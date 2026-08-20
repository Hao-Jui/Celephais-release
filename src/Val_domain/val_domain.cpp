/*
    Copyright 2017 Philippe Grandclement

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"
#include "For_Kadath/Diagnostics/kernel_profile.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Val_domain/der_abs_lane_batch.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Kadath
{
    namespace
    {
        constexpr std::uint64_t fnv_offset_basis = 1469598103934665603ULL;
        constexpr std::uint64_t fnv_prime = 1099511628211ULL;

        // Second independent FNV-1a lane for a 128-bit cache-key digest. A
        // distinct offset basis and multiplier make the two lanes' collisions
        // independent, so digest equality alone is a collision-safe key
        // (~2^-128). That lets the der_abs cache key drop the owning
        // full-coefficient copy it used to carry only for a fallback compare.
        constexpr std::uint64_t fnv_offset_basis_hi = 14695981039346656037ULL;
        constexpr std::uint64_t fnv_prime_hi = 0x9E3779B97F4A7C15ULL;

        struct Digest128
        {
            std::uint64_t lo = fnv_offset_basis;
            std::uint64_t hi = fnv_offset_basis_hi;

            bool operator==(const Digest128& other) const
            {
                return lo == other.lo && hi == other.hi;
            }
        };

        int& der_abs_cache_scope_depth()
        {
            thread_local int depth = 0;
            return depth;
        }

        const bool der_abs_cache_enabled_flag =
            env_flag_enabled("VAL_DOMAIN_DER_ABS_CACHE", true);

        bool der_abs_cache_enabled()
        {
            if (!der_abs_cache_enabled_flag)
                return false;
            // Prevent Clang from speculating the TLS lookup across the cached
            // process-wide gate; the fence is compiler-only on this target.
            std::atomic_signal_fence(std::memory_order_acquire);
            return der_abs_cache_scope_depth() > 0;
        }

        // Two-lane digest over a byte range. The bulk path folds eight bytes per
        // step and post-mixes each lane so bits diffuse downwards as well as
        // upwards (plain multiply only carries upwards); the tail keeps the
        // byte-wise FNV-1a step. Lane independence -- distinct multiplier and
        // distinct post-mix shift -- is what lets digest equality alone key the
        // cache. Digests are in-run only: nothing persists them, so the exact
        // mixing constants are free to change.
        void hash_bytes(Digest128& digest, const void* data, std::size_t bytes)
        {
            const auto* raw = static_cast<const unsigned char*>(data);
            std::size_t offset = 0;
            for (; offset + sizeof(std::uint64_t) <= bytes; offset += sizeof(std::uint64_t)) {
                std::uint64_t word = 0;
                std::memcpy(&word, raw + offset, sizeof(word));
                digest.lo = (digest.lo ^ word) * fnv_prime;
                digest.lo ^= digest.lo >> 29;
                digest.hi = (digest.hi ^ word) * fnv_prime_hi;
                digest.hi ^= digest.hi >> 32;
            }
            for (; offset < bytes; ++offset) {
                digest.lo = (digest.lo ^ raw[offset]) * fnv_prime;
                digest.hi = (digest.hi ^ raw[offset]) * fnv_prime_hi;
            }
        }

        template <typename T>
        void hash_scalar(Digest128& digest, const T& value)
        {
            hash_bytes(digest, &value, sizeof(T));
        }

        void hash_array_shape(Digest128& digest, const Dim_array& dimensions)
        {
            hash_scalar(digest, dimensions.get_ndim());
            for (int axis = 0; axis < dimensions.get_ndim(); ++axis)
                hash_scalar(digest, dimensions(axis));
        }

        void hash_base_signature(Digest128& digest, const Base_spectral& base, int ndim)
        {
            const int defined = base.is_def() ? 1 : 0;
            hash_scalar(digest, defined);
            hash_scalar(digest, ndim);
            if (defined == 0)
                return;

            for (int axis = 0; axis < ndim; ++axis) {
                const Array<int>* base_axis = base.get_base_1d(axis);
                if (base_axis == nullptr) {
                    hash_scalar(digest, -1);
                    continue;
                }
                const std::size_t count = base_axis->get_nbr();
                hash_scalar(digest, count);
                hash_array_shape(digest, base_axis->get_dimensions());
                if (count > 0)
                    hash_bytes(digest, base_axis->get_data(), count * sizeof(int));
            }
        }

        struct DerAbsCacheKey
        {
            const Domain* domain = nullptr;
            std::size_t coefficient_count = 0;
            Digest128 digest;

            bool operator==(const DerAbsCacheKey& other) const
            {
                return domain == other.domain &&
                       coefficient_count == other.coefficient_count &&
                       digest == other.digest;
            }
        };

        struct DerAbsCacheKeyHash
        {
            std::size_t operator()(const DerAbsCacheKey& key) const
            {
                return static_cast<std::size_t>(key.digest.lo);
            }
        };

        struct DerAbsCacheEntry
        {
            std::vector<std::unique_ptr<Val_domain>> derivatives;
        };

        using DerAbsCache = std::unordered_map<DerAbsCacheKey, DerAbsCacheEntry, DerAbsCacheKeyHash>;

        DerAbsCache& der_abs_cache()
        {
            thread_local DerAbsCache cache;
            return cache;
        }

        DerAbsCacheKey make_der_abs_cache_key(const Domain* domain,
                                              const Base_spectral& base,
                                              const Array<double>& coefficients)
        {
            DerAbsCacheKey key;
            key.domain = domain;
            key.coefficient_count = coefficients.get_nbr();

            // 128-bit digest folds in the domain pointer, the base signature and
            // the coefficient shape plus values in a single two-lane pass. Two
            // independent FNV lanes make a false hit ~2^-128, so the digest is a
            // safe key on its own; the coefficient count rides along as a cheap
            // structural discriminator. The shape and base signatures used to be
            // materialised as heap vectors, stored on the key and re-compared on
            // every lookup even though the digest already covered them -- that is
            // two allocations plus an O(base) compare per lookup, now gone.
            Digest128 digest;
            hash_scalar(digest, key.domain);
            hash_array_shape(digest, coefficients.get_dimensions());
            hash_base_signature(digest, base, domain->get_ndim());
            hash_scalar(digest, key.coefficient_count);
            if (key.coefficient_count > 0)
                hash_bytes(digest, coefficients.get_data(), key.coefficient_count * sizeof(double));
            key.digest = digest;
            return key;
        }

        DerAbsCacheEntry snapshot_der_abs_cache_entry(Val_domain** derivatives, int ndim)
        {
            DerAbsCacheEntry entry;
            entry.derivatives.reserve(static_cast<std::size_t>(ndim));
            for (int axis = 0; axis < ndim; ++axis) {
                if (derivatives[axis] == nullptr)
                    entry.derivatives.push_back(nullptr);
                else
                    entry.derivatives.push_back(std::make_unique<Val_domain>(*derivatives[axis]));
            }
            return entry;
        }

        void install_cached_der_abs(const DerAbsCacheEntry& entry, Val_domain** derivatives, int ndim)
        {
            if (entry.derivatives.size() != static_cast<std::size_t>(ndim))
                KADATH_THROW("Val_domain der_abs cache entry has incompatible dimension");
            for (int axis = 0; axis < ndim; ++axis) {
                delete derivatives[axis];
                derivatives[axis] = entry.derivatives[static_cast<std::size_t>(axis)] == nullptr
                                        ? nullptr
                                        : new Val_domain(*entry.derivatives[static_cast<std::size_t>(axis)]);
            }
        }

    } // namespace

    ValDomainDerAbsAssemblyCacheScope::ValDomainDerAbsAssemblyCacheScope()
    {
        ++der_abs_cache_scope_depth();
    }

    ValDomainDerAbsAssemblyCacheScope::~ValDomainDerAbsAssemblyCacheScope()
    {
        --der_abs_cache_scope_depth();
        if (der_abs_cache_scope_depth() == 0)
            reset_val_domain_der_abs_cache();
    }

    void reset_val_domain_der_abs_cache()
    {
        der_abs_cache().clear();
    }

    void Val_domain::allocate_derivative_storage()
    {
        for (Val_domain*& derivative : inline_derivatives)
            derivative = nullptr;
        p_der_var = nullptr;
        p_der_abs = nullptr;

        const int ndim = zone->get_ndim();
        if (ndim <= inline_derivative_dimensions) {
            p_der_var = inline_derivatives;
            p_der_abs = inline_derivatives + inline_derivative_dimensions;
            return;
        }

        p_der_var = MemoryMapper::get_memory<Val_domain*>(ndim);
        try {
            p_der_abs = MemoryMapper::get_memory<Val_domain*>(ndim);
        } catch (...) {
            MemoryMapper::release_memory<Val_domain*>(p_der_var, ndim);
            p_der_var = nullptr;
            throw;
        }
        for (int i = 0; i < ndim; ++i) {
            p_der_var[i] = nullptr;
            p_der_abs[i] = nullptr;
        }
    }

    void Val_domain::release_derivative_storage() noexcept
    {
        if (p_der_var != nullptr && p_der_var != inline_derivatives)
            MemoryMapper::release_memory<Val_domain*>(p_der_var, zone->get_ndim());
        if (p_der_abs != nullptr && p_der_abs != inline_derivatives + inline_derivative_dimensions)
            MemoryMapper::release_memory<Val_domain*>(p_der_abs, zone->get_ndim());
        p_der_var = nullptr;
        p_der_abs = nullptr;
    }

    void Val_domain::move_derivative_storage_from(Val_domain& so) noexcept
    {
        for (Val_domain*& derivative : inline_derivatives)
            derivative = nullptr;

        if (so.p_der_var == so.inline_derivatives) {
            p_der_var = inline_derivatives;
            p_der_abs = inline_derivatives + inline_derivative_dimensions;
            for (int i = 0; i < 2 * inline_derivative_dimensions; ++i) {
                inline_derivatives[i] = so.inline_derivatives[i];
                so.inline_derivatives[i] = nullptr;
            }
        } else {
            p_der_var = so.p_der_var;
            p_der_abs = so.p_der_abs;
        }
        so.p_der_var = nullptr;
        so.p_der_abs = nullptr;
    }

    void Val_domain::swap_derivative_storage(Val_domain& so) noexcept
    {
        const bool this_inline = p_der_var == inline_derivatives;
        const bool so_inline = so.p_der_var == so.inline_derivatives;
        if (this_inline && so_inline) {
            for (int i = 0; i < 2 * inline_derivative_dimensions; ++i)
                std::swap(inline_derivatives[i], so.inline_derivatives[i]);
            return;
        }
        if (!this_inline && !so_inline) {
            std::swap(p_der_var, so.p_der_var);
            std::swap(p_der_abs, so.p_der_abs);
            return;
        }
        if (!this_inline) {
            so.swap_derivative_storage(*this);
            return;
        }

        Val_domain** external_var = so.p_der_var;
        Val_domain** external_abs = so.p_der_abs;
        for (int i = 0; i < 2 * inline_derivative_dimensions; ++i) {
            so.inline_derivatives[i] = inline_derivatives[i];
            inline_derivatives[i] = nullptr;
        }
        so.p_der_var = so.inline_derivatives;
        so.p_der_abs = so.inline_derivatives + inline_derivative_dimensions;
        p_der_var = external_var;
        p_der_abs = external_abs;
    }

    Val_domain::Val_domain(const Domain* dom)
        : zone(dom), base(zone->get_ndim()), is_zero(false), c(nullptr), cf(nullptr), in_conf(false), in_coef(false)
    {
        allocate_derivative_storage();
    }

    Val_domain::Val_domain(const Val_domain& so, bool copie)
        : zone(so.zone), base(so.base), is_zero(so.is_zero), in_conf(so.in_conf), in_coef(so.in_coef)
    {

        c = ((so.c != nullptr) && copie) ? new Array<double>(*so.c) : nullptr;
        cf = ((so.cf != nullptr) && copie) ? new Array<double>(*so.cf) : nullptr;
        if (!copie) {
            in_conf = false;
            in_coef = false;
        }

        allocate_derivative_storage();
        for (int i = 0; i < zone->get_ndim(); i++) {
            p_der_var[i] = ((so.p_der_var[i] != nullptr) && copie) ? new Val_domain(*so.p_der_var[i]) : nullptr;
            p_der_abs[i] = ((so.p_der_abs[i] != nullptr) && copie) ? new Val_domain(*so.p_der_abs[i]) : nullptr;
        }
    }

    Val_domain::Val_domain(const Domain* dom, const Val_domain& so)
        : zone(dom), base(so.base), is_zero(so.is_zero), in_conf(so.in_conf), in_coef(so.in_coef)
    {

        c = ((so.c != nullptr)) ? new Array<double>(*so.c) : nullptr;
        cf = ((so.cf != nullptr)) ? new Array<double>(*so.cf) : nullptr;

        allocate_derivative_storage();
        for (int i = 0; i < zone->get_ndim(); i++) {
            p_der_var[i] = ((so.p_der_var[i] != nullptr)) ? new Val_domain(dom, *so.p_der_var[i]) : nullptr;
            p_der_abs[i] = ((so.p_der_abs[i] != nullptr)) ? new Val_domain(dom, *so.p_der_abs[i]) : nullptr;
        }
    }

    Val_domain::Val_domain(Val_domain&& so) noexcept
        : zone{so.zone}, base{std::move(so.base)}, is_zero{so.is_zero}, c{so.c}, cf{so.cf}, in_conf{so.in_conf},
          in_coef{so.in_coef}
    {
        move_derivative_storage_from(so);
        so.c = nullptr;
        so.cf = nullptr;
    }

    void Val_domain::swap(Val_domain& so) noexcept
    {
        assert(zone == so.zone);
        base.swap(so.base);
        std::swap(is_zero, so.is_zero);
        std::swap(c, so.c);
        std::swap(cf, so.cf);
        std::swap(in_conf, so.in_conf);
        std::swap(in_coef, so.in_coef);
        swap_derivative_storage(so);
    }

    Val_domain& Val_domain::operator=(Val_domain&& so) noexcept
    {
        assert(zone == so.zone);
        base = std::move(so.base);
        is_zero = so.is_zero;
        in_conf = so.in_conf;
        in_coef = so.in_coef;
        std::swap(c, so.c);
        std::swap(cf, so.cf);
        swap_derivative_storage(so);
        return *this;
    }

    Val_domain::~Val_domain()
    {
        del_deriv();
        release_derivative_storage();

        if (c != nullptr)
            delete c;
        if (cf != nullptr)
            delete cf;
    }

    Val_domain::Val_domain(const Domain* so, BinarySource& source) : zone(so), base(source)
    {
        zone->validate_spectral_parity(base);
        const int indic_zero = source.read<int>();
        is_zero = (indic_zero != 0);
        const int indic_conf = source.read<int>();
        in_conf = (indic_conf == 0);
        c = in_conf ? new Array<double>(source) : nullptr;
        const int indic_coef = source.read<int>();
        in_coef = (indic_coef == 0);
        cf = in_coef ? new Array<double>(source) : nullptr;

        allocate_derivative_storage();
    }

    void Val_domain::save(BinarySink& sink) const
    {
        base.save(sink);
        sink.write<int>(is_zero ? 1 : 0);
        sink.write<int>(in_conf ? 0 : 1);
        if (in_conf)
            c->save(sink);
        sink.write<int>(in_coef ? 0 : 1);
        if (in_coef)
            cf->save(sink);
    }

    void Val_domain::del_deriv() const
    {
        if (p_der_var == nullptr)
            return;
        for (int i = 0; i < zone->get_ndim(); i++) {
            if (p_der_var[i] != nullptr) {
                delete p_der_var[i];
                p_der_var[i] = nullptr;
            }
            if (p_der_abs[i] != nullptr) {
                delete p_der_abs[i];
                p_der_abs[i] = nullptr;
            }
        }
    }

    void Val_domain::operator=(const Val_domain& so)
    {
        assert(zone == so.zone);
        if (this == &so)
            return;
        this->assign_vals(so);
    }

    void Val_domain::assign_vals(const Val_domain& so)
    {
        is_zero = so.is_zero;
        in_conf = so.in_conf;
        in_coef = so.in_coef;
        if (in_conf) {
            if (c != nullptr && c->get_dimensions() == so.c->get_dimensions())
                *c = *so.c;
            else {
                delete c;
                c = new Array<double>(*so.c);
            }
        } else {
            delete c;
            c = nullptr;
        }
        if (in_coef) {
            if (cf != nullptr && cf->get_dimensions() == so.cf->get_dimensions())
                *cf = *so.cf;
            else {
                delete cf;
                cf = new Array<double>(*so.cf);
            }
        } else {
            delete cf;
            cf = nullptr;
        }
        if (so.base.is_def())
            base = so.base;
        else
            base.set_non_def();
        del_deriv();
        for (int i = 0; i < zone->get_ndim(); i++) {
            p_der_var[i] = (so.p_der_var[i] == nullptr) ? nullptr : new Val_domain(*so.p_der_var[i]);
            p_der_abs[i] = (so.p_der_abs[i] == nullptr) ? nullptr : new Val_domain(*so.p_der_abs[i]);
        }
    }

    void Val_domain::operator=(double xx)
    {
        if (xx == 0)
            set_zero();
        else {
            is_zero = false;
            allocate_conf();
            del_deriv();
            *c = xx;
        }
    }

    void Val_domain::annule_hard()
    {
        allocate_conf();
        del_deriv();
        *c = 0.;
    }

    void Val_domain::annule_hard_coef()
    {
        allocate_coef();
        del_deriv();
        *cf = 0.;
    }

    double& Val_domain::set(const Index& index)
    {
        set_in_conf();
        del_deriv();
        return c->set(index);
    }

    double& Val_domain::set_coef(const Index& index)
    {
        set_in_coef();
        del_deriv();
        return cf->set(index);
    }

    double Val_domain::get_coef(const Index& index) const
    {
        if (is_zero)
            return 0.;
        else {
            assert(in_coef);
            return cf->set(index);
        }
    }

    double Val_domain::operator()(const Index& index) const
    {
        coef_i();
        return (*c)(index);
    }

    void Val_domain::set_in_conf()
    {
        if (cf != nullptr) {
            delete cf;
            cf = nullptr;
        }
        in_conf = true;
        in_coef = false;
    }

    void Val_domain::set_in_coef()
    {
        if (c != nullptr) {
            delete c;
            c = nullptr;
        }
        in_conf = false;
        in_coef = true;
    }

    void Val_domain::allocate_conf()
    {
        set_in_conf();
        is_zero = false;
        if (c != nullptr)
            delete c;
        c = new Array<double>(zone->get_nbr_points());
    }

    void Val_domain::allocate_coef()
    {
        set_in_coef();
        is_zero = false;
        if (cf != nullptr)
            delete cf;
        cf = new Array<double>(zone->get_nbr_coefs());
    }

    void Val_domain::set_zero()
    {
        if (!is_zero) {
            del_deriv();

            if (in_conf) {
                delete c;
                c = nullptr;
                in_conf = false;
            }

            if (in_coef) {
                delete cf;
                cf = nullptr;
                in_coef = false;
            }

            base.set_non_def();
        }
        is_zero = true;
    }

    void Val_domain::std_base()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_r_base()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_r_base(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_r_base(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_anti_base()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_anti_cheb_base(base);
                break;
            case LEG_TYPE:
                zone->set_anti_legendre_base(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_anti_base");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base(int m)
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_with_m(base, m);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_with_m(base, m);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_anti_base(int m)
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_anti_cheb_base_with_m(base, m);
                break;
            case LEG_TYPE:
                zone->set_anti_legendre_base_with_m(base, m);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_anti_base");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_rt_spher()
    {
        int typeb(zone->get_type_base());
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_rt_spher(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_rt_spher");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_rp_spher()
    {
        int typeb(zone->get_type_base());
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_rp_spher(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_rp_spher");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_tp_spher()
    {
        int typeb(zone->get_type_base());
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_tp_spher(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_tp_spher");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_xy_cart()
    {
        int typeb(zone->get_type_base());
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_xy_cart(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_xy_cart");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_xz_cart()
    {
        int typeb(zone->get_type_base());
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_xz_cart(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_xz_cart");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_yz_cart()
    {
        int typeb(zone->get_type_base());
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_yz_cart(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_yz_cart");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_r_spher()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_r_spher(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_r_spher(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_r_spher");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_t_spher()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_t_spher(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_t_spher(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_t_spher");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_p_spher()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_p_spher(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_p_spher(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_p_spher");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_x_cart()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_x_cart(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_x_cart(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_x_cart");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_y_cart()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_y_cart(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_y_cart(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_y_cart");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_z_cart()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_z_cart(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_z_cart(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_z_cart");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_r_mtz()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_r_mtz(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_r_mtz(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_r_mtz");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_t_mtz()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_t_mtz(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_t_mtz(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_t_mtz");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_p_mtz()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_p_mtz(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_p_mtz(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_p_mtz");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_xodd_base()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_xodd_base(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_xodd_base(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_xodd_base");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_todd_base()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_todd_base(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_todd_base(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_todd_base");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_xodd_todd_base()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_xodd_todd_base(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_xodd_todd_base(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_xodd_todd_base");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::std_base_odd()
    {
        // recupere le type :
        int typeb = zone->get_type_base();
        switch (typeb) {
            case CHEB_TYPE:
                zone->set_cheb_base_odd(base);
                break;
            case LEG_TYPE:
                zone->set_legendre_base_odd(base);
                break;
            default:
                KADATH_THROW("Unknown type of base in Val_domain::std_base_odd");
        }
        zone->validate_spectral_parity(base);
    }

    void Val_domain::coef() const
    {
        if ((in_coef) || (is_zero)) {
            return;
        } else {
            if (base.is_def() == false) {
                KADATH_THROW("Base not defined in Val_domain::coef");
            } else {
                assert(in_conf);
                if (cf != nullptr)
                    delete cf;
                cf = new Array<double>(base.coef(zone->get_nbr_coefs(), *c));
            }
            in_coef = true;
        }
    }

    void Val_domain::coef_i() const
    {
        if ((in_conf) || (is_zero)) {
            return;
        } else {
            if (base.is_def() == false) {
                KADATH_THROW("Base not defined in Val_domain::coef");
            } else {
                assert(in_coef);
                if (c != nullptr)
                    delete c;
                c = new Array<double>(base.coef_i(zone->get_nbr_points(), *cf));
            }
            in_conf = true;
        }
    }

    ostream& operator<<(ostream& o, const Val_domain& so)
    {

        if (so.is_zero)
            o << "Null Val_domain" << endl;

        if (so.in_conf) {
            o << "Configuration space : " << endl;
            o << *so.c << endl;
        }

        if (so.in_coef) {
            o << "Coefficient space : " << endl;
            o << *so.cf << endl;
        }
        return o;
    }

    Val_domain Val_domain::der_var(int var) const
    {
        assert((var > 0) && (var <= zone->get_ndim()));
        if (is_zero)
            return *this;
        else {
            if (p_der_var[var - 1] == nullptr)
                compute_der_var();
            return *p_der_var[var - 1];
        }
    }

    Val_domain Val_domain::der_abs(int var) const
    {
        assert((var >= 0) && (var <= zone->get_ndim()));
        if (var == 0) {
            Val_domain zero(get_domain());
            zero = 0.0;
            return zero;
        }
        if (is_zero)
            return *this;
        else {
            if (p_der_abs[var - 1] == nullptr)
                compute_der_abs();
            return *p_der_abs[var - 1];
        }
    }

    Val_domain Val_domain::der_spher(int var) const
    {
        assert((var > 0) and (var <= zone->get_ndim()));
        if (is_zero)
            return *this;
        else {
            Val_domain zero(zone);
            zero = 0.0;
            if (var == 0)
                return zero;
            else if (var == 1)
                return this->der_r();
            else if (var == 2)
                return this->der_t();
            else if (var == 3)
                return this->der_p();
            else {
                KADATH_THROW("bad index in der_spher");
            }
        }
    }

    Val_domain Val_domain::der_r() const
    {
        return zone->der_r(*this);
    }

    Val_domain Val_domain::der_t() const
    {
        return zone->der_t(*this);
    }

    Val_domain Val_domain::der_p() const
    {
        return zone->der_p(*this);
    }

    Val_domain Val_domain::der_r_rtwo() const
    {
        return zone->der_r_rtwo(*this);
    }

    Val_domain Val_domain::div_r() const
    {
        if (is_zero)
            return *this;
        else
            return zone->div_r(*this);
    }

    Val_domain Val_domain::mult_r() const
    {
        if (is_zero)
            return *this;
        else
            return zone->mult_r(*this);
    }
    double Val_domain::integrale() const
    {
        if (is_zero)
            return 0.;
        else
            return zone->integrale(*this);
    }

    double Val_domain::integ_volume() const
    {
        if (is_zero)
            return 0.;
        else
            return zone->integ_volume(*this);
    }

    void Val_domain::compute_der_var() const
    {
        KernelTimer kernel_timer(KernelId::ComputeDerVar);
        coef();
        for (int var = 0; var < zone->get_ndim(); var++) {
            Val_domain res(zone);
            res.base = base;
            res.cf = new Array<double>(base.ope_der_1d(var, *cf, res.base));
            res.in_coef = true;
            // p_der_var[var] was allocated via scalar `new Val_domain(...)`
            // (line below + the destructor pattern at the top of the file
            // uses `delete p_der_var[i]`), so the matching deallocation must
            // also be scalar. Prior `delete[]` was C++ UB even though it
            // happened to work in practice via MemoryMappable's identical
            // release_memory signature for array/scalar deletes.
            if (p_der_var[var] != nullptr)
                delete p_der_var[var];
            // res is a dying local: move it in. Steals base + cf via the
            // non-allocating move ctor (faf83abd) -- no Base_spectral
            // basis-metadata copy and no Array<double> coefficient
            // copy. Bit-exact (move transfers buffers, COO byte-hash gate).
            // Beachhead for the move-semantics workstream. NB: the earlier
            // A/B that found this neutral weighed only the c/cf data copy,
            // not the Base_spectral object-copy traffic the leaf board
            // attributes to Val_domain copies; the win is object churn +
            // intent (move dying temporaries), not a measured res9 speedup.
            p_der_var[var] = new Val_domain(std::move(res));
        }
    }

    bool Val_domain::derivative_lane_tiling_enabled()
    {
        static const bool enabled = env_flag_enabled("DERIVATIVE_LANE_TILING", true);
        return enabled;
    }

    void Val_domain::prepare_der_var_batch(std::span<const Val_domain* const> fields)
    {
        if (fields.empty())
            return;
        if (fields.size() > static_cast<std::size_t>(Term_eq::max_derivative_lanes))
            KADATH_THROW("Val_domain::prepare_der_var_batch exceeds the packed tile bound");
        if (!derivative_lane_tiling_enabled())
            return;
        if (fields.front() == nullptr)
            KADATH_THROW("Val_domain::prepare_der_var_batch received a null field");

        const Domain* domain = fields.front()->zone;
        for (std::size_t lane = 0; lane < fields.size(); ++lane) {
            const Val_domain* field = fields[lane];
            if (field == nullptr)
                KADATH_THROW("Val_domain::prepare_der_var_batch received a null field");
            if (field->zone != domain)
                KADATH_THROW("Val_domain::prepare_der_var_batch fields must belong to one Domain");
            for (std::size_t previous = 0; previous < lane; ++previous)
                if (fields[previous] == field)
                    KADATH_THROW("Val_domain::prepare_der_var_batch received a repeated field");
        }

        std::array<const Val_domain*, Term_eq::max_derivative_lanes> active{};
        std::size_t active_count = 0;
        for (const Val_domain* field : fields) {
            if (field->is_zero)
                continue;
            bool needs_derivative = false;
            for (int axis = 0; axis < domain->get_ndim(); ++axis)
                needs_derivative = needs_derivative || field->p_der_var[axis] == nullptr;
            if (needs_derivative) {
                field->coef();
                active[active_count++] = field;
            }
        }
        if (active_count == 0)
            return;

        KernelTimer kernel_timer(KernelId::ComputeDerVar);
        const int dimensions = domain->get_ndim();
        std::vector<std::unique_ptr<Val_domain>> staged(
            active_count * static_cast<std::size_t>(dimensions));
        for (std::size_t lane = 0; lane < active_count; ++lane) {
            for (int axis = 0; axis < dimensions; ++axis) {
                auto result = std::make_unique<Val_domain>(domain);
                result->base = active[lane]->base;
                result->cf = new Array<double>(active[lane]->cf->get_dimensions());
                const std::size_t offset =
                    lane * static_cast<std::size_t>(dimensions) + static_cast<std::size_t>(axis);
                staged[offset] = std::move(result);
            }
        }

        std::array<const Base_spectral*, Term_eq::max_derivative_lanes> bases_in{};
        std::array<const Array<double>*, Term_eq::max_derivative_lanes> inputs{};
        std::array<Base_spectral*, Term_eq::max_derivative_lanes> bases_out{};
        std::array<Array<double>*, Term_eq::max_derivative_lanes> outputs{};
        for (std::size_t lane = 0; lane < active_count; ++lane) {
            bases_in[lane] = &active[lane]->base;
            inputs[lane] = active[lane]->cf;
        }

        for (int axis = 0; axis < dimensions; ++axis) {
            for (std::size_t lane = 0; lane < active_count; ++lane) {
                const std::size_t offset =
                    lane * static_cast<std::size_t>(dimensions) + static_cast<std::size_t>(axis);
                bases_out[lane] = &staged[offset]->base;
                outputs[lane] = staged[offset]->cf;
            }
            Base_spectral::ope_der_1d_batch(axis,
                                            bases_in.data(),
                                            inputs.data(),
                                            bases_out.data(),
                                            outputs.data(),
                                            static_cast<int>(active_count));
            for (std::size_t lane = 0; lane < active_count; ++lane) {
                const std::size_t offset =
                    lane * static_cast<std::size_t>(dimensions) + static_cast<std::size_t>(axis);
                staged[offset]->in_coef = true;
            }
        }

        // Every output is complete before any source cache changes.
        for (std::size_t lane = 0; lane < active_count; ++lane) {
            for (int axis = 0; axis < dimensions; ++axis) {
                const std::size_t offset =
                    lane * static_cast<std::size_t>(dimensions) + static_cast<std::size_t>(axis);
                delete active[lane]->p_der_var[axis];
                active[lane]->p_der_var[axis] = staged[offset].release();
            }
        }
    }

    void Val_domain::prepare_der_abs_batch(std::span<const Val_domain* const> fields)
    {
        if (fields.empty())
            return;
        if (fields.size() > static_cast<std::size_t>(Term_eq::max_derivative_lanes))
            KADATH_THROW("Val_domain::prepare_der_abs_batch exceeds the packed tile bound");
        if (!derivative_lane_tiling_enabled())
            return;
        if (fields.front() == nullptr)
            KADATH_THROW("Val_domain::prepare_der_abs_batch received a null field");

        const Domain* domain = fields.front()->zone;
        for (std::size_t lane = 0; lane < fields.size(); ++lane) {
            const Val_domain* field = fields[lane];
            if (field == nullptr)
                KADATH_THROW("Val_domain::prepare_der_abs_batch received a null field");
            if (field->zone != domain)
                KADATH_THROW("Val_domain::prepare_der_abs_batch fields must belong to one Domain");
            for (std::size_t previous = 0; previous < lane; ++previous)
                if (fields[previous] == field)
                    KADATH_THROW("Val_domain::prepare_der_abs_batch received a repeated field");
        }

        std::array<const Val_domain*, Term_eq::max_derivative_lanes> active{};
        std::size_t active_count = 0;
        std::vector<DerAbsCacheKey> cache_keys;
        if (der_abs_cache_enabled())
            cache_keys.reserve(fields.size());

        for (const Val_domain* field : fields) {
            if (field->is_zero)
                continue;

            bool has_any = false;
            bool has_all = true;
            for (int axis = 0; axis < domain->get_ndim(); ++axis) {
                has_any = has_any || field->p_der_abs[axis] != nullptr;
                has_all = has_all && field->p_der_abs[axis] != nullptr;
            }
            if (has_all)
                continue;
            if (has_any)
                KADATH_THROW("Val_domain::prepare_der_abs_batch found a partial derivative cache");

            if (der_abs_cache_enabled()) {
                field->coef();
                DerAbsCacheKey key = make_der_abs_cache_key(domain, field->base, *field->cf);
                DerAbsCache& cache = der_abs_cache();
                auto cached = cache.find(key);
                if (cached != cache.end()) {
                    install_cached_der_abs(cached->second, field->p_der_abs, domain->get_ndim());
                    continue;
                }
                cache_keys.push_back(std::move(key));
            }
            active[active_count++] = field;
        }
        if (active_count == 0)
            return;

        KernelTimer kernel_timer(KernelId::ComputeDerAbs);
        const std::span<const Val_domain* const> active_fields(active.data(), active_count);
        prepare_der_var_batch(active_fields);
        DerAbsLaneBatch batch(active_fields);
        domain->do_der_abs_from_der_var_lanes(batch);
        batch.commit();

        if (der_abs_cache_enabled()) {
            DerAbsCache& cache = der_abs_cache();
            for (std::size_t lane = 0; lane < active_count; ++lane)
                cache.emplace(std::move(cache_keys[lane]),
                              snapshot_der_abs_cache_entry(active[lane]->p_der_abs, domain->get_ndim()));
        }
    }

    void Val_domain::compute_der_abs() const
    {
        KernelTimer kernel_timer(KernelId::ComputeDerAbs);
        if (der_abs_cache_enabled()) {
            coef();
            DerAbsCacheKey key = make_der_abs_cache_key(zone, base, *cf);

            DerAbsCache& cache = der_abs_cache();
            auto cached = cache.find(key);
            if (cached != cache.end()) {
                install_cached_der_abs(cached->second, p_der_abs, zone->get_ndim());
                return;
            }

            for (int i = 0; i < zone->get_ndim(); i++)
                if (p_der_var[i] == nullptr)
                    compute_der_var();
            zone->do_der_abs_from_der_var(p_der_var, p_der_abs);

            cache.emplace(std::move(key), snapshot_der_abs_cache_entry(p_der_abs, zone->get_ndim()));
            return;
        }

        for (int i = 0; i < zone->get_ndim(); i++)
            if (p_der_var[i] == nullptr)
                compute_der_var();
        zone->do_der_abs_from_der_var(p_der_var, p_der_abs);
    }

} // namespace Kadath
