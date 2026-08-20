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

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 *   2026-08-06  RAII/span modernization.
 */

#include "For_Kadath/Kadath_point_h/kadath.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Val_domain/der_abs_lane_batch.hpp"
#include "For_Kadath/Diagnostics/matching_lane_profile.hpp"
#include "../Term_eq/term_eq_derivative_lane_layout.hpp"
#include <cxxabi.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>
namespace Kadath
{
    /**
     * Bounded memo behind \c Domain::multiplied_base. Only a handful of distinct
     * basis layouts occur per domain, so a linear scan over a short table beats
     * hashing the key, and the cap keeps domain-lifetime retention bounded even
     * for user-defined bases.
     * ponytail: linear scan over max_entries; move to a digest-keyed map only if
     * a workload ever shows more than max_entries distinct operand pairs.
     */
    struct Domain::BasisMultCache
    {
        static constexpr std::size_t max_entries = 16;

        struct Entry
        {
            std::vector<int> key;
            Base_spectral result;
        };

        std::mutex mutex;
        std::vector<int> lookup_key;
        std::vector<Entry> entries;
    };

    struct Domain::ImportPlanCache
    {
        static constexpr std::size_t max_entries = 32;

        struct Key
        {
            int target_domain = -1;
            int bound = 0;
            int layout = 0;
            int n_ope = 0;
            std::uint64_t lookup_precision_bits = 0;
            std::vector<int> source_domains;
            std::vector<std::uint64_t> source_shape_fingerprints;

            bool operator==(const Key& other) const
            {
                return target_domain == other.target_domain && bound == other.bound &&
                       layout == other.layout && n_ope == other.n_ope &&
                       lookup_precision_bits == other.lookup_precision_bits &&
                       source_domains == other.source_domains &&
                       source_shape_fingerprints == other.source_shape_fingerprints;
            }
        };

        struct PointPlan
        {
            int first = 0;
            int second = 0;
            int source_domain = -1;
            std::array<double, 3> numerical{};
        };

        struct Entry
        {
            Key key;
            std::vector<PointPlan> points;
        };

        std::vector<Entry> entries;
    };

    namespace
    {
        constexpr std::uint64_t import_hash_offset = 1469598103934665603ULL;
        constexpr std::uint64_t import_hash_prime = 1099511628211ULL;

        void import_hash_bytes(std::uint64_t& hash, const void* data, std::size_t size)
        {
            const auto* bytes = static_cast<const unsigned char*>(data);
            for (std::size_t index = 0; index < size; ++index) {
                hash ^= bytes[index];
                hash *= import_hash_prime;
            }
        }

        template <typename T>
        void import_hash_value(std::uint64_t& hash, const T& value)
        {
            import_hash_bytes(hash, &value, sizeof(value));
        }

        std::uint64_t import_double_bits(double value)
        {
            std::uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        }

        void copy_tensor_metadata(const Tensor& source, Tensor& destination)
        {
            if ((source.get_parameters() != nullptr) && (destination.get_parameters() == nullptr))
                destination.set_parameters() = new Param_tensor(*source.get_parameters());

            if (source.is_name_affected()) {
                destination.set_name_affected();
                const char* source_names = source.get_name_ind();
                for (int index = 0; index < source.get_valence(); index++)
                    destination.set_name_ind(index, source_names[index]);
            }
        }

        void prepare_cartesian_derivative_lanes(int domain, const Term_eq& source)
        {
            if (!Val_domain::derivative_lane_tiling_enabled() || source.get_p_der_t() == nullptr)
                return;
            if (!derivative_lane_detail::all_derivative_lanes_match_value_layout(source))
                return;

            const int lane_count = source.get_derivative_lane_count();
            if (lane_count > Term_eq::max_derivative_lanes)
                KADATH_THROW("Domain derivative lane count exceeds the packed tile bound");

            const Tensor& value = source.get_val_t();
            for (int component = 0; component < value.get_n_comp(); ++component) {
                std::array<const Val_domain*, Term_eq::max_derivative_lanes> fields{};
                std::size_t field_count = 0;
                for (int lane = 0; lane < lane_count; ++lane) {
                    const Tensor* derivative = source.get_p_der_t(lane);
                    if (derivative == nullptr)
                        continue;
                    const Array<int> index(derivative->indices(component));
                    fields[field_count++] = &(*derivative)(index)(domain);
                }
                Val_domain::prepare_der_abs_batch(
                    std::span<const Val_domain* const>(fields.data(), field_count));
            }
        }

        template <typename FillDerivative>
        Term_eq make_tensor_term_with_derivative_lanes(
            int domain,
            const Term_eq& source,
            const Tensor& value_result,
            FillDerivative fill_derivative)
        {
            if (source.get_p_der_t() == nullptr)
                return Term_eq(domain, value_result);

            Tensor primary_derivative(value_result, false);
            fill_derivative(source.get_der_t(), primary_derivative);

            Term_eq result(domain, value_result, primary_derivative);
            result.set_derivative_lane_count(source.get_derivative_lane_count());
            for (int lane = 1; lane < source.get_derivative_lane_count(); ++lane) {
                if (!source.has_der_t(lane))
                    continue;
                Tensor derivative_result(value_result, false);
                fill_derivative(source.get_der_t(lane), derivative_result);
                result.set_der_t(lane, derivative_result);
            }
            return result;
        }

        template <typename IntegrateDerivative>
        Term_eq make_double_term_with_derivative_lanes(
            int domain,
            double value_result,
            const Term_eq& source,
            IntegrateDerivative integrate_derivative)
        {
            if (source.get_p_der_t() == nullptr)
                return Term_eq(domain, value_result);

            Term_eq result(domain, value_result, integrate_derivative(source.get_der_t()));
            result.set_derivative_lane_count(source.get_derivative_lane_count());
            for (int lane = 1; lane < source.get_derivative_lane_count(); ++lane) {
                if (source.has_der_t(lane))
                    result.set_der_d(lane, integrate_derivative(source.get_der_t(lane)));
            }
            return result;
        }
    }

    void Domain::clear_import_plan_cache() const
    {
        import_plan_cache_.reset();
    }

    bool Domain::fingerprint_import_shape_component(const Val_domain& component,
                                                    std::uint64_t& fingerprint) const
    {
        const bool is_zero = component.check_if_zero();
        import_hash_value(fingerprint, is_zero);
        if (is_zero)
            return true;

        const Base_spectral& base = component.get_base();
        if (!base.is_def())
            return false;

        component.coef();
        const Array<double>& coefficients = component.get_coef_ref();
        const Dim_array dimensions = coefficients.get_dimensions();
        const int dimension_count = dimensions.get_ndim();
        import_hash_value(fingerprint, dimension_count);
        for (int axis = 0; axis < dimension_count; ++axis)
            import_hash_value(fingerprint, dimensions(axis));

        for (int axis = 0; axis < dimension_count; ++axis) {
            const Array<int>* axis_base = base.get_base_1d(axis);
            const std::size_t base_count = axis_base == nullptr ? 0 : axis_base->get_nbr();
            import_hash_value(fingerprint, base_count);
            if (axis_base != nullptr)
                import_hash_bytes(fingerprint, axis_base->get_data(), base_count * sizeof(int));
        }

        const std::size_t coefficient_count = coefficients.get_nbr();
        import_hash_value(fingerprint, coefficient_count);
        import_hash_bytes(fingerprint, coefficients.get_data(), coefficient_count * sizeof(double));
        return true;
    }

    void Domain::snapshot_mapping_component(const Val_domain& component,
                                            std::vector<Val_domain>& out,
                                            std::size_t& idx) const
    {
        if (idx < out.size()) {
            if (out[idx].get_domain() != component.get_domain())
                KADATH_THROW("Domain mapping snapshot topology changed");
            out[idx] = component;
        } else if (idx == out.size()) {
            out.push_back(component);
        } else {
            KADATH_THROW("Domain mapping snapshot index is not contiguous");
        }
        ++idx;
    }

    void Domain::snapshot_mapping_into(std::vector<Val_domain>& out, std::size_t& idx) const
    {
        // Compatibility path for a derived Domain that has not provided the
        // reusable hook. Current adapted production domains override this and
        // write directly into their stable slots.
        std::vector<Val_domain> appended;
        snapshot_mapping(appended);
        for (const Val_domain& component : appended)
            snapshot_mapping_component(component, out, idx);
    }

    // standard constructor
    Domain::Domain(int num, int ttype, const Dim_array& res)
        : num_dom(num), ndim(res.get_ndim()), nbr_points(res), nbr_coefs(ndim), type_base(ttype)
    {

        for (int i = 0; i < ndim; i++)
            nbr_coefs.set(i) = 0;
        coloc = new Array<double>*[ndim];
        for (int i = 0; i < ndim; i++)
            coloc[i] = nullptr;
        absol = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            absol[i] = nullptr;
        cart = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            cart[i] = nullptr;
        cart_surr = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            cart_surr[i] = nullptr;
        radius = nullptr;
    }

    // Constructor by copy
    Domain::Domain(const Domain& so)
        : num_dom(so.num_dom), ndim(so.ndim), nbr_points(so.nbr_points), nbr_coefs(so.nbr_coefs),
          type_base(so.type_base)
    {
        coloc = new Array<double>*[ndim];
        for (int i = 0; i < ndim; i++)
            coloc[i] = (so.coloc[i] == nullptr) ? nullptr : new Array<double>(*so.coloc[i]);
        absol = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            absol[i] = (so.absol[i] == nullptr) ? nullptr : new Val_domain(*so.absol[i]);
        cart = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            cart[i] = (so.cart[i] == nullptr) ? nullptr : new Val_domain(*so.cart[i]);
        cart_surr = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            cart_surr[i] = (so.cart_surr[i] == nullptr) ? nullptr : new Val_domain(*so.cart_surr[i]);
        radius = (so.radius == nullptr) ? nullptr : new Val_domain(*so.radius);
    }

    // Constructor by copy to new, identical Space
    Domain::Domain(const Domain& so, bool import)
        : num_dom(so.num_dom), ndim(so.ndim), nbr_points(so.nbr_points), nbr_coefs(so.nbr_coefs),
          type_base(so.type_base)
    {
        const Domain& so_copy = *this;

        coloc = new Array<double>*[ndim];
        for (int i = 0; i < ndim; i++)
            coloc[i] = (so.coloc[i] == nullptr) ? nullptr : new Array<double>(*so.coloc[i]);
        absol = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            absol[i] = (so.absol[i] == nullptr) ? nullptr : new Val_domain(&so_copy, *so.absol[i]);
        cart = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            cart[i] = (so.cart[i] == nullptr) ? nullptr : new Val_domain(&so_copy, *so.cart[i]);
        cart_surr = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            cart_surr[i] = (so.cart_surr[i] == nullptr) ? nullptr : new Val_domain(&so_copy, *so.cart_surr[i]);
        radius = (so.radius == nullptr) ? nullptr : new Val_domain(&so_copy, *so.radius);
    }

    // Destructor
    Domain::~Domain()
    {
        del_deriv();
        delete[] absol;
        delete[] cart;
        delete[] coloc;
        delete[] cart_surr;
    }

    Domain::Domain(int num, BinarySource& source)
        : num_dom(num), nbr_points(source), nbr_coefs(source)
    {
        ndim = source.read<int>();
        type_base = source.read<int>();
        assert(ndim == nbr_points.get_ndim());
        assert(ndim == nbr_coefs.get_ndim());
        coloc = new Array<double>*[ndim];
        for (int i = 0; i < ndim; i++)
            coloc[i] = nullptr;
        absol = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            absol[i] = nullptr;
        cart = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            cart[i] = nullptr;
        cart_surr = new Val_domain*[ndim];
        for (int i = 0; i < ndim; i++)
            cart_surr[i] = nullptr;
        radius = nullptr;
    }

    void Domain::validate_spectral_parity(const Base_spectral& base) const
    {
        if (spectral_parity_guard_checked || !base.is_def())
            return;

        enum class RequiredParity { None, Odd, Even };
        std::string domain_type = typeid(*this).name();
        int demangle_status = 0;
        char* demangled_type = abi::__cxa_demangle(domain_type.c_str(), nullptr, nullptr, &demangle_status);
        if (demangled_type != nullptr) {
            domain_type = demangled_type;
            std::free(demangled_type);
        }
        const std::size_t scope = domain_type.rfind("::");
        if (scope != std::string::npos)
            domain_type.erase(0, scope + 2);
        if (domain_type.empty())
            domain_type = "Domain";

        for (int axis = 0; axis < ndim; ++axis) {
            const Array<int>* axis_base = base.get_base_1d(axis);
            if (axis_base == nullptr || axis_base->get_nbr() == 0)
                continue;

            RequiredParity required = RequiredParity::None;
            Index index(axis_base->get_dimensions());
            do {
                RequiredParity entry = RequiredParity::None;
                switch ((*axis_base)(index)) {
                    case CHEB:
                    case CHEB_EVEN:
                    case CHEB_ODD:
                    case COS:
                    case COS_EVEN:
                    case COS_ODD:
                    case SIN:
                    case SIN_EVEN:
                    case SIN_ODD:
                        entry = RequiredParity::Odd;
                        break;
                    case COSSIN:
                    case COSSIN_EVEN:
                    case COSSIN_ODD:
                        entry = RequiredParity::Even;
                        break;
                    default:
                        break;
                }

                if (entry != RequiredParity::None) {
                    if (required != RequiredParity::None && required != entry) {
                        std::ostringstream message;
                        message << "Spectral parity guard found mixed basis contracts for "
                                << domain_type << " axis " << axis;
                        KADATH_THROW(message.str());
                    }
                    required = entry;
                }
            } while (index.inc());

            if (required == RequiredParity::None)
                continue;

            const bool parity_matches =
                (required == RequiredParity::Odd) ? (nbr_points(axis) % 2 != 0) : (nbr_points(axis) % 2 == 0);
            if (!parity_matches) {
                std::ostringstream message;
                message << "Spectral parity guard rejected " << domain_type << " axis " << axis << ": given "
                        << nbr_points(axis) << " points; "
                        << ((required == RequiredParity::Odd) ? "folding bases" : "COSSIN bases")
                        << " require an " << ((required == RequiredParity::Odd) ? "odd" : "even")
                        << " line length";
                KADATH_THROW(message.str());
            }
        }

        spectral_parity_guard_checked = true;
    }

    void Domain::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
    }

    // assignement operator : not implemented
    void Domain::operator=(const Domain&)
    {
        KADATH_THROW("Domain::operator= not implemented...");
    }

    // destructor of the derived quantities
    void Domain::del_deriv() const
    {
        for (int l = 0; l < ndim; l++) {
            if (coloc[l] != nullptr)
                delete coloc[l];
            if (absol[l] != nullptr)
                delete absol[l];
            if (cart[l] != nullptr)
                delete cart[l];
            if (cart_surr[l] != nullptr)
                delete cart_surr[l];
            coloc[l] = nullptr;
            absol[l] = nullptr;
            cart_surr[l] = nullptr;
            cart[l] = nullptr;
        }
        if (radius != nullptr)
            delete radius;
        radius = nullptr;
    }
    // Returns absolute coordinates
    Val_domain Domain::get_absol(int i) const
    {
        assert((i > 0) && (i <= ndim));
        if (absol[i - 1] == nullptr)
            do_absol();
        return *absol[i - 1];
    }

    // Returns the generalized radius
    Val_domain Domain::get_radius() const
    {
        if (radius == nullptr)
            do_radius();
        return *radius;
    }
    // Returns cartesian coordinate
    Val_domain Domain::get_cart(int i) const
    {
        assert((i > 0) && (i <= ndim));
        if (cart[i - 1] == nullptr)
            do_cart();
        return *cart[i - 1];
    }

    // Returns cartesian coordinate over radius
    Val_domain Domain::get_cart_surr(int i) const
    {
        assert((i > 0) && (i <= ndim));
        if (cart_surr[i - 1] == nullptr)
            do_cart_surr();
        return *cart_surr[i - 1];
    }

    Array<double> Domain::get_coloc(int i) const
    {
        assert((i > 0) && (i <= ndim));
        assert(coloc[i - 1] != nullptr);
        return *coloc[i - 1];
    }

    Val_domain Domain::laplacian(const Val_domain& so, int m) const
    {

        if (m != 0) {
            KADATH_THROW("In the general case, Laplacian must be called with m=0");
        }

        Val_domain res(so.der_abs(1).der_abs(1));
        for (int j = 2; j <= get_ndim(); j++)
            res += so.der_abs(j).der_abs(j);
        return res;
    }

    Val_domain Domain::laplacian2(const Val_domain&, int) const
    {
        cerr << "laplacian2 not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    bool Domain::import_lanes_point_major(int numdom, int bound, int n_ope, const Array<int>& zedoms,
                                          int tensor_lane_count, Tensor* const* lane_parts,
                                          Tensor** lane_results, ImportLanePlanLayout layout,
                                          double lookup_precision, bool force_cartesian_basis) const
    {
        if (get_ndim() != 3 || tensor_lane_count < 2 ||
            tensor_lane_count > Term_eq::max_derivative_lanes + 1 || n_ope <= 0)
            return false;
        if (layout == ImportLanePlanLayout::RadialBoundary && bound != INNER_BC && bound != OUTER_BC)
            return false;
        if (layout == ImportLanePlanLayout::BisphericRectOuter && bound != OUTER_BC)
            return false;
        for (int part = 0; part < n_ope; ++part)
            if (lane_parts[part] == nullptr)
                return false;

        const Tensor& model = *lane_parts[0];
        for (int lane = 0; lane < tensor_lane_count; ++lane) {
            for (int part = 0; part < n_ope; ++part) {
                Tensor* input = lane_parts[lane * n_ope + part];
                if (input == nullptr)
                    continue;
                if (!derivative_lane_detail::same_tensor_component_layout(model, *input))
                    return false;
                if (input->get_valence() != 0 &&
                    input->get_basis().get_basis(zedoms(part)) != CARTESIAN_BASIS)
                    return false;
            }
        }

        for (int lane = 0; lane < tensor_lane_count; ++lane)
            for (int part = 0; part < n_ope; ++part) {
                Tensor* input = lane_parts[lane * n_ope + part];
                if (input == nullptr)
                    continue;
                for (int component = 0; component < input->get_n_comp(); ++component)
                    input->cmp[component]->set_domain(zedoms(part)).coef();
            }

        const Dim_array points(get_nbr_points());
        const Space& space = model.get_space();
        ImportPlanCache::Key key;
        key.target_domain = numdom;
        key.bound = bound;
        key.layout = static_cast<int>(layout);
        key.n_ope = n_ope;
        key.lookup_precision_bits = import_double_bits(lookup_precision);
        key.source_domains.reserve(static_cast<std::size_t>(n_ope));
        key.source_shape_fingerprints.reserve(static_cast<std::size_t>(n_ope));

        bool cacheable = true;
        for (int part = 0; part < n_ope; ++part) {
            key.source_domains.push_back(zedoms(part));
            std::uint64_t fingerprint = import_hash_offset;
            if (!space.get_domain(zedoms(part))->fingerprint_import_shape(fingerprint))
                cacheable = false;
            key.source_shape_fingerprints.push_back(fingerprint);
        }

        const std::vector<ImportPlanCache::PointPlan>* plan = nullptr;
        if (cacheable && import_plan_cache_ != nullptr) {
            for (const ImportPlanCache::Entry& entry : import_plan_cache_->entries) {
                if (entry.key == key) {
                    plan = &entry.points;
                    break;
                }
            }
        }

        std::vector<ImportPlanCache::PointPlan> rebuilt_plan;
        if (plan != nullptr) {
            ++matching_lane_stats_state().import_plan_cache_hits;
        } else {
            ++matching_lane_stats_state().import_plan_cache_misses;
            ++matching_lane_stats_state().import_plan_cache_rebuilds;

            const Val_domain xx(get_cart(1));
            const Val_domain yy(get_cart(2));
            const Val_domain zz(get_cart(3));
            Index boundary_point(points);

            const auto append_point = [&](int first, int second, int i0, int i1, int i2) {
                boundary_point.set(0) = i0;
                boundary_point.set(1) = i1;
                boundary_point.set(2) = i2;
                Point absolute(3);
                absolute.set(1) = xx(boundary_point);
                absolute.set(2) = yy(boundary_point);
                absolute.set(3) = zz(boundary_point);
                int source = 0;
                while (source < n_ope &&
                       !space.get_domain(zedoms(source))->is_in(absolute, lookup_precision))
                    ++source;
                if (source == n_ope)
                    return false;
                const Point numerical =
                    space.get_domain(zedoms(source))->absol_to_num(absolute);
                rebuilt_plan.push_back({first, second, zedoms(source),
                                        {numerical(1), numerical(2), numerical(3)}});
                return true;
            };

            if (layout == ImportLanePlanLayout::RadialBoundary) {
                const int radial = (bound == INNER_BC) ? 0 : points(0) - 1;
                for (int k = 0; k < points(2); ++k)
                    for (int j = 0; j < points(1); ++j)
                        if (!append_point(j, k, radial, j, k))
                            return false;
            } else {
                const int radial = points(0) - 1;
                const int angular = points(1) - 1;
                for (int k = 0; k < points(2); ++k)
                    if (!append_point(0, k, radial, angular, k))
                        return false;
            }

            if (cacheable) {
                if (import_plan_cache_ == nullptr)
                    import_plan_cache_ = std::make_unique<ImportPlanCache>();
                if (import_plan_cache_->entries.size() == ImportPlanCache::max_entries)
                    import_plan_cache_->entries.erase(import_plan_cache_->entries.begin());
                import_plan_cache_->entries.push_back({std::move(key), std::move(rebuilt_plan)});
                plan = &import_plan_cache_->entries.back().points;
            } else {
                plan = &rebuilt_plan;
            }
        }

        std::vector<std::unique_ptr<Tensor>> outputs;
        outputs.reserve(static_cast<std::size_t>(tensor_lane_count));
        for (int lane = 0; lane < tensor_lane_count; ++lane) {
            // The scalar fallback models each derivative result on part zero's
            // derivative, or on part zero's value for a missing lane.
            Tensor* lane_model = lane_parts[lane * n_ope];
            outputs.push_back(std::make_unique<Tensor>(lane_model == nullptr ? model : *lane_model, false));
            for (int component = 0; component < model.get_n_comp(); ++component) {
                const Array<int> index(model.indices(component));
                outputs.back()->set(index).set_domain(numdom).allocate_conf();
            }
        }

        Index target(points);
        Point numerical(3);
        for (const ImportPlanCache::PointPlan& point : *plan) {
            numerical.set(1) = point.numerical[0];
            numerical.set(2) = point.numerical[1];
            numerical.set(3) = point.numerical[2];
            int source_part = 0;
            while (source_part < n_ope && zedoms(source_part) != point.source_domain)
                ++source_part;
            if (source_part == n_ope)
                KADATH_THROW("cached import owner is absent from the source-domain set");
            for (int component = 0; component < model.get_n_comp(); ++component) {
                for (int lane = 0; lane < tensor_lane_count; ++lane) {
                    Tensor* input = lane_parts[lane * n_ope + source_part];
                    double value = 0.0;
                    if (input != nullptr) {
                        const Val_domain& field = (*input->cmp[component])(point.source_domain);
                        if (!field.check_if_zero())
                            value = field.get_base().summation(numerical, field.get_coef_ref());
                    }
                    if (layout == ImportLanePlanLayout::RadialBoundary) {
                        for (int radial = 0; radial < points(0); ++radial) {
                            target.set(0) = radial;
                            target.set(1) = point.first;
                            target.set(2) = point.second;
                            outputs[static_cast<std::size_t>(lane)]->cmp[component]
                                ->set_domain(numdom).set(target) = value;
                        }
                    } else {
                        for (int radial = 0; radial < points(0); ++radial)
                            for (int angular = 0; angular < points(1); ++angular) {
                                target.set(0) = radial;
                                target.set(1) = angular;
                                target.set(2) = point.second;
                                outputs[static_cast<std::size_t>(lane)]->cmp[component]
                                    ->set_domain(numdom).set(target) = value;
                            }
                    }
                }
            }
        }

        matching_lane_detail::finalize_then_release(
            outputs, lane_results,
            [&](Tensor& output, std::size_t) {
                if (force_cartesian_basis)
                    output.set_basis(numdom) = CARTESIAN_BASIS;
                output.std_base();
            });
        matching_lane_stats_state().import_plan_points += static_cast<long long>(plan->size());
        return true;
    }

    bool Domain::import_lanes_native(int, int, int, const Array<int>&, int, Tensor* const*, Tensor**) const
    {
        return false;
    }

    Term_eq Domain::import(int numdom, int bound, int n_ope, Term_eq** parts) const
    {

        // get indices of domains :
        Array<int> zedoms(n_ope);
        for (int i = 0; i < n_ope; i++)
            zedoms.set(i) = parts[i]->get_dom();

        if (matching_import_lane_batch_enabled()) {
            bool native_derivatives = true;
            int lane_count = 1;
            for (int i = 0; i < n_ope; ++i) {
                native_derivatives = native_derivatives && parts[i]->get_p_der_t() != nullptr &&
                                     derivative_lane_detail::all_derivative_lanes_match_value_layout(*parts[i]);
                lane_count = std::max(lane_count, parts[i]->get_derivative_lane_count());
            }
            // A single tangent is the scalar Jacobian path. Keep that on the
            // established virtual import implementation: batching is useful
            // only when one immutable map can serve multiple lanes.
            if (lane_count > 1) {
                if (native_derivatives) {
                    const int tensor_lane_count = lane_count + 1;
                    std::vector<Tensor*> lane_parts(static_cast<std::size_t>(tensor_lane_count * n_ope), nullptr);
                    for (int part = 0; part < n_ope; ++part) {
                        lane_parts[static_cast<std::size_t>(part)] =
                            const_cast<Tensor*>(parts[part]->get_p_val_t());
                        for (int lane = 0; lane < lane_count; ++lane) {
                            Tensor* derivative = const_cast<Tensor*>(parts[part]->get_p_der_t(lane));
                            lane_parts[static_cast<std::size_t>((lane + 1) * n_ope + part)] = derivative;
                            if (derivative == nullptr)
                                ++matching_lane_stats_state().import_missing_inputs;
                        }
                    }
                    std::vector<Tensor*> raw_results(static_cast<std::size_t>(tensor_lane_count), nullptr);
                    if (import_lanes_native(numdom, bound, n_ope, zedoms, tensor_lane_count, lane_parts.data(),
                                            raw_results.data())) {
                        std::vector<std::unique_ptr<Tensor>> results;
                        results.reserve(raw_results.size());
                        for (Tensor* result : raw_results)
                            results.emplace_back(result);
                        Term_eq packed(numdom, *results[0], *results[1]);
                        packed.set_derivative_lane_count(lane_count);
                        for (int lane = 1; lane < lane_count; ++lane)
                            packed.set_der_t(lane, *results[static_cast<std::size_t>(lane + 1)]);
                        ++matching_lane_stats_state().import_native_calls;
                        return packed;
                    }
                    for (Tensor* result : raw_results)
                        delete result;
                    ++matching_lane_stats_state().import_refusals;
                } else {
                    ++matching_lane_stats_state().import_refusals;
                }
            }
        }
        ++matching_lane_stats_state().import_scalar_fallback_calls;

        // Get val part :
        std::vector<Tensor*> parts_val(n_ope);
        for (int i = 0; i < n_ope; i++)
            parts_val[i] = parts[i]->val_t;

        Tensor res_val(import(numdom, bound, n_ope, zedoms, parts_val.data()));

        // Do the derivative parts ?
        bool doder = true;
        for (int i = 0; i < n_ope; i++)
            if (parts[i]->der_t == nullptr)
                doder = false;

        if (doder) {
            std::vector<Tensor*> parts_der(n_ope);
            for (int i = 0; i < n_ope; i++)
                parts_der[i] = parts[i]->der_t;
            Tensor res_der(import(numdom, bound, n_ope, zedoms, parts_der.data()));
            Term_eq result(numdom, res_val, res_der);
            int lane_count = 1;
            for (int i = 0; i < n_ope; i++)
                lane_count = std::max(lane_count, parts[i]->get_derivative_lane_count());
            result.set_derivative_lane_count(lane_count);
            for (int lane = 1; lane < lane_count; ++lane) {
                std::vector<Tensor*> parts_lane(n_ope);
                std::vector<std::unique_ptr<Tensor>> owned_lane(n_ope);
                for (int i = 0; i < n_ope; i++) {
                    if (parts[i]->has_der_t(lane)) {
                        parts_lane[i] = const_cast<Tensor*>(parts[i]->get_p_der_t(lane));
                    } else {
                        owned_lane[i] = std::make_unique<Tensor>(*parts[i]->val_t, false);
                        parts_lane[i] = owned_lane[i].get();
                        for (int component = 0; component < parts_lane[i]->get_n_comp(); ++component) {
                            const Array<int> index = parts_lane[i]->indices(component);
                            parts_lane[i]->set(index).set_domain(zedoms(i)).set_zero();
                        }
                    }
                }
                Tensor res_lane(import(numdom, bound, n_ope, zedoms, parts_lane.data()));
                result.set_der_t(lane, res_lane);
            }
            return result;
        } else
            return Term_eq(numdom, res_val);
    }

    // Term eq version of operators
    Term_eq Domain::der_normal_term_eq(const Term_eq& so, int bound) const
    {
        return do_comp_by_comp_with_int(so, bound, &Domain::der_normal);
    }

    Term_eq Domain::dr_term_eq(const Term_eq& so) const
    {
        return do_comp_by_comp(so, &Domain::der_r);
    }

    Term_eq Domain::dtime_term_eq(const Term_eq& so) const
    {
        return do_comp_by_comp(so, &Domain::dtime);
    }

    Term_eq Domain::ddtime_term_eq(const Term_eq& so) const
    {
        return do_comp_by_comp(so, &Domain::ddtime);
    }

    Term_eq Domain::lap_term_eq(const Term_eq& so, int mm) const
    {
        return do_comp_by_comp_with_int(so, mm, &Domain::laplacian);
    }

    Term_eq Domain::lap2_term_eq(const Term_eq& so, int mm) const
    {
        return do_comp_by_comp_with_int(so, mm, &Domain::laplacian2);
    }

    Term_eq Domain::mult_r_term_eq(const Term_eq& so) const
    {
        return do_comp_by_comp(so, &Domain::mult_r);
    }

    Term_eq Domain::div_r_term_eq(const Term_eq& so) const
    {
        return do_comp_by_comp(so, &Domain::div_r);
    }

    Term_eq Domain::grad_term_eq(const Term_eq& so) const
    {

        Term_eq res(partial_cart(so));
        if (so.val_t->is_m_quant_affected()) {
            res.val_t->affect_parameters();
            res.val_t->set_parameters()->set_m_quant() = so.val_t->get_parameters()->get_m_quant();
        }

        for (int lane = 0; lane < so.get_derivative_lane_count(); ++lane) {
            if (so.has_der_t(lane) && res.has_der_t(lane) && so.get_der_t(lane).is_m_quant_affected()) {
                res.set_der_t(lane)->affect_parameters();
                res.set_der_t(lane)->set_parameters()->set_m_quant() =
                    so.get_der_t(lane).get_parameters()->get_m_quant();
            }
        }
        return res;
    }
    Term_eq Domain::integ_term_eq(const Term_eq& so, int bound) const
    {

        // Check it is a tensor
        if (so.type_data != TERM_T) {
            KADATH_THROW("integ_term_eq only defined with respect for a tensor");
        }

        if (so.val_t->get_n_comp() != 1) {
            KADATH_THROW("integ_term_eq only defined with respect to a scalar");
        }

        assert(so.dom == num_dom);

        // The value
        Array<int> ind(so.val_t->indices(0));
        Val_domain value((*so.val_t)(ind)(so.dom));
        double resval;
        if (value.check_if_zero())
            resval = 0.;
        else
            resval = integ(value, bound);

        auto integrate_boundary_lane = [this, &ind, bound, domain = so.dom](const Tensor& derivative) {
            Val_domain derivative_value(derivative(ind)(domain));
            return derivative_value.check_if_zero() ? 0. : integ(derivative_value, bound);
        };
        return make_double_term_with_derivative_lanes(so.dom, resval, so, integrate_boundary_lane);
    }

    Term_eq Domain::do_comp_by_comp_with_int(const Term_eq& target, int val,
                                             Val_domain (Domain::*pfunc)(const Val_domain&, int) const) const
    {

        // The value
        const Tensor& target_value = *target.get_p_val_t();
        Tensor resval(target_value, false);

        for (int i = 0; i < target_value.get_n_comp(); i++) {
            const Val_domain& value = (*target_value.cmp[i])(num_dom);
            Scalar& result = *resval.cmp[i];
            if (value.check_if_zero())
                result.set_domain(num_dom).set_zero();
            else
                result.set_domain(num_dom) = (this->*pfunc)(value, val);
        }
        copy_tensor_metadata(target_value, resval);

        if (target.der_t != nullptr) {
            auto apply_derivative_lane = [&](const Tensor& target_derivative) {
                Tensor resder(target_derivative, false);
                for (int i = 0; i < target_derivative.get_n_comp(); i++) {
                    const Val_domain& value = (*target_derivative.cmp[i])(num_dom);
                    Scalar& result = *resder.cmp[i];
                    if (value.check_if_zero())
                        result.set_domain(num_dom).set_zero();
                    else
                        result.set_domain(num_dom) = (this->*pfunc)(value, val);
                }
                copy_tensor_metadata(target_derivative, resder);
                return resder;
            };

            Tensor resder(apply_derivative_lane(*target.get_p_der_t()));
            Term_eq res(num_dom, resval, resder);
            res.set_derivative_lane_count(target.get_derivative_lane_count());
            for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane) {
                if (target.has_der_t(lane))
                    res.set_der_t(lane, apply_derivative_lane(target.get_der_t(lane)));
            }
            return res;
        } else {
            Term_eq res(num_dom, resval);
            return res;
        }
    }

    Term_eq Domain::do_comp_by_comp(const Term_eq& target, Val_domain (Domain::*pfunc)(const Val_domain&) const) const
    {

        // The value
        const Tensor& target_value = *target.get_p_val_t();
        Tensor resval(target_value, false);

        for (int i = 0; i < target_value.get_n_comp(); i++) {
            const Val_domain& value = (*target_value.cmp[i])(num_dom);
            Scalar& result = *resval.cmp[i];
            if (value.check_if_zero())
                result.set_domain(num_dom).set_zero();
            else
                result.set_domain(num_dom) = (this->*pfunc)(value);
        }
        copy_tensor_metadata(target_value, resval);

        if (target.der_t != nullptr) {
            auto apply_derivative_lane = [&](const Tensor& target_derivative) {
                Tensor resder(target_derivative, false);
                for (int i = 0; i < target_derivative.get_n_comp(); i++) {
                    const Val_domain& value = (*target_derivative.cmp[i])(num_dom);
                    Scalar& result = *resder.cmp[i];
                    if (value.check_if_zero())
                        result.set_domain(num_dom).set_zero();
                    else
                        result.set_domain(num_dom) = (this->*pfunc)(value);
                }
                copy_tensor_metadata(target_derivative, resder);
                return resder;
            };

            Tensor resder(apply_derivative_lane(*target.get_p_der_t()));
            Term_eq res(num_dom, resval, resder);
            res.set_derivative_lane_count(target.get_derivative_lane_count());
            for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane) {
                if (target.has_der_t(lane))
                    res.set_der_t(lane, apply_derivative_lane(target.get_der_t(lane)));
            }
            return res;
        } else {
            Term_eq res(num_dom, resval);
            return res;
        }
    }

    Term_eq Domain::partial_cart(const Term_eq& so) const
    {
        int dom = so.get_dom();
        int val_res = so.val_t->get_valence() + 1;
        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = CARTESIAN_BASIS;

        // Tensor for val
        Tensor auxi_val(so.get_val_t().get_space(), val_res, type_ind, basis);

        const Tensor& source_value = *so.val_t;
        const int valence = source_value.get_valence();
        const int derivative_axes = source_value.get_ndim();

        auto fill_cartesian_partial = [dom, valence, derivative_axes](const Tensor& source, Tensor& target) {
            if (valence == 0) {
                const Scalar& source_scalar = *source.cmp[0];
                for (int axis = 0; axis < derivative_axes; axis++)
                    target.cmp[axis]->set_domain(dom) = source_scalar(dom).der_abs(axis + 1);
                return;
            }

            Index source_component(source);
            Index target_component(target);
            do {
                for (int index = 0; index < valence; index++)
                    target_component.set(index + 1) = source_component(index);

                const Scalar& source_scalar = *source.cmp[source.position(source_component)];
                for (int axis = 0; axis < derivative_axes; axis++) {
                    target_component.set(0) = axis;
                    target.cmp[target.position(target_component)]->set_domain(dom) = source_scalar(dom).der_abs(axis + 1);
                }
            } while (source_component.inc());
        };

        fill_cartesian_partial(source_value, auxi_val);

        // Prime all active tangent lanes together; the existing materializer
        // below then reads the committed per-field der_abs caches unchanged.
        prepare_cartesian_derivative_lanes(dom, so);

        return make_tensor_term_with_derivative_lanes(dom, so, auxi_val, fill_cartesian_partial);
    }

    Term_eq Domain::partial_spher(const Term_eq& so) const
    {

        int dom = so.get_dom();
        int val_res = so.val_t->get_valence() + 1;
        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = SPHERICAL_BASIS;

        // Tensor for val
        Tensor auxi_val(so.get_val_t().get_space(), val_res, type_ind, basis);

        auto fill_spherical_partial = [dom](const Tensor& source, Tensor& target) {
            const int source_valence = source.get_valence();
            if (source_valence == 0) {
                const Val_domain& source_scalar = (*source.cmp[0])(dom);
                target.cmp[0]->set_domain(dom) = source_scalar.der_r();
                target.cmp[1]->set_domain(dom) = source_scalar.der_var(2).div_r();
                target.cmp[2]->set_domain(dom) = source_scalar.der_var(3).div_r().div_sin_theta();
                return;
            }

            Index source_component(source);
            Index target_component(target);
            do {
                for (int i = 0; i < source_valence; i++)
                    source_component.set(i) = target_component(i + 1);

                switch (target_component(0)) {
                case 0:
                    // d/dr :
                    target.set(target_component).set_domain(dom) = source(source_component)(dom).der_r();
                    break;
                case 1:
                    // 1/r dtheta
                    target.set(target_component).set_domain(dom) = source(source_component)(dom).der_var(2).div_r();
                    break;
                case 2:
                    // 1/r sint d/dphi
                    target.set(target_component).set_domain(dom) =
                        source(source_component)(dom).der_var(3).div_r().div_sin_theta();
                    break;
                default:
                    KADATH_THROW("Bad indice in Domain::derive_partial_spher");
                }
            } while (target_component.inc());
        };

        fill_spherical_partial(*so.val_t, auxi_val);

        return make_tensor_term_with_derivative_lanes(dom, so, auxi_val, fill_spherical_partial);
    }

    Term_eq Domain::partial_mtz(const Term_eq& so) const
    {

        int dom = so.get_dom();
        int val_res = so.val_t->get_valence() + 1;
        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = MTZ_BASIS;

        // Tensor for val
        Tensor auxi_val(so.get_val_t().get_space(), val_res, type_ind, basis);

        auto fill_mtz_partial = [dom, val_res](const Tensor& source, Tensor& target) {
            Index target_component(target);
            Index source_component(source);
            do {
                for (int i = 0; i < val_res - 1; i++)
                    source_component.set(i) = target_component(i + 1);
                switch (target_component(0)) {
                case 0:
                    // d/dr :
                    target.set(target_component).set_domain(dom) = source(source_component)(dom).der_r();
                    break;
                case 1:
                    // cost/r dtheta
                    target.set(target_component).set_domain(dom) =
                        source(source_component)(dom).der_var(2).div_r().mult_cos_theta();
                    break;
                case 2:
                    // cost/r sint d/dphi
                    target.set(target_component).set_domain(dom) =
                        source(source_component)(dom).der_var(3).div_r().div_sin_theta().mult_cos_theta();
                    break;
                default:
                    KADATH_THROW("Bad indice in Domain::derive_partial_mtz");
                }
            } while (target_component.inc());
        };

        fill_mtz_partial(*so.val_t, auxi_val);

        return make_tensor_term_with_derivative_lanes(dom, so, auxi_val, fill_mtz_partial);
    }

    Term_eq Domain::connection_spher(const Term_eq& so) const
    {

        int dom = so.get_dom();
        assert(dom == num_dom);

        int valence = so.val_t->get_valence();
        int val_res = so.val_t->get_valence() + 1;

        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = SPHERICAL_BASIS;

        Tensor auxi_val(so.get_val_t().get_space(), val_res, type_ind, basis, 3);
        for (int cmp = 0; cmp < auxi_val.get_n_comp(); cmp++)
            auxi_val.set(auxi_val.indices(cmp)) = 0;

        for (int ind_sum = 0; ind_sum < valence; ind_sum++) {

            // Loop on the components :
            Index pos_auxi(auxi_val);
            Index pos_so(*so.val_t);

            do {
                for (int i = 0; i < valence; i++)
                    pos_so.set(i) = pos_auxi(i + 1);
                // Different cases of the derivative index :
                switch (pos_auxi(0)) {
                    case 0:
                        // Dr nothing
                        break;
                    case 1:
                        // Dtheta
                        // Different cases of the source index
                        switch (pos_auxi(ind_sum + 1)) {
                            case 0:
                                // Dtheta S_r
                                pos_so.set(ind_sum) = 1;
                                auxi_val.set(pos_auxi).set_domain(dom) -= (*so.val_t)(pos_so)(dom).div_r();
                                break;
                            case 1:
                                // Dtheta S_theta
                                pos_so.set(ind_sum) = 0;
                                auxi_val.set(pos_auxi).set_domain(dom) += (*so.val_t)(pos_so)(dom).div_r();
                                break;
                            case 2:
                                // Dtheta S_phi
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain::connection_spher");
                        }
                        break;
                    case 2:
                        // Dphi
                        // Different cases of the source index
                        switch (pos_auxi(ind_sum + 1)) {
                            case 0:
                                // Dphi S_r
                                pos_so.set(ind_sum) = 2;
                                auxi_val.set(pos_auxi).set_domain(dom) -= (*so.val_t)(pos_so)(dom).div_r();
                                break;
                            case 1:
                                // Dphi S_theta
                                pos_so.set(ind_sum) = 2;
                                auxi_val.set(pos_auxi).set_domain(dom) -=
                                    (*so.val_t)(pos_so)(dom).div_r().mult_cos_theta().div_sin_theta();
                                break;
                            case 2:
                                // Dphi S_phi
                                pos_so.set(ind_sum) = 0;
                                auxi_val.set(pos_auxi).set_domain(dom) += (*so.val_t)(pos_so)(dom).div_r();
                                pos_so.set(ind_sum) = 1;
                                auxi_val.set(pos_auxi).set_domain(dom) +=
                                    (*so.val_t)(pos_so)(dom).div_r().mult_cos_theta().div_sin_theta();
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain::connection_spher");
                        }
                        break;
                    default:
                        KADATH_THROW("Bad indice in Domain::connection_spher");
                }
            } while (pos_auxi.inc());
        }

        if (so.der_t == nullptr) {
            // No need for derivative :
            return Term_eq(dom, auxi_val);
        } else {

            // Need to compute the derivative :
            // Tensor for der
            Tensor auxi_der(so.get_val_t().get_space(), val_res, type_ind, basis, 3);
            for (int cmp = 0; cmp < auxi_der.get_n_comp(); cmp++)
                auxi_der.set(auxi_der.indices(cmp)) = 0;

            // Loop indice summation on connection symbols
            for (int ind_sum = 0; ind_sum < valence; ind_sum++) {

                // Loop on the components :
                Index pos_auxi_der(auxi_der);
                Index pos_so(*so.der_t);

                do {
                    for (int i = 0; i < valence; i++)
                        pos_so.set(i) = pos_auxi_der(i + 1);
                    // Different cases of the derivative index :
                    switch (pos_auxi_der(0)) {
                        case 0:
                            // Dr nothing
                            break;
                        case 1:
                            // Dtheta
                            // Different cases of the source index
                            switch (pos_auxi_der(ind_sum + 1)) {
                                case 0:
                                    // Dtheta S_r
                                    pos_so.set(ind_sum) = 1;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) -= (*so.der_t)(pos_so)(dom).div_r();
                                    break;
                                case 1:
                                    // Dtheta S_theta
                                    pos_so.set(ind_sum) = 0;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) += (*so.der_t)(pos_so)(dom).div_r();
                                    break;
                                case 2:
                                    // Dtheta S_phi
                                    break;
                                default:
                                    KADATH_THROW("Bad indice in Domain::connection_spher");
                            }
                            break;
                        case 2:
                            // Dphi
                            // Different cases of the source index
                            switch (pos_auxi_der(ind_sum + 1)) {
                                case 0:
                                    // Dphi S_r
                                    pos_so.set(ind_sum) = 2;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) -= (*so.der_t)(pos_so)(dom).div_r();
                                    break;
                                case 1:
                                    // Dphi S_theta
                                    pos_so.set(ind_sum) = 2;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) -=
                                        (*so.der_t)(pos_so)(dom).div_r().mult_cos_theta().div_sin_theta();
                                    break;
                                case 2:
                                    // Dphi S_phi
                                    pos_so.set(ind_sum) = 0;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) += (*so.der_t)(pos_so)(dom).div_r();
                                    pos_so.set(ind_sum) = 1;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) +=
                                        (*so.der_t)(pos_so)(dom).div_r().mult_cos_theta().div_sin_theta();
                                    break;
                                default:
                                    KADATH_THROW("Bad indice in Domain::connection_spher");
                            }
                            break;
                        default:
                            KADATH_THROW("Bad indice in Domain::connection_spher");
                    }
                } while (pos_auxi_der.inc());
            }

            return Term_eq(dom, auxi_val, auxi_der);
        }
    }

    Term_eq Domain::connection_mtz(const Term_eq& so) const
    {

        int dom = so.get_dom();
        assert(dom == num_dom);

        int valence = so.val_t->get_valence();
        int val_res = so.val_t->get_valence() + 1;

        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = MTZ_BASIS;

        Tensor auxi_val(so.get_val_t().get_space(), val_res, type_ind, basis, 3);
        for (int cmp = 0; cmp < auxi_val.get_n_comp(); cmp++)
            auxi_val.set(auxi_val.indices(cmp)) = 0;

        for (int ind_sum = 0; ind_sum < valence; ind_sum++) {

            // Loop on the components :
            Index pos_auxi(auxi_val);
            Index pos_so(*so.val_t);

            do {
                for (int i = 0; i < valence; i++)
                    pos_so.set(i) = pos_auxi(i + 1);
                // Different cases of the derivative index :
                switch (pos_auxi(0)) {
                    case 0:
                        // Dr nothing
                        break;
                    case 1:
                        // Dtheta
                        // Different cases of the source index
                        switch (pos_auxi(ind_sum + 1)) {
                            case 0:
                                // Dtheta S_r
                                pos_so.set(ind_sum) = 1;
                                auxi_val.set(pos_auxi).set_domain(dom) -= (*so.val_t)(pos_so)(dom).div_r();
                                break;
                            case 1:
                                // Dtheta S_theta
                                pos_so.set(ind_sum) = 0;
                                auxi_val.set(pos_auxi).set_domain(dom) += (*so.val_t)(pos_so)(dom).div_r();
                                break;
                            case 2:
                                // Dtheta S_phi
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain::connection_mtz");
                        }
                        break;
                    case 2:
                        // Dphi
                        // Different cases of the source index
                        switch (pos_auxi(ind_sum + 1)) {
                            case 0:
                                // Dphi S_r
                                pos_so.set(ind_sum) = 2;
                                auxi_val.set(pos_auxi).set_domain(dom) -= (*so.val_t)(pos_so)(dom).div_r();
                                break;
                            case 1:
                                // Dphi S_theta
                                pos_so.set(ind_sum) = 2;
                                auxi_val.set(pos_auxi).set_domain(dom) -=
                                    (*so.val_t)(pos_so)(dom).div_r().div_sin_theta();
                                break;
                            case 2:
                                // Dphi S_phi
                                pos_so.set(ind_sum) = 0;
                                auxi_val.set(pos_auxi).set_domain(dom) += (*so.val_t)(pos_so)(dom).div_r();
                                pos_so.set(ind_sum) = 1;
                                auxi_val.set(pos_auxi).set_domain(dom) +=
                                    (*so.val_t)(pos_so)(dom).div_r().div_sin_theta();
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain::connection_mtz");
                        }
                        break;
                    default:
                        KADATH_THROW("Bad indice in Domain::connection_mtz");
                }
            } while (pos_auxi.inc());
        }

        if (so.der_t == nullptr) {
            // No need for derivative :
            return Term_eq(dom, auxi_val);
        } else {

            // Need to compute the derivative :
            // Tensor for der
            Tensor auxi_der(so.get_val_t().get_space(), val_res, type_ind, basis, 3);
            for (int cmp = 0; cmp < auxi_der.get_n_comp(); cmp++)
                auxi_der.set(auxi_der.indices(cmp)) = 0;

            // Loop indice summation on connection symbols
            for (int ind_sum = 0; ind_sum < valence; ind_sum++) {

                // Loop on the components :
                Index pos_auxi_der(auxi_der);
                Index pos_so(*so.der_t);

                do {
                    for (int i = 0; i < valence; i++)
                        pos_so.set(i) = pos_auxi_der(i + 1);
                    // Different cases of the derivative index :
                    switch (pos_auxi_der(0)) {
                        case 0:
                            // Dr nothing
                            break;
                        case 1:
                            // Dtheta
                            // Different cases of the source index
                            switch (pos_auxi_der(ind_sum + 1)) {
                                case 0:
                                    // Dtheta S_r
                                    pos_so.set(ind_sum) = 1;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) -= (*so.der_t)(pos_so)(dom).div_r();
                                    break;
                                case 1:
                                    // Dtheta S_theta
                                    pos_so.set(ind_sum) = 0;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) += (*so.der_t)(pos_so)(dom).div_r();
                                    break;
                                case 2:
                                    // Dtheta S_phi
                                    break;
                                default:
                                    KADATH_THROW("Bad indice in Domain::connection_mtz");
                            }
                            break;
                        case 2:
                            // Dphi
                            // Different cases of the source index
                            switch (pos_auxi_der(ind_sum + 1)) {
                                case 0:
                                    // Dphi S_r
                                    pos_so.set(ind_sum) = 2;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) -= (*so.der_t)(pos_so)(dom).div_r();
                                    break;
                                case 1:
                                    // Dphi S_theta
                                    pos_so.set(ind_sum) = 2;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) -=
                                        (*so.der_t)(pos_so)(dom).div_r().div_sin_theta();
                                    break;
                                case 2:
                                    // Dphi S_phi
                                    pos_so.set(ind_sum) = 0;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) += (*so.der_t)(pos_so)(dom).div_r();
                                    pos_so.set(ind_sum) = 1;
                                    auxi_der.set(pos_auxi_der).set_domain(dom) +=
                                        (*so.der_t)(pos_so)(dom).div_r().div_sin_theta();
                                    break;
                                default:
                                    KADATH_THROW("Bad indice in Domain::connection_mtz");
                            }
                            break;
                        default:
                            KADATH_THROW("Bad indice in Domain::connection_mtz");
                    }
                } while (pos_auxi_der.inc());
            }

            return Term_eq(dom, auxi_val, auxi_der);
        }
    }

    Term_eq Domain::partial_cart_with_cached_value(const Term_eq& so, const Tensor& cached_val) const
    {
        int dom = so.get_dom();
        const Tensor& source_value = *so.val_t;
        const int valence = source_value.get_valence();
        const int derivative_axes = source_value.get_ndim();

        auto fill_cartesian_partial = [dom, valence, derivative_axes](const Tensor& source, Tensor& target) {
            if (valence == 0) {
                const Scalar& source_scalar = *source.cmp[0];
                for (int axis = 0; axis < derivative_axes; axis++)
                    target.cmp[axis]->set_domain(dom) = source_scalar(dom).der_abs(axis + 1);
                return;
            }

            Index source_component(source);
            Index target_component(target);
            do {
                for (int index = 0; index < valence; index++)
                    target_component.set(index + 1) = source_component(index);

                const Scalar& source_scalar = *source.cmp[source.position(source_component)];
                for (int axis = 0; axis < derivative_axes; axis++) {
                    target_component.set(0) = axis;
                    target.cmp[target.position(target_component)]->set_domain(dom) = source_scalar(dom).der_abs(axis + 1);
                }
            } while (source_component.inc());
        };

        prepare_cartesian_derivative_lanes(dom, so);
        return make_tensor_term_with_derivative_lanes(dom, so, cached_val, fill_cartesian_partial);
    }

    Term_eq Domain::partial_spher_with_cached_value(const Term_eq& so, const Tensor& cached_val) const
    {
        int dom = so.get_dom();

        auto fill_spherical_partial = [dom](const Tensor& source, Tensor& target) {
            const int source_valence = source.get_valence();
            if (source_valence == 0) {
                const Val_domain& source_scalar = (*source.cmp[0])(dom);
                target.cmp[0]->set_domain(dom) = source_scalar.der_r();
                target.cmp[1]->set_domain(dom) = source_scalar.der_var(2).div_r();
                target.cmp[2]->set_domain(dom) = source_scalar.der_var(3).div_r().div_sin_theta();
                return;
            }

            Index source_component(source);
            Index target_component(target);
            do {
                for (int i = 0; i < source_valence; i++)
                    source_component.set(i) = target_component(i + 1);

                switch (target_component(0)) {
                case 0:
                    target.set(target_component).set_domain(dom) = source(source_component)(dom).der_r();
                    break;
                case 1:
                    target.set(target_component).set_domain(dom) = source(source_component)(dom).der_var(2).div_r();
                    break;
                case 2:
                    target.set(target_component).set_domain(dom) =
                        source(source_component)(dom).der_var(3).div_r().div_sin_theta();
                    break;
                default:
                    KADATH_THROW("Bad indice in Domain::partial_spher_with_cached_value");
                }
            } while (target_component.inc());
        };

        return make_tensor_term_with_derivative_lanes(dom, so, cached_val, fill_spherical_partial);
    }

    Term_eq Domain::partial_mtz_with_cached_value(const Term_eq& so, const Tensor& cached_val) const
    {
        int dom = so.get_dom();
        int val_res = so.val_t->get_valence() + 1;

        auto fill_mtz_partial = [dom, val_res](const Tensor& source, Tensor& target) {
            Index target_component(target);
            Index source_component(source);
            do {
                for (int i = 0; i < val_res - 1; i++)
                    source_component.set(i) = target_component(i + 1);
                switch (target_component(0)) {
                case 0:
                    target.set(target_component).set_domain(dom) = source(source_component)(dom).der_r();
                    break;
                case 1:
                    target.set(target_component).set_domain(dom) =
                        source(source_component)(dom).der_var(2).div_r().mult_cos_theta();
                    break;
                case 2:
                    target.set(target_component).set_domain(dom) =
                        source(source_component)(dom).der_var(3).div_r().div_sin_theta().mult_cos_theta();
                    break;
                default:
                    KADATH_THROW("Bad indice in Domain::partial_mtz_with_cached_value");
                }
            } while (target_component.inc());
        };

        return make_tensor_term_with_derivative_lanes(dom, so, cached_val, fill_mtz_partial);
    }

    Term_eq Domain::connection_spher_with_cached_value(const Term_eq& so, const Tensor& cached_val) const
    {
        int dom = so.get_dom();
        assert(dom == num_dom);

        if (so.der_t == nullptr) {
            return Term_eq(dom, cached_val);
        }

        int valence = so.val_t->get_valence();
        int val_res = so.val_t->get_valence() + 1;

        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = SPHERICAL_BASIS;

        Tensor auxi_der(so.get_val_t().get_space(), val_res, type_ind, basis, 3);
        for (int cmp = 0; cmp < auxi_der.get_n_comp(); cmp++)
            auxi_der.set(auxi_der.indices(cmp)) = 0;

        for (int ind_sum = 0; ind_sum < valence; ind_sum++) {
            Index pos_auxi_der(auxi_der);
            Index pos_so(*so.der_t);

            do {
                for (int i = 0; i < valence; i++)
                    pos_so.set(i) = pos_auxi_der(i + 1);
                switch (pos_auxi_der(0)) {
                    case 0:
                        break;
                    case 1:
                        switch (pos_auxi_der(ind_sum + 1)) {
                            case 0:
                                pos_so.set(ind_sum) = 1;
                                auxi_der.set(pos_auxi_der).set_domain(dom) -= (*so.der_t)(pos_so)(dom).div_r();
                                break;
                            case 1:
                                pos_so.set(ind_sum) = 0;
                                auxi_der.set(pos_auxi_der).set_domain(dom) += (*so.der_t)(pos_so)(dom).div_r();
                                break;
                            case 2:
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain::connection_spher_with_cached_value");
                        }
                        break;
                    case 2:
                        switch (pos_auxi_der(ind_sum + 1)) {
                            case 0:
                                pos_so.set(ind_sum) = 2;
                                auxi_der.set(pos_auxi_der).set_domain(dom) -= (*so.der_t)(pos_so)(dom).div_r();
                                break;
                            case 1:
                                pos_so.set(ind_sum) = 2;
                                auxi_der.set(pos_auxi_der).set_domain(dom) -=
                                    (*so.der_t)(pos_so)(dom).div_r().mult_cos_theta().div_sin_theta();
                                break;
                            case 2:
                                pos_so.set(ind_sum) = 0;
                                auxi_der.set(pos_auxi_der).set_domain(dom) += (*so.der_t)(pos_so)(dom).div_r();
                                pos_so.set(ind_sum) = 1;
                                auxi_der.set(pos_auxi_der).set_domain(dom) +=
                                    (*so.der_t)(pos_so)(dom).div_r().mult_cos_theta().div_sin_theta();
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain::connection_spher_with_cached_value");
                        }
                        break;
                    default:
                        KADATH_THROW("Bad indice in Domain::connection_spher_with_cached_value");
                }
            } while (pos_auxi_der.inc());
        }

        return Term_eq(dom, cached_val, auxi_der);
    }

    Term_eq Domain::connection_mtz_with_cached_value(const Term_eq& so, const Tensor& cached_val) const
    {
        int dom = so.get_dom();
        assert(dom == num_dom);

        if (so.der_t == nullptr) {
            return Term_eq(dom, cached_val);
        }

        int valence = so.val_t->get_valence();
        int val_res = so.val_t->get_valence() + 1;

        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        Base_tensor basis(so.get_val_t().get_space());
        basis.set_basis(dom) = MTZ_BASIS;

        Tensor auxi_der(so.get_val_t().get_space(), val_res, type_ind, basis, 3);
        for (int cmp = 0; cmp < auxi_der.get_n_comp(); cmp++)
            auxi_der.set(auxi_der.indices(cmp)) = 0;

        for (int ind_sum = 0; ind_sum < valence; ind_sum++) {
            Index pos_auxi_der(auxi_der);
            Index pos_so(*so.der_t);

            do {
                for (int i = 0; i < valence; i++)
                    pos_so.set(i) = pos_auxi_der(i + 1);
                switch (pos_auxi_der(0)) {
                    case 0:
                        break;
                    case 1:
                        switch (pos_auxi_der(ind_sum + 1)) {
                            case 0:
                                pos_so.set(ind_sum) = 1;
                                auxi_der.set(pos_auxi_der).set_domain(dom) -= (*so.der_t)(pos_so)(dom).div_r();
                                break;
                            case 1:
                                pos_so.set(ind_sum) = 0;
                                auxi_der.set(pos_auxi_der).set_domain(dom) += (*so.der_t)(pos_so)(dom).div_r();
                                break;
                            case 2:
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain::connection_mtz_with_cached_value");
                        }
                        break;
                    case 2:
                        switch (pos_auxi_der(ind_sum + 1)) {
                            case 0:
                                pos_so.set(ind_sum) = 2;
                                auxi_der.set(pos_auxi_der).set_domain(dom) -= (*so.der_t)(pos_so)(dom).div_r();
                                break;
                            case 1:
                                pos_so.set(ind_sum) = 2;
                                auxi_der.set(pos_auxi_der).set_domain(dom) -=
                                    (*so.der_t)(pos_so)(dom).div_r().div_sin_theta();
                                break;
                            case 2:
                                pos_so.set(ind_sum) = 0;
                                auxi_der.set(pos_auxi_der).set_domain(dom) += (*so.der_t)(pos_so)(dom).div_r();
                                pos_so.set(ind_sum) = 1;
                                auxi_der.set(pos_auxi_der).set_domain(dom) +=
                                    (*so.der_t)(pos_so)(dom).div_r().div_sin_theta();
                                break;
                            default:
                                KADATH_THROW("Bad indice in Domain::connection_mtz_with_cached_value");
                        }
                        break;
                    default:
                        KADATH_THROW("Bad indice in Domain::connection_mtz_with_cached_value");
                }
            } while (pos_auxi_der.inc());
        }

        return Term_eq(dom, cached_val, auxi_der);
    }

    Term_eq Domain::integ_volume_term_eq(const Term_eq& target) const
    {

        int dom = target.get_dom();

        // Check it is a tensor
        if (target.type_data != TERM_T) {
            KADATH_THROW("Ope_int_volume only defined with respect for a tensor");
        }

        if (target.val_t->get_n_comp() != 1) {
            KADATH_THROW("Ope_int_volume only defined with respect to a scalar");
        }

        // The value
        Array<int> ind(target.val_t->indices(0));
        Val_domain value((*target.val_t)(ind)(dom));
        double resval;
        if (value.check_if_zero())
            resval = 0.;
        else
            resval = value.get_domain()->integ_volume(value);

        auto integrate_volume_lane = [&ind, dom](const Tensor& derivative) {
            Val_domain derivative_value(derivative(ind)(dom));
            return derivative_value.check_if_zero() ? 0. : derivative_value.get_domain()->integ_volume(derivative_value);
        };
        return make_double_term_with_derivative_lanes(dom, resval, target, integrate_volume_lane);
    }

    // List of all the functions that must be implemented for the concerned types of domain
    bool Domain::is_in(const Point&, double) const
    {
        cerr << "is_in not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    const Point Domain::absol_to_num(const Point&) const
    {
        cerr << "Absol_to_num not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    const Point Domain::absol_to_num_bound(const Point&, int) const
    {
        cerr << "Absol_to_num_bound not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::do_der_abs_from_der_var_lanes(DerAbsLaneBatch& batch) const
    {
        const int dimensions = batch.dimension();
        std::vector<Val_domain*> der_var(static_cast<std::size_t>(dimensions));
        std::vector<Val_domain*> der_abs(static_cast<std::size_t>(dimensions), nullptr);
        for (std::size_t lane = 0; lane < batch.lane_count(); ++lane) {
            std::fill(der_abs.begin(), der_abs.end(), nullptr);
            for (int axis = 0; axis < dimensions; ++axis)
                der_var[static_cast<std::size_t>(axis)] =
                    const_cast<Val_domain*>(&batch.der_var(lane, axis));

            try {
                do_der_abs_from_der_var(der_var.data(), der_abs.data());
            } catch (...) {
                for (Val_domain* value : der_abs)
                    delete value;
                throw;
            }

            try {
                for (int axis = 0; axis < dimensions; ++axis) {
                    Val_domain* value = der_abs[static_cast<std::size_t>(axis)];
                    der_abs[static_cast<std::size_t>(axis)] = nullptr;
                    batch.adopt_legacy_result(lane, axis, value);
                }
            } catch (...) {
                for (Val_domain* value : der_abs)
                    delete value;
                throw;
            }
        }
    }

    void Domain::do_der_abs_from_der_var(Val_domain**, Val_domain**) const
    {
        cerr << "do_der_abs_from_der_var not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Base_spectral Domain::mult(const Base_spectral&, const Base_spectral&) const
    {
        cerr << "mult not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Base_spectral Domain::multiplied_base(const Base_spectral& a, const Base_spectral& b) const
    {
        // An undefined operand short-circuits inside every mult override, so it
        // is not worth a cache slot.
        if (!a.is_def() || !b.is_def())
            return mult(a, b);

        if (basis_mult_cache == nullptr)
            basis_mult_cache = std::make_unique<BasisMultCache>();

        const std::lock_guard<std::mutex> cache_lock(basis_mult_cache->mutex);
        std::vector<int>& key = basis_mult_cache->lookup_key;
        key.clear();

        // Mapping state is deliberately absent: the rules read only the
        // complete, ordered basis layouts of their two operands.
        for (const Base_spectral* operand : {&a, &b}) {
            for (int dim = 0; dim < ndim; dim++) {
                const Array<int>* basis = operand->get_base_1d(dim);
                if (basis == nullptr) {
                    key.push_back(-1);
                    continue;
                }
                const Dim_array& dimensions = basis->get_dimensions();
                key.push_back(dimensions.get_ndim());
                for (int d = 0; d < dimensions.get_ndim(); d++)
                    key.push_back(dimensions(d));
                key.insert(key.end(), basis->get_data(), basis->get_data() + basis->get_nbr());
            }
        }

        for (const BasisMultCache::Entry& entry : basis_mult_cache->entries)
            if (entry.key == key)
                return entry.result;

        Base_spectral result(mult(a, b));
        if (basis_mult_cache->entries.size() >= BasisMultCache::max_entries)
            basis_mult_cache->entries.erase(basis_mult_cache->entries.begin());
        basis_mult_cache->entries.push_back({key, result});
        return result;
    }

    void Domain::do_coloc()
    {
        cerr << "do_coloc not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_base(Base_spectral&) const
    {
        cerr << "Symetric Chebyshev spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    void Domain::set_legendre_base(Base_spectral&) const
    {
        cerr << "Symetric Legendre spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_r_base(Base_spectral&) const
    {
        cerr << "Chebyshev spectral base for r not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_r_base(Base_spectral&) const
    {
        cerr << "Legendre spectral base for r not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_anti_cheb_base(Base_spectral&) const
    {
        cerr << "Anti symetric Chebyshev spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_anti_legendre_base(Base_spectral&) const
    {
        cerr << "Anti symetric Legendre spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    void Domain::set_cheb_base_with_m(Base_spectral&, int) const
    {
        cerr << "Symetric Chebyshev spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_base_with_m(Base_spectral&, int) const
    {
        cerr << "Symetric Legendre spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_anti_cheb_base_with_m(Base_spectral&, int) const
    {
        cerr << "Anti symetric Chebyshev spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_anti_legendre_base_with_m(Base_spectral&, int) const
    {
        cerr << "Anti symetric Legendre spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_xodd_base(Base_spectral&) const
    {
        cerr << "Chebyshev with xodd spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_xodd_base(Base_spectral&) const
    {
        cerr << "Legendre with xodd spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_todd_base(Base_spectral&) const
    {
        cerr << "Chebyshev with todd spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_todd_base(Base_spectral&) const
    {
        cerr << "Legendre with todd spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_xodd_todd_base(Base_spectral&) const
    {
        cerr << "Chebyshev with X and T odd spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_xodd_todd_base(Base_spectral&) const
    {
        cerr << "Odd Legendre with X and T odd spectral base not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_base_r_spher(Base_spectral&) const
    {
        cerr << "Cheb base r not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_base_t_spher(Base_spectral&) const
    {
        cerr << "Cheb base t not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_base_p_spher(Base_spectral&) const
    {
        cerr << "Cheb base p not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    void Domain::set_cheb_base_r_mtz(Base_spectral&) const
    {
        cerr << "Cheb base r MTZ not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_base_t_mtz(Base_spectral&) const
    {
        cerr << "Cheb base t MTZ not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_base_p_mtz(Base_spectral&) const
    {
        cerr << "Cheb base p MTZ not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    void Domain::set_cheb_base_rt_spher(Base_spectral&) const
    {
        cerr << "Cheb base rt not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_base_rp_spher(Base_spectral&) const
    {
        cerr << "Cheb base rp not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_base_tp_spher(Base_spectral&) const
    {
        cerr << "Cheb base tp not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    void Domain::set_legendre_base_r_spher(Base_spectral&) const
    {
        cerr << "Legendre base r not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_base_t_spher(Base_spectral&) const
    {
        cerr << "Legendre base t not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_base_p_spher(Base_spectral&) const
    {
        cerr << "Legendre base p not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_base_r_mtz(Base_spectral&) const
    {
        cerr << "Legendre base r MTZ not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_base_t_mtz(Base_spectral&) const
    {
        cerr << "Legendre base t MTZ not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_base_p_mtz(Base_spectral&) const
    {
        cerr << "Legendre base p MTZ not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    void Domain::set_cheb_base_odd(Base_spectral&) const
    {
        cerr << "Cheb base with odd cosines not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_legendre_base_odd(Base_spectral&) const
    {
        cerr << "Legendre base with odd cosines not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    void Domain::set_cheb_base_xy_cart(Base_spectral&) const
    {
        cerr << "Cheb base xy not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_base_xz_cart(Base_spectral&) const
    {
        cerr << "Cheb base xz not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_cheb_base_yz_cart(Base_spectral&) const
    {
        cerr << "Cheb base yz not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    void Domain::set_cheb_base_x_cart(Base_spectral& ba) const
    {
        // Default version
        set_cheb_base(ba);
    }

    void Domain::set_cheb_base_y_cart(Base_spectral& ba) const
    {
        // Default version
        set_cheb_base(ba);
    }

    void Domain::set_cheb_base_z_cart(Base_spectral& ba) const
    {
        // Default version
        set_anti_cheb_base(ba);
    }

    void Domain::set_legendre_base_x_cart(Base_spectral& ba) const
    {
        // Default version
        set_legendre_base(ba);
    }

    void Domain::set_legendre_base_y_cart(Base_spectral& ba) const
    {
        // Default version
        set_legendre_base(ba);
    }

    void Domain::set_legendre_base_z_cart(Base_spectral& ba) const
    {
        // Default version
        set_anti_legendre_base(ba);
    }

    Val_domain Domain::mult_cos_theta(const Val_domain&) const
    {
        cerr << "Multiplication by cos(theta) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::mult_cos_phi(const Val_domain&) const
    {
        cerr << "Multiplication by cos(phi) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::mult_sin_theta(const Val_domain&) const
    {
        cerr << "Multiplication by sin(theta) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::mult_sin_phi(const Val_domain&) const
    {
        cerr << "Multiplication by sin(phi) not implemented for this type of domain" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::div_sin_theta(const Val_domain&) const
    {
        cerr << "Division by sin(theta) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::div_cos_theta(const Val_domain&) const
    {
        cerr << "Division by cos(theta) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::div_x(const Val_domain&) const
    {
        cerr << "Division by x not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::set_val_inf(Val_domain&, double) const
    {
        cerr << "set_val_inf not defined for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::div_xm1(const Val_domain&) const
    {
        cerr << "Division by (x-1) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::div_xp1(const Val_domain&) const
    {
        cerr << "Division by (x+1) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::div_1mrsL(const Val_domain&) const
    {
        cerr << "Division by 1 - r/L not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::mult_1mrsL(const Val_domain&) const
    {
        cerr << "Multiplication by 1 - r/L  not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::mult_xm1(const Val_domain&) const
    {
        cerr << "Multiplication by (x-1) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::mult_r(const Val_domain&) const
    {
        cerr << "Multiplication by r not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::mult_x(const Val_domain&) const
    {
        cerr << "Multiplication by x not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::mult_cos_time(const Val_domain&) const
    {
        cerr << "Multiplication by cos(time) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::mult_sin_time(const Val_domain&) const
    {
        cerr << "Multiplication by sin(time) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::div_1mx2(const Val_domain&) const
    {
        cerr << "Divison by 1-x^2 not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::srdr(const Val_domain&) const
    {
        cerr << "srdr not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::div_r(const Val_domain&) const
    {
        cerr << "Division by r not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::div_sin_chi(const Val_domain&) const
    {
        cerr << "Division by sin(chi) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::div_chi(const Val_domain&) const
    {
        cerr << "Division by chi not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Tensor Domain::change_basis_cart_to_spher(int, const Tensor&) const
    {
        cerr << "change_basis_cart_to_spher not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Tensor Domain::change_basis_spher_to_cart(int, const Tensor&) const
    {
        cerr << "change_basis_spher_to_cart not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    double Domain::get_rmax() const
    {
        cerr << "No rmax for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    double Domain::get_rmin() const
    {
        cerr << "No rmin for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Point Domain::get_center() const
    {
        cerr << "No center for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    const Val_domain& Domain::get_chi() const
    {
        cerr << "No chi for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    const Val_domain& Domain::get_eta() const
    {
        cerr << "No eta for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::get_X() const
    {
        cerr << "No X for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::get_T() const
    {
        cerr << "No T for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::find_other_dom(int, int, int&, int&) const
    {
        cerr << "find_other_dom not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::der_normal(const Val_domain&, int) const
    {
        cerr << "der_normal not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::der_partial_var(const Val_domain&, int) const
    {
        cerr << "der_partial_var not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::der_r(const Val_domain&) const
    {
        cerr << "der_r not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::der_t(const Val_domain&) const
    {
        cerr << "der_t not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    Val_domain Domain::der_p(const Val_domain&) const
    {
        cerr << "der_p not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::der_r_rtwo(const Val_domain&) const
    {
        cerr << "der_r_rtwo not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::ddr(const Val_domain& so) const
    {
        return so.der_r().der_r();
    }

    Val_domain Domain::ddp(const Val_domain&) const
    {
        cerr << "ddp not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::ddt(const Val_domain&) const
    {
        cerr << "ddt not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::dt(const Val_domain&) const
    {
        cerr << "dt not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::dtime(const Val_domain&) const
    {
        cerr << "dtime not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Val_domain Domain::ddtime(const Val_domain&) const
    {
        cerr << "ddtime not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    double Domain::integ(const Val_domain&, int) const
    {
        cerr << "integ not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    double Domain::integmoment(const Val_domain&, int, int) const
    {
        cerr << "integmoment not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    double Domain::integrale(const Val_domain&) const
    {
        cerr << "integrale not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    int Domain::nbr_unknowns(const Tensor&, int) const
    {
        cerr << "nbr_unknowns not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Array<int> Domain::nbr_conditions(const Tensor&, int, int, int, Array<int>**) const
    {
        cerr << "nbr_conditions not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Array<int> Domain::nbr_conditions_array(const Tensor&, int, const Array<int>&, int, Array<int>**) const
    {
        cerr << "nbr_conditions (with array) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Array<int> Domain::nbr_conditions_boundary(const Tensor&, int, int, int, Array<int>**) const
    {
        cerr << "nbr_conditions_boundary not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Array<int> Domain::nbr_conditions_boundary_array(const Tensor&, int, int, const Array<int>&, int,
                                                     Array<int>**) const
    {
        cerr << "nbr_conditions_boundary (array version) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    Array<int> Domain::nbr_conditions_boundary_one_side(const Tensor&, int, int, int, Array<int>**) const
    {
        cerr << "nbr_conditions_boundary_one_side not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::export_tau(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int, Array<int>**) const
    {
        cerr << "export_tau not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::export_tau_array(const Tensor&, int, const Array<int>&, Array<double>&, int&, const Array<int>&, int,
                                  Array<int>**) const
    {
        cerr << "export_tau (with array) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::export_tau_boundary(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int,
                                     Array<int>**) const
    {
        cerr << "export_tau_boundary not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }
    void Domain::export_tau_boundary_exception(const Tensor&, int, int, Array<double>&, int&, const Array<int>&,
                                               const Param&, int, const Tensor&, int, Array<int>**) const
    {
        cerr << "export_tau_boundary_exception not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::export_tau_boundary_array(const Tensor&, int, int, const Array<int>&, Array<double>&, int&,
                                           const Array<int>&, int, Array<int>**) const
    {
        cerr << "export_tau_boundary (with array) not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::export_tau_boundary_one_side(const Tensor&, int, int, Array<double>&, int&, const Array<int>&, int,
                                              Array<int>**) const
    {
        cerr << "export_tau_boundary_one_side not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::affecte_tau(Tensor&, int, const Array<double>&, int&) const
    {
        cerr << "affecte_tau not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::affecte_tau_one_coef(Tensor&, int, int, int&) const
    {
        cerr << "affecte_tau_one_coef not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    int Domain::phi_coefficient_parity(int k, int phi_basis) const
    {
        switch (phi_basis) {
            case COSSIN:
                return (k % 2) ? -1 : +1;
            case COS:
                return +1;
            case SIN:
                return -1;
            default:
                return 0;
        }
    }

    bool Domain::describe_tau_seed_block(const Tensor&, int,
                                         std::vector<TauSeedDescriptor>& descriptors) const
    {
        descriptors.clear();
        return false;
    }

    bool Domain::describe_volume_residual_rows(
        const Tensor&, int, int, const Array<int>&, int, Array<int>**,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        return false;
    }

    bool Domain::describe_boundary_residual_rows(
        const Tensor&, int, int, const Array<int>&, int, Array<int>**,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        return false;
    }

    bool Domain::residual_tensor_components_in_tau_order(
        const Tensor& eq, int dom, int n_cmp, Array<int>** p_cmp,
        std::vector<int>& components) const
    {
        components.clear();
        if (dom < 0 || dom >= eq.get_space().get_nbr_domains() ||
            eq.get_space().get_domain(dom) != this || n_cmp < -1 ||
            (n_cmp >= 0 && p_cmp == nullptr)) {
            return false;
        }
        if (n_cmp == -1) {
            components.reserve(static_cast<std::size_t>(eq.get_n_comp()));
            for (int component = 0; component < eq.get_n_comp(); ++component)
                components.push_back(component);
            return true;
        }
        components.reserve(static_cast<std::size_t>(n_cmp));
        for (int selected = 0; selected < n_cmp; ++selected) {
            if (p_cmp[selected] == nullptr)
                return false;
            const int component = eq.position(*p_cmp[selected]);
            if (component < 0 || component >= eq.get_n_comp())
                return false;
            components.push_back(component);
        }
        return true;
    }

    bool Domain::append_volume_residual_row(
        const Val_domain& field, int dom, int component, int phi_index,
        ResidualRowDescriptor& descriptor) const
    {
        descriptor = ResidualRowDescriptor{};
        if (component < 0 || phi_index < 0 ||
            phi_index >= nbr_coefs(2) ||
            field.get_base().get_base_1d(2) == nullptr) {
            return false;
        }
        const Array<int>& phi_bases = *field.get_base().get_base_1d(2);
        if (phi_bases.get_nbr() <= 0)
            return false;
        const int phi_basis = phi_bases(0);
        for (std::size_t i = 1; i < phi_bases.get_nbr(); ++i)
            if (phi_bases(i) != phi_basis)
                return false;
        if (phi_coefficient_parity(phi_index, phi_basis) == 0)
            return false;

        descriptor.sides.push_back(
            ResidualRowCoordinate{dom, component, phi_basis, phi_index});
        return true;
    }

    bool Domain::materialize_tau_seed(
        Tensor& target, const Tensor& source, int dom,
        const TauSeedDescriptor& descriptor) const
    {
        if (dom < 0 || dom >= source.get_space().get_nbr_domains() ||
            source.get_space().get_domain(dom) != this ||
            &target.get_space() != &source.get_space() ||
            target.get_n_comp() != source.get_n_comp() ||
            descriptor.component < 0 ||
            descriptor.component >= target.get_n_comp() ||
            descriptor.write_count <= 0 ||
            descriptor.write_count > TauSeedDescriptor::max_writes) {
            return false;
        }

        std::size_t coefficient_count = 1;
        const Dim_array& shape = get_nbr_coefs();
        for (int dimension = 0; dimension < shape.get_ndim(); ++dimension)
            coefficient_count *= static_cast<std::size_t>(shape(dimension));
        for (int write = 0; write < descriptor.write_count; ++write)
            if (descriptor.writes[write].coefficient_offset >= coefficient_count)
                return false;

        for (int component = 0; component < target.get_n_comp(); ++component) {
            const Array<int> index(target.indices(component));
            target.set(index).set_domain(dom).set_zero();
        }

        const Array<int> active_index(target.indices(descriptor.component));
        Val_domain& active = target.set(active_index).set_domain(dom);
        active.set_base() = source(active_index)(dom).get_base();
        active.allocate_coef();
        *active.cf = 0.;
        double* const coefficients = active.cf->get_data();
        for (int write = 0; write < descriptor.write_count; ++write) {
            coefficients[descriptor.writes[write].coefficient_offset] =
                descriptor.writes[write].value;
        }
        return true;
    }

    double Domain::val_boundary(int, const Val_domain&, const Index&) const
    {
        cerr << "val_boundary not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    int Domain::nbr_points_boundary(int, const Base_spectral&) const
    {
        cerr << "nbr_points_boundary not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::do_which_points_boundary(int, const Base_spectral&, Index**, int) const
    {
        cerr << "do_which_points_boundary not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::do_absol() const
    {
        cerr << "do_absol not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::do_cart() const
    {
        cerr << "do_cart not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::do_cart_surr() const
    {
        cerr << "do_cart_surr not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::do_radius() const
    {
        cerr << "do_radius not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    double Domain::multipoles_sym(int, int, int, const Val_domain&, const Array<double>&) const
    {
        cerr << "multipoles sym not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    double Domain::multipoles_asym(int, int, int, const Val_domain&, const Array<double>&) const
    {
        cerr << "multipoles asym not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::multipoles_sym(int, int, int, const Term_eq&, const Array<double>&) const
    {
        cerr << "multipoles sym not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::multipoles_asym(int, int, int, const Term_eq&, const Array<double>&) const
    {
        cerr << "multipoles asym not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::radial_part_sym(const Space&, int, int, const Term_eq&,
                                    Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&),
                                    const Param&) const
    {
        cerr << "radial part sym not implemented for" << endl;
        cerr << *this << endl;
        std::ostringstream oss;
        oss << "and function " << f << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::radial_part_asym(const Space&, int, int, const Term_eq&,
                                     Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&),
                                     const Param&) const
    {
        cerr << "radial part asym not implemented for" << endl;
        cerr << *this << endl;
        std::ostringstream oss;
        oss << "and function " << f << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::harmonics_sym(const Term_eq&, const Term_eq&, int,
                                  Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&), const Param&,
                                  const Array<double>&) const
    {
        cerr << "harmonics_sym not implemented for" << endl;
        cerr << *this << endl;
        std::ostringstream oss;
        oss << "and function " << f << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::harmonics_asym(const Term_eq&, const Term_eq&, int,
                                   Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&), const Param&,
                                   const Array<double>&) const
    {
        cerr << "harmonics_asym not implemented for" << endl;
        cerr << *this << endl;
        std::ostringstream oss;
        oss << "and function " << f << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::der_multipoles_sym(int, int, int, const Term_eq&, const Array<double>&) const
    {
        cerr << "der multipoles sym not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::der_multipoles_asym(int, int, int, const Term_eq&, const Array<double>&) const
    {
        cerr << "der multipoles asym not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::der_radial_part_sym(const Space&, int, int, const Term_eq&,
                                        Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&),
                                        const Param&) const
    {
        cerr << "der radial part sym not implemented for" << endl;
        cerr << *this << endl;
        std::ostringstream oss;
        oss << "and function " << f << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::der_radial_part_asym(const Space&, int, int, const Term_eq&,
                                         Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&),
                                         const Param&) const
    {
        cerr << "der radial part asym not implemented for" << endl;
        cerr << *this << endl;
        std::ostringstream oss;
        oss << "and function " << f << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::der_harmonics_sym(const Term_eq&, const Term_eq&, int,
                                      Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&), const Param&,
                                      const Array<double>&) const
    {
        cerr << "der_harmonics_sym not implemented for" << endl;
        cerr << *this << endl;
        std::ostringstream oss;
        oss << "and function " << f << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::der_harmonics_asym(const Term_eq&, const Term_eq&, int,
                                       Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&), const Param&,
                                       const Array<double>&) const
    {
        cerr << "der harmonics_asym not implemented for" << endl;
        cerr << *this << endl;
        std::ostringstream oss;
        oss << "and function " << f << endl;
        KADATH_THROW(oss.str());
    }

    const Term_eq* Domain::give_normal(int, int) const
    {
        cerr << "give_normal not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::derive_flat_spher(int, char, const Term_eq&, const Metric*) const
    {
        cerr << "derive_flat_spher not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::derive_flat_mtz(int, char, const Term_eq&, const Metric*) const
    {
        cerr << "derive_flat_mtz not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::derive_flat_cart(int, char, const Term_eq&, const Metric*) const
    {
        cerr << "derive_flat_cart not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    int Domain::give_place_var(char*) const
    {
        return -1;
    }

    Tensor Domain::import(int, int, int, const Array<int>&, Tensor**) const
    {
        cerr << "import not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::update_term_eq(Term_eq* so) const
    {
        so->set_der_zero();
    }

    void Domain::accumulate_term_eq_mapping_derivative(Term_eq* so) const
    {
        // General exact fallback: preserve the already seeded field tangent,
        // let the established replacement interface materialize the mapping
        // tangent, then restore mapping + field in the pertinent domain only.
        // BNS_nosym's adapted shells override this to eliminate the snapshot.
        const int term_domain = so->get_dom();
        Tensor field_derivative(one_domain_storage, term_domain, so->get_der_t());
        update_term_eq(so);
        so->set_der_t(add_one_dom(term_domain, so->get_der_t(), field_derivative));
    }

    double Domain::integ_volume(const Val_domain&) const
    {
        cerr << "Integ volume not implemented for " << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    void Domain::filter(Tensor&, int, double) const
    {
        cerr << "Filter not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Term_eq Domain::div_1mx2_term_eq(const Term_eq& so) const
    {
        return do_comp_by_comp(so, &Domain::div_1mx2);
    }
} // namespace Kadath
