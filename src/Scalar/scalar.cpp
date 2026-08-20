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
 *   2026-08-11  Added size-neutral direct-domain storage for sparse Scalar shells.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <memory>
#include <limits>
#include <sstream>
#include <utility>
#include "For_Kadath/Scalar/scalar.hpp"

namespace Kadath
{
    namespace
    {
        struct BoolSlabRelease
        {
            int count;

            void operator()(bool* memory) const noexcept
            {
                MemoryMapper::release_memory<bool>(memory, count);
            }
        };

        using BoolSlab = std::unique_ptr<bool[], BoolSlabRelease>;

        struct DomainTableRelease
        {
            int count;

            void operator()(Val_domain** domains) const noexcept
            {
                if (domains == nullptr)
                    return;
                for (int domain = 0; domain < count; ++domain)
                    delete domains[domain];
                MemoryMapper::release_memory<Val_domain*>(domains, count);
            }
        };

        using DomainTable = std::unique_ptr<Val_domain*[], DomainTableRelease>;

        DomainTable make_empty_domain_table(int count)
        {
            Val_domain** const domains = MemoryMapper::get_memory<Val_domain*>(count);
            for (int domain = 0; domain < count; ++domain)
                domains[domain] = nullptr;
            return DomainTable(domains, DomainTableRelease{count});
        }

        DomainTable clone_domain_table(Val_domain* const* source, int count,
                                       bool copy)
        {
            DomainTable domains = make_empty_domain_table(count);
            for (int domain = 0; domain < count; ++domain) {
                if (source[domain] != nullptr)
                    domains[domain] = new Val_domain(*source[domain], copy);
            }
            return domains;
        }
    }

    void Scalar::copy_domain_storage_from(const Scalar& source, bool copy)
    {
        assert(domain_storage_word == nullptr);
        assert(scalar_domain_storage_tag == 0U);

        if (source.uses_direct_domain_storage()) {
            assert(source.direct_domain_storage() != nullptr);
            std::unique_ptr<Val_domain> direct(
                new Val_domain(*source.direct_domain_storage(), copy));
            domain_storage_word = direct.release();
            scalar_domain_storage_tag = source.scalar_domain_storage_tag;
            return;
        }

        Val_domain** const source_domains = source.dense_domain_storage();
        if (source_domains == nullptr)
            return;
        DomainTable domains = clone_domain_table(source_domains, ndom, copy);
        domain_storage_word = domains.release();
    }

    void Scalar::assign_domain_storage_from(const Scalar& source)
    {
        assert(this != &source);

        if (!source.uses_direct_domain_storage()) {
            Val_domain** const source_domains = source.dense_domain_storage();
            if (source_domains == nullptr) {
                release_domain_storage();
                return;
            }

            if (!uses_direct_domain_storage()) {
                Val_domain** const target_domains = dense_domain_storage();
                if (target_domains != nullptr) {
                    // Preserve the persistent dense table. Each replacement is
                    // complete before its old slot is released, so a failed copy
                    // leaves every slot owning a valid object.
                    for (int domain = 0; domain < ndom; ++domain) {
                        std::unique_ptr<Val_domain> replacement;
                        if (source_domains[domain] != nullptr)
                            replacement.reset(new Val_domain(*source_domains[domain]));
                        delete target_domains[domain];
                        target_domains[domain] = replacement.release();
                    }
                    return;
                }
            }

            DomainTable replacement = clone_domain_table(source_domains, ndom, true);
            release_domain_storage();
            domain_storage_word = replacement.release();
            return;
        }

        assert(source.direct_domain_storage() != nullptr);
        std::unique_ptr<Val_domain> replacement(
            new Val_domain(*source.direct_domain_storage()));
        const std::uint8_t replacement_tag = source.scalar_domain_storage_tag;
        release_domain_storage();
        domain_storage_word = replacement.release();
        scalar_domain_storage_tag = replacement_tag;
    }

    void Scalar::release_domain_storage() noexcept
    {
        void* const storage = domain_storage_word;
        const std::uint8_t storage_tag = scalar_domain_storage_tag;
        domain_storage_word = nullptr;
        scalar_domain_storage_tag = 0U;

        if (storage_tag != 0U) {
            delete static_cast<Val_domain*>(storage);
            return;
        }
        DomainTable domains(static_cast<Val_domain**>(storage),
                            DomainTableRelease{ndom});
    }

    Scalar::Scalar(const Space& sp) : Tensor(sp)
    {
        DomainTable domains = make_empty_domain_table(ndom);
        for (int l = 0; l < ndom; l++)
            domains[l] = new Val_domain(sp.get_domain(l));
        domain_storage_word = domains.release();
        cmp[0] = this;
    }

    Scalar::Scalar(OneDomainStorageTag, int active_domain, const Space& sp) : Tensor(sp)
    {
        if (active_domain < 0 || active_domain >= ndom)
            KADATH_THROW("one-domain Scalar storage index is outside the Space");
        if (active_domain < static_cast<int>(std::numeric_limits<std::uint8_t>::max())) {
            std::unique_ptr<Val_domain> direct(new Val_domain(sp.get_domain(active_domain)));
            domain_storage_word = direct.release();
            scalar_domain_storage_tag = static_cast<std::uint8_t>(active_domain + 1);
        } else {
            DomainTable domains = make_empty_domain_table(ndom);
            domains[active_domain] = new Val_domain(sp.get_domain(active_domain));
            domain_storage_word = domains.release();
        }
        cmp[0] = this;
    }

    Scalar::Scalar(const Scalar& so, bool copie) : Tensor(so.espace)
    {
        copy_domain_storage_from(so, copie);
        cmp[0] = this;
    }

    Scalar::Scalar(const Tensor& so, bool copie) : Tensor(so.espace)
    {

        assert(so.valence == 0);
        copy_domain_storage_from(*so.cmp[0], copie);
        cmp[0] = this;
    }

    Scalar::Scalar(const Space& sp, BinarySource& source) : Tensor(sp)
    {
        DomainTable domains = make_empty_domain_table(ndom);
        for (int l = 0; l < ndom; l++)
            domains[l] = new Val_domain(sp.get_domain(l), source);
        domain_storage_word = domains.release();
        cmp[0] = this;
    }

    Scalar::Scalar(Scalar&& so) noexcept : Tensor{so.espace}
    {
        cmp[0] = this;
        domain_storage_word = so.domain_storage_word;
        scalar_domain_storage_tag = so.scalar_domain_storage_tag;
        so.domain_storage_word = nullptr;
        so.scalar_domain_storage_tag = 0U;
    }

    Scalar& Scalar::operator=(Scalar&& so) noexcept
    {
        assert(&espace == &so.espace);
        std::swap(domain_storage_word, so.domain_storage_word);
        std::swap(scalar_domain_storage_tag, so.scalar_domain_storage_tag);
        return *this;
    }

    Scalar::~Scalar()
    {
        release_domain_storage();
        cmp[0] = nullptr;
    }

    void Scalar::save(BinarySink& sink) const
    {
        for (int i = 0; i < get_nbr_domains(); i++)
            domain_storage(i).save(sink);
    }

    void Scalar::operator=(const Scalar& so)
    {
        assert(&espace == &so.espace);
        if (this == &so)
            return;
        assign_domain_storage_from(so);
    }

    void Scalar::operator=(const Tensor& so)
    {
        assert(&espace == &so.espace);
        assert(so.valence == 0);
        const Scalar& source = *so.cmp[0];
        if (this == &source)
            return;
        assign_domain_storage_from(source);
    }

    void Scalar::operator=(double xx)
    {
        for (int i = 0; i < ndom; i++)
            set_domain(i) = xx;
    }

    void Scalar::annule_hard()
    {
        for (int i = 0; i < ndom; i++)
            set_domain(i).annule_hard();
    }

    void Scalar::annule_hard_coef()
    {
        for (int i = 0; i < ndom; i++)
            set_domain(i).annule_hard_coef();
    }

    void Scalar::set_in_conf()
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).set_in_conf();
    }

    void Scalar::set_in_coef()
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).set_in_coef();
    }

    void Scalar::std_base()
    {
        if (!is_m_quant_affected()) {
            for (int l = 0; l < ndom; l++)
                domain_storage(l).std_base();
        } else {
            for (int l = 0; l < ndom; l++)
                domain_storage(l).std_base(parameters->get_m_quant());
        }
    }

    void Scalar::std_anti_base()
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).std_anti_base();
    }

    void Scalar::std_base(int m)
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).std_base(m);
    }

    void Scalar::std_anti_base(int m)
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).std_anti_base(m);
    }

    void Scalar::std_base_r_spher()
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).std_base_r_spher();
    }

    void Scalar::std_base_t_spher()
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).std_base_t_spher();
    }

    void Scalar::std_base_p_spher()
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).std_base_p_spher();
    }

    void Scalar::std_base_domain(int d)
    {
        domain_storage(d).std_base();
    }

    void Scalar::std_anti_base_domain(int d)
    {
        domain_storage(d).std_anti_base();
    }

    void Scalar::std_base_domain(int d, int m)
    {
        domain_storage(d).std_base(m);
    }

    void Scalar::std_base_x_cart_domain(int d)
    {
        domain_storage(d).std_base_x_cart();
    }

    void Scalar::std_base_y_cart_domain(int d)
    {
        domain_storage(d).std_base_y_cart();
    }

    void Scalar::std_base_z_cart_domain(int d)
    {
        domain_storage(d).std_base_z_cart();
    }

    void Scalar::std_base_r_spher_domain(int d)
    {
        domain_storage(d).std_base_r_spher();
    }

    void Scalar::std_base_t_spher_domain(int d)
    {
        domain_storage(d).std_base_t_spher();
    }

    void Scalar::std_base_p_spher_domain(int d)
    {
        domain_storage(d).std_base_p_spher();
    }

    void Scalar::std_base_xy_cart_domain(int d)
    {
        domain_storage(d).std_base_xy_cart();
    }

    void Scalar::std_base_xz_cart_domain(int d)
    {
        domain_storage(d).std_base_xz_cart();
    }

    void Scalar::std_base_yz_cart_domain(int d)
    {
        domain_storage(d).std_base_yz_cart();
    }

    void Scalar::std_base_rt_spher_domain(int d)
    {
        domain_storage(d).std_base_rt_spher();
    }

    void Scalar::std_base_rp_spher_domain(int d)
    {
        domain_storage(d).std_base_rp_spher();
    }

    void Scalar::std_base_tp_spher_domain(int d)
    {
        domain_storage(d).std_base_tp_spher();
    }

    void Scalar::std_xodd_base()
    {
        domain_storage(0).std_xodd_base();
        for (int l = 1; l < ndom; l++)
            domain_storage(l).std_base();
    }

    void Scalar::std_todd_base()
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).std_todd_base();
    }

    void Scalar::std_xodd_todd_base()
    {
        domain_storage(0).std_xodd_todd_base();
        for (int l = 1; l < ndom; l++)
            domain_storage(l).std_todd_base();
    }

    void Scalar::std_base_odd()
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).std_base_odd();
    }

    void Scalar::std_base_r_mtz_domain(int d)
    {
        domain_storage(d).std_base_r_mtz();
    }

    void Scalar::std_base_t_mtz_domain(int d)
    {
        domain_storage(d).std_base_t_mtz();
    }

    void Scalar::std_base_p_mtz_domain(int d)
    {
        domain_storage(d).std_base_p_mtz();
    }

    void Scalar::allocate_conf()
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).allocate_conf();
    }

    void Scalar::allocate_coef()
    {
        for (int l = 0; l < ndom; l++)
            domain_storage(l).allocate_coef();
    }

    Val_domain& Scalar::domain_storage(int i) const
    {
        assert((i >= 0) && (i < ndom));
        if (uses_direct_domain_storage()) {
            Val_domain* const direct = direct_domain_storage();
            assert(direct != nullptr);
            if (direct_domain_index() == i)
                return *direct;

            // Build the promoted representation completely before publishing
            // it. The existing direct object is transferred only after the
            // requested Val_domain has been constructed successfully.
            DomainTable promoted = make_empty_domain_table(ndom);
            promoted[i] = new Val_domain(espace.get_domain(i));
            promoted[direct_domain_index()] = direct;
            Val_domain* const requested = promoted[i];
            domain_storage_word = promoted.release();
            scalar_domain_storage_tag = 0U;
            return *requested;
        }

        Val_domain** domains = dense_domain_storage();
        if (domains == nullptr) {
            DomainTable materialized = make_empty_domain_table(ndom);
            materialized[i] = new Val_domain(espace.get_domain(i));
            Val_domain* const requested = materialized[i];
            domain_storage_word = materialized.release();
            return *requested;
        }
        if (domains[i] == nullptr) {
            std::unique_ptr<Val_domain> requested(
                new Val_domain(espace.get_domain(i)));
            domains[i] = requested.release();
        }
        return *domains[i];
    }

    const Val_domain& Scalar::operator()(int i) const
    {
        return domain_storage(i);
    }
    const Val_domain& Scalar::at(int i) const
    {
        return operator()(i);
    }

    Val_domain& Scalar::set_domain(int l)
    {
        return domain_storage(l);
    }

    void Scalar::set_val_inf(double x)
    {
        Val_domain& domain = domain_storage(ndom - 1);
        domain.get_domain()->set_val_inf(domain, x);
    }

    void Scalar::set_val_inf(double x, int l)
    {
        Val_domain& domain = domain_storage(l);
        domain.get_domain()->set_val_inf(domain, x);
    }

    double Scalar::val_point(const Point& xx, int sens) const
    {

        assert((sens == +1) || (sens == -1));

        BoolSlab inside(MemoryMapper::get_memory<bool>(ndom), BoolSlabRelease{ndom});

        for (int l = ndom - 1; l >= 0; l--)
            inside[l] = get_domain(l)->is_in(xx);
        // First domain in which the point is :
        int ld = -1;
        if (sens == -1) {
            for (int l = ndom - 1; l >= 0; l--)
                if ((ld == -1) && (inside[l]))
                    ld = l;
        } else {
            for (int l = 0; l < ndom; l++)
                if ((ld == -1) && (inside[l]))
                    ld = l;
        }

        if (ld == -1) {
            std::ostringstream oss;
            oss << "Point " << xx << "not found in the computational space..." << endl;
            KADATH_THROW(oss.str());
        } else {
            const Val_domain& domain = operator()(ld);
            if (domain.check_if_zero())
                return 0.;
            else {
                Point num(get_domain(ld)->absol_to_num(xx));

                coef();
                return domain.base.summation(num, *domain.cf);
            }
        }
    }

    double Scalar::val_point_zeronotdef(const Point& xx, int sens) const
    {

        assert((sens == +1) || (sens == -1));

        BoolSlab inside(MemoryMapper::get_memory<bool>(ndom), BoolSlabRelease{ndom});
        for (int l = ndom - 1; l >= 0; l--)
            inside[l] = get_domain(l)->is_in(xx);
        // First domain in which the point is :
        int ld = -1;
        if (sens == -1) {
            for (int l = ndom - 1; l >= 0; l--)
                if ((ld == -1) && (inside[l]))
                    ld = l;
        } else {
            for (int l = 0; l < ndom; l++)
                if ((ld == -1) && (inside[l]))
                    ld = l;
        }

        if (ld == -1) {
            return 0;
        } else {
            const Val_domain& domain = operator()(ld);
            if (domain.check_if_zero())
                return 0.;
            else {
                Point num(get_domain(ld)->absol_to_num(xx));

                coef();
                return domain.base.summation(num, *domain.cf);
            }
        }
    }
    void Scalar::coef() const
    {
        // Boucles sur les domaines
        for (int l = 0; l < ndom; l++)
            domain_storage(l).coef();
    }

    void Scalar::coef_i() const
    {
        // Boucles sur les domaines
        for (int l = 0; l < ndom; l++)
            domain_storage(l).coef_i();
    }

    void Scalar::filter_phi(int dom, int ncf)
    {
        coef();
        Index pcf(get_space().get_domain(dom)->get_nbr_coefs());
        int np = get_space().get_domain(dom)->get_nbr_coefs()(2);
        do {
            if (pcf(2) + ncf > np - 1)
                set_domain(dom).set_coef(pcf) = 0;
        } while (pcf.inc());
    }

    Scalar Scalar::der_var(int var) const
    {
        Scalar res(*this, false);
        for (int dom = 0; dom < ndom; dom++)
            res.set_domain(dom) = operator()(dom).der_var(var);
        return res;
    }

    Scalar Scalar::der_abs(int var) const
    {
        Scalar res(*this, false);
        for (int dom = 0; dom < ndom; dom++)
            res.set_domain(dom) = operator()(dom).der_abs(var);

        return res;
    }

    Scalar Scalar::der_spher(int var) const
    {
        Scalar res(*this, false);
        for (int dom(0); dom < ndom; ++dom)
            res.set_domain(dom) = operator()(dom).der_spher(var);
        return res;
    }

    Scalar Scalar::der_r() const
    {
        Scalar res(*this, false);
        for (int dom = 0; dom < ndom; dom++)
            res.set_domain(dom) = operator()(dom).der_r();

        return res;
    }

    double Scalar::integrale() const
    {
        double res = 0;
        for (int dom = 0; dom < ndom; dom++)
            res += operator()(dom).integrale();

        return res;
    }

    unique_ptr<Scalar> Scalar::clone() const
    {
        unique_ptr<Scalar> res(new Scalar(*this)); // USE make_unique when g++ 4.9 and std=c++14 available
        return res;
    }

    ostream& operator<<(ostream& o, const Scalar& so)
    {

        o << "Scalar" << endl;
        for (int l = 0; l < so.ndom; l++) {
            o << "Domain : " << l << endl;
            o << so(l) << endl;
        }
        return o;
    }

    double Scalar::integ_volume() const
    {
        double res = 0;
        for (int dom = 0; dom < ndom; dom++)
            res += operator()(dom).integ_volume();

        return res;
    }

    Scalar Scalar::zero(Space const& espace)
    {
        Scalar res(espace);
        res = 0.0;
        return res;
    }
    double Scalar::val_point_bound(const Point& xx, int bound) const
    {

        BoolSlab inside(MemoryMapper::get_memory<bool>(ndom), BoolSlabRelease{ndom});

        for (int l = ndom - 1; l >= 0; l--)
            inside[l] = get_domain(l)->is_in(xx);
        // First domain in which the point is :
        int ld = -1;
        for (int l = 0; l < ndom; l++)
            if ((ld == -1) && (inside[l]))
                ld = l;

        if (ld == -1) {
            std::ostringstream oss;
            oss << "Point " << xx << "not found in the computational space..." << endl;
            KADATH_THROW(oss.str());
        } else {
            const Val_domain& domain = operator()(ld);
            if (domain.check_if_zero())
                return 0.;
            else {
                Point num(get_domain(ld)->absol_to_num_bound(xx, bound));

                coef();
                return domain.base.summation(num, *domain.cf);
            }
        }
    }

} // namespace Kadath
