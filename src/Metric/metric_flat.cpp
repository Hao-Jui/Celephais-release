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

#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "For_Kadath/Tensor/metric_tensor.hpp"
#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"

#include <chrono>
namespace Kadath
{
    namespace
    {
        void set_derivative_result_names(Term_eq& result, const Term_eq& source, char derivative_name, int valence)
        {
            result.set_val_t()->set_name_affected();
            result.set_val_t()->set_name_ind(0, derivative_name);
            for (int index = 1; index < valence; index++)
                result.set_val_t()->set_name_ind(index, source.get_val_t().get_name_ind()[index - 1]);

            for (int lane = 0; lane < result.get_derivative_lane_count(); ++lane) {
                if (!result.has_der_t(lane))
                    continue;
                Tensor* derivative = result.set_der_t(lane);
                derivative->set_name_affected();
                derivative->set_name_ind(0, derivative_name);
                for (int index = 1; index < valence; index++)
                    derivative->set_name_ind(index, source.get_val_t().get_name_ind()[index - 1]);
            }
        }

        Term_eq make_summed_one_domain_term(int domain, const Term_eq& source)
        {
            Tensor summed_value(source.get_val_t().do_summation_one_dom(domain));
            if (source.get_p_der_t() == nullptr)
                return Term_eq(domain, summed_value);

            Tensor summed_derivative(source.get_der_t().do_summation_one_dom(domain));
            Term_eq result(domain, summed_value, summed_derivative);
            result.set_derivative_lane_count(source.get_derivative_lane_count());
            for (int lane = 1; lane < source.get_derivative_lane_count(); ++lane) {
                if (source.has_der_t(lane))
                    result.set_der_t(lane, source.get_der_t(lane).do_summation_one_dom(domain));
            }
            return result;
        }
    }

    Metric_flat::Metric_flat(const Space& sp, const Base_tensor& bb) : Metric(sp), basis(bb) {}

    Metric_flat::Metric_flat(const Metric_flat& so) : Metric(so), basis(so.basis) {}

    Metric_flat::~Metric_flat() {}

    void Metric_flat::update()
    {
        // Nothing to do everything is constant
    }

    void Metric_flat::update(int)
    {
        // Nothing to do everything is constant
    }

    void Metric_flat::compute_cov(int dd) const
    {
        Metric_tensor res(espace, COV, basis);

        for (int i = 1; i <= espace.get_ndim(); i++)
            for (int j = i; j <= espace.get_ndim(); j++)
                res.set(i, j).set_domain(dd) = (i == j) ? 1. : 0;
        res.std_base();

        if (p_met_cov[dd] == nullptr)
            p_met_cov[dd] = std::make_unique<Term_eq>(dd, res);
        else
            *p_met_cov[dd] = Term_eq(dd, res);
        p_met_cov[dd]->set_der_zero();
    }

    int Metric_flat::give_type(int dd) const
    {
        return basis.get_basis(dd);
    }

    void Metric_flat::compute_con(int dd) const
    {

        Metric_tensor res(espace, CON, basis);
        for (int i = 1; i <= espace.get_ndim(); i++)
            for (int j = i; j <= espace.get_ndim(); j++)
                res.set(i, j).set_domain(dd) = (i == j) ? 1. : 0;
        res.std_base();
        if (p_met_con[dd] == nullptr)
            p_met_con[dd] = std::make_unique<Term_eq>(dd, res);
        else
            *p_met_con[dd] = Term_eq(dd, res);
        p_met_con[dd]->set_der_zero();
    }

    void Metric_flat::compute_christo(int) const
    {
        KADATH_THROW("Computation of Christo not explicit for Metric_flat");
    }

    void Metric_flat::compute_ricci_tensor(int dd) const
    {

        Metric_tensor res(espace, COV, basis);
        switch (basis.get_basis(dd)) {
            case CARTESIAN_BASIS:
                res = 0;
                break;
            case SPHERICAL_BASIS:
                res = 0;
                break;
            case MTZ_BASIS:
                for (int i = 1; i <= 3; i++)
                    for (int j = i; j <= 3; j++)
                        res.set(i, j).annule_hard();
                res.set(2, 2).set_domain(dd) = -2 / pow(espace.get_domain(dd)->get_radius(), 2);
                res.set(3, 3).set_domain(dd) = res(2, 2)(dd);
                break;
            default:
                KADATH_THROW("Unknown tensorial basis in Metric_flat::compute_ricci_tensor");
        }

        res.std_base();
        if (p_ricci_tensor[dd] == nullptr)
            p_ricci_tensor[dd] = std::make_unique<Term_eq>(dd, res);
        else
            *p_ricci_tensor[dd] = Term_eq(dd, res);

        p_ricci_tensor[dd]->set_der_zero();
    }

    void Metric_flat::compute_ricci_scalar(int dd) const
    {

        Scalar res(espace);
        switch (basis.get_basis(dd)) {
            case CARTESIAN_BASIS:
                res = 0;
                break;
            case SPHERICAL_BASIS:
                res = 0;
                break;
            case MTZ_BASIS:
                res.set_domain(dd) = -4 / pow(espace.get_domain(dd)->get_radius(), 2);
                break;
            default:
                KADATH_THROW("Unknown tensorial basis in Metric_flat::compute_ricci_tensor");
        }

        res.std_base();
        if (p_ricci_scalar[dd] == nullptr)
            p_ricci_scalar[dd] = std::make_unique<Term_eq>(dd, res);
        else
            *p_ricci_scalar[dd] = Term_eq(dd, res);

        p_ricci_scalar[dd]->set_der_zero();
    }

    void Metric_flat::manipulate_ind(Term_eq& so, int ind) const
    {
        // Just change the type of the indice !
        so.set_val_t()->set_index_type(ind) *= -1;
        for (int lane = 0; lane < so.get_derivative_lane_count(); ++lane)
            if (so.has_der_t(lane))
                so.set_der_t(lane)->set_index_type(ind) *= -1;
    }

    Term_eq Metric_flat::derive_partial_cart(int type_der, char ind_der, const Term_eq& so) const
    {

        int dom = so.get_dom();

        // Affecte names or not ?
        bool donames = ((so.val_t->is_name_affected()) || (so.val_t->get_valence() == 0)) ? true : false;

        bool need_sum = false;
        if (donames)
            for (int i = 0; i < so.val_t->get_valence(); i++)
                if (ind_der == so.val_t->get_name_ind()[i])
                    need_sum = true;

        const Domain* domain = espace.get_domain(dom);
        if (so.der_t != nullptr && need_sum && ((dynamic_cast<const Domain_shell_inner_adapted*>(domain) != nullptr) ||
                                                (dynamic_cast<const Domain_shell_outer_adapted*>(domain) != nullptr))) {
            return domain->derive_flat_cart(type_der, ind_der, so, this);
        }

        // Computation of flat gradient :
        Term_eq auxi(domain->partial_cart(so));

        int val_res = auxi.val_t->get_valence();

        if (donames)
            set_derivative_result_names(auxi, so, ind_der, val_res);

        // Manipulate if Contravariant version
        if (type_der == CON)
            manipulate_ind(auxi, 0);

        if (!need_sum)
            return auxi;
        else {
            return make_summed_one_domain_term(dom, auxi);
        }
    }

    Term_eq Metric_flat::derive_partial_spher(int type_der, char ind_der, const Term_eq& so) const
    {

        int dom = so.get_dom();
        bool donames = ((so.val_t->is_name_affected()) || (so.val_t->get_valence() == 0)) ? true : false;

        // Computation of flat gradient :
        Term_eq auxi(espace.get_domain(dom)->partial_spher(so));

        int val_res = auxi.val_t->get_valence();

        if (donames)
            set_derivative_result_names(auxi, so, ind_der, val_res);

        // Manipulate if Contravariant version
        if (type_der == CON)
            manipulate_ind(auxi, 0);

        bool need_sum = false;
        if (donames)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;

        if (!need_sum)
            return auxi;
        else {
            return make_summed_one_domain_term(dom, auxi);
        }
    }

    Term_eq Metric_flat::derive_partial_mtz(int type_der, char ind_der, const Term_eq& so) const
    {

        int dom = so.get_dom();
        bool donames = ((so.val_t->is_name_affected()) || (so.val_t->get_valence() == 0)) ? true : false;

        // Computation of flat gradient :
        Term_eq auxi(espace.get_domain(dom)->partial_mtz(so));

        int val_res = auxi.val_t->get_valence();

        if (donames)
            set_derivative_result_names(auxi, so, ind_der, val_res);

        // Manipulate if Contravariant version
        if (type_der == CON)
            manipulate_ind(auxi, 0);

        bool need_sum = false;
        if (donames)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;

        if (!need_sum)
            return auxi;
        else {
            return make_summed_one_domain_term(dom, auxi);
        }
    }

    Term_eq Metric_flat::derive_partial(int type_der, char ind_der, const Term_eq& so) const
    {

        int dom = so.get_dom();

        if (p_met_con[dom] == nullptr)
            compute_con(dom);
        if (p_met_cov[dom] == nullptr)
            compute_cov(dom);

        // so must be tensor :
        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("Metric_flat::derive partial only defined for tensor data");
        }

        switch (basis.get_basis(dom)) {
            case CARTESIAN_BASIS:
                return derive_partial_cart(type_der, ind_der, so);
            case SPHERICAL_BASIS:
                return derive_partial_spher(type_der, ind_der, so);
            case MTZ_BASIS:
                return derive_partial_mtz(type_der, ind_der, so);
            default:
                KADATH_THROW("Unknown tensorial basis in Metric_flat::derive_partial");
        }
    }

    Term_eq Metric_flat::derive_cart(int type_der, char ind_der, const Term_eq& so) const
    {
        return derive_partial(type_der, ind_der, so);
    }

    Term_eq Metric_flat::derive_spher(int type_der, char ind_der, const Term_eq& so) const
    {

        // Computation of flat gradient :
        Term_eq part_der(derive_partial(type_der, ind_der, so));

        int dom = so.get_dom();
        bool donames = ((so.val_t->is_name_affected()) || (so.val_t->get_valence() == 0)) ? true : false;

        Term_eq auxi(espace.get_domain(dom)->connection_spher(so));
        int val_res = auxi.val_t->get_valence();

        if (donames)
            set_derivative_result_names(auxi, so, ind_der, val_res);

        // Manipulate if Contravariant version
        if (type_der == CON)
            manipulate_ind(auxi, 0);

        bool need_sum = false;
        if (donames)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;

        if (!need_sum) {
            return (part_der + auxi);
        } else {
            return (part_der + make_summed_one_domain_term(dom, auxi));
        }
    }

    Term_eq Metric_flat::derive_mtz(int type_der, char ind_der, const Term_eq& so) const
    {

        // Computation of flat gradient :
        Term_eq part_der(derive_partial(type_der, ind_der, so));

        int dom = so.get_dom();
        bool donames = ((so.val_t->is_name_affected()) || (so.val_t->get_valence() == 0)) ? true : false;

        Term_eq auxi(espace.get_domain(dom)->connection_mtz(so));
        int val_res = auxi.val_t->get_valence();

        if (donames)
            set_derivative_result_names(auxi, so, ind_der, val_res);

        // Manipulate if Contravariant version
        if (type_der == CON)
            manipulate_ind(auxi, 0);

        bool need_sum = false;
        if (donames)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;

        if (!need_sum) {
            return (part_der + auxi);
        } else {
            return (part_der + make_summed_one_domain_term(dom, auxi));
        }
    }

    namespace
    {
        OpeDerDispatchClass classify_dispatch(int basis_id, int type_der, char ind_der,
                                              const Term_eq& so, const Domain* domain)
        {
            const bool donames = so.get_val_t().is_name_affected() ||
                                 so.get_val_t().get_valence() == 0;
            bool need_sum = false;
            if (donames) {
                const int v = so.get_val_t().get_valence();
                const char* names = so.get_val_t().get_name_ind();
                if (names != nullptr) {
                    for (int i = 0; i < v; ++i) {
                        if (names[i] == ind_der) {
                            need_sum = true;
                            break;
                        }
                    }
                }
            }
            const bool adapted =
                (dynamic_cast<const Domain_shell_inner_adapted*>(domain) != nullptr) ||
                (dynamic_cast<const Domain_shell_outer_adapted*>(domain) != nullptr);
            switch (basis_id) {
                case CARTESIAN_BASIS:
                    if (adapted && so.get_p_der_t() != nullptr && need_sum)
                        return OpeDerDispatchClass::Cart_AdaptedBypass;
                    if (type_der == COV && !need_sum) return OpeDerDispatchClass::Cart_Cov_NoSum;
                    if (type_der == CON && !need_sum) return OpeDerDispatchClass::Cart_Con_NoSum;
                    if (type_der == COV) return OpeDerDispatchClass::Cart_Cov_Sum;
                    return OpeDerDispatchClass::Cart_Con_Sum;
                case SPHERICAL_BASIS:
                    if (type_der == COV && !need_sum) return OpeDerDispatchClass::Spher_Cov_NoSum;
                    if (type_der == CON && !need_sum) return OpeDerDispatchClass::Spher_Con_NoSum;
                    return OpeDerDispatchClass::Spher_Sum;
                case MTZ_BASIS:
                    if (type_der == COV && !need_sum) return OpeDerDispatchClass::Mtz_Cov_NoSum;
                    if (type_der == CON && !need_sum) return OpeDerDispatchClass::Mtz_Con_NoSum;
                    return OpeDerDispatchClass::Mtz_Sum;
                default:
                    return OpeDerDispatchClass::UnknownBasis;
            }
        }
    }

    Term_eq Metric_flat::derive(int type_der, char ind_der, const Term_eq& so) const
    {

        int dom = so.get_dom();

        if (p_met_con[dom] == nullptr)
            compute_con(dom);
        if (p_met_cov[dom] == nullptr)
            compute_cov(dom);

        // so must be tensor :
        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("Metric_flat::derive only defined for tensor data");
        }

        const int basis_id = basis.get_basis(dom);
        const bool census_on = ope_der_dispatch_census_enabled();
        OpeDerDispatchClass dispatch_class = OpeDerDispatchClass::UnknownBasis;
        std::chrono::steady_clock::time_point census_start;
        if (census_on) {
            dispatch_class = classify_dispatch(basis_id, type_der, ind_der, so,
                                               espace.get_domain(dom));
            census_start = std::chrono::steady_clock::now();
        }
        Term_eq result = [&] {
            switch (basis_id) {
                case CARTESIAN_BASIS:
                    return derive_cart(type_der, ind_der, so);
                case SPHERICAL_BASIS:
                    return derive_spher(type_der, ind_der, so);
                case MTZ_BASIS:
                    return derive_mtz(type_der, ind_der, so);
                default:
                    KADATH_THROW("Unknown tensorial basis in Metric_flat::derive");
            }
        }();
        if (census_on) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - census_start).count();
            ope_der_dispatch_census_record(dispatch_class, elapsed);
        }
        return result;
    }
    Term_eq Metric_flat::derive_with_other_cart(int type_der, char ind_der, const Term_eq& so,
                                                const Metric* manipulator) const
    {
        int dom = so.get_dom();
        // Call the domain version
        return so.val_t->get_space().get_domain(dom)->derive_flat_cart(type_der, ind_der, so, manipulator);
    }

    Term_eq Metric_flat::derive_with_other_spher(int type_der, char ind_der, const Term_eq& so,
                                                 const Metric* manipulator) const
    {
        int dom = so.get_dom();

        // Call the domain version
        return so.val_t->get_space().get_domain(dom)->derive_flat_spher(type_der, ind_der, so, manipulator);
    }

    Term_eq Metric_flat::derive_with_other_mtz(int type_der, char ind_der, const Term_eq& so,
                                               const Metric* manipulator) const
    {
        int dom = so.get_dom();

        // Call the domain version
        return so.val_t->get_space().get_domain(dom)->derive_flat_mtz(type_der, ind_der, so, manipulator);
    }

    Term_eq Metric_flat::derive_with_other(int type_der, char ind_der, const Term_eq& so,
                                           const Metric* manipulator) const
    {

        int dom = so.get_dom();

        if (p_met_con[dom] == nullptr)
            compute_con(dom);
        if (p_met_cov[dom] == nullptr)
            compute_cov(dom);

        // so must be tensor :
        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("Metric_flat::derive_with_other only defined for tensor data");
        }

        switch (basis.get_basis(dom)) {
            case CARTESIAN_BASIS:
                return derive_with_other_cart(type_der, ind_der, so, manipulator);
            case SPHERICAL_BASIS:
                return derive_with_other_spher(type_der, ind_der, so, manipulator);
            case MTZ_BASIS:
                return derive_with_other_mtz(type_der, ind_der, so, manipulator);
            default:
                KADATH_THROW("Unknown tensorial basis in Metric_flat::derive_with_other");
        }
    }

    Term_eq Metric_flat::derive_partial_cart_with_cached_value(int type_der, char ind_der,
                                                                const Term_eq& so,
                                                                const Tensor& cached_val) const
    {
        int dom = so.get_dom();

        bool donames = ((so.val_t->is_name_affected()) || (so.val_t->get_valence() == 0)) ? true : false;

        bool need_sum = false;
        if (donames)
            for (int i = 0; i < so.val_t->get_valence(); i++)
                if (ind_der == so.val_t->get_name_ind()[i])
                    need_sum = true;

        const Domain* domain = espace.get_domain(dom);
        if (so.der_t != nullptr && need_sum && ((dynamic_cast<const Domain_shell_inner_adapted*>(domain) != nullptr) ||
                                                (dynamic_cast<const Domain_shell_outer_adapted*>(domain) != nullptr))) {
            return derive_partial_cart(type_der, ind_der, so);
        }

        Term_eq auxi(domain->partial_cart_with_cached_value(so, cached_val));

        int val_res = auxi.val_t->get_valence();

        if (donames)
            set_derivative_result_names(auxi, so, ind_der, val_res);

        // v2a: cached_val already encodes post-manipulate_ind (CON) index_type
        // for the val side, and partial_cart_with_cached_value's
        // make_tensor_term_with_derivative_lanes constructs the derivative
        // tensors from cached_val's shape so they inherit the same index_type.
        // No flip is needed for either COV or CON dispatch.

        if (!need_sum)
            return auxi;
        else {
            return make_summed_one_domain_term(dom, auxi);
        }
    }

    Term_eq Metric_flat::derive_partial_spher_with_cached_value(int type_der, char ind_der,
                                                                 const Term_eq& so,
                                                                 const Tensor& cached_val) const
    {
        int dom = so.get_dom();
        bool donames = ((so.val_t->is_name_affected()) || (so.val_t->get_valence() == 0)) ? true : false;

        Term_eq auxi(espace.get_domain(dom)->partial_spher_with_cached_value(so, cached_val));

        int val_res = auxi.val_t->get_valence();

        if (donames)
            set_derivative_result_names(auxi, so, ind_der, val_res);

        // v2a: cached_val already encodes the post-manipulate_ind index_type,
        // and partial_spher_with_cached_value's derivative lanes inherit that
        // index_type from cached_val's shape. No flip needed.
        (void)type_der;

        bool need_sum = false;
        if (donames)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;

        if (!need_sum)
            return auxi;
        else {
            return make_summed_one_domain_term(dom, auxi);
        }
    }

    Term_eq Metric_flat::derive_partial_mtz_with_cached_value(int type_der, char ind_der,
                                                              const Term_eq& so,
                                                              const Tensor& cached_val) const
    {
        int dom = so.get_dom();
        bool donames = ((so.val_t->is_name_affected()) || (so.val_t->get_valence() == 0)) ? true : false;

        Term_eq auxi(espace.get_domain(dom)->partial_mtz_with_cached_value(so, cached_val));

        int val_res = auxi.val_t->get_valence();

        if (donames)
            set_derivative_result_names(auxi, so, ind_der, val_res);

        // v2a: see derive_partial_spher_with_cached_value note above.
        (void)type_der;

        bool need_sum = false;
        if (donames)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;

        if (!need_sum)
            return auxi;
        else {
            return make_summed_one_domain_term(dom, auxi);
        }
    }

    Term_eq Metric_flat::derive_partial_with_cached_value(int type_der, char ind_der,
                                                          const Term_eq& so,
                                                          const Tensor& cached_val) const
    {
        int dom = so.get_dom();

        if (p_met_con[dom] == nullptr)
            compute_con(dom);
        if (p_met_cov[dom] == nullptr)
            compute_cov(dom);

        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("Metric_flat::derive_partial_with_cached_value only defined for tensor data");
        }

        switch (basis.get_basis(dom)) {
            case CARTESIAN_BASIS:
                return derive_partial_cart_with_cached_value(type_der, ind_der, so, cached_val);
            case SPHERICAL_BASIS:
                return derive_partial_spher_with_cached_value(type_der, ind_der, so, cached_val);
            case MTZ_BASIS:
                return derive_partial_mtz_with_cached_value(type_der, ind_der, so, cached_val);
            default:
                KADATH_THROW("Unknown tensorial basis in Metric_flat::derive_partial_with_cached_value");
        }
    }

    Term_eq Metric_flat::derive_cart_with_cached_value(int type_der, char ind_der,
                                                       const Term_eq& so,
                                                       const Tensor& cached_val) const
    {
        return derive_partial_with_cached_value(type_der, ind_der, so, cached_val);
    }

    namespace
    {
        // Replace summed.val_t with cached_val (preserving name_affected from
        // cached_val), keeping summed's derivative lanes intact. set_val_t in
        // Term_eq drops name_affected, so we build a fresh Term_eq from
        // cached_val + summed's derivative side via the (dom, val, der) ctor
        // which preserves names on both.
        Term_eq replace_val_preserve_derivatives(const Term_eq& summed, const Tensor& cached_val)
        {
            const int dom = summed.get_dom();
            if (summed.get_p_der_t() == nullptr) {
                return Term_eq(dom, cached_val);
            }
            Term_eq result(dom, cached_val, summed.get_der_t());
            result.set_derivative_lane_count(summed.get_derivative_lane_count());
            for (int lane = 1; lane < summed.get_derivative_lane_count(); ++lane) {
                if (summed.has_der_t(lane))
                    result.set_der_t(lane, summed.get_der_t(lane));
            }
            return result;
        }
    }

    Term_eq Metric_flat::derive_spher_with_cached_value(int type_der, char ind_der,
                                                        const Term_eq& so,
                                                        const Tensor& cached_val) const
    {
        // derive_spher = derive_partial(so) + connection_spher(so). Use the
        // derivative-only partial + connection paths so neither recomputes its
        // own value side; the per-piece val_t fields are stand-in copies of
        // cached_val. After summing, the result's derivative lanes are the
        // sum we want, and we replace the (incorrect) summed val with the
        // cached_val while preserving names.
        Term_eq part_der(derive_partial_with_cached_value(type_der, ind_der, so, cached_val));

        int dom = so.get_dom();
        bool donames = ((so.val_t->is_name_affected()) || (so.val_t->get_valence() == 0)) ? true : false;

        Term_eq auxi(espace.get_domain(dom)->connection_spher_with_cached_value(so, cached_val));
        int val_res = auxi.val_t->get_valence();

        if (donames)
            set_derivative_result_names(auxi, so, ind_der, val_res);

        if (type_der == CON)
            manipulate_ind(auxi, 0);

        bool need_sum = false;
        if (donames)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;

        Term_eq summed = need_sum ? (part_der + make_summed_one_domain_term(dom, auxi))
                                  : (part_der + auxi);
        return replace_val_preserve_derivatives(summed, cached_val);
    }

    Term_eq Metric_flat::derive_mtz_with_cached_value(int type_der, char ind_der,
                                                      const Term_eq& so,
                                                      const Tensor& cached_val) const
    {
        Term_eq part_der(derive_partial_with_cached_value(type_der, ind_der, so, cached_val));

        int dom = so.get_dom();
        bool donames = ((so.val_t->is_name_affected()) || (so.val_t->get_valence() == 0)) ? true : false;

        Term_eq auxi(espace.get_domain(dom)->connection_mtz_with_cached_value(so, cached_val));
        int val_res = auxi.val_t->get_valence();

        if (donames)
            set_derivative_result_names(auxi, so, ind_der, val_res);

        if (type_der == CON)
            manipulate_ind(auxi, 0);

        bool need_sum = false;
        if (donames)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;

        Term_eq summed = need_sum ? (part_der + make_summed_one_domain_term(dom, auxi))
                                  : (part_der + auxi);
        return replace_val_preserve_derivatives(summed, cached_val);
    }

    Term_eq Metric_flat::derive_with_cached_value(int type_der, char ind_der,
                                                  const Term_eq& so,
                                                  const Tensor& cached_val) const
    {
        int dom = so.get_dom();

        if (p_met_con[dom] == nullptr)
            compute_con(dom);
        if (p_met_cov[dom] == nullptr)
            compute_cov(dom);

        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("Metric_flat::derive_with_cached_value only defined for tensor data");
        }

        switch (basis.get_basis(dom)) {
            case CARTESIAN_BASIS:
                return derive_cart_with_cached_value(type_der, ind_der, so, cached_val);
            case SPHERICAL_BASIS:
                return derive_spher_with_cached_value(type_der, ind_der, so, cached_val);
            case MTZ_BASIS:
                return derive_mtz_with_cached_value(type_der, ind_der, so, cached_val);
            default:
                KADATH_THROW("Unknown tensorial basis in Metric_flat::derive_with_cached_value");
        }
    }

    void Metric_flat::set_system(System_of_eqs& ss, const char* name)
    {

        syst = &ss;
        if (syst->met != nullptr) {
            KADATH_THROW("Metric already set for the system");
        }

        ss.attach_metric(*this, name);
    }
} // namespace Kadath
