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

#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Metric/metric_nophi.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "For_Kadath/Tensor/metric_tensor.hpp"
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

    Metric_flat_nophi::Metric_flat_nophi(const Space& sp, const Base_tensor& bb) : Metric(sp), basis(bb)
    {

        for (int d = 0; d < sp.get_nbr_domains(); d++)
            if (bb.get_basis(d) != SPHERICAL_BASIS) {
                KADATH_THROW("Metric_flat_nophi only defined wrt spherical tensorial basis for now...");
            }
    }

    Metric_flat_nophi::Metric_flat_nophi(const Metric_flat_nophi& so) : Metric(so), basis(so.basis) {}

    Metric_flat_nophi::~Metric_flat_nophi() {}

    void Metric_flat_nophi::update()
    {
        // Nothing to do everything is constant
    }

    void Metric_flat_nophi::update(int)
    {
        // Nothing to do everything is constant
    }

    void Metric_flat_nophi::compute_cov(int dd) const
    {
        Metric_tensor res(espace, COV, basis);

        for (int i = 1; i <= 3; i++)
            for (int j = i; j <= 3; j++)
                res.set(i, j).set_domain(dd) = (i == j) ? 1. : 0;
        res.std_base();

        if (p_met_cov[dd] == nullptr)
            p_met_cov[dd] = std::make_unique<Term_eq>(dd, res);
        else
            *p_met_cov[dd] = Term_eq(dd, res);
        p_met_cov[dd]->set_der_zero();
    }

    int Metric_flat_nophi::give_type(int dd) const
    {
        return basis.get_basis(dd);
    }

    void Metric_flat_nophi::compute_con(int dd) const
    {

        Metric_tensor res(espace, CON, basis);
        for (int i = 1; i <= 3; i++)
            for (int j = i; j <= 3; j++)
                res.set(i, j).set_domain(dd) = (i == j) ? 1. : 0;
        res.std_base();
        if (p_met_con[dd] == nullptr)
            p_met_con[dd] = std::make_unique<Term_eq>(dd, res);
        else
            *p_met_con[dd] = Term_eq(dd, res);
        p_met_con[dd]->set_der_zero();
    }

    void Metric_flat_nophi::compute_christo(int) const
    {
        KADATH_THROW("Computation of Christo not explicit for Metric_flat_nophi");
    }

    void Metric_flat_nophi::manipulate_ind(Term_eq& so, int ind) const
    {
        // Just change the type of the indice !
        so.set_val_t()->set_index_type(ind) *= -1;
        for (int lane = 0; lane < so.get_derivative_lane_count(); ++lane)
            if (so.has_der_t(lane))
                so.set_der_t(lane)->set_index_type(ind) *= -1;
    }

    Term_eq Metric_flat_nophi::derive_partial_spher(int type_der, char ind_der, const Term_eq& so) const
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

    Term_eq Metric_flat_nophi::derive_partial(int type_der, char ind_der, const Term_eq& so) const
    {

        int dom = so.get_dom();

        if (p_met_con[dom] == nullptr)
            compute_con(dom);
        if (p_met_cov[dom] == nullptr)
            compute_cov(dom);

        // so must be tensor :
        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("Metric_flat_nophi::derive partial only defined for tensor data");
        }

        switch (basis.get_basis(dom)) {
            case SPHERICAL_BASIS:
                return derive_partial_spher(type_der, ind_der, so);
            default:
                KADATH_THROW("Unknown tensorial basis in Metric_flat_nophi::derive_partial");
        }
    }

    Term_eq Metric_flat_nophi::derive_spher(int type_der, char ind_der, const Term_eq& so) const
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

    Term_eq Metric_flat_nophi::derive(int type_der, char ind_der, const Term_eq& so) const
    {

        int dom = so.get_dom();

        if (p_met_con[dom] == nullptr)
            compute_con(dom);
        if (p_met_cov[dom] == nullptr)
            compute_cov(dom);

        // so must be tensor :
        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("Metric_flat_nophi::derive only defined for tensor data");
        }

        switch (basis.get_basis(dom)) {
            case SPHERICAL_BASIS:
                return derive_spher(type_der, ind_der, so);
            default:
                KADATH_THROW("Unknown tensorial basis in Metric_flat_nophi::derive");
        }
    }

    Term_eq Metric_flat_nophi::derive_with_other_spher(int type_der, char ind_der, const Term_eq& so,
                                                       const Metric* manipulator) const
    {
        int dom = so.get_dom();

        // Call the domain version
        return so.val_t->get_space().get_domain(dom)->derive_flat_spher(type_der, ind_der, so, manipulator);
    }

    Term_eq Metric_flat_nophi::derive_with_other(int type_der, char ind_der, const Term_eq& so,
                                                 const Metric* manipulator) const
    {

        int dom = so.get_dom();

        if (p_met_con[dom] == nullptr)
            compute_con(dom);
        if (p_met_cov[dom] == nullptr)
            compute_cov(dom);

        // so must be tensor :
        if (so.get_type_data() != TERM_T) {
            KADATH_THROW("Metric_flat_nophi::derive_with_other only defined for tensor data");
        }

        switch (basis.get_basis(dom)) {
            case SPHERICAL_BASIS:
                return derive_with_other_spher(type_der, ind_der, so, manipulator);
            default:
                KADATH_THROW("Unknown tensorial basis in Metric_flat_nophi::derive_with_other");
        }
    }

    void Metric_flat_nophi::set_system(System_of_eqs& ss, const char* name)
    {

        syst = &ss;
        if (syst->met != nullptr) {
            KADATH_THROW("Metric already set for the system");
        }

        ss.attach_metric(*this, name);
    }
} // namespace Kadath
