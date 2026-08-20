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

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Eq_matching_order_array::Eq_matching_order_array(const Domain* zedom, int dd, int bb, int other_dd, int other_bb,
                                                     const Array<int>& ord, Ope_eq* lhs, Ope_eq* rhs, int nused,
                                                     Array<int>** pused)
        : Equation(zedom, dd, 2, nused, pused), bound(bb), other_dom(other_dd), other_bound(other_bb), order(ord)
    {
        parts[0].reset(lhs);
        parts[1].reset(rhs);
    }

    Eq_matching_order_array::~Eq_matching_order_array() {}

    void Eq_matching_order_array::export_val(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        assert(residus[conte + 1]->get_type_data() == TERM_T);

        const Tensor& local_value = *residus[conte]->get_p_val_t();
        const Tensor& other_value = *residus[conte + 1]->get_p_val_t();
        int start = pos_res;
        dom->export_tau_boundary_array(local_value, ndom, bound, order, sec, pos_res, *n_cond, n_cmp_used,
                                       p_cmp_used);

        Array<double> auxi(pos_res - start);
        auxi = 0.;
        int zero = 0;
        other_value.get_space().get_domain(other_dom)->export_tau_boundary_array(
            other_value, other_dom, other_bound, order, auxi, zero, *n_cond, n_cmp_used, p_cmp_used);

        for (int i = start; i < pos_res; i++)
            sec.set(i) -= auxi(i - start);
        conte += 2;
    }

    void Eq_matching_order_array::export_der(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        assert(residus[conte + 1]->get_type_data() == TERM_T);
        const Tensor& local_derivative = *residus[conte]->get_p_der_t();
        const Tensor& other_value = *residus[conte + 1]->get_p_val_t();
        const Tensor& other_derivative = *residus[conte + 1]->get_p_der_t();
        int start = pos_res;
        dom->export_tau_boundary_array(local_derivative, ndom, bound, order, sec, pos_res, *n_cond, n_cmp_used,
                                       p_cmp_used);
        Array<double> auxi(pos_res - start);
        auxi = 0.;
        int zero = 0;
        other_value.get_space().get_domain(other_dom)->export_tau_boundary_array(
            other_derivative, other_dom, other_bound, order, auxi, zero, *n_cond, n_cmp_used, p_cmp_used);
        for (int i = start; i < pos_res; i++)
            sec.set(i) -= auxi(i - start);
        conte += 2;
    }

    Array<int> Eq_matching_order_array::do_nbr_conditions(const Tensor& tt) const
    {
        return dom->nbr_conditions_boundary_array(tt, ndom, bound, order, n_cmp_used, p_cmp_used);
    }

    bool Eq_matching_order_array::take_into_account(int target) const
    {
        if ((target == ndom) || (target == other_dom))
            return true;
        else
            return false;
    }

} // namespace Kadath
