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
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Eq_order::Eq_order(const Domain* zedom, int dd, int ord, Ope_eq* so, int nused, Array<int>** pused)
        : Equation(zedom, dd, 1, nused, pused)
    {
        order = ord;
        parts[0].reset(so);
    }

    Eq_order::~Eq_order() {}

    void Eq_order::export_val(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        const Tensor& residual_value = *residus[conte]->get_p_val_t();
        dom->export_tau(residual_value, ndom, order, sec, pos_res, *n_cond, n_cmp_used, p_cmp_used);
        conte++;
    }

    void Eq_order::export_der(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        const Tensor& residual_derivative = *residus[conte]->get_p_der_t();
        dom->export_tau(residual_derivative, ndom, order, sec, pos_res, *n_cond, n_cmp_used, p_cmp_used);
        conte++;
    }

    // Lane-direct packed export. Selector `order` matches export_der;
    // delegates to the shared helper, which skips missed lanes (engine
    // pre-zeroes this equation's active rows). See
    // Equation::export_der_lanes_single_tau for the byte-identity rationale.
    void Eq_order::export_der_lanes(int& conte, Term_eq** residus,
                                    int lane_count,
                                    Array<double>* const* secs,
                                    int* pos_res_arr) const
    {
        export_der_lanes_single_tau(conte, residus, lane_count, secs, pos_res_arr, order);
    }

    Array<int> Eq_order::do_nbr_conditions(const Tensor& tt) const
    {
        return dom->nbr_conditions(tt, ndom, order, n_cmp_used, p_cmp_used);
    }

    bool Eq_order::describe_residual_rows(
        int& conte, Term_eq** residuals, int equation_index,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        return describe_residual_rows_single_tau(
            conte, residuals, equation_index, order, descriptors);
    }

    bool Eq_order::take_into_account(int target) const
    {
        if (target == ndom)
            return true;
        else
            return false;
    }

} // namespace Kadath
