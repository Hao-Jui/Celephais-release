/*
    Copyright 2020 Philippe Grandclement

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
    Eq_bc_exception::Eq_bc_exception(const Domain* zedom, int dd, int bb, Ope_eq* so, Ope_eq* constant)
        : Equation(zedom, dd, 2), bound(bb)
    {
        parts[0].reset(so);
        parts[1].reset(constant);
    }

    Eq_bc_exception::~Eq_bc_exception() {}

    void Eq_bc_exception::export_val(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        const Tensor& residual_value = *residus[conte]->get_p_val_t();
        assert(residual_value.get_valence() == 0); // only defined for a scalar field so far

        assert(residus[conte + 1]->get_type_data() == TERM_T);
        const Tensor& constant_value = *residus[conte + 1]->get_p_val_t();
        assert(constant_value.get_valence() == 0);

        int old_pos = pos_res;

        dom->export_tau_boundary(residual_value, ndom, bound, sec, pos_res, *n_cond);

        // Get the first coef (a bit long probably but anyway...)
        Array<double> auxi(pos_res - old_pos);
        auxi = 0.;
        int zero = 0;
        dom->export_tau_boundary(constant_value, ndom, bound, auxi, zero, *n_cond);

        // Put the coef in the right place
        sec.set(old_pos) = auxi(0);

        conte += 2;
    }

    void Eq_bc_exception::export_der(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        const Tensor& residual_derivative = *residus[conte]->get_p_der_t();
        assert(residual_derivative.get_valence() == 0); // only defined for a scalar field so far

        assert(residus[conte + 1]->get_type_data() == TERM_T);
        const Tensor& constant_derivative = *residus[conte + 1]->get_p_der_t();
        assert(constant_derivative.get_valence() == 0);

        int old_pos = pos_res;

        dom->export_tau_boundary(residual_derivative, ndom, bound, sec, pos_res, *n_cond);

        // Get the first coef (a bit long probably but anyway...)
        Array<double> auxi(pos_res - old_pos);
        auxi = 0.;
        int zero = 0;
        dom->export_tau_boundary(constant_derivative, ndom, bound, auxi, zero, *n_cond);

        // Put the coef in the right place
        sec.set(old_pos) = auxi(0);

        conte += 2;
    }

    Array<int> Eq_bc_exception::do_nbr_conditions(const Tensor& tt) const
    {
        return dom->nbr_conditions_boundary(tt, ndom, bound);
    }

    bool Eq_bc_exception::take_into_account(int target) const
    {
        if (target == ndom)
            return true;
        else
            return false;
    }

} // namespace Kadath
