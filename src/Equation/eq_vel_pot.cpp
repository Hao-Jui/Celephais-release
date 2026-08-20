/*
    Copyright 2019 Philippe Grandclement

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
    Eq_vel_pot::Eq_vel_pot(const Domain* zedom, int dd, int ord, Ope_eq* so,
                           Ope_eq* constant, bool same_sector)
        : Equation(zedom, dd, 2), order(ord),
          same_reflection_sector(same_sector)
    {
        parts[0].reset(so);
        parts[1].reset(constant);
    }

    Eq_vel_pot::~Eq_vel_pot() {}

    void Eq_vel_pot::export_val(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        const Tensor& residual_value = *residus[conte]->get_p_val_t();
        assert(residual_value.get_valence() == 0); // only defined for a scalar field so far

        assert(residus[conte + 1]->get_type_data() == TERM_T);
        const Tensor& constant_value = *residus[conte + 1]->get_p_val_t();
        assert(constant_value.get_valence() == 0);

        int old_pos = pos_res;

        dom->export_tau(residual_value, ndom, order, sec, pos_res, *n_cond);

        // Get the first coef (a bit long probably but anyway...)
        Array<double> auxi(pos_res - old_pos);
        auxi = 0.;
        int zero = 0;
        dom->export_tau(constant_value, ndom, order, auxi, zero, *n_cond);

        // Put the coef in the right place
        sec.set(old_pos) = auxi(0);

        conte += 2;
    }

    void Eq_vel_pot::export_der(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        const Tensor& residual_derivative = *residus[conte]->get_p_der_t();
        assert(residual_derivative.get_valence() == 0); // only defined for a scalar field so far

        assert(residus[conte + 1]->get_type_data() == TERM_T);
        const Tensor& constant_derivative = *residus[conte + 1]->get_p_der_t();
        assert(constant_derivative.get_valence() == 0);

        int old_pos = pos_res;

        dom->export_tau(residual_derivative, ndom, order, sec, pos_res, *n_cond);

        // Get the first coef (a bit long probably but anyway...)
        Array<double> auxi(pos_res - old_pos);
        auxi = 0.;
        int zero = 0;
        dom->export_tau(constant_derivative, ndom, order, auxi, zero, *n_cond);

        // Put the coef in the right place
        sec.set(old_pos) = auxi(0);

        conte += 2;
    }

    Array<int> Eq_vel_pot::do_nbr_conditions(const Tensor& tt) const
    {
        return dom->nbr_conditions(tt, ndom, order);
    }

    bool Eq_vel_pot::describe_residual_rows(
        int& conte, Term_eq** residuals, int equation_index,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        if (!called || !same_reflection_sector || residuals == nullptr ||
            residuals[conte] == nullptr ||
            residuals[conte + 1] == nullptr ||
            residuals[conte]->get_type_data() != TERM_T ||
            residuals[conte + 1]->get_type_data() != TERM_T) {
            conte += 2;
            return false;
        }

        std::vector<ResidualRowDescriptor> ordinary;
        std::vector<ResidualRowDescriptor> constant;
        const Tensor& ordinary_value = *residuals[conte]->get_p_val_t();
        const Tensor& constant_value = *residuals[conte + 1]->get_p_val_t();
        conte += 2;
        if (!dom->describe_volume_residual_rows(
                ordinary_value, ndom, order, *n_cond, -1, nullptr, ordinary) ||
            !dom->describe_volume_residual_rows(
                constant_value, ndom, order, *n_cond, -1, nullptr, constant) ||
            ordinary.size() != constant.size() ||
            ordinary.size() != static_cast<std::size_t>(n_cond_tot)) {
            return false;
        }

        for (std::size_t row = 0; row < ordinary.size(); ++row) {
            if (ordinary[row].family != ResidualRowEquationFamily::Unavailable ||
                constant[row].family != ResidualRowEquationFamily::Unavailable ||
                ordinary[row].available || constant[row].available ||
                ordinary[row].equation_index != -1 ||
                constant[row].equation_index != -1 ||
                ordinary[row].explicit_sector != 0 ||
                constant[row].explicit_sector != 0 ||
                ordinary[row].sides.size() != 1 ||
                constant[row].sides.size() != 1) {
                return false;
            }
            const ResidualRowCoordinate& lhs = ordinary[row].sides.front();
            const ResidualRowCoordinate& rhs = constant[row].sides.front();
            const auto coordinate_in_bounds = [](
                                                  const Tensor& residual,
                                                  const ResidualRowCoordinate& coordinate) {
                return coordinate.domain >= 0 &&
                       coordinate.domain <
                           residual.get_space().get_nbr_domains() &&
                       coordinate.component >= 0 &&
                       coordinate.component < residual.get_n_comp() &&
                       coordinate.phi_index >= 0 &&
                       coordinate.phi_index <
                           residual.get_space()
                               .get_domain(coordinate.domain)
                               ->get_nbr_coefs()(2);
            };
            if (!coordinate_in_bounds(ordinary_value, lhs) ||
                !coordinate_in_bounds(constant_value, rhs) ||
                lhs.domain != rhs.domain || lhs.component != rhs.component ||
                lhs.phi_basis != rhs.phi_basis ||
                lhs.phi_index != rhs.phi_index) {
                return false;
            }
            ordinary[row].family = ResidualRowEquationFamily::Field;
            ordinary[row].equation_index = equation_index;
            ordinary[row].available = true;
        }
        descriptors = std::move(ordinary);
        return true;
    }

    bool Eq_vel_pot::take_into_account(int target) const
    {
        if (target == ndom)
            return true;
        else
            return false;
    }

} // namespace Kadath
