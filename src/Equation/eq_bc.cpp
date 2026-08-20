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

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
namespace Kadath
{
    namespace
    {
        bool is_projected_boundary_domain(const Domain* domain)
        {
            return dynamic_cast<const Domain_shell_inner_adapted_nosym*>(domain) != nullptr ||
                   dynamic_cast<const Domain_shell_outer_adapted_nosym*>(domain) != nullptr;
        }

        Array<int> projected_boundary_counts(const Domain* domain, const Tensor& value, int dom_index,
                                             int bound, int n_cmp, Array<int>** p_cmp)
        {
            if (value.get_valence() != 0)
                KADATH_THROW("Projected boundary conditions are implemented for scalar equations only");
            return domain->nbr_conditions_boundary(value, dom_index, bound, n_cmp, p_cmp);
        }

        bool export_projected_boundary(const Domain* domain, const Tensor& value, int dom_index, int bound,
                                       Array<double>& sec, int& pos_res, const Array<int>& ncond,
                                       int n_cmp, Array<int>** p_cmp)
        {
            if (value.get_valence() != 0)
                KADATH_THROW("Projected boundary conditions are implemented for scalar equations only");

            auto export_scalar = [&](const auto* adapted_domain) {
                if (!value.is_m_order_affected())
                    adapted_domain->export_tau_val_domain_boundary_matching(value()(dom_index), 0, bound, sec,
                                                                            pos_res, ncond(0));
                else
                    adapted_domain->export_tau_val_domain_boundary_matching(
                        value()(dom_index), value.get_parameters()->get_m_order(), bound, sec, pos_res, ncond(0));
            };

            if (const auto* inner = dynamic_cast<const Domain_shell_inner_adapted_nosym*>(domain)) {
                export_scalar(inner);
                return true;
            }
            if (const auto* outer = dynamic_cast<const Domain_shell_outer_adapted_nosym*>(domain)) {
                export_scalar(outer);
                return true;
            }
            return false;
        }
    }

    Eq_bc::Eq_bc(const Domain* zedom, int dd, int bb, Ope_eq* so, int nused, Array<int>** pused)
        : Eq_bc(zedom, dd, bb, so, false, nused, pused)
    {
    }

    Eq_bc::Eq_bc(const Domain* zedom, int dd, int bb, Ope_eq* so, bool projected, int nused, Array<int>** pused)
        : Equation(zedom, dd, 1, nused, pused), bound(bb), projected_boundary(projected)
    {
        parts[0].reset(so);
    }

    Eq_bc::~Eq_bc() {}

    void Eq_bc::export_val(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        const Tensor& residual_value = *residus[conte]->get_p_val_t();
        if (!projected_boundary ||
            !export_projected_boundary(dom, residual_value, ndom, bound, sec, pos_res, *n_cond, n_cmp_used,
                                       p_cmp_used))
            dom->export_tau_boundary(residual_value, ndom, bound, sec, pos_res, *n_cond, n_cmp_used, p_cmp_used);
        conte++;
    }

    void Eq_bc::export_der(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        const Tensor& residual_derivative = *residus[conte]->get_p_der_t();
        if (!projected_boundary ||
            !export_projected_boundary(dom, residual_derivative, ndom, bound, sec, pos_res, *n_cond, n_cmp_used,
                                       p_cmp_used))
            dom->export_tau_boundary(residual_derivative, ndom, bound, sec, pos_res, *n_cond, n_cmp_used, p_cmp_used);
        conte++;
    }

    bool Eq_bc::describe_residual_rows(
        int& conte, Term_eq** residuals, int equation_index,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        if (!called || residuals == nullptr || residuals[conte] == nullptr ||
            residuals[conte]->get_type_data() != TERM_T) {
            ++conte;
            return false;
        }

        const Tensor& residual = *residuals[conte]->get_p_val_t();
        ++conte;
        if (!dom->describe_boundary_residual_rows(
                residual, ndom, bound, *n_cond, n_cmp_used, p_cmp_used,
                descriptors) ||
            descriptors.size() != static_cast<std::size_t>(n_cond_tot)) {
            descriptors.clear();
            return false;
        }
        for (ResidualRowDescriptor& descriptor : descriptors) {
            if (descriptor.family != ResidualRowEquationFamily::Unavailable ||
                descriptor.available || descriptor.equation_index != -1 ||
                descriptor.explicit_sector != 0 || descriptor.sides.size() != 1) {
                descriptors.clear();
                return false;
            }
            const ResidualRowCoordinate& coordinate = descriptor.sides.front();
            if (coordinate.domain != ndom || coordinate.component < 0 ||
                coordinate.component >= residual.get_n_comp() ||
                coordinate.phi_index < 0 || coordinate.phi_basis == 0 ||
                coordinate.phi_index >= dom->get_nbr_coefs()(2)) {
                descriptors.clear();
                return false;
            }
            descriptor.family = ResidualRowEquationFamily::Field;
            descriptor.equation_index = equation_index;
            descriptor.available = true;
        }
        return true;
    }

    Array<int> Eq_bc::do_nbr_conditions(const Tensor& tt) const
    {
        if (projected_boundary && is_projected_boundary_domain(dom))
            return projected_boundary_counts(dom, tt, ndom, bound, n_cmp_used, p_cmp_used);
        return dom->nbr_conditions_boundary(tt, ndom, bound, n_cmp_used, p_cmp_used);
    }

    bool Eq_bc::take_into_account(int target) const
    {
        if (target == ndom)
            return true;
        else
            return false;
    }
} // namespace Kadath
