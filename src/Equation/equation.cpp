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
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Diagnostics/kernel_profile.hpp"
namespace Kadath
{
    Equation::Equation(const Domain* zedom, int nd, int np, int nused, Array<int>** pused)
        : dom(zedom), ndom(nd), n_ope(np), parts(n_ope), called(false), n_cond(nullptr), n_cmp_used(nused),
          p_cmp_used(pused)
    {}

    Equation::~Equation()
    {
        if (called)
            delete n_cond;
    }

    void Equation::apply(int& conte, Term_eq** res)
    {
        KernelTimer kernel_timer(KernelId::EquationApply);
        ApplyScope apply_scope;
        int old_conte = conte;
        for (int i = 0; i < n_ope; i++) {
            if (res[conte] != nullptr) {
                if (ope_action_write_into_enabled())
                    parts[i]->action_into(*res[conte]);
                else
                    *res[conte] = parts[i]->action();
            }
            else
                res[conte] = new Term_eq(parts[i]->action());
            conte++;
        }
        if (!called) {
            Tensor copie(*res[old_conte]->get_p_val_t());
            n_cond = new Array<int>(do_nbr_conditions(copie));
            n_comp = n_cond->get_size(0);
            n_cond_tot = 0;
            for (int i = 0; i < n_cond->get_size(0); i++)
                n_cond_tot += (*n_cond)(i);
            called = true;
        }
    }

    // Default packed-export fallback: per-lane local primary-slot swap on the
    // n_ope Term_eq instances this equation reads, then dispatch to the scalar
    // export_der once per lane. Byte-identical to the legacy global-swap path
    // because only slots this equation reads are mutated, and other equations'
    // slots are handled by their own dispatch.
    void Equation::export_der_lanes(int& conte, Term_eq** residus,
                                     int lane_count,
                                     Array<double>* const* secs,
                                     int* pos_res_arr) const
    {
        const int conte_at_start = conte;
        for (int lane = 0; lane < lane_count; ++lane) {
            if (lane != 0) {
                for (int t = conte_at_start; t < conte_at_start + n_ope; ++t) {
                    Term_eq* result = residus[t];
                    if (result == nullptr)
                        continue;
                    if (result->get_type_data() == TERM_D) {
                        result->set_der_d(result->has_der_d(lane) ? result->get_der_d(lane) : 0.0);
                    } else {
                        if (result->has_der_t(lane))
                            result->set_der_t(result->get_der_t(lane));
                        else
                            result->set_der_zero(0);
                    }
                }
            }
            int lane_conte = conte_at_start;
            int lane_pos = pos_res_arr[lane];
            export_der(lane_conte, residus, *secs[lane], lane_pos);
            pos_res_arr[lane] = lane_pos;
        }
        conte = conte_at_start + n_ope;
    }

    bool Equation::describe_residual_rows(
        int& conte, Term_eq**, int,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        conte += n_ope;
        descriptors.clear();
        return false;
    }

    bool Equation::describe_residual_rows_single_tau(
        int& conte, Term_eq** residuals, int equation_index, int selector,
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
        if (!dom->describe_volume_residual_rows(
                residual, ndom, selector, *n_cond, n_cmp_used, p_cmp_used,
                descriptors) ||
            descriptors.size() != static_cast<std::size_t>(n_cond_tot)) {
            descriptors.clear();
            return false;
        }
        for (ResidualRowDescriptor& descriptor : descriptors) {
            if (descriptor.family != ResidualRowEquationFamily::Unavailable ||
                descriptor.available || descriptor.equation_index != -1 ||
                descriptor.explicit_sector != 0 ||
                descriptor.sides.size() != 1) {
                descriptors.clear();
                return false;
            }
            const ResidualRowCoordinate& coordinate = descriptor.sides.front();
            if (coordinate.domain != ndom || coordinate.component < 0 ||
                coordinate.component >= residual.get_n_comp() ||
                coordinate.phi_index < 0 ||
                coordinate.domain < 0 ||
                coordinate.domain >= residual.get_space().get_nbr_domains() ||
                coordinate.phi_index >=
                    residual.get_space()
                        .get_domain(coordinate.domain)
                        ->get_nbr_coefs()(2) ||
                coordinate.phi_basis == 0) {
                descriptors.clear();
                return false;
            }
            descriptor.family = ResidualRowEquationFamily::Field;
            descriptor.equation_index = equation_index;
            descriptor.available = true;
        }
        return true;
    }

    // Shared lane-direct fast path for n_ope=1 single-TERM_T equations whose
    // scalar export_der is exactly export_tau(*get_p_der_t(), ndom, selector,
    // ...). Extracted verbatim from the Eq_full / Eq_inside overrides (only the
    // selector differs), so it is byte-identical to those paths. Relies on the
    // engine having zeroed this equation's active row range before dispatch: a
    // missed lane skips export_tau and merely advances pos_res_arr, leaving the
    // pre-zeroed rows in place rather than allocating a zero tensor.
    void Equation::export_der_lanes_single_tau(int& conte, Term_eq** residus,
                                               int lane_count,
                                               Array<double>* const* secs,
                                               int* pos_res_arr,
                                               int selector) const
    {
        assert(residus[conte]->get_type_data() == TERM_T);
        Term_eq* result = residus[conte];
        for (int lane = 0; lane < lane_count; ++lane) {
            if (!result->has_der_t(lane)) {
                pos_res_arr[lane] += n_cond_tot;
                continue;
            }
            const Tensor& residual_derivative = *result->get_p_der_t(lane);
            dom->export_tau(residual_derivative, ndom, selector, *secs[lane],
                            pos_res_arr[lane], *n_cond, n_cmp_used, p_cmp_used);
        }
        conte++;
    }

} // namespace Kadath
