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
#include "For_Kadath/Domain/spheric_nosym_regularization.hpp"
#include "For_Kadath/Diagnostics/matching_lane_profile.hpp"
namespace Kadath
{
    namespace
    {
        int matching_val_count(const Val_domain& value, const Dim_array& nbr_coefs, int mlim)
        {
            int res = 0;
            const int kmin = 2 * mlim + 2;

            for (int k = 0; k < nbr_coefs(2) - 1; k++)
                if (k != 1) {
                    const int baset = (*value.get_base().get_base_1d(1))(k);
                    for (int j = 0; j < nbr_coefs(1); j++) {
                        switch (baset) {
                            case COS:
                            case SIN:
                                if (detail::spheric_nosym_true_theta_coef(baset, j, k, kmin, nbr_coefs(1)))
                                    res++;
                                break;
                            default:
                                KADATH_THROW("Unknown theta basis in adapted_nosym import matching count");
                        }
                    }
                }
            return res;
        }

        template <typename CountValDomain>
        Array<int> count_tensor_boundary_matching(const Tensor& tt, int dom_index, int n_cmp, Array<int>** p_cmp,
                                                  CountValDomain count_val_domain)
        {
            const int size = (n_cmp == -1) ? tt.get_n_comp() : n_cmp;
            Array<int> res(size);
            const int val = tt.get_valence();
            switch (val) {
                case 0:
                    if (!tt.is_m_order_affected())
                        res.set(0) = count_val_domain(tt()(dom_index), 0);
                    else
                        res.set(0) = count_val_domain(tt()(dom_index), tt.get_parameters()->get_m_order());
                    break;
                case 1: {
                    bool found = false;
                    if (tt.get_basis().get_basis(dom_index) == CARTESIAN_BASIS) {
                        if (n_cmp == -1) {
                            res.set(0) = count_val_domain(tt(1)(dom_index), 0);
                            res.set(1) = count_val_domain(tt(2)(dom_index), 0);
                            res.set(2) = count_val_domain(tt(3)(dom_index), 0);
                        } else {
                            for (int i = 0; i < n_cmp; i++) {
                                if ((*p_cmp[i])(0) == 1)
                                    res.set(i) = count_val_domain(tt(1)(dom_index), 0);
                                if ((*p_cmp[i])(0) == 2)
                                    res.set(i) = count_val_domain(tt(2)(dom_index), 0);
                                if ((*p_cmp[i])(0) == 3)
                                    res.set(i) = count_val_domain(tt(3)(dom_index), 0);
                            }
                        }
                        found = true;
                    }
                    if (tt.get_basis().get_basis(dom_index) == SPHERICAL_BASIS) {
                        if (n_cmp == -1) {
                            res.set(0) = count_val_domain(tt(1)(dom_index), 0);
                            res.set(1) = count_val_domain(tt(2)(dom_index), 1);
                            res.set(2) = count_val_domain(tt(3)(dom_index), 1);
                        } else {
                            for (int i = 0; i < n_cmp; i++) {
                                if ((*p_cmp[i])(0) == 1)
                                    res.set(i) = count_val_domain(tt(1)(dom_index), 0);
                                if ((*p_cmp[i])(0) == 2)
                                    res.set(i) = count_val_domain(tt(2)(dom_index), 1);
                                if ((*p_cmp[i])(0) == 3)
                                    res.set(i) = count_val_domain(tt(3)(dom_index), 1);
                            }
                        }
                        found = true;
                    }
                    if (!found)
                        KADATH_THROW("Unknown vector basis in adapted_nosym import matching count");
                } break;
                case 2: {
                    bool found = false;
                    if ((tt.get_basis().get_basis(dom_index) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                        if (n_cmp == -1) {
                            res.set(0) = count_val_domain(tt(1, 1)(dom_index), 0);
                            res.set(1) = count_val_domain(tt(1, 2)(dom_index), 0);
                            res.set(2) = count_val_domain(tt(1, 3)(dom_index), 0);
                            res.set(3) = count_val_domain(tt(2, 2)(dom_index), 0);
                            res.set(4) = count_val_domain(tt(2, 3)(dom_index), 0);
                            res.set(5) = count_val_domain(tt(3, 3)(dom_index), 0);
                        } else {
                            for (int i = 0; i < n_cmp; i++) {
                                if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                    res.set(i) = count_val_domain(tt(1, 1)(dom_index), 0);
                                if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                    res.set(i) = count_val_domain(tt(1, 2)(dom_index), 0);
                                if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                    res.set(i) = count_val_domain(tt(1, 3)(dom_index), 0);
                                if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                    res.set(i) = count_val_domain(tt(2, 2)(dom_index), 0);
                                if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                    res.set(i) = count_val_domain(tt(2, 3)(dom_index), 0);
                                if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                    res.set(i) = count_val_domain(tt(3, 3)(dom_index), 0);
                            }
                        }
                        found = true;
                    }
                    if (!found)
                        KADATH_THROW("Unknown tensor basis in adapted_nosym import matching count");
                } break;
                default:
                    KADATH_THROW("Valence not implemented in adapted_nosym import matching count");
            }
            return res;
        }

        template <typename ExportValDomain>
        void export_tensor_boundary_matching(const Tensor& tt, int dom_index, int bound, Array<double>& res,
                                             int& pos_res, const Array<int>& ncond, int n_cmp, Array<int>** p_cmp,
                                             ExportValDomain export_val_domain)
        {
            const int val = tt.get_valence();
            switch (val) {
                case 0:
                    if (!tt.is_m_order_affected())
                        export_val_domain(tt()(dom_index), 0, bound, res, pos_res, ncond(0));
                    else
                        export_val_domain(tt()(dom_index), tt.get_parameters()->get_m_order(), bound, res, pos_res,
                                          ncond(0));
                    break;
                case 1: {
                    bool found = false;
                    if (tt.get_basis().get_basis(dom_index) == CARTESIAN_BASIS) {
                        if (n_cmp == -1) {
                            export_val_domain(tt(1)(dom_index), 0, bound, res, pos_res, ncond(0));
                            export_val_domain(tt(2)(dom_index), 0, bound, res, pos_res, ncond(1));
                            export_val_domain(tt(3)(dom_index), 0, bound, res, pos_res, ncond(2));
                        } else {
                            for (int i = 0; i < n_cmp; i++) {
                                if ((*p_cmp[i])(0) == 1)
                                    export_val_domain(tt(1)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if ((*p_cmp[i])(0) == 2)
                                    export_val_domain(tt(2)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if ((*p_cmp[i])(0) == 3)
                                    export_val_domain(tt(3)(dom_index), 0, bound, res, pos_res, ncond(i));
                            }
                        }
                        found = true;
                    }
                    if (tt.get_basis().get_basis(dom_index) == SPHERICAL_BASIS) {
                        if (n_cmp == -1) {
                            export_val_domain(tt(1)(dom_index), 0, bound, res, pos_res, ncond(0));
                            export_val_domain(tt(2)(dom_index), 1, bound, res, pos_res, ncond(1));
                            export_val_domain(tt(3)(dom_index), 1, bound, res, pos_res, ncond(2));
                        } else {
                            for (int i = 0; i < n_cmp; i++) {
                                if ((*p_cmp[i])(0) == 1)
                                    export_val_domain(tt(1)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if ((*p_cmp[i])(0) == 2)
                                    export_val_domain(tt(2)(dom_index), 1, bound, res, pos_res, ncond(i));
                                if ((*p_cmp[i])(0) == 3)
                                    export_val_domain(tt(3)(dom_index), 1, bound, res, pos_res, ncond(i));
                            }
                        }
                        found = true;
                    }
                    if (!found)
                        KADATH_THROW("Unknown vector basis in adapted_nosym import matching export");
                } break;
                case 2:
                    KADATH_THROW("2-tensor adapted_nosym import matching export not implemented");
                default:
                    KADATH_THROW("Valence not implemented in adapted_nosym import matching export");
            }
        }

        Array<int> adapted_matching_counts(const Domain* domain, const Tensor& tt, int dom_index, int n_cmp,
                                           Array<int>** p_cmp)
        {
            if (const auto* inner = dynamic_cast<const Domain_shell_inner_adapted_nosym*>(domain)) {
                return count_tensor_boundary_matching(tt, dom_index, n_cmp, p_cmp,
                    [&](const Val_domain& val, int mlim) {
                        return matching_val_count(val, inner->get_nbr_coefs(), mlim);
                    });
            }
            if (const auto* outer = dynamic_cast<const Domain_shell_outer_adapted_nosym*>(domain)) {
                return count_tensor_boundary_matching(tt, dom_index, n_cmp, p_cmp,
                    [&](const Val_domain& val, int mlim) {
                        return matching_val_count(val, outer->get_nbr_coefs(), mlim);
                    });
            }
            return domain->nbr_conditions_boundary(tt, dom_index, INNER_BC, n_cmp, p_cmp);
        }

        bool export_adapted_matching(const Domain* domain, const Tensor& tt, int dom_index, int bound,
                                     Array<double>& sec, int& pos_res, const Array<int>& ncond, int n_cmp,
                                     Array<int>** p_cmp)
        {
            if (const auto* inner = dynamic_cast<const Domain_shell_inner_adapted_nosym*>(domain)) {
                export_tensor_boundary_matching(tt, dom_index, bound, sec, pos_res, ncond, n_cmp, p_cmp,
                    [&](const Val_domain& val, int mlim, int val_bound, Array<double>& val_sec, int& val_pos,
                        int val_ncond) {
                        inner->export_tau_val_domain_boundary_matching(val, mlim, val_bound, val_sec, val_pos,
                                                                       val_ncond);
                    });
                return true;
            }
            if (const auto* outer = dynamic_cast<const Domain_shell_outer_adapted_nosym*>(domain)) {
                export_tensor_boundary_matching(tt, dom_index, bound, sec, pos_res, ncond, n_cmp, p_cmp,
                    [&](const Val_domain& val, int mlim, int val_bound, Array<double>& val_sec, int& val_pos,
                        int val_ncond) {
                        outer->export_tau_val_domain_boundary_matching(val, mlim, val_bound, val_sec, val_pos,
                                                                       val_ncond);
                    });
                return true;
            }
            return false;
        }
    }

    Eq_matching_import::Eq_matching_import(const Domain* zedom, int dd, int bb, Ope_eq* so, const Array<int>& ozers,
                                           int nused, Array<int>** pused)
        : Equation(zedom, dd, 1, nused, pused), bound(bb), other_doms(ozers.get_size(1)),
          other_bounds(ozers.get_size(1))
    {
        parts[0].reset(so);
        for (int i = 0; i < other_doms.get_size(0); i++) {
            other_doms.set(i) = ozers(0, i);
            other_bounds.set(i) = ozers(1, i);
        }
    }

    Eq_matching_import::~Eq_matching_import() {}

    void Eq_matching_import::export_val(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        const Tensor& residual_value = *residus[conte]->get_p_val_t();
        if (!export_adapted_matching(dom, residual_value, ndom, bound, sec, pos_res, *n_cond, n_cmp_used, p_cmp_used))
            dom->export_tau_boundary(residual_value, ndom, bound, sec, pos_res, *n_cond, n_cmp_used, p_cmp_used);
        conte++;
    }

    void Eq_matching_import::export_der(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {
        assert(residus[conte]->get_type_data() == TERM_T);
        const Tensor& residual_derivative = *residus[conte]->get_p_der_t();
        if (!export_adapted_matching(dom, residual_derivative, ndom, bound, sec, pos_res, *n_cond, n_cmp_used,
                                     p_cmp_used))
            dom->export_tau_boundary(residual_derivative, ndom, bound, sec, pos_res, *n_cond, n_cmp_used, p_cmp_used);
        conte++;
    }

    void Eq_matching_import::export_der_lanes(int& conte, Term_eq** residus, int lane_count,
                                              Array<double>* const* secs, int* pos_res_arr) const
    {
        MatchingLaneStats& stats = matching_lane_stats_state();
        if (!matching_lane_export_enabled()) {
            ++stats.export_scalar_fallback_calls;
            Equation::export_der_lanes(conte, residus, lane_count, secs, pos_res_arr);
            return;
        }

        assert(residus[conte]->get_type_data() == TERM_T);
        Term_eq* result = residus[conte];
        for (int lane = 0; lane < lane_count; ++lane) {
            const Tensor* derivative = result->get_p_der_t(lane);
            if (derivative == nullptr) {
                const int start = pos_res_arr[lane];
                for (int row = 0; row < n_cond_tot; ++row)
                    secs[lane]->set(start + row) = 0.0;
                pos_res_arr[lane] += n_cond_tot;
                ++stats.export_missing_lanes;
                continue;
            }
            if (!export_adapted_matching(dom, *derivative, ndom, bound, *secs[lane], pos_res_arr[lane],
                                         *n_cond, n_cmp_used, p_cmp_used))
                dom->export_tau_boundary(*derivative, ndom, bound, *secs[lane], pos_res_arr[lane], *n_cond,
                                         n_cmp_used, p_cmp_used);
        }
        ++conte;
        ++stats.export_native_calls;
    }

    bool Eq_matching_import::describe_residual_rows(
        int& conte, Term_eq** residuals, int equation_index,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        if (!called || parts[0] == nullptr ||
            !parts[0]->preserves_reflection_parity() || residuals == nullptr ||
            residuals[conte] == nullptr ||
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
            // The import operator contract certifies reflection-sector
            // preservation. Remote coefficient provenance is non-unique for
            // configuration-space interpolation, so the canonical local
            // coordinate is intentionally the descriptor's only side.
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

    Array<int> Eq_matching_import::do_nbr_conditions(const Tensor& tt) const
    {
        if ((dynamic_cast<const Domain_shell_inner_adapted_nosym*>(dom) != nullptr) ||
            (dynamic_cast<const Domain_shell_outer_adapted_nosym*>(dom) != nullptr))
            return adapted_matching_counts(dom, tt, ndom, n_cmp_used, p_cmp_used);
        return dom->nbr_conditions_boundary(tt, ndom, bound, n_cmp_used, p_cmp_used);
    }

    bool Eq_matching_import::take_into_account(int target) const
    {

        bool res = (target == ndom) ? true : false;
        for (int i = 0; i < other_doms.get_size(0); i++)
            if (target == other_doms(i))
                res = true;
        return res;
    }
} // namespace Kadath
