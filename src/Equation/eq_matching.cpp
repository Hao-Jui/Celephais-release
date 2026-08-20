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
#include "For_Kadath/Diagnostics/kernel_profile.hpp"
#include "For_Kadath/Diagnostics/matching_lane_profile.hpp"
namespace Kadath
{
    namespace
    {
        template <typename ExportValDomain>
        void export_tensor_boundary(const Tensor& tt, int dom_index, int bound, Array<double>& res, int& pos_res,
                                    const Array<int>& ncond, int n_cmp, Array<int>** p_cmp,
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
                        KADATH_THROW("Unknown vector basis in Eq_matching adapted_nosym boundary export");
                } break;
                case 2: {
                    bool found = false;
                    if ((tt.get_basis().get_basis(dom_index) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                        if (n_cmp == -1) {
                            export_val_domain(tt(1, 1)(dom_index), 0, bound, res, pos_res, ncond(0));
                            export_val_domain(tt(1, 2)(dom_index), 0, bound, res, pos_res, ncond(1));
                            export_val_domain(tt(1, 3)(dom_index), 0, bound, res, pos_res, ncond(2));
                            export_val_domain(tt(2, 2)(dom_index), 0, bound, res, pos_res, ncond(3));
                            export_val_domain(tt(2, 3)(dom_index), 0, bound, res, pos_res, ncond(4));
                            export_val_domain(tt(3, 3)(dom_index), 0, bound, res, pos_res, ncond(5));
                        } else {
                            for (int i = 0; i < n_cmp; i++) {
                                if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                    export_val_domain(tt(1, 1)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                    export_val_domain(tt(1, 2)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                    export_val_domain(tt(1, 3)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                    export_val_domain(tt(2, 2)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                    export_val_domain(tt(2, 3)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                    export_val_domain(tt(3, 3)(dom_index), 0, bound, res, pos_res, ncond(i));
                            }
                        }
                        found = true;
                    }
                    if ((tt.get_basis().get_basis(dom_index) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                        if (n_cmp == -1) {
                            export_val_domain(tt(1, 1)(dom_index), 0, bound, res, pos_res, ncond(0));
                            export_val_domain(tt(1, 2)(dom_index), 0, bound, res, pos_res, ncond(1));
                            export_val_domain(tt(1, 3)(dom_index), 0, bound, res, pos_res, ncond(2));
                            export_val_domain(tt(2, 1)(dom_index), 0, bound, res, pos_res, ncond(3));
                            export_val_domain(tt(2, 2)(dom_index), 0, bound, res, pos_res, ncond(4));
                            export_val_domain(tt(2, 3)(dom_index), 0, bound, res, pos_res, ncond(5));
                            export_val_domain(tt(3, 1)(dom_index), 0, bound, res, pos_res, ncond(6));
                            export_val_domain(tt(3, 2)(dom_index), 0, bound, res, pos_res, ncond(7));
                            export_val_domain(tt(3, 3)(dom_index), 0, bound, res, pos_res, ncond(8));
                        } else {
                            for (int i = 0; i < n_cmp; i++) {
                                if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                    export_val_domain(tt(1, 1)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                    export_val_domain(tt(1, 2)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                    export_val_domain(tt(1, 3)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                    export_val_domain(tt(2, 1)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                    export_val_domain(tt(2, 2)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                    export_val_domain(tt(2, 3)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                    export_val_domain(tt(3, 1)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                    export_val_domain(tt(3, 2)(dom_index), 0, bound, res, pos_res, ncond(i));
                                if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                    export_val_domain(tt(3, 3)(dom_index), 0, bound, res, pos_res, ncond(i));
                            }
                        }
                        found = true;
                    }
                    if (!found)
                        KADATH_THROW("Unknown 2-tensor basis in Eq_matching adapted_nosym boundary export");
                } break;
                default:
                    KADATH_THROW("Valence not implemented in Eq_matching adapted_nosym boundary export");
            }
        }

        void export_boundary_for_matching(const Domain* domain, const Tensor& value, int domain_index, int bound,
                                          Array<double>& sec, int& pos_sec, const Array<int>& ncond, int n_cmp,
                                          Array<int>** p_cmp)
        {
            if (const auto* inner = dynamic_cast<const Domain_shell_inner_adapted_nosym*>(domain)) {
                export_tensor_boundary(value, domain_index, bound, sec, pos_sec, ncond, n_cmp, p_cmp,
                    [&](const Val_domain& val, int mlim, int val_bound, Array<double>& val_sec, int& val_pos,
                        int val_ncond) {
                        inner->export_tau_val_domain_boundary_matching(val, mlim, val_bound, val_sec, val_pos,
                                                                       val_ncond);
                    });
                return;
            }
            if (const auto* outer = dynamic_cast<const Domain_shell_outer_adapted_nosym*>(domain)) {
                export_tensor_boundary(value, domain_index, bound, sec, pos_sec, ncond, n_cmp, p_cmp,
                    [&](const Val_domain& val, int mlim, int val_bound, Array<double>& val_sec, int& val_pos,
                        int val_ncond) {
                        outer->export_tau_val_domain_boundary_matching(val, mlim, val_bound, val_sec, val_pos,
                                                                       val_ncond);
                    });
                return;
            }
            domain->export_tau_boundary(value, domain_index, bound, sec, pos_sec, ncond, n_cmp, p_cmp);
        }
    }

    Eq_matching::Eq_matching(const Domain* zedom, int dd, int bb, int other_dd, int other_bb, Ope_eq* lhs, Ope_eq* rhs,
                             int nused, Array<int>** pused)
        : Equation(zedom, dd, 2, nused, pused), bound(bb), other_dom(other_dd), other_bound(other_bb)
    {
        parts[0].reset(lhs);
        parts[1].reset(rhs);
    }

    Eq_matching::~Eq_matching() {}

    // No apply() override: Eq_matching inherits the base Equation::apply() (as at
    // 74e7f645), which materialises a Tensor copy of the evaluated residual and
    // counts via do_nbr_conditions(). The post-74e7f645 apply() override bypassed
    // that copy and drifted the single-star NS_nosym count out of square; removing
    // it restores the A.5 control flow. The adapted-nosym matching EXPORT
    // (export_val/export_der below) and do_nbr_conditions() remain overridden.

    void Eq_matching::export_val(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        assert(residus[conte + 1]->get_type_data() == TERM_T);

        const Tensor& local_value = *residus[conte]->get_p_val_t();
        const Tensor& other_value = *residus[conte + 1]->get_p_val_t();
        int start = pos_res;
        export_boundary_for_matching(dom, local_value, ndom, bound, sec, pos_res, *n_cond, n_cmp_used, p_cmp_used);
        Array<double> auxi(pos_res - start);
        auxi = 0.;
        int zero = 0;
        export_boundary_for_matching(other_value.get_space().get_domain(other_dom), other_value, other_dom, other_bound,
                                     auxi, zero, *n_cond, n_cmp_used, p_cmp_used);
        for (int i = start; i < pos_res; i++)
            sec.set(i) -= auxi(i - start);
        conte += 2;
    }

    void Eq_matching::export_der(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        assert(residus[conte]->get_type_data() == TERM_T);
        assert(residus[conte + 1]->get_type_data() == TERM_T);
        const Tensor& local_derivative = *residus[conte]->get_p_der_t();
        const Tensor& other_value = *residus[conte + 1]->get_p_val_t();
        const Tensor& other_derivative = *residus[conte + 1]->get_p_der_t();
        int start = pos_res;
        export_boundary_for_matching(dom, local_derivative, ndom, bound, sec, pos_res, *n_cond, n_cmp_used,
                                     p_cmp_used);
        Array<double> auxi(pos_res - start);
        auxi = 0.;
        int zero = 0;
        export_boundary_for_matching(other_value.get_space().get_domain(other_dom), other_derivative, other_dom,
                                     other_bound, auxi, zero, *n_cond, n_cmp_used, p_cmp_used);
        for (int i = start; i < pos_res; i++)
            sec.set(i) -= auxi(i - start);
        conte += 2;
    }

    void Eq_matching::export_der_lanes(int& conte, Term_eq** residus, int lane_count,
                                       Array<double>* const* secs, int* pos_res_arr) const
    {
        MatchingLaneStats& stats = matching_lane_stats_state();
        if (!matching_lane_export_enabled()) {
            ++stats.export_scalar_fallback_calls;
            Equation::export_der_lanes(conte, residus, lane_count, secs, pos_res_arr);
            return;
        }

        assert(residus[conte]->get_type_data() == TERM_T);
        assert(residus[conte + 1]->get_type_data() == TERM_T);
        Term_eq* local_result = residus[conte];
        Term_eq* other_result = residus[conte + 1];
        Array<double> other_rows(n_cond_tot);

        for (int lane = 0; lane < lane_count; ++lane) {
            Array<double>& sec = *secs[lane];
            const int start = pos_res_arr[lane];
            const Tensor* local = local_result->get_p_der_t(lane);
            const Tensor* other = other_result->get_p_der_t(lane);

            if (local != nullptr) {
                export_boundary_for_matching(dom, *local, ndom, bound, sec, pos_res_arr[lane], *n_cond,
                                             n_cmp_used, p_cmp_used);
            } else {
                for (int row = 0; row < n_cond_tot; ++row)
                    sec.set(start + row) = 0.0;
                pos_res_arr[lane] += n_cond_tot;
                ++stats.export_missing_lanes;
            }

            if (other == nullptr) {
                ++stats.export_missing_lanes;
                continue;
            }

            other_rows = 0.0;
            int other_pos = 0;
            export_boundary_for_matching(other->get_space().get_domain(other_dom), *other, other_dom,
                                         other_bound, other_rows, other_pos, *n_cond, n_cmp_used, p_cmp_used);
            assert(other_pos == n_cond_tot);
            for (int row = 0; row < n_cond_tot; ++row)
                sec.set(start + row) -= other_rows(row);
        }

        conte += 2;
        ++stats.export_native_calls;
    }

    bool Eq_matching::describe_residual_rows(
        int& conte, Term_eq** residuals, int equation_index,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        if (!called || residuals == nullptr || residuals[conte] == nullptr ||
            residuals[conte + 1] == nullptr ||
            residuals[conte]->get_type_data() != TERM_T ||
            residuals[conte + 1]->get_type_data() != TERM_T) {
            conte += 2;
            return false;
        }

        const Tensor& local = *residuals[conte]->get_p_val_t();
        const Tensor& remote = *residuals[conte + 1]->get_p_val_t();
        conte += 2;

        std::vector<ResidualRowDescriptor> local_rows;
        std::vector<ResidualRowDescriptor> remote_rows;
        const Domain* remote_domain = remote.get_space().get_domain(other_dom);
        if (!dom->describe_boundary_residual_rows(
                local, ndom, bound, *n_cond, n_cmp_used, p_cmp_used,
                local_rows) ||
            !remote_domain->describe_boundary_residual_rows(
                remote, other_dom, other_bound, *n_cond, n_cmp_used,
                p_cmp_used, remote_rows) ||
            local_rows.size() != static_cast<std::size_t>(n_cond_tot) ||
            remote_rows.size() != local_rows.size()) {
            return false;
        }

        descriptors.reserve(local_rows.size());
        for (std::size_t row = 0; row < local_rows.size(); ++row) {
            const ResidualRowDescriptor& local_descriptor = local_rows[row];
            const ResidualRowDescriptor& remote_descriptor = remote_rows[row];
            if (local_descriptor.family != ResidualRowEquationFamily::Unavailable ||
                remote_descriptor.family != ResidualRowEquationFamily::Unavailable ||
                local_descriptor.available || remote_descriptor.available ||
                local_descriptor.equation_index != -1 ||
                remote_descriptor.equation_index != -1 ||
                local_descriptor.explicit_sector != 0 ||
                remote_descriptor.explicit_sector != 0 ||
                local_descriptor.sides.size() != 1 ||
                remote_descriptor.sides.size() != 1) {
                descriptors.clear();
                return false;
            }
            const ResidualRowCoordinate& local_coordinate =
                local_descriptor.sides.front();
            const ResidualRowCoordinate& remote_coordinate =
                remote_descriptor.sides.front();
            if (local_coordinate.domain != ndom ||
                remote_coordinate.domain != other_dom ||
                local_coordinate.component < 0 ||
                local_coordinate.component >= local.get_n_comp() ||
                remote_coordinate.component < 0 ||
                remote_coordinate.component >= remote.get_n_comp() ||
                local_coordinate.component != remote_coordinate.component ||
                local_coordinate.phi_index < 0 ||
                remote_coordinate.phi_index < 0 ||
                local_coordinate.phi_basis == 0 ||
                remote_coordinate.phi_basis == 0 ||
                local_coordinate.phi_index >= dom->get_nbr_coefs()(2) ||
                remote_coordinate.phi_index >=
                    remote_domain->get_nbr_coefs()(2)) {
                descriptors.clear();
                return false;
            }

            ResidualRowDescriptor paired;
            paired.family = ResidualRowEquationFamily::Field;
            paired.equation_index = equation_index;
            paired.available = true;
            paired.sides.push_back(local_coordinate);
            paired.sides.push_back(remote_coordinate);
            descriptors.push_back(std::move(paired));
        }
        return true;
    }

    Array<int> Eq_matching::do_nbr_conditions(const Tensor& tt) const
    {
        return dom->nbr_conditions_boundary(tt, ndom, bound, n_cmp_used, p_cmp_used);
    }

    bool Eq_matching::take_into_account(int target) const
    {
        if ((target == ndom) || (target == other_dom))
            return true;
        else
            return false;
    }

} // namespace Kadath
