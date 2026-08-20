// Packed W-lane verification oracle for the Jacobian column engine.
//
// Split out of jacobian_column_engine.cpp: compute_packed_wlaneN_columns_for_verification<W>
// and its W={2,4,8} wrappers produce DENSE reference columns (no drop tolerance)
// that the production compute_packed_wlaneN_columns_sparse<W> path is checked
// against. Called only from System_of_eqs::validate_packed_wlane*_columns (see
// solver_diagnostics.cpp) and the unit tests -- never on the Newton hot path.
//
// The lane-seeding helper below is intentionally a verbatim copy of the
// production helper in jacobian_column_engine.cpp: an independent oracle must
// not share code with the path it verifies.
#include "For_Kadath/System_of_eqs/jacobian_column_engine.hpp"

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "mpi.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <unordered_map>

namespace Kadath {

namespace {

void set_primary_zero_and_release_extra_derivative_lanes(Term_eq* term,
                                                         int lane_count)
{
    if (term == nullptr)
        return;
    term->reset_derivative_tile(lane_count);
}

void set_primary_zero_and_release_extra_derivative_lanes(const std::unique_ptr<Term_eq>& term,
                                                         int lane_count)
{
    set_primary_zero_and_release_extra_derivative_lanes(term.get(), lane_count);
}

} // namespace

template <int W>
bool JacobianColumnEngine::compute_packed_wlaneN_columns_for_verification(
    PackedJacobianColumns<W> columns,
    PackedJacobianOutputs<W> outputs,
    std::string& failure_reason)
{
    static_assert(W == 2 || W == 4 || W == 8 || W == 16 || W == 32,
                  "packed W-lane verification path supports only W in {2,4,8,16,32}");
    failure_reason.clear();
    if (sys_.nbr_conditions == -1) {
        KADATH_THROW("Number of conditions unknown ; call sec_member first");
    }
    for (int k = 0; k < W; ++k) {
        if (columns[k] < 0 || columns[k] >= sys_.nbr_unknowns) {
            failure_reason = "selected column is outside the system unknown range";
            return false;
        }
    }
    for (int a = 0; a < W; ++a) {
        for (int b = a + 1; b < W; ++b) {
            if (columns[a] == columns[b]) {
                failure_reason = "packed W-lane verification requires distinct columns";
                return false;
            }
        }
    }
    const auto expected_output_size = static_cast<std::size_t>(sys_.nbr_conditions);
    for (int k = 0; k < W; ++k) {
        if (outputs[k] == nullptr || outputs[k]->get_nbr() != expected_output_size) {
            failure_reason = "packed W-lane verification output buffers do not match nbr_conditions";
            return false;
        }
    }

    std::vector<ColumnInfo> column_map;
    sys_.build_column_map(column_map);
    for (int k = 0; k < W; ++k) {
        if (column_map[static_cast<std::size_t>(columns[k])].is_var_domain) {
            failure_reason = "variable-domain columns are not supported by the W-lane verification path";
            return false;
        }
    }

    // Prepare/restore mirror the production path's logical derivative state:
    // widen ALL five derivative-bearing categories (cst / term / term_double /
    // results / def) to W lanes and zero them, then return to scalar state. This
    // independent oracle deliberately keeps the destructive reset: production
    // retains dormant capacity, while verification releases it, so a stale
    // occupancy bug cannot be shared by both paths and escape comparison.
    auto prepare_lane_derivatives = [&]() {
        for (int i = 0; i < sys_.nterm_cst; ++i)
            set_primary_zero_and_release_extra_derivative_lanes(sys_.cst[i], W);
        for (int i = 0; i < sys_.nterm; ++i)
            set_primary_zero_and_release_extra_derivative_lanes(sys_.term[i], W);
        for (int i = 0; i < sys_.nterm_double; ++i)
            set_primary_zero_and_release_extra_derivative_lanes(sys_.term_double[i], W);
        for (auto& result : sys_.results)
            if (result)
                set_primary_zero_and_release_extra_derivative_lanes(result.get(), W);
        for (int i = 0; i < sys_.ndef; ++i)
            set_primary_zero_and_release_extra_derivative_lanes(sys_.def[i]->get_res(), W);
    };

    auto restore_scalar_derivative_lanes = [&]() {
        for (int i = 0; i < sys_.nterm_cst; ++i)
            set_primary_zero_and_release_extra_derivative_lanes(sys_.cst[i], 1);
        for (int i = 0; i < sys_.nterm; ++i)
            set_primary_zero_and_release_extra_derivative_lanes(sys_.term[i], 1);
        for (int i = 0; i < sys_.nterm_double; ++i)
            set_primary_zero_and_release_extra_derivative_lanes(sys_.term_double[i], 1);
        for (auto& result : sys_.results)
            if (result)
                set_primary_zero_and_release_extra_derivative_lanes(result.get(), 1);
        for (int i = 0; i < sys_.ndef; ++i)
            set_primary_zero_and_release_extra_derivative_lanes(sys_.def[i]->get_res(), 1);
    };

    auto seed_column_into_lane = [&](int column, int lane, bool& has_scalar_column) -> bool {
        int column_counter = 0;
        Array<int> affected_variable_domains(2);
        affected_variable_domains = -1;
        sys_.espace.affecte_coef_to_variable_domains(column_counter, column, affected_variable_domains);
        if (affected_variable_domains(0) != -1 || affected_variable_domains(1) != -1) {
            failure_reason = "variable-domain derivative appeared during W-lane verification seeding";
            return false;
        }

        for (int variable_index = 0; variable_index < sys_.nvar_double; ++variable_index) {
            if (column_counter == column) {
                for (int domain = sys_.dom_min; domain <= sys_.dom_max; ++domain) {
                    sys_.term_double[static_cast<std::size_t>(
                        variable_index * sys_.ndom + (domain - sys_.dom_min))]->set_der_d(lane, 1.0);
                }
                has_scalar_column = true;
            }
            ++column_counter;
        }

        for (int term_index = 0; term_index < sys_.nterm; ++term_index) {
            const int domain = sys_.term[term_index]->get_dom();
            const Tensor& term_value = *sys_.term[term_index]->get_p_val_t();
            Tensor derivative_seed(term_value, false);
            sys_.espace.get_domain(domain)->affecte_tau_one_coef(
                derivative_seed, domain, column, column_counter);

            bool seed_is_nonzero = false;
            for (int component = 0; component < derivative_seed.get_n_comp(); ++component) {
                const Array<int> index = derivative_seed.indices(component);
                if (!derivative_seed(index)(domain).check_if_zero()) {
                    seed_is_nonzero = true;
                    if (!derivative_seed(index)(domain).get_base().is_def()) {
                        derivative_seed.set(index).set_domain(domain).set_base() =
                            term_value(index)(domain).get_base();
                    }
                }
            }
            if (seed_is_nonzero) {
                sys_.term[term_index]->set_der_t(lane, derivative_seed);
            }
        }
        return true;
    };

    auto export_all_equations_packed = [&]() {
        std::array<int, W> positions{};
        for (int k = 0; k < W; ++k)
            *outputs[k] = 0.0;
        for (int i = 0; i < sys_.neq_int; ++i) {
            for (int k = 0; k < W; ++k) {
                outputs[k]->set(positions[k]) = sys_.eq_int[i]->get_der(k);
                ++positions[k];
            }
        }
        Term_eq** results_raw = sys_.results_shadow_view();
        std::array<Array<double>*, W> secs_arr;
        for (int k = 0; k < W; ++k)
            secs_arr[k] = outputs[k];
        int conte = 0;
        for (int equation_index = 0; equation_index < sys_.neq; ++equation_index) {
            sys_.eq[equation_index]->export_der_lanes(
                conte, results_raw, W, secs_arr.data(), positions.data());
        }
    };

    try {
        prepare_lane_derivatives();

        bool has_scalar_column = false;
        for (int k = 0; k < W; ++k) {
            if (!seed_column_into_lane(columns[k], k, has_scalar_column)) {
                restore_scalar_derivative_lanes();
                return false;
            }
        }

        if (sys_.met != nullptr) {
            for (int domain = sys_.dom_min; domain <= sys_.dom_max; ++domain)
                sys_.met->update(domain);
        }
        for (int i = 0; i < sys_.ndef; ++i)
            sys_.def[i]->compute_res();

        int operator_offset = 0;
        Term_eq** results_raw = sys_.results_shadow_view();
        for (int equation_index = 0; equation_index < sys_.neq; ++equation_index)
            sys_.eq[equation_index]->apply(operator_offset, results_raw);
        sys_.results_shadow_sync();

        export_all_equations_packed();

        restore_scalar_derivative_lanes();
        (void)has_scalar_column;
        return true;
    } catch (...) {
        restore_scalar_derivative_lanes();
        throw;
    }
}

bool JacobianColumnEngine::compute_packed_wlane2_columns_for_verification(int first_column,
                                                                          int second_column,
                                                                          Array<double>& first_output,
                                                                          Array<double>& second_output,
                                                                          std::string& failure_reason)
{
    const std::array<int, 2> columns{ first_column, second_column };
    std::array<Array<double>*, 2> outputs{ &first_output, &second_output };
    return compute_packed_wlaneN_columns_for_verification<2>(columns, outputs, failure_reason);
}

bool JacobianColumnEngine::compute_packed_wlane4_columns_for_verification(
    const std::array<int, 4>& columns,
    std::array<Array<double>*, 4>& outputs,
    std::string& failure_reason)
{
    return compute_packed_wlaneN_columns_for_verification<4>(columns, outputs, failure_reason);
}

bool JacobianColumnEngine::compute_packed_wlane8_columns_for_verification(
    const std::array<int, 8>& columns,
    std::array<Array<double>*, 8>& outputs,
    std::string& failure_reason)
{
    return compute_packed_wlaneN_columns_for_verification<8>(columns, outputs, failure_reason);
}

bool JacobianColumnEngine::compute_packed_wlane16_columns_for_verification(
    const std::array<int, 16>& columns,
    std::array<Array<double>*, 16>& outputs,
    std::string& failure_reason)
{
    return compute_packed_wlaneN_columns_for_verification<16>(columns, outputs, failure_reason);
}

bool JacobianColumnEngine::compute_packed_wlane32_columns_for_verification(
    const std::array<int, 32>& columns,
    std::array<Array<double>*, 32>& outputs,
    std::string& failure_reason)
{
    return compute_packed_wlaneN_columns_for_verification<32>(columns, outputs, failure_reason);
}

} // namespace Kadath
