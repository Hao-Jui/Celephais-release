/*
    Added 2026 Hao-Jui Kuan
    Newton step via exact matrix-free Jv and sparse MUMPS right preconditioner.
*/

#include "mpi.h"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Diagnostics/kernel_profile.hpp"
#include "For_Kadath/Diagnostics/matching_lane_profile.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"

#include "newton_norms.hpp"

#include "Linear_algebra/jacobian_assembler.hpp"
#include "Linear_algebra/jacobian_parity_mask.hpp"
#include "Linear_algebra/krylov_solver.hpp"

#ifdef CELEPHAIS_USE_MUMPS
#include "Linear_algebra/mumps_tree_cache.hpp"
#include "Linear_algebra/sparse_direct_mumps_solve.hpp"
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <system_error>
#include <utility>
#include <vector>


namespace Kadath
{
    extern "C"
    {
    void dump_ope_action_profile();
    void reset_ope_action_profile();
    }

    namespace
    {
#ifdef CELEPHAIS_USE_MUMPS
        double elapsed_time(std::chrono::time_point<std::chrono::steady_clock> const& begin)
        {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        }

        void report_timing(int rank, int nproc, const char* label, double seconds)
        {
            double seconds_max = 0.0;
            double seconds_min = 0.0;
            double seconds_sum = 0.0;
            MPI_Reduce(&seconds, &seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&seconds, &seconds_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&seconds, &seconds_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                // Headline the MAX -- the wall-clock cost of the phase (the run waits
                // for the slowest rank). For sub-communicator work (e.g. the MUMPS
                // factor on a few ranks of many) the average is diluted by the idle
                // ranks' near-zero times and badly understates the real cost, so it is
                // demoted to a parenthetical.
                std::cout << label << ": " << seconds_max << " s (max; avg "
                          << seconds_sum / nproc << ", min " << seconds_min << " over "
                          << nproc << " ranks)" << std::endl;
            }
        }

        template<std::size_t N>
        void report_timing_batch(
            int rank,
            int nproc,
            const std::array<const char*, N>& labels,
            const std::array<double, N>& seconds)
        {
            std::array<double, N> seconds_max{};
            std::array<double, N> seconds_min{};
            std::array<double, N> seconds_sum{};
            MPI_Reduce(seconds.data(), seconds_max.data(), static_cast<int>(N),
                       MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(seconds.data(), seconds_min.data(), static_cast<int>(N),
                       MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(seconds.data(), seconds_sum.data(), static_cast<int>(N),
                       MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) {
                for (std::size_t i = 0; i < N; ++i) {
                    std::cout << labels[i] << ": " << seconds_max[i]
                              << " s (max; avg " << seconds_sum[i] / nproc
                              << ", min " << seconds_min[i] << " over "
                              << nproc << " ranks)" << std::endl;
                }
            }
        }

        double global_max_seconds(double local_seconds)
        {
            double maximum = 0.0;
            MPI_Allreduce(&local_seconds, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            return maximum;
        }

        template <typename Operation>
        std::string collective_phase_error(
            MPI_Comm communicator, const char* phase, Operation&& operation)
        {
            int rank = 0;
            int size = 1;
            MPI_Comm_rank(communicator, &rank);
            MPI_Comm_size(communicator, &size);

            std::string local_error;
            try {
                operation();
            } catch (const std::exception& error) {
                local_error = error.what();
            } catch (...) {
                local_error = "unknown non-standard exception";
            }

            const int local_failure_rank = local_error.empty() ? size : rank;
            int failure_rank = size;
            MPI_Allreduce(&local_failure_rank, &failure_rank, 1, MPI_INT, MPI_MIN,
                          communicator);
            if (failure_rank == size)
                return {};

            int message_size = rank == failure_rank
                                   ? static_cast<int>(std::min<std::size_t>(
                                         local_error.size(),
                                         static_cast<std::size_t>(
                                             std::numeric_limits<int>::max())))
                                   : 0;
            MPI_Bcast(&message_size, 1, MPI_INT, failure_rank, communicator);
            std::string message(static_cast<std::size_t>(message_size), '\0');
            if (rank == failure_rank)
                message.assign(local_error.data(), static_cast<std::size_t>(message_size));
            if (message_size > 0)
                MPI_Bcast(message.data(), message_size, MPI_CHAR, failure_rank,
                          communicator);
            return std::string(phase) + " failed collectively: " + message;
        }

        template <typename Operation>
        void run_collective_phase(
            MPI_Comm communicator, const char* phase, Operation&& operation)
        {
            const std::string error = collective_phase_error(
                communicator, phase, std::forward<Operation>(operation));
            if (!error.empty())
                throw LinearSolverError(__FILE__, __LINE__, error);
        }

        void broadcast_int_vector(std::vector<int>& values, int root, MPI_Comm communicator)
        {
            int rank = 0;
            MPI_Comm_rank(communicator, &rank);
            int count = rank == root ? static_cast<int>(values.size()) : 0;
            MPI_Bcast(&count, 1, MPI_INT, root, communicator);
            if (rank != root)
                values.resize(static_cast<std::size_t>(count));
            if (count > 0)
                MPI_Bcast(values.data(), count, MPI_INT, root, communicator);
        }

        bool cache_path_has_entry(const std::filesystem::path& path)
        {
            std::error_code error;
            const std::filesystem::file_status status =
                std::filesystem::symlink_status(path, error);
            if (error == std::errc::no_such_file_or_directory ||
                status.type() == std::filesystem::file_type::not_found)
                return false;
            if (error)
                throw std::filesystem::filesystem_error(
                    "cannot inspect MUMPS tree cache", path, error);
            return true;
        }

        void remove_tree_cache_with_log(
            const std::filesystem::path& path, const char* context)
        {
            try {
                (void)remove_mumps_tree_cache(path);
            } catch (const std::exception& error) {
                std::cerr << context << ": could not delete MUMPS tree cache "
                          << path.string() << ": " << error.what() << '\n';
            }
        }

        void require_collective_tree_cache_config_agreement(
            bool enabled, const std::string& path)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            int root_enabled = rank == 0 && enabled ? 1 : 0;
            MPI_Bcast(&root_enabled, 1, MPI_INT, 0, MPI_COMM_WORLD);

            unsigned long long root_path_size =
                rank == 0 && root_enabled != 0
                    ? static_cast<unsigned long long>(path.size())
                    : 0;
            MPI_Bcast(&root_path_size, 1, MPI_UNSIGNED_LONG_LONG, 0,
                      MPI_COMM_WORLD);
            if (root_path_size >
                static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
                throw LinearSolverError(
                    __FILE__, __LINE__,
                    "MUMPS tree cache path exceeds the MPI broadcast limit");
            }

            std::string root_path(static_cast<std::size_t>(root_path_size), '\0');
            if (rank == 0)
                root_path = path;
            if (root_path_size > 0) {
                MPI_Bcast(root_path.data(), static_cast<int>(root_path_size),
                          MPI_CHAR, 0, MPI_COMM_WORLD);
            }

            const int local_mismatch =
                enabled != (root_enabled != 0) ||
                        (root_enabled != 0 && path != root_path)
                    ? 1
                    : 0;
            int any_mismatch = 0;
            MPI_Allreduce(&local_mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD);
            if (any_mismatch != 0) {
                throw LinearSolverError(
                    __FILE__, __LINE__,
                    "MUMPS_TREE_CACHE and its configured path must "
                    "match exactly on every MPI rank");
            }
        }

        void require_collective_adaptive_config_agreement(
            bool adaptive_refresh,
            int refresh_steps,
            int adaptive_max_steps,
            int max_linear_iterations)
        {
            const std::array<int, 4> local_values = {
                adaptive_refresh ? 1 : 0,
                refresh_steps,
                adaptive_max_steps,
                max_linear_iterations};
            std::array<int, 4> minimum_values{};
            std::array<int, 4> maximum_values{};
            MPI_Allreduce(local_values.data(), minimum_values.data(),
                          static_cast<int>(local_values.size()), MPI_INT,
                          MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(local_values.data(), maximum_values.data(),
                          static_cast<int>(local_values.size()), MPI_INT,
                          MPI_MAX, MPI_COMM_WORLD);
            if (minimum_values != maximum_values) {
                KADATH_THROW(
                    "JFNK-MUMPS collective configuration differs across ranks "
                    "(adaptive refresh, refresh cadence, adaptive maximum, or "
                    "GMRES iteration limit)");
            }
        }

        bool exact_jv_der_abs_cache_enabled()
        {
            return env_flag_enabled("DO_JX_DER_ABS_CACHE", false);
        }

        // Eisenstat-Walker forcing-term controller (choice 2 from Eisenstat &
        // Walker 1996), used to pick the GMRES relative tolerance for the
        // next linear solve. With prev residual ||F_{k-1}|| and current
        // residual ||F_k||, choice 2 selects
        //
        //   eta_candidate = gamma * (||F_k|| / ||F_{k-1}||)^alpha
        //
        // Safeguard against the controller pushing too soft when the previous
        // eta was already large:
        //
        //   eta_candidate = max(eta_candidate, gamma * (eta_{k-1})^alpha)
        //
        // Final tolerance is clamped into [eta_min, eta_max]. eta_min is the
        // baseline `linear_relative_tolerance` so we never solve looser than
        // the user's hard floor.
        struct EisenstatWalkerParameters {
            double gamma = 0.9;
            double alpha = 2.0;
            double eta_max = 1e-3;
            double eta_min = 1e-8;
        };

        EisenstatWalkerParameters read_eisenstat_walker_parameters(double eta_min_default)
        {
            EisenstatWalkerParameters params;
            params.gamma = env_double_value("JFNK_EW_GAMMA", 0.9);
            params.alpha = env_double_value("JFNK_EW_ALPHA", 2.0);
            params.eta_max = env_double_value("JFNK_EW_MAX", 1e-3);
            params.eta_min = env_double_value("JFNK_EW_MIN", eta_min_default);
            params.gamma = std::clamp(params.gamma, 0.0, 1.0);
            params.alpha = std::clamp(params.alpha, 1.0, 4.0);
            params.eta_max = std::clamp(params.eta_max,
                                        std::max(params.eta_min, 1e-12), 0.999);
            params.eta_min = std::clamp(params.eta_min, 1e-16, params.eta_max);
            return params;
        }

        double next_eisenstat_walker_eta(const EisenstatWalkerParameters& params,
                                         double prev_residual,
                                         double prev_eta,
                                         double current_residual)
        {
            // First Newton step or after a regime reset: start from the
            // upper bound so the linear solve is loose and cheap.
            if (prev_residual <= 0.0 || current_residual <= 0.0)
                return params.eta_max;

            const double ratio = current_residual / prev_residual;
            double eta = params.gamma * std::pow(ratio, params.alpha);

            // Safeguard from Eisenstat-Walker 1996: avoid volatile drops by
            // refusing to fall below a (gamma * prev_eta^alpha) floor when
            // that floor is itself already large.
            if (prev_eta > 0.0) {
                const double prev_floor = params.gamma * std::pow(prev_eta, params.alpha);
                if (prev_floor > 0.1)
                    eta = std::max(eta, prev_floor);
            }

            return std::clamp(eta, params.eta_min, params.eta_max);
        }

        double euclidean_norm(const std::vector<double>& values)
        {
            double norm_sq = 0.0;
            for (double value : values) {
                norm_sq += value * value;
            }
            return std::sqrt(norm_sq);
        }

        std::vector<double> vector_from_array(const Array<double>& values, int expected_size)
        {
            std::vector<double> result(static_cast<std::size_t>(expected_size), 0.0);
            for (int i = 0; i < expected_size; ++i) {
                result[static_cast<std::size_t>(i)] = values(i);
            }
            return result;
        }

        void copy_vector_to_array(const std::vector<double>& values, Array<double>& result)
        {
            for (int i = 0; i < static_cast<int>(values.size()); ++i) {
                result.set(i) = values[static_cast<std::size_t>(i)];
            }
        }

        // Bit-exact comparison of two residual vectors for the line-search
        // snapshot/restore self-test. "Exact" means every element compares
        // equal under ==; max_abs_diff/mismatches characterise any drift.
        struct ArrayDiff {
            bool exact = true;
            double max_abs_diff = 0.0;
            long mismatches = 0;
            long size_a = 0;
            long size_b = 0;
        };

        ArrayDiff compare_arrays_bitwise(const Array<double>& a, const Array<double>& b)
        {
            ArrayDiff diff;
            diff.size_a = static_cast<long>(a.get_nbr());
            diff.size_b = static_cast<long>(b.get_nbr());
            if (diff.size_a != diff.size_b) {
                diff.exact = false;
                return diff;
            }
            for (long i = 0; i < diff.size_a; ++i) {
                const double va = a(static_cast<int>(i));
                const double vb = b(static_cast<int>(i));
                if (va != vb) {
                    ++diff.mismatches;
                    const double d = std::fabs(va - vb);
                    if (d > diff.max_abs_diff)
                        diff.max_abs_diff = d;
                }
            }
            diff.exact = (diff.mismatches == 0);
            return diff;
        }

        void report_selftest(std::ostream& os, const char* label, const ArrayDiff& diff)
        {
            os << "  [self-test] " << label << ": ";
            if (diff.size_a != diff.size_b) {
                os << "SIZE MISMATCH (" << diff.size_a << " vs " << diff.size_b << ")\n";
                return;
            }
            os << (diff.exact ? "exact" : "DIFFERS") << " (n=" << diff.size_a
               << ", mismatches=" << diff.mismatches << ", max_abs_diff=" << std::setprecision(3)
               << std::scientific << diff.max_abs_diff << std::defaultfloat << ")\n";
        }

        const char* gmres_status_name(GmresStatus::Code code)
        {
            switch (code) {
            case GmresStatus::Code::Converged:
                return "converged";
            case GmresStatus::Code::MaxIterations:
                return "max_iterations";
            case GmresStatus::Code::InvalidInput:
                return "invalid_input";
            case GmresStatus::Code::Breakdown:
                return "breakdown";
            }
            return "unknown";
        }
#endif
    } // namespace

    bool System_of_eqs::do_newton_jfnk_mumps(double precision,
                                             double& error,
                                             const SolverRuntimeConfig& config)
    {
        set_solver_runtime_config(config);
#ifdef CELEPHAIS_USE_MUMPS
        int rank = 0;
        int nproc = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        const int column_count = nbr_unknowns;
        auto& mumps_state = mumps_runtime_state;
        const bool tree_cache_configured =
            config.mumps.tree_cache_enabled &&
            !config.mumps.tree_cache_path.empty();
        const std::filesystem::path tree_cache_path(config.mumps.tree_cache_path);

        require_collective_tree_cache_config_agreement(
            config.mumps.tree_cache_enabled, config.mumps.tree_cache_path);

        // Validate a surviving cache before even an already-converged return.
        // The root performs all filesystem access; every rank receives the
        // same disposition before entering the residual/Newton collectives.
        if (!mumps_state.jfnk_tree_cache_stage_entry_checked) {
            run_collective_phase(
                MPI_COMM_WORLD, "MUMPS tree cache stage registration", [&]() {
                    if (rank != 0)
                        return;
                    if (tree_cache_configured)
                        set_active_mumps_tree_cache_path(tree_cache_path);
                    else
                        clear_active_mumps_tree_cache_path();
                });
            int cache_disposition = 0; // 0=disabled/missing, 1=valid, 2=rejected
            if (tree_cache_configured && rank == 0) {
                try {
                    if (cache_path_has_entry(tree_cache_path)) {
                        (void)read_mumps_tree_cache(tree_cache_path, column_count);
                        cache_disposition = 1;
                    }
                } catch (const std::exception& cache_error) {
                    cache_disposition = 2;
                    std::cerr << "JFNK-MUMPS tree cache stage-entry rejection: "
                              << cache_error.what() << '\n';
                    remove_tree_cache_with_log(
                        tree_cache_path, "JFNK-MUMPS tree cache stage entry");
                }
            }
            MPI_Bcast(&cache_disposition, 1, MPI_INT, 0, MPI_COMM_WORLD);
            mumps_state.jfnk_tree_cache_stage_entry_checked = true;
        }

        // All-rank collective residual: reuses the do_JX MPI row partition
        // once built (replicated until then / on fallback). When the previous
        // iteration's update_fields refresh forwarded its residual and the
        // unknowns are untouched since (the slot is cleared by every state
        // mutation), consume it instead of recomputing — one full evaluation
        // saved per Newton step. The shared consumer also performs the opt-in
        // collective byte-for-byte self-test before accepting a candidate.
        Array<double> residual =
            take_forwarded_residual_or_compute("update_fields");
        error = infinity_norm(residual);
        const double nonlinear_error_before_step = error;
        print_error_init_diagnostic(residual, error);

        if (error < precision) {
            return true;
        }

        const int row_count = static_cast<int>(residual.get_nbr());
        if (row_count != column_count) {
            if (rank == 0) {
                std::cerr << "do_newton_jfnk_mumps: non-square system m=" << row_count
                          << " n=" << column_count << "; falling back to do_newton()." << std::endl;
            }
            return do_newton(precision, error, System_of_eqs::SOLVER::NEWTON_RAPHSON);
        }

        const bool timing_enabled = config.diagnostics.timing;
        // The adaptive-recovery diagnostic measures the discarded outer attempt.
        const auto step_start = std::chrono::steady_clock::now();

        // A symmetry-certified Newton step lives entirely in the + invariant
        // subspace. Build the same structural plan used by sparse-direct
        // Newton before the first PC Jacobian so MUMPS never materializes the
        // inactive block. The full entry residual remains the fail-closed
        // oracle for F_- and for later symmetry drift.
        const char* local_parity_mass_path =
            std::getenv("JACOBIAN_PARITY_MASS");
        const int local_parity_mass_probe_requested =
            local_parity_mass_path != nullptr &&
                local_parity_mass_path[0] != '\0' &&
                std::string(local_parity_mass_path) != "0"
            ? 1
            : 0;
        int parity_mass_probe_requested_any = 0;
        MPI_Allreduce(
            &local_parity_mass_probe_requested,
            &parity_mass_probe_requested_any, 1, MPI_INT, MPI_MAX,
            MPI_COMM_WORLD);
        const bool parity_mass_probe_requested =
            parity_mass_probe_requested_any != 0;
        const int local_pre_j1_requested =
            config.sparse_sector_reduce && !jacobian_parity_mask_state() &&
                !parity_mass_probe_requested
            ? 1
            : 0;
        int pre_j1_requested_all = 0;
        MPI_Allreduce(&local_pre_j1_requested, &pre_j1_requested_all, 1,
                      MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (pre_j1_requested_all != 0) {
            JacobianParityRowPrediction row_prediction =
                predict_jacobian_parity_rows(*this);
            JacobianParityColumnGrading column_grading =
                grade_jacobian_parity_columns(*this);
            JacobianPreJ1SelectionPlanBuild pre_j1 =
                make_jacobian_pre_j1_selection_plan(
                    row_prediction, column_grading, residual.get_data(),
                    row_count);

            const int local_eligible = pre_j1 ? 1 : 0;
            int eligible_all = 0;
            MPI_Allreduce(&local_eligible, &eligible_all, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD);
            if (eligible_all != 0) {
                auto state = std::make_shared<JacobianParityMaskState>();
                state->n = column_count;
                state->column_sector = std::move(column_grading.sector);
                state->row_sector = std::move(row_prediction.sector);
                state->selection_plan = std::move(pre_j1.plan);
                state->decision =
                    JacobianParityMaskState::Decision::Engaged;
                state->structural_labels_available = true;
                state->row_grading_source_label = "structural";
                state->reduction_decision =
                    JacobianParityMaskState::ReductionDecision::Eligible;
                jacobian_parity_mask_state() = std::move(state);
            }
        } else if (config.sparse_sector_reduce &&
                   !jacobian_parity_mask_state() &&
                   parity_mass_probe_requested && rank == 0) {
            std::cout
                << "JFNK sector reduction: pre-J1 diagnostic full-J "
                   "cross-check requested by JACOBIAN_PARITY_MASS\n";
        }

        std::shared_ptr<const JacobianSelectionPlan> step_selection_plan;
        if (config.sparse_sector_reduce) {
            const std::shared_ptr<JacobianParityMaskState>& parity_state =
                jacobian_parity_mask_state();
            if (parity_state && parity_state->n == column_count &&
                parity_state->reduction_decision ==
                    JacobianParityMaskState::ReductionDecision::Eligible) {
                step_selection_plan = parity_state->selection_plan;
            }
        }

        if (step_selection_plan) {
            const JacobianSelectionNorms residual_norms =
                measure_jacobian_selection_norms(
                    std::span<const double>{
                        residual.get_data(), residual.get_nbr()},
                    step_selection_plan->selected_rows());
            std::shared_ptr<JacobianParityMaskState>& parity_state =
                jacobian_parity_mask_state();
            JacobianForbiddenResidualCheck forbidden_check;
            if (!parity_state->forbidden_baseline_installed) {
                forbidden_check.limit = std::max(
                    jacobian_pre_j1_forbidden_relative_tolerance *
                        residual_norms.active_linf,
                    jacobian_pre_j1_forbidden_absolute_floor);
                forbidden_check.allowed = residual_norms &&
                    std::isfinite(residual_norms.active_linf) &&
                    std::isfinite(residual_norms.forbidden_linf) &&
                    residual_norms.active_linf >= 0.0 &&
                    residual_norms.forbidden_linf >= 0.0 &&
                    residual_norms.forbidden_linf <= forbidden_check.limit;
                if (forbidden_check.allowed) {
                    parity_state->forbidden_baseline =
                        residual_norms.forbidden_linf;
                    parity_state->forbidden_baseline_installed = true;
                }
            } else {
                forbidden_check = check_jacobian_forbidden_residual(
                    residual_norms, parity_state->forbidden_baseline,
                    parity_state->forbidden_baseline_installed);
            }

            std::vector<double> inactive_state;
            std::string inactive_state_failure;
            const bool local_state_valid = read_inactive_jacobian_state(
                *step_selection_plan, inactive_state,
                inactive_state_failure);
            double inactive_state_drift =
                local_state_valid
                    ? install_or_measure_jacobian_inactive_state_drift(
                          inactive_state,
                          parity_state->inactive_state_baseline,
                          parity_state->inactive_state_baseline_installed)
                    : std::numeric_limits<double>::infinity();
            const int local_guards_valid =
                forbidden_check.allowed && local_state_valid &&
                    jacobian_inactive_state_drift_allowed(
                        inactive_state_drift)
                ? 1
                : 0;
            int guards_valid_all = 0;
            double inactive_state_drift_max = inactive_state_drift;
            MPI_Allreduce(&local_guards_valid, &guards_valid_all, 1, MPI_INT,
                          MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&inactive_state_drift,
                          &inactive_state_drift_max, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            if (rank == 0 && timing_enabled) {
                std::cout
                    << "JFNK sector reduction residual: active_Linf="
                    << residual_norms.active_linf
                    << " forbidden_Linf=" << residual_norms.forbidden_linf
                    << " limit=" << forbidden_check.limit << '\n';
                std::cout
                    << "JFNK sector reduction: inactive state drift Linf="
                    << inactive_state_drift_max << " (limit=1e-14)\n";
            }
            if (guards_valid_all == 0) {
                const std::string reason =
                    !forbidden_check.allowed
                        ? "forbidden residual norm exceeded its runtime guard"
                        : (!local_state_valid
                               ? "inactive state drift check is unsupported: " +
                                     inactive_state_failure
                               : "inactive state drift exceeds 1e-14");
                abandon_jacobian_parity_reduction(
                    *parity_state, reason, rank);
                step_selection_plan.reset();
            }
        }

        const int linear_dimension = step_selection_plan
            ? static_cast<int>(
                  step_selection_plan->selected_columns().size())
            : column_count;
        std::vector<double> rhs;
        if (step_selection_plan) {
            const JacobianSelectedValues gathered =
                gather_jacobian_selected_values(
                    std::span<const double>{
                        residual.get_data(), residual.get_nbr()},
                    step_selection_plan->selected_rows());
            if (!gathered)
                KADATH_THROW("reduced JFNK RHS gather failed: " +
                             gathered.failure_reason);
            rhs = gathered.values;
        } else {
            rhs = vector_from_array(residual, column_count);
        }
        report_memory_mapper_phase("jfnk.step_entry", MPI_COMM_WORLD);

        const int mumps_ordering = config.mumps.ordering;
        const MumpsOutOfCoreMode out_of_core_mode = config.mumps.out_of_core;
        const double out_of_core_touch = config.mumps.out_of_core_touch;
        const double out_of_core_safety = config.mumps.out_of_core_safety;
        const double out_of_core_budget_mb =
            config.mumps.out_of_core_budget_mb;
        const int blr_icntl35 = config.mumps.blr;
        const int refresh_steps = std::max(1, config.jfnk_mumps.preconditioner_refresh_steps);
        const bool adaptive_refresh = config.jfnk_mumps.adaptive_preconditioner_refresh;
        const int adaptive_max_steps = std::max(
            refresh_steps, config.jfnk_mumps.adaptive_preconditioner_max_steps);
        require_collective_adaptive_config_agreement(
            adaptive_refresh,
            refresh_steps,
            adaptive_max_steps,
            config.jfnk_mumps.max_linear_iterations);
        const int factor_ranks_per_node =
            resolve_sparse_direct_mumps_ranks_per_node(
                config.mumps, MPI_COMM_WORLD);
        JacobianEmissionCaps emission_caps;
        emission_caps.selected_block_supported = true;
        emission_caps.physical_payload_supported = true;
        emission_caps.parity_mass_probe_requested =
            parity_mass_probe_requested;
        JacobianEmissionPlan emission_plan = plan_jacobian_emission(
            *this, column_count, step_selection_plan, emission_caps);
        require_collective_jacobian_emission_plan_agreement(
            emission_plan, MPI_COMM_WORLD);
        // Fused full-system assembly produces two physical sector factors even
        // without the classic post-hoc split opt-in. A reduced system produces
        // only the physical + block.
        const bool parity_split_requested =
            emission_plan.fingerprint.parity_split_requested;
        const bool preconditioner_settings_changed =
            mumps_state.jfnk_preconditioner_dimension != column_count ||
            mumps_state.jfnk_preconditioner_factor_dimension !=
                linear_dimension ||
            mumps_state.jfnk_preconditioner_selection_plan.get() !=
                step_selection_plan.get() ||
            !mumps_state.jfnk_preconditioner_emission_fingerprint ||
            *mumps_state.jfnk_preconditioner_emission_fingerprint !=
                emission_plan.fingerprint ||
            mumps_state.jfnk_preconditioner_refresh_steps != refresh_steps ||
            mumps_state.jfnk_preconditioner_ordering != mumps_ordering ||
            mumps_state.jfnk_preconditioner_out_of_core_mode !=
                out_of_core_mode ||
            mumps_state.jfnk_preconditioner_out_of_core_touch !=
                out_of_core_touch ||
            mumps_state.jfnk_preconditioner_out_of_core_safety !=
                out_of_core_safety ||
            mumps_state.jfnk_preconditioner_out_of_core_budget_mb !=
                out_of_core_budget_mb ||
            mumps_state.jfnk_preconditioner_blr != blr_icntl35 ||
            mumps_state.jfnk_preconditioner_ranks_per_node !=
                factor_ranks_per_node;
        if (preconditioner_settings_changed) {
            mumps_state.jfnk_preconditioner.reset();
            mumps_state.jfnk_preconditioner_emission_fingerprint.reset();
            mumps_state.jfnk_preconditioner_selection_plan.reset();
            mumps_state.jfnk_preconditioner_factor_dimension = 0;
            mumps_state.jfnk_step_index = 0;
            mumps_state.jfnk_preconditioner_step_index = 0;
            mumps_state.jfnk_eisenstat_walker_prev_residual = -1.0;
            mumps_state.jfnk_eisenstat_walker_prev_eta = -1.0;
            mumps_state.jfnk_previous_start_error = -1.0;
            mumps_state.jfnk_preconditioner_nnz = 0;
            mumps_state.jfnk_last_preconditioner_refresh_seconds = -1.0;
            mumps_state.jfnk_last_krylov_seconds = -1.0;
            mumps_state.jfnk_last_krylov_iterations = 0;
            mumps_state.jfnk_last_krylov_status = -1;
            mumps_state.jfnk_force_preconditioner_recovery = false;
        }

        constexpr double kPreconditionerRefreshErrorGrowthFactor = 10.0;
        const bool has_previous_start_error = mumps_state.jfnk_previous_start_error > 0.0;
        const double start_error_ratio =
            has_previous_start_error
                ? nonlinear_error_before_step / mumps_state.jfnk_previous_start_error
                : std::numeric_limits<double>::infinity();
        const double recycle_reset_converging_ratio =
            std::clamp(env_double_value("JFNK_MUMPS_PC_REUSE_CONVERGING_RATIO", 0.1),
                       0.0,
                       1.0);
        const bool reset_recycle_for_convergence =
            !adaptive_refresh &&
            mumps_state.jfnk_preconditioner &&
            recycle_reset_converging_ratio > 0.0 &&
            start_error_ratio <= recycle_reset_converging_ratio;
        if (reset_recycle_for_convergence) {
            mumps_state.jfnk_preconditioner_step_index = 0;
        }
        const bool force_refresh_for_error_growth =
            mumps_state.jfnk_preconditioner &&
            has_previous_start_error &&
            nonlinear_error_before_step >
                kPreconditionerRefreshErrorGrowthFactor * mumps_state.jfnk_previous_start_error;
        const int converged_status = static_cast<int>(GmresStatus::Code::Converged);
        const int max_iterations_status = static_cast<int>(GmresStatus::Code::MaxIterations);
        const int preconditioner_age_before_decision =
            mumps_state.jfnk_preconditioner_step_index;
        const bool previous_krylov_usable =
            mumps_state.jfnk_last_krylov_iterations > 0 &&
            (mumps_state.jfnk_last_krylov_status == converged_status ||
             mumps_state.jfnk_last_krylov_status == max_iterations_status);
        const bool had_preconditioner =
            static_cast<bool>(mumps_state.jfnk_preconditioner);
        const JfnkPreconditionerRefreshDecision refresh_decision =
            decide_jfnk_preconditioner_refresh({
                had_preconditioner,
                mumps_state.jfnk_force_preconditioner_recovery,
                force_refresh_for_error_growth,
                adaptive_refresh,
                previous_krylov_usable,
                preconditioner_age_before_decision,
                refresh_steps,
                adaptive_max_steps,
                mumps_state.jfnk_last_krylov_iterations,
                config.jfnk_mumps.max_linear_iterations,
                start_error_ratio,
                mumps_state.jfnk_last_krylov_seconds,
                mumps_state.jfnk_last_preconditioner_refresh_seconds,
            });
        const bool refresh_preconditioner = refresh_decision.refresh;
        const bool adaptively_deferred_refresh = refresh_decision.adaptively_deferred;

        if (refresh_preconditioner) {
            const auto preconditioner_refresh_start = std::chrono::steady_clock::now();
            mumps_state.jfnk_preconditioner_step_index = 0;
            mumps_state.jfnk_force_preconditioner_recovery = false;
            if (rank == 0 && force_refresh_for_error_growth) {
                std::cout << "JFNK-MUMPS PC refresh: nonlinear error grew by > "
                          << kPreconditionerRefreshErrorGrowthFactor
                          << "x (previous_start_error="
                          << mumps_state.jfnk_previous_start_error
                          << ", current_start_error=" << nonlinear_error_before_step
                          << ")" << std::endl;
            }
            if (rank == 0 && adaptive_refresh &&
                refresh_decision.reason != JfnkPreconditionerRefreshReason::InitialBuild &&
                !force_refresh_for_error_growth) {
                std::cout << "JFNK-MUMPS adaptive PC refresh: reason="
                          << jfnk_preconditioner_refresh_reason_name(refresh_decision.reason)
                          << " age=" << preconditioner_age_before_decision
                          << " fixed=" << refresh_steps
                          << " max=" << adaptive_max_steps << std::endl;
            }
            const double configured_drop_tol =
                (config.mumps.drop_tol > 0.0) ? config.mumps.drop_tol : 1e-8;
            constexpr double kDropTolMin = 1e-16;
            const double drop_tol =
                std::max(kDropTolMin,
                         configured_drop_tol * std::sqrt(std::sqrt(std::max(error, 0.0))));

            const auto build_start = std::chrono::steady_clock::now();
            JacobianAssembler assembler(*this, MPI_COMM_WORLD);
            AssembledJacobianCoo assembled_jacobian = assembler.assemble(
                drop_tol, emission_plan);
            if (assembled_jacobian.n != linear_dimension) {
                KADATH_THROW(
                    "JFNK preconditioner dimension does not match the frozen selection role");
            }
            report_memory_mapper_phase("jfnk.post_assembly", MPI_COMM_WORLD);
            const long long preconditioner_nnz = assembled_jacobian.nnz;
            if (rank == 0) {
                const std::shared_ptr<JacobianParityMaskState>& parity_state =
                    jacobian_parity_mask_state();
                const SparseDirectMumpsSystemMode system_mode =
                    sparse_direct_mumps_system_mode(
                        step_selection_plan != nullptr, parity_state.get());
                print_sparse_direct_mumps_system_summary(
                    std::cout, system_mode, linear_dimension,
                    preconditioner_nnz);
            }
            // Per-step Jacobian-assembly wall is always reported (the key per-step
            // cost); the heavier per-column / operator-action profiles stay gated.
            report_sparse_direct_mumps_jacobian_timing(
                MPI_COMM_WORLD, rank == 0 ? &std::cout : nullptr,
                elapsed_time(build_start),
                sparse_direct_mumps_coo_allocated_bytes(
                    assembled_jacobian));
            if (timing_enabled) {
                assembler.dump_column_profile();
                dump_ope_action_profile();
                reset_ope_action_profile();
                if (rank == 0) {
                    const MatchingLaneStats lane_stats = matching_lane_stats();
                    std::cout << "matching lane stats (rank 0): import_native="
                              << lane_stats.import_native_calls
                              << " import_scalar_fallback="
                              << lane_stats.import_scalar_fallback_calls
                              << " import_refusals=" << lane_stats.import_refusals
                              << " import_plan_points="
                              << lane_stats.import_plan_points
                              << " import_missing_inputs="
                              << lane_stats.import_missing_inputs
                              << " import_plan_cache_hits="
                              << lane_stats.import_plan_cache_hits
                              << " import_plan_cache_misses="
                              << lane_stats.import_plan_cache_misses
                              << " import_plan_cache_rebuilds="
                              << lane_stats.import_plan_cache_rebuilds
                              << " export_native="
                              << lane_stats.export_native_calls
                              << " export_scalar_fallback="
                              << lane_stats.export_scalar_fallback_calls
                              << " export_missing_lanes="
                              << lane_stats.export_missing_lanes << std::endl;
                }
                reset_matching_lane_stats();
                // diagonal_stats reads the gathered rank-0 COO.
                assembler.diagonal_stats(assembled_jacobian, drop_tol);
            }
            if (rank == 0 && !mumps_state.settings_printed) {
                if (blr_icntl35 > 0) {
                    std::cout << "do_newton_jfnk_mumps: enabling MUMPS BLR (ICNTL(35)="
                              << blr_icntl35 << ")" << std::endl;
                }
                mumps_state.settings_printed = true;
            }

            const std::shared_ptr<JacobianParityMaskState>& parity_state =
                jacobian_parity_mask_state();
            const bool split_candidate = sparse_direct_mumps_split_candidate(
                assembled_jacobian, parity_state.get(),
                parity_split_requested);

            // A MUMPSTREE archive describes one full-matrix analysis pair. It
            // remains eligible for the ordinary path, but is never composed
            // into either retained parity sector.
            MumpsTreeCache replay_cache;
            std::vector<int> replay_composed_jcn;
            SparseDirectMumpsReplayOptions replay_options;
            const SparseDirectMumpsReplayOptions* replay_policy = nullptr;
            const bool replay_would_be_requested = should_replay_mumps_tree_cache(
                config.mumps.tree_cache_enabled,
                mumps_state.jfnk_tree_cache_replay_disabled,
                refresh_preconditioner,
                had_preconditioner,
                !config.mumps.tree_cache_path.empty());
            const bool replay_requested =
                replay_would_be_requested && !split_candidate &&
                !step_selection_plan;
            if (replay_would_be_requested && split_candidate &&
                !step_selection_plan && rank == 0) {
                std::cout
                    << "JFNK-MUMPS tree cache replay bypassed: parity split "
                       "preconditioner requested\n";
            }
            if (replay_would_be_requested && step_selection_plan &&
                rank == 0) {
                std::cout
                    << "JFNK-MUMPS tree cache replay bypassed: reduced J++ "
                       "preconditioner requested\n";
            }
            if (replay_requested) {
                int cache_read_succeeded = 0;
                std::string cache_read_error;
                if (rank == 0) {
                    try {
                        replay_cache = read_mumps_tree_cache(
                            tree_cache_path, column_count);
                        cache_read_succeeded = 1;
                    } catch (const std::exception& cache_error) {
                        cache_read_error = cache_error.what();
                        remove_tree_cache_with_log(
                            tree_cache_path, "JFNK-MUMPS tree cache replay");
                    }
                }
                MPI_Bcast(&cache_read_succeeded, 1, MPI_INT, 0, MPI_COMM_WORLD);

                std::string replay_error;
                if (cache_read_succeeded != 0) {
                    int matching_applied =
                        rank == 0 && replay_cache.matching_applied ? 1 : 0;
                    MPI_Bcast(&matching_applied, 1, MPI_INT, 0, MPI_COMM_WORLD);
                    if (rank != 0)
                        replay_cache.matching_applied = matching_applied != 0;
                    broadcast_int_vector(
                        replay_cache.column_permutation_1based, 0,
                        MPI_COMM_WORLD);
                    broadcast_int_vector(
                        replay_cache.symmetric_permutation_1based, 0,
                        MPI_COMM_WORLD);

                    replay_error = collective_phase_error(
                        MPI_COMM_WORLD, "MUMPS tree cache column composition",
                        [&]() {
                            if (rank == 0) {
                                replay_composed_jcn =
                                    compose_mumps_tree_column_indices_1based(
                                        assembled_jacobian.jcn,
                                        replay_cache.column_permutation_1based,
                                        column_count);
                            }
                        });
                } else {
                    replay_error = rank == 0
                                       ? "MUMPS tree cache read failed: " +
                                             cache_read_error
                                       : "MUMPS tree cache read failed";
                }

                if (!replay_error.empty()) {
                    if (rank == 0)
                        std::cerr << "JFNK-MUMPS tree cache fallback: "
                                  << replay_error << '\n';
                    if (rank == 0 && cache_read_succeeded != 0)
                        remove_tree_cache_with_log(
                            tree_cache_path, "JFNK-MUMPS tree cache fallback");
                    mumps_state.jfnk_tree_cache_replay_disabled = true;
                } else {
                    replay_options.column_indices_1based =
                        &replay_composed_jcn;
                    replay_options.symmetric_permutation_1based =
                        &replay_cache.symmetric_permutation_1based;
                    replay_options.solution_column_permutation_1based =
                        &replay_cache.column_permutation_1based;
                    replay_policy = &replay_options;
                }
            }

            SparseDirectMumpsSolveOptions solve_options;
            solve_options.ordering = mumps_ordering;
            solve_options.out_of_core_mode = out_of_core_mode;
            solve_options.blr = blr_icntl35;
            solve_options.communicator = MPI_COMM_WORLD;
            solve_options.ranks_per_node = factor_ranks_per_node;
            solve_options.out_of_core_touch = out_of_core_touch;
            solve_options.out_of_core_safety = out_of_core_safety;
            solve_options.out_of_core_budget_mb = out_of_core_budget_mb;
            solve_options.parity_split_requested = parity_split_requested;
            solve_options.ordinary_factor_lifecycle =
                SparseDirectMumpsFactorLifecycle::Retained;
            solve_options.split_factor_lifecycle =
                SparseDirectMumpsFactorLifecycle::Retained;
            solve_options.measure_phases = true;
            solve_options.memory_phase_prefix = "jfnk";
            solve_options.diagnostic = rank == 0 ? &std::cout : nullptr;
            solve_options.ordinary_replay = replay_policy;

            // The cache contract captures the first successful ordinary
            // JOB=1 immediately, before JOB=2. Split analyses deliberately do
            // not emit a full-matrix cache entry.
            if (tree_cache_configured && !step_selection_plan &&
                !split_candidate &&
                refresh_decision.reason ==
                    JfnkPreconditionerRefreshReason::InitialBuild) {
                solve_options.ordinary_analysis_observer =
                    [&](const SparseDirectMumpsAnalysisSnapshot& snapshot) {
                        try {
                            MumpsTreeCacheView cache_view;
                            cache_view.dimension = column_count;
                            cache_view.pattern_nnz = preconditioner_nnz;
                            cache_view.matching_applied =
                                snapshot.matching_applied;
                            cache_view.column_permutation_1based =
                                snapshot.column_permutation_1based;
                            cache_view.symmetric_permutation_1based =
                                snapshot.symmetric_permutation_1based;
                            write_mumps_tree_cache(tree_cache_path, cache_view);
                            std::cout << "JFNK-MUMPS tree cache saved: "
                                      << tree_cache_path.string()
                                      << " n=" << column_count
                                      << " nnz=" << preconditioner_nnz << '\n';
                        } catch (const std::exception& cache_error) {
                            std::cerr
                                << "JFNK-MUMPS tree cache write skipped: "
                                << cache_error.what() << '\n';
                        }
                    };
            }
            if (replay_policy != nullptr) {
                solve_options.ordinary_replay_success_observer = [&]() {
                    if (rank == 0) {
                        std::cout << "JFNK-MUMPS tree cache replayed: "
                                  << tree_cache_path.string()
                                  << " stored_nnz=" << replay_cache.pattern_nnz
                                  << " current_nnz=" << preconditioner_nnz
                                  << " matching="
                                  << (replay_cache.matching_applied ? 1 : 0)
                                  << '\n';
                    }
                };
                solve_options.ordinary_replay_failure_observer =
                    [&](const std::string& replay_error) {
                        if (rank == 0) {
                            std::cerr << "JFNK-MUMPS tree cache fallback: "
                                      << replay_error << '\n';
                            remove_tree_cache_with_log(
                                tree_cache_path,
                                "JFNK-MUMPS tree cache fallback");
                        }
                        mumps_state.jfnk_tree_cache_replay_disabled = true;
                    };
            }

            SparseDirectMumpsSolveResult solve_result =
                run_sparse_direct_mumps_solve(
                    assembled_jacobian, nullptr, parity_state.get(),
                    mumps_state.icntl14, solve_options);
            if (!solve_result.retained_factor) {
                KADATH_THROW(
                    "direct MUMPS primitive did not retain the JFNK "
                    "preconditioner factor");
            }

            // Phase boundary: gathered COO triples just freed. At res=13 this
            // releases ~6.6 GB on rank 0. Pages above libmalloc's mmap
            // threshold (~128 KB on macOS) munmap immediately; the rest goes
            // to libmalloc's heap free-list. Ask libmalloc to release that
            // too before MUMPS holds its LU through the GMRES iters. No-op
            // unless RELEASE_ALLOCATOR_PAGES=1.
            release_allocator_pages();
            mumps_state.jfnk_preconditioner_dimension = column_count;
            mumps_state.jfnk_preconditioner_refresh_steps = refresh_steps;
            mumps_state.jfnk_preconditioner_ordering = mumps_ordering;
            mumps_state.jfnk_preconditioner_out_of_core_mode =
                out_of_core_mode;
            mumps_state.jfnk_preconditioner_out_of_core_touch =
                out_of_core_touch;
            mumps_state.jfnk_preconditioner_out_of_core_safety =
                out_of_core_safety;
            mumps_state.jfnk_preconditioner_out_of_core_budget_mb =
                out_of_core_budget_mb;
            mumps_state.jfnk_preconditioner_blr = blr_icntl35;
            mumps_state.jfnk_preconditioner_ranks_per_node =
                factor_ranks_per_node;
            JacobianEmissionFingerprint retained_plan_fingerprint =
                emission_plan.fingerprint;
            retained_plan_fingerprint.parity_split_ready =
                solve_result.parity_split;
            const bool retained_parity_structure_valid =
                solve_result.parity_split && parity_state &&
                parity_state->n == column_count &&
                parity_state->row_sector.size() ==
                    static_cast<std::size_t>(column_count) &&
                parity_state->column_sector.size() ==
                    static_cast<std::size_t>(column_count);
            retained_plan_fingerprint.row_sector =
                retained_parity_structure_valid
                    ? parity_state->row_sector
                    : std::vector<signed char>{};
            retained_plan_fingerprint.column_sector =
                retained_parity_structure_valid
                    ? parity_state->column_sector
                    : std::vector<signed char>{};
            mumps_state.jfnk_preconditioner_emission_fingerprint =
                std::make_shared<const JacobianEmissionFingerprint>(
                    std::move(retained_plan_fingerprint));
            mumps_state.jfnk_preconditioner_selection_plan =
                step_selection_plan;
            mumps_state.jfnk_preconditioner_factor_dimension =
                linear_dimension;
            mumps_state.jfnk_preconditioner_nnz = preconditioner_nnz;
            mumps_state.jfnk_preconditioner =
                std::move(solve_result.retained_factor);
            if (adaptive_refresh) {
                mumps_state.jfnk_last_preconditioner_refresh_seconds =
                    global_max_seconds(elapsed_time(preconditioner_refresh_start));
            }
        } else {
            if (rank == 0) {
                const std::shared_ptr<JacobianParityMaskState>& parity_state =
                    jacobian_parity_mask_state();
                const SparseDirectMumpsSystemMode system_mode =
                    sparse_direct_mumps_system_mode(
                        step_selection_plan != nullptr, parity_state.get());
                print_sparse_direct_mumps_system_summary(
                    std::cout, system_mode, linear_dimension,
                    mumps_state.jfnk_preconditioner_nnz);
                std::cout << "JFNK-MUMPS PC reuse: step="
                          << mumps_state.jfnk_preconditioner_step_index
                          << " refresh=" << refresh_steps;
                if (reset_recycle_for_convergence) {
                    std::cout << " (recycle reset: error_ratio="
                              << start_error_ratio
                              << " <= " << recycle_reset_converging_ratio << ")";
                }
                if (adaptively_deferred_refresh) {
                    std::cout << " (adaptive: projected_stale="
                              << refresh_decision.projected_stale_krylov_seconds
                              << "s < refresh="
                              << mumps_state.jfnk_last_preconditioner_refresh_seconds
                              << "s, error_ratio=" << start_error_ratio
                              << ", max_age=" << adaptive_max_steps << ")";
                }
                std::cout << std::endl;
            }
        }

        LinearSolver& preconditioner = *mumps_state.jfnk_preconditioner;

        const auto krylov_start = std::chrono::steady_clock::now();
        Array<double> matvec_delta_array(column_count);
        Array<double> applied_delta_array(column_count);
        double exact_jv_input_copy_seconds = 0.0;
        double exact_jv_apply_seconds = 0.0;
        double exact_jv_output_copy_seconds = 0.0;
        double preconditioner_input_copy_seconds = 0.0;
        double preconditioner_solve_seconds = 0.0;
        const auto matvec = [this, column_count, linear_dimension,
                             timing_enabled, step_selection_plan,
                             &matvec_delta_array, &applied_delta_array,
                             &exact_jv_input_copy_seconds, &exact_jv_apply_seconds,
                             &exact_jv_output_copy_seconds](
                                const std::vector<double>& trial_delta,
                                std::vector<double>& jacobian_delta) {
            // Per-kernel timing probe context. Enabled by
            // KERNEL_PROFILE=1; default-off cost is a single
            // pointer compare on kernel entry.
            KernelProfileScope kernel_profile_exact_jv(KernelContext::ExactJv);
            const auto load_trial = [&]() {
                if (trial_delta.size() !=
                    static_cast<std::size_t>(linear_dimension)) {
                    KADATH_THROW(
                        "reduced JFNK matvec input has the wrong dimension");
                }
                if (!step_selection_plan) {
                    std::copy(trial_delta.begin(), trial_delta.end(),
                              matvec_delta_array.set_data());
                    return;
                }
                std::fill_n(matvec_delta_array.set_data(),
                            static_cast<std::size_t>(column_count), 0.0);
                const std::vector<int>& columns =
                    step_selection_plan->selected_columns();
                for (std::size_t reduced = 0; reduced < columns.size();
                     ++reduced) {
                    matvec_delta_array.set(columns[reduced]) =
                        trial_delta[reduced];
                }
            };
            const auto store_result = [&]() {
                if (!step_selection_plan) {
                    jacobian_delta.assign(
                        applied_delta_array.set_data(),
                        applied_delta_array.set_data() + column_count);
                    return;
                }
                const std::vector<int>& rows =
                    step_selection_plan->selected_rows();
                jacobian_delta.resize(rows.size());
                for (std::size_t reduced = 0; reduced < rows.size();
                     ++reduced) {
                    jacobian_delta[reduced] =
                        applied_delta_array(rows[reduced]);
                }
            };
            if (timing_enabled) {
                const auto copy_start = std::chrono::steady_clock::now();
                load_trial();
                exact_jv_input_copy_seconds += elapsed_time(copy_start);

                const auto apply_start = std::chrono::steady_clock::now();
                do_JX(matvec_delta_array, applied_delta_array);
                exact_jv_apply_seconds += elapsed_time(apply_start);

                const auto output_start = std::chrono::steady_clock::now();
                store_result();
                exact_jv_output_copy_seconds += elapsed_time(output_start);
            } else {
                load_trial();
                do_JX(matvec_delta_array, applied_delta_array);
                store_result();
            }
        };
        const auto right_preconditioner =
            [timing_enabled, &preconditioner, &preconditioner_input_copy_seconds,
             &preconditioner_solve_seconds](
                const std::vector<double>& krylov_vector,
                std::vector<double>& preconditioned_delta) {
                if (timing_enabled) {
                    const auto copy_start = std::chrono::steady_clock::now();
                    preconditioned_delta = krylov_vector;
                    preconditioner_input_copy_seconds += elapsed_time(copy_start);
                } else {
                    preconditioned_delta = krylov_vector;
                }
                const auto solve_start = std::chrono::steady_clock::now();
                preconditioner.solve(preconditioned_delta.data());
                preconditioner_solve_seconds += elapsed_time(solve_start);
            };

        GmresConfig gmres_config;
        gmres_config.max_iters = config.jfnk_mumps.max_linear_iterations;
        const double rhs_norm = euclidean_norm(rhs);
        const double linear_floor = config.jfnk_mumps.linear_relative_tolerance;
        const bool ew_enabled = env_flag_enabled("JFNK_EW", true);
        double eta_used = linear_floor;
        if (ew_enabled) {
            const EisenstatWalkerParameters ew_params =
                read_eisenstat_walker_parameters(linear_floor);
            eta_used = next_eisenstat_walker_eta(
                ew_params,
                mumps_state.jfnk_eisenstat_walker_prev_residual,
                mumps_state.jfnk_eisenstat_walker_prev_eta,
                rhs_norm);
        }
        gmres_config.tolerance =
            std::max(std::numeric_limits<double>::epsilon(), eta_used * rhs_norm);
        GmresTiming gmres_timing;
        if (timing_enabled) {
            gmres_config.timing = &gmres_timing;
        }

        std::unique_ptr<ValDomainDerAbsAssemblyCacheScope> exact_jv_der_abs_cache_scope;
        if (exact_jv_der_abs_cache_enabled()) {
            exact_jv_der_abs_cache_scope =
                std::make_unique<ValDomainDerAbsAssemblyCacheScope>();
        }

        std::vector<double> delta(
            static_cast<std::size_t>(linear_dimension), 0.0);
        const GmresStatus gmres_status =
            right_preconditioned_gmres(rhs, delta, matvec, right_preconditioner, gmres_config);
        const double krylov_seconds_local = elapsed_time(krylov_start);
        const double krylov_seconds = adaptive_refresh
                                          ? global_max_seconds(krylov_seconds_local)
                                          : krylov_seconds_local;

        if (rank == 0) {
            const double relative_linear_residual =
                rhs_norm > 0.0 ? gmres_status.residual_norm / rhs_norm : gmres_status.residual_norm;
            std::cout << "JFNK-MUMPS GMRES: " << gmres_status_name(gmres_status.code)
                      << " in " << gmres_status.iterations
                      << " iters, rel_resid=" << relative_linear_residual << '\n';
            if (ew_enabled) {
                std::cout << "  Eisenstat-Walker: eta=" << eta_used
                          << "  |F| " << mumps_state.jfnk_eisenstat_walker_prev_residual
                          << " -> " << rhs_norm << '\n';
            }
            std::cout << std::flush;
        }
        report_sparse_direct_mumps_apply_timing(
            MPI_COMM_WORLD, rank == 0 ? &std::cout : nullptr,
            preconditioner_solve_seconds);

        mumps_state.jfnk_last_krylov_seconds = krylov_seconds;
        mumps_state.jfnk_last_krylov_iterations = gmres_status.iterations;
        mumps_state.jfnk_last_krylov_status = static_cast<int>(gmres_status.code);
        report_memory_mapper_phase("jfnk.post_gmres", MPI_COMM_WORLD);

        if (timing_enabled) {
            constexpr std::size_t phase_count = 12;
            const std::array<const char*, phase_count> labels = {
                "JFNK-MUMPS GMRES solve",
                "  GMRES preconditioner callback",
                "    Krylov-vector copy",
                "    MUMPS solve+broadcast",
                "  GMRES exact-Jv callback",
                "    std::vector -> Array copy",
                "    do_JX exact apply",
                "    Array -> std::vector copy",
                "  GMRES orthogonalization",
                "  GMRES vector norms",
                "  GMRES least squares",
                "  GMRES solution update"};
            const std::array<double, phase_count> seconds = {
                krylov_seconds_local,
                gmres_timing.precondition_seconds,
                preconditioner_input_copy_seconds,
                preconditioner_solve_seconds,
                gmres_timing.matvec_seconds,
                exact_jv_input_copy_seconds,
                exact_jv_apply_seconds,
                exact_jv_output_copy_seconds,
                gmres_timing.orthog_seconds,
                gmres_timing.vector_norm_seconds,
                gmres_timing.least_squares_seconds,
                gmres_timing.update_seconds};
            report_timing_batch(rank, nproc, labels, seconds);
            if (rank == 0) {
                // Replicated GMRES executes callbacks in the same order on every
                // rank; the root-local counts expose ordering drift without
                // adding another diagnostic collective.
                std::cout << "  GMRES callback counts: matvecs=" << gmres_timing.matvecs
                          << ", preconditions=" << gmres_timing.preconditions << std::endl;
            }
        }

        // An adaptively deferred build is speculative, but never commits an
        // unusable direction. Retry the same nonlinear step immediately with a
        // fresh factor before any field or adapted-domain state is mutated.
        // Ordinary max-iteration exits remain usable when GMRES reduced the
        // residual: the line search guards the nonlinear correction and the
        // next-step progress check will force a refresh on stagnation.
        const bool finite_linear_residual = std::isfinite(gmres_status.residual_norm);
        const bool local_unusable_deferred_correction =
            adaptively_deferred_refresh &&
            (gmres_status.code == GmresStatus::Code::InvalidInput ||
             gmres_status.code == GmresStatus::Code::Breakdown ||
             gmres_status.iterations == 0 ||
             !finite_linear_residual ||
             (rhs_norm > 0.0 && gmres_status.residual_norm >= rhs_norm));
        int unusable_deferred_correction =
            local_unusable_deferred_correction ? 1 : 0;
        if (adaptively_deferred_refresh) {
            int any_unusable_deferred_correction = 0;
            MPI_Allreduce(&unusable_deferred_correction,
                          &any_unusable_deferred_correction,
                          1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            unusable_deferred_correction = any_unusable_deferred_correction;
        }
        if (unusable_deferred_correction) {
            if (rank == 0) {
                std::cout << "JFNK-MUMPS adaptive PC recovery: stale GMRES "
                          << gmres_status_name(gmres_status.code)
                          << " after " << gmres_status.iterations
                          << " iterations; rebuilding before applying a correction"
                          << std::endl;
            }
            report_timing(rank, nproc,
                          "JFNK-MUMPS discarded stale GMRES",
                          krylov_seconds_local);
            mumps_state.jfnk_force_preconditioner_recovery = true;
            exact_jv_der_abs_cache_scope.reset();
            const bool recovered = do_newton_jfnk_mumps(precision, error, config);
            report_timing(rank, nproc,
                          "JFNK-MUMPS adaptive recovery total",
                          elapsed_time(step_start));
            return recovered;
        }

        if (ew_enabled) {
            mumps_state.jfnk_eisenstat_walker_prev_residual = rhs_norm;
            mumps_state.jfnk_eisenstat_walker_prev_eta = eta_used;
        }

        if (!gmres_status.converged) {
            if (rank == 0) {
                std::cerr << "do_newton_jfnk_mumps: GMRES did not meet tolerance; "
                          << "using best available Krylov correction."
                          << std::endl;
            }
            if (gmres_status.code == GmresStatus::Code::InvalidInput ||
                gmres_status.iterations == 0) {
                // The compact apply metric above intentionally describes the
                // retained-factor applications requested by GMRES. This rare
                // post-GMRES fallback constructs a correction after an invalid
                // Krylov run and is not a GMRES preconditioner callback.
                delta = rhs;
                preconditioner.solve(delta.data());
            }
        }

        Array<double> delta_array(column_count);
        if (step_selection_plan) {
            const JacobianSelectedValues scattered =
                scatter_jacobian_selected_values(
                    delta, column_count,
                    step_selection_plan->selected_columns());
            if (!scattered)
                KADATH_THROW("reduced JFNK correction scatter failed: " +
                             scattered.failure_reason);
            copy_vector_to_array(scattered.values, delta_array);
        } else {
            copy_vector_to_array(delta, delta_array);
        }
        const double step_scale = env_double_value("JFNK_STEP_SCALE",
                                                   env_double_value("NEWTON_STEP_SCALE", 1.0));
        if (step_scale != 1.0) {
            for (int i = 0; i < delta_array.get_size(0); ++i)
                delta_array.set(i) *= step_scale;
            if (rank == 0)
                std::cout << "JFNK-MUMPS Newton step scaled by " << step_scale << std::endl;
        }
        // Apply the (optionally scaled) Newton correction to the unknowns: this
        // both moves the adapted-domain surfaces (xx_to_vars_variable_domains)
        // and updates the field coefficients (xx_to_vars_delta). Factored into a
        // lambda so the line-search self-test can apply the same step repeatedly.
        const auto apply_correction = [this](Array<double>& correction) {
            int offset = 0;
            espace.xx_to_vars_variable_domains(this, correction, offset);
            xx_to_vars_delta(correction, offset);
        };

        // P1 line-search self-test (JFNK_LS_SELFTEST, one-shot at step 0):
        // prove snapshot_state/restore_state round-trips the full unknown state
        // bit-exactly across an adapted-domain step before the line search relies
        // on it. Two independent checks: (1) restore returns to the pre-trial
        // residual (catches an incomplete restore), (2) re-applying the same step
        // reproduces the residual (catches a nondeterministic retry). Exactly one
        // accepted full step is left committed.
        if (env_flag_enabled("JFNK_LS_SELFTEST", false) && mumps_state.jfnk_step_index == 0) {
            const Array<double> pretrial_residual(sec_member());
            State_snapshot& pretrial = jfnk_line_search_snapshot_;
            snapshot_state_into(pretrial);

            apply_correction(delta_array);
            const Array<double> first_trial(sec_member());

            restore_state(pretrial);
            const Array<double> restored_residual(sec_member());
            const ArrayDiff restore_diff = compare_arrays_bitwise(pretrial_residual, restored_residual);

            apply_correction(delta_array);
            const Array<double> second_trial(sec_member());
            const ArrayDiff retry_diff = compare_arrays_bitwise(first_trial, second_trial);

            // Rank 0 owns the authoritative verdict; broadcast it so every rank
            // exits with the same status (a clean machine-readable gate signal).
            int verdict = (restore_diff.exact && retry_diff.exact) ? 0 : 1;
            if (rank == 0) {
                report_selftest(std::cout, "restore -> pretrial residual", restore_diff);
                report_selftest(std::cout, "re-apply reproducibility", retry_diff);
                std::cout << "JFNK-MUMPS line-search self-test: "
                          << (verdict == 0 ? "PASS (snapshot/restore is bit-exact)"
                                           : "FAIL (snapshot/restore is NOT exact; see diffs above)")
                          << std::endl;
                std::cout.flush();
            }
            // The self-test is a one-shot diagnostic gate, not production: exit
            // collectively after the verdict so it costs one Newton step, not a
            // full solve. Every rank must MPI_Finalize so the launcher exits.
            MPI_Bcast(&verdict, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            std::exit(verdict);
        } else if (config.jfnk_mumps.line_search) {
            // Guarded backtracking: commit the full step when it does not worsen
            // the residual, otherwise damp via snapshot/restore (the line search
            // applies the step itself).
            error = apply_jfnk_line_search(delta_array, nonlinear_error_before_step, rank);
        } else {
            apply_correction(delta_array);
            Array<double> trial_residual(sec_member_partitioned());
            error = infinity_norm(trial_residual);
        }
        report_memory_mapper_phase("jfnk.step_exit", MPI_COMM_WORLD);

        // Per-Newton-step per-kernel report (env-gated, default off).
        kernel_profile_report_and_reset(rank, mumps_state.jfnk_step_index);
        mumps_state.jfnk_previous_start_error = nonlinear_error_before_step;
        ++mumps_state.jfnk_step_index;
        ++mumps_state.jfnk_preconditioner_step_index;
        return false;
#else
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "do_newton_jfnk_mumps: built without MUMPS.\n";
        }
        KADATH_THROW("CELEPHAIS_SOLVER=jfnk-mumps requires a binary built with CELEPHAIS_USE_MUMPS");
#endif
    }
} // namespace Kadath
