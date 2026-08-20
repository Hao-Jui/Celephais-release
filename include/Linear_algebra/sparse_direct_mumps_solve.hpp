/*
    Added 2026 Hao-Jui Kuan
    Reusable sparse-direct MUMPS solve orchestration.
*/

#pragma once

#include "For_Kadath/System_of_eqs/solver_runtime_config.hpp"
#include "Linear_algebra/linear_solver.hpp"

#include <mpi.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace Kadath
{
    struct AssembledJacobianCoo;
    struct JacobianParityMaskState;

    /** Reachable sparse-MUMPS system layouts.  Sector reduction requires the
     * parity mask, so a reduced/unmasked state is deliberately unrepresentable.
     */
    enum class SparseDirectMumpsSystemMode
    {
        FullUnmasked,
        FullMasked,
        ReducedMasked,
    };

    SparseDirectMumpsSystemMode sparse_direct_mumps_system_mode(
        bool reduced_sector,
        const JacobianParityMaskState* parity_state) noexcept;

    void print_sparse_direct_mumps_system_summary(
        std::ostream& diagnostic, SparseDirectMumpsSystemMode mode,
        int dof, long long nnz);

    /** Allocated storage of the centralized COO value/index buffers. */
    std::size_t sparse_direct_mumps_coo_allocated_bytes(
        const AssembledJacobianCoo& assembled_jacobian) noexcept;

    /** Collective Jacobian timing plus rank-0 COO allocation renderer. */
    void report_sparse_direct_mumps_jacobian_timing(
        MPI_Comm communicator, std::ostream* diagnostic, double seconds,
        std::size_t coo_allocated_bytes);

    /** Collective factor timing and memory summary. Memory is the MUMPS
     * max-rank INFOG(21) effective use and INFOG(18) allocation, in decimal MB.
     */
    void report_sparse_direct_mumps_factorization(
        MPI_Comm communicator, std::ostream* diagnostic, const char* label,
        double analyze_seconds, double factorize_seconds,
        const std::string& ordering, bool out_of_core,
        int factor_memory_used_mb, int factor_memory_allocated_mb);

    /** Report only the slowest-rank elapsed time for applying factored MUMPS. */
    void report_sparse_direct_mumps_apply_timing(
        MPI_Comm communicator, std::ostream* diagnostic, double seconds);

    int resolve_sparse_direct_mumps_ranks_per_node(
        const MumpsRuntimeConfig& config, MPI_Comm communicator);

    void sparse_direct_collective_throw_if_failed(
        MPI_Comm communicator, const char* phase,
        const std::string& local_error);

    enum class SparseDirectMumpsFactorLifecycle
    {
        Transient,
        Retained,
    };

    struct SparseDirectMumpsAnalysisSnapshot
    {
        bool matching_applied = false;
        std::vector<int> column_permutation_1based;
        std::vector<int> symmetric_permutation_1based;
    };

    /**
     * Full-matrix replay metadata consumed by the direct-owned ordinary path.
     * Sector splitting deliberately ignores this policy: a permutation learned
     * for the full matrix cannot be applied independently to either sector.
     */
    struct SparseDirectMumpsReplayOptions
    {
        int ordering = 1;
        const std::vector<int>* column_indices_1based = nullptr;
        const std::vector<int>* symmetric_permutation_1based = nullptr;
        const std::vector<int>* solution_column_permutation_1based = nullptr;
        bool detect_null_pivots = true;
        double null_pivot_threshold = 0.0;
    };

    struct SparseDirectMumpsSolveOptions
    {
        int ordering = 7;
        MumpsOutOfCoreMode out_of_core_mode = MumpsOutOfCoreMode::Auto;
        int blr = 0;
        MPI_Comm communicator = MPI_COMM_WORLD;
        int ranks_per_node = 0;
        double out_of_core_touch = kMumpsOutOfCoreTouchDefault;
        double out_of_core_safety = kMumpsOutOfCoreSafetyDefault;
        double out_of_core_budget_mb = kMumpsOutOfCoreBudgetUnset;
        bool parity_split_requested = false;
        SparseDirectMumpsFactorLifecycle ordinary_factor_lifecycle =
            SparseDirectMumpsFactorLifecycle::Transient;
        SparseDirectMumpsFactorLifecycle split_factor_lifecycle =
            SparseDirectMumpsFactorLifecycle::Transient;
        bool measure_phases = false;
        bool report_apply_timing = false;
        const char* memory_phase_prefix = nullptr;
        std::ostream* diagnostic = nullptr;
        const SparseDirectMumpsReplayOptions* ordinary_replay = nullptr;
        std::function<void(const SparseDirectMumpsAnalysisSnapshot&)>
            ordinary_analysis_observer;
        std::function<void()> ordinary_replay_success_observer;
        std::function<void(const std::string&)>
            ordinary_replay_failure_observer;
    };

    enum class SparseDirectMumpsParityLayout
    {
        None,
        Plus,
        PlusMinus,
    };

    struct SparseDirectMumpsSolveResult
    {
        std::unique_ptr<LinearSolver> retained_factor;
        bool parity_split = false;
        SparseDirectMumpsParityLayout parity_layout =
            SparseDirectMumpsParityLayout::None;
        bool replay_attempted = false;
        bool replay_succeeded = false;
        std::string replay_failure_reason;
        double analyze_seconds = 0.0;
        double factorize_seconds = 0.0;
        // Sum of solve+broadcast calls measured for phase or compact apply
        // reporting.
        double solve_seconds = 0.0;
    };

    bool sparse_direct_mumps_split_candidate(
        const AssembledJacobianCoo& assembled_jacobian,
        const JacobianParityMaskState* parity_state,
        bool parity_split_requested) noexcept;

    /**
     * Consume either one legacy centralized COO or the assembler's ordered
     * physical parity payload. A physical + block is factored once; a +,-
     * payload is factored sector by sector without repartitioning. The legacy
     * COO can still be split in place by the classic opt-in policy. Solve
     * @p rhs_inout when it is non-null.
     * ICNTL(14) is carried from the first successful sector into the second and
     * returned collectively through @p icntl14, matching sparse-direct Newton.
     *
     * Factor lifetime is caller policy. A transient split factors, solves, and
     * destroys each sector before constructing the next. A retained split keeps
     * both sector factors resident so later calls through the returned
     * LinearSolver can route two triangular solves by the immutable sector maps.
     * Consequently, split retention reduces only factorization front/workspace
     * peak; resident LU memory is the sum of both sector factors, not their max.
     */
    SparseDirectMumpsSolveResult run_sparse_direct_mumps_solve(
        AssembledJacobianCoo& assembled_jacobian,
        double* rhs_inout,
        const JacobianParityMaskState* parity_state,
        int& icntl14,
        const SparseDirectMumpsSolveOptions& options);
} // namespace Kadath
