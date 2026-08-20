/*
    Runtime solver configuration for Celephais.

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include "Linear_algebra/mumps_out_of_core_mode.hpp"

#include <string>

namespace Kadath
{
    inline constexpr double kSparseChordEntryTolerance = 1e-3;

    enum class NewtonBackend {
        Dense,
        Mumps,
        JfnkMumps,
        JfnkDense,
        JfnkSchur
    };

    struct MumpsRuntimeConfig {
        double drop_tol = 1e-14;
        int ordering = 7;
        MumpsOutOfCoreMode out_of_core = MumpsOutOfCoreMode::Auto;
        double out_of_core_touch = kMumpsOutOfCoreTouchDefault;
        double out_of_core_safety = kMumpsOutOfCoreSafetyDefault;
        // Test-only override for the probed node-available memory in MB.
        // Safety is still applied. The exact -1 sentinel uses the platform probe.
        double out_of_core_budget_mb = kMumpsOutOfCoreBudgetUnset;
        int blr = 0;
        // Persist the initial JFNK-MUMPS analysis pair beside the stage's final
        // converged dataset, then replay it only for later PC refreshes. The
        // enable gate is parsed from MUMPS_TREE_CACHE; the path is
        // supplied explicitly by the stage so checkpoints cannot redirect it.
        bool tree_cache_enabled = false;
        std::string tree_cache_path;
        // Factor MUMPS on only the first `ranks_per_node` ranks of each node (a
        // sub-communicator); assembly/do_JX/GMRES stay on all ranks. The factor wall
        // is a U-curve in factor-rank count (compute-bound with too few, comm-bound
        // with too many; see ASSESSMENT B.2), so the default targets ~1/4 of a node.
        //   -1 (default, env unset) = max(floor(local_ranks/4), 1)
        //    0                      = all ranks
        //    N                      = N factor-ranks per node
        int ranks_per_node = -1;
        // Sparse direct Newton freezes the first error-scaled numerical
        // threshold throughout one nonlinear solve and reuses MUMPS JOB=1
        // while later support is contained by the first analyzed pattern. A
        // support miss still forces safe superset growth + reanalysis. Negative
        // means use the frozen numerical threshold for the pattern too,
        // avoiding a lower-threshold RSS-heavy superset by default.
        bool sparse_analyze_reuse = false;
        // Retain an ordinary sparse-direct factorization and use chord Newton
        // corrections while they contract the nonlinear residual sufficiently.
        bool sparse_chord_reuse = true;
        double sparse_pattern_drop_tol = -1.0;
        // Refuse the prototype for the remainder of a solve if the candidate
        // superset exceeds this multiple of the ordinary numerical nnz.
        double sparse_superset_max_nnz_ratio = 2.0;
    };

    struct SparseDirectDropPolicy {
        double numerical_drop_tol = 1e-8;
        double pattern_drop_tol = 1e-8;
    };

    struct SparseDirectDropState {
        int dimension = 0;
        double configured_drop_tol = -1.0;
        double frozen_numerical_drop_tol = -1.0;

        void reset() noexcept
        {
            dimension = 0;
            configured_drop_tol = -1.0;
            frozen_numerical_drop_tol = -1.0;
        }
    };

    // On the first step of a solve, reproduce the historical error-scaled
    // threshold exactly. Later steps reuse that value until state is reset or
    // the dimension/configured tolerance changes. An explicit pattern
    // threshold remains clamped to the frozen numerical threshold.
    SparseDirectDropPolicy resolve_sparse_direct_drop_policy(
        const MumpsRuntimeConfig& config,
        double nonlinear_error,
        int dimension,
        SparseDirectDropState& state) noexcept;

    struct SolverDiagnosticsConfig {
        bool timing = false;
        bool def_filter = true;
        // Diagnostic-only exact sparse-direct matrix/RHS capture. An empty
        // path disables the writer and leaves numerical controls unchanged.
        // The ordinal is counted
        // over sparse-direct Jacobian assemblies in one process and makes it
        // possible to select a later staged system without dumping every
        // matrix. Capture deliberately refuses the analyze-reuse superset path
        // because its factor input is not the newly assembled COO stream.
        std::string direct_replay_capture_path;
        int direct_replay_capture_ordinal = 1;
    };

    struct JfnkMumpsRuntimeConfig {
        int max_linear_iterations = 64;
        int preconditioner_refresh_steps = 10;
        // Optional cost-aware extension of the fixed refresh cadence. The
        // ordinary `preconditioner_refresh_steps` cadence remains the default
        // and the hard fallback whenever the controller lacks trustworthy
        // measurements. When enabled, a healthy/cheap stale preconditioner may
        // be retained up to `adaptive_preconditioner_max_steps`; nonlinear
        // stagnation, Krylov failure, or the hard limit forces a refresh.
        bool adaptive_preconditioner_refresh = false;
        int adaptive_preconditioner_max_steps = 20;
        double linear_relative_tolerance = 1e-8;
        // Guarded backtracking line search. Global default ON: the
        // snapshot/restore primitive is now bit-exact on every adapted space,
        // including the 2D polar adapted domains (Space_polar_adapted / NS2d),
        // after adapted_polar.hpp gained the snapshot_mapping/restore_mapping
        // overrides its 3D counterparts already had. It is a no-op on healthy
        // trajectories and only engages when a full Newton step would increase
        // |F|. JFNK_LINESEARCH overrides everywhere.
        bool line_search = true;
    };

    enum class JfnkPreconditionerRefreshReason {
        InitialBuild,
        RecoveryRetry,
        NonlinearGrowth,
        FixedCadence,
        AdaptiveHardLimit,
        AdaptiveInsufficientEvidence,
        AdaptivePoorProgress,
        AdaptiveKrylovFailure,
        AdaptiveKrylovCost,
        ReuseBeforeCadence,
        AdaptiveReuse
    };

    struct JfnkPreconditionerRefreshInput {
        bool has_preconditioner = false;
        bool recovery_retry = false;
        bool nonlinear_error_grew = false;
        bool adaptive_enabled = false;
        bool previous_krylov_usable = false;
        int preconditioner_age = 0;
        int fixed_refresh_steps = 10;
        int adaptive_max_steps = 20;
        int previous_krylov_iterations = 0;
        int max_krylov_iterations = 64;
        double nonlinear_error_ratio = 0.0;
        double previous_krylov_seconds = 0.0;
        double previous_refresh_seconds = 0.0;
    };

    struct JfnkPreconditionerRefreshDecision {
        bool refresh = true;
        bool adaptively_deferred = false;
        JfnkPreconditionerRefreshReason reason =
            JfnkPreconditionerRefreshReason::InitialBuild;
        // Cost estimate used for the economic decision: measured solve time
        // scaled by the square of the projected/observed Krylov-depth ratio.
        // The projection doubles the previous iteration count (with a
        // four-iteration minimum allowance), capped at the configured limit.
        double projected_stale_krylov_seconds = 0.0;
    };

    JfnkPreconditionerRefreshDecision decide_jfnk_preconditioner_refresh(
        const JfnkPreconditionerRefreshInput& input) noexcept;

    bool should_replay_mumps_tree_cache(
        bool cache_enabled,
        bool replay_disabled_for_stage,
        bool refresh_requested,
        bool had_preconditioner,
        bool path_configured) noexcept;

    const char* jfnk_preconditioner_refresh_reason_name(
        JfnkPreconditionerRefreshReason reason) noexcept;

    inline constexpr double kSparseChordAcceptanceTheta = 0.9;
    inline constexpr int kSparseChordConsecutiveStepLimit = 64;

    enum class SparseChordReuseAction {
        AttemptChord,
        AcceptChord,
        RefreshJacobian
    };

    enum class SparseChordReuseReason {
        WithinConsecutiveLimit,
        SufficientContraction,
        InsufficientContraction,
        ConsecutiveLimit,
        InvalidInput
    };

    // Call before applying a chord correction with candidate_evaluated false,
    // then after re-evaluating the residual with it true. The consecutive count
    // is the number of accepted chord steps since the retained factorization
    // was built.
    struct SparseChordReuseInput {
        bool candidate_evaluated = false;
        int consecutive_chord_steps = 0;
        double previous_error = 0.0;
        double candidate_error = 0.0;
    };

    struct SparseChordReuseDecision {
        SparseChordReuseAction action = SparseChordReuseAction::RefreshJacobian;
        SparseChordReuseReason reason = SparseChordReuseReason::InvalidInput;
    };

    SparseChordReuseDecision decide_sparse_chord_reuse(
        const SparseChordReuseInput& input) noexcept;

    // The fixed threshold is a policy boundary, not a runtime tuning control.
    bool sparse_chord_entry_allowed(double initial_error) noexcept;

    struct SolverRuntimeConfig {
        NewtonBackend backend = NewtonBackend::JfnkMumps;
        bool backend_explicitly_selected = false;
        bool sparse_parity_mask = true;
        // Certified exact-symmetry sector reduction for sparse-direct and
        // JFNK-MUMPS. Complete structural grading plus the entry-residual gate
        // selects the square block before J1; refusal retains the full-J
        // measurement-derived fallback.
        bool sparse_sector_reduce = true;
        // Factor the two y-parity sector blocks of a fused-mask Jacobian one
        // after the other instead of handing MUMPS both at once.  The blocks
        // are disconnected, so the elimination work is unchanged; the peak
        // factor residency becomes the larger block instead of their sum.
        bool sparse_parity_split_solve = false;
        MumpsRuntimeConfig mumps;
        JfnkMumpsRuntimeConfig jfnk_mumps;
        SolverDiagnosticsConfig diagnostics;

        static SolverRuntimeConfig from_environment();
        SolverRuntimeConfig with_stage_default_backend(NewtonBackend stage_default) const;
        SolverRuntimeConfig with_mumps_tree_cache_path(std::string path) const;
    };
} // namespace Kadath
