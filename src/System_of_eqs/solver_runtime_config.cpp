/*
    Runtime solver configuration for Celephais.

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "For_Kadath/System_of_eqs/solver_runtime_config.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>

namespace Kadath
{
    namespace
    {
        const char* getenv_nonempty(const char* name)
        {
            const char* value = std::getenv(name);
            return (value != nullptr && value[0] != '\0') ? value : nullptr;
        }

        bool env_flag_truthy(const char* name, bool default_value)
        {
            // Defer to the canonical case-insensitive parser (disabled iff the
            // whole value is one of 0/false/off/no). The previous first-char
            // test mis-read "off" as truthy ('o') — CELEPHAIS_TIMING=off enabled
            // timing.
            return env_flag_enabled(name, default_value);
        }

        MumpsOutOfCoreMode env_mumps_out_of_core_mode(
            MumpsOutOfCoreMode default_value)
        {
            const char* value = getenv_nonempty("MUMPS_OOC");
            if (value == nullptr)
                return default_value;
            const std::string mode(value);
            if (mode == "0")
                return MumpsOutOfCoreMode::Off;
            if (mode == "1")
                return MumpsOutOfCoreMode::On;
            if (mode == "auto")
                return MumpsOutOfCoreMode::Auto;
            return default_value;
        }

        int env_int(const char* name, int default_value, int min_value, int max_value)
        {
            const char* value = getenv_nonempty(name);
            if (value == nullptr)
                return default_value;

            errno = 0;
            char* end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (errno == 0 && end != value && *end == '\0' &&
                parsed >= min_value && parsed <= max_value &&
                parsed >= std::numeric_limits<int>::min() &&
                parsed <= std::numeric_limits<int>::max())
                return static_cast<int>(parsed);
            return default_value;
        }

        int env_nonnegative_int_or_flag(const char* name, int default_value)
        {
            const char* value = getenv_nonempty(name);
            if (value == nullptr) {
                return default_value;
            }
            errno = 0;
            char* endp = nullptr;
            const long parsed = std::strtol(value, &endp, 10);
            if (endp != value) {
                if (errno != 0 || *endp != '\0' ||
                    parsed > std::numeric_limits<int>::max())
                    return default_value;
                return static_cast<int>(parsed < 0 ? 0 : parsed);
            }
            return env_flag_truthy(name, default_value != 0) ? 1 : 0;
        }

        double env_positive_double(const char* name, double default_value)
        {
            const char* value = getenv_nonempty(name);
            if (value == nullptr) {
                return default_value;
            }
            try {
                const double parsed = std::stod(value);
                if (parsed > 0.0) {
                    return parsed;
                }
            } catch (...) {
            }
            return default_value;
        }

        double env_nonnegative_double(const char* name, double default_value)
        {
            const char* value = getenv_nonempty(name);
            if (value == nullptr)
                return default_value;
            try {
                const double parsed = std::stod(value);
                if (std::isfinite(parsed) && parsed >= 0.0)
                    return parsed;
            } catch (...) {
            }
            return default_value;
        }

        double env_ooc_double(const char* name, double default_value,
                              bool allow_zero)
        {
            const char* value = getenv_nonempty(name);
            if (value == nullptr)
                return default_value;
            try {
                std::size_t consumed = 0;
                const std::string input(value);
                const double parsed = std::stod(input, &consumed);
                if (consumed == input.size() && std::isfinite(parsed) &&
                    (allow_zero ? parsed >= 0.0 : parsed > 0.0)) {
                    return parsed;
                }
            } catch (...) {
            }
            return default_value;
        }

        NewtonBackend env_backend(bool& explicitly_selected)
        {
            explicitly_selected = false;
            const char* value = getenv_nonempty("CELEPHAIS_SOLVER");
            if (value == nullptr) {
                return NewtonBackend::JfnkMumps;
            }
            const std::string solver(value);
            if (solver == "dense") {
                explicitly_selected = true;
                return NewtonBackend::Dense;
            }
            if (solver == "mumps") {
                explicitly_selected = true;
                return NewtonBackend::Mumps;
            }
            if (solver == "jfnk-mumps" || solver == "jfnk_mumps") {
                explicitly_selected = true;
                return NewtonBackend::JfnkMumps;
            }
            if (solver == "jfnk-dense" || solver == "jfnk_dense") {
                explicitly_selected = true;
                return NewtonBackend::JfnkDense;
            }
            if (solver == "jfnk-schur" || solver == "jfnk_schur") {
                explicitly_selected = true;
                return NewtonBackend::JfnkSchur;
            }
            return NewtonBackend::JfnkMumps;
        }

        struct SparseParityPermissions
        {
            bool mask = false;
            bool split = false;
            bool reduce = false;
        };

        SparseParityPermissions parse_sparse_parity_permissions(
            const std::string& rung)
        {
            if (rung == "off")
                return {false, false, false};
            if (rung == "mask")
                return {true, false, false};
            if (rung == "split")
                return {true, true, false};
            if (rung == "reduce")
                return {true, true, true};
            KADATH_THROW("SPARSE_PARITY has invalid value '" + rung +
                         "'; expected one of: off, mask, split, reduce");
        }

        void apply_sparse_parity_permissions(SolverRuntimeConfig& config)
        {
            const char* value = std::getenv("SPARSE_PARITY");
            if (value == nullptr)
                return;

            const std::string rung(value);
            const SparseParityPermissions permissions =
                parse_sparse_parity_permissions(rung);
            const auto reject_conflict =
                [&](const char* alias, bool alias_enabled, bool expected) {
                    const char* alias_value = std::getenv(alias);
                    if (alias_value != nullptr && alias_enabled != expected) {
                        KADATH_THROW("SPARSE_PARITY=" + rung +
                                     " conflicts with " + alias + "=" +
                                     alias_value);
                    }
                };
            reject_conflict("SPARSE_PARITY_MASK", config.sparse_parity_mask,
                            permissions.mask);
            reject_conflict("SPARSE_PARITY_SPLIT_SOLVE",
                            config.sparse_parity_split_solve,
                            permissions.split);
            reject_conflict("SPARSE_SECTOR_REDUCE",
                            config.sparse_sector_reduce,
                            permissions.reduce);

            config.sparse_parity_mask = permissions.mask;
            config.sparse_parity_split_solve = permissions.split;
            config.sparse_sector_reduce = permissions.reduce;
        }
    } // namespace

    SparseDirectDropPolicy resolve_sparse_direct_drop_policy(
        const MumpsRuntimeConfig& config,
        double nonlinear_error,
        int dimension,
        SparseDirectDropState& state) noexcept
    {
        constexpr double kFallbackDropTol = 1e-8;
        constexpr double kDropTolMin = 1e-16;
        const double configured_drop_tol =
            std::isfinite(config.drop_tol) && config.drop_tol > 0.0
                ? config.drop_tol
                : kFallbackDropTol;
        const bool state_matches =
            dimension > 0 && state.dimension == dimension &&
            state.configured_drop_tol == configured_drop_tol &&
            std::isfinite(state.frozen_numerical_drop_tol) &&
            state.frozen_numerical_drop_tol > 0.0;
        if (!state_matches) {
            const double finite_error =
                std::isfinite(nonlinear_error) && nonlinear_error > 0.0
                    ? nonlinear_error
                    : 0.0;
            state.dimension = dimension;
            state.configured_drop_tol = configured_drop_tol;
            state.frozen_numerical_drop_tol = std::max(
                kDropTolMin,
                configured_drop_tol * std::sqrt(std::sqrt(finite_error)));
        }
        const double numerical_drop_tol = state.frozen_numerical_drop_tol;
        const double pattern_drop_tol =
            std::isfinite(config.sparse_pattern_drop_tol) &&
                    config.sparse_pattern_drop_tol >= 0.0
                ? std::min(numerical_drop_tol, config.sparse_pattern_drop_tol)
                : numerical_drop_tol;
        return {numerical_drop_tol, pattern_drop_tol};
    }

    JfnkPreconditionerRefreshDecision decide_jfnk_preconditioner_refresh(
        const JfnkPreconditionerRefreshInput& input) noexcept
    {
        using Reason = JfnkPreconditionerRefreshReason;
        const auto refresh = [](Reason reason, double projected_seconds = 0.0) {
            return JfnkPreconditionerRefreshDecision{
                true, false, reason, projected_seconds};
        };
        const auto reuse = [](Reason reason, bool deferred, double projected_seconds = 0.0) {
            return JfnkPreconditionerRefreshDecision{
                false, deferred, reason, projected_seconds};
        };

        if (!input.has_preconditioner)
            return refresh(Reason::InitialBuild);
        if (input.recovery_retry)
            return refresh(Reason::RecoveryRetry);
        if (input.nonlinear_error_grew)
            return refresh(Reason::NonlinearGrowth);

        const int fixed_steps = std::max(input.fixed_refresh_steps, 1);
        if (input.preconditioner_age < fixed_steps)
            return reuse(Reason::ReuseBeforeCadence, false);
        if (!input.adaptive_enabled)
            return refresh(Reason::FixedCadence);

        const int adaptive_max = std::max(input.adaptive_max_steps, fixed_steps);
        if (input.preconditioner_age >= adaptive_max)
            return refresh(Reason::AdaptiveHardLimit);

        // A stale factor is extended only after a usable Krylov correction
        // produced material nonlinear progress. The default line search makes
        // this a measured property of the accepted step rather than a guess
        // based on linear iterations alone.
        if (!input.previous_krylov_usable)
            return refresh(Reason::AdaptiveKrylovFailure);
        if (!std::isfinite(input.nonlinear_error_ratio) ||
            input.nonlinear_error_ratio <= 0.0 ||
            input.nonlinear_error_ratio > 0.8)
            return refresh(Reason::AdaptivePoorProgress);
        if (!std::isfinite(input.previous_krylov_seconds) ||
            !std::isfinite(input.previous_refresh_seconds) ||
            input.previous_krylov_seconds <= 0.0 ||
            input.previous_refresh_seconds <= 0.0 ||
            input.previous_krylov_iterations <= 0 ||
            input.max_krylov_iterations <= 0)
            return refresh(Reason::AdaptiveInsufficientEvidence);

        // Estimate the next stale solve at twice the observed Krylov count,
        // allowing at least four extra iterations. Scale the measured solve by
        // the square of the Krylov-depth ratio: this bounds the linear
        // matvec/preconditioner work with the steeper unrestarted-GMRES
        // orthogonalization/back-substitution envelope. Deferral wins only if
        // that cost projection is still below a measured rebuild+factor wall.
        const long long previous_iterations = input.previous_krylov_iterations;
        const long long projected_iterations_wide = std::min<long long>(
            input.max_krylov_iterations,
            std::max(previous_iterations + 4, 2 * previous_iterations));
        const int projected_iterations =
            static_cast<int>(projected_iterations_wide);
        const double krylov_depth_ratio =
            static_cast<double>(projected_iterations) /
            input.previous_krylov_iterations;
        const double projected_seconds =
            input.previous_krylov_seconds * krylov_depth_ratio * krylov_depth_ratio;
        if (!std::isfinite(projected_seconds) ||
            projected_seconds >= input.previous_refresh_seconds)
            return refresh(Reason::AdaptiveKrylovCost, projected_seconds);

        return reuse(Reason::AdaptiveReuse, true, projected_seconds);
    }

    bool should_replay_mumps_tree_cache(
        bool cache_enabled,
        bool replay_disabled_for_stage,
        bool refresh_requested,
        bool had_preconditioner,
        bool path_configured) noexcept
    {
        return cache_enabled && !replay_disabled_for_stage &&
               refresh_requested && had_preconditioner && path_configured;
    }

    const char* jfnk_preconditioner_refresh_reason_name(
        JfnkPreconditionerRefreshReason reason) noexcept
    {
        switch (reason) {
        case JfnkPreconditionerRefreshReason::InitialBuild:
            return "initial-build";
        case JfnkPreconditionerRefreshReason::RecoveryRetry:
            return "recovery-retry";
        case JfnkPreconditionerRefreshReason::NonlinearGrowth:
            return "nonlinear-growth";
        case JfnkPreconditionerRefreshReason::FixedCadence:
            return "fixed-cadence";
        case JfnkPreconditionerRefreshReason::AdaptiveHardLimit:
            return "adaptive-hard-limit";
        case JfnkPreconditionerRefreshReason::AdaptiveInsufficientEvidence:
            return "adaptive-insufficient-evidence";
        case JfnkPreconditionerRefreshReason::AdaptivePoorProgress:
            return "adaptive-poor-progress";
        case JfnkPreconditionerRefreshReason::AdaptiveKrylovFailure:
            return "adaptive-krylov-failure";
        case JfnkPreconditionerRefreshReason::AdaptiveKrylovCost:
            return "adaptive-krylov-cost";
        case JfnkPreconditionerRefreshReason::ReuseBeforeCadence:
            return "reuse-before-cadence";
        case JfnkPreconditionerRefreshReason::AdaptiveReuse:
            return "adaptive-reuse";
        }
        return "unknown";
    }

    SparseChordReuseDecision decide_sparse_chord_reuse(
        const SparseChordReuseInput& input) noexcept
    {
        using Action = SparseChordReuseAction;
        using Reason = SparseChordReuseReason;

        if (input.consecutive_chord_steps < 0)
            return {Action::RefreshJacobian, Reason::InvalidInput};
        if (!std::isfinite(input.previous_error) || input.previous_error < 0.0)
            return {Action::RefreshJacobian, Reason::InvalidInput};
        if (input.consecutive_chord_steps >=
            kSparseChordConsecutiveStepLimit)
            return {Action::RefreshJacobian, Reason::ConsecutiveLimit};
        if (!input.candidate_evaluated)
            return {Action::AttemptChord, Reason::WithinConsecutiveLimit};

        if (!std::isfinite(input.candidate_error) ||
            input.candidate_error < 0.0)
            return {Action::RefreshJacobian, Reason::InvalidInput};
        if (input.candidate_error <
            kSparseChordAcceptanceTheta * input.previous_error)
            return {Action::AcceptChord, Reason::SufficientContraction};
        return {Action::RefreshJacobian, Reason::InsufficientContraction};
    }

    bool sparse_chord_entry_allowed(double initial_error) noexcept
    {
        return std::isfinite(initial_error) && initial_error >= 0.0 &&
               initial_error < kSparseChordEntryTolerance;
    }

    SolverRuntimeConfig SolverRuntimeConfig::from_environment()
    {
        SolverRuntimeConfig config;
        config.backend = env_backend(config.backend_explicitly_selected);
        config.sparse_parity_mask =
            env_flag_zero_opt_out("SPARSE_PARITY_MASK");
        config.sparse_sector_reduce =
            env_flag_zero_opt_out("SPARSE_SECTOR_REDUCE");
        config.sparse_parity_split_solve =
            env_flag_enabled("SPARSE_PARITY_SPLIT_SOLVE",
                             config.sparse_parity_split_solve);
        apply_sparse_parity_permissions(config);
        config.mumps.drop_tol = env_positive_double("DROP_TOL", config.mumps.drop_tol);
        config.mumps.ordering = env_int("MUMPS_ORDERING", config.mumps.ordering, 0, 7);
        config.mumps.out_of_core =
            env_mumps_out_of_core_mode(config.mumps.out_of_core);
        config.mumps.out_of_core_touch =
            env_ooc_double("MUMPS_OOC_TOUCH",
                           config.mumps.out_of_core_touch, false);
        config.mumps.out_of_core_safety =
            env_ooc_double("MUMPS_OOC_SAFETY",
                           config.mumps.out_of_core_safety, false);
        config.mumps.out_of_core_budget_mb =
            env_ooc_double("MUMPS_OOC_BUDGET_MB",
                           config.mumps.out_of_core_budget_mb, true);
        config.mumps.blr = env_nonnegative_int_or_flag("MUMPS_BLR", config.mumps.blr);
        // Default OFF; opt in with MUMPS_TREE_CACHE=1 (any value other
        // than 0/false/off/no enables).
        config.mumps.tree_cache_enabled =
            env_flag_enabled("MUMPS_TREE_CACHE");
        // Unset -> the struct default -1 (resolve to max(floor(local_ranks/4), 1) at
        // factor time); 0 = all ranks; N = fixed N/node. env_int returns the default
        // for unset or non-numeric input, so a stale "auto" harmlessly falls through.
        config.mumps.ranks_per_node =
            env_int("MUMPS_RANKS_PER_NODE", config.mumps.ranks_per_node, 0, 4096);
        config.mumps.sparse_analyze_reuse =
            env_flag_enabled("SPARSE_MUMPS_ANALYZE_REUSE",
                             config.mumps.sparse_analyze_reuse);
        config.mumps.sparse_chord_reuse =
            env_flag_zero_opt_out("SPARSE_CHORD_REUSE");
        config.mumps.sparse_pattern_drop_tol =
            env_nonnegative_double("SPARSE_MUMPS_PATTERN_DROP_TOL",
                                   config.mumps.sparse_pattern_drop_tol);
        config.mumps.sparse_superset_max_nnz_ratio =
            env_positive_double("SPARSE_MUMPS_SUPERSET_MAX_NNZ_RATIO",
                                config.mumps.sparse_superset_max_nnz_ratio);
        config.jfnk_mumps.max_linear_iterations =
            env_int("JFNK_MAX_ITERS", config.jfnk_mumps.max_linear_iterations, 1, 4096);
        config.jfnk_mumps.preconditioner_refresh_steps =
            env_int("JFNK_MUMPS_PC_REFRESH",
                    config.jfnk_mumps.preconditioner_refresh_steps,
                    1,
                    4096);
        config.jfnk_mumps.adaptive_preconditioner_refresh =
            env_flag_enabled("JFNK_MUMPS_PC_ADAPTIVE",
                             config.jfnk_mumps.adaptive_preconditioner_refresh);
        config.jfnk_mumps.adaptive_preconditioner_max_steps =
            env_int("JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS",
                    config.jfnk_mumps.adaptive_preconditioner_max_steps,
                    1,
                    4096);
        config.jfnk_mumps.adaptive_preconditioner_max_steps =
            std::max(config.jfnk_mumps.adaptive_preconditioner_max_steps,
                     config.jfnk_mumps.preconditioner_refresh_steps);
        config.jfnk_mumps.linear_relative_tolerance =
            env_positive_double("JFNK_RTOL", config.jfnk_mumps.linear_relative_tolerance);
        config.jfnk_mumps.line_search =
            env_flag_enabled("JFNK_LINESEARCH", config.jfnk_mumps.line_search);
        config.diagnostics.timing = env_flag_enabled("CELEPHAIS_TIMING", config.diagnostics.timing);
        config.diagnostics.def_filter = env_flag_enabled("DEF_FILTER", config.diagnostics.def_filter);
        if (const char* capture_path =
                getenv_nonempty("DIRECT_REPLAY_CAPTURE")) {
            config.diagnostics.direct_replay_capture_path = capture_path;
        }
        config.diagnostics.direct_replay_capture_ordinal =
            env_int("DIRECT_REPLAY_CAPTURE_ORDINAL",
                    config.diagnostics.direct_replay_capture_ordinal,
                    1,
                    std::numeric_limits<int>::max());
        return config;
    }

    SolverRuntimeConfig SolverRuntimeConfig::with_stage_default_backend(
        NewtonBackend stage_default) const
    {
        SolverRuntimeConfig config = *this;
        if (!config.backend_explicitly_selected)
            config.backend = stage_default;
        return config;
    }

    SolverRuntimeConfig SolverRuntimeConfig::with_mumps_tree_cache_path(
        std::string path) const
    {
        SolverRuntimeConfig config = *this;
        config.mumps.tree_cache_path = std::move(path);
        return config;
    }
} // namespace Kadath
