/*
    MUMPS out-of-core runtime policy for Celephais.

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include <cmath>

namespace Kadath
{

// Requested MUMPS factor-storage policy. Auto is retained through analysis so
// the wrapper can make its factor-time memory-budget decision before JOB=2.
enum class MumpsOutOfCoreMode
{
    Off,
    On,
    Auto,
};

inline constexpr double kMumpsOutOfCoreTouchDefault = 1.3;
inline constexpr double kMumpsOutOfCoreSafetyDefault = 0.7;
inline constexpr double kMumpsOutOfCoreBudgetUnset = -1.0;

struct MumpsOutOfCoreDecision
{
    double expected_mb_per_rank = 0.0;
    double node_expected_mb = 0.0;
    double budget_mb = 0.0;
    bool use_out_of_core = false;
    bool valid = false;
};

// Convert MUMPS' ICNTL(14)-inflated INFOG(16) reservation back to its base
// factor-memory prediction, then apply the measured resident-memory touch and
// compare the whole host node against its safety-adjusted available memory.
// Invalid or non-finite inputs fail closed to in-core with valid=false.
inline MumpsOutOfCoreDecision decide_mumps_out_of_core(
    double estimated_factor_memory_mb,
    int icntl14,
    double touch,
    double safety,
    int factor_ranks_on_host_node,
    double node_available_memory_mb) noexcept
{
    if (!std::isfinite(estimated_factor_memory_mb) ||
        estimated_factor_memory_mb < 0.0 || icntl14 < 0 ||
        !std::isfinite(touch) || touch <= 0.0 ||
        !std::isfinite(safety) || safety <= 0.0 ||
        factor_ranks_on_host_node <= 0 ||
        !std::isfinite(node_available_memory_mb) ||
        node_available_memory_mb < 0.0) {
        return {};
    }

    const double relaxation = 1.0 + static_cast<double>(icntl14) / 100.0;
    const double expected_mb_per_rank =
        estimated_factor_memory_mb / relaxation * touch;
    const double node_expected_mb =
        expected_mb_per_rank * static_cast<double>(factor_ranks_on_host_node);
    const double budget_mb = node_available_memory_mb * safety;
    if (!std::isfinite(relaxation) || relaxation <= 0.0 ||
        !std::isfinite(expected_mb_per_rank) ||
        !std::isfinite(node_expected_mb) || !std::isfinite(budget_mb)) {
        return {};
    }

    return {expected_mb_per_rank,
            node_expected_mb,
            budget_mb,
            node_expected_mb > budget_mb,
            true};
}

} // namespace Kadath
