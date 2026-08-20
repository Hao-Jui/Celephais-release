#pragma once

#include "For_Kadath/System_of_eqs/Jacobian/column_types.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace Kadath
{

    struct JacobianGroupPlannerOptions {
        int nproc = 1;
        bool use_wlane32 = true;
        bool use_wlane16 = true;
        bool use_wlane8 = true;
        bool use_wlane4 = true;
        bool use_wlane2 = true;
    };

    struct JacobianColumnGroup {
        std::vector<int> columns;
        int owner_rank = -1;
        long long estimated_cost = 0;
        bool direct = false;

        bool operator==(const JacobianColumnGroup&) const = default;
    };

    struct JacobianGlobalGroupPlan {
        // Groups remain in bucket/domain order. Assignment uses a separate
        // deterministic longest-processing-time pass, so per-rank group lists can
        // preserve domain locality without giving up cost-aware balancing.
        std::vector<JacobianColumnGroup> groups;
        std::vector<std::vector<std::size_t>> group_indices_by_rank;
        std::vector<long long> estimated_cost_by_rank;

        bool operator==(const JacobianGlobalGroupPlan&) const = default;
    };

    // Build W32 -> W16 -> W8 -> W4 -> W2 groups globally, before MPI ownership.
    // Direct and non-bucketable columns remain singleton groups. Every group is
    // assigned whole to exactly one rank by deterministic LPT scheduling.
    JacobianGlobalGroupPlan build_global_jacobian_group_plan(const std::vector<ColumnMetadata>& metadata,
                                                             const std::vector<bool>& direct_columns,
                                                             const JacobianGroupPlannerOptions& options);

} // namespace Kadath
