#include "Linear_algebra/jacobian_group_planner.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>

namespace Kadath
{
    namespace
    {

        struct PairingBucket {
            ColumnClass column_class = ColumnClass::Unknown;
            int domain = -1;
            std::string variable_name;

            bool operator<(const PairingBucket& other) const
            {
                return std::tie(column_class, domain, variable_name) <
                       std::tie(other.column_class, other.domain, other.variable_name);
            }
        };

        bool column_can_use_bucket(const ColumnMetadata& column)
        {
            return column.column_class != ColumnClass::VarDomain && column.column_class != ColumnClass::Unknown;
        }

        PairingBucket make_bucket(const ColumnMetadata& column)
        {
            return {column.column_class, column.domain, column.var_name};
        }

        long long estimated_group_cost(std::size_t width, bool direct)
        {
            // The AD traversal dominates a column sweep, while lane export/threshold
            // work grows with width. These integer work units deliberately encode only
            // that ordering; the default-off planner must be benchmark-calibrated
            // before promotion. Direct columns bypass AD and are correspondingly cheap.
            constexpr long long ad_traversal_cost = 32;
            constexpr long long per_lane_cost = 1;
            constexpr long long direct_emit_cost = 1;
            return direct ? direct_emit_cost : ad_traversal_cost + per_lane_cost * static_cast<long long>(width);
        }

        void append_group(JacobianGlobalGroupPlan& plan, const std::vector<int>& columns, std::size_t begin,
                          std::size_t width, bool direct)
        {
            JacobianColumnGroup group;
            group.columns.assign(columns.begin() + static_cast<std::ptrdiff_t>(begin),
                                 columns.begin() + static_cast<std::ptrdiff_t>(begin + width));
            group.estimated_cost = estimated_group_cost(width, direct);
            group.direct = direct;
            plan.groups.push_back(std::move(group));
        }

    } // namespace

    JacobianGlobalGroupPlan build_global_jacobian_group_plan(const std::vector<ColumnMetadata>& metadata,
                                                             const std::vector<bool>& direct_columns,
                                                             const JacobianGroupPlannerOptions& options)
    {
        if (options.nproc <= 0)
            throw std::invalid_argument("Jacobian group planner requires nproc > 0");
        if (metadata.size() != direct_columns.size())
            throw std::invalid_argument("Jacobian group planner metadata/direct-column sizes differ");

        JacobianGlobalGroupPlan plan;
        std::map<PairingBucket, std::vector<int>> columns_by_bucket;
        std::vector<bool> grouped(metadata.size(), false);

        for (std::size_t column = 0; column < metadata.size(); ++column) {
            if (direct_columns[column] || !column_can_use_bucket(metadata[column]))
                continue;
            columns_by_bucket[make_bucket(metadata[column])].push_back(static_cast<int>(column));
            grouped[column] = true;
        }

        const std::array<std::pair<std::size_t, bool>, 5> widths = {{
            {32, options.use_wlane32},
            {16, options.use_wlane16},
            {8, options.use_wlane8},
            {4, options.use_wlane4},
            {2, options.use_wlane2},
        }};
        for (const auto& bucket : columns_by_bucket) {
            const std::vector<int>& columns = bucket.second;
            std::size_t begin = 0;
            for (const auto& width_and_enabled : widths) {
                const std::size_t width = width_and_enabled.first;
                if (!width_and_enabled.second)
                    continue;
                while (begin + width <= columns.size()) {
                    append_group(plan, columns, begin, width, false);
                    begin += width;
                }
            }
            while (begin < columns.size()) {
                append_group(plan, columns, begin, 1, false);
                ++begin;
            }
        }

        // Match the legacy clustered traversal: direct and non-bucketable columns
        // follow all bucketed columns in ascending global-column order.
        for (std::size_t column = 0; column < metadata.size(); ++column) {
            if (grouped[column])
                continue;
            const std::vector<int> singleton = {static_cast<int>(column)};
            append_group(plan, singleton, 0, 1, direct_columns[column]);
        }

        plan.group_indices_by_rank.resize(static_cast<std::size_t>(options.nproc));
        plan.estimated_cost_by_rank.assign(static_cast<std::size_t>(options.nproc), 0);

        std::vector<std::size_t> scheduling_order(plan.groups.size());
        std::iota(scheduling_order.begin(), scheduling_order.end(), 0);
        std::stable_sort(scheduling_order.begin(), scheduling_order.end(), [&](std::size_t lhs, std::size_t rhs) {
            return plan.groups[lhs].estimated_cost > plan.groups[rhs].estimated_cost;
        });

        for (std::size_t group_index : scheduling_order) {
            int owner = 0;
            for (int rank = 1; rank < options.nproc; ++rank) {
                if (plan.estimated_cost_by_rank[static_cast<std::size_t>(rank)] <
                    plan.estimated_cost_by_rank[static_cast<std::size_t>(owner)]) {
                    owner = rank;
                }
            }
            plan.groups[group_index].owner_rank = owner;
            plan.estimated_cost_by_rank[static_cast<std::size_t>(owner)] += plan.groups[group_index].estimated_cost;
        }

        // Re-scan semantic group order so each rank visits bucket/domain groups
        // contiguously even though ownership was chosen in LPT order.
        for (std::size_t group_index = 0; group_index < plan.groups.size(); ++group_index) {
            const int owner = plan.groups[group_index].owner_rank;
            plan.group_indices_by_rank[static_cast<std::size_t>(owner)].push_back(group_index);
        }
        return plan;
    }

} // namespace Kadath
