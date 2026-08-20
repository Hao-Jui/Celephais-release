#pragma once

#include "For_Kadath/Utilities/runtime_env.hpp"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace Kadath
{
    namespace matching_lane_detail
    {
        // Preserve all-or-nothing ownership transfer: finalization may allocate
        // and throw, so no raw pointer escapes until every owned output has
        // completed that potentially-throwing work.
        template <typename Output, typename Finalizer>
        void finalize_then_release(std::vector<std::unique_ptr<Output>>& outputs,
                                   Output** raw_outputs, Finalizer&& finalize)
        {
            auto&& finalize_output = finalize;
            for (std::size_t lane = 0; lane < outputs.size(); ++lane)
                finalize_output(*outputs[lane], lane);
            for (std::size_t lane = 0; lane < outputs.size(); ++lane)
                raw_outputs[lane] = outputs[lane].release();
        }
    } // namespace matching_lane_detail

    struct MatchingLaneStats {
        long long export_native_calls = 0;
        long long export_scalar_fallback_calls = 0;
        long long export_missing_lanes = 0;
        long long import_native_calls = 0;
        long long import_scalar_fallback_calls = 0;
        long long import_refusals = 0;
        long long import_plan_points = 0;
        long long import_missing_inputs = 0;
        long long import_plan_cache_hits = 0;
        long long import_plan_cache_misses = 0;
        long long import_plan_cache_rebuilds = 0;
    };

    inline MatchingLaneStats& matching_lane_stats_state()
    {
        thread_local MatchingLaneStats stats;
        return stats;
    }

    inline MatchingLaneStats matching_lane_stats()
    {
        return matching_lane_stats_state();
    }

    inline void reset_matching_lane_stats()
    {
        matching_lane_stats_state() = {};
    }

    inline bool matching_lane_export_enabled()
    {
        return env_flag_enabled("MATCHING_LANE_EXPORT", true);
    }

    inline bool matching_import_lane_batch_enabled()
    {
        return env_flag_enabled("MATCHING_IMPORT_LANE_BATCH", true);
    }
} // namespace Kadath
