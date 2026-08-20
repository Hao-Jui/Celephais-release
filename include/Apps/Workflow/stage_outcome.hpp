#pragma once

#include <cstdlib>

namespace KadathApps {

// Facts reported by a stage or driver. Workflow code decides what to do with
// these facts: reload fields, continue to a boost, advance resolution, or stop.
enum class StageEvent {
    Completed,
    ReusedCurrentDataset,
    LoadedDifferentDataset,
    RequestBinaryBoost,
    Failed
};

struct StageOutcome {
    StageEvent event{StageEvent::Completed};
    int code{EXIT_SUCCESS};

    bool requires_field_reload() const
    {
        return event == StageEvent::LoadedDifferentDataset;
    }

    bool requests_binary_boost() const
    {
        return event == StageEvent::RequestBinaryBoost;
    }

    bool completed_without_reload() const
    {
        return event == StageEvent::Completed ||
               event == StageEvent::ReusedCurrentDataset;
    }

    bool failed() const
    {
        return event == StageEvent::Failed;
    }
};

inline StageOutcome completed_stage(int code = EXIT_SUCCESS)
{
    return {code == EXIT_SUCCESS ? StageEvent::Completed : StageEvent::Failed, code};
}

inline StageOutcome reused_current_dataset()
{
    return {StageEvent::ReusedCurrentDataset, EXIT_SUCCESS};
}

inline StageOutcome loaded_different_dataset()
{
    return {StageEvent::LoadedDifferentDataset, EXIT_SUCCESS};
}

inline StageOutcome request_binary_boost()
{
    return {StageEvent::RequestBinaryBoost, EXIT_SUCCESS};
}

} // namespace KadathApps
