#pragma once

#include "Apps/Workflow/stage_outcome.hpp"

#include <string>
#include <utility>

namespace KadathApps {

enum class DatasetResumeEvent {
    Missing,
    ReusedCurrentDataset,
    LoadedDifferentDataset
};

struct DatasetResumeOutcome {
    DatasetResumeEvent event{DatasetResumeEvent::Missing};
    std::string before;
    std::string after;

    bool found() const
    {
        return event != DatasetResumeEvent::Missing;
    }

    bool requires_field_reload() const
    {
        return event == DatasetResumeEvent::LoadedDifferentDataset;
    }

    StageOutcome as_stage_outcome() const
    {
        switch (event) {
            case DatasetResumeEvent::ReusedCurrentDataset:
                return reused_current_dataset();
            case DatasetResumeEvent::LoadedDifferentDataset:
                return loaded_different_dataset();
            case DatasetResumeEvent::Missing:
                return completed_stage();
        }
        return completed_stage(EXIT_FAILURE);
    }
};

inline DatasetResumeOutcome missing_dataset(std::string before)
{
    return {DatasetResumeEvent::Missing, std::move(before), {}};
}

inline DatasetResumeOutcome resumed_dataset(std::string before, std::string after)
{
    const bool same_dataset = before == after;
    return {same_dataset ? DatasetResumeEvent::ReusedCurrentDataset
                         : DatasetResumeEvent::LoadedDifferentDataset,
            std::move(before),
            std::move(after)};
}

} // namespace KadathApps
