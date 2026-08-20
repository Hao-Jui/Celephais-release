#pragma once
// Shared stage-dispatch loop for the binary XCTS solvers (BNS + BHNS).
//
// Binary drivers run an ordered set of stages, each gated by [stages] and each
// aborting the solve on a non-EXIT_SUCCESS return. The caller supplies the
// physics order as a list of {STAGE, thunk}. This helper owns only the part that
// is identical and load-bearing: walk the list IN ORDER, skip disabled stages,
// run the enabled ones, and throw on failure.

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Config/config_enums.hpp"

#include <array>
#include <functional>
#include <initializer_list>
#include <utility>

namespace Kadath {

// Run the given stages in list order: each enabled stage's thunk is invoked and
// must return EXIT_SUCCESS or the solve aborts. Returns the exit status of the
// last stage that ran (EXIT_SUCCESS if none ran). The caller owns any banner,
// stage-gate validation, and barriers.
inline int run_binary_stage_sequence(const std::array<bool, NUM_STAGES_V>& stage_enabled,
                                     std::initializer_list<std::pair<STAGE, std::function<int()>>> stages)
{
    int exit_status = EXIT_SUCCESS;
    for (const auto& [stage, run] : stages) {
        if (!stage_enabled[to_int(stage)])
            continue;
        exit_status = run();
        if (exit_status != EXIT_SUCCESS)
            KADATH_THROW("Stage failed");
    }
    return exit_status;
}

} // namespace Kadath
