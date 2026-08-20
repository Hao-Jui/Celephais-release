#pragma once
// Shared resolution-stepping skeleton for the app drivers.
//
// Every app (NS, BNS, BH) runs the SAME continuation ladder: start at
// `init_res`, solve, then walk the resolution up to `final_res` in steps of
// +2, regridding and re-solving on each rung. This header single-sources that
// policy:
//
//   * the [init, final] ladder bootstrap + odd-resolution validation
//     (re-exported from Apps/Policy/app_resolution.hpp), and
//   * the +2 step increment (`next_resolution`), and
//   * a minimal `run_resolution_sequence(...)` driver for the apps whose
//     control flow is a clean solve/regrid loop.
//
// Apps whose stepping is fused into a per-stage field-reload state machine
// (NS/BH stage reloads) cannot express the loop as a plain callback pair
// without changing their convergence protocol; they instead use
// `make_resolution_ladder` + `next_resolution` directly so the ladder policy
// stays single-sourced while their state machine is preserved.

#include "Apps/Policy/app_resolution.hpp"

#include <utility>

namespace KadathApps {

// The continuation ladder steps the resolution by +2 each rung (Phillipe
// convention: only odd radial resolutions are well-formed). The next rung never
// overshoots `final_res`.
inline int next_resolution(int current_res, int final_res)
{
    const int stepped = current_res + 2;
    return stepped < final_res ? stepped : final_res;
}

// Minimal continuation driver for apps with a clean solve/regrid loop.
//
//   regrid_to_res(new_res): bring the fields onto a grid of resolution
//       `new_res` (rank-aware regrid + broadcast handled by the callback).
//   solve_at_current():     solve on the current grid.
//
// Pattern: solve at `init`, then for each +2 rung up to `final`, regrid then
// solve. `regrid_to_res` may signal early termination by returning true (e.g.
// a stop-after-regrid debug gate); in that case the loop returns immediately
// without the trailing solve.
template <typename RegridToRes, typename SolveAtCurrent>
void run_resolution_sequence(const Kadath::ResolutionLadder& ladder,
                             RegridToRes&& regrid_to_res,
                             SolveAtCurrent&& solve_at_current)
{
    solve_at_current();

    int current_res = ladder.init;
    while (current_res < ladder.final) {
        current_res = next_resolution(current_res, ladder.final);
        if (regrid_to_res(current_res))
            return;
        solve_at_current();
    }
}

} // namespace KadathApps
