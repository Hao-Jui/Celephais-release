#pragma once
// Warm-start floor for the resolution continuation ladder.
//
// The continuation ladder (Apps/Policy/app_resolution.hpp) walks the radial
// resolution from `init` up to `final` in +2 rungs, uniform-regridding between
// rungs. On a COLD start that is correct: seed at a low `init`, climb to the
// target. On a WARM start (resuming from an existing .dat) it is a trap: if
// `init` is below the resolution the loaded solution already carries, the first
// uniform regrid DOWNSAMPLES the saved grid -- discarding any per-domain AMR
// refinement -- only to re-climb back up. (Observed on BNS: a converged res-11
// AMR grid warm-started with initial_resolution=5 collapses to uniform res-7
// before AMR re-runs from scratch.)
//
// make_resume_ladder floors `init` at the loaded grid's resolution so a warm
// start never begins below where the saved solution already is. `final` is
// raised to match when the loaded grid already exceeds the configured target,
// so the ladder degrades to a no-op (resume in place) instead of erroring.

#include "Apps/Policy/app_resolution.hpp"
#include "For_Kadath/IO/be_file_source.hpp"

#include <algorithm>
#include <filesystem>
#include <string>

namespace KadathApps {

// Highest radial point count across all domains of the space serialized in
// `spacein`. Equals domain 0's radial resolution on a uniform space; on an AMR
// p-refined (mixed-resolution) space it is the largest radial count any domain
// carries -- the resolution the ladder must not start below.
template <typename space_t>
int loaded_space_max_radial(const std::string& spacein)
{
    Kadath::BeFileSource source(spacein);
    space_t space(source);
    int highest = 0;
    for (int d = 0; d < space.get_nbr_domains(); ++d)
        highest = std::max(highest, space.get_domain(d)->get_nbr_points()(0));
    return highest;
}

// Pure floor arithmetic (no I/O): raise `requested_init` to at least
// `loaded_res`, and `final` to at least the resulting init, then build the
// validated ladder. Factored out of make_resume_ladder so the floor logic is
// unit-testable without a space file on disk.
inline Kadath::ResolutionLadder floor_ladder_to_loaded(int requested_init, int final, int loaded_res)
{
    const int init = std::max(requested_init, loaded_res);
    return Kadath::make_resolution_ladder(init, std::max(final, init));
}

// Continuation ladder with the warm-start floor applied. On a cold start
// (warm_start == false) this is the plain init->final ladder. On a warm start
// it reads the loaded grid's resolution from `bconfig.space_filename()` and
// raises `init` to it (and `final` to at least `init`), so the ladder never
// downsamples the solution it is resuming from.
template <typename space_t, typename config_t>
Kadath::ResolutionLadder make_resume_ladder(config_t& bconfig, int requested_init, int final, bool warm_start)
{
    // The floor only applies to a genuine warm start: a resume flag with no
    // backing .dat (the downstream solve would fail to load it anyway) falls
    // back to the plain cold-start ladder rather than reading a missing file.
    const std::string spacein = bconfig.space_filename();
    if (!warm_start || !std::filesystem::exists(spacein))
        return Kadath::make_resolution_ladder(requested_init, final);

    const int loaded = loaded_space_max_radial<space_t>(spacein);
    return floor_ladder_to_loaded(requested_init, final, loaded);
}

} // namespace KadathApps
