#include <catch2/catch_test_macros.hpp>

#include "Apps/Workflow/resume_ladder.hpp"

// Warm-start floor for the continuation ladder. The bug these tests pin: a warm
// start whose ladder init sits below the loaded grid's resolution makes the
// first uniform regrid downsample (and discard) the saved solution before
// re-climbing. floor_ladder_to_loaded raises init to the loaded resolution so
// the ladder never starts below where the saved solution already is.

using KadathApps::floor_ladder_to_loaded;

TEST_CASE("resume ladder floors init at the loaded resolution", "[resume_ladder]")
{
    SECTION("loaded above requested init: resume in place, no downsample")
    {
        // The reported BNS case: converged res-11 grid, initial_resolution=5.
        // Before the fix the ladder was {5, 11} and the first regrid collapsed
        // res-11 -> res-7. The floor makes it a no-op {11, 11}.
        const auto ladder = floor_ladder_to_loaded(/*requested_init=*/5, /*final=*/11, /*loaded_res=*/11);
        CHECK(ladder.init == 11);
        CHECK(ladder.final == 11);
    }

    SECTION("loaded above a stale/hand-edited target: final raised to avoid downsample and error")
    {
        // res field hand-edited to 7 while the .dat is still res-11. init floors
        // to 11; final is raised to 11 so make_resolution_ladder does not throw
        // on init > final and no downsample to 7 happens.
        const auto ladder = floor_ladder_to_loaded(/*requested_init=*/5, /*final=*/7, /*loaded_res=*/11);
        CHECK(ladder.init == 11);
        CHECK(ladder.final == 11);
    }

    SECTION("loaded above init but below target: resume at loaded, climb to target")
    {
        const auto ladder = floor_ladder_to_loaded(/*requested_init=*/5, /*final=*/15, /*loaded_res=*/11);
        CHECK(ladder.init == 11);
        CHECK(ladder.final == 15);
    }

    SECTION("loaded below requested init: floor is inert, requested init honoured")
    {
        const auto ladder = floor_ladder_to_loaded(/*requested_init=*/9, /*final=*/15, /*loaded_res=*/7);
        CHECK(ladder.init == 9);
        CHECK(ladder.final == 15);
    }

    SECTION("loaded equals requested init and target: unchanged")
    {
        const auto ladder = floor_ladder_to_loaded(/*requested_init=*/9, /*final=*/9, /*loaded_res=*/9);
        CHECK(ladder.init == 9);
        CHECK(ladder.final == 9);
    }
}
