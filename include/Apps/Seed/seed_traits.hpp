#pragma once
/*
 * ObjectSeeder<s_type, space_t, theory_t> — extension point for BCO initial-data generation.
 *
 * To add a new (object × space × theory) combination, specialize ObjectSeeder in a new
 * header under bco_seed/<Theory>/ and include it from the relevant solver's regrid file.
 *
 * The free function Kadath::Seed::setup_co<s_type, space_t>() is the public entry point.
 * It lives in the Kadath::Seed namespace to avoid shadowing the legacy ::setup_co<s_type>
 * template (still used by 2-D polar solvers with their own space types).
 */
#include "For_Kadath/Config/config_bco.hpp"

namespace Kadath::Seed {

struct GR {};

// Primary template — a static_assert fires if no specialization is found.
// This catches missing includes at compile time rather than at link time.
template <NODES s_type, typename space_t, typename theory_t = GR>
struct ObjectSeeder {
    template <typename config_t>
    static void execute(config_t&) {
        static_assert(sizeof(config_t) == 0,
            "No ObjectSeeder specialization for this (s_type, space_t, theory_t) triple. "
            "Add the matching #include \"Apps/Seed/GR/<object>_seed.hpp\" "
            "(or the relevant theory header) to your solver's regrid file.");
    }
};

// Public entry point.  Explicit template args: s_type + space_t required; theory_t defaults to GR.
// config_t is deduced from the argument.
template <NODES s_type, typename space_t, typename config_t, typename theory_t = GR>
void setup_co(config_t& bconfig, bool use_config_vars = false) {
    ObjectSeeder<s_type, space_t, theory_t>::execute(bconfig, use_config_vars);
}

} // namespace Kadath::Seed
