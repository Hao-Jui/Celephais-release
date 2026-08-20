// Body fragment #included via stage_helper.cpp (which is included
// by solver.hpp). Not a standalone translation unit.
// Stage body shared across formalisms: Apps/Formalism/Shared/Stages/ns_3d_xcts_solve_stages.hpp.
#include "mpi.h"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Formalism/Shared/Stages/ns_3d_xcts_solve_stages.hpp"

template <class eos_t, typename config_t, typename space_t>
int ns_3d_xcts_solver<eos_t, config_t, space_t>::uniform_rot_stage()
{
    const SolverRuntimeConfig solver_config = SolverRuntimeConfig::from_environment();
    return ns_3d_xcts_run_uniform_rot_stage</*WithScalar=*/false>(
        *this, "Rotating",
        "Stilde = 3 * press * delta + press * h * Wsquare * Usquare",
        /*print_initial_residual_norm=*/false, solver_config,
        [](System_of_eqs&) {},
        [](System_of_eqs& syst, int d, bool has_matter, bool has_shift) {
            gr_xcts::add_xcts_field_equations(syst, d, has_matter, has_shift);
        },
        [](System_of_eqs&) {});
}
