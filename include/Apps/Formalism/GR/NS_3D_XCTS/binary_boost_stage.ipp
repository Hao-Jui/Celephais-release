// Body fragment #included via stage_helper.cpp (which is included
// by solver.hpp). Not a standalone translation unit.
//
// The boost-stage body is shared across the sym formalisms in
// Apps/Formalism/Shared/Stages/ns_3d_xcts_binary_boost_stage.hpp; this wrapper binds
// the GR specialization: plain XCTS field equations, no scalar field.
#include "mpi.h"
#include "Apps/Bco_utils/bco_io.hpp"

template <class eos_t, typename config_t, typename space_t>
int ns_3d_xcts_solver<eos_t, config_t, space_t>::binary_boost_stage(kadath_config<BIN_INFO>& binconfig, NODES bco)
{
    return ns_3d_xcts_run_binary_boost_stage</*WithScalar=*/false>(
        *this, binconfig, bco, "Binary boosted NS",
        /*emit_advection_defs=*/[](System_of_eqs&) {},
        /*emit_scalar_setup=*/[](System_of_eqs&) {},
        /*emit_field_equations=*/
        [](System_of_eqs& syst, int d, bool has_matter, bool has_shift) {
            gr_xcts::add_xcts_field_equations(syst, d, has_matter, has_shift);
        },
        /*emit_scalar_global_eqs=*/[](System_of_eqs&) {});
}
