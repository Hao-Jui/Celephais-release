#pragma once
// Shared continuation-driver loop for the isolated compact-object solvers
// (isolated NS in ns_resolution_sequence.hpp, isolated BH in
// BH_3D_XCTS/solver_imp.ipp). Both walk the identical skeleton:
//
//   exit_status = loaded_different_dataset();           // prime the first pass
//   while (reload || (allow_binary_boost && boost)) {
//       exit_status = <per-driver body>(bconfig, exit_status);
//       MPI_Barrier(MPI_COMM_WORLD);
//   }
//   return exit_status;
//
// The per-driver body — load the saved fields, construct the solver, dispatch
// the enabled stages, manage stage flags, regrid on completion, and (BH only)
// run the inline binary boost — is supplied by the `step` callback and stays in
// each driver where it is convergence-critical and individually verifiable.
//
// This template owns only the loop discipline that is easy to get subtly wrong:
// the while-condition (including whether binary boost re-enters the same loop,
// which is true for BH and false for the isolated NS — NS runs its boost in a
// separate post-ladder driver) and the once-per-iteration barrier.

#include "mpi.h"
#include "Apps/Helper/solver_base.hpp"      // Kadath::status_requires_field_reload / _requests_binary_boost
#include "Apps/Workflow/stage_outcome.hpp"  // KadathApps::loaded_different_dataset

namespace KadathApps {

// `step` is invoked as step(bconfig, prev_exit_status) and returns the new
// exit_status that gates the next iteration. `allow_binary_boost` controls
// whether a RequestBinaryBoost status re-enters the loop (BH: true, inline boost)
// or terminates it (isolated NS: false, boost handled by a separate driver).
template <typename config_t, typename Step>
int run_isolated_driver_loop(config_t& bconfig, bool allow_binary_boost, Step&& step)
{
    int exit_status =
        Kadath::legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
    while (Kadath::status_requires_field_reload(exit_status) ||
           (allow_binary_boost && Kadath::status_requests_binary_boost(exit_status))) {
        exit_status = step(bconfig, exit_status);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    return exit_status;
}

} // namespace KadathApps
