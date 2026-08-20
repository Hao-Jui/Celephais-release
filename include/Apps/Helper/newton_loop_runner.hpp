#pragma once

#include "mpi.h"
#include "For_Kadath/System_of_eqs/solver_runtime_config.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

namespace Kadath {

/**
 * One Newton step that all MPI ranks exit together.
 *
 * Calls \c System_of_eqs::do_newton_configured on every rank, then performs an
 * \c MPI_Allreduce with \c MPI_MIN on the per-rank convergence flag. Returns
 * \c true only when every rank reports convergence in the same step, so the
 * caller's outer \c while(!endloop) loop terminates in lock-step across the
 * communicator. Without this consensus check a single rank that converges
 * earlier than its peers can break the loop while others are mid-iteration,
 * leaving \c MPI_Comm_world in an inconsistent state.
 */
inline bool newton_step_with_consensus(System_of_eqs& syst,
                                       double precision,
                                       double& error,
                                       const SolverRuntimeConfig& config,
                                       bool reset_runtime_state = false) {
    if (reset_runtime_state)
        syst.reset_solver_runtime_state();
    const bool local = syst.do_newton_configured(precision, error, config);
    int local_done = local ? 1 : 0;
    int global_done = 0;
    // NOLINTNEXTLINE(bugprone-casting-through-void) — OpenMPI handle macros expand through void*.
    MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return global_done != 0;
}

} // namespace Kadath
