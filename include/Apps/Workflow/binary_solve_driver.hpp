#pragma once
// Stage-sequence drivers for the binary XCTS solvers (BNS + BHNS).
//
// The workflow order is explicit and intentionally independent of STAGE enum
// value order:
//
//   QUASI_EQUIL  -> fixed-Omega hydro-rescale initializer
//   FORCE_BALANCE -> free-Omega hydrostatic / first-integral solve
//   ECC_RED      -> fixed-Omega hydro-rescale eccentricity-reduction solve
//
// Pure orchestration: walk the enabled stages in workflow order via
// run_binary_stage_sequence and dispatch each to a solver-supplied thunk. The
// per-formalism physical diagnostics live one layer down in
// Apps/Formalism/Shared/binary_driver_common.hpp.

#include "mpi.h"
#include "Apps/Helper/solver_base.hpp"
#include "Apps/Formalism/Shared/omega_mode.hpp"
#include "Apps/Workflow/binary_stage_dispatch.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <array>
#include <iostream>
#include <string>
#include <utility>

namespace Kadath {

inline constexpr std::array<STAGE, 3> BINARY_WORKFLOW_ORDER{
    QUASI_EQUIL,
    FORCE_BALANCE,
    ECC_RED,
};

inline const char* binary_workflow_stage_name(STAGE stage)
{
    switch (stage) {
    case QUASI_EQUIL:
        return "QUASI_EQUIL";
    case FORCE_BALANCE:
        return "FORCE_BALANCE";
    case ECC_RED:
        return "ECC_RED";
    default:
        KADATH_THROW("Unsupported binary workflow stage");
    }
}

inline std::string binary_workflow_last_stage(const std::array<bool, NUM_STAGES_V>& stage_enabled)
{
    for (auto it = BINARY_WORKFLOW_ORDER.rbegin(); it != BINARY_WORKFLOW_ORDER.rend(); ++it) {
        if (stage_enabled[to_int(*it)])
            return binary_workflow_stage_name(*it);
    }
    KADATH_THROW("No binary stage enabled: set quasi_equilibrium, force_balance, and/or ecc_red in [stages].");
}

template <typename Solver, typename config_t, typename PrintBanner, typename RunHydroRescale, typename RunHydrostatic>
int binary_xcts_solve_common(Solver& solver, config_t& bconfig, PrintBanner&& print_banner,
                             RunHydroRescale&& run_hydro_rescale, RunHydrostatic&& run_hydrostatic)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto stage_enabled = bconfig.return_stages();
    (void)binary_workflow_last_stage(stage_enabled);

    if (rank == 0)
        std::forward<PrintBanner>(print_banner)();

    const int exit_status = run_binary_stage_sequence(stage_enabled, {
        {QUASI_EQUIL, [&]() {
             return run_hydro_rescale(QUASI_EQUIL, "QUASI_EQUIL");
         }},
        {FORCE_BALANCE, [&]() {
             return run_hydrostatic(FORCE_BALANCE, "FORCE_BALANCE");
         }},
        {ECC_RED, [&]() {
             return run_hydro_rescale(ECC_RED, "ECC_RED");
         }},
    });

    MPI_Barrier(MPI_COMM_WORLD);
    return exit_status;
}

template <typename Solver, typename config_t, typename PrintBanner>
int bns_xcts_solve_common(Solver& solver, config_t& bconfig, PrintBanner&& print_banner)
{
    return binary_xcts_solve_common(
        solver, bconfig, std::forward<PrintBanner>(print_banner),
        [&](STAGE stage, const std::string& stage_text) {
            return solver.hydro_rescaling_stage(stage, stage_text);
        },
        [&](STAGE, const std::string&) {
            return solver.hydrostatic_equilibrium_stage();
        });
}

template <typename Solver, typename config_t>
int bhns_xcts_solve_common(Solver& solver, config_t& bconfig, bool want_warmup)
{
    (void)want_warmup;
    auto print_banner = [&]() {
        std::cout << "=================================" << std::endl;
        std::cout << "BHNS grav input" << std::endl;
        std::cout << "Distance: " << bconfig(DIST) << std::endl;
        std::cout << "Omega guess: " << bconfig(GOMEGA) << std::endl;
        std::cout << "Units: " << (4.0 * M_PI) << std::endl;
        std::cout << "=================================" << std::endl;
    };

    const int status = binary_xcts_solve_common(
        solver, bconfig, print_banner,
        [&](STAGE stage, const std::string& stage_text) {
            return solver.hydro_rescaling_stages(stage, OmegaMode::Fixed, stage_text);
        },
        [&](STAGE stage, const std::string& stage_text) {
            return solver.hydrostatic_equilibrium_stage(stage, OmegaMode::Free, stage_text);
        });

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << "Exit status: " << status << "\n";
    return status;
}

} // namespace Kadath
