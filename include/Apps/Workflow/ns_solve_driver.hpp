#pragma once
// Single-solve stage dispatch + resolution-read helpers for ns_3d_xcts solvers.
//
// Pure orchestration: apply the enabled stages (NOROT / UNIROT) to one loaded
// space via solver-supplied callbacks, and read/assert the radial resolution of
// a restart file. The resolution-sequencing state machine lives alongside in
// ns_resolution_sequence.hpp; the per-formalism physical diagnostics live one
// layer down in Apps/Formalism/Shared/ns_driver_common.hpp.

#include "mpi.h"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "Apps/Helper/solver_base.hpp"

#include <sstream>
#include <array>
#include <string>

namespace Kadath {

template <typename space_t>
int ns_3d_xcts_read_space_resolution(const std::string& spacein)
{
    if (!fs::exists(spacein)) {
        std::ostringstream oss;
        oss << "File: " << spacein << " not found.";
        KADATH_THROW(oss.str());
    }

    BeFileSource source(spacein);
    space_t space(source);
    return space.get_domain(0)->get_nbr_points()(0);
}

template <typename space_t>
void ns_3d_xcts_assert_space_resolution(const std::string& spacein, int expected_res, const std::string& label)
{
    const int actual_res = ns_3d_xcts_read_space_resolution<space_t>(spacein);
    if (actual_res == expected_res)
        return;

    std::ostringstream oss;
    oss << "Resolution mismatch for " << label << ": " << spacein << " has radial resolution " << actual_res
        << ", but expected " << expected_res << ". Make initial_resolution match the existing .dat file "
        << "or remove the stale restart files.";
    KADATH_THROW(oss.str());
}

template <typename Solver, typename config_t>
int ns_3d_xcts_solve_common(Solver& solver, config_t& bconfig)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exit_status = EXIT_SUCCESS;

    std::array<bool, NUM_STAGES_V> stage_enabled = bconfig.return_stages();
    auto [last_stage, last_stage_idx] = get_last_enabled(MSTAGE(), stage_enabled);
    if (rank == 0)
        std::cout << "Last stage: " << last_stage << "\n";

    if (stage_enabled[to_int(NOROT)]) {
        double const initial_chi = bconfig(CHI);
        exit_status = solver.norot_stage(false);
        bconfig(CHI) = initial_chi;
        if (status_requires_field_reload(exit_status))
            return exit_status;
        return legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
    }

    if (stage_enabled[to_int(UNIROT)]) {
        bconfig.set_stage(NOROT) = false;
        bconfig(CHI) = bconfig.template seq_setting_as<double>(FINAL_CHI);
        exit_status = solver.uniform_rot_stage();
        if (status_requires_field_reload(exit_status))
            return exit_status;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return exit_status;
}

} // namespace Kadath
