#pragma once

#include "mpi.h"
#include "For_Kadath/Array/exceptions.hpp"
#include <sstream>
#include "bh_2d_msqi_regrid.hpp"
#include "bh2d_solver_utils.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"

namespace Kadath {

template <typename config_t> inline int bh_2d_msqi_driver(config_t& bconfig, std::string outputdir)
{
    int exit_status = legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#if defined(__APPLE__)
    if (std::abs(bconfig(CHI)) > 0.9999992) {
        std::cerr << "For spin > 0.9992, please use workstation or cluster.\n";
        return EXIT_FAILURE;
    }
#endif
    if (!bconfig.has_seq_setting(INIT_RES))
        bconfig.seq_setting(INIT_RES) = 9;

    const int final_res = bconfig(BCO_PARAMS::BCO_RES);
    const int init_res = bconfig.template seq_setting_as<int>(INIT_RES);
    bool res_inc = init_res < final_res;

    int current_rs = init_res;

    if (rank == 0) {
        std::cout << "\nInitial res: " << current_rs << " Final res: " << final_res << std::endl;
        std::cout << "Solutions will be stored in: " << outputdir << "\n";
        fs::create_directory(outputdir);
    }

    if (res_inc)
        bconfig.set(BCO_RES) = init_res;
    if (!fs::exists(bconfig.space_filename())) {
        bconfig.set_filename("initbh");

        if (rank == 0)
            setup_co<BH>(bconfig);
        MPI_Barrier(MPI_COMM_WORLD);

        bconfig.open_config();
    }

    std::string spacein = bconfig.space_filename();

    {
        int file_ok = (rank == 0) ? (fs::exists(spacein) ? 1 : 0) : 0;
        MPI_Bcast(&file_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!file_ok) {
            if (rank == 0) {
                std::ostringstream oss;
                oss << "File: " << spacein << " not found.";
                KADATH_THROW(oss.str());
            }
        }
    }

    if (rank == 0)
        std::cout << "\nExit status: " << exit_status << std::endl << std::endl;
    while (status_requires_field_reload(exit_status)) {
        spacein = bconfig.space_filename();
        // just so you really know
        if (rank == 0) {
            std::cout << "Config File: " << bconfig.config_filename_abs() << std::endl
                      << "Fields File: " << spacein << std::endl
                      << bconfig << std::endl;
        }
        BeFileSource ff1(spacein);
        Space_adapted_bh_polar space(ff1);
        Scalar conf(space, ff1);
        Scalar lapse(space, ff1);
        Scalar metQ(space, ff1);
        Scalar shift(space, ff1);
        Scalar varscal(space, ff1);

        if (!outputdir.empty())
            bconfig.set_outputdir(outputdir);

        bh_2d_msqi_solver<decltype(bconfig), decltype(space)> bh_solver(bconfig, space, conf, lapse, metQ, shift,
                                                                        varscal);
        exit_status = bh_solver.solve();

        if (res_inc && status_completed_without_reload(exit_status)) {
            current_rs += 2;
            bconfig.set(BCO_RES) = current_rs;
            res_inc = (bconfig(BCO_RES) < final_res);

            if (rank == 0) {
                exit_status = bh_2d_msqi_regrid(bconfig, bconfig(BCO_RES), "bh_regrid");
            }
            bconfig.set_filename("bh_regrid");
            MPI_Barrier(MPI_COMM_WORLD);
            bconfig.open_config();
            exit_status = legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    return exit_status;
}

template <typename config_t, typename space_t> int bh_2d_msqi_solver<config_t, space_t>::solve()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exit_status = EXIT_SUCCESS;
    std::array<bool, NUM_STAGES_V> stage_enabled = bconfig.return_stages();
    auto [last_stage, last_stage_idx] = get_last_enabled(MSTAGE(), stage_enabled);
    if (rank == 0) {
        std::cout << "Last stage: " << last_stage << "\n";
    }

    if (stage_enabled[to_int(VON_NEUMANN)]) {

        // Chi ramping disabled for 2D BH solver

        exit_status = fixed_lapse_stage();
        if (status_requires_field_reload(exit_status))
            return exit_status;
    }
    // Barrier needed in case we need to read from the previous output
    MPI_Barrier(MPI_COMM_WORLD);
    return exit_status;
}


} // namespace Kadath
