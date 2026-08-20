#pragma once

#include "Apps/Workflow/solver_sequence.hpp"
#include "Apps/Workflow/resume_ladder.hpp"
#include "Apps/Workflow/isolated_driver_loop.hpp"
#include "Apps/Seed/GR/ns_seed.hpp"
#include "Apps/Workflow/ns_solve_driver.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Config/config_binary.hpp"

#include <mpi.h>
#include <array>
#include <iostream>
#include <string>
#include <utility>

namespace Kadath {

// ---------------------------------------------------------------------------
// Isolated NS sequence engine — cold-start seed, init->final +2 resolution
// ladder, regrid-between-rungs, and the per-stage field-reload solve loop.
//
// Callbacks:
//   solve_from_file(config, spacein) -> exit_status
//     Read fields from spacein, EOS dispatch, construct solver, call solve().
//   regrid_func(config, new_res, filename) -> exit_status
//     Interpolate onto a new grid.
// ---------------------------------------------------------------------------
template <typename space_t, typename config_t, typename SolveFromFile, typename RegridFunc, typename SeedFunc,
          typename ResumeLadderFunc, typename AssertSpaceResolutionFunc>
config_t ns_3d_xcts_isolated_driver_impl(config_t& bconfig, std::string outputdir,
                                         SolveFromFile&& solve_from_file, RegridFunc&& regrid_func,
                                         SeedFunc&& seed_func, ResumeLadderFunc&& resume_ladder_func,
                                         AssertSpaceResolutionFunc&& assert_space_resolution)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto iterative_config = bconfig;

    // Warm starts resume from an existing .dat: floor the ladder init at the
    // loaded grid's resolution so a continuation never restarts below (and
    // regrids away) the saved solution. With init == loaded the resolution
    // assertions below also pass instead of forcing a manual initial_resolution.
    const ResolutionLadder ladder = resume_ladder_func(
        bconfig, bconfig.template seq_setting_as<int>(INIT_RES), bconfig(BCO_RES),
        /*warm_start=*/fs::exists(bconfig.space_filename()));
    const int final_res = ladder.final;

    int current_res = ladder.init;
    bool res_inc = (current_res < final_res);
    bconfig.set(BCO_RES) = current_res;

    bconfig.seq_setting(FINAL_CHI) = bconfig(CHI);

    if (rank == 0)
        std::cout << "Solutions will be stored in: " << outputdir << "\n"
                  << "Directory will be created if it doesn't exist.\n";
    fs::create_directory(outputdir);

    if (std::isnan(bconfig.set(MADM)) && (bconfig(MB) == 0. || std::isnan(bconfig.set(MB)))) {
        if (rank == 0)
            KADATH_THROW("Config error.  No madm nor mb found. \\n\\n");
    }

    if (!fs::exists(bconfig.space_filename())) {
        bconfig.set_filename("initns");
        std::forward<SeedFunc>(seed_func)(bconfig, current_res, final_res);
        MPI_Barrier(MPI_COMM_WORLD);
        bconfig.open_config();
        current_res = static_cast<int>(bconfig(BCO_RES));
        if (current_res > final_res)
            KADATH_THROW("Generated NS seed resolution exceeds requested final resolution.");
        res_inc = (current_res < final_res);
    }

    std::array<bool, NUM_STAGES_V> stage_enabled = bconfig.return_stages();
    auto [last_stage, last_stage_idx] = get_last_enabled(MSTAGE(), stage_enabled);

    std::string spacein = bconfig.space_filename();
    if (!fs::exists(spacein)) {
        if (rank == 0)
            std::cerr << "File: " << spacein << " not found.\n\n";
        else
            std::cerr << "File: " << spacein << " not found for another rank.\n\n";
        KADATH_THROW("Stage failed");
    }
    assert_space_resolution(bconfig, spacein, current_res, "initial_resolution");

    // Isolated NS continuation loop. Binary boost runs in a SEPARATE driver
    // (ns_3d_xcts_boosted_driver_impl), so allow_binary_boost = false here. The
    // body below is the per-rung step (unchanged from the hand-written loop): the
    // NOROT-consume / UNIROT-promotion stage management and the +2 regrid.
    KadathApps::run_isolated_driver_loop(
        bconfig, /*allow_binary_boost=*/false,
        [&](config_t& bconfig, int exit_status) -> int {
            std::string spacein = bconfig.space_filename();
            assert_space_resolution(bconfig, spacein, current_res, "active resolution");
            bconfig.set(BCO_RES) = current_res;
            if (rank == 0) {
                std::cout << "Config File: " << bconfig.config_filename_abs() << std::endl
                          << "Fields File: " << spacein << std::endl
                          << bconfig << std::endl;
            }

            if (outputdir != "")
                bconfig.set_outputdir(outputdir);

            exit_status = solve_from_file(bconfig, spacein);

            if (bconfig.set_stage(NOROT) == true && status_requires_field_reload(exit_status)) {
                // A NOROT rung is terminal only once the requested final resolution is reached.
                const bool norot_is_final_stage = last_stage_idx == to_int(NOROT);
                if (!norot_is_final_stage || !res_inc)
                    bconfig.set_stage(NOROT) = false;
                iterative_config = bconfig;
                if (bconfig.set_stage(UNIROT) == false) {
                    exit_status = EXIT_SUCCESS;
                }
            }

            if (res_inc && status_completed_without_reload(exit_status)) {
                bconfig.set_stage(NOROT) = last_stage_idx == to_int(NOROT);
                const int next_res = KadathApps::next_resolution(current_res, final_res);
                bconfig(BCO_RES) = next_res;
                if (rank == 0)
                    exit_status = regrid_func(bconfig, next_res, std::string("initns"));
                current_res = next_res;
                res_inc = (current_res < final_res);
                bconfig.set_filename("initns");
                MPI_Barrier(MPI_COMM_WORLD);
                bconfig.open_config();
                bconfig.set(BCO_RES) = current_res;
                exit_status = legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
            } else if (bconfig.set_stage(NOROT) == false && bconfig.set_stage(UNIROT) == false) {
                exit_status = EXIT_SUCCESS;
            }

            return exit_status;
        });
    return iterative_config;
}

template <typename space_t, typename config_t, typename SolveFromFile, typename RegridFunc, typename SeedFunc>
config_t ns_3d_xcts_isolated_driver_impl(config_t& bconfig, std::string outputdir,
                                         SolveFromFile&& solve_from_file, RegridFunc&& regrid_func,
                                         SeedFunc&& seed_func)
{
    auto resume_ladder_func = [](config_t& bc, int requested_init, int final, bool warm_start) {
        return KadathApps::make_resume_ladder<space_t>(bc, requested_init, final, warm_start);
    };
    auto assert_space_resolution = [](config_t& /*bc*/, const std::string& spacein, int expected_res,
                                      const std::string& label) {
        ns_3d_xcts_assert_space_resolution<space_t>(spacein, expected_res, label);
    };

    return ns_3d_xcts_isolated_driver_impl<space_t>(
        bconfig, std::move(outputdir), std::forward<SolveFromFile>(solve_from_file),
        std::forward<RegridFunc>(regrid_func), std::forward<SeedFunc>(seed_func), resume_ladder_func,
        assert_space_resolution);
}

template <typename space_t, typename config_t, typename SolveFromFile, typename RegridFunc>
config_t ns_3d_xcts_isolated_driver_impl(config_t& bconfig, std::string outputdir,
                                         SolveFromFile&& solve_from_file, RegridFunc&& regrid_func)
{
    auto seed_func = [](config_t& bc, int /*current_res*/, int /*final_res*/) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            Kadath::Seed::setup_co<NS, space_t>(bc);
    };

    return ns_3d_xcts_isolated_driver_impl<space_t>(
        bconfig, std::move(outputdir), std::forward<SolveFromFile>(solve_from_file),
        std::forward<RegridFunc>(regrid_func), seed_func);
}

// ---------------------------------------------------------------------------
// Boosted NS continuation — the binary-boost stage loop.
//
// Callback:
//   boost_from_file(config, spacein, binconfig, bco) -> exit_status
//     Read fields from spacein, EOS dispatch, construct solver, call
//     binary_boost_stage().
// ---------------------------------------------------------------------------
template <typename config_t, typename BoostFromFile>
int ns_3d_xcts_boosted_driver_impl(config_t& nsconfig, config_t& /*iterative_config*/,
                                   std::string outputdir, kadath_config<BIN_INFO> binconfig,
                                   NODES bco, BoostFromFile&& boost_from_file)
{
    int exit_status = legacy_status_from_stage_outcome(KadathApps::request_binary_boost());
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto bconfig = nsconfig;

    while (status_requests_binary_boost(exit_status)) {
        auto spacein = bconfig.space_filename();
        if (rank == 0) {
            std::cout << "Config File: " << bconfig.config_filename_abs() << std::endl
                      << "Fields File: " << spacein << std::endl
                      << bconfig << std::endl;
        }

        if (outputdir != "")
            bconfig.set_outputdir(outputdir);

        exit_status = boost_from_file(bconfig, spacein, binconfig, bco);

        MPI_Barrier(MPI_COMM_WORLD);
    }
    nsconfig = bconfig;
    return exit_status;
}

} // namespace Kadath
