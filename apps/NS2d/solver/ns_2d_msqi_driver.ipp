// ns_2d_msqi_driver.cpp
#pragma once
#include "ns_2d_msqi_regrid.hpp"
#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include <sstream>
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"

namespace Kadath {

template <typename config_t> inline int ns_2d_msqi_driver(config_t& bconfig, std::string outputdir)
{
    int exit_status = legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (!bconfig.has_seq_setting(INIT_RES))
        bconfig.seq_setting(INIT_RES) = 9;

    const int final_res = bconfig(BCO_PARAMS::BCO_RES);
    const int init_res = bconfig.template seq_setting_as<int>(INIT_RES);
    bool res_inc = init_res < final_res;

    int current_rs = init_res;

    if (rank == 0) {
        std::cout << "\nInitial res: " << current_rs << " Final res: " << final_res << endl;
        std::cout << "Solutions will be stored in: " << outputdir << "\n";
        fs::create_directory(outputdir);
    }

    if (res_inc)
        bconfig.set(BCO_RES) = init_res;

    if (std::isnan(bconfig.set(MADM)) && (bconfig(MB) == 0. || std::isnan(bconfig.set(MB)))) {
        if (rank == 0) {
            std::cout << "Config error.  No madm nor mb found.\n\n";
        }
        KADATH_THROW("Stage failed");
    }

    if (!fs::exists(bconfig.space_filename())) {
        bconfig.set_filename("initns");

        if (rank == 0) {
            setup_co<NS>(bconfig);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        bconfig.open_config();
    }

    const std::string spacein = bconfig.space_filename();

    if (!fs::exists(spacein)) {
        if (rank == 0) {
            std::cerr << "File: " << spacein << " not found.\n\n";
        } else {
            std::cerr << "File: " << spacein << " not found for another rank.\n\n";
        }
        KADATH_THROW("Stage failed");
    }

    if (rank == 0)
        cout << "\nExit status: " << exit_status << endl << endl;
    while (status_requires_field_reload(exit_status)) {
        const std::string spacein = bconfig.space_filename();
        if (rank == 0) {
            std::cout << "Config File: " << bconfig.config_filename_abs() << std::endl
                      << "Fields File: " << spacein << std::endl
                      << bconfig << std::endl;
        }

        BeFileSource ff1(spacein);
        Space_polar_adapted space(ff1);
        Scalar conf(space, ff1);
        Scalar lapse(space, ff1);
        Scalar shift(space, ff1);
        Scalar metQ(space, ff1);
        Scalar logh(space, ff1);
        Scalar Omg(space, ff1);

        if (!outputdir.empty())
            bconfig.set_outputdir(outputdir);

        // load and setup the EOS
        const double h_cut = bconfig.template eos<double>(HCUT);
        const std::string eos_file = bconfig.template eos<std::string>(EOSFILE);
        const std::string eos_type = bconfig.template eos<std::string>(EOSTYPE);

        auto run_solver = [&](auto eos_tag, auto&&... eos_args) {
            using eos_t = decltype(eos_tag);
            EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut, std::forward<decltype(eos_args)>(eos_args)...);
            ns_2d_msqi_diff_solver<eos_t, decltype(bconfig), decltype(space)> ns_diff_solver(
                bconfig, space, conf, lapse, logh, shift, metQ, Omg);
            exit_status = ns_diff_solver.solve();
        };

        if (eos_type == "Cold_PWPoly") {
            using eos_t = Kadath::Margherita::Cold_PWPoly;
            run_solver(eos_t{});
        } else if (eos_type == "Cold_Table") {
            using eos_t = Kadath::Margherita::Cold_Table;
            const int interp_pts =
                (bconfig.template eos<int>(INTERP_PTS) == 0) ? 2000 : bconfig.template eos<int>(INTERP_PTS);
            run_solver(eos_t{}, interp_pts);
        } else {
            KADATH_THROW("Unknown EOSTYPE.");
        }

        if (rank == 0)
            cout << "Exit status: " << exit_status << "\n";

        if (res_inc && status_completed_without_reload(exit_status)) {
            current_rs += 2;
            bconfig.set(BCO_RES) = current_rs;
            res_inc = (bconfig(BCO_RES) < final_res);

            if (rank == 0) {
                exit_status = ns_2d_msqi_regrid(bconfig, bconfig(BCO_RES), "ns_regrid");
            }
            bconfig.set_filename("ns_regrid");
            MPI_Barrier(MPI_COMM_WORLD);
            bconfig.open_config();
            exit_status = legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
        } else if (bconfig.set_stage(NOROT) == false && bconfig.set_stage(UNIROT) == false) {
            exit_status = EXIT_SUCCESS;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    return exit_status;
}


} // namespace Kadath
