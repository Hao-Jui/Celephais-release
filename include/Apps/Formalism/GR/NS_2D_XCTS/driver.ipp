// NS_2D_XCTS driver.
#pragma once
#include "regrid.hpp"
#include "seed_utils.hpp"
#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include <sstream>
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include <cmath>

namespace Kadath {

constexpr double NS_2D_XCTS_CHI_CONTINUATION = 0.3;
constexpr double NS_2D_XCTS_CHI_TOL = 1.e-14;

template <typename config_t> inline bool ns_2d_xcts_needs_chi_continuation(const config_t& bconfig,
                                                                           const double final_chi)
{
    if (std::fabs(final_chi) <= NS_2D_XCTS_CHI_CONTINUATION + NS_2D_XCTS_CHI_TOL)
        return false;

    if (!fs::exists(bconfig.space_filename()) || !fs::exists(bconfig.config_filename_abs()))
        return true;

    config_t current_dataset(bconfig.config_filename_abs());
    const double current_chi = current_dataset(CHI);
    if (std::fabs(current_chi - final_chi) <= NS_2D_XCTS_CHI_TOL)
        return false;

    const bool same_spin_direction = std::signbit(current_chi) == std::signbit(final_chi);
    return !same_spin_direction || std::fabs(current_chi) < NS_2D_XCTS_CHI_CONTINUATION - NS_2D_XCTS_CHI_TOL;
}

template <typename config_t>
inline int ns_2d_xcts_driver_single_chi(config_t& bconfig, std::string outputdir,
                                        const std::string& initial_basename)
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

    const bool overwrite_initial = bconfig.config_filename() == initial_basename + ".toml";
    if (overwrite_initial || !fs::exists(bconfig.space_filename())) {
        bconfig.set_filename(initial_basename);

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
            ns_2d_xcts_diff_solver<eos_t, decltype(bconfig), decltype(space)> ns_diff_solver(
                bconfig, space, conf, lapse, logh, shift, Omg);
            exit_status = ns_diff_solver.solve();
        };

        if (eos_type == "Cold_PWPoly") {
            using eos_t = Kadath::Margherita::Cold_PWPoly;
            run_solver(eos_t{});
        } else if (eos_type == "Cold_Table") {
            using eos_t = Kadath::Margherita::Cold_Table;
            const int interp_pts =
                (bconfig.template eos<int>(INTERP_PTS) == 0) ? 2000 : bconfig.template eos<int>(INTERP_PTS);
            run_solver(eos_t{}, interp_pts, bconfig.template eos<double>(MNUC_CGS));
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
                exit_status = ns_2d_xcts_regrid_from_fields(
                    bconfig, space, conf, lapse, shift, logh, Omg,
                    bconfig(BCO_RES), "ns_regrid");
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

template <typename config_t>
inline int ns_2d_xcts_driver(config_t& bconfig, std::string outputdir, std::string initial_basename)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const double final_chi = bconfig(CHI);
    ns_2d_xcts_core::validate_spin(final_chi);
    if (!ns_2d_xcts_needs_chi_continuation(bconfig, final_chi))
        return ns_2d_xcts_driver_single_chi(bconfig, outputdir, initial_basename);

    auto intermediate_config = bconfig;
    const double intermediate_chi = std::copysign(NS_2D_XCTS_CHI_CONTINUATION, final_chi);
    intermediate_config.set(CHI) = intermediate_chi;

    if (rank == 0)
        std::cout << "Solving intermediate NS2d XCTS spin chi " << intermediate_chi
                  << " before target chi " << final_chi << ".\n";

    int exit_status = ns_2d_xcts_driver_single_chi(intermediate_config, outputdir, initial_basename);
    if (exit_status != EXIT_SUCCESS)
        return exit_status;

    bconfig = intermediate_config;
    bconfig.set(CHI) = final_chi;
    bconfig.seq_setting(INIT_RES) = static_cast<int>(bconfig(BCO_RES));

    if (rank == 0)
        std::cout << "Continuing NS2d XCTS spin from chi " << intermediate_chi
                  << " to target chi " << final_chi << " at resolution " << bconfig(BCO_RES) << ".\n";

    return ns_2d_xcts_driver_single_chi(bconfig, outputdir, initial_basename);
}


} // namespace Kadath
