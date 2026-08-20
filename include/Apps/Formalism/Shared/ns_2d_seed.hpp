#pragma once

#include "Apps/Formalism/GR/NS_2D_XCTS/lifting.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "mpi.h"

#include <cmath>
#include <iostream>
#include <string>

namespace Kadath {

template <typename stages_t>
stages_t ns_2d_xcts_lifted_stages(stages_t stages)
{
    stages[to_int(NOROT)] = false;
    stages[to_int(UNIROT)] = true;
    return stages;
}

template <typename target_space_t,
          ns_2d_xcts_lift_field_layout field_layout = ns_2d_xcts_lift_field_layout::gr,
          typename config_t, typename Solve2d>
void ns_2d_xcts_seed_initial_data(config_t& bc, const int current_res, const int final_res,
                                  Solve2d&& solve_2d, const std::string& seed_label,
                                  const std::string& initial_basename,
                                  const double tilt_degrees = 0.)
{
    if (!std::isfinite(tilt_degrees))
        KADATH_THROW("2D XCTS seed lift requires a finite tilt angle");

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto seed_config = bc;
    const auto lifted_stages = ns_2d_xcts_lifted_stages(bc.return_stages());
    const double final_chi = bc.template seq_setting_as<double>(FINAL_CHI);
    const double seed_chi = std::fabs(final_chi) > 0.2 ? std::copysign(0.1, final_chi) : final_chi;

    seed_config.set(DIM) = 2;
    seed_config.set(BCO_RES) = final_res;
    seed_config.seq_setting(INIT_RES) = current_res;
    seed_config.set(CHI) = seed_chi;
    seed_config.set(DEG) = 0.;
    seed_config.set_stage(NOROT) = false;
    seed_config.set_stage(UNIROT) = true;
    seed_config.set_stage(BIN_BOOST) = false;
    seed_config.set_filename(initial_basename);

    if (rank == 0) {
        std::cout << "Generating " << seed_label
                  << " seed from a converged 2D XCTS solve at resolution "
                  << current_res << " -> " << final_res << " and chi " << seed_chi
                  << " (target chi " << final_chi << ")";
        if (std::fabs(tilt_degrees) > 1.e-15)
            std::cout << ", then rotating it to tilt " << tilt_degrees << " degrees";
        std::cout << ".\n";
    }

    const int seed_status = solve_2d(
        seed_config, seed_config.config_outputdir(), initial_basename);
    if (seed_status != EXIT_SUCCESS)
        KADATH_THROW("2D XCTS seed solve failed before 3D " + seed_label + " lift");

    if (std::fabs(final_chi - seed_chi) > 1.e-14) {
        seed_config.set(CHI) = final_chi;
        seed_config.seq_setting(INIT_RES) = final_res;
        if (rank == 0)
            std::cout << "Continuing 2D XCTS seed from chi " << seed_chi
                      << " to target chi " << final_chi << " at resolution " << final_res
                      << " before 3D lift.\n";
        const int continuation_status = solve_2d(
            seed_config, seed_config.config_outputdir(), initial_basename);
        if (continuation_status != EXIT_SUCCESS)
            KADATH_THROW("2D XCTS seed chi continuation failed before 3D " + seed_label + " lift");
    }

    if (rank == 0) {
        seed_config.return_stages() = lifted_stages;
        seed_config.set(DIM) = 3;
        seed_config.set(BCO_RES) = final_res;
        seed_config.set(CHI) = final_chi;
        seed_config.set(DEG) = tilt_degrees;
        constexpr double degrees_to_radians = M_PI / 180.;
        ns_2d_xcts_lift_to_3d_as<target_space_t, field_layout>(
            seed_config, final_res, "initns", true, tilt_degrees * degrees_to_radians);
    }
}

} // namespace Kadath
