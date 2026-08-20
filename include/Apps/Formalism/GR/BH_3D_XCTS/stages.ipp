/*
 * Copyright 2022
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 *
 * Author:
 * Samuel D. Tootle <tootle@itp.uni-frankfurt.de>
 * L. Jens Papenfort <papenfort@th.physik.uni-frankfurt.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "mpi.h"
#include "Apps/Helper/newton_loop_runner.hpp"

namespace Kadath {

template <typename config_t, typename space_t> int bh_3d_xcts_solver<config_t, space_t>::von_Neumann_stage()
{
    int exit_status = EXIT_SUCCESS;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto resume = this->load_existing_solution("VON_NEUMANN");
    if (resume.found()) {
        if (!resume.requires_field_reload()) {
            if (rank == 0)
                std::cout << "Solved previously: " << bconfig.config_filename_abs() << std::endl;
            return EXIT_SUCCESS;
        }
        return legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
    }

    if (rank == 0)
        std::cout << "############################" << std::endl
                  << "Total system with von Neumann BC" << std::endl
                  << "############################" << std::endl;
    bconfig.set(MIRR) = bco_utils::mirr_from_mch(bconfig(CHI), bconfig(MCH));

    double xo = 0.;

    System_of_eqs syst(space, 0, ndom - 1);
    syst_init(syst);

    // in the event we want a linear boost
    syst.add_cst("xboost", bconfig(BVELX));
    syst.add_cst("yboost", bconfig(BVELY));

    syst.add_def("intS  = A_ij * mg^i * sm^j / 8. / PI");

    // definition of constraint equations - conformally flat/maximal slice only
    // K = 0 -> R = 0; \partial_t(metric) = 0
    syst.add_def("eqP = D^i D_i P + A_ij * A^ij / P^7 / 8");
    syst.add_def("eqNP = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
    syst.add_def("eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");

    // Add equations and continuity constraints to the numerical space
    space.add_eq(syst, "eqNP = 0", "N", "dn(N)");
    space.add_eq(syst, "eqP = 0", "P", "dn(P)");
    space.add_eq(syst, "eqbet^i = 0", "bet^i", "dn(bet^i)");

    // boundary conditions at infinity
    space.add_bc_inf(syst, "NP = 1");
    space.add_bc_inf(syst, "P = 1");
    space.add_bc_inf(syst, "bet^i = xboost * ex^i + yboost * ey^i");

    // excision boundary conditions
    space.add_bc_bh(syst, "dn(NP) = 0");
    space.add_bc_bh(syst, "sm^j * D_j P + P / 4 * D^j sm_j + A_ij * sm^i * sm^j / P^3 / 4 = 0");
    space.add_bc_bh(syst, "bet^i = N / P^2 * sm^i + ome * mg^i");

    // equations fixing the radius of the BH based on (MIRR)
    // and the spin angular momentum based on fixed dimensionless spin (CHI)
    space.add_eq_int_bh(syst, "integ(intS) - chi * CM * CM = 0 ");
    space.add_eq_int_bh(syst, "integ(intMsq) - M * M = 0 ");

    for (int i : excluded_doms) {
        syst.add_eq_full(i, "N = 0");
        syst.add_eq_full(i, "P = 0");
        syst.add_eq_full(i, "bet^i = 0");
    }
    bool endloop = false;
    int ite = 1;
    double conv = 0;
    const SolverRuntimeConfig solver_config = this->with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), "VON_NEUMANN");

    if (rank == 0) {
        // syst.print_system_structure(std::cout);
        print_diagnostics(syst, ite, conv);
    }

    while (!endloop) {
        endloop = newton_step_with_consensus(syst, bconfig.template seq_setting_as<double>(PREC), conv, solver_config, ite == 1);

        std::stringstream ss;
        ss << "bh_total_bc_" << ite - 1;
        bconfig.set_filename(ss.str());
        if (rank == 0) {
            print_diagnostics(syst, ite, conv);
            if (bconfig.control(CHECKPOINT)) {
                bco_utils::save_to_file(space, bconfig, conf, lapse, shift);
            }
        }

        update_fields_co(cfields, coord_vectors, {}, xo, &syst);
        ite++;
        if (check_max_iter_exceeded(rank, ite, conv))
            break;
    }

    bconfig.set(RMID) = bco_utils::get_radius(space.get_domain(1), OUTER_BC);
    bconfig.set(FIXED_LAPSE) = bco_utils::get_boundary_val(2, lapse, INNER_BC);
    bconfig.set_filename(converged_filename("VON_NEUMANN"));
    if (rank == 0)
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift);

    return exit_status;
}

template <typename config_t, typename space_t> int bh_3d_xcts_solver<config_t, space_t>::fixed_lapse_stage()
{
    int exit_status = EXIT_SUCCESS;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (bconfig.return_stages()[to_int(VON_NEUMANN)] &&
        bconfig.config_filename() == converged_filename("VON_NEUMANN") + ".toml") {
        return EXIT_SUCCESS;
    }

    const auto resume = this->load_existing_solution("DIRICHLET_LAPSE");
    if (resume.found()) {
        if (!resume.requires_field_reload()) {
            if (rank == 0)
                std::cout << "Solved previously: " << bconfig.config_filename_abs() << std::endl;
            return EXIT_SUCCESS;
        }
        return legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
    }

    if (rank == 0)
        std::cout << "############################" << std::endl
                  << "Total system with Fixed Lapse BC" << std::endl
                  << "############################" << std::endl;
    bconfig.set(MIRR) = bco_utils::mirr_from_mch(bconfig(CHI), bconfig(MCH));

    double xo = 0.;

    System_of_eqs syst(space, 0, ndom - 1);
    syst_init(syst);

    // in the event we want a linear boost
    syst.add_cst("xboost", bconfig(BVELX));
    syst.add_cst("yboost", bconfig(BVELY));
    syst.add_cst("n0", bconfig(FIXED_LAPSE));

    syst.add_def("intS  = A_ij * mg^i * sm^j / 8. / PI");

    // definition of constraint equations - conformally flat/maximal slice only
    // K = 0 -> R = 0; \partial_t(metric) = 0
    syst.add_def("eqP = D^i D_i P + A_ij * A^ij / P^7 / 8");
    syst.add_def("eqNP = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
    syst.add_def("eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");

    // Add equations and continuity constraints to the numerical space
    space.add_eq(syst, "eqNP = 0", "N", "dn(N)");
    space.add_eq(syst, "eqP = 0", "P", "dn(P)");
    space.add_eq(syst, "eqbet^i = 0", "bet^i", "dn(bet^i)");

    // boundary conditions at infinity
    space.add_bc_inf(syst, "NP = 1");
    space.add_bc_inf(syst, "P = 1");
    space.add_bc_inf(syst, "bet^i = xboost * ex^i + yboost * ey^i");

    // excision boundary conditions
    space.add_bc_bh(syst, "N = n0");
    space.add_bc_bh(syst, "sm^j * D_j P + P / 4 * D^j sm_j + A_ij * sm^i * sm^j / P^3 / 4 = 0");
    space.add_bc_bh(syst, "bet^i = N / P^2 * sm^i + ome * mg^i");

    // equations fixing the radius of the BH based on (MIRR)
    // and the spin angular momentum based on fixed dimensionless spin (CHI)
    space.add_eq_int_bh(syst, "integ(intS) - chi * CM * CM = 0 ");
    space.add_eq_int_bh(syst, "integ(intMsq) - M * M = 0 ");

    for (int i : excluded_doms) {
        syst.add_eq_full(i, "N = 0");
        syst.add_eq_full(i, "P = 0");
        syst.add_eq_full(i, "bet^i = 0");
    }
    bool endloop = false;
    int ite = 1;
    double conv = 0;
    const SolverRuntimeConfig solver_config = this->with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), "DIRICHLET_LAPSE");
    if (rank == 0)
        print_diagnostics(syst, ite, conv);
    while (!endloop) {
        endloop = newton_step_with_consensus(syst, bconfig.template seq_setting_as<double>(PREC), conv, solver_config, ite == 1);

        std::stringstream ss;
        ss << "bh_total_" << ite - 1;
        bconfig.set_filename(ss.str());
        if (rank == 0) {
            print_diagnostics(syst, ite, conv);
            if (bconfig.control(CHECKPOINT)) {
                bco_utils::save_to_file(space, bconfig, conf, lapse, shift);
            }
        }

        update_fields_co(cfields, coord_vectors, {}, xo, &syst);
        ite++;
        if (check_max_iter_exceeded(rank, ite, conv))
            break;
    }

    bconfig.set(RMID) = bco_utils::get_radius(space.get_domain(1), OUTER_BC);
    bconfig.set_filename(converged_filename("DIRICHLET_LAPSE"));
    if (rank == 0)
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift);

    return exit_status;
}

template <typename config_t, typename space_t>
int bh_3d_xcts_solver<config_t, space_t>::binary_boost_stage(kadath_config<BIN_INFO>& binconfig, NODES bco)
{
    int exit_status = EXIT_SUCCESS;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << "############################" << std::endl
                  << "Binary boost using von Neumann BC" << std::endl
                  << "############################" << std::endl;
    bconfig.set(MIRR) = bco_utils::mirr_from_mch(bconfig(CHI), bconfig(MCH));

    // generate filename string unique to this binary setup
    std::stringstream ss;
    ss << "BIN_BOOST" << "_" << binconfig(DIST) << "_" << binconfig(GOMEGA);
    auto const boost_converged_filename{ss.str()};

    // check if we have a non-boosted solution
    const auto resume = this->load_existing_solution(boost_converged_filename);
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Solved previously: " << bconfig.config_filename_abs() << std::endl;
        return EXIT_SUCCESS;
    }

    double xo = 0.;

    update_fields_co(cfields, coord_vectors, {}, xo);

    System_of_eqs syst(space, 0, ndom - 1);
    syst_init(syst);

    syst.add_def("intS = A_ij * mg^i * sm^j / 8. / PI");

    // boost based on binary config
    syst.add_cst("omega_boost", binconfig(GOMEGA));
    syst.add_def("B^i = bet^i + omega_boost * mg^i");

    // definition of constraint equations - conformally flat/maximal slice only
    // K = 0 -> R = 0; \partial_t(metric) = 0
    syst.add_def("eqP = D^i D_i P + A_ij * A^ij / P^7 / 8");
    syst.add_def("eqNP = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
    syst.add_def("eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");

    // Add equations and continuity constraints to the numerical space
    space.add_eq(syst, "eqNP = 0", "N", "dn(N)");
    space.add_eq(syst, "eqP = 0", "P", "dn(P)");
    space.add_eq(syst, "eqbet^i = 0", "bet^i", "dn(bet^i)");

    // boundary conditions at infinity
    space.add_bc_inf(syst, "NP = 1");
    space.add_bc_inf(syst, "P = 1");
    space.add_bc_inf(syst, "bet^i = 0");

    // excision boundary conditions
    space.add_bc_bh(syst, "dn(NP) = 0");
    space.add_bc_bh(syst, "sm^j * D_j P + P / 4 * D^j sm_j + A_ij * sm^i * sm^j / P^3 / 4 = 0");
    space.add_bc_bh(syst, "B^i = N / P^2 * sm^i + ome * mg^i");

    // equations fixing the radius of the BH based on (MIRR)
    // and the spin angular momentum based on fixed dimensionless spin (CHI)
    space.add_eq_int_bh(syst, "integ(intS) - chi * CM * CM = 0 ");
    space.add_eq_int_bh(syst, "integ(intMsq) - M * M = 0 ");

    for (int i : excluded_doms) {
        syst.add_eq_full(i, "N = 0");
        syst.add_eq_full(i, "P = 0");
        syst.add_eq_full(i, "bet^i = 0");
    }
    bool endloop = false;
    int ite = 1;
    double conv = 0;
    const SolverRuntimeConfig solver_config = this->with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), boost_converged_filename);

    if (rank == 0) {
        // syst.print_system_structure(std::cout);
        print_diagnostics(syst, ite, conv);
    }
    while (!endloop) {
        endloop = newton_step_with_consensus(syst, bconfig.template seq_setting_as<double>(PREC), conv, solver_config, ite == 1);

        std::stringstream ss;
        ss << "bh_bin_boost_" << ite - 1;
        bconfig.set_filename(ss.str());
        if (rank == 0) {
            print_diagnostics(syst, ite, conv);
            if (bconfig.control(CHECKPOINT)) {
                bco_utils::save_to_file(space, bconfig, conf, lapse, shift);
            }
        }

        update_fields_co(cfields, coord_vectors, {}, xo, &syst);
        ite++;
        if (check_max_iter_exceeded(rank, ite, conv))
            break;
    }

    bconfig.set(RMID) = bco_utils::get_radius(space.get_domain(1), OUTER_BC);
    bconfig.set_filename(converged_filename(boost_converged_filename));
    if (rank == 0)
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift);

    return exit_status;
}


} // namespace Kadath
