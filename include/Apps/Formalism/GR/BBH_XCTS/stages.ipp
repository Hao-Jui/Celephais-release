#pragma once
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"
#include "mpi.h"
#include "Apps/Helper/newton_loop_runner.hpp"

namespace Kadath {

inline void print_stage(std::string stage_name)
{
    std::cout << "############################" << std::endl
              << stage_name << std::endl
              << "############################" << std::endl;
}

template <typename config_t, typename space_t>
int bbh_xcts_solver<config_t, space_t>::solve_stage(STAGE stage, OmegaMode omega_mode, std::string stage_text)
{
    int exit_status = EXIT_SUCCESS;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Fixed Omega holds "ome" constant and drops the quasi-equilibrium
    // constraint; Free solves for it. ECC_RED is treated as fixed-Omega below
    // regardless of `omega_mode` (it carries its own constant-ome / no-QE rules),
    // exactly as the original keyed those choices off (FIXED_GOMEGA || ECC_RED).
    const bool fixed_omega = (omega_mode == OmegaMode::Fixed);
    if (stage != ECC_RED && fixed_omega)
        stage_text += "_FIXED_OMEGA";

    std::string const lapse_cond =
        (bconfig.control(USE_FIXED_LAPSE)) ? "Total system with Fixed Lapse BC" : "Total system with v.Neumann BC";
    if (rank == 0 && stage == ECC_RED)
        print_stage("Ecc. Reduction - Fixed Omega & Adot");
    if (rank == 0 && bconfig.control(COROT_BIN))
        print_stage(lapse_cond + " - COROT");
    else if (rank == 0)
        print_stage(lapse_cond);

    if (stage == ECC_RED) {
        // determine whether to use PN estimates of the orbital frequency and adot
        // in case of the eccentricity stage
        if (std::isnan(bconfig.set(ADOT)) || std::isnan(bconfig.set(ECC_OMEGA)) || bconfig.control(USE_PN)) {

            const auto pn_orbital_params =
                bco_utils::binary_pn_orbital_seed(bconfig, bconfig(MCH, BCO1), bconfig(MCH, BCO2));
            bconfig.set(ECC_OMEGA) = pn_orbital_params.omega;
            bconfig.set(ADOT) = pn_orbital_params.adot;
            bconfig.set(GOMEGA) = pn_orbital_params.omega;

            if (rank == 0)
                std::cout << "### Using PN estimate for adot and omega! ###" << std::endl;
        }
        bconfig.set(GOMEGA) = bconfig(ECC_OMEGA);
    }

    // setup background position vector field - only needed for ECC_RED stage
    Vector CART(space, CON, basis);
    CART = cfields.cart();

    System_of_eqs syst(space, 0, ndom - 1);
    // allow for a shift of the "center of mass" on the x-axis
    // enforced by a vanishing ADM linear momentum
    // determine whether orbital frequency is fixed
    if (fixed_omega || stage == ECC_RED) {
        syst.add_cst("ome", bconfig(GOMEGA));
    } else {
        syst.add_var("ome", bconfig(GOMEGA));
    }
    syst.add_var("xaxis", bconfig(COM));
    syst.add_var("yaxis", bconfig(COMY));
    syst_init(syst);

    std::string bigB{"B^i = bet^i + ome * Morb^i"};
    if (stage == ECC_RED) {
        syst.add_cst("adot", bconfig(ADOT));
        syst.add_cst("r", CART);
        syst.add_def("comr^i = r^i - xaxis * ex^i + yaxis * ey^i");
        // add contribution of ADOT to the total shift definition
        bigB += " + adot * comr^i";
    }

    // total shift = inertial + orbital contributions
    syst.add_def(bigB.c_str());

    // vacuum XCTS constraint equations
    syst.add_def("eqP     = D^i D_i P + A_ij * A^ij / P^7 / 8");
    syst.add_def("eqNP    = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
    syst.add_def("eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");
    // end definitions

    // system of equations - see space for details regarding add_eq
    space.add_eq(syst, "eqNP    = 0", "N", "dn(N)");
    space.add_eq(syst, "eqP     = 0", "P", "dn(P)");
    space.add_eq(syst, "eqbet^i = 0", "bet^i", "dn(bet^i)");

    // boundary conditions on variable fields at infinity
    syst.add_eq_bc(ndom - 1, OUTER_BC, "N     = 1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "P     = 1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "bet^i = 0");

    // quasi-equallibrium condition if binary orbital frequency is not fixed
    if (!fixed_omega && stage != ECC_RED) {
        space.add_eq_int_inf(syst, "integ(dn(N) + 2 * dn(P)) = 0");
    }
    // minimize ADM linear momenta at infinity, Pz is zero by symmetry
    // space.add_eq_int_inf(syst, "integ(COMx) - xaxis = 0");
    space.add_eq_int_inf(syst, "integ(intPx) = 0");
    space.add_eq_int_inf(syst, "integ(intPy) = 0");

    // Lapse condition BCs
    if (bconfig.control(USE_FIXED_LAPSE)) {
        syst.add_cst("nm", bconfig(FIXED_LAPSE, BCO1));
        syst.add_cst("np", bconfig(FIXED_LAPSE, BCO2));

        space.add_bc_sphere_one(syst, "N = nm");
        space.add_bc_sphere_two(syst, "N = np");
    } else {
        space.add_bc_sphere_one(syst, "dn(NP) = 0");
        space.add_bc_sphere_two(syst, "dn(NP) = 0");
    }

    // Apparent Horizon condition BCs
    space.add_bc_sphere_one(syst, "sm^j * D_j P + P / 4 * D^j sm_j + A_ij * sm^i * sm^j / P^3 / 4 = 0");
    space.add_bc_sphere_two(syst, "sp^j * D_j P + P / 4 * D^j sp_j + A_ij * sp^i * sp^j / P^3 / 4 = 0");

    // Irreducible Mass BCs
    space.add_eq_int_sphere_one(syst, "integ(intMsq) - Mm * Mm = 0 ");
    space.add_eq_int_sphere_two(syst, "integ(intMsq) - Mp * Mp = 0 ");

    // in the case of corotation, the tangential term is zero by definition
    if (bconfig.control(COROT_BIN)) {
        space.add_bc_sphere_one(syst, "B^i = N / P^2 * sm^i");
        space.add_bc_sphere_two(syst, "B^i = N / P^2 * sp^i");
    } else {
        // otherwise we include equations on the boundary with the addition of the tangential spin component
        // note that these assume a conformally flat background metric
        space.add_bc_sphere_one(syst, "B^i = N / P^2 * sm^i + ExOme^i");
        if (std::isnan(bconfig.set(FIXED_BCOMEGA, BCO1)))
            space.add_eq_int_sphere_one(syst, "integ(intSm) - chim * CMm * CMm = 0 ");

        space.add_bc_sphere_two(syst, "B^i = N / P^2 * sp^i + ExOme^i");
        if (std::isnan(bconfig.set(FIXED_BCOMEGA, BCO2)))
            space.add_eq_int_sphere_two(syst, "integ(intSp) - chip * CMp * CMp = 0 ");
    }

    // exclude the excised regions
    for (int i : excluded_doms) {
        syst.add_eq_full(i, "N = 0");
        syst.add_eq_full(i, "P = 0");
        syst.add_eq_full(i, "bet^i = 0");
    }

    if (rank == 0)
        print_diagnostics(syst, 0, 0);

    // begin iterative solver
    bool endloop = false;
    int ite = 1;
    double conv;
    const SolverRuntimeConfig solver_config = this->with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), stage_text);
    while (!endloop) {
        endloop = newton_step_with_consensus(syst, bconfig.template seq_setting_as<double>(PREC), conv, solver_config, ite == 1);

        std::stringstream ss;
        ss << "bbh_" + stage_text << ite - 1;
        bconfig.set_filename(ss.str());
        if (rank == 0) {
            print_diagnostics(syst, ite, conv);

            if (bconfig.control(CHECKPOINT))
                bco_utils::save_to_file(ss, space, bconfig, conf, lapse, shift);
        }
        update_fields(cfields, coord_vectors, {}, xo, xc1, xc2, &syst);
        ite++;
        if (check_max_iter_exceeded(rank, ite, conv))
            break;
    }

    // in the case of corotation or fixed local spin frequency, we calculate MCH accordingly
    if (!std::isnan(bconfig.set(FIXED_BCOMEGA, BCO1)) || bconfig.control(COROT_BIN))
        bconfig(MCH, BCO1) = bco_utils::syst_mch(syst, space, "intSm", space.BH1 + 2);
    if (!std::isnan(bconfig.set(FIXED_BCOMEGA, BCO2)) || bconfig.control(COROT_BIN))
        bconfig(MCH, BCO2) = bco_utils::syst_mch(syst, space, "intSp", space.BH2 + 2);

    bconfig.set_filename(converged_filename(stage_text));
    if (rank == 0)
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift);

    return exit_status;
}


} // namespace Kadath
