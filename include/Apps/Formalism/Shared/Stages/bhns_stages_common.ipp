#pragma once
// Shared bodies of the BHNS hydrostatic-equilibrium and hydro-rescaling stages.
//
// The sym (bhns_xcts_solver) and nosym (bhns_xcts_nosym_solver) stage bodies are
// identical except for the in-star shift (eqbet) definition: the nosym solver
// appends the rotational-gauge forcing "+ muz1 * mmz^i" on the NS domains when
// rotational_gauge_enabled(). That one divergence is supplied by the
// matter_eqbet_def(d) callback; everything else is shared here.
//
// Templated on the solver type and befriended by both BHNS solver classes so the
// bodies can reach the derived-private field references (conf/lapse/logh/phi/
// shift, xc1/xc2/xo, excluded_doms) and base-protected helpers
// (check_max_iter_exceeded). Mirrors ns_3d_xcts_solve_stages.hpp.

#include "mpi.h"
#include "Apps/Helper/solver_base.hpp"
#include "Apps/Formalism/Shared/omega_mode.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"
#include "Apps/Helper/newton_loop_runner.hpp"

#include <sstream>
#include <string>

namespace Kadath {

template <typename Solver, typename MatterEqbetDef, typename PzVelocity, typename PzConstraint>
int bhns_run_hydrostatic_equilibrium_stage(Solver& solver, STAGE stage, OmegaMode omega_mode,
                                           const std::string stage_text, MatterEqbetDef&& matter_eqbet_def,
                                           PzVelocity&& pz_velocity, PzConstraint&& pz_constraint)
{

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exit_status = EXIT_SUCCESS;

    // Fixed Omega holds "ome"/"xaxis" constant and drops the ADM-momentum
    // integrals; Free solves for them. Replaces the in-body config-flag read.
    const bool fixed_omega = (omega_mode == OmegaMode::Fixed);

    auto current_file = solver.bconfig.config_filename_abs();
    bco_utils::sync_mirr_from_mch(solver.bconfig, BCO2);

    // get central values of the logarithmic enthalpy
    double loghc = bco_utils::get_boundary_val(solver.space.NS, solver.logh, INNER_BC);

    update_fields(solver.cfields, solver.coord_vectors, {}, solver.xo, solver.xc1, solver.xc2);

    if (rank == 0 && stage == FORCE_BALANCE)
        std::cout << "############################################" << std::endl
                  << "FORCE_BALANCE - Hydrostatic equilibrium stage\n"
                  << "using von Neumann lapse condition on the BH\n"
                  << "############################################" << std::endl;
    else if (rank == 0 && stage == QUASI_EQUIL)
        std::cout << "############################################" << std::endl
                  << "QUASI_EQUIL - Hydrostatic equilibrium stage\n"
                  << "using fixed lapse BC on the BH\n"
                  << "############################################" << std::endl;

    update_fields(solver.cfields, solver.coord_vectors, {}, solver.xo, solver.xc1, solver.xc2);
    // setup a system of equations
    System_of_eqs syst(solver.space, 0, solver.ndom - 1);

    // the actual solution fields of the equations, i.e.
    // conformal factor, lapse, shift and (logarithmic) enthalpy
    // set this first before running syst_init()
    syst.add_var("H", solver.logh);

    // populate all the boiler-plate constants, variables, and definitions
    solver.syst_init(syst);

    // center of mass on the x-axis connecting both companions
    // and orbital angular frequency parameter
    // in this case, both are fixed by the two central force-balance
    // equations
    syst.add_var("yaxis", solver.bconfig(COMY));
    if (fixed_omega) {
        syst.add_cst("ome", solver.bconfig(GOMEGA));
        syst.add_cst("xaxis", solver.bconfig(COM));
    } else {
        syst.add_var("xaxis", solver.bconfig(COM));
        syst.add_var("ome", solver.bconfig(GOMEGA));
    }

    // central (logarithmic) enthalpy is a variable, fixed by the baryonic mass integral
    syst.add_var("Hc1", loghc);

    // orbital rotation vector field, corrected by the "center of mass" shift
    syst.add_def("Morb^i = mg^i + xaxis * ey^i + yaxis * ex^i");

    // full shift vector field, incoporating the orbital part
    // nosym: when omega is free (the ADM Px/Py integrals are active below) append
    // the uniform z-velocity closure (+ zvel * ez^i) so a tilted spin's ADM Pz has
    // an unknown to absorb it, pinned by integ(intPz) = 0 below. Fixed-omega drops
    // the momentum integrals, so zvel must be dropped too to keep the system square.
    // Empty (no-op) for the symmetric solver under z-reflection symmetry.
    std::string bigB{"B^i= bet^i + ome * Morb^i"};
    if (!fixed_omega)
        bigB += pz_velocity(syst, solver.bconfig);
    syst.add_def(bigB.c_str());

    // the actual equations, defined differently in the different domains
    for (int d = 0; d < solver.ndom; d++) {
        // if outside the stellar domains, without matter sources
        // resort to the source-free constraint equations
        // and set matter (and velocity potential) to zero
        if (d >= solver.space.ADAPTEDNS + 1) {
            if (!solver.bconfig.control(COROT_BIN))
                syst.add_eq_full(d, "phi= 0");

            syst.add_eq_full(d, "H  = 0");

            syst.add_def(d, "eqP     = D^i D_i P + A_ij * A^ij / P^7 / 8");
            syst.add_def(d, "eqNP    = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
            syst.add_def(d, "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");

        }
        // in case of the domains describing the NS interior
        // define all derived matter related quantities
        // and enforce the transformed contraint equations (i.e. multiplied by p / rho)
        else {
            // 3-velocity and first integral of the Euler equation
            // in case of corotation
            if (solver.bconfig.control(COROT_BIN)) {
                syst.add_def(d, "U^i    = B^i / N");
                syst.add_def(d, "Usquare= P^4 * U_i * U^i");
                syst.add_def(d, "Wsquare= 1 / (1 - Usquare)");
                syst.add_def(d, "W      = sqrt(Wsquare)");
                syst.add_def(d, "firstint = log(h * N / W)");
            }
            // 3-velocity, (approximate) first integral of the Euler equation
            // and velocity potential equation
            // in case of irrotational or spinning companions
            else {
                syst.add_def(d, "Wsquare= eta^i * eta_i / h^2 / P^4 + 1.");
                syst.add_def(d, "W      = sqrt(Wsquare)");
                syst.add_def(d, "U^i    = eta^i / P^4 / h / W");
                syst.add_def(d, "Usquare= P^4 * U_i * U^i");
                syst.add_def(d, "V^i    = N * U^i - B^i");
                syst.add_def(d, "firstint = log(h * N / W + D_i phi * V^i)");
                syst.add_def(d, "eqphi  = P^6 * W * V^i * D_i H + dHdlnrho * D_i (P^6 * W * V^i)");
            }
            // transformed source terms
            syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
            syst.add_def(d, "Stilde = 3 * press * delta + (Etilde + press * delta) * Usquare");
            syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");

            // transformed constraint equations
            syst.add_def(d, "eqP    = delta * D^i D_i P + A_ij * A^ij / P^7 / 8 * delta + 4piG / 2. * P^5 * Etilde");
            syst.add_def(d, "eqNP   = delta * D^i D_i NP - 7. / 8. * NP / P^8 * delta * A_ij *A^ij "
                            "- 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
            syst.add_def(d, matter_eqbet_def(d).c_str());

            // baryonic mass volume integrant
            syst.add_def(d, "intMb  = P^6 * rho * W");
            // quasi-local ADM mass volume integrant
            syst.add_def(d, "intM   = - D_i D^i P * 2. / 4piG");
        }
    }
    // add equations to the system and demand continuity
    // along the normal of the domains
    solver.space.add_eq(syst, "eqNP= 0", "N", "dn(N)");
    solver.space.add_eq(syst, "eqP= 0", "P", "dn(P)");
    solver.space.add_eq(syst, "eqbet^i= 0", "bet^i", "dn(bet^i)");

    // boundary conditions at infinity
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "N=1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "P=1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "bet^i=0");

    // boundary conditions defining the boundary of the adapted domains, i.e. vanishing matter
    syst.add_eq_bc(solver.space.ADAPTEDNS, OUTER_BC, "H = 0");

    // in case of irrotational or spinning companions
    // solve also for the velocity potential
    if (!solver.bconfig.control(COROT_BIN)) {
        syst.add_eq_bc(solver.space.ADAPTEDNS, OUTER_BC, "V^i * D_i H = 0");

        for (int i = solver.space.NS; i < solver.space.ADAPTEDNS; ++i) {
            syst.add_eq_vel_pot(i, 2, "eqphi = 0", "phi=0");
            syst.add_eq_matching(i, OUTER_BC, "phi");
            syst.add_eq_matching(i, OUTER_BC, "dn(phi)");
        }
        syst.add_eq_vel_pot(solver.space.ADAPTEDNS, 2, "eqphi = 0", "phi=0");

        if (std::isnan(solver.bconfig.set(FIXED_BCOMEGA, BCO1)))
            solver.space.add_eq_int_outer_NS(syst, "integ(intS1) / Madm1 / Madm1 = chi1");

        solver.space.add_bc_sphere_two(syst, "B^i = N / P^2 * sp^i + s^i");
        if (std::isnan(solver.bconfig.set(FIXED_BCOMEGA, BCO2)))
            solver.space.add_eq_int_BH(syst, "integ(intS2) - chi2 * Mch * Mch = 0 ");
    } else {
        solver.space.add_bc_sphere_two(syst, "B^i = N / P^2 * sp^i");
    }

    // force-balance equations at the center of each star
    // to fix the orbital frequency as well as the "center of mass"
    // (the latter can get very inaccurate in extreme configurations
    // with large residual linear momenta, for which the next stage is given)
    Index posori1(solver.space.get_domain(solver.space.NS)->get_nbr_points());
    syst.add_eq_val(solver.space.NS, "ex^i * D_i H", posori1);

    // add the first integral to get (approximate) hydrostatic equilibrium
    syst.add_eq_first_integral(solver.space.NS, solver.space.ADAPTEDNS, "firstint", "H - Hc1");

    // fix the central enthalpy by baryonic mass volume integrals
    solver.space.add_eq_int_volume(syst, solver.space.NS, solver.space.ADAPTEDNS, "integvolume(intM) = qlMadm1");

    // compute a quasi-local approximation of the ADM component masses
    solver.space.add_eq_int_volume(syst, solver.space.NS, solver.space.ADAPTEDNS, "integvolume(intMb) = Mb1");

    if (!fixed_omega) {
        solver.space.add_eq_int_inf(syst, "integ(intPy) = 0");
        solver.space.add_eq_int_inf(syst, "integ(intPx) = 0");
        // nosym: pin the z-velocity closure (zvel) with integ(intPz) = 0 so a tilted
        // spin's out-of-plane ADM momentum is controlled (the in-plane Px/Py are
        // closed above). zvel is only registered on this same free-omega path, so the
        // unknown/equation count stays square. No-op for the symmetric solver.
        pz_constraint(solver.space, syst);
    }

    if (stage == QUASI_EQUIL) {
        // For QUASI_EQUIL we use the fixed lapse condition.
        solver.space.add_bc_sphere_two(syst, "N = n0");
    } else {
        // FORCE_BALANCE and ECC_RED use the von Neumann lapse condition.
        solver.space.add_bc_sphere_two(syst, "sp^j * D_j NP = 0");
    }

    solver.space.add_bc_sphere_two(syst, "sp^j * D_j P + P / 4 * D^j sp_j + A_ij * sp^i * sp^j / P^3 / 4 = 0");
    solver.space.add_eq_int_BH(syst, "integ(intMsq) - Mirr * Mirr = 0 ");

    // excised domains of the BH
    for (int i : solver.excluded_doms) {
        syst.add_eq_full(i, "N = 0");
        syst.add_eq_full(i, "P = 0");
        syst.add_eq_full(i, "bet^i = 0");
    }

    // print initial diagnostics
    if (rank == 0)
        solver.print_diagnostics(syst, 0, 0);

    // iterative solver variables
    bool endloop = false;
    int ite = 1;
    double conv;
    const std::string final_stage = solver.bconfig.control(COROT_BIN)
                                        ? stage_text + "_COROT"
                                        : stage_text;
    const SolverRuntimeConfig solver_config = solver.with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), final_stage);

    // loop until desired convergence is achieved
    while (!endloop) {
        // do exactly one Newton step
        endloop = newton_step_with_consensus(syst, solver.bconfig.template seq_setting_as<double>(PREC), conv, solver_config, ite == 1);

        // recompute all coordinate dependent fields
        // to make sure that they are updated correctly along
        // with the changing adapted domains
        update_fields(solver.cfields, solver.coord_vectors, {}, solver.xo, solver.xc1, solver.xc2, &syst);

        // generate output filename for this iteration
        std::stringstream ss;
        ss << stage_text << ite - 1;
        solver.bconfig.set_filename(ss.str());

        // print diagnostics and output configuration as well as the binary data
        if (rank == 0) {
            solver.print_diagnostics(syst, ite, conv);
            if (solver.bconfig.control(CHECKPOINT)) {
                bco_utils::sync_mirr_from_mch(solver.bconfig, BCO2);
                bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift, solver.logh, solver.phi);
            }
        }

        ite++;
        if (solver.check_max_iter_exceeded(rank, ite, conv))
            break;
    }

    // since the ADM mass at infinite separation is not known
    // a priori for corotating stars,
    // we use the quasi-local measurement to update the ADM masses
    // approximately
    if (solver.bconfig.control(COROT_BIN)) {
        solver.bconfig.set(MADM, BCO1) = solver.bconfig(QLMADM, BCO1);
    }
    solver.bconfig.set_filename(solver.converged_filename(final_stage));

    // output final configuration and binary data
    if (rank == 0) {
        bco_utils::sync_mirr_from_mch(solver.bconfig, BCO2);
        bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift, solver.logh, solver.phi);
    }
    return exit_status;
}

template <typename Solver, typename MatterEqbetDef, typename PzVelocity, typename PzConstraint>
int bhns_run_hydro_rescaling_stages(Solver& solver, STAGE stage, OmegaMode omega_mode,
                                    std::string stage_text, MatterEqbetDef&& matter_eqbet_def,
                                    PzVelocity&& pz_velocity, PzConstraint&& pz_constraint)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exit_status = EXIT_SUCCESS;

    // This stage always holds the orbital frequency constant (fixed Omega); the
    // mode argument exists only for a uniform stage signature across binary apps.
    (void)omega_mode;
    bco_utils::sync_mirr_from_mch(solver.bconfig, BCO2);

    /*if(solution_exists(stage_text)) {
      if(rank == 0)
        std::cout << "Solved previously: " << solver.bconfig.config_filename_abs() << std::endl;
      return EXIT_SUCCESS;
    }*/

    if (rank == 0) {
        if (stage == ECC_RED) {
            std::cout << "############################" << std::endl
                      << "Eccentricity reduction step with fixed omega, chi = " << solver.bconfig(CHI, BCO1) << ","
                      << solver.bconfig(CHI, BCO2) << " and q = " << solver.bconfig(Q) << std::endl
                      << "############################" << std::endl;
        } else {
            std::cout << "############################" << std::endl
                      << "Hydro Rescaling stage using fixed omega, chi = " << solver.bconfig(CHI, BCO1) << ","
                      << solver.bconfig(CHI, BCO2) << " and q = " << solver.bconfig(Q) << std::endl
                      << "############################" << std::endl;
        }
    }

    if (stage == QUASI_EQUIL) {
        const auto pn_orbital_params =
            bco_utils::binary_pn_orbital_seed(solver.bconfig, solver.bconfig(MADM, BCO1), solver.bconfig(MCH, BCO2));
        solver.bconfig.set(ECC_OMEGA) = pn_orbital_params.omega;
        if (std::isnan(solver.bconfig.set(ADOT)))
            solver.bconfig.set(ADOT) = pn_orbital_params.adot;
        solver.bconfig.set(GOMEGA) = pn_orbital_params.omega;

        if (rank == 0)
            std::cout << "### Using PN estimate for omega! ###" << std::endl;
    } else if (stage == ECC_RED) {
        // determine whether to use PN estimates of the orbital frequency and adot
        // in case of the eccentricity stage
        if (std::isnan(solver.bconfig.set(ADOT)) || std::isnan(solver.bconfig.set(ECC_OMEGA)) || solver.bconfig.control(USE_PN)) {

            const auto pn_orbital_params =
                bco_utils::binary_pn_orbital_seed(solver.bconfig, solver.bconfig(MADM, BCO1), solver.bconfig(MCH, BCO2));
            solver.bconfig.set(ECC_OMEGA) = pn_orbital_params.omega;
            solver.bconfig.set(ADOT) = pn_orbital_params.adot;
            solver.bconfig.set(GOMEGA) = pn_orbital_params.omega;

            if (rank == 0)
                std::cout << "### Using PN estimate for adot and omega! ###" << std::endl;
        }
        solver.bconfig.set(GOMEGA) = solver.bconfig(ECC_OMEGA);
    }

    // setup background position vector field - only needed for ECC_RED stage
    Vector CART(solver.space, CON, solver.basis);
    CART = solver.cfields.cart();

    // residual scaling factors for the ethalpy
    // used to correct the baryonic mass
    double H_scale = 0;

    // set the shift of the "center of mass" along the y-axis
    // to zero, potentially fixing x-components of the ADM linear
    // momentum at infinity introduced by the eccentricity reduction
    // parameters
    if (std::isnan(solver.bconfig.set(COMY)))
        solver.bconfig.set(COMY) = 0.;

    // "background", unscaled logarithmic enthalpy from the previous step
    Scalar logh_const(solver.logh);
    logh_const.std_base();

    update_fields(solver.cfields, solver.coord_vectors, {}, solver.xo, solver.xc1, solver.xc2);

    // setup a system of equations
    System_of_eqs syst(solver.space, 0, solver.ndom - 1);

    // constant "background" enthalpy
    syst.add_cst("Hconst", logh_const);

    // residual scaling factors for the ethalpy
    syst.add_var("Hscale", H_scale);

    // definition of H
    // - rescaled in the star
    // - constant everywhere else
    for (int d = 0; d < solver.ndom; ++d) {
        if (d <= solver.space.ADAPTEDNS && d >= solver.space.NS)
            syst.add_def(d, "H  = Hconst * (1. + Hscale)");
        else
            syst.add_def(d, "H  = Hconst");
    }

    // populate all the boiler-plate constants, variables, and definitions
    solver.syst_init(syst);

    if (stage != ECC_RED) {
        // "center of mass" on the x-axis, connecting both stellar centers
        // These are fixed on the initial ID import as integrals at INF
        // cause instabilities in the initial solution
        syst.add_cst("xaxis", solver.bconfig(COM));
        syst.add_cst("yaxis", solver.bconfig(COMY));
    } else {
        // "center of mass" on the x-axis, connecting both stellar centers
        // fixed by the vanishing of the ADM linear momentum at infinity
        syst.add_var("xaxis", solver.bconfig(COM));

        // same on the y-axis in case finite momenta develope by
        // the eccentricity reduction parameters
        syst.add_var("yaxis", solver.bconfig(COMY));
    }

    // no additional force-balance is computed,
    // the matter distribution is fixed modulo the scaling factos above,
    // therefore the orbital frequency is a constant similar to
    // eccentricity reduced ID
    syst.add_cst("ome", solver.bconfig(GOMEGA));

    // orbital rotation vector field, corrected by the "center of mass" shift
    syst.add_def("Morb^i = mg^i + xaxis * ey^i + yaxis * ex^i");

    std::string bigB{"B^i = bet^i + ome * Morb^i"};
    if (stage == ECC_RED) {
        syst.add_cst("adot", solver.bconfig(ADOT));
        syst.add_cst("r", CART);
        syst.add_def("comr^i = r^i - xaxis * ex^i + yaxis * ey^i");
        // add contribution of ADOT to the total shift definition
        bigB += " + adot * comr^i";
        // nosym: close Pz with a uniform z-velocity unknown (paired with
        // integ(intPz) = 0 below). No-op (empty) for the symmetric solver.
        bigB += pz_velocity(syst, solver.bconfig);
    }

    // full shift vector including inertial + orbital contributions
    syst.add_def(bigB.c_str());

    // the actual equations, defined differently in the different domains
    for (int d = 0; d < solver.ndom; d++) {
        // if outside the stellar domains, without matter sources
        // resort to the source-free constraint equations
        // and set matter (and velocity potential) to zero
        if (d >= solver.space.ADAPTEDNS + 1) {
            if (!solver.bconfig.control(COROT_BIN))
                syst.add_eq_full(d, "phi= 0");

            syst.add_def(d, "eqP     = D^i D_i P + A_ij * A^ij / P^7 / 8");
            syst.add_def(d, "eqNP    = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
            syst.add_def(d, "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");

        }
        // in case of the domains harboring the two stars
        // define all derived matter related quantities
        // and enforce the transformed contraint equations (i.e. multiplied by p / rho)
        else {
            // 3-velocity and first integral of the Euler equation
            // in case of corotation
            if (solver.bconfig.control(COROT_BIN)) {
                syst.add_def(d, "U^i    = B^i / N");
                syst.add_def(d, "Usquare= P^4 * U_i * U^i");
                syst.add_def(d, "Wsquare= 1 / (1 - Usquare)");
                syst.add_def(d, "W      = sqrt(Wsquare)");
                syst.add_def(d, "firstint = log(h * N / W)");
            }
            // 3-velocity, (approximate) first integral of the Euler equation
            // and velocity potential equation
            // in case of irrotational or spinning companions
            else {
                syst.add_def(d, "Wsquare= eta^i * eta_i / h^2 / P^4 + 1.");
                syst.add_def(d, "W      = sqrt(Wsquare)");
                syst.add_def(d, "U^i    = eta^i / P^4 / h / W");
                syst.add_def(d, "Usquare= P^4 * U_i * U^i");
                syst.add_def(d, "V^i    = N * U^i - B^i");
                syst.add_def(d, "firstint = log(h * N / W + D_i phi * V^i)");
                syst.add_def(d, "eqphi  = P^6 * W * V^i * D_i H + dHdlnrho * D_i (P^6 * W * V^i)");
            }
            // transformed source terms
            syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
            syst.add_def(d, "Stilde = 3 * press * delta + (Etilde + press * delta) * Usquare");
            syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");

            // transformed constraint equations
            syst.add_def(d, "eqP    = delta * D^i D_i P + A_ij * A^ij / P^7 / 8 * delta + 4piG / 2. * P^5 * Etilde");
            syst.add_def(d, "eqNP   = delta * D^i D_i NP - 7. / 8. * NP / P^8 * delta * A_ij *A^ij "
                            "- 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
            syst.add_def(d, matter_eqbet_def(d).c_str());

            // baryonic mass volume integrant
            syst.add_def(d, "intMb  = P^6 * rho * W");
            // quasi-local ADM mass volume integrant
            syst.add_def(d, "intM   = - D_i D^i P * 2. / 4piG");
        }
    }
    // add equations to the system and demand continuity
    // along the normal of the domains
    solver.space.add_eq(syst, "eqNP= 0", "N", "dn(N)");
    solver.space.add_eq(syst, "eqP= 0", "P", "dn(P)");
    solver.space.add_eq(syst, "eqbet^i= 0", "bet^i", "dn(bet^i)");

    // boundary conditions at infinity
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "N=1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "P=1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "bet^i=0");

    // boundary conditions defining the boundary of the adapted domains, i.e. vanishing matter
    syst.add_eq_bc(solver.space.ADAPTEDNS, OUTER_BC, "H = 0");

    // determine equations to add based on corotation
    // or mixed spin binaries based on chi or fixed omega
    if (!solver.bconfig.control(COROT_BIN)) {
        syst.add_eq_bc(solver.space.ADAPTEDNS, OUTER_BC, "V^i * D_i H = 0");

        for (int i = solver.space.NS; i < solver.space.ADAPTEDNS; ++i) {
            syst.add_eq_vel_pot(i, 2, "eqphi = 0", "phi=0");
            syst.add_eq_matching(i, OUTER_BC, "phi");
            syst.add_eq_matching(i, OUTER_BC, "dn(phi)");
        }
        syst.add_eq_vel_pot(solver.space.ADAPTEDNS, 2, "eqphi = 0", "phi=0");

        if (std::isnan(solver.bconfig.set(FIXED_BCOMEGA, BCO1)))
            solver.space.add_eq_int_outer_NS(syst, "integ(intS1) / Madm1 / Madm1 = chi1");

        solver.space.add_bc_sphere_two(syst, "B^i = N / P^2 * sp^i + s^i");
        if (std::isnan(solver.bconfig.set(FIXED_BCOMEGA, BCO2)))
            solver.space.add_eq_int_BH(syst, "integ(intS2) - chi2 * Mch * Mch = 0 ");
    } else {
        solver.space.add_bc_sphere_two(syst, "B^i = N / P^2 * sp^i");
    }

    solver.space.add_eq_int_volume(syst, solver.space.NS, solver.space.ADAPTEDNS, "integvolume(intM) = qlMadm1");
    solver.space.add_eq_int_volume(syst, solver.space.NS, solver.space.ADAPTEDNS, "integvolume(intMb) = Mb1");

    if (stage == ECC_RED) {
        solver.space.add_eq_int_inf(syst, "integ(intPx) = 0");
        solver.space.add_eq_int_inf(syst, "integ(intPy) = 0");
        // nosym: pin the z-velocity closure with integ(intPz) = 0. zvel is only
        // registered on this same ECC_RED path, keeping the system square. No-op for sym.
        pz_constraint(solver.space, syst);
    }

    if (stage == QUASI_EQUIL)
        solver.space.add_bc_sphere_two(syst, "N = n0");
    else
        solver.space.add_bc_sphere_two(syst, "sp^j * D_j NP = 0");
    solver.space.add_bc_sphere_two(syst, "sp^j * D_j P + P / 4 * D^j sp_j + A_ij * sp^i * sp^j / P^3 / 4 = 0");

    solver.space.add_eq_int_BH(syst, "integ(intMsq) - Mirr * Mirr = 0 ");

    // excised domains of the BH
    for (int i : solver.excluded_doms) {
        syst.add_eq_full(i, "N = 0");
        syst.add_eq_full(i, "P = 0");
        syst.add_eq_full(i, "bet^i = 0");
    }

    // print initial diagnostics
    if (rank == 0)
        solver.print_diagnostics(syst, 0, 0);

    // iterative solver variables
    bool endloop = false;
    int ite = 1;
    double conv;
    const std::string final_stage = solver.bconfig.control(COROT_BIN)
                                        ? stage_text + "_COROT"
                                        : stage_text;
    const SolverRuntimeConfig solver_config = solver.with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), final_stage);

    // loop until desired convergence is achieved
    while (!endloop) {
        endloop = newton_step_with_consensus(syst, solver.bconfig.template seq_setting_as<double>(PREC), conv, solver_config, ite == 1);

        // recompute all coordinate dependent fields
        // to make sure that they are updated correctly along
        // with the changing adapted domains
        update_fields(solver.cfields, solver.coord_vectors, {}, solver.xo, solver.xc1, solver.xc2, &syst);

        // generate output filename for this iteration
        std::stringstream ss;
        ss << "total_" << ite - 1;
        solver.bconfig.set_filename(ss.str());

        // overwrite logh with scaled version for later use and output
        solver.logh = syst.give_val_def("H");

        // print diagnostics and output configuration as well as the binary data
        if (rank == 0) {
            solver.print_diagnostics(syst, ite, conv);
            if (solver.bconfig.control(CHECKPOINT)) {
                bco_utils::sync_mirr_from_mch(solver.bconfig, BCO2);
                bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift, solver.logh, solver.phi);
            }
        }

        ite++;
        if (solver.check_max_iter_exceeded(rank, ite, conv))
            break;
    }

    // since the ADM mass at infinite separation is not known
    // a priori for corotating stars,
    // we use the quasi-local measurement to update the ADM masses
    // approximately
    if (solver.bconfig.control(COROT_BIN)) {
        solver.bconfig.set(MADM, BCO1) = solver.bconfig(QLMADM, BCO1);
    }
    solver.bconfig.set_filename(solver.converged_filename(final_stage));

    // output final configuration and binary data
    if (rank == 0) {
        bco_utils::sync_mirr_from_mch(solver.bconfig, BCO2);
        bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift, solver.logh, solver.phi);
    }
    return exit_status;
}

} // namespace Kadath
