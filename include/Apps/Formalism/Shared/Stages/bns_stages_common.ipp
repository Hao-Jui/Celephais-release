#pragma once
// Shared bodies of the GR binary-NS force-balance and hydro-rescaling stages.
//
// The sym (bns_xcts_solver) and nosym (bns_xcts_nosym_solver) stage bodies are
// identical except for two nosym-only closures, both supplied here by callbacks:
//
//   * matter_eqbet_def(d) -> the in-star shift-equation definition string. The
//     nosym solver appends a per-star rigid-rotation gauge forcing
//     ("+ muz1 * mmz^i" on NS1, "+ muz2 * mpz^i" on NS2) when
//     rotational_gauge_enabled(); the sym solver returns the plain definition.
//
//   * pz_velocity(syst, bconfig) / pz_constraint(space, syst) -> the ADM
//     z-momentum closure used by the eccentricity stage of the nosym binary
//     (no z-reflection symmetry). Both are no-ops for the symmetric solver.
//
// Templated on the solver type and befriended by both BNS solver classes so the
// bodies can reach the derived-private field references (conf/lapse/logh/phi/
// shift, xc1/xc2/xo) and base-protected helpers (run_newton_loop). Mirrors
// Apps/Formalism/Shared/Stages/bhns_stages_common.ipp.

#include "mpi.h"
#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"
#include "Apps/Formalism/Shared/Stages/bns_stage_diagnostics.hpp"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace Kadath {
using namespace bns_diagnostics;

// =============================================================================
// Force-balance (hydrostatic equilibrium) stage
//
// The binary is solved using force balance equations in addition to solving
// the relativistic Euler equation to obtain a binary in hydrostatic equilibrium.
// =============================================================================

template <typename Solver, typename MatterEqbetDef, typename PzVelocity, typename PzConstraint>
int bns_run_force_balance_stage(Solver& solver, MatterEqbetDef&& matter_eqbet_def, PzVelocity&& pz_velocity,
                                PzConstraint&& pz_constraint)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exit_status = EXIT_SUCCESS;

    // Get central enthalpy values
    double loghc1 = bco_utils::get_boundary_val(solver.space.NS1, solver.logh, INNER_BC);
    double loghc2 = bco_utils::get_boundary_val(solver.space.NS2, solver.logh, INNER_BC);

    update_fields(solver.cfields, solver.coord_vectors, {}, solver.xo, solver.xc1, solver.xc2);

    // Print stage banner
    if (rank == 0) {
        std::cout << "############################" << std::endl
                  << "Force balance: Euler first integral, Omega solved, Mb fixed" << std::endl
                  << "chi = " << solver.bconfig(CHI, BCO1) << "," << solver.bconfig(CHI, BCO2)
                  << " and q = " << solver.bconfig(Q) << std::endl
                  << "############################" << std::endl;
    }

    // ---------------------------------------------------------------------------
    // System setup
    // ---------------------------------------------------------------------------
    System_of_eqs syst(solver.space, 0, solver.ndom - 1);

    syst.add_var("H", solver.logh);
    solver.syst_init(syst);

    const bool hydrostatic_y_cm_probe = env_flag("BNS_HYDRO_Y_CM", false);
    if (hydrostatic_y_cm_probe && std::isnan(solver.bconfig.set(COMY)))
        solver.bconfig.set(COMY) = 0.;

    // Variables fixed by force-balance equations
    syst.add_var("xaxis", solver.bconfig(COM));
    if (hydrostatic_y_cm_probe)
        syst.add_var("yaxis", solver.bconfig(COMY));
    syst.add_var("ome", solver.bconfig(GOMEGA));

    // Central enthalpy variables (fixed by baryonic mass integrals)
    syst.add_var("Hc1", loghc1);
    syst.add_var("Hc2", loghc2);

    // Orbital rotation with center-of-mass shift
    if (hydrostatic_y_cm_probe)
        syst.add_def("Morb^i = mg^i + xaxis * ey^i + yaxis * ex^i");
    else
        syst.add_def("Morb^i = mg^i + xaxis * ey^i");
    // nosym: append the uniform z-velocity closure (+ zvel * ez^i) so a tilted
    // spin's ADM Pz has an unknown to absorb it (pinned by integ(intPz) = 0
    // below). Empty (no-op) for the symmetric solver under z-reflection symmetry.
    std::string bigB{"B^i = bet^i + ome * Morb^i"};
    bigB += pz_velocity(syst, solver.bconfig);
    syst.add_def(bigB.c_str());

    // ---------------------------------------------------------------------------
    // Domain-specific definitions
    // ---------------------------------------------------------------------------
    for (int d = 0; d < solver.ndom; d++) {
        // Outside stellar domains: vacuum equations
        if (!((d >= solver.space.NS1 && d <= solver.space.ADAPTED1) ||
              (d >= solver.space.NS2 && d <= solver.space.ADAPTED2))) {
            if (!solver.bconfig.control(COROT_BIN))
                syst.add_eq_full(d, "phi = 0");

            syst.add_eq_full(d, "H = 0");
            syst.add_def(d, "eqP     = D^i D_i P + A_ij * A^ij / P^7 / 8");
            syst.add_def(d, "eqNP    = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
            syst.add_def(d, "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");
        }
        // Inside stellar domains: constraint equations (multiplied by p/rho)
        else {
            if (solver.bconfig.control(COROT_BIN)) {
                // Corotating case
                syst.add_def(d, "U^i     = B^i / N");
                syst.add_def(d, "Usquare = P^4 * U_i * U^i");
                syst.add_def(d, "Wsquare = 1 / (1 - Usquare)");
                syst.add_def(d, "W       = sqrt(Wsquare)");
                syst.add_def(d, "firstint = log(h * N / W)");
            } else {
                // Irrotational/spinning case
                syst.add_def(d, "Wsquare = eta^i * eta_i / h^2 / P^4 + 1.");
                syst.add_def(d, "W       = sqrt(Wsquare)");
                syst.add_def(d, "U^i     = eta^i / P^4 / h / W");
                syst.add_def(d, "Usquare = P^4 * U_i * U^i");
                syst.add_def(d, "V^i     = N * U^i - B^i");
                syst.add_def(d, "firstint = log(h * N / W + D_i phi * V^i)");
                syst.add_def(d, "eqphi   = P^6 * W * V^i * D_i H + dHdlnrho * D_i (P^6 * W * V^i)");
            }

            syst.add_def(d, "Etilde   = press * h * Wsquare - press * delta");
            syst.add_def(d, "Stilde   = 3 * press * delta + press * h * Wsquare * Usquare");
            syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");

            // Constraint equations (eqbet supplied by the solver: nosym appends gauge forcing)
            syst.add_def(d, "eqP    = delta * D^i D_i P + A_ij * A^ij / P^7 / 8 * delta + 4piG / 2. * P^5 * Etilde");
            syst.add_def(d, "eqNP   = delta * D^i D_i NP - 7. / 8. * NP / P^8 * delta * A_ij * A^ij "
                            "- 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
            syst.add_def(d, matter_eqbet_def(d).c_str());

            // Volume integrands
            syst.add_def(d, "intMb = P^6 * rho * W");
            syst.add_def(d, "intM  = -D_i D^i P * 2. / 4piG");
        }
    }

    // ---------------------------------------------------------------------------
    // Equations and boundary conditions
    // ---------------------------------------------------------------------------
    solver.space.add_eq(syst, "eqNP = 0", "N", "dn(N)");
    solver.space.add_eq(syst, "eqP = 0", "P", "dn(P)");
    solver.space.add_eq(syst, "eqbet^i = 0", "bet^i", "dn(bet^i)");

    // Infinity boundary conditions
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "N = 1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "P = 1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "bet^i = 0");

    // Stellar surface boundary conditions
    syst.add_eq_bc(solver.space.ADAPTED1, OUTER_BC, "H = 0");
    syst.add_eq_bc(solver.space.ADAPTED2, OUTER_BC, "H = 0");

    // Velocity potential equations (irrotational/spinning case)
    if (!solver.bconfig.control(COROT_BIN)) {
        syst.add_eq_bc_projected(solver.space.ADAPTED1, OUTER_BC,
                                 "V^i * D_i H = 0", -1, nullptr, "phi");
        syst.add_eq_bc_projected(solver.space.ADAPTED2, OUTER_BC,
                                 "V^i * D_i H = 0", -1, nullptr, "phi");

        for (int i = solver.space.NS1; i < solver.space.ADAPTED1; ++i) {
            syst.add_eq_vel_pot(i, 2, "eqphi = 0", "phi = 0", "phi", true);
            syst.add_eq_matching(i, OUTER_BC, "phi");
            syst.add_eq_matching(i, OUTER_BC, "dn(phi)");
        }
        syst.add_eq_vel_pot(solver.space.ADAPTED1, 2, "eqphi = 0", "phi = 0", "phi", true);

        for (int i = solver.space.NS2; i < solver.space.ADAPTED2; ++i) {
            syst.add_eq_vel_pot(i, 2, "eqphi = 0", "phi = 0", "phi", true);
            syst.add_eq_matching(i, OUTER_BC, "phi");
            syst.add_eq_matching(i, OUTER_BC, "dn(phi)");
        }
        syst.add_eq_vel_pot(solver.space.ADAPTED2, 2, "eqphi = 0", "phi = 0", "phi", true);

        // Spin equations
        if (std::isnan(solver.bconfig.set(FIXED_BCOMEGA, BCO1))) {
            solver.space.add_eq_int_outer_sphere_one(syst, "integ(intS1) / Madm1 / Madm1 = chi1");
            syst.set_last_eq_int_reflection_sector(+1);
        }
        if (std::isnan(solver.bconfig.set(FIXED_BCOMEGA, BCO2))) {
            solver.space.add_eq_int_outer_sphere_two(syst, "integ(intS2) / Madm2 / Madm2 = chi2");
            syst.set_last_eq_int_reflection_sector(+1);
        }
    }

    // Force balance at stellar centers
    Index posori1(solver.space.get_domain(solver.space.NS1)->get_nbr_points());
    syst.add_eq_val(solver.space.NS1, "ex^i * D_i H", posori1);
    syst.set_last_eq_int_reflection_sector(+1);

    Index posori2(solver.space.get_domain(solver.space.NS2)->get_nbr_points());
    syst.add_eq_val(solver.space.NS2, "ex^i * D_i H", posori2);
    syst.set_last_eq_int_reflection_sector(+1);

    if (hydrostatic_y_cm_probe) {
        solver.space.add_eq_int_inf(syst, "integ(intPy) = 0");
        syst.set_last_eq_int_reflection_sector(-1);
    }

    // nosym: pin the z-velocity closure (zvel) with integ(intPz) = 0 so a tilted
    // spin's out-of-plane ADM momentum is controlled (the in-plane Px/Py are
    // closed by the x-axis centering above). No-op for the symmetric solver.
    pz_constraint(solver.space, syst);

    // First integral for hydrostatic equilibrium
    syst.add_eq_first_integral(solver.space.NS1, solver.space.ADAPTED1,
                               "firstint", "H - Hc1", true);
    syst.add_eq_first_integral(solver.space.NS2, solver.space.ADAPTED2,
                               "firstint", "H - Hc2", true);

    // Mass integrals
    solver.space.add_eq_int_volume(syst, solver.space.NS1, solver.space.ADAPTED1, "integvolume(intMb) = Mb1");
    syst.set_last_eq_int_reflection_sector(+1);
    solver.space.add_eq_int_volume(syst, solver.space.NS2, solver.space.ADAPTED2, "integvolume(intMb) = Mb2");
    syst.set_last_eq_int_reflection_sector(+1);
    solver.space.add_eq_int_volume(syst, solver.space.NS1, solver.space.ADAPTED1, "integvolume(intM) = qlMadm1");
    syst.set_last_eq_int_reflection_sector(+1);
    solver.space.add_eq_int_volume(syst, solver.space.NS2, solver.space.ADAPTED2, "integvolume(intM) = qlMadm2");
    syst.set_last_eq_int_reflection_sector(+1);

    // ---------------------------------------------------------------------------
    // Pre-Newton baryon-mass rebalance (regrid diagnostic landed in evidence
    // bundles bns-res9to11-error-init-breakdown / bns-res11to13-error-init-
    // breakdown): at regrid, eq_int[4] = integvolume(intMb) - Mb1 and
    // eq_int[5] = integvolume(intMb) - Mb2 own ~100 % of the iter-0
    // Newton residual infinity-norm; field PDE residuals are 11-20× smaller.
    // Secant on Hc1 then Hc2 with logh constant-shift updates nulls those
    // two integrals before the Newton loop. The default path evaluates only
    // each selected intMb row and its definition closure; the legacy full
    // residual path remains available for qualification.
    //
    // Env-gated. The usual irrotational, non-FIXED_BCOMEGA FORCE_BALANCE
    // layout is:
    //   [0] chi1   [1] chi2   [2] Px   [3] Py
    //   [4] Mb1    [5] Mb2    [6] qlMadm1   [7] qlMadm2
    // The code resolves the Mb rows by expression and permits an explicit
    // BNS_HC_PRECORRECT_IDX_{MB1,MB2}=<idx> override.
    if (env_flag("BNS_HC_PRECORRECT", true)) {
        // Resolve the baryon-mass integral-equation indices by EXPRESSION, so the
        // pre-correction is robust to the active constraint set (spin and nosym
        // momentum rows shift the fixed eq_int positions; the old hard-coded 4/5
        // was only valid for the irrotational symmetric layout). An explicit env
        // index wins; otherwise auto-detect; a star with no Mb constraint is skipped.
        const int mb1_env = env_int("BNS_HC_PRECORRECT_IDX_MB1", -1);
        const int mb2_env = env_int("BNS_HC_PRECORRECT_IDX_MB2", -1);
        const int mb1_idx = (mb1_env >= 0) ? mb1_env : syst.eq_int_index_of("intMb) = Mb1");
        const int mb2_idx = (mb2_env >= 0) ? mb2_env : syst.eq_int_index_of("intMb) = Mb2");
        const int max_iters = env_int("BNS_HC_PRECORRECT_MAX_ITERS", 20);
        const double tol = env_double("BNS_HC_PRECORRECT_TOL", 1e-6);
        const double initial_step = env_double("BNS_HC_PRECORRECT_STEP", 0.01);
        bool selective_residual = env_flag("BNS_HC_PRECORRECT_SELECTIVE", true);
        const bool selective_selftest =
            selective_residual && env_flag("BNS_HC_PRECORRECT_SELECTIVE_SELFTEST", false);
        int selective_selftest_checks = 0;
        bool selective_selftest_fallback = false;

        Scalar logh_snapshot(solver.logh);

        auto apply_hc_shift = [&](int dom_lo, int dom_hi, double delta) {
            for (int d = dom_lo; d <= dom_hi; ++d) {
                solver.logh.set_domain(d) = logh_snapshot(d) + delta;
            }
            solver.logh.std_base();
        };

        auto eval_residual = [&](int dom_lo, int dom_hi, double delta,
                                 double& hc_var, double hc_init, int eq_idx) {
            apply_hc_shift(dom_lo, dom_hi, delta);
            hc_var = hc_init + delta;
            if (!selective_residual) {
                Array<double> res = syst.sec_member();
                return res(eq_idx);
            }

            const double selected = syst.sec_member_eq_int(eq_idx);
            if (!selective_selftest)
                return selected;

            Array<double> reference = syst.sec_member();
            const double expected = reference(eq_idx);
            ++selective_selftest_checks;
            if (std::memcmp(&selected, &expected, sizeof(double)) != 0) {
                selective_residual = false;
                selective_selftest_fallback = true;
                if (rank == 0) {
                    std::cout << "BNS Hc precorrect selective self-test: mismatch at eq_int["
                              << eq_idx << "]; falling back to full residuals" << std::endl;
                }
            }
            // Qualification follows the legacy value even on a mismatch, so
            // the self-test cannot perturb the Hc secant trajectory.
            return expected;
        };

        auto secant_one_star = [&](const char* star_label,
                                   int dom_lo, int dom_hi,
                                   double& hc_var, int eq_idx) -> double {
            const double hc_init = hc_var;
            double delta0 = 0.0;
            double r0 = eval_residual(dom_lo, dom_hi, delta0, hc_var, hc_init, eq_idx);
            if (rank == 0) {
                std::cout << "BNS Hc precorrect " << star_label
                          << ": initial residual=" << r0
                          << " (hc=" << hc_init << ")\n";
            }
            if (std::abs(r0) <= tol) {
                if (rank == 0)
                    std::cout << "BNS Hc precorrect " << star_label
                              << ": already within tol, skipping" << std::endl;
                apply_hc_shift(dom_lo, dom_hi, 0.0);
                hc_var = hc_init;
                return r0;
            }
            double delta1 = (r0 > 0) ? -initial_step : initial_step;
            double r1 = eval_residual(dom_lo, dom_hi, delta1, hc_var, hc_init, eq_idx);
            double best_delta = (std::abs(r1) < std::abs(r0)) ? delta1 : delta0;
            double best_res = std::min(std::abs(r0), std::abs(r1));
            for (int it = 0; it < max_iters; ++it) {
                if (std::abs(r1) <= tol)
                    break;
                if (std::abs(r1 - r0) < 1e-18)
                    break;
                double delta2 = delta1 - r1 * (delta1 - delta0) / (r1 - r0);
                const double clamp = 4.0 * std::abs(initial_step);
                delta2 = std::max(-clamp, std::min(clamp, delta2));
                double r2 = eval_residual(dom_lo, dom_hi, delta2, hc_var, hc_init, eq_idx);
                if (std::abs(r2) < best_res) {
                    best_res = std::abs(r2);
                    best_delta = delta2;
                }
                delta0 = delta1;
                r0 = r1;
                delta1 = delta2;
                r1 = r2;
            }
            double final_residual = r1;
            if (std::abs(r1) > best_res)
                final_residual = eval_residual(dom_lo, dom_hi, best_delta, hc_var, hc_init, eq_idx);
            if (rank == 0) {
                std::cout << "BNS Hc precorrect " << star_label
                          << ": final residual=" << final_residual
                          << " delta_hc=" << (hc_var - hc_init)
                          << " hc_new=" << hc_var << std::endl;
            }
            return final_residual;
        };

        if (rank == 0 && mb1_idx >= 0 && mb2_idx >= 0) {
            double mb1_start = 0.0;
            double mb2_start = 0.0;
            if (selective_residual) {
                mb1_start = eval_residual(solver.space.NS1, solver.space.ADAPTED1, 0.0,
                                          loghc1, loghc1, mb1_idx);
                mb2_start = eval_residual(solver.space.NS2, solver.space.ADAPTED2, 0.0,
                                          loghc2, loghc2, mb2_idx);
            } else {
                const Array<double> res_pre = syst.sec_member();
                mb1_start = res_pre(mb1_idx);
                mb2_start = res_pre(mb2_idx);
            }
            std::cout << "BNS Hc precorrect: starting Mb1=" << mb1_start
                      << " Mb2=" << mb2_start << std::endl;
        }
        if (mb1_idx >= 0)
            (void)secant_one_star("NS1", solver.space.NS1, solver.space.ADAPTED1, loghc1, mb1_idx);
        logh_snapshot = solver.logh;
        if (mb2_idx >= 0)
            (void)secant_one_star("NS2", solver.space.NS2, solver.space.ADAPTED2, loghc2, mb2_idx);
        if (rank == 0 && (mb1_idx >= 0 || mb2_idx >= 0)) {
            Array<double> res_post = syst.sec_member();
            double max_post = 0.0;
            for (std::size_t i = 0; i < res_post.get_nbr(); ++i)
                max_post = std::max(max_post, std::abs(res_post(static_cast<int>(i))));
            std::cout << "BNS Hc precorrect: post Error init = " << max_post;
            if (mb1_idx >= 0)
                std::cout << " Mb1_res=" << res_post(mb1_idx);
            if (mb2_idx >= 0)
                std::cout << " Mb2_res=" << res_post(mb2_idx);
            std::cout << std::endl;
            if (selective_selftest) {
                std::cout << "BNS Hc precorrect selective self-test: "
                          << (selective_selftest_fallback ? "FALLBACK" : "PASS")
                          << " (checks=" << selective_selftest_checks << ")" << std::endl;
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Newton iteration
    // ---------------------------------------------------------------------------
    if (rank == 0) {
        if (env_flag("BNS_PRINT_STRUCTURE", false))
            syst.print_system_structure(std::cout);
        solver.print_diagnostics(syst, 0, 0);
        if (env_flag("BNS_PRINT_DIMENSIONS", false)) {
            syst.sec_member();
            std::cout << "System dimensions: " << syst.get_nbr_conditions()
                      << " x " << syst.get_nbr_unknowns() << std::endl;
        }
        print_selected_equation_rows(syst);
        probe_jacobian_columns(syst);
        compare_jx_columns(syst);
    }

    if (env_flag("BNS_STOP_AFTER_INIT", false))
        return exit_status;

    const auto coord_binding = bind_coordinate_fields(syst, solver.coord_vectors);
    const std::string final_stage = solver.bconfig.control(COROT_BIN)
                                        ? "FORCE_BALANCE_COROT"
                                        : "FORCE_BALANCE";
    const SolverRuntimeConfig stage_solver_config =
        solver.with_stage_mumps_tree_cache(
            SolverRuntimeConfig::from_environment()
                .with_stage_default_backend(NewtonBackend::JfnkMumps),
            final_stage);

    const bool aborted = solver.run_newton_loop(syst, stage_solver_config, [&](int ite, double& conv) -> bool {
        update_fields(solver.cfields, solver.coord_vectors, {}, solver.xo, solver.xc1, solver.xc2, coord_binding);

        std::stringstream ss;
        ss << "total_" << ite - 1;
        solver.bconfig.set_filename(ss.str());

        if (rank == 0) {
            solver.print_diagnostics(syst, ite, conv);
            print_selected_equation_rows(syst);
            if (solver.bconfig.control(CHECKPOINT))
                bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift,
                                        solver.logh, solver.phi);
        }

        if (env_int("BNS_STOP_AFTER_STEP", 0) == ite)
            return false;  // request early stage abort

        return true;       // continue
    });

    if (aborted)
        return exit_status;

    // ---------------------------------------------------------------------------
    // Finalize
    // ---------------------------------------------------------------------------
    if (solver.bconfig.control(COROT_BIN)) {
        solver.bconfig.set(MADM, BCO1) = solver.bconfig(QLMADM, BCO1);
        solver.bconfig.set(MADM, BCO2) = solver.bconfig(QLMADM, BCO2);
    }
    solver.bconfig.set_filename(solver.converged_filename(final_stage));

    if (rank == 0)
        bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift,
                                solver.logh, solver.phi);

    return exit_status;
}

// =============================================================================
// Fixed-Omega hydro-rescaling stage used by QUASI_EQUIL and ECC_RED.
//
// The binary is solved with fixed orbital velocity (PN estimate or
// user-supplied ecc_omega/adot). Matter fields are rescaled based on the
// fixed baryonic mass of each star; the ADM momentum integrals close the
// center-of-mass unknowns. The nosym solver additionally closes the ADM
// z-momentum via the pz_velocity / pz_constraint hooks (no-ops for sym).
// =============================================================================

template <typename Solver, typename MatterEqbetDef, typename PzVelocity, typename PzConstraint>
int bns_run_hydro_rescaling_stage(Solver& solver, STAGE stage, const std::string& stage_text,
                                  MatterEqbetDef&& matter_eqbet_def, PzVelocity&& pz_velocity,
                                  PzConstraint&& pz_constraint)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exit_status = EXIT_SUCCESS;
    const bool eccentricity_stage = (stage == ECC_RED);

    // Print stage banner
    if (rank == 0) {
        std::cout << "############################" << std::endl;
        if (eccentricity_stage)
            std::cout << "Eccentricity reduction step with fixed omega, ";
        else
            std::cout << "Hydro Rescaling stage using PN fixed omega, ";
        std::cout << "chi = " << solver.bconfig(CHI, BCO1) << "," << solver.bconfig(CHI, BCO2)
                  << " and q = " << solver.bconfig(Q) << std::endl
                  << "############################" << std::endl;
    }

    // Setup PN parameters for the fixed-Omega rescale stages.
    if (stage == QUASI_EQUIL) {
        const auto pn_orbital_params =
            bco_utils::binary_pn_orbital_seed(solver.bconfig, solver.bconfig(MADM, BCO1), solver.bconfig(MADM, BCO2));
        solver.bconfig.set(ECC_OMEGA) = pn_orbital_params.omega;
        if (std::isnan(solver.bconfig.set(ADOT)))
            solver.bconfig.set(ADOT) = pn_orbital_params.adot;
        solver.bconfig.set(GOMEGA) = pn_orbital_params.omega;
        if (rank == 0)
            std::cout << "### Using PN estimate for omega! ###" << std::endl;
    } else if (stage == ECC_RED) {
        if (std::isnan(solver.bconfig.set(ADOT)) || std::isnan(solver.bconfig.set(ECC_OMEGA)) ||
            solver.bconfig.control(USE_PN)) {
            const auto pn_orbital_params =
                bco_utils::binary_pn_orbital_seed(solver.bconfig, solver.bconfig(MADM, BCO1), solver.bconfig(MADM, BCO2));
            solver.bconfig.set(ECC_OMEGA) = pn_orbital_params.omega;
            solver.bconfig.set(ADOT) = pn_orbital_params.adot;
            solver.bconfig.set(GOMEGA) = pn_orbital_params.omega;
            if (rank == 0)
                std::cout << "### Using PN estimate for adot and omega! ###" << std::endl;
        }
        solver.bconfig.set(GOMEGA) = solver.bconfig(ECC_OMEGA);
    } else {
        KADATH_THROW("Unsupported BNS hydro-rescaling stage");
    }

    // Enthalpy scaling factors
    double H_scale1 = 0;
    double H_scale2 = 0;

    if (std::isnan(solver.bconfig.set(COMY)))
        solver.bconfig.set(COMY) = 0.;

    // Background enthalpy from previous step
    Scalar logh_const(solver.logh);
    logh_const.std_base();

    // Only the eccentricity-reduction formulation consumes the background
    // Cartesian position vector. Keep it alive for as long as the equation
    // system may reference its constant value.
    std::optional<Vector> CART;

    update_fields(solver.cfields, solver.coord_vectors, {}, solver.xo, solver.xc1, solver.xc2);

    // ---------------------------------------------------------------------------
    // System setup
    // ---------------------------------------------------------------------------
    System_of_eqs syst(solver.space, 0, solver.ndom - 1);

    syst.add_cst("Hconst", logh_const);
    syst.add_var("Hscale1", H_scale1);
    syst.add_var("Hscale2", H_scale2);

    // Enthalpy definitions by domain
    for (int d = 0; d < solver.ndom; d++) {
        if (!((d >= solver.space.NS1 && d <= solver.space.ADAPTED1) ||
              (d >= solver.space.NS2 && d <= solver.space.ADAPTED2))) {
            syst.add_def(d, "H = Hconst");
        }
    }
    for (int d = solver.space.NS1; d <= solver.space.ADAPTED1; ++d) {
        syst.add_def(d, "H = Hconst * (1. + Hscale1)");
    }
    for (int d = solver.space.NS2; d <= solver.space.ADAPTED2; ++d) {
        syst.add_def(d, "H = Hconst * (1. + Hscale2)");
    }

    solver.syst_init(syst);

    // Center-of-mass variables
    if (eccentricity_stage) {
        syst.add_var("xaxis", solver.bconfig(COM));
        syst.add_var("yaxis", solver.bconfig(COMY));
    } else {
        syst.add_cst("xaxis", solver.bconfig(COM));
        syst.add_cst("yaxis", solver.bconfig(COMY));
    }
    syst.add_cst("ome", solver.bconfig(GOMEGA));

    // Orbital rotation definition
    syst.add_def("Morb^i = mg^i + xaxis * ey^i + yaxis * ex^i");

    std::string bigB{"B^i = bet^i + ome * Morb^i"};
    if (eccentricity_stage) {
        CART.emplace(solver.space, CON, solver.basis);
        *CART = solver.cfields.cart();
        syst.add_cst("adot", solver.bconfig(ADOT));
        syst.add_cst("r", *CART);
        syst.add_def("comr^i = r^i - xaxis * ex^i + yaxis * ey^i");
        bigB += " + adot * comr^i";
        // nosym: close Pz with a uniform z-velocity unknown (paired with
        // integ(intPz) = 0 below). No-op (empty) for the symmetric solver.
        bigB += pz_velocity(syst, solver.bconfig);
    }
    syst.add_def(bigB.c_str());

    // ---------------------------------------------------------------------------
    // Domain-specific definitions
    // ---------------------------------------------------------------------------
    for (int d = 0; d < solver.ndom; d++) {
        // Outside stellar domains
        if (!((d >= solver.space.NS1 && d <= solver.space.ADAPTED1) ||
              (d >= solver.space.NS2 && d <= solver.space.ADAPTED2))) {
            if (!solver.bconfig.control(COROT_BIN))
                syst.add_eq_full(d, "phi = 0");

            syst.add_def(d, "eqP     = D^i D_i P + A_ij * A^ij / P^7 / 8");
            syst.add_def(d, "eqNP    = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
            syst.add_def(d, "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");
        }
        // Inside stellar domains
        else {
            if (solver.bconfig.control(COROT_BIN)) {
                syst.add_def(d, "U^i     = B^i / N");
                syst.add_def(d, "Usquare = P^4 * U_i * U^i");
                syst.add_def(d, "Wsquare = 1 / (1 - Usquare)");
                syst.add_def(d, "W       = sqrt(Wsquare)");
                syst.add_def(d, "firstint = log(h * N / W)");
            } else {
                syst.add_def(d, "Wsquare = eta^i * eta_i / h^2 / P^4 + 1.");
                syst.add_def(d, "W       = sqrt(Wsquare)");
                syst.add_def(d, "U^i     = eta^i / P^4 / h / W");
                syst.add_def(d, "Usquare = P^4 * U_i * U^i");
                syst.add_def(d, "V^i     = N * U^i - B^i");
                syst.add_def(d, "firstint = log(h * N / W + D_i phi * V^i)");
                syst.add_def(d, "eqphi   = P^6 * W * V^i * D_i H + dHdlnrho * D_i (P^6 * W * V^i)");
            }

            syst.add_def(d, "Etilde   = press * h * Wsquare - press * delta");
            syst.add_def(d, "Stilde   = 3 * press * delta + press * h * Wsquare * Usquare");
            syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");

            syst.add_def(d, "eqP    = delta * D^i D_i P + A_ij * A^ij / P^7 / 8 * delta + 4piG / 2. * P^5 * Etilde");
            syst.add_def(d, "eqNP   = delta * D^i D_i NP - 7. / 8. * NP / P^8 * delta * A_ij * A^ij "
                            "- 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
            syst.add_def(d, matter_eqbet_def(d).c_str());

            syst.add_def(d, "intMb = P^6 * rho * W");
            syst.add_def(d, "intM  = -D_i D^i P * 2. / 4piG");
        }
    }

    // ---------------------------------------------------------------------------
    // Equations and boundary conditions
    // ---------------------------------------------------------------------------
    solver.space.add_eq(syst, "eqNP = 0", "N", "dn(N)");
    solver.space.add_eq(syst, "eqP = 0", "P", "dn(P)");
    solver.space.add_eq(syst, "eqbet^i = 0", "bet^i", "dn(bet^i)");

    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "N = 1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "P = 1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "bet^i = 0");

    syst.add_eq_bc(solver.space.ADAPTED1, OUTER_BC, "H = 0");
    syst.add_eq_bc(solver.space.ADAPTED2, OUTER_BC, "H = 0");

    if (!solver.bconfig.control(COROT_BIN)) {
        syst.add_eq_bc_projected(solver.space.ADAPTED1, OUTER_BC,
                                 "V^i * D_i H = 0", -1, nullptr, "phi");
        syst.add_eq_bc_projected(solver.space.ADAPTED2, OUTER_BC,
                                 "V^i * D_i H = 0", -1, nullptr, "phi");

        for (int i = solver.space.NS1; i < solver.space.ADAPTED1; ++i) {
            syst.add_eq_vel_pot(i, 2, "eqphi = 0", "phi = 0", "phi", true);
            syst.add_eq_matching(i, OUTER_BC, "phi");
            syst.add_eq_matching(i, OUTER_BC, "dn(phi)");
        }
        syst.add_eq_vel_pot(solver.space.ADAPTED1, 2, "eqphi = 0", "phi = 0", "phi", true);

        for (int i = solver.space.NS2; i < solver.space.ADAPTED2; ++i) {
            syst.add_eq_vel_pot(i, 2, "eqphi = 0", "phi = 0", "phi", true);
            syst.add_eq_matching(i, OUTER_BC, "phi");
            syst.add_eq_matching(i, OUTER_BC, "dn(phi)");
        }
        syst.add_eq_vel_pot(solver.space.ADAPTED2, 2, "eqphi = 0", "phi = 0", "phi", true);

        if (std::isnan(solver.bconfig.set(FIXED_BCOMEGA, BCO1))) {
            solver.space.add_eq_int_outer_sphere_one(syst, "integ(intS1) / Madm1 / Madm1 = chi1");
            syst.set_last_eq_int_reflection_sector(+1);
        }
        if (std::isnan(solver.bconfig.set(FIXED_BCOMEGA, BCO2))) {
            solver.space.add_eq_int_outer_sphere_two(syst, "integ(intS2) / Madm2 / Madm2 = chi2");
            syst.set_last_eq_int_reflection_sector(+1);
        }
    }

    // ADM momentum constraints
    if (eccentricity_stage) {
        solver.space.add_eq_int_inf(syst, "integ(intPx) = 0");
        // Px closes yaxis through the x-directed shift term.
        syst.set_last_eq_int_reflection_sector(-1);
        solver.space.add_eq_int_inf(syst, "integ(intPy) = 0");
        // Py closes xaxis through the y-directed shift term.
        syst.set_last_eq_int_reflection_sector(+1);
        // nosym: pin the z-velocity closure with integ(intPz) = 0. No-op for sym.
        pz_constraint(solver.space, syst);
    }

    // Mass integrals
    solver.space.add_eq_int_volume(syst, solver.space.NS1, solver.space.ADAPTED1, "integvolume(intMb) = Mb1");
    syst.set_last_eq_int_reflection_sector(+1);
    solver.space.add_eq_int_volume(syst, solver.space.NS2, solver.space.ADAPTED2, "integvolume(intMb) = Mb2");
    syst.set_last_eq_int_reflection_sector(+1);
    solver.space.add_eq_int_volume(syst, solver.space.NS1, solver.space.ADAPTED1, "integvolume(intM) = qlMadm1");
    syst.set_last_eq_int_reflection_sector(+1);
    solver.space.add_eq_int_volume(syst, solver.space.NS2, solver.space.ADAPTED2, "integvolume(intM) = qlMadm2");
    syst.set_last_eq_int_reflection_sector(+1);

    // ---------------------------------------------------------------------------
    // Newton iteration
    // ---------------------------------------------------------------------------
    if (rank == 0) {
        if (env_flag("BNS_PRINT_STRUCTURE", false))
            syst.print_system_structure(std::cout);
        solver.print_diagnostics(syst, 0, 0);
    }

    const auto coord_binding = bind_coordinate_fields(syst, solver.coord_vectors);
    const std::string final_stage = solver.bconfig.control(COROT_BIN)
                                        ? stage_text + "_COROT"
                                        : stage_text;
    const SolverRuntimeConfig stage_solver_config =
        solver.with_stage_mumps_tree_cache(
            SolverRuntimeConfig::from_environment()
                .with_stage_default_backend(NewtonBackend::JfnkMumps),
            final_stage);

    solver.run_newton_loop(syst, stage_solver_config, [&](int ite, double& conv) {
        update_fields(solver.cfields, solver.coord_vectors, {}, solver.xo, solver.xc1, solver.xc2, coord_binding);
        for (int d = 0; d < solver.ndom; ++d)
            solver.logh.set_domain(d) = syst.give_val_def_scalar_domain("H", d);

        std::stringstream ss;
        ss << "total_" << ite - 1;
        solver.bconfig.set_filename(ss.str());

        if (rank == 0) {
            solver.print_diagnostics(syst, ite, conv);
            if (solver.bconfig.control(CHECKPOINT))
                bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift,
                                        solver.logh, solver.phi);
        }
    });

    // ---------------------------------------------------------------------------
    // Finalize
    // ---------------------------------------------------------------------------
    if (solver.bconfig.control(COROT_BIN)) {
        solver.bconfig.set(MADM, BCO1) = solver.bconfig(QLMADM, BCO1);
        solver.bconfig.set(MADM, BCO2) = solver.bconfig(QLMADM, BCO2);
    }
    solver.bconfig.set_filename(solver.converged_filename(final_stage));

    if (rank == 0)
        bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift,
                                solver.logh, solver.phi);

    return exit_status;
}

} // namespace Kadath
