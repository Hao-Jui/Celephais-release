#pragma once

#include "Apps/Helper/solver_base.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Formalism/Shared/ns_binary_boost_kinematics.hpp"

#include <cmath>
#include <sstream>

namespace Kadath {

template <bool WithScalar, typename Solver, typename AdvectionDefs, typename ScalarSetup,
          typename FieldEqs, typename ScalarGlobalEqs>
int ns_3d_xcts_run_binary_boost_stage(Solver& solver, kadath_config<BIN_INFO>& binconfig, NODES bco,
                                      const char* banner, AdvectionDefs&& emit_advection_defs,
                                      ScalarSetup&& emit_scalar_setup, FieldEqs&& emit_field_equations,
                                      ScalarGlobalEqs&& emit_scalar_global_eqs)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // generate filename string unique to this binary setup
    std::stringstream stage_ss;
    const auto component_center = ns_binary_boost::component_center(binconfig(DIST), bco);
    // Legacy binary inputs can omit com_velz; use the configured physical
    // default, as the binary no-symmetry momentum closure does.
    if (std::isnan(binconfig.set(COM_VELZ)))
        binconfig.set(COM_VELZ) = 0.;
    stage_ss << "BIN_BOOST_BCO" << (bco == BCO1 ? 1 : 2)
             << "_" << binconfig(DIST) << "_" << binconfig(GOMEGA)
             << "_" << binconfig(COM) << "_" << binconfig(COM_VELZ);
    auto const boost_converged_filename{stage_ss.str()};
    const auto resume = solver.load_existing_solution(boost_converged_filename);
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Solved previously: " << solver.bconfig.config_filename_abs() << std::endl;
        return EXIT_SUCCESS;
    }

    // Since we assume a comoving frame in the COM of the Binary
    // we need to using the same system of equation as in the Binary
    // this includes splitting the fluid velocity into an irrotational
    // term plus a rotation term
    Scalar phi(solver.space);
    phi.annule_hard();
    phi.std_base();

    // flag needs to be set in order to be used during import into
    // a BNS or BHNS setup
    solver.bconfig.set_field(PHI) = true;

    double H_scale = 0;
    Scalar logh_const(solver.logh);
    logh_const.std_base();

    double xo = 0.0;

    if (rank == 0)
        std::cout << "############################" << std::endl
                  << banner << std::endl
                  << "############################" << std::endl;

    update_fields_co(solver.cfields, solver.coord_vectors, {}, xo);
    System_of_eqs syst(solver.space, 0, solver.ndom - 1);

    // irrotational part of the fluid velocity
    syst.add_var("phi", phi);
    syst.add_var("qlMadm", solver.bconfig(QLMADM));
    syst.add_cst("Hconst", logh_const);
    syst.add_var("Hscale", H_scale);
    syst.add_cst("ex", *solver.coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *solver.coord_vectors[to_int(coord_vector::EY)]);
    syst.add_cst("ez", *solver.coord_vectors[to_int(coord_vector::EZ)]);
    for (int d = 0; d < solver.ndom; d++)
        // the enthalpy is equal to the constant part everywhere
        // outside of the stars, i.e. zero
        if (d > 1)
            syst.add_def(d, "H  = Hconst");
        else
            syst.add_def(d, "H  = Hconst * (1. + Hscale)");

    solver.syst_init(syst);

    syst.add_cst("chi", solver.bconfig(CHI));
    syst.add_var("ome", solver.bconfig(OMEGA));

    // MB and MADM are now fixed as they would be in a binary
    syst.add_cst("Mb", solver.bconfig(MB));
    syst.add_cst("Madm", solver.bconfig(MADM));

    syst.add_var("bet", solver.shift);

    // The isolated grid uses local coordinates r about the star. Restore the
    // constant part of the binary's global helical generator evaluated at the
    // component centre C_a:
    //   B_a(r) = bet + Omega zhat x r
    //                  + (-Omega C_a,y, Omega(C_a,x + xaxis), zvel).
    const double omega_boost = binconfig(GOMEGA);
    const auto boost = ns_binary_boost::local_translation(
        component_center, omega_boost, binconfig(COM), binconfig(COM_VELZ));
    syst.add_cst("omega_boost", omega_boost);
    syst.add_cst("boostx", boost.x);
    syst.add_cst("boosty", boost.y);
    syst.add_cst("boostz", boost.z);
    syst.add_def("B^i = bet^i + omega_boost * mg^i"
                 " + boostx * ex^i + boosty * ey^i + boostz * ez^i");

    emit_advection_defs(syst);

    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / "
                 "2. / Ntilde");

    syst.add_def(solver.ndom - 1, "intJ = multr(A_ij * mg^j * einf^i) / 2. / 4piG");
    syst.add_def("intS = A_ij * mg^i * sm^j / 2 / 4piG");

    // define the irrotational + spinning parts of the fluid velocity
    for (int d = 0; d <= 1; ++d) {
        syst.add_def(d, "s^i  = ome * mg^i");
        syst.add_def(d, "eta_i  = D_i phi + P^4 * s_i");
    }

    emit_scalar_setup(syst);

    for (int d = 0; d < solver.ndom; d++) {
        switch (d) {
            case 0:
            case 1:
                // definitions for the fluid 3-velocity
                // and its Lorentz factor
                syst.add_def(d, "Wsquare= eta^i * eta_i / h^2 / P^4 + 1.");
                syst.add_def(d, "W      = sqrt(Wsquare)");
                syst.add_def(d, "U^i    = eta^i / P^4 / h / W");
                syst.add_def(d, "Usquare= P^4 * U_i * U^i");
                syst.add_def(d, "V^i    = N * U^i - B^i");
                syst.add_def(d, "firstint = log(h * N / W + D_i phi * V^i)");
                syst.add_def(d, "eqphi  = P^6 * W * V^i * D_i H + dHdlnrho * D_i (P^6 * W * V^i)");

                // rescaled sources and constraint equations.
                // The Stilde leading coefficient is 3 (the spatial-stress trace
                // term 3P, since press * delta maps to P via the Etilde
                // definition); the eqP/eqNP/eqbet block is supplied by
                // emit_field_equations.
                syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
                syst.add_def(d, "Stilde = 3 * press * delta + press * h * Wsquare * Usquare");
                syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");

                emit_field_equations(syst, d, /*has_matter=*/true, /*has_shift=*/true);

                // integrant of the baryonic mass integral
                syst.add_def(d, "intMb = P^6 * rho(h) * W");

                break;
            default:
                // outside the star the matter is absent and the sources are zero
                // syst.add_eq_full(d, "H = 0");
                syst.add_eq_full(d, "phi = 0");

                emit_field_equations(syst, d, /*has_matter=*/false, /*has_shift=*/true);
                break;
        }
    }

    solver.space.add_eq(syst, "eqNP= 0", "N", "dn(N)");
    solver.space.add_eq(syst, "eqP= 0", "P", "dn(P)");
    solver.space.add_eq(syst, "eqbet^i= 0", "bet^i", "dn(bet^i)");

    // formalism-specific global scalar equation(s); no-op for GR
    emit_scalar_global_eqs(syst);

    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "N=1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "P=1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "bet^i=0");

    syst.add_eq_bc(1, OUTER_BC, "H = 0");
    syst.add_eq_bc(1, OUTER_BC, "V^i * D_i H = 0");
    // in case of the stellar domains
    syst.add_eq_vel_pot(0, 2, "eqphi = 0", "phi=0", "phi", true);
    syst.add_eq_matching(0, OUTER_BC, "phi");
    syst.add_eq_matching(0, OUTER_BC, "dn(phi)");
    syst.add_eq_vel_pot(1, 2, "eqphi = 0", "phi=0", "phi", true);

    solver.space.add_eq_int(syst, 2, OUTER_BC, "integ(intS) - chi * Madm * Madm = 0");
    syst.set_last_eq_int_reflection_sector(+1);
    solver.space.add_eq_int_inf(syst, "integ(intMadmalt) = qlMadm");
    syst.set_last_eq_int_reflection_sector(+1);

    solver.space.add_eq_int_volume(syst, 2, "integvolume(intMb) = Mb");
    syst.set_last_eq_int_reflection_sector(+1);

    if (rank == 0)
        solver.print_diagnostics(syst, 0, 0);
    SolverRuntimeConfig solver_config = SolverRuntimeConfig::from_environment();
    solver_config = solver.with_stage_mumps_tree_cache(
        solver_config, boost_converged_filename);
    if (solver_config.backend == NewtonBackend::JfnkMumps) {
        // The boosted single-star setup is small but sensitive to inexact
        // early Krylov corrections. Use the same sparse Jacobian with a direct
        // MUMPS Newton step so the binary handoff is robust.
        solver_config.backend = NewtonBackend::Mumps;
    }
    solver.run_newton_loop(syst, solver_config, [&](int ite, double& conv) {
        std::stringstream ss;
        ss << "ns_bin_boost_" << ite - 1;
        solver.bconfig.set_filename(ss.str());

        // overwrite logh with scaled version for later use and output
        solver.logh = syst.give_val_def("H");
        if constexpr (requires { solver.update_scalar_branch_cstB_config(); })
            solver.update_scalar_branch_cstB_config();

        if (rank == 0) {
            solver.print_diagnostics(syst, ite, conv);
            if (solver.bconfig.control(CHECKPOINT)) {
                if constexpr (WithScalar) {
                    if constexpr (requires { solver.scalar_slot(); }) {
                        Scalar scalar_slot = solver.scalar_slot();
                        solver.bconfig.set_field(SWEIGHT) = solver.has_scalar_weight_slot();
                        if (solver.has_scalar_weight_slot()) {
                            Scalar Sweight = solver.scalar_weight_slot();
                            bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse,
                                                    solver.shift, solver.logh, scalar_slot, Sweight);
                        } else {
                            bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse,
                                                    solver.shift, solver.logh, scalar_slot);
                        }
                    } else {
                        bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse,
                                                solver.shift, solver.logh, solver.scalar_field_slot());
                    }
                } else {
                    bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse,
                                            solver.shift, solver.logh);
                }
            }
        }
        update_fields_co(solver.cfields, solver.coord_vectors, {}, xo, &syst);
    });

    if constexpr (requires { solver.update_scalar_branch_cstB_config(); })
        solver.update_scalar_branch_cstB_config();
    solver.bconfig.set_filename(solver.converged_filename(boost_converged_filename));
    if (rank == 0) {
        if constexpr (WithScalar) {
            if constexpr (requires { solver.scalar_slot(); }) {
                Scalar scalar_slot = solver.scalar_slot();
                solver.bconfig.set_field(SWEIGHT) = solver.has_scalar_weight_slot();
                if (solver.has_scalar_weight_slot()) {
                    Scalar Sweight = solver.scalar_weight_slot();
                    bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse,
                                            solver.shift, solver.logh, phi, scalar_slot, Sweight);
                } else {
                    bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse,
                                            solver.shift, solver.logh, phi, scalar_slot);
                }
            } else {
                bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift,
                                        solver.logh, phi, solver.scalar_field_slot());
            }
        } else {
            bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift,
                                    solver.logh, phi);
        }
    }
    return EXIT_SUCCESS;
}

} // namespace Kadath
