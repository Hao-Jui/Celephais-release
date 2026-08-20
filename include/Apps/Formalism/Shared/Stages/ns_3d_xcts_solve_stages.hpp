#pragma once

#include "Apps/Helper/solver_base.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/System_of_eqs/solver_runtime_config.hpp"
#include "For_Kadath/Space/spheric_homothetic.hpp"

#include <sstream>
#include <string>
#include <type_traits>

namespace Kadath {

template <bool WithScalar, typename Solver>
void ns_3d_xcts_save_stage_fields(Solver& solver)
{
    if constexpr (WithScalar) {
        if constexpr (requires { solver.scalar_slot(); }) {
            Scalar scalar_slot = solver.scalar_slot();
            solver.bconfig.set_field(SWEIGHT) = solver.has_scalar_weight_slot();
            if (solver.has_scalar_weight_slot()) {
                Scalar Sweight = solver.scalar_weight_slot();
                bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift,
                                        solver.logh, scalar_slot, Sweight);
            } else {
                bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift,
                                        solver.logh, scalar_slot);
            }
        } else {
            bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift,
                                    solver.logh, solver.scalar_field_slot());
        }
    } else {
        bco_utils::save_to_file(solver.space, solver.bconfig, solver.conf, solver.lapse, solver.shift,
                                solver.logh);
    }
}

template <typename space_t>
void ns_3d_xcts_add_norot_surface_condition(space_t& space, System_of_eqs& syst, bool fixed)
{
    if (fixed) {
        syst.add_eq_bc(1, OUTER_BC, "lev = 0");
    } else if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<space_t>>,
                                         Space_spheric_homothetic>) {
        Index pos_cf(space.get_domain(1)->get_nbr_coefs());
        pos_cf.set(0) = 0;
        syst.add_eq_mode(1, OUTER_BC, "H", pos_cf, 0.);
    } else {
        syst.add_eq_bc(1, OUTER_BC, "H = 0");
    }
}

template <bool WithScalar, typename Solver, typename ScalarSetup, typename FieldEqs, typename ScalarGlobalEqs>
int ns_3d_xcts_run_norot_stage(Solver& solver, bool fixed, const char* banner_head,
                               const SolverRuntimeConfig& solver_config, ScalarSetup&& emit_scalar_setup,
                               FieldEqs&& emit_field_equations, ScalarGlobalEqs&& emit_scalar_global_eqs)
{
    int exit_status = EXIT_SUCCESS;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double loghc = std::log(solver.bconfig(HC));
    std::string stagename = (fixed) ? "NOROT_FIXED" : "NOROT";
    const SolverRuntimeConfig stage_solver_config =
        solver.with_stage_mumps_tree_cache(solver_config, stagename);

    const auto resume = solver.load_existing_solution(stagename);
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Current: " << resume.before << "\nSolved previously: "
                      << solver.bconfig.config_filename_abs() << std::endl;
        return legacy_status_from_dataset_resume(resume);
    }

    if (fixed) {
        if (rank == 0)
            std::cout << "############################" << std::endl
                      << banner_head << " with a fixed radius" << std::endl
                      << "############################" << std::endl;
    } else {
        if (rank == 0) {
            std::cout << "############################" << std::endl
                      << banner_head << " with a resolved surface" << std::endl;
            if (solver.bconfig.control(MB_FIXING))
                std::cout << "with Baryonic Mass fixing\n";
            else
                std::cout << "with ADM Mass fixing\n";
            std::cout << "############################" << std::endl;
        }
    }

    // set up radius and level field in case of "fixed"
    scalar_ary_t coord_scalars;
    coord_scalars[to_int(coord_scalar::R_BCO1)] = Scalar(solver.space);

    update_fields_co(solver.cfields, solver.coord_vectors, coord_scalars, 0.);

    // a level function, defining a root at a given fixed radius
    // helper construction to force the system to attain a fixed radius
    // instead resolving the correct surface
    Scalar level(solver.space);
    level = (*coord_scalars[to_int(coord_scalar::R_BCO1)]) * (*coord_scalars[to_int(coord_scalar::R_BCO1)]) -
            solver.bconfig(RMID) * solver.bconfig(RMID);
    level.std_base();

    // setup a system of equations
    System_of_eqs syst(solver.space, 0, solver.ndom - 1);
    syst.add_var("H", solver.logh);
    solver.syst_init(syst);
    syst.add_def(solver.ndom - 1, "intJ = 0 * P");

    solver.add_mass_fixing(syst);

    // in case of a fixed radius solve the TOV with the given fixed central enthalpy
    // in case of a resolved surface, solve for the central enthalpy
    if (fixed)
        syst.add_cst("Hc", loghc);
    else
        syst.add_var("Hc", loghc);

    // in case of "fixed" domain radii
    syst.add_cst("lev", level);

    emit_scalar_setup(syst);

    for (int d = 0; d < solver.ndom; d++) {
        switch (d) {
            // in the star the constraint equations are sourced by the matter
            case 0:
            case 1:
                // sources
                syst.add_def(d, "Etilde = press * h - press * delta");
                syst.add_def(d, "Stilde = 3 * press * delta");

                // constraint equations
                emit_field_equations(syst, d, /*has_matter=*/true, /*has_shift=*/false);

                // definition for the baryonic mass integral
                syst.add_def(d, "intMb = P^6 * rho");
                // first integral of the euler equation for a static, non-rotating star, i.e. a TOV
                syst.add_def(d, "firstint = H + log(N)");

                break;
            // outside the matter is absent and the sources are zero
            default:
                syst.add_eq_full(d, "H = 0");

                emit_field_equations(syst, d, /*has_matter=*/false, /*has_shift=*/false);
                break;
        }
    }

    // add the constraint equations and demand continuity their normal derivative across domain boundaries
    solver.space.add_eq(syst, "eqNP= 0", "N", "dn(N)");
    solver.space.add_eq(syst, "eqP = 0", "P", "dn(P)");

    // formalism-specific global scalar equation(s); no-op for GR
    emit_scalar_global_eqs(syst);

    // boundary conditions at infinity
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "N=1");
    syst.add_eq_bc(solver.ndom - 1, OUTER_BC, "P=1");

    // Fixed-radius NOROT uses the level field. Resolved NOROT closes either all
    // adapted surface-shape modes or, for the homothetic space, only the
    // spherical radius mode.
    ns_3d_xcts_add_norot_surface_condition(solver.space, syst, fixed);

    // first integral in the innermost domains with non-zero matter content
    // and condition on the central value, either fixed directly or by the
    // integral below
    syst.add_eq_first_integral(0, 1, "firstint", "H - Hc");

    // if surface is resolved, fix the central enthalpy by one of these integrals
    if (!fixed) {
        solver.space.add_eq_int_volume(syst, 2, "integvolume(intMb) = Mb");
        solver.space.add_eq_int_inf(syst, "integ(intMadm) = Madm");
    }

    // print the variation of the surface radius over the whole star
    if (rank == 0) {
        auto rs = bco_utils::get_rmin_rmax(solver.space, 1);
        std::cout << "[Rmin, Rmax] : [" << rs[0] << ", " << rs[1] << "]\n";
    }

    // solve until convergence is achieved
    solver.run_newton_loop(syst, stage_solver_config, [&](int ite, double& conv) {
        solver.update_config_quantities(loghc);
        // output files at this iteration and print diagnostics
        std::stringstream ss;
        ss << "norot_3d_";
        if (fixed)
            ss << "fixed";
        else
            ss << "norot_bc";
        ss << "_" << ite - 1;
        solver.bconfig.set(QLMADM) = solver.bconfig(MADM);
        solver.bconfig.set_filename(ss.str());
        if (rank == 0) {
            solver.print_diagnostics(syst, ite, conv);
            if (solver.bconfig.control(CHECKPOINT))
                ns_3d_xcts_save_stage_fields<WithScalar>(solver);
        }

        // update all coordinate fields, in case the domain extents have changed
        update_fields_co(solver.cfields, solver.coord_vectors, coord_scalars, 0.);
    });

    if constexpr (requires { solver.update_scalar_branch_cstB_config(); })
        solver.update_scalar_branch_cstB_config();
    solver.bconfig.set_filename(solver.converged_filename(stagename));
    if (rank == 0)
        ns_3d_xcts_save_stage_fields<WithScalar>(solver);
    return exit_status;
}

template <bool WithScalar, typename Solver, typename ScalarSetup, typename FieldEqs, typename ScalarGlobalEqs>
int ns_3d_xcts_run_uniform_rot_stage(Solver& solver, const char* banner_head, const char* stilde_def,
                                     bool print_initial_residual_norm, const SolverRuntimeConfig& solver_config,
                                     ScalarSetup&& emit_scalar_setup, FieldEqs&& emit_field_equations,
                                     ScalarGlobalEqs&& emit_scalar_global_eqs)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const SolverRuntimeConfig stage_solver_config =
        solver.with_stage_mumps_tree_cache(solver_config, "UNIROT");

    const auto resume = solver.load_existing_solution("UNIROT");
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Solved previously: " << solver.bconfig.config_filename_abs() << std::endl;
        return legacy_status_from_dataset_resume(resume);
    }

    double loghc = std::log(solver.bconfig(HC));
    double xo = 0.0;
    update_fields_co(solver.cfields, solver.coord_vectors, {}, xo);

    if (rank == 0 && solver.bconfig.control(MB_FIXING))
        std::cout << "###################################" << std::endl
                  << banner_head << " - with fixed Baryonic Mass" << std::endl
                  << "###################################" << std::endl;
    else if (rank == 0)
        std::cout << "###################################" << std::endl
                  << banner_head << " - with fixed ADM Mass" << std::endl
                  << "###################################" << std::endl;

    System_of_eqs syst(solver.space, 0, solver.ndom - 1);
    syst.add_var("H", solver.logh);
    solver.syst_init(syst);

    syst.add_cst("chi", solver.bconfig(CHI));
    syst.add_var("ome", solver.bconfig(OMEGA));
    syst.add_var("Hc", loghc);

    solver.add_mass_fixing(syst);

    syst.add_var("bet", solver.shift);

    syst.add_def("omega^i = bet^i + ome * mg^i");

    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / "
                 "2. / Ntilde");

    syst.add_def(solver.ndom - 1, "intJ = multr(A_ij * mg^j * einf^i) / 2. / 4piG");

    syst.add_def(2, "intS = A_ij * mg^i * sm^j / 2. / 4piG");

    emit_scalar_setup(syst);

    for (int d = 0; d < solver.ndom; d++) {
        switch (d) {
            case 0:
            case 1:
                syst.add_def(d, "U^i = omega^i / N");
                syst.add_def(d, "Usquare = P^4 * U_i * U^i");
                syst.add_def(d, "Wsquare = 1. / (1. - Usquare)");
                syst.add_def(d, "W = sqrt(Wsquare)");

                syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
                syst.add_def(d, stilde_def);
                syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");

                emit_field_equations(syst, d, /*has_matter=*/true, /*has_shift=*/true);

                syst.add_def(d, "intMb = P^6 * rho(h) * W");
                syst.add_def(d, "firstint = H + log(N) - log(W)");

                break;
            default:
                syst.add_eq_full(d, "H = 0");

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

    syst.add_eq_first_integral(0, 1, "firstint", "H - Hc");
    solver.space.add_eq_int_volume(syst, 2, "integvolume(intMb) = Mb");

    solver.space.add_eq_int_inf(syst, "integ(intJ) - chi * Madm * Madm = 0");
    solver.space.add_eq_int_inf(syst, "integ(intMadm) = Madm");

    if (rank == 0) {
        if (print_initial_residual_norm) {
            // report the true iteration-0 residual infinity norm instead of 0
            Array<double> initial_residual(syst.sec_member());
            double initial_error = 0.0;
            for (int i = 0; i < initial_residual.get_size(0); ++i) {
                const double residual_abs = std::abs(initial_residual(i));
                if (residual_abs > initial_error)
                    initial_error = residual_abs;
            }
            solver.print_diagnostics(syst, 0, initial_error);
        } else {
            solver.print_diagnostics(syst, 0, 0);
        }
    }
    solver.run_newton_loop(syst, stage_solver_config, [&](int ite, double& conv) {
        solver.update_config_quantities(loghc);
        std::stringstream ss;
        ss << "rot_3d_total" << ite - 1;
        solver.bconfig.set(QLMADM) = solver.bconfig(MADM);
        solver.bconfig.set_filename(ss.str());
        if (rank == 0) {
            solver.print_diagnostics(syst, ite, conv);
            if (solver.bconfig.control(CHECKPOINT))
                ns_3d_xcts_save_stage_fields<WithScalar>(solver);
        }
        update_fields_co(solver.cfields, solver.coord_vectors, {}, xo, &syst);
    });

    if constexpr (requires { solver.update_scalar_branch_cstB_config(); })
        solver.update_scalar_branch_cstB_config();
    solver.bconfig.set_filename(solver.converged_filename("UNIROT"));
    if (rank == 0)
        ns_3d_xcts_save_stage_fields<WithScalar>(solver);
    return EXIT_SUCCESS;
}

} // namespace Kadath
