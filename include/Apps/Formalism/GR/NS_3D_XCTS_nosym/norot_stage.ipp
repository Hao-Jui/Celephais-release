// Body fragment #included via stage_helper.cpp (which is included
// by solver.hpp). Not a standalone translation unit.
#include "mpi.h"
#include <cmath>
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"

template <class eos_t, typename config_t, typename space_t>
int ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::norot_stage(bool fixed)
{
    int exit_status = EXIT_SUCCESS;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double loghc = std::log(bconfig(HC));
    std::string stagename = (fixed) ? "NOROT_FIXED" : "NOROT";

    const auto resume = this->load_existing_solution(stagename);
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Current: " << resume.before << "\nSolved previously: " << bconfig.config_filename_abs()
                      << std::endl;
        return legacy_status_from_dataset_resume(resume);
    }

    if (fixed) {
        if (rank == 0)
            std::cout << "############################" << std::endl
                      << "TOV with a fixed radius" << std::endl
                      << "############################" << std::endl;
    } else {
        if (rank == 0) {
            std::cout << "############################" << std::endl
                      << "TOV with a homothetic surface" << std::endl;
            if (bconfig.control(MB_FIXING))
                std::cout << "with Baryonic Mass fixing\n";
            else
                std::cout << "with ADM Mass fixing\n";

            std::cout << "############################" << std::endl;
        }
    }

    // set up radius and leve field in case of "fixed"
    scalar_ary_t coord_scalars;
    coord_scalars[to_int(coord_scalar::R_BCO1)] = Scalar(space);

    update_fields_co(cfields, coord_vectors, coord_scalars, 0.);

    // a level function, defining a root at a given fixed radius
    // helper construction to force the system to attain a fixed radius
    // instead resolving the correct surface
    Scalar level(space);
    level = (*coord_scalars[to_int(coord_scalar::R_BCO1)]) * (*coord_scalars[to_int(coord_scalar::R_BCO1)]) -
            bconfig(RMID) * bconfig(RMID);
    level.std_base();

    // setup a system of equations
    System_of_eqs syst(space, 0, ndom - 1);
    syst.add_var("H", logh);
    syst_init(syst);
    syst.add_def(ndom - 1, "intJ = 0 * P");

    add_mass_fixing(syst);

    // in case of a fixed radius solve the TOV with the given fixed central enthalpy
    // in case of a resolved surface, solve for the central enthalpy
    if (fixed) {
        syst.add_cst("Hc", loghc);
    } else {
        syst.add_var("Hc", loghc);
    }

    // in case of "fixed" domain radii
    syst.add_cst("lev", level);

    for (int d = 0; d < ndom; d++) {
        switch (d) {
            // in the star the constraint equations are sourced by the matter
            case 0:
            case 1:
                // sources
                syst.add_def(d, "Etilde = press * h - press * delta");
                syst.add_def(d, "Stilde = 3 * press * delta");

                // constraint equations
                syst.add_def(d, "eqP    = delta * D^i D_i P + 4piG / 2. * P^5 * Etilde");
                syst.add_def(d, "eqNP   = delta * D^i D_i NP - 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");

                // definition for the baryonic mass integral
                syst.add_def(d, "intMb = P^6 * rho");
                // first integral of the euler equation for a static, non-rotating star, i.e. a TOV
                syst.add_def(d, "firstint = H + log(N)");

                break;
            // outside the matter is absent and the sources are zero
            default:
                syst.add_eq_full(d, "H = 0");

                syst.add_def(d, "eqP = D^i D_i P");
                syst.add_def(d, "eqNP = D^i D_i NP");
                break;
        }
    }

    // add the constraint equations and demand continuity their normal derivative across domain boundaries
    space.add_eq(syst, "eqNP= 0", "N", "dn(N)");
    space.add_eq(syst, "eqP = 0", "P", "dn(P)");

    // boundary conditions at infinity
    syst.add_eq_bc(ndom - 1, OUTER_BC, "N=1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "P=1");

    // if the radius of the stellar surface domain is fixed
    // use the helper construction, i.e. a level function with a root defining the radius
    if (fixed) {
        syst.add_eq_bc(1, OUTER_BC, "lev = 0");
    }
    // if the surface is resolved, define it to be where the matter vanishes
    else {
        syst.add_eq_bc(1, OUTER_BC, "H = 0");
    }

    // first integral in the innermost domains with non-zero matter content
    // and condition on the central value, either fixed directly or by the
    // integral below
    syst.add_eq_first_integral(0, 1, "firstint", "H - Hc", true);

    // if surface is resolved, fix the central enthalpy by one of these integrals
    if (!fixed) {
        space.add_eq_int_volume(syst, 2, "integvolume(intMb) = Mb");
        syst.set_last_eq_int_reflection_sector(+1);
        space.add_eq_int_inf(syst, "integ(intMadm) = Madm");
        syst.set_last_eq_int_reflection_sector(+1);
    }

    auto coord_binding = bind_coordinate_fields(syst, coord_vectors, coord_scalars);

    // print the variation of the surface radius over the whole star
    if (rank == 0) {
        auto rs = bco_utils::get_rmin_rmax(space, 1);
        std::cout << "[Rmin, Rmax] : [" << rs[0] << ", " << rs[1] << "]\n";
    }

    // solve until convergence is achieved
    const SolverRuntimeConfig solver_config = this->with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), stagename);
    run_newton_loop(syst, solver_config, [&](int ite, double& conv) {
        update_config_quantities(loghc);
        // Refresh coordinate-dependent constants after adapted domains move,
        // then test/print the exact residual already forwarded by that refresh.
        update_fields_co(cfields, coord_vectors, coord_scalars, 0., coord_binding);
        conv = ns_nosym_forwarded_residual_infinity_norm(syst);
        // output files at this iteration and print diagnostics
        std::stringstream ss;
        ss << "norot_3d_" << (fixed ? "fixed" : "norot_bc") << "_" << ite - 1;
        bconfig.set(QLMADM) = bconfig(MADM);
        bconfig.set_filename(ss.str());
        if (rank == 0) {
            print_diagnostics(syst, ite, conv);
            if (bconfig.control(CHECKPOINT))
                bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh);
        }
    });

    save_converged_solution(stagename);
    return exit_status;
}
