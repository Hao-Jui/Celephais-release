// Body fragment #included via stage_helper.cpp (which is included
// by solver.hpp). Not a standalone translation unit.
#include "mpi.h"
#include <cmath>
#include "Apps/Bco_utils/bco_io.hpp"

template <class eos_t, typename config_t, typename space_t>
int ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::binary_boost_stage(kadath_config<BIN_INFO>& binconfig, NODES bco)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // generate filename string unique to this binary setup
    std::stringstream stage_ss;
    stage_ss << "BIN_BOOST" << "_" << binconfig(DIST) << "_" << binconfig(GOMEGA);
    auto const boost_converged_filename{stage_ss.str()};
    const auto resume = this->load_existing_solution(boost_converged_filename);
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Solved previously: " << bconfig.config_filename_abs() << std::endl;
        return EXIT_SUCCESS;
    }

    // Since we assume a comoving frame in the COM of the Binary
    // we need to using the same system of equation as in the Binary
    // this includes splitting the fluid velocity into an irrotational
    // term plus a rotation term
    Scalar phi(space);
    phi.annule_hard();
    phi.std_base();

    // flag needs to be set in order to be used during import into
    // a BNS or BHNS setup
    bconfig.set_field(PHI) = true;

    double H_scale = 0;
    Scalar logh_const(logh);
    logh_const.std_base();

    double xo = 0.0;

    if (rank == 0)
        std::cout << "############################" << std::endl
                  << "Binary boosted NS" << std::endl
                  << "############################" << std::endl;

    activate_coordinate_vector(coord_vectors, space, coord_vector::EX);
    activate_coordinate_vector(coord_vectors, space, coord_vector::EY);
    update_fields_co(cfields, coord_vectors, {}, xo);
    System_of_eqs syst(space, 0, ndom - 1);

    // irrotational part of the fluid velocity
    syst.add_var("phi", phi);
    syst.add_var("qlMadm", bconfig(QLMADM));
    syst.add_cst("Hconst", logh_const);
    syst.add_var("Hscale", H_scale);
    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);
    for (int d = 0; d < ndom; d++)
        // the enthalpy is equal to the constant part everywhere
        // outside of the stars, i.e. zero
        if (d > 1)
            syst.add_def(d, "H  = Hconst");
        else
            syst.add_def(d, "H  = Hconst * (1. + Hscale)");

    syst_init(syst);

    add_spin_constants(syst);

    // MB and MADM are now fixed as they would be in a binary
    syst.add_cst("Mb", bconfig(MB));
    syst.add_cst("Madm", bconfig(MADM));

    syst.add_var("bet", shift);

    // boost based on binary config
    syst.add_cst("omega_boost", binconfig(GOMEGA));
    syst.add_def("B^i = bet^i + omega_boost * (mg^i)");
    syst.add_def("mm^i = cos(angs) * mmz^i + sin(angs) * mmx^i");

    add_extrinsic_curvature(syst);
    syst.add_def("intS = A_ij * mm^i * sm^j / 2 / 4piG");

    // define the irrotational + spinning parts of the fluid velocity
    for (int d = 0; d <= 1; ++d) {
        syst.add_def(d, "s^i  = ome * mm^i");
        syst.add_def(d, "eta_i  = D_i phi + P^4 * s_i");
    }

    for (int d = 0; d < ndom; d++) {
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

                // rescaled sources and constraint equations
                syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
                syst.add_def(d, "Stilde = 3 * press * delta + (Etilde + press * delta) * Usquare");
                syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");

                add_constraint_equations(syst, d);

                // integrant of the baryonic mass integral
                syst.add_def(d, "intMb = P^6 * rho(h) * W");

                break;
            default:
                // outside the star the matter is absent and the sources are zero
                syst.add_eq_full(d, "phi = 0");
                add_constraint_equations(syst, d);
                break;
        }
    }

    add_conformal_field_bcs(syst);

    syst.add_eq_bc(1, OUTER_BC, "H = 0");
    syst.add_eq_bc(1, OUTER_BC, "V^i * D_i H = 0");
    // in case of the stellar domains
    syst.add_eq_vel_pot(0, 2, "eqphi = 0", "phi=0", "phi", true);
    syst.add_eq_matching(0, OUTER_BC, "phi");
    syst.add_eq_matching(0, OUTER_BC, "dn(phi)");
    syst.add_eq_vel_pot(1, 2, "eqphi = 0", "phi=0", "phi", true);

    space.add_eq_int(syst, 2, OUTER_BC, "integ(intS) - chi * Madm * Madm = 0");
    syst.set_last_eq_int_reflection_sector(+1);
    space.add_eq_int_inf(syst, "integ(intMadmalt) = qlMadm");
    syst.set_last_eq_int_reflection_sector(+1);

    space.add_eq_int_volume(syst, 2, "integvolume(intMb) = Mb");
    syst.set_last_eq_int_reflection_sector(+1);

    auto coord_binding = bind_coordinate_fields(syst, coord_vectors);

    if (rank == 0)
        print_diagnostics(syst, 0, 0);

    SolverRuntimeConfig solver_config = SolverRuntimeConfig::from_environment();
    solver_config = this->with_stage_mumps_tree_cache(
        solver_config, boost_converged_filename);
    if (solver_config.backend == NewtonBackend::JfnkMumps) {
        // The boosted single-star setup is small but sensitive to inexact
        // early Krylov corrections. Use the same sparse Jacobian with a direct
        // MUMPS Newton step so the binary handoff is robust.
        solver_config.backend = NewtonBackend::Mumps;
    }
    run_newton_loop(syst, solver_config, [&](int ite, double& conv) {
        std::stringstream ss;
        ss << "ns_bin_boost_" << ite - 1;
        bconfig.set_filename(ss.str());

        update_fields_co(cfields, coord_vectors, {}, xo, coord_binding);
        conv = ns_nosym_forwarded_residual_infinity_norm(syst);

        // Overwrite logh with the scaled version for later use and output.
        // H is defined independently on every domain; copy each domain view
        // immediately so neither a full-space Tensor nor a second H evaluation
        // is materialized.
        for (int d = 0; d < ndom; ++d)
            logh.set_domain(d) = syst.give_val_def_scalar_domain("H", d);

        if (rank == 0) {
            print_diagnostics(syst, ite, conv);
            if (bconfig.control(CHECKPOINT))
                bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh);
        }
    });

    save_converged_solution(boost_converged_filename, phi);
    return EXIT_SUCCESS;
}
