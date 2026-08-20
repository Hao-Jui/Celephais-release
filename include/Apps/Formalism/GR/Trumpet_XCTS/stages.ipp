#pragma once

#include "mpi.h"

namespace Kadath {

template <typename config_t, typename space_t>
int trumpet_3d_xcts_solver<config_t, space_t>::trumpet_stage()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto resume = this->load_existing_solution("TRUMPET");
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Solved previously: " << bconfig.config_filename_abs() << std::endl;
        return legacy_status_from_dataset_resume(resume);
    }

    if (rank == 0)
        std::cout << "############################" << std::endl
                  << "Trumpet BH with resolved horizon" << std::endl
                  << "############################" << std::endl;

    Scalar trumpet_conf(space);
    Scalar trumpet_lapse(space);
    Scalar trumpet_lapse_conf(space);
    Scalar horizon_level(space);
    Vector trumpet_shift(space, CON, basis);
    fill_trumpet_background(space, bconfig, trumpet_conf, trumpet_lapse, trumpet_lapse_conf, trumpet_shift);
    fill_paper_horizon_level(space, bconfig, horizon_level);

    constexpr int first_exterior_domain = 2;
    copy_trumpet_interior_background(space, trumpet_conf, trumpet_lapse, conf, lapse, first_exterior_domain);
    copy_trumpet_shift(trumpet_shift, shift);

    System_of_eqs syst(space, first_exterior_domain, ndom - 1);
    syst_init(syst, trumpet_conf, trumpet_lapse, trumpet_lapse_conf, trumpet_shift);

    for (int d = first_exterior_domain; d < ndom; ++d)
        gr_xcts::add_xcts_field_equations(syst, d, /*has_matter=*/false, /*has_shift=*/true);

    space.add_eq(syst, "eqNP = 0", "N", "dn(N)");
    space.add_eq(syst, "eqP = 0", "P", "dn(P)");

    space.add_bc_inf(syst, "P = 1");
    space.add_bc_inf(syst, "N = 1");

    space.add_bc_horizon(syst, "P = P0");
    space.add_bc_horizon(syst, "N = N0");
    syst.add_cst("lev", horizon_level);
    space.add_bc_horizon(syst, "lev = 0");

    if (rank == 0) {
        syst.print_system_structure(std::cout);
        print_diagnostics(syst, 0, 0.);
    }

    const SolverRuntimeConfig solver_config = this->with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), "TRUMPET");
    run_newton_loop(syst, solver_config, [&](int ite, double& conv) {
        update_fields_co(cfields, coord_vectors, {}, 0.);
        refresh_trumpet_coordinate_constants(syst, coord_vectors, first_exterior_domain, ndom);
        fill_trumpet_background(space, bconfig, trumpet_conf, trumpet_lapse, trumpet_lapse_conf, trumpet_shift);
        fill_paper_horizon_level(space, bconfig, horizon_level);
        refresh_trumpet_scalar_constant(syst, horizon_level, "lev", first_exterior_domain, ndom);
        copy_trumpet_interior_background(space, trumpet_conf, trumpet_lapse, conf, lapse, first_exterior_domain);
        refresh_trumpet_background_constants(syst, trumpet_conf, trumpet_lapse, trumpet_lapse_conf, trumpet_shift,
                                             first_exterior_domain);
        copy_trumpet_shift(trumpet_shift, shift);

        std::stringstream ss;
        ss << "trumpet_total_bc_" << ite - 1;
        bconfig.set_filename(ss.str());

        if (rank == 0) {
            print_diagnostics(syst, ite, conv);
            if (bconfig.control(CHECKPOINT))
                bco_utils::save_to_file(space, bconfig, conf, lapse, shift);
        }
    });

    copy_trumpet_shift(trumpet_shift, shift);
    copy_trumpet_interior_background(space, trumpet_conf, trumpet_lapse, conf, lapse, first_exterior_domain);
    bconfig.set(RMID) = bco_utils::get_radius(space.get_domain(1), OUTER_BC);

    Val_domain integMsq(syst.give_val_def("intMsq")()(2));
    bconfig.set(MIRR) = std::sqrt(space.get_domain(2)->integ(integMsq, INNER_BC));
    Val_domain integMadm(syst.give_val_def("intMadm")()(ndom - 1));
    bconfig.set(MADM) = space.get_domain(ndom - 1)->integ(integMadm, OUTER_BC);

    bconfig.set_filename(converged_filename("TRUMPET"));
    if (rank == 0)
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift);

    return EXIT_SUCCESS;
}

} // namespace Kadath
