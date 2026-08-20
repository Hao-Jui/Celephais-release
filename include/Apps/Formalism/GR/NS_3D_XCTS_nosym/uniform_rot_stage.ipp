// Body fragment #included via stage_helper.cpp (which is included
// by solver.hpp). Not a standalone translation unit.
#include "mpi.h"
#include <cmath>
#include "Apps/Bco_utils/bco_io.hpp"

template <class eos_t, typename config_t, typename space_t>
int ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::uniform_rot_stage()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto resume = this->load_existing_solution("UNIROT");
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Solved previously: " << bconfig.config_filename_abs() << std::endl;
        return legacy_status_from_dataset_resume(resume);
    }

    double loghc = std::log(bconfig(HC));
    double xo = 0.0;
    update_fields_co(cfields, coord_vectors, {}, xo);

    if (rank == 0 && bconfig.control(MB_FIXING))
        std::cout << "###################################" << std::endl
                  << "Rotating - with fixed Baryonic Mass" << std::endl
                  << "###################################" << std::endl;
    else if (rank == 0)
        std::cout << "###################################" << std::endl
                  << "Rotating - with fixed ADM Mass" << std::endl
                  << "###################################" << std::endl;

    System_of_eqs syst(space, 0, ndom - 1);
    syst.add_var("H", logh);
    syst_init(syst);

    add_spin_constants(syst);
    syst.add_var("Hc", loghc);

    add_mass_fixing(syst);

    syst.add_var("bet", shift);

    syst.add_def("mm^i = cos(angs) * mmz^i + sin(angs) * mmx^i");
    syst.add_def("omega^i = bet^i + ome * mm^i");

    add_extrinsic_curvature(syst);

    syst.add_def(2, "intS = A_ij * mm^i * sm^j / 2. / 4piG");

    for (int d = 0; d < ndom; d++) {
        switch (d) {
            case 0:
            case 1:
                syst.add_def(d, "U^i = omega^i / N");
                syst.add_def(d, "Usquare = P^4 * U_i * U^i");
                syst.add_def(d, "Wsquare = 1. / (1. - Usquare)");
                syst.add_def(d, "W = sqrt(Wsquare)");

                syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
                syst.add_def(d, "Stilde = 3 * press * delta + (Etilde + press * delta) * Usquare");
                syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");

                add_constraint_equations(syst, d);

                syst.add_def(d, "intMb = P^6 * rho(h) * W");
                syst.add_def(d, "firstint = H + log(N) - log(W)");

                break;
            default:
                syst.add_eq_full(d, "H = 0");
                add_constraint_equations(syst, d);
                break;
        }
    }

    add_conformal_field_bcs(syst);

    syst.add_eq_bc(1, OUTER_BC, "H = 0");

    syst.add_eq_first_integral(0, 1, "firstint", "H - Hc", true);
    space.add_eq_int_volume(syst, 2, "integvolume(intMb) = Mb");
    syst.set_last_eq_int_reflection_sector(+1);

    space.add_eq_int_inf(syst, "integ(intJ) - chi * Madm * Madm = 0");
    syst.set_last_eq_int_reflection_sector(+1);
    space.add_eq_int_inf(syst, "integ(intMadm) = Madm");
    syst.set_last_eq_int_reflection_sector(+1);

    auto coord_binding = bind_coordinate_fields(syst, coord_vectors);

    if (rank == 0)
        print_diagnostics(syst, 0, 0);

    const SolverRuntimeConfig solver_config = this->with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), "UNIROT");
    run_newton_loop(syst, solver_config, [&](int ite, double& conv) {
        update_config_quantities(loghc);
        std::stringstream ss;
        ss << "rot_3d_total" << ite - 1;
        bconfig.set(QLMADM) = bconfig(MADM);
        bconfig.set_filename(ss.str());
        update_fields_co(cfields, coord_vectors, {}, xo, coord_binding);
        conv = ns_nosym_forwarded_residual_infinity_norm(syst);
        if (rank == 0) {
            print_diagnostics(syst, ite, conv);
            if (bconfig.control(CHECKPOINT))
                bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh);
        }
    });

    save_converged_solution("UNIROT");
    return EXIT_SUCCESS;
}
