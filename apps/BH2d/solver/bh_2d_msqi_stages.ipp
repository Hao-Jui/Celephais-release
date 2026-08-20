#pragma once
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "mpi.h"
#include "Apps/Helper/newton_loop_runner.hpp"

namespace Kadath {
template <typename config_t, typename space_t>
int bh_2d_msqi_solver<config_t, space_t>::run_stage(bool fixed_lapse, const char* resume_tag, const char* banner,
                                                    const char* fname_tag, bool update_fixed_lapse)
{
    int exit_status = EXIT_SUCCESS;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto resume = this->load_existing_solution(resume_tag);
    if (resume.found()) {
        if (resume.requires_field_reload())
            return legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
        if (rank == 0) {
            std::cout << "Solved previously: " << resume.before << std::endl;
        }
        return EXIT_SUCCESS;
    }

    if (rank == 0) {
        std::cout << "############################" << std::endl
                  << banner << std::endl
                  << "############################" << std::endl;
    }
    bconfig.set(MIRR) = bco_utils::mirr_from_mch(bconfig(CHI), bconfig(MCH));

    System_of_eqs syst(space, 0, ndom - 1);
    syst_init(syst);
    add_common_defs(syst, fixed_lapse);
    add_domain_equations(syst, fixed_lapse);
    add_boundary_conditions(syst, fixed_lapse);
    if (rank == 0) {
        // syst.dump_eq_dependency_coloring(std::cout); // FOR DEVELOPER
    }

    const auto zero_excluded_domains = [&]() {
        for (int i : excluded_doms) {
            lapse.set_domain(i).annule_hard();
            metA.set_domain(i).annule_hard();
            metB.set_domain(i).annule_hard();
            beta.set_domain(i).annule_hard();
        }
    };

    bool endloop = false;
    int ite = 0;
    double conv = 0;
    [[maybe_unused]] double conv_old;

    if (rank == 0) {
        // syst.print_system_structure(std::cout);
        print_diagnostics(syst, ite, conv);
    }

    const SolverRuntimeConfig solver_config = this->with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), resume_tag);

    while (!endloop) {

        endloop = newton_step_with_consensus(syst, bconfig.template seq_setting_as<double>(PREC), conv, solver_config, ite == 1);
        ite++;
        std::stringstream ss;
        ss << fname_tag << ite - 1;
        bconfig.set_filename(ss.str());
        if (rank == 0) {
            print_diagnostics(syst, ite, conv);
        }
        if (bconfig.control(CHECKPOINT)) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                zero_excluded_domains();
                bco_utils::save_to_file(space, bconfig, lapse, metA, metB, beta, varscal);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        check_max_iter_exceeded(rank, ite, conv);
    }

    bconfig.set(RMID) = bco_utils::get_radius(space.get_domain(1), OUTER_BC);
    if (update_fixed_lapse) {
        bconfig.set(FIXED_LAPSE) = bco_utils::get_boundary_val(2, lapse, INNER_BC);
    }
    bconfig.set_filename(converged_filename(resume_tag));
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        zero_excluded_domains();
        bco_utils::save_to_file(space, bconfig, lapse, metA, metB, beta, varscal);
        cout << "Data written!" << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return exit_status;
}

template <typename config_t, typename space_t> int bh_2d_msqi_solver<config_t, space_t>::von_Neumann_stage()
{
    return run_stage(false, "VON_NEUMANN", "Total system with von Neumann BC", "bh_total_bc_", true);
}

template <typename config_t, typename space_t> int bh_2d_msqi_solver<config_t, space_t>::fixed_lapse_stage()
{
    return run_stage(true, "DIRICHLET_LAPSE", "Total system with Fixed Lapse BC", "bh_total_", false);
}


} // namespace Kadath
