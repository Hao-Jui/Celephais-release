#include "mpi.h"

namespace Kadath {

template <class eos_t, typename config_t, typename space_t>
int ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::uniform_stage()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int maybe_skip = maybe_skip_stage(rank, "UNIROT");
    if (maybe_skip != -1) {
        return maybe_skip;
    }

    double loghc = 0.0;
    init_stage_fields(loghc);

    const std::string law = bconfig.template diffrot<std::string>(DIFF_LAW);

    if (rank == 0) {
        std::cout << "###################################" << std::endl
                  << "Uniformly Rotating models       " << std::endl
                  << "Rotation law: " << law << std::endl
                  << "###################################" << "\n\n";
    }
    System_of_eqs syst(space, 0, ndom - 1);
    syst_init(syst);

    syst.add_var("Hc", loghc);
    syst.add_def("rotlaw = Omg - ome");

    add_common_defs(syst);
    add_domain_equations(syst, false);
    add_omega_constraints(syst);
    add_boundary_conditions(syst);

    if (rank == 0)
        print_diagnostics(syst, 0, 0);
    MPI_Barrier(MPI_COMM_WORLD);

    run_newton_loop(syst, loghc, rank);
    return EXIT_SUCCESS;
}


} // namespace Kadath