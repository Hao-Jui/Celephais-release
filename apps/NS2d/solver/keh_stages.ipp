#include "mpi.h"

namespace Kadath {

template <class eos_t, typename config_t, typename space_t>
int ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::keh_stage()
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
    const double cstA = bconfig.template diffrot<double>(CSTA);
    const int adapted_outer = space.ADAPTED_OUTER;

    auto npts = space.get_domain(adapted_outer)->get_nbr_points();
    Index pos_origin(space.get_domain(0)->get_nbr_points());
    Index pos_eq(npts);
    pos_eq.set(0) = npts(0) - 1;
    pos_eq.set(1) = npts(1) - 1;
    double R0 = space.get_domain(adapted_outer)->get_radius()(pos_eq);

    if (rank == 0) {
        std::cout << "###################################" << std::endl
                  << "Differential Rotating models       " << std::endl
                  << "Rotation law: " << law << std::endl
                  << "Fixed A / R0: " << cstA << std::endl
                  << "Initial R0: " << R0 << std::endl
                  << "###################################" << "\n\n";
    }
    System_of_eqs syst(space, 0, ndom - 1);
    syst_init(syst);

    syst.add_var("Hc", loghc);
    double diffA = cstA * R0;
    syst.add_cst("cstA", cstA);
    syst.add_var("R0", R0);
    syst.add_var("diffA", diffA);
    syst.add_def("diffAField = ones * diffA");
    syst.add_def("r = multr(ones)");
    syst.add_def("intF = - 0.5 * F^2 / diffA^2");
    syst.add_def("rotlaw = Omg - ome + F / diffA^2");

    add_common_defs(syst);
    add_domain_equations(syst, true);
    add_omega_constraints(syst);

    syst.add_eq_val(0, "diffAField/R0 - cstA", pos_origin);
    syst.add_eq_val(adapted_outer, "r/R0 - 1", pos_eq);

    add_boundary_conditions(syst);

    if (rank == 0)
        print_diagnostics(syst, 0, 0);
    MPI_Barrier(MPI_COMM_WORLD);

    run_newton_loop(syst, loghc, rank);
    return EXIT_SUCCESS;
}


} // namespace Kadath