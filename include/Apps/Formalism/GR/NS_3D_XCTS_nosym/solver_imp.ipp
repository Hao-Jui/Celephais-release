#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include <sstream>
#include "mpi.h"
#include "regrid.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include <cmath>
#include <type_traits>
#include <utility>
#include "Apps/Formalism/Shared/ns_driver_common.hpp"
#include "Apps/Workflow/ns_solve_driver.hpp"
#include "Apps/Workflow/ns_resolution_sequence.hpp"
#include "Apps/Formalism/Shared/xcts_syst_init.ipp"
#include "Apps/Formalism/Shared/ns_2d_seed.hpp"
#include "Apps/Formalism/GR/NS_2D_XCTS/solver.hpp"

using namespace Kadath::Margherita;

namespace Kadath {


template <class eos_t, typename config_t, typename space_t>
ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::ns_3d_xcts_nosym_solver(config_t& config_in, space_t& space_in,
                                                               Base_tensor& base_in, Scalar& conf_in, Scalar& lapse_in,
                                                               Scalar& logh_in, Vector& shift_in)
    : XCTS_Solver<config_t, space_t>(config_in, space_in, base_in), conf(conf_in), lapse(lapse_in), logh(logh_in),
      shift(shift_in), fmet(Metric_flat(space_in, base_in))
{
    // The isolated stages need only the rotational and radial coordinate
    // fields. The boost stage activates its two Cartesian directions on demand.
    activate_coordinate_vector(coord_vectors, space, coord_vector::GLOBAL_ROT);
    activate_coordinate_vector(coord_vectors, space, coord_vector::BCO1_ROTx);
    activate_coordinate_vector(coord_vectors, space, coord_vector::BCO1_ROTz);
    activate_coordinate_vector(coord_vectors, space, coord_vector::S_BCO1);
    activate_coordinate_vector(coord_vectors, space, coord_vector::S_INF);

    update_fields_co(cfields, coord_vectors, {}, 0.);
}

// standardized filename for each converged dataset at the end of each stage.
template <class eos_t, typename config_t, typename space_t>
std::string ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::converged_filename(const std::string& stage) const
{
    return converged_gr_ns_nosym_filename(bconfig, space, stage);
}

template <class eos_t, typename config_t, typename space_t> int ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::solve()
{
    return ns_3d_xcts_solve_common(*this, bconfig);
}

template <typename config_t, typename Run>
int with_loaded_ns_solver_from_file(config_t& bconfig, const std::string& spacein, Run&& run)
{
    BeFileSource ff1(spacein);
    Space_spheric_adapted_nosym space(ff1);
    Scalar conf(space, ff1);
    Scalar lapse(space, ff1);
    Vector shift(space, ff1);
    Scalar logh(space, ff1);
    Base_tensor basis(space, CARTESIAN_BASIS);

    // load and setup the EOS
    const double h_cut = bconfig.template eos<double>(HCUT);
    const std::string eos_file = bconfig.template eos<std::string>(EOSFILE);
    const std::string eos_type = bconfig.template eos<std::string>(EOSTYPE);

    if (eos_type == "Cold_PWPoly") {
        using eos_t = Kadath::Margherita::Cold_PWPoly;

        EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
        ns_3d_xcts_nosym_solver<eos_t, decltype(bconfig), decltype(space)> ns_solver(bconfig, space, basis, conf, lapse,
                                                                                     logh, shift);
        return run(ns_solver);
    } else if (eos_type == "Cold_Table") {
        using eos_t = Kadath::Margherita::Cold_Table;

        const int interp_pts =
            (bconfig.template eos<int>(INTERP_PTS) == 0) ? 2000 : bconfig.template eos<int>(INTERP_PTS);

        EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut, interp_pts,
                                             bconfig.template eos<double>(MNUC_CGS));
        ns_3d_xcts_nosym_solver<eos_t, decltype(bconfig), decltype(space)> ns_solver(bconfig, space, basis, conf, lapse,
                                                                                     logh, shift);
        return run(ns_solver);
    } else {
        KADATH_THROW("Unknown EOSTYPE.");
    }
}

// Load the saved space + fields, initialise the EOS, construct the solver, and
// invoke `run` on it. The boosted driver keeps its own print/load wrapper; the
// isolated driver uses the shared orchestration helper with a solve callback.
template <typename config_t, typename Run>
int with_loaded_ns_solver(config_t& bconfig, const std::string& outputdir, Run&& run)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const auto spacein = bconfig.space_filename();
    if (rank == 0) {
        std::cout << "Config File: " << bconfig.config_filename_abs() << std::endl
                  << "Fields File: " << spacein << std::endl
                  << bconfig << std::endl;
    }

    if (outputdir != "")
        bconfig.set_outputdir(outputdir);
    return with_loaded_ns_solver_from_file(bconfig, spacein, std::forward<Run>(run));
}

template <typename config_t>
int gr_ns_nosym_solve_from_file(config_t& bconfig, const std::string& spacein)
{
    return with_loaded_ns_solver_from_file(bconfig, spacein, [](auto& ns_solver) { return ns_solver.solve(); });
}

template <typename config_t> config_t ns_3d_xcts_isolated_driver(config_t& bconfig, std::string outputdir)
{
    auto seed_from_rotated_2d_xcts = [](config_t& bc, int current_res, int final_res) {
        ns_2d_xcts_seed_initial_data<Space_spheric_adapted_nosym>(
            bc, current_res, final_res,
            [](auto& config, const std::string& seed_outputdir, const std::string& basename) {
                return ns_2d_xcts_driver(config, seed_outputdir, basename);
            },
            "NS_nosym", "initns_2d_xcts", bc(DEG));
    };

    return ns_3d_xcts_isolated_driver_impl<Space_spheric_adapted_nosym>(
        bconfig, outputdir,
        gr_ns_nosym_solve_from_file<config_t>,
        [](config_t& bc, int new_res, std::string filename) {
            return ns_3d_xcts_nosym_interpolate_on_new_grid(bc, new_res, filename);
        },
        seed_from_rotated_2d_xcts);
}

template <typename config_t>
inline int ns_3d_xcts_boosted_driver(config_t& nsconfig, config_t& iterative_config, std::string outputdir,
                                     kadath_config<BIN_INFO> binconfig, NODES bco)
{
    int exit_status = legacy_status_from_stage_outcome(KadathApps::request_binary_boost());
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto bconfig = nsconfig;

    while (status_requests_binary_boost(exit_status)) {
        exit_status = with_loaded_ns_solver(
            bconfig, outputdir,
            [&](auto& ns_solver) { return ns_solver.binary_boost_stage(binconfig, bco); });
        MPI_Barrier(MPI_COMM_WORLD);
    } // end obtaining the rotating solution
    nsconfig = bconfig;
    return exit_status;
}

template <typename config_t>
inline int ns_3d_xcts_nosym_driver(config_t& bconfig, std::string outputdir, kadath_config<BIN_INFO> binconfig,
                             NODES bco)
{
    if (!bconfig.has_seq_setting(INIT_RES))
        bconfig.seq_setting(INIT_RES) = Kadath::kDefaultInitialResolution;

    int exit_status = EXIT_SUCCESS;
    std::array<bool, NUM_STAGES_V> stage_enabled = bconfig.return_stages();
    auto [last_stage, last_stage_idx] = get_last_enabled(MSTAGE(), stage_enabled);

    auto iterative_config = ns_3d_xcts_isolated_driver(bconfig, outputdir);

    if (last_stage_idx == to_int(BIN_BOOST))
        exit_status = ns_3d_xcts_boosted_driver(bconfig, iterative_config, outputdir, binconfig, bco);

    return exit_status;
}

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::syst_init(System_of_eqs& syst)
{
    using namespace Kadath::Margherita;


    auto& space = syst.get_space();
    const int ndom = space.get_nbr_domains();
    // call the (flat) conformal metric "f"
    xcts::add_flat_metric(fmet, syst);

    // define numerical constants
    xcts::add_four_pi_g(syst);

    // include the coordinate fields (mmx/mmz interleaved between mg and sm — a
    // nosym-specific spelling, kept per-class; the shared mg/sm/einf trio helper
    // would reorder it and is therefore not wired here)
    syst.add_cst("mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
    syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
    syst.add_cst("sm", *coord_vectors[to_int(coord_vector::S_BCO1)]);
    syst.add_cst("einf", *coord_vectors[to_int(coord_vector::S_INF)]);

    // the basic fields, conformal factor, lapse and (log) enthalpy
    xcts::add_conformal_lapse_vars(syst, conf, lapse);

    // common conformal combinations + ADM-mass / Komar integrands at infinity
    xcts::add_conformal_combinations_and_adm_integrands(syst, ndom);

    // define the EOS operators
    Param p;
    xcts::add_eos_operators<eos_t>(syst, p);

    // enthalpy + thermodynamic defs (h, rho, eps, press, dHdlnrho, delta)
    xcts::add_thermodynamic_defs(syst);
}

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::print_diagnostics(System_of_eqs const& syst, const int ite,
                                                                    const double conv) const
{
    ns_3d_xcts_print_diagnostics(space, bconfig, syst, ite, conv);
    std::cout << "=======================================" << "\n\n";
} // end print diagnostics

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_nosym_solver<eos_t, config_t, space_t>::update_config_quantities(const double& loghc)
{
    xcts::update_config_quantities<eos_t>(bconfig, loghc);
}

} // namespace Kadath
