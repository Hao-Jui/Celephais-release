#pragma once

#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_utils_toml.hpp"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "Apps/Policy/app_resolution.hpp"
#include "Apps/Formalism/Shared/ns_solver_instantiate.hpp"
#include "Apps/Workflow/ns_resolution_sequence.hpp"

#include <mpi.h>
#include <array>
#include <cstdlib>
#include <string>

namespace KadathApps::NsPolicyCommon
{

// Construct the formalism's NS solver from a checkpoint on `PrimarySpace` and run
// `action` on it. The solver is not movable, so it is built in place at the point
// `action` runs; the constructor arity (with / without the trailing scalar field)
// is selected by Traits::has_scalar.
template <typename Traits, typename PrimarySpace, typename config_t, typename Action>
int ns_from_file_as(config_t& bconfig, const std::string& spacein, Action&& action)
{
    return Kadath::ns_xcts::instantiate_from_file_as<PrimarySpace>(
        bconfig, spacein,
        [&](auto& fields) { return Traits::load_extra(bconfig, fields); },
        [&](auto tag, auto& f, Kadath::Base_tensor& basis, auto& extra) {
            using eos_t = typename decltype(tag)::eos_t;
            using Solver = typename Traits::template solver_t<eos_t, config_t, PrimarySpace>;
            if constexpr (Traits::has_scalar) {
                Solver ns_solver(bconfig, f.space, basis, f.conf, f.lapse, f.logh, f.shift, extra);
                return action(ns_solver);
            } else {
                Solver ns_solver(bconfig, f.space, basis, f.conf, f.lapse, f.logh, f.shift);
                return action(ns_solver);
            }
        });
}

// Solve the loaded NS. GR loads the NOROT stage onto Traits::norot_space and every
// other stage onto Traits::rotating_space; scalar formalisms set both to the
// adapted space (so the branch collapses to a single grid).
template <typename Traits, typename config_t>
int ns_solve_from_file(config_t& bconfig, const std::string& spacein)
{
    using namespace Kadath;
    auto solve = [](auto& ns_solver) { return ns_solver.solve(); };
    if (bconfig.return_stages()[to_int(NOROT)])
        return ns_from_file_as<Traits, typename Traits::norot_space>(bconfig, spacein, solve);
    return ns_from_file_as<Traits, typename Traits::rotating_space>(bconfig, spacein, solve);
}

// Binary-boost the loaded NS onto the rotating (adapted) space.
template <typename Traits, typename config_t>
int ns_boost_from_file(config_t& bconfig, const std::string& spacein,
                       kadath_config<BIN_INFO>& binconfig, NODES bco)
{
    return ns_from_file_as<Traits, typename Traits::rotating_space>(
        bconfig, spacein,
        [&](auto& ns_solver) { return ns_solver.binary_boost_stage(binconfig, bco); });
}

// Isolated resolution ladder on the formalism's primary space, driven by its
// solve-from-file and regrid.
template <typename Traits, typename config_t>
config_t ns_isolated_driver(config_t& bconfig, std::string outputdir)
{
    auto seed_initial_data = [](config_t& bc, int current_res, int final_res) {
        if constexpr (requires { Traits::template seed_initial_data<config_t>(bc, current_res, final_res); }) {
            Traits::template seed_initial_data<config_t>(bc, current_res, final_res);
        } else if constexpr (requires { Traits::template seed_initial_data<config_t>(bc, current_res); }) {
            Traits::template seed_initial_data<config_t>(bc, current_res);
        } else {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
                Kadath::Seed::setup_co<NS, typename Traits::ladder_space>(bc);
        }
    };
    auto resume_ladder = [](config_t& bc, int requested_init, int final, bool warm_start) {
        if (bc.return_stages()[to_int(NOROT)])
            return KadathApps::make_resume_ladder<typename Traits::norot_space>(
                bc, requested_init, final, warm_start);
        return KadathApps::make_resume_ladder<typename Traits::rotating_space>(
            bc, requested_init, final, warm_start);
    };
    auto assert_space_resolution = [](config_t& bc, const std::string& spacein, int expected_res,
                                      const std::string& label) {
        if (bc.return_stages()[to_int(NOROT)]) {
            Kadath::ns_3d_xcts_assert_space_resolution<typename Traits::norot_space>(
                spacein, expected_res, label);
            return;
        }
        Kadath::ns_3d_xcts_assert_space_resolution<typename Traits::rotating_space>(
            spacein, expected_res, label);
    };

    return Kadath::ns_3d_xcts_isolated_driver_impl<typename Traits::ladder_space>(
        bconfig, outputdir,
        ns_solve_from_file<Traits, config_t>,
        [](config_t& bc, int new_res, std::string filename) { return Traits::regrid(bc, new_res, filename); },
        seed_initial_data, resume_ladder, assert_space_resolution);
}

template <typename Traits, typename config_t>
int ns_boosted_driver(config_t& nsconfig, config_t& iterative_config, std::string outputdir,
                      kadath_config<BIN_INFO> binconfig, NODES bco)
{
    return Kadath::ns_3d_xcts_boosted_driver_impl(
        nsconfig, iterative_config, outputdir, binconfig, bco, ns_boost_from_file<Traits, config_t>);
}

// Control computation of a 3D NS from a setup config/.dat file pair: run the
// isolated resolution ladder, then (when BIN_BOOST is the last enabled stage) the
// binary-boost stage. `binconfig`/`bco` are only consulted by the boost.
template <typename Traits, typename config_t>
int solve_ns_xcts(config_t& bconfig, std::string outputdir,
                  kadath_config<BIN_INFO> binconfig = kadath_config<BIN_INFO>{}, NODES bco = BCO1)
{
    using namespace Kadath;

    if (!bconfig.has_seq_setting(INIT_RES))
        bconfig.seq_setting(INIT_RES) = kDefaultInitialResolution;

    int exit_status = EXIT_SUCCESS;
    std::array<bool, NUM_STAGES_V> stage_enabled = bconfig.return_stages();
    auto [last_stage, last_stage_idx] = get_last_enabled(MSTAGE(), stage_enabled);

    auto iterative_config = ns_isolated_driver<Traits>(bconfig, outputdir);

    if (last_stage_idx == to_int(BIN_BOOST))
        exit_status = ns_boosted_driver<Traits>(bconfig, iterative_config, outputdir, binconfig, bco);

    return exit_status;
}

} // namespace KadathApps::NsPolicyCommon
