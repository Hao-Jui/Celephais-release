#pragma once

#include "Apps/AMR/refinement_policy_state.hpp"
#include "Apps/Helper/solver_base.hpp"
#include "Apps/Startup/app_startup.hpp"
#include "Apps/Workflow/resume_ladder.hpp"
#include "Apps/Workflow/solver_sequence.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Config/config_binary.hpp"

#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <mpi.h>
#include <string>
#include <type_traits>
#include <utility>

namespace KadathApps
{
    namespace bns_workflow_detail
    {
        template <typename Policy>
        concept AmrRefineHook = requires {
            { Policy::amr_refine(
                std::declval<typename Policy::config_t&>(),
                std::declval<const std::string&>(),
                std::declval<int>(),
                std::declval<int>()) } -> std::convertible_to<bool>;
        };

        template <typename Policy>
        concept StatefulAmrRefineHook = requires {
            { Policy::amr_refine(
                std::declval<typename Policy::config_t&>(),
                std::declval<const std::string&>(),
                std::declval<int>(),
                std::declval<int>(),
                std::declval<bns_amr::RefinementPolicyState&>()) } -> std::convertible_to<bool>;
        };

        inline bool mpi_is_initialized()
        {
            int initialized = 0;
            MPI_Initialized(&initialized);
            return initialized != 0;
        }

        inline void mpi_barrier_if_initialized()
        {
            if (mpi_is_initialized())
                MPI_Barrier(MPI_COMM_WORLD);
        }

        template <typename Config>
        void reload_root_written_config_if_needed(Config& bconfig, int rank)
        {
            if (rank != 0 && mpi_is_initialized())
                bconfig.open_config();
        }

        template <typename Policy, typename... Args>
        int invoke_policy_solve(typename Policy::config_t& bconfig,
                                const std::string& outputdir,
                                Args&&... args)
        {
            if constexpr (std::is_void_v<decltype(Policy::solve(
                              bconfig, outputdir, std::forward<Args>(args)...))>) {
                Policy::solve(bconfig, outputdir, std::forward<Args>(args)...);
                return EXIT_SUCCESS;
            } else {
                return Policy::solve(bconfig, outputdir, std::forward<Args>(args)...);
            }
        }

        template <typename Policy, typename SolveAtCurrent>
        void run_bns_amr_gate(typename Policy::config_t& bconfig,
                              int rank,
                              const std::string& outputdir,
                              SolveAtCurrent&& solve_at_current)
        {
            if (!bconfig.template amr_setting_as<bool>(AMR_ENABLED))
                return;

            if constexpr (StatefulAmrRefineHook<Policy> || AmrRefineHook<Policy>) {
                const int max_cycles = std::max(
                    0, bconfig.template amr_setting_as<int>(AMR_MAX_CYCLES));
                bns_amr::RefinementPolicyState amr_policy_state;
                for (int cycle = 0; cycle < max_cycles; ++cycle) {
                    bool refined = false;
                    if constexpr (StatefulAmrRefineHook<Policy>)
                        refined = Policy::amr_refine(
                            bconfig, outputdir, rank, cycle, amr_policy_state);
                    else
                        refined = Policy::amr_refine(bconfig, outputdir, rank, cycle);
                    if (!refined)
                        return;
                    solve_at_current();
                }
            } else {
                KADATH_THROW(
                    "adaptive_mesh_refinement.enabled=true requires BNS Policy::amr_refine("
                    "config, outputdir, rank, cycle[, state])");
            }
        }
    } // namespace bns_workflow_detail

    template <typename Policy>
    void run_bns_solver_sequence(typename Policy::config_t& bconfig,
                                 int rank,
                                 bool setup_first,
                                 const std::string& outputdir)
    {
        const int requested_init = !bconfig.has_seq_setting(INIT_RES)
            ? 7
            : bconfig.template seq_setting_as<int>(INIT_RES);
        // Warm starts (setup_first == false) resume from an existing .dat: floor
        // the ladder init at the loaded grid's resolution so the first uniform
        // regrid never downsamples (and discards) the saved AMR refinement.
        const Kadath::ResolutionLadder ladder =
            KadathApps::make_resume_ladder<typename Policy::space_t>(
                bconfig, requested_init, bconfig(BIN_RES), /*warm_start=*/!setup_first);
        const int init_res = ladder.init;
        const int final_res = ladder.final;
        const bool res_inc = init_res < final_res;
        const auto final_stages = bconfig.return_stages();
        auto& stages = bconfig.return_stages();

        if (res_inc) {
            stages.fill(false);
            stages[to_int(Policy::low_res_stage())] = true;
        }

        if (setup_first) {
            bconfig.set_filename("initbin");
            bconfig.set(BIN_RES) = init_res;
            Policy::setup_bin_config(bconfig);
            Policy::setup_space(bconfig);
            if (Policy::stop_after_setup()) {
                return;
            }
        }

        auto regrid = [&]() -> bool {
            std::string fname{Policy::regrid_filename()};

            if (rank == 0) {
                Policy::regrid(bconfig, fname);
            }
            bconfig.set_filename(fname);
            bns_workflow_detail::mpi_barrier_if_initialized();
            bns_workflow_detail::reload_root_written_config_if_needed(bconfig, rank);

            stages = final_stages;
            return Policy::stop_after_regrid();
        };

        if (bconfig.control(REGRID) && !setup_first) {
            if (regrid()) {
                return;
            }
        }

        // Some binary policies keep a 3-arg solve signature for compatibility
        // with older fixed-Omega warm-up plumbing. The unified stage driver uses
        // explicit [stages] gates for QUASI_EQUIL / FORCE_BALANCE / ECC_RED.
        bool first_solve = true;
        auto solve_once = [&]() -> int {
            if constexpr (requires { Policy::solve(bconfig, outputdir, true); }) {
                const bool want_warmup = first_solve && bconfig.control(FIXED_GOMEGA);
                return bns_workflow_detail::invoke_policy_solve<Policy>(
                    bconfig, outputdir, want_warmup);
            } else {
                return bns_workflow_detail::invoke_policy_solve<Policy>(
                    bconfig, outputdir);
            }
        };
        auto solve_current_grid = [&]() {
            int status = EXIT_SUCCESS;
            do {
                status = solve_once();
                first_solve = false;
            } while (Kadath::status_requires_field_reload(status));

            if (!Kadath::status_completed_without_reload(status))
                KADATH_THROW("BNS solve failed");
        };
        auto run_solve = [&]() {
            solve_current_grid();
        };

        bool stopped_after_regrid = false;
        auto regrid_to_res = [&](int new_res) -> bool {
            bconfig.set(BIN_RES) = new_res;
            stopped_after_regrid = regrid();
            return stopped_after_regrid;
        };

        const bool amr_uses_loaded_final_grid =
            bconfig.template amr_setting_as<bool>(AMR_ENABLED) &&
            !setup_first && !res_inc;

        if (!amr_uses_loaded_final_grid)
            run_resolution_sequence(ladder, regrid_to_res, run_solve);
        if (!stopped_after_regrid) {
            bns_workflow_detail::run_bns_amr_gate<Policy>(
                bconfig, rank, outputdir, solve_current_grid);
        }
    }

    template <typename Policy>
    int run_bns_app(int argc, char** argv)
    {
        const int rank = init_mpi(argc, argv);
        Policy::configure_defaults();

        const int rc = guarded_run([&] {
            using config_t = typename Policy::config_t;
            if (argc < 2) {
                Policy::write_example(rank);
                return;
            }

            auto sr = KadathApps::parse_kadath_config_toml_startup<config_t>(argc, argv, rank);
            config_t bconfig = std::move(sr.bconfig);

            run_bns_solver_sequence<Policy>(bconfig, rank, sr.setup_first, sr.outputdir);
        });

        MPI_Finalize();
        return rc;
    }
} // namespace KadathApps
