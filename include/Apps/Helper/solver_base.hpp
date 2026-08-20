#pragma once

#include "mpi.h"
#include "For_Kadath/Kadath_point_h/kadath.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "celephais_paths.h"
#include "For_Kadath/System_of_eqs/solver_runtime_config.hpp"
#include "Apps/Helper/converged_filename.hpp"
#include "Apps/Helper/newton_loop_runner.hpp"
#include "Apps/Workflow/dataset_resume.hpp"

#include <cstdlib>
#include <string>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <array>
#include <sstream>
#include <system_error>
#include <type_traits>

namespace fs = std::filesystem;

namespace Kadath
{
    inline std::string get_cos_path()
    {
        // Prefer runtime env var override; fall back to compile-time CELEPHAIS_ROOT_DIR
        const char* home_env = std::getenv("HOME_CELEPHAIS");
        const std::string root = home_env ? home_env : CELEPHAIS_ROOT_DIR;
        return root + "/data/convergence/";
    }

    inline fs::path normalized_directory_for_compare(const std::string& dir)
    {
        std::error_code ec;
        const fs::path input = dir.empty() ? fs::path(".") : fs::path(dir);
        fs::path normalized = fs::weakly_canonical(input, ec);
        if (!ec)
            return normalized.lexically_normal();

        ec.clear();
        normalized = fs::absolute(input, ec);
        return ec ? input.lexically_normal() : normalized.lexically_normal();
    }

    inline bool is_central_convergence_outputdir(const std::string& outputdir)
    {
        return normalized_directory_for_compare(outputdir) ==
               normalized_directory_for_compare(get_cos_path());
    }

    inline std::ostream& FORMAT(std::ostream& os)
    {
        return os << std::setw(13) << std::left << std::showpos << std::showpoint;
    }

    // Solver operation modes
    constexpr int RELOAD_FILE = 2;
    constexpr int RUN_BOOST = 3;

    inline KadathApps::StageOutcome stage_outcome_from_legacy_status(int status)
    {
        if (status == EXIT_SUCCESS)
            return KadathApps::completed_stage();
        if (status == RELOAD_FILE)
            return KadathApps::loaded_different_dataset();
        if (status == RUN_BOOST)
            return KadathApps::request_binary_boost();
        return KadathApps::completed_stage(status);
    }

    inline int legacy_status_from_stage_outcome(const KadathApps::StageOutcome& outcome)
    {
        if (outcome.requires_field_reload())
            return RELOAD_FILE;
        if (outcome.requests_binary_boost())
            return RUN_BOOST;
        return outcome.code;
    }

    inline int legacy_status_from_dataset_resume(const KadathApps::DatasetResumeOutcome& outcome)
    {
        return legacy_status_from_stage_outcome(outcome.as_stage_outcome());
    }

    inline bool status_requires_field_reload(int status)
    {
        return stage_outcome_from_legacy_status(status).requires_field_reload();
    }

    inline bool status_requests_binary_boost(int status)
    {
        return stage_outcome_from_legacy_status(status).requests_binary_boost();
    }

    inline bool status_completed_without_reload(int status)
    {
        return stage_outcome_from_legacy_status(status).completed_without_reload();
    }

    template <typename config_t, typename space_t> class Solver
    {
      public:
        using base_config_t = std::decay_t<config_t>;
        using base_space_t = std::decay_t<space_t>;

      protected:
        space_t& space;
        config_t& bconfig;
        const int ndom;

      public:
        /**
         * @brief Constructs a Solver instance.
         * @param config_in Configuration object reference.
         * @param space_in Space object reference.
         */
        Solver(config_t& config_in, space_t& space_in)
            : space(space_in), bconfig(config_in), ndom(space_in.get_nbr_domains())
        {
        }

        virtual ~Solver() = default;

        // Delete copy operations to prevent unintended copies
        Solver(const Solver&) = delete;
        Solver& operator=(const Solver&) = delete;

        // Allow move operations
        Solver(Solver&&) = default;
        Solver& operator=(Solver&&) = default;

      protected:
        virtual void print_diagnostics(const System_of_eqs& syst, const int ite, const double conv) const = 0;

        virtual std::string converged_filename(const std::string& stage) const = 0;

        SolverRuntimeConfig with_stage_mumps_tree_cache(
            SolverRuntimeConfig config,
            const std::string& final_stage) const
        {
            if (!config.mumps.tree_cache_enabled)
                return config;

            fs::path cache_path =
                fs::path(bconfig.config_outputdir()) / converged_filename(final_stage);
            cache_path += ".mumpstree";
            return config.with_mumps_tree_cache_path(cache_path.string());
        }

        // Returns true when the iteration cap is hit but the residual already
        // sits within 10 * PREC: the caller stops iterating and keeps the
        // solution (converged-file write still happens). Throws only when the
        // cap is hit away from the solution. ite and conv are rank-consistent
        // (consensus residual), so all ranks take the same branch.
        bool check_max_iter_exceeded(const int& rank, const int& ite, const double& conv) const
        {
            const int max_iter = bconfig.template seq_setting_as<int>(MAX_ITER);
            const double precision = bconfig.template seq_setting_as<double>(PREC);
            const double tolerance = 10.0 * precision;

            if (ite > max_iter && conv < tolerance) {
                if (rank == 0)
                    std::cout << "Maximum iterations exceeded at precision: " << conv
                              << ". Finishing since precision < 10 * PREC."
                              << " Running at higher resolution may help.\n";
                return true;
            }
            if (ite > max_iter) {
                std::ostringstream oss;
                oss << "Maximum iterations exceeded at precision: " << conv;
                KADATH_THROW(oss.str());
            }
            return false;
        }

        KadathApps::DatasetResumeOutcome load_existing_solution(const std::string& last_stage = "")
        {
            const std::string current_abs = bconfig.config_filename_abs();
            const std::string prev_name = converged_filename(last_stage);
            const std::string prev_abs = bconfig.config_outputdir() + prev_name;
            const std::string central_abs = get_cos_path() + prev_name;
            const bool search_central_cache =
                is_central_convergence_outputdir(bconfig.config_outputdir());

            // When the run was launched directly with a stage's own converged
            // dataset, do not report "solved previously" and skip: the warm-start
            // fields for that dataset are already loaded, so re-run the stage
            // instead. The Newton driver evaluates the residual before assembling
            // any Jacobian and returns immediately once it is below precision, so
            // an unchanged dataset costs a single residual evaluation, while edited
            // inputs (e.g. ecc_omega/adot in an eccentricity-reduction iteration)
            // are honoured and actually re-solve -- rather than being silently
            // discarded because a matching converged file exists. Ladder resume is
            // unaffected: launching from initbin keeps config_filename != prev_name,
            // so a cached converged stage is still resumed without recomputation.
            if (bconfig.config_filename() == prev_name + ".toml") {
                return KadathApps::missing_dataset(current_abs);
            }

            auto check_and_update_config = [&](const std::string& path) -> bool {
                base_config_t old_solution(path + ".toml");

                // Verify shell configuration compatibility
                if (bconfig.set(NSHELLS) != old_solution.set(NSHELLS)) {
                    return false;
                }

                // Reject a resume whose resolution differs from the active
                // ladder rung. A half-stale central cache can hold two stage
                // datasets converged at different resolutions (e.g. DIRICHLET at
                // res 11, VON_NEUMANN at res 9); resuming the off-rung one makes
                // the driver's reload loop flip-flop between them forever instead
                // of converging. Treat such a dataset as absent so the stage
                // re-solves fresh at the current rung. Binary configs key the
                // resolution on BIN_RES, isolated-object (BCO) configs on
                // BCO_RES -- select the right enum for this config type.
                constexpr bool is_binary_config =
                    std::is_same_v<base_config_t, kadath_config<BIN_INFO>>;
                if constexpr (is_binary_config) {
                    if (bconfig.set(BIN_RES) != old_solution.set(BIN_RES))
                        return false;
                } else {
                    if (bconfig.set(BCO_RES) != old_solution.set(BCO_RES))
                        return false;
                }

                // Transfer current stages, controls, and settings to old solution
                for (size_t idx = 0; idx < NUM_STAGES_V; ++idx) {
                    old_solution.set_stage(idx) = bconfig.set_stage(idx);
                }
                for (size_t idx = 0; idx < NUM_CONTROLS_V; ++idx) {
                    old_solution.control(idx) = bconfig.control(idx);
                }
                for (size_t idx = 0; idx < NUM_SEQ_SETTINGS_V; ++idx) {
                    old_solution.seq_setting(idx) = bconfig.seq_setting(idx);
                }

                // Preserve the caller's requested spin magnitude (CHI) and spin-axis
                // tilt (DEG) across the resume. A cached seed records the *building*
                // body's CHI/DEG, and some seed names are spin-agnostic (e.g. the
                // non-rotating NOROT seed carries no z<DEG> tag), so a body with a
                // different requested spin can match and resume from it. Adopting
                // old_solution's CHI/DEG wholesale then silently overwrites the
                // requested values -- e.g. a tilted DEG=30 body reusing a DEG=0
                // seed loses its tilt, and a DEG=0 body reusing a DEG=30 NOROT seed
                // gains a spurious tilt. Treat CHI/DEG as current-run intent (like
                // stages/controls/settings above) so the requested spin survives.
                if constexpr (is_binary_config) {
                    for (auto& bco : {BCO1, BCO2}) {
                        old_solution.set(CHI, bco) = bconfig.set(CHI, bco);
                        old_solution.set(DEG, bco) = bconfig.set(DEG, bco);
                    }
                } else {
                    old_solution.set(CHI) = bconfig.set(CHI);
                    old_solution.set(DEG) = bconfig.set(DEG);
                }

                // Update configuration with loaded solution
                bconfig = old_solution;
                return true;
            };

            // Check local output directory first
            if (fs::exists(prev_abs + ".toml") && fs::exists(prev_abs + ".dat")) {
                if (check_and_update_config(prev_abs))
                    return KadathApps::resumed_dataset(current_abs, bconfig.config_filename_abs());
                return KadathApps::missing_dataset(current_abs);
            }
            else if (search_central_cache &&
                     fs::exists(central_abs + ".toml") &&
                     fs::exists(central_abs + ".dat")) {
                if (check_and_update_config(central_abs))
                    return KadathApps::resumed_dataset(current_abs, bconfig.config_filename_abs());
                return KadathApps::missing_dataset(current_abs);
            }

            return KadathApps::missing_dataset(current_abs);
        }

        bool solution_exists(const std::string& last_stage = "")
        {
            return load_existing_solution(last_stage).found();
        }
    };

    /**
     * @brief XCTS (Extended Conformal Thin Sandwich) solver implementation.
     *
     * Specialized solver for XCTS formulation of Einstein's equations.
     *
     * @tparam config_t Configuration type
     * @tparam space_t Space type
     */
    template <typename config_t, typename space_t> class XCTS_Solver : public Solver<config_t, space_t>
    {
      public:
        using cfgen_t = CoordFields<space_t>;
        using cfary_t = std::array<std::optional<Vector>, NUM_VECTORS_V>;
        using base_config_t = std::decay_t<config_t>;
        using base_space_t = std::decay_t<space_t>;

      protected:
        using Solver<config_t, space_t>::space;
        using Solver<config_t, space_t>::bconfig;
        using Solver<config_t, space_t>::check_max_iter_exceeded;

        Base_tensor& basis;
        cfgen_t cfields;
        cfary_t coord_vectors;

      public:
        XCTS_Solver(config_t& config_in, space_t& space_in, Base_tensor& base_in)
            : Solver<config_t, space_t>(config_in, space_in), basis(base_in), cfields(space), coord_vectors()
        {
        }

        // Delete copy operations
        XCTS_Solver(const XCTS_Solver&) = delete;
        XCTS_Solver& operator=(const XCTS_Solver&) = delete;

        // Allow move operations
        XCTS_Solver(XCTS_Solver&&) = default;
        XCTS_Solver& operator=(XCTS_Solver&&) = default;

      protected:
        // Register the baryonic/ADM mass unknown split shared by every XCTS NS
        // stage: when MB_FIXING is set Mb is held constant and Madm solved for,
        // otherwise the roles swap. Only instantiated when a derived stage calls
        // it, so configs lacking MB/MADM/MB_FIXING are unaffected.
        void add_mass_fixing(System_of_eqs& syst)
        {
            if (bconfig.control(MB_FIXING)) {
                syst.add_cst("Mb", bconfig(MB));
                syst.add_var("Madm", bconfig(MADM));
            } else {
                syst.add_var("Mb", bconfig(MB));
                syst.add_cst("Madm", bconfig(MADM));
            }
        }

        // Shared Newton-iteration driver: steps newton_step_with_consensus until
        // convergence, invoking per_iteration(ite, conv) after each step for the
        // stage-specific bookkeeping (filenames, diagnostics, checkpoint saves).
        // After the consensus loop reports convergence, any stage whose
        // per_iteration refreshes adapted-coordinate constants (and recomputes
        // conv) gets a second collective check so a residual that crept above
        // precision on the refreshed system reopens the loop; stages that leave
        // conv at the converged Newton value see this as a no-op.
        // per_iteration may return void (no early stop, used by NS stages) or
        // bool, where `false` requests an early stage abort (used by BNS stages
        // for BNS_STOP_AFTER_STEP). The if-constexpr keeps both forms
        // compiling without disturbing existing void callers. Returns true iff
        // the loop was aborted early.
        template <typename PerIteration>
        bool run_newton_loop(System_of_eqs& syst, SolverRuntimeConfig solver_config, PerIteration&& per_iteration)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            bool endloop = false;
            int ite = 1;
            double conv = 0.;
            while (!endloop) {
                const double precision = bconfig.template seq_setting_as<double>(PREC);
                endloop = newton_step_with_consensus(syst, precision, conv, solver_config, ite == 1);
                if constexpr (std::is_void_v<std::invoke_result_t<PerIteration&, int, double&>>) {
                    per_iteration(ite, conv);
                } else {
                    if (!per_iteration(ite, conv))
                        return true;  // early stage abort requested
                }
                if (endloop) {
                    const int local_done = conv < precision ? 1 : 0;
                    int global_done = 0;
                    MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                    endloop = global_done != 0;
                    if (!endloop && rank == 0) {
                        std::cout << "Refreshed adapted-coordinate residual " << conv
                                  << " is above precision " << precision
                                  << "; continuing Newton iterations." << std::endl;
                    }
                }
                ite++;
                if (check_max_iter_exceeded(rank, ite, conv))
                    break;
            }
            return false;  // loop completed normally
        }
    };

} // namespace Kadath
