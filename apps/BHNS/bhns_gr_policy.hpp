#pragma once

// BHNS app policy — mirrors apps/BNS/bns_gr_policy.hpp.
//
// The generic binary orchestration (res-increment sequence, MPI/config
// startup, the field-loading XCTS driver) lives in the shared workflow
// (Apps/Workflow/bns_app_workflow.hpp) and BnsPolicyCommon::solve_xcts, which
// are reused verbatim. Only the component-2 = BH specifics differ from BNS and
// live here: NS(MADM)+BH(MCH) binary config, NS+BH component setup, and the
// NS+BH superimposed import. The BHNS_XCTS solver now lives in the shared
// Formalism tree (include/Apps/Formalism/GR/BHNS_XCTS/); only the BH-specific
// config/setup/import below remain app-local.

#include "Apps/AMR/bhns_amr_policy.hpp"                  // amr_refine_bhns_fields_if_needed
#include "Apps/Policy/bns_policy_common.hpp"            // BnsPolicyCommon::{EosType, solve_xcts}
#include "Apps/Policy/ns_policy_common.hpp"             // NsPolicyCommon::solve_ns_xcts (NS seed)
#include "Apps/Startup/app_startup.hpp"                  // write_example_toml
#include "Apps/Formalism/Shared/PreBinary/bco_binary_setup.hpp"   // solve_NS_from_binary, solve_BH_from_binary, check_dist
#include "Apps/Formalism/GR/BHNS_XCTS/solver.hpp"       // Space_bhns, bhns_xcts_solver, bhns_setup_boosted_3d
#include "Apps/Formalism/GR/BHNS_XCTS/regrid.hpp"       // bhns_xcts_regrid
#include "Apps/Formalism/GR/NS_3D_XCTS/solver.hpp"      // GrNsDriverTraits (NS seed)
#include "Apps/Formalism/GR/BH_3D_XCTS/solver.hpp"      // bh_3d_xcts_driver (BH seed)
#include "Apps/Formalism/Shared/Stages/bns_stage_diagnostics.hpp" // env_flag

#include <array>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <string>

namespace KadathApps
{
    struct BhnsGrPolicy
    {
        using config_t = kadath_config<BIN_INFO>;
        using space_t = Kadath::Space_bhns;

        static void configure_defaults()
        {
            // Line search is the global default now (bit-exact on every space);
            // it bounds the BHNS momentum-sector overshoot in the free-Omega
            // von-Neumann pass. Convergence of that pass ALSO needs tight GMRES
            // + a fresh preconditioner near the solution — those live in the run
            // recipe, not here; see apps/BHNS/sandbox/README.md.
        }

        static void write_example(int rank)
        {
            KadathApps::write_example_toml<config_t>(rank, "initial_bhns.toml", [](config_t& example_config) {
                example_config.initialize_binary({"ns", "bh"});
                example_config.set_defaults();
            });
        }

        static const char* regrid_filename() { return "bhns_regrid"; }
        static bool stop_after_setup()
        {
            return Kadath::bns_diagnostics::env_flag("BHNS_STOP_AFTER_SETUP", false);
        }
        static bool stop_after_regrid()
        {
            return Kadath::bns_diagnostics::env_flag("BHNS_STOP_AFTER_REGRID", false);
        }

        // Low-resolution short-circuit stage for the res-increment sequence.
        static auto low_res_stage() { return QUASI_EQUIL; }

        static void setup_bin_config(config_t& bconfig)
        {
            // Component 1 = NS (MADM, BCO1), component 2 = BH (MCH, BCO2).
            // NS surface radius and BH excision radius both live in RMID.
            Kadath::check_dist(bconfig(DIST),
                {bconfig(MADM, BCO1), bconfig(RMID, BCO1), /*deformable=*/true },
                {bconfig(MCH , BCO2), bconfig(RMID, BCO2), /*deformable=*/false});

            bconfig.set(REXT) = 2 * bconfig(DIST);
            bconfig.set(Q) = bconfig(MADM, BCO1) / bconfig(MCH, BCO2);
            bconfig.set(COM) =
                bco_utils::com_estimate(bconfig(DIST), bconfig(MADM, BCO1), bconfig(MCH, BCO2));

            const auto pn_orbital_params =
                bco_utils::binary_pn_orbital_seed(bconfig, bconfig(MADM, BCO1), bconfig(MCH, BCO2));
            bconfig.set(ECC_OMEGA) = pn_orbital_params.omega;
            bconfig.set(ADOT) = pn_orbital_params.adot;
            bconfig.set(GOMEGA) = pn_orbital_params.omega;
            bconfig.set(ADOT) = std::nan("1");
        }

        static void regrid(config_t& bconfig, const std::string& filename)
        {
            Kadath::bhns_xcts_regrid(bconfig, filename);
        }

        static bool amr_refine(config_t& bconfig,
                               const std::string& outputdir,
                               int rank,
                               int cycle,
                               bns_amr::RefinementPolicyState& policy_state)
        {
            return bns_amr::amr_refine_bhns_fields_if_needed<config_t, Kadath::Space_bhns>(
                bconfig, outputdir, rank, cycle,
                [](config_t& config, const std::string& filename,
                   const std::vector<Kadath::Dim_array>& res_per_domain) {
                    Kadath::bhns_xcts_regrid(config, filename, res_per_domain);
                },
                &policy_state);
        }

        static bool amr_refine(config_t& bconfig, const std::string& outputdir, int rank, int cycle)
        {
            bns_amr::RefinementPolicyState policy_state;
            return amr_refine(bconfig, outputdir, rank, cycle, policy_state);
        }

        static int solve(config_t& bconfig, const std::string& outputdir, bool want_warmup)
        {
            // Generic field-loading XCTS driver, identical to BNS modulo Space/Solver.
            return BnsPolicyCommon::solve_xcts<Kadath::Space_bhns, Kadath::bhns_xcts_solver>(bconfig, outputdir,
                                                                                            want_warmup);
        }

        static void setup_space(config_t& bconfig)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            // Smoke/infra hook: the two component seed solves below (a full NS and
            // BH XCTS Newton solve each) are the bulk of BHNS setup and are far too
            // heavy for a smoke test. When set, skip them so the run still exercises
            // launch + MPI init + TOML parse + binary-config setup and then exits
            // cleanly (paired with stop_after_setup()). The space is left unbuilt.
            if (Kadath::bns_diagnostics::env_flag("BHNS_STOP_AFTER_SETUP", false)) {
                if (rank == 0)
                    std::cout << "BHNS setup phase reached: launch/parse/MPI/config OK "
                                 "(component seed solves skipped)" << std::endl;
                return;
            }

            std::array<std::string, 2> filenames;

            filenames[0] = Kadath::solve_NS_from_binary(
                bconfig, BCO1,
                [](auto& nsconfig, const std::string& seed_outputdir, auto& binconfig, NODES) {
                    nsconfig.set_stage(BIN_BOOST) = true;
                    NsPolicyCommon::solve_ns_xcts<Kadath::GrNsDriverTraits>(nsconfig, seed_outputdir, binconfig);
                });
            filenames[1] = Kadath::solve_BH_from_binary(
                bconfig, BCO2,
                [](auto& bhconfig, const std::string& seed_outputdir, auto& binconfig, NODES bco) {
                    bhconfig.set_stage(BIN_BOOST) = true;
                    Kadath::bh_3d_xcts_driver(bhconfig, seed_outputdir, binconfig, bco);
                });

            for (auto& filename : filenames) {
                if (rank == 0)
                    std::cout << filename << std::endl;
            }

            if (rank == 0)
                superimposed_import(bconfig, filenames);
            MPI_Barrier(MPI_COMM_WORLD);
            bconfig.open_config();
        }

        static void superimposed_import(config_t& bconfig, const std::array<std::string, 2>& filenames)
        {
            kadath_config<BCO_NS_INFO> ns_config(filenames[0]);
            kadath_config<BCO_BH_INFO> bh_config(filenames[1]);

            const double h_cut = ns_config.eos<double>(HCUT);
            const std::string eos_file = ns_config.eos<std::string>(EOSFILE);
            const std::string eos_type = ns_config.eos<std::string>(EOSTYPE);

            auto setup = [&](auto eos_tag, auto&&... eos_args) {
                using eos_t = typename decltype(eos_tag)::type;
                ::EOS<eos_t, ::eos_var_t::PRESSURE>::init(
                    eos_file, h_cut, std::forward<decltype(eos_args)>(eos_args)...);
                Kadath::bhns_setup_boosted_3d<eos_t>(ns_config, bh_config, bconfig);
            };

            if (eos_type == "Cold_PWPoly") {
                setup(BnsPolicyCommon::EosType<Kadath::Margherita::Cold_PWPoly>{});
            } else if (eos_type == "Cold_Table") {
                const int interp_pts = ns_config.eos<int>(INTERP_PTS) == 0
                    ? 2000
                    : ns_config.eos<int>(INTERP_PTS);
                setup(BnsPolicyCommon::EosType<Kadath::Margherita::Cold_Table>{}, interp_pts,
                      ns_config.eos<double>(MNUC_CGS));
            } else {
                KADATH_THROW("Unknown EOSTYPE.");
            }
        }
    };
} // namespace KadathApps
