#pragma once

#include "Apps/AMR/bns_amr_policy.hpp"
#include "Apps/Policy/bns_policy_common.hpp"
#include "Apps/Startup/app_startup.hpp"
#include "Apps/Formalism/GR/BNS_XCTS_nosym/solver.hpp"
#include "Apps/Formalism/GR/BNS_XCTS_nosym/regrid.hpp"
#include "Apps/Formalism/GR/NS_3D_XCTS_nosym/solver.hpp"
#include "Apps/Formalism/Shared/PreBinary/bco_binary_setup.hpp"
#include "Apps/Formalism/Shared/Stages/bns_stage_diagnostics.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace KadathApps
{
    struct BnsGrNosymPolicy
    {
        using config_t = kadath_config<BIN_INFO>;
        using space_t = Kadath::Space_bin_ns_nosym;

        static void configure_defaults()
        {
            set_default_env("JFNK_EW", "0");
            set_default_env("JFNK_RTOL", "1e-10");
            set_default_env("JFNK_MAX_ITERS", "256");
        }

        static void write_example(int rank)
        {
            KadathApps::write_example_toml<config_t>(rank, "initial_bns.toml", [](config_t& example_config) {
                example_config.initialize_binary({"ns", "ns"});
                example_config.set_defaults();
            });
        }

        static const char* regrid_filename()
        {
            return "bns_regrid";
        }

        // Low-resolution short-circuit stage for the res-increment sequence.
        static auto low_res_stage() { return QUASI_EQUIL; }

        static bool stop_after_setup()
        {
            return Kadath::bns_diagnostics::env_flag("BNS_STOP_AFTER_SETUP", false);
        }

        static bool stop_after_regrid()
        {
            return Kadath::bns_diagnostics::env_flag("BNS_STOP_AFTER_REGRID", false);
        }

        static void setup_bin_config(config_t& bconfig)
        {
            BnsPolicyCommon::setup_binary_ns_config(bconfig,
                [](double dist, double m1, double r1, double m2, double r2) {
                    // Both components are neutron stars (deformable).
                    Kadath::check_dist(dist, {m1, r1, true}, {m2, r2, true});
                });
        }

        static void regrid(config_t& bconfig, const std::string& filename)
        {
            Kadath::bns_xcts_nosym_regrid(bconfig, filename);
        }

        static int solve(config_t& bconfig, const std::string& outputdir)
        {
            return BnsPolicyCommon::solve_xcts<Kadath::Space_bin_ns_nosym, Kadath::bns_xcts_nosym_solver>(
                bconfig, outputdir);
        }

        static bool amr_refine(config_t& bconfig,
                               const std::string& outputdir,
                               int rank,
                               int cycle,
                               bns_amr::RefinementPolicyState& policy_state)
        {
            return bns_amr::amr_refine_common_fields_if_needed<config_t, Kadath::Space_bin_ns_nosym>(
                bconfig, outputdir, rank, cycle,
                [](config_t& config, const std::string& filename, const std::vector<Kadath::Dim_array>& res_per_domain) {
                    Kadath::bns_xcts_nosym_regrid(config, filename, res_per_domain);
                },
                &policy_state);
        }

        static bool amr_refine(config_t& bconfig, const std::string& outputdir, int rank, int cycle)
        {
            bns_amr::RefinementPolicyState policy_state;
            return amr_refine(bconfig, outputdir, rank, cycle, policy_state);
        }

        static void setup_space(config_t& bconfig)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::array<NODES, 2> bcos{BCO1, BCO2};
            std::array<std::string, 2> filenames;

            [[maybe_unused]] bool tilted_spin = false;  // consumed by the commented FORCE_BALANCE guard below
            for (auto bco : bcos) {
                if (std::isnan(bconfig.set(DEG, bco))) {
                    bconfig.set(DEG, bco) = 0.;
                }
                if (bconfig.set(DEG, bco) != 0.)
                    tilted_spin = true;
            }
            // The free-Omega FORCE_BALANCE pass assumes the spins lie along the
            // orbital axis (force balance evaluated in the equatorial plane). A
            // tilted-spin nosym seed (DEG != 0 — the nosym app's reason to exist)
            // makes that pass ill-posed: it diverges on the cold seed and SIGABRTs
            // before ECC_RED ever runs. Gate it off so the cold path is
            // ECC_RED-direct, which converges on the tilted seed.
            // if (tilted_spin && bconfig.set_stage(FORCE_BALANCE)) {
            //     bconfig.set_stage(FORCE_BALANCE) = false;
            //     if (rank == 0)
            //         std::cout << "BNS_nosym setup: tilted spin (DEG != 0) -> disabling "
            //                      "FORCE_BALANCE; cold path is ECC_RED-direct" << std::endl;
            // }
            if (rank == 0) {
                std::cout << "BNS_nosym setup: using nosym isolated-NS seed" << std::endl;
            }

            for (int i = 0; i < 2; ++i) {
                filenames[static_cast<std::size_t>(i)] = Kadath::solve_NS_from_binary(
                    bconfig, bcos[i],
                    [](auto& nsconfig, const std::string& seed_outputdir, auto& binconfig, NODES) {
                        nsconfig.set_stage(BIN_BOOST) = true;
                        Kadath::ns_3d_xcts_nosym_driver(nsconfig, seed_outputdir, binconfig);
                    });
            }

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
            BnsPolicyCommon::superimposed_import(
                bconfig, filenames,
                [](auto eos_tag,
                   kadath_config<BCO_NS_INFO>& ns1_config,
                   kadath_config<BCO_NS_INFO>& ns2_config,
                   kadath_config<BIN_INFO>& binary_config) {
                    using eos_t = typename decltype(eos_tag)::type;
                    Kadath::bns_setup_boosted_3d_nosym<eos_t>(ns1_config, ns2_config, binary_config);
                });
        }

    };
} // namespace KadathApps
