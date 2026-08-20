#pragma once

#include "Apps/AMR/bns_amr_policy.hpp"
#include "Apps/Policy/bns_policy_common.hpp"
#include "Apps/Policy/ns_policy_common.hpp"
#include "Apps/Startup/app_startup.hpp"
#include "Apps/Formalism/GR/BNS_XCTS/solver.hpp"
#include "Apps/Formalism/GR/BNS_XCTS/regrid.hpp"
#include "Apps/Formalism/GR/NS_3D_XCTS/solver.hpp"
#include "Apps/Formalism/Shared/PreBinary/bco_binary_setup.hpp"

#include <array>
#include <iostream>
#include <string>

namespace KadathApps
{
    struct BnsGrPolicy
    {
        using config_t = kadath_config<BIN_INFO>;
        using space_t = Kadath::Space_bin_ns;

        static void configure_defaults()
        {
            // Line search is the global default now (bit-exact on every space);
            // it is load-bearing for BNS (d=32 plateau rescue).
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
            return false;
        }

        static bool stop_after_regrid()
        {
            return false;
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
            Kadath::bns_xcts_regrid(bconfig, filename);
        }

        static int solve(config_t& bconfig, const std::string& outputdir)
        {
            return BnsPolicyCommon::solve_xcts<Kadath::Space_bin_ns, Kadath::bns_xcts_solver>(bconfig, outputdir);
        }

        static bool amr_refine(config_t& bconfig,
                               const std::string& outputdir,
                               int rank,
                               int cycle,
                               bns_amr::RefinementPolicyState& policy_state)
        {
            return bns_amr::amr_refine_common_fields_if_needed<config_t, Kadath::Space_bin_ns>(
                bconfig, outputdir, rank, cycle,
                [](config_t& config, const std::string& filename, const std::vector<Kadath::Dim_array>& res_per_domain) {
                    Kadath::bns_xcts_regrid(config, filename, res_per_domain);
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

            for (int i = 0; i < 2; ++i)
                filenames[static_cast<std::size_t>(i)] = Kadath::solve_NS_from_binary(
                    bconfig, bcos[i],
                    [](auto& nsconfig, const std::string& seed_outputdir, auto& binconfig, NODES bco) {
                        nsconfig.set_stage(BIN_BOOST) = true;
                        NsPolicyCommon::solve_ns_xcts<Kadath::GrNsDriverTraits>(
                            nsconfig, seed_outputdir, binconfig, bco);
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
            BnsPolicyCommon::superimposed_import(
                bconfig, filenames,
                [](auto eos_tag,
                   kadath_config<BCO_NS_INFO>& ns1_config,
                   kadath_config<BCO_NS_INFO>& ns2_config,
                   kadath_config<BIN_INFO>& binary_config) {
                    using eos_t = typename decltype(eos_tag)::type;
                    Kadath::bns_setup_boosted_3d<eos_t>(ns1_config, ns2_config, binary_config);
                });
        }
    };
} // namespace KadathApps
