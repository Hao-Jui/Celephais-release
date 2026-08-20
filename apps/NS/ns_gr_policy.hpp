#pragma once

#include "Apps/Startup/app_startup.hpp"
#include "Apps/Formalism/GR/NS_3D_XCTS/solver.hpp"
#include "Apps/Policy/ns_policy_common.hpp"

#include <string>

namespace KadathApps
{
    struct NsGrPolicy
    {
        using config_t = kadath_config<BCO_NS_INFO>;

        static void configure_defaults() {}

        static void write_example(int rank)
        {
            KadathApps::write_example_toml<config_t>(rank, "initial_ns.toml");
        }

        static const char* regrid_filename()
        {
            return "initns_regrid";
        }

        static void regrid(config_t& bconfig, int final_resolution, std::string& filename)
        {
            Kadath::ns_3d_xcts_interpolate_on_new_grid(bconfig, final_resolution, filename);
        }

        static int solve(config_t& bconfig, const std::string& outputdir)
        {
            return NsPolicyCommon::solve_ns_xcts<Kadath::GrNsDriverTraits>(bconfig, outputdir);
        }
    };
} // namespace KadathApps
