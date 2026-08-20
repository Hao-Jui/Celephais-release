#pragma once

#include "Apps/Startup/app_startup.hpp"
#include "Apps/Formalism/GR/NS_3D_XCTS_nosym/solver.hpp"

#include <cstdlib>
#include <string>

namespace KadathApps
{
    struct NsGrNosymPolicy
    {
        using config_t = kadath_config<BCO_NS_INFO>;

        static void configure_defaults()
        {
        }

        static void write_example(int rank)
        {
            KadathApps::write_example_toml<config_t>(rank, "initial_ns_nosym.toml");
        }

        static const char* regrid_filename()
        {
            return "initns_regrid";
        }

        static void regrid(config_t& bconfig, int final_resolution, std::string& filename)
        {
            Kadath::ns_3d_xcts_nosym_interpolate_on_new_grid(bconfig, final_resolution, filename);
        }

        static int solve(config_t& bconfig, const std::string& outputdir)
        {
            return Kadath::ns_3d_xcts_nosym_driver(bconfig, outputdir);
        }
    };
} // namespace KadathApps
