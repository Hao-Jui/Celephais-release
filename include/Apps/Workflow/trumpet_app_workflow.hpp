#pragma once

#include "Apps/Formalism/GR/Trumpet_XCTS/solver.hpp"
#include "Apps/Startup/app_startup.hpp"

#include <mpi.h>
#include <utility>

namespace KadathApps
{
    inline int run_trumpet_app(int argc, char** argv)
    {
        const int rank = init_mpi(argc, argv);

        const int rc = guarded_run([&] {
            using config_t = kadath_config<BCO_BH_INFO>;
            if (argc < 2) {
	                write_example_toml<config_t>(rank, "initial_trumpet.toml", [](config_t& example_config) {
	                    example_config.set_defaults();
	                    example_config.set(TRUMPET_BH_SEED) = true;
	                    example_config.set(CHI) = 0.;
	                    Kadath::Seed::NonRotatingTrumpetProfile profile(static_cast<double>(example_config(MCH)));
	                    example_config.set(RMID) = profile.horizon_isotropic_radius();
	                    example_config.set(RIN) = 0.5 * example_config(RMID);
	                    example_config.set(ROUT) = 4. * example_config(MCH);
	                    example_config.seq_setting(INIT_RES) = example_config(BCO_RES);
	                });
                return;
            }

            auto sr = KadathApps::parse_kadath_config_toml_startup<config_t>(argc, argv, rank);
            config_t bconfig = std::move(sr.bconfig);
            throw_if_amr_unsupported(bconfig);
            bconfig.set(TRUMPET_BH_SEED) = true;
            bconfig.set(CHI) = 0.;

            Kadath::trumpet_3d_xcts_driver(bconfig, sr.outputdir);
        });

        MPI_Finalize();
        return rc;
    }
} // namespace KadathApps
