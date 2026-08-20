#include "mpi.h"

#include "solver/ns_2d_msqi_solver.hpp"
#include "Apps/Startup/app_startup.hpp"

using namespace Kadath;

int main(int argc, char** argv) {
  int rank = KadathApps::init_mpi(argc, argv);

  int rc = KadathApps::guarded_run([&] {
    using config_t = kadath_config<BCO_NS_INFO>;
    if (argc < 2) {
      KadathApps::write_example_toml<config_t>(rank, "initial_ns.toml", [](config_t& example_config) {
        example_config.set_defaults();
        example_config.set(DIM) = 2;
        example_config.set(BCO_RES) = 13;
        example_config.seq_setting(INIT_RES) = 13;
      });
      return;
    }

    auto sr = KadathApps::parse_kadath_config_toml_startup<config_t>(argc, argv, rank);
    config_t bconfig = std::move(sr.bconfig);
    KadathApps::throw_if_amr_unsupported(bconfig);

    ns_2d_msqi_driver(bconfig, sr.outputdir);
  });

  MPI_Finalize();
  return rc;
}
