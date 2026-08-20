#include "mpi.h"
#include "solver/bh_2d_msqi_solver.hpp"
#include "solver/bh_2d_msqi_regrid.hpp"
#include "Apps/Startup/app_startup.hpp"

using namespace Kadath;

int main(int argc, char** argv) {
  int rank = KadathApps::init_mpi(argc, argv);

  int rc = KadathApps::guarded_run([&] {
    using config_t = kadath_config<BCO_BH_INFO>;
    if (argc < 2) {
      KadathApps::write_example_toml<config_t>(rank, "initial_bh.toml", [](config_t& example_config) {
        example_config.set_defaults();
        example_config.set(DIM) = 2;
        example_config.set(BCO_RES) = 11;
        example_config.seq_setting(INIT_RES) = 11;
      });
      return;
    }

    auto sr = KadathApps::parse_kadath_config_toml_startup<config_t>(argc, argv, rank);
    config_t bconfig = std::move(sr.bconfig);

    const int final_resolution = static_cast<int>(bconfig(BCO_RES));

    auto regrid = [&]() -> void {
      std::string fname{"bh2d_regrid"};
      if (rank == 0) {
        bh_2d_msqi_regrid(bconfig, final_resolution, fname);
      }
      bconfig.set_filename(fname);
      MPI_Barrier(MPI_COMM_WORLD);
    };

    if (bconfig.control(REGRID) && !sr.setup_first) {
      regrid();
    }

    bh_2d_msqi_driver(bconfig, sr.outputdir);
  });

  MPI_Finalize();
  return rc;
}
