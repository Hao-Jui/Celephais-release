#pragma once
// Workflow-layer entry point for the isolated black-hole solver, consistent
// with ns_app_workflow.hpp / bns_app_workflow.hpp.
//
// The BH continuation ladder (init_res -> final_res, +2 per rung) lives in
// bh_3d_xcts_driver because that driver is shared with the binary-boost path
// (Apps/Formalism/Shared/PreBinary/bco_binary_setup_imp.ipp) and its resolution stepping
// is fused with the chi-continuation / binary-boost state machine. The +2 step
// policy itself is single-sourced through solver_sequence.hpp (next_resolution),
// the same policy NS and BNS use.

#include "Apps/Startup/app_startup.hpp"
#include "Apps/Workflow/solver_sequence.hpp"
#include "Apps/Formalism/GR/BH_3D_XCTS/solver.hpp"

#include <mpi.h>
#include <utility>

namespace KadathApps
{
    inline int run_bh_app(int argc, char** argv)
    {
        const int rank = init_mpi(argc, argv);

        const int rc = guarded_run([&] {
            using config_t = kadath_config<BCO_BH_INFO>;
            if (argc < 2) {
                write_example_toml<config_t>(rank, "initial_bh.toml");
                return;
            }

            auto sr = KadathApps::parse_kadath_config_toml_startup<config_t>(argc, argv, rank);
            config_t bconfig = std::move(sr.bconfig);
            throw_if_amr_unsupported(bconfig);

            Kadath::bh_3d_xcts_driver(bconfig, sr.outputdir);
        });

        MPI_Finalize();
        return rc;
    }
} // namespace KadathApps
