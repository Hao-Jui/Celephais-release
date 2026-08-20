#pragma once
// App-layer orchestration for the isolated neutron-star solver, consistent with
// bns_app_workflow.hpp / bh_app_workflow.hpp: run_ns_solver_sequence drives the
// optional pre-solve regrid and hands the run to Policy::solve; run_ns_app is the
// main() entry (MPI init, config parse, example-template emission).
//
// The Kadath-namespace resolution-sequencing engine that Policy::solve
// ultimately dispatches into (ns_3d_xcts_isolated_driver_impl /
// ns_3d_xcts_boosted_driver_impl) lives one layer down in
// Apps/Workflow/ns_resolution_sequence.hpp, consumed by NsPolicyCommon::solve_ns_xcts
// via the per-formalism NsDriverTraits — mirroring how binary stage dispatch
// lives in Apps/Formalism/Shared/binary_driver_common.hpp.

#include "Apps/Startup/app_startup.hpp"
#include "For_Kadath/Config/config_binary.hpp"

#include <mpi.h>
#include <string>

namespace KadathApps
{
    // Policy-facing entry for the isolated-NS resolution sequence. Mirrors
    // run_bns_solver_sequence: the Workflow layer owns the continuation, and the
    // per-rung work goes through Policy::solve, which dispatches into
    // NsPolicyCommon::solve_ns_xcts (the resolution ladder + binary-boost
    // engine defined in ns_resolution_sequence.hpp).
    template <typename Policy>
    void run_ns_solver_sequence(typename Policy::config_t& bconfig,
                                int rank,
                                bool setup_first,
                                const std::string& outputdir)
    {
        const int final_resolution = static_cast<int>(bconfig(BCO_RES));
        auto regrid = [&]() {
            std::string fname{Policy::regrid_filename()};
            if (rank == 0)
                Policy::regrid(bconfig, final_resolution, fname);
            bconfig.set_filename(fname);
            MPI_Barrier(MPI_COMM_WORLD);
        };

        if (bconfig.control(REGRID) && !setup_first)
            regrid();

        Policy::solve(bconfig, outputdir);
    }

    template <typename Policy>
    int run_ns_app(int argc, char** argv)
    {
        const int rank = init_mpi(argc, argv);
        Policy::configure_defaults();

        const int rc = guarded_run([&] {
            using config_t = typename Policy::config_t;
            if (argc < 2) {
                Policy::write_example(rank);
                return;
            }

            auto sr = KadathApps::parse_kadath_config_toml_startup<config_t>(argc, argv, rank);
            config_t bconfig = std::move(sr.bconfig);
            throw_if_amr_unsupported(bconfig);

            run_ns_solver_sequence<Policy>(bconfig, rank, sr.setup_first, sr.outputdir);
        });

        MPI_Finalize();
        return rc;
    }
} // namespace KadathApps
