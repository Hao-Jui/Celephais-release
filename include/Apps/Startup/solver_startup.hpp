#pragma once

#include "For_Kadath/Array/exceptions.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <mpi.h>
#include <string>
#include <utility>

namespace KadathApps
{
    // Default KadathError handler: print a clear diagnostic and report a non-zero
    // status so the caller (run_*_app) finalizes MPI and exits cleanly -- no
    // MPI_Abort, hence no SIGABRT on an ordinary solver failure. The solver's
    // terminal failures (Newton non-convergence, a singular MUMPS factorization,
    // a non-finite residual) are collective: every rank reaches the same verdict
    // through an MPI_Allreduce and so enters this handler together, making the
    // subsequent collective MPI_Finalize safe.
    inline int report_failure_on_error(const Kadath::KadathError& error)
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        int rank = 0;
        if (initialized)
            MPI_Comm_rank(MPI_COMM_WORLD, &rank); // NOLINT(bugprone-casting-through-void)
        std::cerr << "KadathError [rank=" << rank << "]: " << error.what() << '\n'
                  << "Solver stage did not converge/complete; exiting with failure status.\n";
        return EXIT_FAILURE;
    }

    // Hard-abort handler, retained for genuinely rank-divergent (non-collective)
    // failures where letting one rank return would hang the others at the next
    // collective. Opt back in by passing it explicitly as the guarded_run on_error.
    inline int mpi_abort_on_error(const Kadath::KadathError& error)
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        int rank = 0;
        if (initialized)
            MPI_Comm_rank(MPI_COMM_WORLD, &rank); // NOLINT(bugprone-casting-through-void)
        std::cerr << "KadathError [rank=" << rank << "]: " << error.what() << '\n';
        if (initialized)
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); // NOLINT(bugprone-casting-through-void)
        std::_Exit(EXIT_FAILURE);
    }

    template <class Callable, class OnError = decltype(&report_failure_on_error)>
    int guarded_run(Callable&& callable, OnError&& on_error = &report_failure_on_error)
    {
        try {
            std::forward<Callable>(callable)();
            return 0;
        } catch (const Kadath::KadathError& e) {
            return std::forward<OnError>(on_error)(e);
        }
    }

    // Kadath parallelism is pure-MPI: the solver has no thread parallelism, only
    // the BLAS/LAPACK inside the MUMPS factor would spawn threads. When several
    // ranks share a node (the normal case, ranks ~= cores) every BLAS thread is
    // pure oversubscription -- np ranks x N threads >> cores -- and the MUMPS
    // factor anti-scales (measured: BNS res9 np=8 factor 5.5 s -> 22-32 s, wall
    // +1.5x). Pin one BLAS thread per rank unless the user explicitly asked for
    // threads (a deliberate hybrid MPI+threads run on an undersubscribed node).
    inline void pin_blas_to_single_thread_for_mpi(int world_size, int rank)
    {
        if (world_size <= 1) {
            return; // a single rank cannot oversubscribe -- let BLAS use the node
        }
        for (const char* var : {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                "GOTO_NUM_THREADS", "BLIS_NUM_THREADS"}) {
            const char* value = std::getenv(var);
            if (value != nullptr && value[0] != '\0') {
                return; // explicit user thread setting -- respect it
            }
        }
        // Env first: covers any BLAS that reads it lazily, child processes, and
        // makes the setting visible to diagnostics.
        ::setenv("OMP_NUM_THREADS", "1", 1);
        ::setenv("OPENBLAS_NUM_THREADS", "1", 1);
        ::setenv("MKL_NUM_THREADS", "1", 1);
        ::setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
        // The BLAS fixes its thread count in a load-time constructor, so the env
        // above is already too late -- override the live thread pool through the
        // library's runtime setter. dlsym(RTLD_DEFAULT, ...) finds it in whichever
        // library is loaded, with no link-time dependency, and is a no-op if
        // absent. omp_set_num_threads is the effective control for an OpenMP-built
        // OpenBLAS (its GEMM parallelism is OpenMP regions, which
        // openblas_set_num_threads does not gate) and also caps MKL via its OMP
        // runtime; the others cover the pthread-OpenBLAS and explicit-MKL builds.
        //
        // CRITICAL: only the C entry points are called here -- each takes int by
        // value. Do NOT add the lowercase Fortran aliases (omp_set_num_threads_,
        // mkl_set_num_threads in libmkl_gf): those take int* by reference, so a
        // dlsym call passing the value 1 dereferences address 0x1 and segfaults.
        // MKL's C API is the CamelCase MKL_Set_Num_Threads.
        using set_threads_t = void (*)(int);
        for (const char* symbol : {"omp_set_num_threads", "openblas_set_num_threads",
                                   "goto_set_num_threads", "MKL_Set_Num_Threads"}) {
            if (auto* setter = reinterpret_cast<set_threads_t>(
                    ::dlsym(RTLD_DEFAULT, symbol))) {
                setter(1);
            }
        }
        if (rank == 0) {
            std::cout << "[kadath] pure-MPI run on " << world_size
                      << " ranks: pinned BLAS to 1 thread/rank "
                      << "(export OMP_NUM_THREADS to override)\n";
        }
    }

    inline int init_mpi(int argc, char** argv)
    {
        int rc = MPI_Init(&argc, &argv);
        if (rc != MPI_SUCCESS) {
            std::cerr << "Error starting MPI" << '\n';
            MPI_Abort(MPI_COMM_WORLD, rc); // NOLINT(bugprone-casting-through-void)
        }
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); // NOLINT(bugprone-casting-through-void)
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size); // NOLINT(bugprone-casting-through-void)
        pin_blas_to_single_thread_for_mpi(world_size, rank);
        return rank;
    }

    inline std::string toml_config_path_from_reader_input(std::string input_path)
    {
        if (input_path.ends_with(".toml")) {
            return input_path;
        }
        if (input_path.ends_with(".dat")) {
            input_path.resize(input_path.size() - 4);
            input_path += ".toml";
        }
        return input_path;
    }

    template <class config_t>
    struct StartupResult
    {
        config_t bconfig;
        std::string outputdir;
        bool example_setup;
        bool setup_first;
    };
} // namespace KadathApps
