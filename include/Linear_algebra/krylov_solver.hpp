#pragma once

#include <functional>
#include <vector>

namespace Kadath
{
    // Per-phase wall-clock accumulators for right_preconditioned_gmres.
    // All values are seconds. Phases account for the kernel's inner loop;
    // setup/teardown outside the loop is not split.
    struct GmresTiming {
        double precondition_seconds = 0.0;   // right_preconditioner(...) (e.g. MUMPS solve apply)
        double matvec_seconds = 0.0;         // matvec(...) (e.g. exact Jv)
        double orthog_seconds = 0.0;         // modified Gram-Schmidt, including final ||w||
        double vector_norm_seconds = 0.0;    // standalone norms, including initial ||r||
        double least_squares_seconds = 0.0;  // Givens + prefix rank guards + final back-substitution
        double update_seconds = 0.0;         // apply_solution_update on the retained solution basis
        int matvecs = 0;
        int preconditions = 0;
    };

    struct GmresConfig {
        int max_iters = 80;
        double tolerance = 1e-8;
        // Optional. If non-null, the kernel writes per-phase wall accumulators
        // into the pointee. Caller owns the storage and is responsible for
        // zeroing between solves. Null skips timing-clock reads.
        GmresTiming* timing = nullptr;
        // Optional (diagnostic). If non-null, the kernel appends the exact GMRES
        // residual norm after each iteration (|rotated_rhs[k+1]|). Caller owns
        // and clears the storage between solves. Used by the p-coarse PC probe to
        // report per-100-iteration convergence slope; zero overhead when null.
        std::vector<double>* residual_history = nullptr;
        // Optional (diagnostic). If both are non-null, on a Converged or
        // MaxIterations return the kernel MOVES the Arnoldi basis V (one entry per
        // basis column v_k, each length n) and the RAW (pre-Givens) upper-Hessenberg
        // H (row-major, capacity (m+1) x m; only the leading arnoldi_basis_out->size()
        // columns are populated) into the pointees. This exposes the Krylov data
        // needed for a post-hoc Ritz-value spectral analysis of the preconditioned
        // operator A*M (eigenvalues of the leading m x m block of H). Caller owns
        // the storage; both pointers must be set together. Zero overhead when null
        // (no raw copy is maintained, no behavior change).
        std::vector<std::vector<double>>* arnoldi_basis_out = nullptr;
        std::vector<std::vector<double>>* hessenberg_raw_out = nullptr;
    };

    struct GmresStatus {
        enum class Code {
            Converged,
            MaxIterations,
            InvalidInput,
            Breakdown
        };

        Code code = Code::InvalidInput;
        bool converged = false;
        int iterations = 0;
        double residual_norm = 0.0;
    };

    using KrylovOperator = std::function<void(const std::vector<double>&, std::vector<double>&)>;

    // x is output-only: any input contents are discarded before a returned solution.
    GmresStatus right_preconditioned_gmres(const std::vector<double>& b,
                                           std::vector<double>& x,
                                           const KrylovOperator& matvec,
                                           const KrylovOperator& right_preconditioner,
                                           const GmresConfig& config);
} // namespace Kadath
