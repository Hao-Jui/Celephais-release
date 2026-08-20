/*
    Added 2026 Hao-Jui Kuan
    JFNK Newton step (CELEPHAIS_SOLVER=jfnk-dense): exact matrix-free Jacobian-vector
    products (do_JX) preconditioned by a DISTRIBUTED DENSE LU factorization of the
    assembled Jacobian (ScaLAPACK pdgetrf/pdgetrs, 2D block-cyclic). The dense
    factor is built once and reused (lagged) across Newton steps and across every
    Krylov apply -- the dense analogue of do_newton_jfnk_mumps's sparse MUMPS PC.

    Build path: the Jacobian is assembled by JacobianAssembler (the W-lane
    packed-column AD -- W=32/16/8/4/2 columns per sweep + der_abs cache, the same
    fast assembler jfnk-mumps uses) into COO, then the COO triplets are SCATTERED
    into the 2D block-cyclic local blocks and pdgetrf-factored. The earlier
    do_col_J-direct build was ~4x slower because do_col_J is the scalar single-column
    path with no W-lane batching; routing the fast assembler's COO into the dense
    block-cyclic layout keeps the fast build and feeds ScaLAPACK its native format.

    A dense LU has no fill-in and needs no fill-reducing ordering, so it sidesteps
    sparse-ordering pathologies; the price is O(n^2/p) per-rank memory + O(n^3)
    factor, so this targets moderate n.
*/

#include "mpi.h"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"

#include "newton_norms.hpp"
#include "Linear_algebra/jacobian_assembler.hpp"
#include "Linear_algebra/krylov_solver.hpp"
#include "Linear_algebra/linear_solver.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace Kadath
{
    extern "C"
    {
    void sl_init_(int*, int*, int*);
    void blacs_gridexit_(int*);
    void blacs_gridinfo_(int*, int*, int*, int*, int*);
    int numroc_(int*, int*, int*, int*, int*);
    void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
    void pdgetrf_(int*, int*, double*, int*, int*, int*, int*, int*);
    void pdgetrs_(char*, int*, int*, double*, int*, int*, int*, int*, double*, int*, int*, int*, int*);
    }

    namespace
    {
        double elapsed_time_since(std::chrono::time_point<std::chrono::system_clock> begin)
        {
            return std::chrono::duration<double>(std::chrono::system_clock::now() - begin).count();
        }

        double euclidean_norm(const std::vector<double>& values)
        {
            double sum = 0.0;
            for (double v : values)
                sum += v * v;
            return std::sqrt(sum);
        }

        void copy_vector_to_array(const std::vector<double>& values, Array<double>& result)
        {
            for (std::size_t i = 0; i < values.size(); ++i)
                result.set(static_cast<int>(i)) = values[i];
        }

        const char* gmres_status_name(GmresStatus::Code code)
        {
            switch (code) {
            case GmresStatus::Code::Converged:     return "converged";
            case GmresStatus::Code::MaxIterations: return "max_iterations";
            case GmresStatus::Code::InvalidInput:  return "invalid_input";
            case GmresStatus::Code::Breakdown:     return "breakdown";
            }
            return "unknown";
        }

        void split_process_grid(int target, int& low, int& up)
        {
            up = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(target))));
            while (target % up)
                ++up;
            low = target / up;
        }

        void report_step_timing(int rank, int nproc, const char* label, double seconds)
        {
            double s_max = 0.0, s_min = 0.0, s_sum = 0.0;
            MPI_Reduce(&seconds, &s_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&seconds, &s_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&seconds, &s_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0)
                std::cout << label << ": " << s_sum / nproc << " / " << s_min << " / " << s_max
                          << " seconds (avg/min/max)" << std::endl;   // flush: long dense
                          // factors get SIGTERM'd at the time limit; unflushed phase
                          // lines would be lost (log looks stuck after the DOF print).
        }

        // Distributed dense LU preconditioner (ScaLAPACK). Owns its 2D block-cyclic
        // BLACS context + the factored block-cyclic block + pivots, reused for every
        // solve and lagged across Newton steps. Built from an assembled COO scattered
        // into the block-cyclic layout (no rank-0 dense gather). solve() distributes
        // the full (replicated) Krylov vector to a block-cyclic RHS, runs pdgetrs,
        // gathers the solution, and broadcasts it so every rank holds the same result.
        class ScalapackDenseFactor : public LinearSolver
        {
          public:
            ScalapackDenseFactor(int n, MPI_Comm comm, const AssembledJacobianCoo& coo)
                : comm_(comm), n_(n)
            {
                MPI_Comm_rank(comm_, &rank_);
                MPI_Comm_size(comm_, &nproc_);
                int zero_i = 0, one_i = 1, info = 0;
                int m = n_;

                const auto t_build = std::chrono::system_clock::now();
                split_process_grid(nproc_, nprow_, npcol_);
                sl_init_(&ictxt_, &nprow_, &npcol_);
                blacs_gridinfo_(&ictxt_, &nprow_, &npcol_, &prow_, &pcol_);
                nrow_ = numroc_(&m, &bsize_, &prow_, &zero_i, &nprow_);
                ncol_ = numroc_(&n_, &bsize_, &pcol_, &zero_i, &npcol_);
                J_.assign(static_cast<std::size_t>(nrow_) * static_cast<std::size_t>(ncol_), 0.0);
                descinit_(J_desc_.data(), &m, &n_, &bsize_, &bsize_, &zero_i, &zero_i,
                          &ictxt_, &nrow_, &info);
                scatter_coo_into_block_cyclic(coo);
                build_seconds_ = elapsed_time_since(t_build);

                const auto t_factor = std::chrono::system_clock::now();
                ipiv_.assign(static_cast<std::size_t>(nrow_ + bsize_), 0);
                pdgetrf_(&m, &n_, J_.data(), &one_i, &one_i, J_desc_.data(), ipiv_.data(), &info);
                if (info != 0) {
                    throw LinearSolverError(__FILE__, __LINE__,
                        "jfnk-dense: pdgetrf failed (INFO=" + std::to_string(info) +
                        (info > 0 ? "; zero pivot, Jacobian singular)" : "; illegal argument)"));
                }
                factor_seconds_ = elapsed_time_since(t_factor);
            }

            ~ScalapackDenseFactor() override { release(); }

            double build_seconds() const { return build_seconds_; }
            double factor_seconds() const { return factor_seconds_; }

            // LinearSolver interface: this PC is built+factored in the ctor and only
            // serves solves, so the COO-oriented entry points are inert.
            void set_pattern(int, long long, const int*, const int*) override {}
            void factor(const double*) override {}

            void solve(double* rhs_inout) override   // rhs_inout: full length-n vector, replicated
            {
                int zero_i = 0, one_i = 1, info = 0;
                int m = n_;
                int nrow_B = numroc_(&m, &bsize_, &prow_, &zero_i, &nprow_);
                int ncol_B = numroc_(&one_i, &bsize_, &pcol_, &zero_i, &npcol_);
                std::vector<double> B(static_cast<std::size_t>(nrow_B) * static_cast<std::size_t>(ncol_B), 0.0);
                if (pcol_ == 0) {
                    int idx = 0;
                    const int nblocks = (m + bsize_ - 1) / bsize_;
                    for (int gb = prow_; gb < nblocks; gb += nprow_) {
                        const int start = gb * bsize_;
                        const int limit = std::min(bsize_, m - start);
                        for (int k = 0; k < limit; ++k)
                            B[static_cast<std::size_t>(idx++)] = rhs_inout[start + k];
                    }
                }
                Array<int> B_desc(9);
                descinit_(B_desc.set_data(), &m, &one_i, &bsize_, &one_i, &zero_i, &zero_i,
                          &ictxt_, &nrow_B, &info);
                char trans = 'N';
                pdgetrs_(&trans, &n_, &one_i, J_.data(), &one_i, &one_i, J_desc_.data(),
                         ipiv_.data(), B.data(), &one_i, &one_i, B_desc.set_data(), &info);

                // Gather the block-cyclic solution column onto rank 0, reconstruct the
                // natural-order vector, broadcast back into rhs_inout.
                std::vector<int> n_elements(static_cast<std::size_t>(nproc_), 0);
                std::vector<int> disp(static_cast<std::size_t>(nproc_), 0);
                int offset = 0;
                for (int i = 0; i < nprow_; ++i) {
                    for (int j = 0; j < npcol_; ++j) {
                        const int r = i * npcol_ + j;
                        n_elements[static_cast<std::size_t>(r)] =
                            (j == 0) ? numroc_(&n_, &bsize_, &i, &zero_i, &nprow_) : 0;
                        disp[static_cast<std::size_t>(r)] = offset;
                        offset += n_elements[static_cast<std::size_t>(r)];
                    }
                }
                std::vector<double> recv(rank_ == 0 ? static_cast<std::size_t>(n_) : 0u);
                MPI_Gatherv(B.data(), n_elements[static_cast<std::size_t>(rank_)], MPI_DOUBLE,
                            rank_ == 0 ? recv.data() : nullptr, n_elements.data(), disp.data(),
                            MPI_DOUBLE, 0, comm_);
                if (rank_ == 0) {
                    const int nblocks = (n_ + bsize_ - 1) / bsize_;
                    for (int kb = 0; kb < nblocks; ++kb) {
                        const int proc_row = kb % nprow_;
                        const int r = proc_row * npcol_;
                        const int src_off = disp[static_cast<std::size_t>(r)] + (kb / nprow_) * bsize_;
                        const int dst_off = kb * bsize_;
                        const int copy_size = std::min(bsize_, n_ - dst_off);
                        std::memcpy(rhs_inout + dst_off, recv.data() + src_off,
                                    static_cast<std::size_t>(copy_size) * sizeof(double));
                    }
                }
                MPI_Bcast(rhs_inout, n_, MPI_DOUBLE, 0, comm_);
            }

            void reset() override { release(); }

          private:
            // Route the assembled COO (1-based, centralized on rank 0) into each
            // rank's 2D block-cyclic local block. Rank 0 bins every (gi,gj,v) by its
            // block-cyclic owner and Scatterv's the triplets; each owner places them
            // using its own local leading dimension.
            // ponytail: centralized scatter -- rank 0 holds the full COO + a packed
            // send buffer (~3*nnz doubles) transiently. Switch to distributed-COO
            // (ICNTL(18)=3) routing if that rank-0 transient ever bites.
            void scatter_coo_into_block_cyclic(const AssembledJacobianCoo& coo)
            {
                const int B = bsize_;
                std::vector<double> sendbuf;
                std::vector<int> sendcounts(static_cast<std::size_t>(nproc_), 0);
                std::vector<int> displs(static_cast<std::size_t>(nproc_), 0);
                if (rank_ == 0) {
                    for (long long k = 0; k < coo.nnz; ++k) {
                        const int gi = coo.irn[static_cast<std::size_t>(k)] - 1;
                        const int gj = coo.jcn[static_cast<std::size_t>(k)] - 1;
                        const int owner = ((gi / B) % nprow_) * npcol_ + ((gj / B) % npcol_);
                        sendcounts[static_cast<std::size_t>(owner)] += 3;
                    }
                    long long total = 0;
                    for (int r = 0; r < nproc_; ++r) {
                        displs[static_cast<std::size_t>(r)] = static_cast<int>(total);
                        total += sendcounts[static_cast<std::size_t>(r)];
                    }
                    sendbuf.resize(static_cast<std::size_t>(total));
                    std::vector<int> pos(displs);
                    for (long long k = 0; k < coo.nnz; ++k) {
                        const int gi = coo.irn[static_cast<std::size_t>(k)] - 1;
                        const int gj = coo.jcn[static_cast<std::size_t>(k)] - 1;
                        const double v = coo.a[static_cast<std::size_t>(k)];
                        const int owner = ((gi / B) % nprow_) * npcol_ + ((gj / B) % npcol_);
                        int& p = pos[static_cast<std::size_t>(owner)];
                        sendbuf[static_cast<std::size_t>(p)] = gi;
                        sendbuf[static_cast<std::size_t>(p + 1)] = gj;
                        sendbuf[static_cast<std::size_t>(p + 2)] = v;
                        p += 3;
                    }
                }
                int my_count = 0;
                MPI_Scatter(rank_ == 0 ? sendcounts.data() : nullptr, 1, MPI_INT,
                            &my_count, 1, MPI_INT, 0, comm_);
                std::vector<double> recv(static_cast<std::size_t>(my_count));
                MPI_Scatterv(rank_ == 0 ? sendbuf.data() : nullptr,
                             rank_ == 0 ? sendcounts.data() : nullptr,
                             rank_ == 0 ? displs.data() : nullptr, MPI_DOUBLE,
                             recv.data(), my_count, MPI_DOUBLE, 0, comm_);
                for (int t = 0; t + 2 < my_count; t += 3) {
                    const int gi = static_cast<int>(recv[static_cast<std::size_t>(t)]);
                    const int gj = static_cast<int>(recv[static_cast<std::size_t>(t + 1)]);
                    const double v = recv[static_cast<std::size_t>(t + 2)];
                    const int li = (gi / (B * nprow_)) * B + gi % B;
                    const int lj = (gj / (B * npcol_)) * B + gj % B;
                    J_[static_cast<std::size_t>(li) + static_cast<std::size_t>(lj) * nrow_] = v;
                }
            }

            void release()
            {
                if (ictxt_ >= 0) {
                    blacs_gridexit_(&ictxt_);
                    ictxt_ = -1;
                }
                std::vector<double>().swap(J_);
                std::vector<int>().swap(ipiv_);
            }

            MPI_Comm comm_;
            int rank_ = 0, nproc_ = 1;
            int n_ = 0;
            int bsize_ = 256;
            int ictxt_ = -1, nprow_ = 1, npcol_ = 1, prow_ = 0, pcol_ = 0;
            int nrow_ = 0, ncol_ = 0;
            std::array<int, 9> J_desc_{};
            std::vector<double> J_;      // factored block-cyclic local block
            std::vector<int> ipiv_;
            double build_seconds_ = 0.0;   // assemble-COO scatter into block-cyclic
            double factor_seconds_ = 0.0;  // pdgetrf
        };
    } // namespace

    bool System_of_eqs::do_newton_jfnk_dense(double precision, double& error,
                                             const SolverRuntimeConfig& config)
    {
        set_solver_runtime_config(config);
        int rank = 0, nproc = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);

        Array<double> residual(sec_member_partitioned());
        error = infinity_norm(residual);
        print_error_init_diagnostic(residual, error);
        if (error < precision)
            return true;

        const int row_count = static_cast<int>(residual.get_nbr());
        const int column_count = nbr_unknowns;
        if (row_count != column_count) {
            if (rank == 0)
                std::cerr << "do_newton_jfnk_dense: non-square system m=" << row_count
                          << " n=" << column_count << "; falling back to do_newton().\n";
            return do_newton(precision, error, System_of_eqs::SOLVER::NEWTON_RAPHSON);
        }

        const auto step_start = std::chrono::system_clock::now();
        const std::vector<double> rhs = [&]() {
            std::vector<double> out(static_cast<std::size_t>(column_count));
            for (int i = 0; i < column_count; ++i)
                out[static_cast<std::size_t>(i)] = residual(i);
            return out;
        }();

        // ---- Build or REUSE the distributed dense LU preconditioner -------------
        // Lagged across Newton steps (like jfnk-mumps): re-assemble + re-factor only
        // every refresh_steps (or on >10x error growth). Held in
        // mumps_runtime_state.jfnk_preconditioner.
        const double pre_step_error = error;
        auto& pc_state = mumps_runtime_state;
        const int refresh_steps = std::max(1, config.jfnk_mumps.preconditioner_refresh_steps);
        if (pc_state.jfnk_preconditioner_dimension != column_count ||
            pc_state.jfnk_preconditioner_refresh_steps != refresh_steps) {
            pc_state.jfnk_preconditioner.reset();
            pc_state.jfnk_step_index = 0;
            pc_state.jfnk_preconditioner_step_index = 0;
            pc_state.jfnk_previous_start_error = -1.0;
        }
        constexpr double kErrorGrowthRefresh = 10.0;
        const bool grew = pc_state.jfnk_preconditioner &&
                          pc_state.jfnk_previous_start_error > 0.0 &&
                          pre_step_error > kErrorGrowthRefresh * pc_state.jfnk_previous_start_error;
        const bool refresh = !pc_state.jfnk_preconditioner ||
                             pc_state.jfnk_preconditioner_step_index >= refresh_steps || grew;
        if (refresh) {
            pc_state.jfnk_preconditioner_step_index = 0;
            // Free the OLD factor (its O(n^2/p) block + BLACS context) BEFORE building
            // the new one. Otherwise the refresh transiently holds old factor + new
            // factor + the COO + the rank-0 scatter buffer simultaneously -- at large n
            // (res15: rank-0 RSS already ~18 GB) that doubled peak OOMs/segfaults the
            // first scheduled refresh. A refresh discards the old PC regardless, so
            // releasing it first is free. (res7 had headroom, so it never tripped --
            // this is the size-gated failure.) ponytail: the 8.5 GB rank-0 scatter
            // buffer is the remaining transient; distributed-COO scatter would remove it.
            pc_state.jfnk_preconditioner.reset();
            const double drop_tol = (config.mumps.drop_tol > 0.0) ? config.mumps.drop_tol : 1e-8;
            const auto assemble_start = std::chrono::system_clock::now();
            JacobianAssembler assembler(*this, MPI_COMM_WORLD);
            AssembledJacobianCoo coo = assembler.assemble(drop_tol);
            print_sparse_dof_rank_diagnostic(column_count, row_count, nproc, coo.nnz);
            report_step_timing(rank, nproc, "Jacobian assemble (W-lane COO)",
                               elapsed_time_since(assemble_start));
            auto pc_new = std::make_unique<ScalapackDenseFactor>(column_count, MPI_COMM_WORLD, coo);
            report_step_timing(rank, nproc, "dense scatter COO -> block-cyclic",
                               pc_new->build_seconds());
            report_step_timing(rank, nproc, "dense factorize (ScaLAPACK pdgetrf)",
                               pc_new->factor_seconds());
            pc_state.jfnk_preconditioner_dimension = column_count;
            pc_state.jfnk_preconditioner_refresh_steps = refresh_steps;
            pc_state.jfnk_preconditioner = std::move(pc_new);
        } else if (rank == 0) {
            std::cout << "JFNK-DENSE PC reuse: step=" << pc_state.jfnk_preconditioner_step_index
                      << "/" << refresh_steps << " (distributed dense factor lagged)\n";
        }
        LinearSolver& pc = *pc_state.jfnk_preconditioner;

        // ---- Matrix-free GMRES: exact Jv (do_JX) + distributed dense LU right-PC -
        Array<double> matvec_delta_array(column_count);
        const auto matvec = [this, column_count, &matvec_delta_array](
                                const std::vector<double>& trial_delta,
                                std::vector<double>& jacobian_delta) {
            copy_vector_to_array(trial_delta, matvec_delta_array);
            Array<double> applied_delta = do_JX(matvec_delta_array);
            jacobian_delta.resize(static_cast<std::size_t>(column_count));
            for (int i = 0; i < column_count; ++i)
                jacobian_delta[static_cast<std::size_t>(i)] = applied_delta(i);
        };
        const auto right_preconditioner =
            [&pc](const std::vector<double>& krylov_vector,
                  std::vector<double>& preconditioned_delta) {
                preconditioned_delta = krylov_vector;
                pc.solve(preconditioned_delta.data());
            };

        GmresConfig gmres_config;
        gmres_config.max_iters = config.jfnk_mumps.max_linear_iterations;
        const double rhs_norm = euclidean_norm(rhs);
        const double rtol = config.jfnk_mumps.linear_relative_tolerance;
        gmres_config.tolerance =
            std::max(std::numeric_limits<double>::epsilon(), rtol * rhs_norm);

        const auto krylov_start = std::chrono::system_clock::now();
        std::vector<double> delta(static_cast<std::size_t>(column_count), 0.0);
        const GmresStatus gmres_status =
            right_preconditioned_gmres(rhs, delta, matvec, right_preconditioner, gmres_config);
        if (rank == 0) {
            const double rel = rhs_norm > 0.0 ? gmres_status.residual_norm / rhs_norm
                                              : gmres_status.residual_norm;
            std::cout << "JFNK-DENSE GMRES: " << gmres_status_name(gmres_status.code) << " in "
                      << gmres_status.iterations << " iters, rel_resid=" << rel << '\n';
        }
        if (config.diagnostics.timing)
            report_step_timing(rank, nproc, "JFNK-DENSE GMRES solve", elapsed_time_since(krylov_start));

        // ---- Apply the correction (same path as do_newton_sparse) --------------
        Array<double> delta_array(column_count);
        copy_vector_to_array(delta, delta_array);
        int offset = 0;
        espace.xx_to_vars_variable_domains(this, delta_array, offset);
        xx_to_vars_delta(delta_array, offset);

        Array<double> trial_residual(sec_member_partitioned());
        error = infinity_norm(trial_residual);

        pc_state.jfnk_previous_start_error = pre_step_error;
        ++pc_state.jfnk_step_index;
        ++pc_state.jfnk_preconditioner_step_index;

        report_step_timing(rank, nproc, "JFNK-DENSE Newton step total", elapsed_time_since(step_start));
        return false;
    }
} // namespace Kadath
