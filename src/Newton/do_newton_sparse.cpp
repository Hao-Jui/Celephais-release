/*
    Added 2026 Hao-Jui Kuan
    Newton step via MUMPS direct solve:
    direct Newton with analytically-assembled sparse-approximate Jacobian (MUMPS factorization)
*/

#include "mpi.h"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/System_of_eqs/system_dof_record.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"

#include "Linear_algebra/jacobian_assembler.hpp"
#include "Linear_algebra/jacobian_parity_mask.hpp"
#include "Linear_algebra/sparse_direct_mumps_solve.hpp"

#ifdef CELEPHAIS_USE_MUMPS
#include "Linear_algebra/captured_linear_system.hpp"
#include "Linear_algebra/mumps_linear_solver.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace Kadath
{
    extern "C"
    {
    void dump_ope_mult_profile();
    void reset_ope_mult_profile();
    void dump_ope_action_profile();
    void reset_ope_action_profile();
    }

    void print_sparse_direct_mumps_system_summary(
        std::ostream& diagnostic, SparseDirectMumpsSystemMode mode,
        int dof, long long nnz)
    {
        const bool reduced =
            mode == SparseDirectMumpsSystemMode::ReducedMasked;
        const bool masked =
            mode != SparseDirectMumpsSystemMode::FullUnmasked;
        diagnostic << "Sector: " << (reduced ? "reduced" : "full")
                   << ", Mask: " << (masked ? "on" : "off")
                   << " | System: " << dof << " (" << nnz << " nnz)"
                   << std::endl;
    }

    SparseDirectMumpsSystemMode sparse_direct_mumps_system_mode(
        bool reduced_sector,
        const JacobianParityMaskState* parity_state) noexcept
    {
        if (reduced_sector)
            return SparseDirectMumpsSystemMode::ReducedMasked;
        if (parity_state != nullptr &&
            parity_state->decision ==
                JacobianParityMaskState::Decision::Engaged) {
            return SparseDirectMumpsSystemMode::FullMasked;
        }
        return SparseDirectMumpsSystemMode::FullUnmasked;
    }

    std::size_t sparse_direct_mumps_coo_allocated_bytes(
        const AssembledJacobianCoo& assembled_jacobian) noexcept
    {
        const auto block_bytes = [](const AssembledJacobianCooBlock& block) {
            return block.irn.capacity() * sizeof(int) +
                   block.jcn.capacity() * sizeof(int) +
                   block.a.capacity() * sizeof(double);
        };
        std::size_t bytes =
            assembled_jacobian.irn.capacity() * sizeof(int) +
            assembled_jacobian.jcn.capacity() * sizeof(int) +
            assembled_jacobian.a.capacity() * sizeof(double);
        for (const AssembledJacobianCooBlock& block :
             assembled_jacobian.parity_blocks) {
            bytes += block_bytes(block);
        }
        return bytes;
    }

    void report_sparse_direct_mumps_jacobian_timing(
        MPI_Comm communicator, std::ostream* diagnostic, double seconds,
        std::size_t coo_allocated_bytes)
    {
        double seconds_max = 0.0;
        unsigned long long bytes_local =
            static_cast<unsigned long long>(coo_allocated_bytes);
        unsigned long long bytes_max = 0;
        MPI_Reduce(&seconds, &seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                   communicator);
        MPI_Reduce(&bytes_local, &bytes_max, 1, MPI_UNSIGNED_LONG_LONG,
                   MPI_MAX, 0, communicator);

        int rank = 0;
        MPI_Comm_rank(communicator, &rank);
        if (rank == 0 && diagnostic != nullptr) {
            constexpr double bytes_per_decimal_mb = 1.0e6;
            *diagnostic << "Jacobian build+gather: " << seconds_max
                        << " s (COO: "
                        << static_cast<double>(bytes_max) /
                               bytes_per_decimal_mb
                        << " MB)" << std::endl;
        }
    }

    void report_sparse_direct_mumps_factorization(
        MPI_Comm communicator, std::ostream* diagnostic, const char* label,
        double analyze_seconds, double factorize_seconds,
        const std::string& ordering, bool out_of_core,
        int factor_memory_used_mb, int factor_memory_allocated_mb)
    {
        double analyze_seconds_max = 0.0;
        double factorize_seconds_max = 0.0;
        MPI_Reduce(&analyze_seconds, &analyze_seconds_max, 1, MPI_DOUBLE,
                   MPI_MAX, 0, communicator);
        MPI_Reduce(&factorize_seconds, &factorize_seconds_max, 1, MPI_DOUBLE,
                   MPI_MAX, 0, communicator);

        int rank = 0;
        MPI_Comm_rank(communicator, &rank);
        if (rank == 0 && diagnostic != nullptr) {
            *diagnostic << label << ": " << analyze_seconds_max << " + "
                        << factorize_seconds_max << " s (" << ordering
                        << ", OOC " << (out_of_core ? "on" : "off")
                        << ", " << factor_memory_used_mb << " MB used, "
                        << factor_memory_allocated_mb << " MB allocated)"
                        << std::endl;
        }
    }

    void report_sparse_direct_mumps_apply_timing(
        MPI_Comm communicator, std::ostream* diagnostic, double seconds)
    {
        double seconds_max = 0.0;
        MPI_Reduce(&seconds, &seconds_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                   communicator);

        int rank = 0;
        MPI_Comm_rank(communicator, &rank);
        if (rank == 0 && diagnostic != nullptr) {
            *diagnostic << "MUMPS apply: " << seconds_max << " s"
                        << std::endl;
        }
    }

#ifdef CELEPHAIS_USE_MUMPS
    int resolve_sparse_direct_mumps_ranks_per_node(
        const MumpsRuntimeConfig& config, MPI_Comm communicator)
    {
        if (config.ranks_per_node >= 0)
            return config.ranks_per_node;

        int local_ranks = 1;
        MPI_Comm node_comm = MPI_COMM_NULL;
        MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &node_comm);
        MPI_Comm_size(node_comm, &local_ranks);
        MPI_Comm_free(&node_comm);
        return std::max(local_ranks / 4, 1);
    }

    void sparse_direct_collective_throw_if_failed(
        MPI_Comm communicator, const char* phase,
        const std::string& local_error)
    {
        int rank = 0;
        int size = 1;
        MPI_Comm_rank(communicator, &rank);
        MPI_Comm_size(communicator, &size);

        const int local_failure_rank = local_error.empty() ? size : rank;
        int failure_rank = size;
        MPI_Allreduce(&local_failure_rank, &failure_rank, 1, MPI_INT, MPI_MIN,
                      communicator);
        if (failure_rank == size)
            return;

        int message_size = rank == failure_rank
                               ? static_cast<int>(std::min<std::size_t>(
                                     local_error.size(),
                                     static_cast<std::size_t>(
                                         std::numeric_limits<int>::max())))
                               : 0;
        MPI_Bcast(&message_size, 1, MPI_INT, failure_rank, communicator);
        std::string message(static_cast<std::size_t>(message_size), '\0');
        if (rank == failure_rank)
            message.assign(local_error.data(), static_cast<std::size_t>(message_size));
        if (message_size > 0) {
            MPI_Bcast(message.data(), message_size, MPI_CHAR, failure_rank,
                      communicator);
        }

        throw LinearSolverError(
            __FILE__, __LINE__,
            std::string(phase) + " failed collectively: " + message);
    }

    void zero_selected_sparse_corrections(Array<double>& correction,
                                          int column_count,
                                          std::ostream* diagnostic)
    {
        const std::vector<int> selected =
            env_int_list("SPARSE_ZERO_COLUMN_LIST");
        if (selected.empty())
            return;

        if (diagnostic != nullptr)
            *diagnostic << "Zeroing selected sparse Newton correction columns:";
        for (int col : selected) {
            if (col < 0 || col >= column_count)
                continue;
            correction.set(col) = 0.0;
            if (diagnostic != nullptr)
                *diagnostic << ' ' << col;
        }
        if (diagnostic != nullptr)
            *diagnostic << std::endl;
    }
#endif

    namespace
    {
#ifdef CELEPHAIS_USE_MUMPS
        std::uint64_t sparse_direct_capture_candidate_count = 0;

        std::uint64_t stable_capture_name_hash(const std::string& value)
        {
            // FNV-1a is deliberately fixed here rather than using std::hash,
            // whose result is not required to be stable across processes or
            // standard-library versions.
            std::uint64_t hash = UINT64_C(14695981039346656037);
            for (unsigned char byte : value) {
                hash ^= static_cast<std::uint64_t>(byte);
                hash *= UINT64_C(1099511628211);
            }
            return hash;
        }

        CapturedRowTaxonomy capture_row_taxonomy(RowTaxonomy taxonomy)
        {
            switch (taxonomy) {
                case RowTaxonomy::Vol:
                    return CapturedRowTaxonomy::Vol;
                case RowTaxonomy::TauBc:
                    return CapturedRowTaxonomy::TauBc;
                case RowTaxonomy::TauMatch:
                    return CapturedRowTaxonomy::TauMatch;
                case RowTaxonomy::GlobalInt:
                    return CapturedRowTaxonomy::GlobalInt;
                case RowTaxonomy::Unknown:
                    break;
            }
            throw std::runtime_error("direct replay capture encountered unknown row taxonomy");
        }

        CapturedColumnClass capture_column_class(ColumnClass column_class)
        {
            switch (column_class) {
                case ColumnClass::FieldUnknown:
                    return CapturedColumnClass::FieldUnknown;
                case ColumnClass::FieldInterior:
                    return CapturedColumnClass::FieldInterior;
                case ColumnClass::FieldBoundary:
                    return CapturedColumnClass::FieldBoundary;
                case ColumnClass::FieldInteriorVol:
                    return CapturedColumnClass::FieldInteriorVol;
                case ColumnClass::FieldBoundaryTau:
                    return CapturedColumnClass::FieldBoundaryTau;
                case ColumnClass::FieldOuterShellTau:
                    return CapturedColumnClass::FieldOuterShellTau;
                case ColumnClass::FieldMatching:
                    return CapturedColumnClass::FieldMatching;
                case ColumnClass::FieldGauge:
                    return CapturedColumnClass::FieldGauge;
                case ColumnClass::VarDomain:
                    return CapturedColumnClass::VarDomain;
                case ColumnClass::ScalarGlobal:
                    return CapturedColumnClass::ScalarGlobal;
                case ColumnClass::Unknown:
                    break;
            }
            throw std::runtime_error("direct replay capture encountered unknown column class");
        }

        template <typename Operation>
        void run_sparse_direct_collective_phase(
            MPI_Comm communicator, const char* phase, Operation&& operation)
        {
            std::string local_error;
            try {
                operation();
            } catch (const std::exception& error) {
                local_error = error.what();
            } catch (...) {
                local_error = "unknown non-standard exception";
            }
            sparse_direct_collective_throw_if_failed(
                communicator, phase, local_error);
        }

        double elapsed_time(std::chrono::time_point<std::chrono::system_clock> const& begin)
        {
            return std::chrono::duration<double>(std::chrono::system_clock::now() - begin).count();
        }

        long long current_resident_bytes()
        {
#if defined(__APPLE__)
            mach_task_basic_info_data_t info;
            mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
            if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                          reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
                return static_cast<long long>(info.resident_size);
#endif
            struct rusage usage;
            if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
                return static_cast<long long>(usage.ru_maxrss);
#else
                return static_cast<long long>(usage.ru_maxrss) * 1024LL;
#endif
            }
            return -1;
        }

        double max_resident_mb()
        {
            const long long resident_bytes = current_resident_bytes();
            const double local_mb = resident_bytes >= 0
                                        ? static_cast<double>(resident_bytes) /
                                              (1024.0 * 1024.0)
                                        : -1.0;
            double maximum_mb = -1.0;
            MPI_Reduce(&local_mb, &maximum_mb, 1, MPI_DOUBLE, MPI_MAX, 0,
                       MPI_COMM_WORLD);
            return maximum_mb;
        }

        void compact_coo_to_numerical_drop(AssembledJacobianCoo& coo,
                                           double numerical_drop_tol,
                                           int rank)
        {
            long long numerical_nnz = 0;
            if (rank == 0) {
                for (long long read = 0; read < coo.nnz; ++read) {
                    const std::size_t source = static_cast<std::size_t>(read);
                    if (std::abs(coo.a[source]) <= numerical_drop_tol)
                        continue;
                    const std::size_t target = static_cast<std::size_t>(numerical_nnz);
                    if (target != source) {
                        coo.irn[target] = coo.irn[source];
                        coo.jcn[target] = coo.jcn[source];
                        coo.a[target] = coo.a[source];
                    }
                    ++numerical_nnz;
                }
                coo.irn.resize(static_cast<std::size_t>(numerical_nnz));
                coo.jcn.resize(static_cast<std::size_t>(numerical_nnz));
                coo.a.resize(static_cast<std::size_t>(numerical_nnz));
            }
            MPI_Bcast(&numerical_nnz, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            coo.nnz = numerical_nnz;
            coo.drop_tol_used = numerical_drop_tol;
        }

        class RetainedParitySplitMumpsFactor final : public LinearSolver
        {
          public:
            struct Sector
            {
                std::vector<int> rows;
                std::vector<int> columns;
                std::unique_ptr<MumpsLinearSolver> solver;
                std::vector<double> solve_workspace;
            };

            RetainedParitySplitMumpsFactor(
                int dimension, std::array<Sector, 2> sectors)
                : dimension_(dimension), sectors_(std::move(sectors)),
                  rhs_copy_(static_cast<std::size_t>(dimension))
            {
            }

            void set_pattern(int, long long, const int*, const int*) override
            {
                throw LinearSolverError(
                    __FILE__, __LINE__,
                    "retained parity-split factor cannot replace its pattern");
            }

            void factor(const double*) override
            {
                throw LinearSolverError(
                    __FILE__, __LINE__,
                    "retained parity-split factor cannot be refactored");
            }

            void solve(double* rhs_inout) override
            {
                if (rhs_inout == nullptr || sectors_[0].solver == nullptr ||
                    sectors_[1].solver == nullptr) {
                    throw LinearSolverError(
                        __FILE__, __LINE__,
                        "retained parity-split factor is not available");
                }

                std::copy(rhs_inout, rhs_inout + dimension_, rhs_copy_.begin());
                for (Sector& sector : sectors_) {
                    for (std::size_t reduced = 0; reduced < sector.rows.size();
                         ++reduced) {
                        sector.solve_workspace[reduced] =
                            rhs_copy_[static_cast<std::size_t>(
                                sector.rows[reduced])];
                    }
                    sector.solver->solve(sector.solve_workspace.data());
                    for (std::size_t reduced = 0;
                         reduced < sector.columns.size(); ++reduced) {
                        rhs_inout[sector.columns[reduced]] =
                            sector.solve_workspace[reduced];
                    }
                }
            }

            void reset() override
            {
                for (Sector& sector : sectors_) {
                    sector.solver.reset();
                    std::vector<double>{}.swap(sector.solve_workspace);
                }
                std::vector<double>{}.swap(rhs_copy_);
            }

          private:
            int dimension_ = 0;
            std::array<Sector, 2> sectors_;
            std::vector<double> rhs_copy_;
        };

        void report_sparse_direct_memory_phase(
            const SparseDirectMumpsSolveOptions& options, const char* suffix)
        {
            if (options.memory_phase_prefix == nullptr)
                return;
            const std::string phase =
                std::string(options.memory_phase_prefix) + suffix;
            report_memory_mapper_phase(phase.c_str(), options.communicator);
        }

        SparseDirectMumpsParityLayout validate_physical_parity_blocks(
            const AssembledJacobianCoo& assembled_jacobian,
            const SparseDirectMumpsSolveOptions& options,
            int rank)
        {
            const int local_block_count = static_cast<int>(
                assembled_jacobian.parity_blocks.size());
            int minimum_block_count = 0;
            int maximum_block_count = 0;
            MPI_Allreduce(&local_block_count, &minimum_block_count, 1,
                          MPI_INT, MPI_MIN, options.communicator);
            MPI_Allreduce(&local_block_count, &maximum_block_count, 1,
                          MPI_INT, MPI_MAX, options.communicator);

            std::string local_error;
            try {
                if (minimum_block_count != maximum_block_count) {
                    throw std::runtime_error(
                        "physical parity block count disagrees across ranks");
                }
                if (local_block_count == 0)
                    return SparseDirectMumpsParityLayout::None;
                if (local_block_count != 1 && local_block_count != 2) {
                    throw std::runtime_error(
                        "physical parity COO must contain one or two blocks");
                }
                if (!assembled_jacobian.irn.empty() ||
                    !assembled_jacobian.jcn.empty() ||
                    !assembled_jacobian.a.empty()) {
                    throw std::runtime_error(
                        "physical parity COO cannot also contain a top-level payload");
                }

                long long aggregate_nnz = 0;
                int aggregate_dimension = 0;
                for (int block_index = 0;
                     block_index < local_block_count; ++block_index) {
                    const AssembledJacobianCooBlock& block =
                        assembled_jacobian.parity_blocks[
                            static_cast<std::size_t>(block_index)];
                    const int expected_label = block_index == 0 ? 1 : -1;
                    if (block.parity_label != expected_label) {
                        throw std::runtime_error(
                            "physical parity COO blocks must be ordered + then -");
                    }
                    if (block.n <= 0 || block.nnz < 0) {
                        throw std::runtime_error(
                            "physical parity COO block has an invalid size");
                    }
                    if (!block.selection_plan ||
                        block.selection_plan->selected_block() !=
                            expected_label ||
                        block.selection_plan->selected_rows().size() !=
                            static_cast<std::size_t>(block.n) ||
                        block.selection_plan->selected_columns().size() !=
                            static_cast<std::size_t>(block.n)) {
                        throw std::runtime_error(
                            "physical parity COO block has an incompatible selection plan");
                    }
                    const std::size_t expected_nnz =
                        static_cast<std::size_t>(block.nnz);
                    const bool payload_sizes_match =
                        block.irn.size() == expected_nnz &&
                        block.jcn.size() == expected_nnz &&
                        block.a.size() == expected_nnz;
                    if ((rank == 0 && !payload_sizes_match) ||
                        (rank != 0 && (!block.irn.empty() ||
                                       !block.jcn.empty() ||
                                       !block.a.empty()))) {
                        throw std::runtime_error(
                            "physical parity COO payload ownership or size is invalid");
                    }
                    if (rank == 0) {
                        for (std::size_t entry = 0; entry < expected_nnz;
                             ++entry) {
                            if (block.irn[entry] < 1 ||
                                block.irn[entry] > block.n ||
                                block.jcn[entry] < 1 ||
                                block.jcn[entry] > block.n) {
                                throw std::runtime_error(
                                    "physical parity COO index is outside its sector");
                            }
                        }
                    }
                    aggregate_nnz += block.nnz;
                    aggregate_dimension += block.n;
                }
                if (aggregate_nnz != assembled_jacobian.nnz) {
                    throw std::runtime_error(
                        "physical parity COO nnz does not match its aggregate");
                }
                const int expected_dimension =
                    local_block_count == 1
                        ? assembled_jacobian.parity_blocks.front().n
                        : aggregate_dimension;
                if (assembled_jacobian.n != expected_dimension) {
                    throw std::runtime_error(
                        "physical parity COO dimension does not match its blocks");
                }
            } catch (const std::exception& error) {
                local_error = error.what();
            }
            sparse_direct_collective_throw_if_failed(
                options.communicator, "physical parity COO validation",
                local_error);
            return local_block_count == 1
                       ? SparseDirectMumpsParityLayout::Plus
                       : SparseDirectMumpsParityLayout::PlusMinus;
        }

        SparseDirectMumpsSolveResult run_sparse_direct_parity_split(
            AssembledJacobianCoo& assembled_jacobian,
            double* rhs_inout,
            const JacobianParityMaskState* parity_state,
            const std::vector<
                std::shared_ptr<const JacobianSelectionPlan>>& sector_plans,
            int& icntl14,
            const SparseDirectMumpsSolveOptions& options,
            int rank)
        {
            if (options.split_factor_lifecycle ==
                    SparseDirectMumpsFactorLifecycle::Transient &&
                rhs_inout == nullptr) {
                throw LinearSolverError(
                    __FILE__, __LINE__,
                    "transient parity-split solve requires a right-hand side");
            }
            const bool physical_blocks =
                !assembled_jacobian.parity_blocks.empty();
            std::vector<int> row_in_sector;
            std::vector<int> column_in_sector;
            if (!physical_blocks) {
                if (parity_state == nullptr) {
                    throw LinearSolverError(
                        __FILE__, __LINE__,
                        "legacy parity split requires a parity state");
                }
                row_in_sector.assign(
                    static_cast<std::size_t>(assembled_jacobian.n), -1);
                column_in_sector.assign(
                    static_cast<std::size_t>(assembled_jacobian.n), -1);
                for (const auto& plan : sector_plans) {
                    const std::vector<int>& rows = plan->selected_rows();
                    const std::vector<int>& columns = plan->selected_columns();
                    for (std::size_t reduced = 0; reduced < rows.size();
                         ++reduced) {
                        row_in_sector[static_cast<std::size_t>(rows[reduced])] =
                            static_cast<int>(reduced);
                        column_in_sector[static_cast<std::size_t>(
                            columns[reduced])] = static_cast<int>(reduced);
                    }
                }
            }

            long long symmetric_nnz = 0;
            if (!physical_blocks) {
                run_sparse_direct_collective_phase(
                    options.communicator, "parity sector COO partition", [&]() {
                        if (rank != 0)
                            return;
                        long long front = 0;
                        long long back = assembled_jacobian.nnz;
                        while (front < back) {
                            const std::size_t head =
                                static_cast<std::size_t>(front);
                            const std::size_t row = static_cast<std::size_t>(
                                assembled_jacobian.irn[head] - 1);
                            const std::size_t column = static_cast<std::size_t>(
                                assembled_jacobian.jcn[head] - 1);
                            if (parity_state->row_sector[row] !=
                                parity_state->column_sector[column]) {
                                throw std::runtime_error(
                                    "masked Jacobian still holds a cross-sector "
                                    "entry");
                            }
                            if (parity_state->column_sector[column] > 0) {
                                ++front;
                                continue;
                            }
                            --back;
                            const std::size_t back_index =
                                static_cast<std::size_t>(back);
                            std::swap(assembled_jacobian.irn[head],
                                      assembled_jacobian.irn[back_index]);
                            std::swap(assembled_jacobian.jcn[head],
                                      assembled_jacobian.jcn[back_index]);
                            std::swap(assembled_jacobian.a[head],
                                      assembled_jacobian.a[back_index]);
                        }
                        symmetric_nnz = front;
                        for (long long entry = 0;
                             entry < assembled_jacobian.nnz; ++entry) {
                            const std::size_t index =
                                static_cast<std::size_t>(entry);
                            assembled_jacobian.irn[index] =
                                row_in_sector[static_cast<std::size_t>(
                                    assembled_jacobian.irn[index] - 1)] + 1;
                            assembled_jacobian.jcn[index] =
                                column_in_sector[static_cast<std::size_t>(
                                    assembled_jacobian.jcn[index] - 1)] + 1;
                        }
                    });
                MPI_Bcast(&symmetric_nnz, 1, MPI_LONG_LONG, 0,
                          options.communicator);
            }

            std::vector<double> original_rhs;
            if (rhs_inout != nullptr) {
                original_rhs.assign(
                    rhs_inout, rhs_inout + assembled_jacobian.n);
            }
            std::array<RetainedParitySplitMumpsFactor::Sector, 2>
                retained_sectors;
            SparseDirectMumpsSolveResult result;
            result.parity_split = true;
            result.parity_layout =
                SparseDirectMumpsParityLayout::PlusMinus;
            long long block_offset = 0;
            for (std::size_t block = 0; block < sector_plans.size(); ++block) {
                const JacobianSelectionPlan& plan = *sector_plans[block];
                AssembledJacobianCooBlock* physical_block =
                    physical_blocks
                        ? &assembled_jacobian.parity_blocks[block]
                        : nullptr;
                const int sector_dimension =
                    physical_block != nullptr
                        ? physical_block->n
                        : static_cast<int>(
                              plan.selected_columns().size());
                const long long sector_nnz = physical_block != nullptr
                    ? physical_block->nnz
                    : (block == 0 ? symmetric_nnz
                                  : assembled_jacobian.nnz - symmetric_nnz);
                const int* block_irn = physical_block != nullptr
                    ? (physical_block->irn.empty()
                           ? nullptr
                           : physical_block->irn.data())
                    : (assembled_jacobian.irn.empty()
                           ? nullptr
                           : assembled_jacobian.irn.data() + block_offset);
                const int* block_jcn = physical_block != nullptr
                    ? (physical_block->jcn.empty()
                           ? nullptr
                           : physical_block->jcn.data())
                    : (assembled_jacobian.jcn.empty()
                           ? nullptr
                           : assembled_jacobian.jcn.data() + block_offset);
                const double* block_a = physical_block != nullptr
                    ? (physical_block->a.empty()
                           ? nullptr
                           : physical_block->a.data())
                    : (assembled_jacobian.a.empty()
                           ? nullptr
                           : assembled_jacobian.a.data() + block_offset);
                const int parity = physical_block != nullptr
                    ? physical_block->parity_label
                    : plan.selected_block();
                const char* parity_label = parity > 0 ? "+" : "-";

                std::unique_ptr<MumpsLinearSolver> sector_solver;
                run_sparse_direct_collective_phase(
                    options.communicator, "MUMPS construction/setup", [&]() {
                        sector_solver = std::make_unique<MumpsLinearSolver>(
                            sector_dimension, options.ordering,
                            options.out_of_core_mode, options.blr, icntl14,
                            options.communicator, options.ranks_per_node,
                            false, options.out_of_core_touch,
                            options.out_of_core_safety,
                            options.out_of_core_budget_mb);
                        sector_solver->set_auto_out_of_core_diagnostic(
                            nullptr, "");
                        sector_solver->set_pattern(sector_dimension, sector_nnz,
                                                   block_irn, block_jcn);
                    });
                if (rank == 0 && options.diagnostic != nullptr) {
                    *options.diagnostic << "[MUMPS: Parity " << parity_label
                                        << "]" << std::endl;
                }
                const auto analyze_start =
                    std::chrono::system_clock::now();
                run_sparse_direct_collective_phase(
                    options.communicator, "MUMPS analysis",
                    [&]() { sector_solver->analyze_pattern(); });
                const double sector_analyze_seconds =
                    elapsed_time(analyze_start);
                result.analyze_seconds += sector_analyze_seconds;
                report_sparse_direct_memory_phase(options, ".post_analyze");

                const auto factor_start =
                    std::chrono::system_clock::now();
                run_sparse_direct_collective_phase(
                    options.communicator, "MUMPS factorization",
                    [&]() { sector_solver->factor_analyzed(block_a); });
                const double sector_factor_seconds = elapsed_time(factor_start);
                result.factorize_seconds += sector_factor_seconds;
                report_sparse_direct_mumps_factorization(
                    options.communicator, options.diagnostic,
                    "analyze+factorize", sector_analyze_seconds,
                    sector_factor_seconds,
                    mumps_ordering_name(
                        sector_solver->last_actual_ordering()),
                    sector_solver->out_of_core_enabled(),
                    sector_solver->factor_memory_mb(),
                    sector_solver->factor_allocated_memory_mb());
                sector_solver->release_centralized_coo_input();
                if (physical_block != nullptr) {
                    AssembledJacobianCooBlock::IndexVector{}.swap(
                        physical_block->irn);
                    AssembledJacobianCooBlock::IndexVector{}.swap(
                        physical_block->jcn);
                    AssembledJacobianCooBlock::ValueVector{}.swap(
                        physical_block->a);
                }
                report_sparse_direct_memory_phase(
                    options, ".post_factor_pre_coo_release");

                if (rank == 0) {
                    icntl14 = sector_solver->last_icntl14();
                }
                MPI_Bcast(&icntl14, 1, MPI_INT, 0, options.communicator);

                if (options.split_factor_lifecycle ==
                    SparseDirectMumpsFactorLifecycle::Retained) {
                    RetainedParitySplitMumpsFactor::Sector& retained =
                        retained_sectors[block];
                    retained.rows = plan.selected_rows();
                    retained.columns = plan.selected_columns();
                    retained.solve_workspace.resize(
                        static_cast<std::size_t>(sector_dimension));
                    retained.solver = std::move(sector_solver);
                } else {
                    std::vector<double> sector_rhs(
                        static_cast<std::size_t>(sector_dimension));
                    for (std::size_t reduced = 0;
                         reduced < plan.selected_rows().size(); ++reduced) {
                        sector_rhs[reduced] = original_rhs[
                            static_cast<std::size_t>(
                                plan.selected_rows()[reduced])];
                    }
                    const auto solve_start =
                        std::chrono::system_clock::now();
                    sector_solver->solve(sector_rhs.data());
                    result.solve_seconds += elapsed_time(solve_start);
                    for (std::size_t reduced = 0;
                         reduced < plan.selected_columns().size(); ++reduced) {
                        rhs_inout[plan.selected_columns()[reduced]] =
                            sector_rhs[reduced];
                    }
                }
                block_offset += sector_nnz;
            }

            AssembledJacobianCoo::IndexVector{}.swap(
                assembled_jacobian.irn);
            AssembledJacobianCoo::IndexVector{}.swap(
                assembled_jacobian.jcn);
            AssembledJacobianCoo::ValueVector{}.swap(assembled_jacobian.a);
            assembled_jacobian.parity_blocks.clear();
            report_sparse_direct_memory_phase(options, ".post_coo_release");

            if (options.split_factor_lifecycle ==
                SparseDirectMumpsFactorLifecycle::Retained) {
                auto retained_factor =
                    std::make_unique<RetainedParitySplitMumpsFactor>(
                        assembled_jacobian.n, std::move(retained_sectors));
                if (rhs_inout != nullptr) {
                    const auto solve_start =
                        std::chrono::system_clock::now();
                    retained_factor->solve(rhs_inout);
                    result.solve_seconds += elapsed_time(solve_start);
                }
                result.retained_factor = std::move(retained_factor);
            }
            if (rhs_inout != nullptr) {
                report_sparse_direct_mumps_apply_timing(
                    options.communicator, options.diagnostic,
                    result.solve_seconds);
            }
            return result;
        }

        SparseDirectMumpsSolveResult run_sparse_direct_ordinary_mumps(
            AssembledJacobianCoo& assembled_jacobian,
            AssembledJacobianCooBlock* parity_block,
            double* rhs_inout,
            int& icntl14,
            const SparseDirectMumpsSolveOptions& options,
            int rank)
        {
            if (options.ordinary_factor_lifecycle ==
                    SparseDirectMumpsFactorLifecycle::Transient &&
                rhs_inout == nullptr) {
                throw LinearSolverError(
                    __FILE__, __LINE__,
                    "transient ordinary MUMPS solve requires a right-hand side");
            }

            SparseDirectMumpsSolveResult result;
            if (parity_block != nullptr) {
                result.parity_layout =
                    SparseDirectMumpsParityLayout::Plus;
            }
            std::unique_ptr<MumpsLinearSolver> solver;
            const bool grouped_parity_diagnostic = parity_block != nullptr;
            const bool measure_factor_phases = true;
            const int factor_dimension = parity_block != nullptr
                ? parity_block->n
                : assembled_jacobian.n;
            const long long factor_nnz = parity_block != nullptr
                ? parity_block->nnz
                : assembled_jacobian.nnz;
            const int* factor_irn = parity_block != nullptr
                ? (parity_block->irn.empty() ? nullptr
                                             : parity_block->irn.data())
                : (assembled_jacobian.irn.empty()
                       ? nullptr
                       : assembled_jacobian.irn.data());
            const int* factor_jcn = parity_block != nullptr
                ? (parity_block->jcn.empty() ? nullptr
                                             : parity_block->jcn.data())
                : (assembled_jacobian.jcn.empty()
                       ? nullptr
                       : assembled_jacobian.jcn.data());
            const double* factor_values = parity_block != nullptr
                ? (parity_block->a.empty() ? nullptr
                                           : parity_block->a.data())
                : (assembled_jacobian.a.empty()
                       ? nullptr
                       : assembled_jacobian.a.data());
            if (rank == 0 && options.diagnostic != nullptr &&
                grouped_parity_diagnostic) {
                *options.diagnostic << "[MUMPS: Parity +]" << std::endl;
            }
            const auto build_attempt = [&](const SparseDirectMumpsReplayOptions*
                                               replay) {
                const char* construction_phase =
                    replay == nullptr
                        ? "MUMPS construction/setup"
                        : "MUMPS tree cache preconditioner initialization";
                run_sparse_direct_collective_phase(
                    options.communicator, construction_phase, [&]() {
                        solver = std::make_unique<MumpsLinearSolver>(
                            factor_dimension,
                            replay == nullptr ? options.ordering
                                              : replay->ordering,
                            options.out_of_core_mode, options.blr, icntl14,
                            options.communicator, options.ranks_per_node, false,
                            options.out_of_core_touch,
                            options.out_of_core_safety,
                            options.out_of_core_budget_mb);
                        solver->set_auto_out_of_core_diagnostic(nullptr, "");
                        if (replay != nullptr) {
                            if (replay->column_indices_1based == nullptr ||
                                replay->symmetric_permutation_1based == nullptr ||
                                replay->solution_column_permutation_1based ==
                                    nullptr) {
                                throw std::runtime_error(
                                    "MUMPS replay metadata is incomplete");
                            }
                            solver->set_pattern(
                                factor_dimension, factor_nnz,
                                factor_irn,
                                replay->column_indices_1based->data());
                            solver->set_user_permutation_1based(
                                *replay->symmetric_permutation_1based);
                            solver->set_solution_column_permutation_1based(
                                *replay
                                     ->solution_column_permutation_1based);
                            if (replay->detect_null_pivots) {
                                solver->enable_null_pivot_detection(
                                    true, replay->null_pivot_threshold);
                            }
                        }
                    });
                if (replay == nullptr) {
                    // Preserve the direct path's existing failure boundary:
                    // ordinary set_pattern is local, while construction and
                    // every MUMPS phase are promoted collectively.
                    solver->set_pattern(
                        factor_dimension, factor_nnz,
                        factor_irn, factor_jcn);
                }

                const auto analyze_start = measure_factor_phases
                                               ? std::chrono::system_clock::now()
                                               : std::chrono::time_point<
                                                     std::chrono::system_clock>{};
                run_sparse_direct_collective_phase(
                    options.communicator,
                    replay == nullptr ? "MUMPS analysis"
                                      : "MUMPS tree cache analysis",
                    [&]() { solver->analyze_pattern(); });
                if (measure_factor_phases)
                    result.analyze_seconds += elapsed_time(analyze_start);

                if (parity_block == nullptr && replay == nullptr &&
                    options.ordinary_analysis_observer) {
                    run_sparse_direct_collective_phase(
                        options.communicator,
                        "MUMPS ordinary analysis observer", [&]() {
                            if (rank != 0)
                                return;
                            SparseDirectMumpsAnalysisSnapshot snapshot;
                            solver->copy_column_permutation_1based(
                                snapshot.column_permutation_1based,
                                snapshot.matching_applied);
                            solver->copy_symmetric_permutation_1based(
                                snapshot.symmetric_permutation_1based);
                            options.ordinary_analysis_observer(snapshot);
                        });
                }
                report_sparse_direct_memory_phase(options, ".post_analyze");

                const auto factor_start = measure_factor_phases
                                              ? std::chrono::system_clock::now()
                                              : std::chrono::time_point<
                                                    std::chrono::system_clock>{};
                run_sparse_direct_collective_phase(
                    options.communicator,
                    replay == nullptr ? "MUMPS factorization"
                                      : "MUMPS tree cache factorization",
                    [&]() {
                        solver->factor_analyzed(factor_values);
                    });
                if (measure_factor_phases)
                    result.factorize_seconds += elapsed_time(factor_start);
                report_sparse_direct_memory_phase(
                    options, ".post_factor_pre_coo_release");

                if (replay != nullptr && replay->detect_null_pivots) {
                    int null_pivot_count =
                        rank == 0 ? solver->last_null_pivot_count() : 0;
                    MPI_Bcast(&null_pivot_count, 1, MPI_INT, 0,
                              options.communicator);
                    if (null_pivot_count > 0) {
                        throw LinearSolverError(
                            __FILE__, __LINE__,
                            "MUMPS tree cache factorization reported " +
                                std::to_string(null_pivot_count) +
                                " null pivot(s)");
                    }
                }
            };

            if (parity_block == nullptr &&
                options.ordinary_replay != nullptr) {
                result.replay_attempted = true;
                try {
                    build_attempt(options.ordinary_replay);
                    result.replay_succeeded = true;
                    if (options.ordinary_replay_success_observer)
                        options.ordinary_replay_success_observer();
                } catch (const std::exception& error) {
                    result.replay_failure_reason = error.what();
                    solver.reset();
                    if (options.ordinary_replay_failure_observer) {
                        options.ordinary_replay_failure_observer(
                            result.replay_failure_reason);
                    }
                }
            }
            if (!result.replay_succeeded)
                build_attempt(nullptr);

            solver->release_centralized_coo_input();
            AssembledJacobianCoo::IndexVector{}.swap(
                assembled_jacobian.irn);
            AssembledJacobianCoo::IndexVector{}.swap(
                assembled_jacobian.jcn);
            AssembledJacobianCoo::ValueVector{}.swap(assembled_jacobian.a);
            if (parity_block != nullptr) {
                AssembledJacobianCooBlock::IndexVector{}.swap(
                    parity_block->irn);
                AssembledJacobianCooBlock::IndexVector{}.swap(
                    parity_block->jcn);
                AssembledJacobianCooBlock::ValueVector{}.swap(
                    parity_block->a);
                assembled_jacobian.parity_blocks.clear();
            }
            report_sparse_direct_memory_phase(options, ".post_coo_release");

            if (rank == 0) {
                icntl14 = solver->last_icntl14();
            }
            MPI_Bcast(&icntl14, 1, MPI_INT, 0, options.communicator);
            report_sparse_direct_mumps_factorization(
                options.communicator, options.diagnostic,
                grouped_parity_diagnostic ? "analyze+factorize"
                                          : "MUMPS analyze+factorize",
                result.analyze_seconds, result.factorize_seconds,
                mumps_ordering_name(solver->last_actual_ordering()),
                solver->out_of_core_enabled(), solver->factor_memory_mb(),
                solver->factor_allocated_memory_mb());

            if (rhs_inout != nullptr) {
                const bool report_apply =
                    options.report_apply_timing || grouped_parity_diagnostic;
                const bool measure_solve =
                    options.measure_phases || report_apply;
                const auto solve_start = measure_solve
                                             ? std::chrono::system_clock::now()
                                             : std::chrono::time_point<
                                                   std::chrono::system_clock>{};
                solver->solve(rhs_inout);
                if (measure_solve)
                    result.solve_seconds += elapsed_time(solve_start);
                if (report_apply) {
                    report_sparse_direct_mumps_apply_timing(
                        options.communicator, options.diagnostic,
                        result.solve_seconds);
                }
            }
            if (options.ordinary_factor_lifecycle ==
                SparseDirectMumpsFactorLifecycle::Retained) {
                result.retained_factor = std::move(solver);
            }
            return result;
        }
#endif
    } // namespace

#ifdef CELEPHAIS_USE_MUMPS
    bool sparse_direct_mumps_split_candidate(
        const AssembledJacobianCoo& assembled_jacobian,
        const JacobianParityMaskState* parity_state,
        bool parity_split_requested) noexcept
    {
        return !assembled_jacobian.parity_blocks.empty() ||
               (parity_split_requested &&
               assembled_jacobian.parity_sector_block_diagonal &&
               parity_state != nullptr &&
               parity_state->n == assembled_jacobian.n &&
               parity_state->row_sector.size() ==
                   static_cast<std::size_t>(assembled_jacobian.n) &&
               parity_state->column_sector.size() ==
                   static_cast<std::size_t>(assembled_jacobian.n));
    }

    SparseDirectMumpsSolveResult run_sparse_direct_mumps_solve(
        AssembledJacobianCoo& assembled_jacobian,
        double* rhs_inout,
        const JacobianParityMaskState* parity_state,
        int& icntl14,
        const SparseDirectMumpsSolveOptions& options)
    {
        int rank = 0;
        MPI_Comm_rank(options.communicator, &rank);

        const SparseDirectMumpsParityLayout physical_layout =
            validate_physical_parity_blocks(
                assembled_jacobian, options, rank);

        std::vector<std::shared_ptr<const JacobianSelectionPlan>> sector_plans;
        std::string split_refusal_reason;
        bool split_sector_solve =
            physical_layout == SparseDirectMumpsParityLayout::PlusMinus ||
            (physical_layout == SparseDirectMumpsParityLayout::None &&
             sparse_direct_mumps_split_candidate(
                 assembled_jacobian, parity_state,
                 options.parity_split_requested));
        if (physical_layout == SparseDirectMumpsParityLayout::PlusMinus) {
            for (const AssembledJacobianCooBlock& block :
                 assembled_jacobian.parity_blocks) {
                sector_plans.push_back(block.selection_plan);
            }
        } else if (split_sector_solve) {
            const std::vector<JacobianSelectionPlan::BlockLabel> row_labels(
                parity_state->row_sector.begin(),
                parity_state->row_sector.end());
            const std::vector<JacobianSelectionPlan::BlockLabel> column_labels(
                parity_state->column_sector.begin(),
                parity_state->column_sector.end());
            for (const JacobianSelectionPlan::BlockLabel sector : {1, -1}) {
                JacobianSelectionPlanBuild build =
                    make_jacobian_selection_plan(row_labels, column_labels,
                                                 sector, -sector);
                if (!build.plan) {
                    split_sector_solve = false;
                    split_refusal_reason = build.fallback_reason;
                    break;
                }
                sector_plans.push_back(std::move(build.plan));
            }
        }
        {
            const int split_local = split_sector_solve ? 1 : 0;
            int split_all = 0;
            MPI_Allreduce(&split_local, &split_all, 1, MPI_INT, MPI_MIN,
                          options.communicator);
            if (split_all == 0 && split_sector_solve) {
                split_sector_solve = false;
                split_refusal_reason = "collective disagreement";
            }
        }
        if (!split_refusal_reason.empty() && rank == 0 &&
            options.diagnostic != nullptr) {
            *options.diagnostic
                << "Jacobian parity split solve: refused, "
                << split_refusal_reason
                << "; solving both sector blocks together\n";
        }

        if (split_sector_solve) {
            return run_sparse_direct_parity_split(
                assembled_jacobian, rhs_inout, parity_state, sector_plans,
                icntl14, options, rank);
        }
        AssembledJacobianCooBlock* plus_block =
            physical_layout == SparseDirectMumpsParityLayout::Plus
                ? &assembled_jacobian.parity_blocks.front()
                : nullptr;
        return run_sparse_direct_ordinary_mumps(
            assembled_jacobian, plus_block, rhs_inout, icntl14, options, rank);
    }
#endif

    bool System_of_eqs::do_newton_sparse(double precision, double& error, double user_drop_tol)
    {
        SolverRuntimeConfig config = solver_runtime_config;
        if (std::isfinite(user_drop_tol) && user_drop_tol > 0.0) {
            config.mumps.drop_tol = user_drop_tol;
        }
        return do_newton_sparse(precision, error, config);
    }

    bool System_of_eqs::do_newton_sparse(double precision, double& error, const SolverRuntimeConfig& config)
    {
        set_solver_runtime_config(config);
        last_solved_system_dof() = nbr_unknowns;  // stamped into datasets saved after this solve
#ifdef CELEPHAIS_USE_MUMPS
        int rank = 0;
        int nproc = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        const bool timing_enabled = config.diagnostics.timing;
        auto& mumps_state = mumps_runtime_state;
        const bool analyze_reuse_requested = config.mumps.sparse_analyze_reuse;
        const bool direct_replay_capture_requested = !config.diagnostics.direct_replay_capture_path.empty();
        validate_captured_linear_system_collective_request(
            MPI_COMM_WORLD, config.diagnostics.direct_replay_capture_path,
            config.diagnostics.direct_replay_capture_ordinal, analyze_reuse_requested);
        auto release_sparse_direct_chord_factor = [&]() {
            mumps_state.sparse_direct_solver.reset();
            mumps_state.sparse_direct_chord_factor_retained = false;
            mumps_state.sparse_direct_consecutive_chord_steps = 0;
            mumps_state.sparse_direct_selection_plan.reset();
            mumps_state.sparse_direct_factor_dimension = 0;
        };
        auto clear_sparse_direct_reuse_state = [&]() {
            mumps_state.sparse_direct_solver.reset();
            mumps_state.sparse_direct_chord_factor_retained = false;
            mumps_state.sparse_direct_consecutive_chord_steps = 0;
            mumps_state.sparse_direct_selection_plan.reset();
            mumps_state.sparse_direct_factor_dimension = 0;
            std::vector<int>{}.swap(mumps_state.sparse_direct_pattern_irn);
            std::vector<int>{}.swap(mumps_state.sparse_direct_pattern_jcn);
            std::vector<long long>{}.swap(
                mumps_state.sparse_direct_pattern_column_offsets);
            std::vector<double>{}.swap(
                mumps_state.sparse_direct_aligned_values);
            mumps_state.sparse_direct_dimension = 0;
            mumps_state.sparse_direct_ordering = -1;
            mumps_state.sparse_direct_blr = -1;
            mumps_state.sparse_direct_out_of_core_mode =
                MumpsOutOfCoreMode::Off;
            mumps_state.sparse_direct_out_of_core_touch =
                kMumpsOutOfCoreTouchDefault;
            mumps_state.sparse_direct_out_of_core_safety =
                kMumpsOutOfCoreSafetyDefault;
            mumps_state.sparse_direct_out_of_core_budget_mb =
                kMumpsOutOfCoreBudgetUnset;
            mumps_state.sparse_direct_ranks_per_node = -1;
            mumps_state.sparse_direct_pattern_drop_tol = -1.0;
            mumps_state.sparse_direct_analyze_count = 0;
            mumps_state.sparse_direct_reuse_count = 0;
        };
        auto read_inactive_state_drift = [&](const JacobianSelectionPlan& plan,
                                             bool install_baseline,
                                             double& drift_linf,
                                             std::string& failure_reason) {
            const std::shared_ptr<JacobianParityMaskState>& parity_state =
                jacobian_parity_mask_state();
            if (!parity_state) {
                failure_reason = "parity state is absent";
                return false;
            }
            if (!install_baseline &&
                !parity_state->inactive_state_baseline_installed) {
                failure_reason = "inactive state baseline is not installed";
                return false;
            }
            std::vector<double> current;
            if (!read_inactive_jacobian_state(plan, current, failure_reason))
                return false;
            drift_linf = install_or_measure_jacobian_inactive_state_drift(
                current, parity_state->inactive_state_baseline,
                parity_state->inactive_state_baseline_installed);
            return true;
        };
        auto check_forbidden_residual =
            [&](const JacobianSelectionNorms& norms) {
                const std::shared_ptr<JacobianParityMaskState>& parity_state =
                    jacobian_parity_mask_state();
                if (!parity_state)
                    return JacobianForbiddenResidualCheck{};
                return check_jacobian_forbidden_residual(
                    norms, parity_state->forbidden_baseline,
                    parity_state->forbidden_baseline_installed);
            };

        // Reuse the residual evaluated after the previous direct-sparse trial
        // when no unknown-state mutation invalidated it in between. The shared
        // consumer falls back to the all-rank partitioned/fresh evaluation and
        // provides the opt-in collective byte self-test.
        Array<double> residual =
            take_forwarded_residual_or_compute("direct-sparse trial");
        error = 0.0;
        for (std::size_t i = 0; i < residual.get_nbr(); ++i)
            error = std::max(error, std::abs(residual(static_cast<int>(i))));
        print_error_init_diagnostic(residual, error);

        if (!mumps_state.sparse_direct_chord_entry_checked) {
            mumps_state.sparse_direct_chord_entry_checked = true;
            if (!sparse_chord_entry_allowed(error)) {
                mumps_state.sparse_direct_chord_disabled_for_solve = true;
                if (rank == 0 && config.mumps.sparse_chord_reuse) {
                    std::cout << "Sparse chord reuse: disabled for this solve at "
                              "initial error="
                              << error << " entry_tol="
                              << kSparseChordEntryTolerance << '\n';
                    mumps_state.sparse_direct_chord_exclusion_printed = true;
                }
            }
        }

        if (error < precision &&
            !mumps_state.sparse_direct_masked_full_rebuild_pending) {
            // Never carry a live factorization past convergence: its JOB_END
            // must not run after MPI_Finalize.
            if (mumps_state.sparse_direct_chord_factor_retained) {
                release_sparse_direct_chord_factor();
            }
            // Do not clear retained state on local convergence: the caller's
            // MPI consensus may find another rank unconverged and take another
            // step. Independent loops reset collectively at iteration one via
            // newton_step_with_consensus(..., reset_runtime_state=true).
            return true;
        }

        const int row_count = static_cast<int>(residual.get_nbr());
        const int column_count = nbr_unknowns;
        if (row_count != column_count) {
            if (rank == 0)
                std::cerr << "do_newton_sparse: non-square system m=" << row_count
                          << " n=" << column_count << "; falling back to do_newton()." << std::endl;
            reset_solver_runtime_state();
            return do_newton(precision, error, System_of_eqs::SOLVER::NEWTON_RAPHSON);
        }

        const bool chord_reuse_requested = config.mumps.sparse_chord_reuse;

        // Structural descriptors and column grading are sufficient to select a
        // square parity block before J1.  Keep the parity-mass diagnostic on its
        // full-J path: it is the opt-in oracle that compares those structural
        // row labels with matrix-derived grading, and assumes a full COO.
        const char* local_parity_mass_path =
            std::getenv("JACOBIAN_PARITY_MASS");
        const int local_parity_mass_probe_requested =
            local_parity_mass_path != nullptr &&
                local_parity_mass_path[0] != '\0' &&
                std::string(local_parity_mass_path) != "0"
            ? 1
            : 0;
        int parity_mass_probe_requested_any = 0;
        MPI_Allreduce(
            &local_parity_mass_probe_requested,
            &parity_mass_probe_requested_any, 1, MPI_INT, MPI_MAX,
            MPI_COMM_WORLD);
        const bool parity_mass_probe_requested =
            parity_mass_probe_requested_any != 0;
        const int local_pre_j1_requested =
            config.sparse_sector_reduce && !jacobian_parity_mask_state() &&
                !parity_mass_probe_requested
            ? 1
            : 0;
        int pre_j1_requested_all = 0;
        MPI_Allreduce(&local_pre_j1_requested, &pre_j1_requested_all, 1,
                      MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (pre_j1_requested_all != 0) {
            JacobianParityRowPrediction row_prediction =
                predict_jacobian_parity_rows(*this);
            JacobianParityColumnGrading column_grading =
                grade_jacobian_parity_columns(*this);
            JacobianPreJ1SelectionPlanBuild pre_j1 =
                make_jacobian_pre_j1_selection_plan(
                    row_prediction, column_grading, residual.get_data(),
                    row_count);

            const int local_eligible = pre_j1 ? 1 : 0;
            int eligible_all = 0;
            MPI_Allreduce(&local_eligible, &eligible_all, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD);
            if (eligible_all != 0) {
                auto state = std::make_shared<JacobianParityMaskState>();
                state->n = column_count;
                state->column_sector = std::move(column_grading.sector);
                state->row_sector = std::move(row_prediction.sector);
                state->selection_plan = std::move(pre_j1.plan);
                state->decision = JacobianParityMaskState::Decision::Engaged;
                state->structural_labels_available = true;
                state->reduction_decision =
                    JacobianParityMaskState::ReductionDecision::Eligible;
                jacobian_parity_mask_state() = std::move(state);
            }
        } else if (config.sparse_sector_reduce &&
                   !jacobian_parity_mask_state() &&
                   parity_mass_probe_requested && rank == 0) {
            std::cout
                << "Jacobian sector reduction: pre-J1 diagnostic full-J "
                   "cross-check requested by JACOBIAN_PARITY_MASS\n";
        }

        // Freeze the selection role at step entry.  The structural path has
        // already installed its plan, so J1 has the selected role.  A failed
        // pre-J1 gate follows the phase-1 full-J path; a plan certified during
        // that assembly becomes the next step's role and its full factor is not
        // retained for chord reuse.
        std::shared_ptr<const JacobianSelectionPlan> step_selection_plan;
        if (config.sparse_sector_reduce) {
            const std::shared_ptr<JacobianParityMaskState>& parity_state =
                jacobian_parity_mask_state();
            if (parity_state && parity_state->n == column_count &&
                parity_state->reduction_decision ==
                    JacobianParityMaskState::ReductionDecision::Eligible) {
                step_selection_plan = parity_state->selection_plan;
            }
        }
        if (mumps_state.sparse_direct_masked_full_rebuild_pending &&
            step_selection_plan) {
            KADATH_THROW(
                "pending masked-full rebuild still has an active selection plan");
        }

        auto validate_reduced_step_entry = [&]() {
            if (step_selection_plan) {
                // Structural reduction installs the inactive-state baseline at
                // J1 entry.  Phase-1 fallback installs it when a full-J plan
                // first becomes active on a later Jacobian.
                double inactive_state_drift = 0.0;
                std::string inactive_state_failure;
                const bool local_state_valid = read_inactive_state_drift(
                    *step_selection_plan, true, inactive_state_drift,
                    inactive_state_failure);
                int state_valid = local_state_valid ? 1 : 0;
                int state_valid_all = 0;
                if (!local_state_valid) {
                    inactive_state_drift =
                        std::numeric_limits<double>::infinity();
                }
                double inactive_state_drift_max = inactive_state_drift;
                MPI_Allreduce(
                    &state_valid, &state_valid_all, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);
                MPI_Allreduce(
                    &inactive_state_drift, &inactive_state_drift_max, 1,
                    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                if (rank == 0 && timing_enabled) {
                    std::cout
                        << "Jacobian sector reduction: inactive state drift Linf="
                        << inactive_state_drift_max << " (limit=1e-14)\n";
                }
                if (state_valid_all == 0 ||
                    !jacobian_inactive_state_drift_allowed(
                        inactive_state_drift_max)) {
                    const std::shared_ptr<JacobianParityMaskState>& parity_state =
                        jacobian_parity_mask_state();
                    if (parity_state) {
                        abandon_jacobian_parity_reduction(
                            *parity_state,
                            state_valid_all == 0
                                ? "inactive state drift check is unsupported: " +
                                      inactive_state_failure
                                : "inactive state drift exceeds 1e-14",
                            rank);
                    }
                    clear_sparse_direct_reuse_state();
                    step_selection_plan.reset();
                }
            }
        };
        validate_reduced_step_entry();
        int linear_dimension = step_selection_plan
            ? static_cast<int>(
                  step_selection_plan->selected_columns().size())
            : column_count;

        const SparseDirectDropPolicy drop_policy =
            resolve_sparse_direct_drop_policy(
                config.mumps, error, column_count,
                mumps_state.sparse_direct_drop_state);
        const double drop_tol = drop_policy.numerical_drop_tol;

        const int mumps_ordering = config.mumps.ordering;
        const MumpsOutOfCoreMode out_of_core_mode = config.mumps.out_of_core;
        const double out_of_core_touch = config.mumps.out_of_core_touch;
        const double out_of_core_safety = config.mumps.out_of_core_safety;
        const double out_of_core_budget_mb =
            config.mumps.out_of_core_budget_mb;
        const int blr_icntl35 = config.mumps.blr;
        const int mumps_ranks_per_node =
            resolve_sparse_direct_mumps_ranks_per_node(
                config.mumps, MPI_COMM_WORLD);
        const double pattern_drop_tol = drop_policy.pattern_drop_tol;

        if (chord_reuse_requested && analyze_reuse_requested) {
            mumps_state.sparse_direct_chord_disabled_for_solve = true;
            if (rank == 0 &&
                !mumps_state.sparse_direct_chord_exclusion_printed) {
                std::cout << "Sparse chord reuse: disabled for this solve because "
                             "MUMPS analyze reuse is requested\n";
                mumps_state.sparse_direct_chord_exclusion_printed = true;
            }
        }
        const bool chord_reuse_active =
            chord_reuse_requested && !analyze_reuse_requested &&
            !mumps_state.sparse_direct_chord_disabled_for_solve;
        auto abandon_reduced_step = [&](const std::string& reason) {
            const std::shared_ptr<JacobianParityMaskState>& parity_state =
                jacobian_parity_mask_state();
            if (parity_state)
                abandon_jacobian_parity_reduction(*parity_state, reason, rank);
            mumps_state.sparse_direct_masked_full_rebuild_pending = true;
            clear_sparse_direct_reuse_state();
        };
        auto restore_rejected_reduced_step =
            [&](const State_snapshot& snapshot,
                Array<double>& restored_residual) {
                restore_state(snapshot);
                // The rejected trial left the equation-term, metric and
                // definition caches at its state.  Re-evaluate after the exact
                // restore so the pending masked-full rebuild starts from the
                // restored entry point with matching caches and residual.
                restored_residual = sec_member_partitioned();
                error = 0.0;
                for (std::size_t i = 0; i < restored_residual.get_nbr(); ++i) {
                    const double magnitude = std::abs(
                        restored_residual(static_cast<int>(i)));
                    if (!std::isfinite(magnitude)) {
                        error = std::numeric_limits<double>::infinity();
                        break;
                    }
                    error = std::max(error, magnitude);
                }
            };

        // A retained chord factor belongs to one exact immutable selection
        // role and OOC policy. End it before use when either changes.
        const bool retained_chord_ooc_policy_matches =
            mumps_state.sparse_direct_out_of_core_mode == out_of_core_mode &&
            mumps_state.sparse_direct_out_of_core_touch ==
                out_of_core_touch &&
            mumps_state.sparse_direct_out_of_core_safety ==
                out_of_core_safety &&
            mumps_state.sparse_direct_out_of_core_budget_mb ==
                out_of_core_budget_mb;
        if (mumps_state.sparse_direct_chord_factor_retained &&
            (!chord_reuse_active ||
             !retained_chord_ooc_policy_matches ||
             !jacobian_selection_factor_compatible(
                 mumps_state.sparse_direct_selection_plan,
                 mumps_state.sparse_direct_factor_dimension,
                 step_selection_plan, linear_dimension))) {
            release_sparse_direct_chord_factor();
        }

        if (chord_reuse_active &&
            mumps_state.sparse_direct_chord_factor_retained) {
            const SparseChordReuseDecision pre_correction_decision =
                decide_sparse_chord_reuse({
                    false,
                    mumps_state.sparse_direct_consecutive_chord_steps,
                    error,
                    0.0});
            if (pre_correction_decision.action ==
                SparseChordReuseAction::RefreshJacobian) {
                if (rank == 0) {
                    std::cout << "Sparse chord reuse: forced refresh error="
                              << error << " consecutive="
                              << mumps_state.sparse_direct_consecutive_chord_steps
                              << '\n';
                }
                release_sparse_direct_chord_factor();
            } else {
                report_memory_mapper_phase(
                    "sparse.step_entry", MPI_COMM_WORLD);
                // Keep the entry residual intact for a full Newton refresh if
                // this speculative correction fails the contraction test.
                {
                    Array<double> chord_rhs(linear_dimension);
                    if (step_selection_plan) {
                        const JacobianSelectedValues gathered =
                            gather_jacobian_selected_values(
                                std::span<const double>{
                                    residual.get_data(), residual.get_nbr()},
                                step_selection_plan->selected_rows());
                        if (!gathered)
                            KADATH_THROW(
                                "reduced chord RHS gather failed: " +
                                gathered.failure_reason);
                        std::copy(gathered.values.begin(),
                                  gathered.values.end(),
                                  chord_rhs.set_data());
                    } else {
                        chord_rhs = residual;
                    }
                    auto* chord_solver = dynamic_cast<MumpsLinearSolver*>(
                        mumps_state.sparse_direct_solver.get());
                    if (chord_solver == nullptr) {
                        KADATH_THROW(
                            "sparse-direct chord state holds a non-MUMPS solver");
                    }
                    const auto chord_solve_start =
                        std::chrono::system_clock::now();
                    chord_solver->solve(chord_rhs.set_data());
                    report_sparse_direct_mumps_apply_timing(
                        MPI_COMM_WORLD, rank == 0 ? &std::cout : nullptr,
                        elapsed_time(chord_solve_start));
                    Array<double> chord_delta(column_count);
                    if (step_selection_plan) {
                        const JacobianSelectedValues scattered =
                            scatter_jacobian_selected_values(
                                std::span<const double>{
                                    chord_rhs.get_data(), chord_rhs.get_nbr()},
                                column_count,
                                step_selection_plan->selected_columns());
                        if (!scattered)
                            KADATH_THROW(
                                "reduced chord correction scatter failed: " +
                                scattered.failure_reason);
                        std::copy(scattered.values.begin(),
                                  scattered.values.end(),
                                  chord_delta.set_data());
                    } else {
                        chord_delta = std::move(chord_rhs);
                    }
                    zero_selected_sparse_corrections(
                        chord_delta, column_count,
                        rank == 0 ? &std::cout : nullptr);
                    if (env_flag_enabled(
                            "SPARSE_PROJECT_ZSYM_ALIGNED", false) ||
                        env_flag_enabled(
                            "DENSE_PROJECT_ZSYM_ALIGNED", false)) {
                        project_z_symmetric_diagnostic_correction(
                            chord_delta, rank == 0 ? &std::cout : nullptr);
                    }

                    if (step_selection_plan) {
                        const JacobianSelectionNorms correction_norms =
                            measure_jacobian_selection_norms(
                                std::span<const double>{
                                    chord_delta.get_data(),
                                    chord_delta.get_nbr()},
                                step_selection_plan->selected_columns());
                        if (!correction_norms ||
                            correction_norms.forbidden_linf > 1e-14) {
                            abandon_reduced_step(
                                "inactive scattered correction exceeds 1e-14");
                            if (env_flag_enabled(
                                    "RESIDUAL_FORWARD", true)) {
                                store_forwarded_residual(std::move(residual));
                            }
                            report_memory_mapper_phase(
                                "sparse.step_exit", MPI_COMM_WORLD);
                            return false;
                        }
                    }

                    State_snapshot chord_snapshot = snapshot_state();
                    int chord_offset = 0;
                    espace.xx_to_vars_variable_domains(
                        this, chord_delta, chord_offset);
                    xx_to_vars_delta(chord_delta, chord_offset);

                    Array<double> chord_residual(sec_member_partitioned());
                    double chord_error = 0.0;
                    for (std::size_t i = 0;
                         i < chord_residual.get_nbr(); ++i) {
                        const double magnitude = std::abs(
                            chord_residual(static_cast<int>(i)));
                        if (!std::isfinite(magnitude)) {
                            chord_error =
                                std::numeric_limits<double>::infinity();
                            break;
                        }
                        chord_error = std::max(chord_error, magnitude);
                    }

                    if (step_selection_plan) {
                        const JacobianSelectionNorms residual_norms =
                            measure_jacobian_selection_norms(
                                std::span<const double>{
                                    chord_residual.get_data(),
                                    chord_residual.get_nbr()},
                                step_selection_plan->selected_rows());
                        const JacobianForbiddenResidualCheck forbidden_check =
                            check_forbidden_residual(residual_norms);
                        if (rank == 0 && timing_enabled) {
                            std::cout
                                << "Jacobian sector reduction residual: active_Linf="
                                << residual_norms.active_linf
                                << " forbidden_Linf="
                                << residual_norms.forbidden_linf
                                << " limit=" << forbidden_check.limit << '\n';
                        }

                        double inactive_state_drift = 0.0;
                        std::string inactive_state_failure;
                        const bool state_valid = read_inactive_state_drift(
                            *step_selection_plan, false,
                            inactive_state_drift, inactive_state_failure);
                        if (!state_valid) {
                            inactive_state_drift =
                                std::numeric_limits<double>::infinity();
                        }
                        if (rank == 0 && timing_enabled) {
                            std::cout
                                << "Jacobian sector reduction: inactive state drift Linf="
                                << inactive_state_drift
                                << " (limit=1e-14)\n";
                        }
                        if (!forbidden_check.allowed ||
                            !state_valid ||
                            !jacobian_inactive_state_drift_allowed(
                                inactive_state_drift)) {
                            abandon_reduced_step(
                                !forbidden_check.allowed
                                    ? "forbidden residual norm exceeded its runtime guard"
                                    : (!state_valid
                                           ? "inactive state drift check is unsupported: " +
                                                 inactive_state_failure
                                           : "inactive state drift exceeds 1e-14"));
                            restore_rejected_reduced_step(
                                chord_snapshot, chord_residual);
                            if (env_flag_enabled(
                                    "RESIDUAL_FORWARD", true)) {
                                store_forwarded_residual(
                                    std::move(chord_residual));
                            }
                            report_memory_mapper_phase(
                                "sparse.step_exit", MPI_COMM_WORLD);
                            return false;
                        }
                    }
                    const SparseChordReuseDecision candidate_decision =
                        decide_sparse_chord_reuse({
                            true,
                            mumps_state.sparse_direct_consecutive_chord_steps,
                            error,
                            chord_error});
                    if (candidate_decision.action ==
                        SparseChordReuseAction::AcceptChord) {
                        const double previous_error = error;
                        error = chord_error;
                        ++mumps_state.sparse_direct_consecutive_chord_steps;
                        if (env_flag_enabled(
                                "RESIDUAL_FORWARD", true)) {
                            store_forwarded_residual(
                                std::move(chord_residual));
                        }
                        if (rank == 0) {
                            std::cout << "Sparse chord reuse: accepted error_prev="
                                      << previous_error << " error_new=" << error
                                      << " consecutive="
                                      << mumps_state
                                             .sparse_direct_consecutive_chord_steps
                                      << '\n';
                        }
                        if (error < precision) {
                            release_sparse_direct_chord_factor();
                        }
                        report_memory_mapper_phase(
                            "sparse.step_exit", MPI_COMM_WORLD);
                        return false;
                    }

                    const double previous_error = error;
                    restore_state(chord_snapshot);
                    release_sparse_direct_chord_factor();
                    // The rejected residual evaluation left the equation-term,
                    // metric, and definition caches at the trial point. Rebuild
                    // them from the restored unknowns before Jacobian assembly,
                    // and use that fresh residual as the full-step RHS.
                    residual = sec_member_partitioned();
                    if (rank == 0) {
                        std::cout << "Sparse chord reuse: rejected error_prev="
                                  << previous_error << " error_new="
                                  << chord_error << "; refreshing Jacobian\n";
                    }
                }
            }
        }

        const bool reuse_configuration_matches =
            mumps_state.sparse_direct_dimension == linear_dimension &&
            mumps_state.sparse_direct_selection_plan.get() ==
                step_selection_plan.get() &&
            mumps_state.sparse_direct_ordering == mumps_ordering &&
            mumps_state.sparse_direct_blr == blr_icntl35 &&
            mumps_state.sparse_direct_out_of_core_mode == out_of_core_mode &&
            mumps_state.sparse_direct_out_of_core_touch ==
                out_of_core_touch &&
            mumps_state.sparse_direct_out_of_core_safety ==
                out_of_core_safety &&
            mumps_state.sparse_direct_out_of_core_budget_mb ==
                out_of_core_budget_mb &&
            mumps_state.sparse_direct_ranks_per_node ==
                mumps_ranks_per_node &&
            mumps_state.sparse_direct_pattern_drop_tol == pattern_drop_tol;
        if (!analyze_reuse_requested) {
            if (mumps_state.sparse_direct_solver ||
                !mumps_state.sparse_direct_pattern_irn.empty() ||
                mumps_state.sparse_direct_dimension != 0) {
                clear_sparse_direct_reuse_state();
            }
            mumps_state.sparse_direct_analyze_reuse_refused = false;
        } else if (!reuse_configuration_matches &&
                   (mumps_state.sparse_direct_dimension != 0 ||
                    mumps_state.sparse_direct_analyze_reuse_refused)) {
            clear_sparse_direct_reuse_state();
            mumps_state.sparse_direct_analyze_reuse_refused = false;
        }
        if (analyze_reuse_requested &&
            mumps_state.sparse_direct_dimension == 0 &&
            !mumps_state.sparse_direct_analyze_reuse_refused) {
            mumps_state.sparse_direct_dimension = linear_dimension;
            mumps_state.sparse_direct_selection_plan = step_selection_plan;
            mumps_state.sparse_direct_ordering = mumps_ordering;
            mumps_state.sparse_direct_blr = blr_icntl35;
            mumps_state.sparse_direct_out_of_core_mode = out_of_core_mode;
            mumps_state.sparse_direct_out_of_core_touch = out_of_core_touch;
            mumps_state.sparse_direct_out_of_core_safety = out_of_core_safety;
            mumps_state.sparse_direct_out_of_core_budget_mb =
                out_of_core_budget_mb;
            mumps_state.sparse_direct_ranks_per_node = mumps_ranks_per_node;
            mumps_state.sparse_direct_pattern_drop_tol = pattern_drop_tol;
        }
        const bool analyze_reuse_active =
            analyze_reuse_requested &&
            !mumps_state.sparse_direct_analyze_reuse_refused;
        const double assembly_drop_tol =
            analyze_reuse_active ? pattern_drop_tol : drop_tol;

        // The full residual remains the nonlinear oracle.  A reduced step
        // gathers its MUMPS RHS by selected rows; the ordinary path preserves
        // the existing move into the in/out solve buffer.
        Array<double> delta(linear_dimension);
        if (step_selection_plan) {
            const JacobianSelectedValues gathered =
                gather_jacobian_selected_values(
                    std::span<const double>{
                        residual.get_data(), residual.get_nbr()},
                    step_selection_plan->selected_rows());
            if (!gathered)
                KADATH_THROW("reduced RHS gather failed: " +
                             gathered.failure_reason);
            std::copy(gathered.values.begin(), gathered.values.end(),
                      delta.set_data());
        } else {
            delta = std::move(residual);
        }
        report_memory_mapper_phase("sparse.step_entry", MPI_COMM_WORLD);

        if (mumps_state.sparse_direct_chord_factor_retained) {
            KADATH_THROW(
                "sparse-direct chord factor must be released before assembly");
        }
        const auto build_start = std::chrono::system_clock::now();
        JacobianEmissionCaps emission_caps;
        emission_caps.selected_block_supported = true;
        emission_caps.physical_payload_supported = true;
        emission_caps.analyze_reuse_requested = analyze_reuse_requested;
        emission_caps.replay_capture_requested =
            direct_replay_capture_requested;
        emission_caps.parity_mass_probe_requested =
            parity_mass_probe_requested;
        JacobianEmissionPlan emission_plan = plan_jacobian_emission(
            *this, column_count, step_selection_plan, emission_caps);
        require_collective_jacobian_emission_plan_agreement(
            emission_plan, MPI_COMM_WORLD);
        JacobianAssembler assembler(*this, MPI_COMM_WORLD);
        AssembledJacobianCoo assembled_jacobian =
            assembler.assemble(assembly_drop_tol, emission_plan);
        if (assembled_jacobian.n != linear_dimension) {
            KADATH_THROW(
                "assembled Jacobian dimension does not match the frozen selection role");
        }
        report_memory_mapper_phase("sparse.post_assembly", MPI_COMM_WORLD);
        std::shared_ptr<const JacobianSelectionPlan> post_assembly_plan;
        if (config.sparse_sector_reduce) {
            const std::shared_ptr<JacobianParityMaskState>& parity_state =
                jacobian_parity_mask_state();
            if (parity_state && parity_state->n == column_count &&
                parity_state->reduction_decision ==
                    JacobianParityMaskState::ReductionDecision::Eligible) {
                post_assembly_plan = parity_state->selection_plan;
            }
        }
        const bool factor_role_still_current =
            jacobian_selection_factor_compatible(
                step_selection_plan, linear_dimension,
                post_assembly_plan, assembled_jacobian.n);
        if (rank == 0) {
            const std::shared_ptr<JacobianParityMaskState>& parity_state =
                jacobian_parity_mask_state();
            const SparseDirectMumpsSystemMode system_mode =
                sparse_direct_mumps_system_mode(
                    step_selection_plan != nullptr, parity_state.get());
            print_sparse_direct_mumps_system_summary(
                std::cout, system_mode, linear_dimension,
                assembled_jacobian.nnz);
        }
        report_sparse_direct_mumps_jacobian_timing(
            MPI_COMM_WORLD, rank == 0 ? &std::cout : nullptr,
            elapsed_time(build_start),
            sparse_direct_mumps_coo_allocated_bytes(assembled_jacobian));
        if (timing_enabled) {
            assembler.dump_column_profile();
            dump_ope_mult_profile();
            reset_ope_mult_profile();
            dump_ope_action_profile();
            reset_ope_action_profile();
            assembler.diagonal_stats(assembled_jacobian, drop_tol);
        }
        const double rss_after_assembly_mb =
            analyze_reuse_requested ? max_resident_mb() : -1.0;

        if (rank == 0 && !mumps_state.settings_printed) {
            if (blr_icntl35 > 0)
                std::cout << "do_newton_sparse: enabling MUMPS BLR (ICNTL(35)=" << blr_icntl35 << ")" << std::endl;
            mumps_state.settings_printed = true;
        }

        MumpsPatternSupersetUpdate superset_update;
        std::vector<int> next_pattern_irn;
        std::vector<int> next_pattern_jcn;
        std::vector<long long> next_pattern_column_offsets;
        std::vector<double> next_aligned_values;
        bool use_cached_analysis = analyze_reuse_active;
        if (use_cached_analysis) {
            if (rank == 0) {
                superset_update = update_mumps_pattern_superset(
                    linear_dimension,
                    assembled_jacobian.nnz,
                    assembled_jacobian.irn.data(),
                    assembled_jacobian.jcn.data(),
                    assembled_jacobian.a.data(),
                    drop_tol,
                    mumps_state.sparse_direct_pattern_irn,
                    mumps_state.sparse_direct_pattern_jcn,
                    mumps_state.sparse_direct_pattern_column_offsets,
                    next_pattern_irn,
                    next_pattern_jcn,
                    next_pattern_column_offsets,
                    next_aligned_values);
            }
            long long update_counts[5] = {
                superset_update.candidate_nnz,
                superset_update.numerical_nnz,
                superset_update.superset_nnz,
                superset_update.new_pattern_entries,
                superset_update.explicit_zero_entries};
            int pattern_changed = superset_update.pattern_changed ? 1 : 0;
            MPI_Bcast(update_counts, 5, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(&pattern_changed, 1, MPI_INT, 0, MPI_COMM_WORLD);
            superset_update.pattern_changed = pattern_changed != 0;
            superset_update.candidate_nnz = update_counts[0];
            superset_update.numerical_nnz = update_counts[1];
            superset_update.superset_nnz = update_counts[2];
            superset_update.new_pattern_entries = update_counts[3];
            superset_update.explicit_zero_entries = update_counts[4];

            const bool superset_over_ratio =
                superset_update.numerical_nnz <= 0 ||
                static_cast<double>(superset_update.superset_nnz) >
                    config.mumps.sparse_superset_max_nnz_ratio *
                        static_cast<double>(superset_update.numerical_nnz);
            if (superset_over_ratio) {
                if (rank == 0) {
                    std::cerr << "MUMPS analyze-reuse REFUSED: superset_nnz="
                              << superset_update.superset_nnz
                              << " numerical_nnz=" << superset_update.numerical_nnz
                              << " ratio_cap="
                              << config.mumps.sparse_superset_max_nnz_ratio
                              << "; falling back to per-step numerical pattern"
                              << std::endl;
                }
                clear_sparse_direct_reuse_state();
                mumps_state.sparse_direct_dimension = linear_dimension;
                mumps_state.sparse_direct_selection_plan =
                    step_selection_plan;
                mumps_state.sparse_direct_ordering = mumps_ordering;
                mumps_state.sparse_direct_blr = blr_icntl35;
                mumps_state.sparse_direct_out_of_core_mode = out_of_core_mode;
                mumps_state.sparse_direct_out_of_core_touch =
                    out_of_core_touch;
                mumps_state.sparse_direct_out_of_core_safety =
                    out_of_core_safety;
                mumps_state.sparse_direct_out_of_core_budget_mb =
                    out_of_core_budget_mb;
                mumps_state.sparse_direct_ranks_per_node =
                    mumps_ranks_per_node;
                mumps_state.sparse_direct_pattern_drop_tol = pattern_drop_tol;
                mumps_state.sparse_direct_analyze_reuse_refused = true;
                // Enforce the cap before ordinary analysis/factorization: do
                // not retain a rejected superset (or its aligned values) next
                // to the compact numerical COO and MUMPS factor workspace.
                std::vector<int>{}.swap(next_pattern_irn);
                std::vector<int>{}.swap(next_pattern_jcn);
                std::vector<long long>{}.swap(next_pattern_column_offsets);
                std::vector<double>{}.swap(next_aligned_values);
                compact_coo_to_numerical_drop(assembled_jacobian, drop_tol, rank);
                use_cached_analysis = false;
            }
        }

        const bool measure_mumps_phases = true;
        std::unique_ptr<MumpsLinearSolver> step_solver;
        MumpsLinearSolver* solver = nullptr;
        bool retained_chord_factor_this_step = false;
        bool analysis_reused = false;
        const char* reuse_action = "disabled";
        double analyze_seconds = 0.0;
        // Applying the correction, re-evaluating the residual and re-checking
        // the reduced-step guards are identical for the single-block solve and
        // for the sequential parity-sector solve, so both paths finish here.
        auto apply_correction_and_finish =
            [&](Array<double>& full_delta) -> bool {
            zero_selected_sparse_corrections(
                full_delta, column_count, rank == 0 ? &std::cout : nullptr);

            if (env_flag_enabled("SPARSE_PROJECT_ZSYM_ALIGNED", false) ||
                env_flag_enabled("DENSE_PROJECT_ZSYM_ALIGNED", false)) {
                project_z_symmetric_diagnostic_correction(
                    full_delta, rank == 0 ? &std::cout : nullptr);
            }

            if (step_selection_plan) {
                const JacobianSelectionNorms correction_norms =
                    measure_jacobian_selection_norms(
                        std::span<const double>{
                            full_delta.get_data(), full_delta.get_nbr()},
                        step_selection_plan->selected_columns());
                if (!correction_norms ||
                    correction_norms.forbidden_linf > 1e-14) {
                    abandon_reduced_step(
                        "inactive scattered correction exceeds 1e-14");
                    if (env_flag_enabled("RESIDUAL_FORWARD", true))
                        store_forwarded_residual(std::move(residual));
                    report_memory_mapper_phase(
                        "sparse.step_exit", MPI_COMM_WORLD);
                    return false;
                }
            }

            std::unique_ptr<State_snapshot> reduced_step_snapshot;
            if (step_selection_plan) {
                reduced_step_snapshot =
                    std::make_unique<State_snapshot>(snapshot_state());
            }
            int offset = 0;
            espace.xx_to_vars_variable_domains(this, full_delta, offset);
            xx_to_vars_delta(full_delta, offset);

            Array<double> trial_residual(sec_member_partitioned());
            double trial_error = 0.0;
            for (std::size_t i = 0; i < trial_residual.get_nbr(); ++i)
                trial_error = std::max(trial_error, std::abs(trial_residual(static_cast<int>(i))));
            error = trial_error;
            if (step_selection_plan) {
                const JacobianSelectionNorms residual_norms =
                    measure_jacobian_selection_norms(
                        std::span<const double>{
                            trial_residual.get_data(), trial_residual.get_nbr()},
                        step_selection_plan->selected_rows());
                const JacobianForbiddenResidualCheck forbidden_check =
                    check_forbidden_residual(residual_norms);
                if (rank == 0 && timing_enabled) {
                    std::cout
                        << "Jacobian sector reduction residual: active_Linf="
                        << residual_norms.active_linf << " forbidden_Linf="
                        << residual_norms.forbidden_linf
                        << " limit=" << forbidden_check.limit << '\n';
                }

                double inactive_state_drift = 0.0;
                std::string inactive_state_failure;
                const bool state_valid = read_inactive_state_drift(
                    *step_selection_plan, false, inactive_state_drift,
                    inactive_state_failure);
                if (!state_valid)
                    inactive_state_drift = std::numeric_limits<double>::infinity();
                if (rank == 0 && timing_enabled) {
                    std::cout
                        << "Jacobian sector reduction: inactive state drift Linf="
                        << inactive_state_drift << " (limit=1e-14)\n";
                }
                if (!forbidden_check.allowed || !state_valid ||
                    !jacobian_inactive_state_drift_allowed(
                        inactive_state_drift)) {
                    abandon_reduced_step(
                        !forbidden_check.allowed
                            ? "forbidden residual norm exceeded its runtime guard"
                            : (!state_valid
                                   ? "inactive state drift check is unsupported: " +
                                         inactive_state_failure
                                   : "inactive state drift exceeds 1e-14"));
                    restore_rejected_reduced_step(
                        *reduced_step_snapshot, trial_residual);
                    if (env_flag_enabled("RESIDUAL_FORWARD", true))
                        store_forwarded_residual(std::move(trial_residual));
                    report_memory_mapper_phase(
                        "sparse.step_exit", MPI_COMM_WORLD);
                    return false;
                }
            }
            if (!step_selection_plan &&
                mumps_state.sparse_direct_masked_full_rebuild_pending) {
                mumps_state.sparse_direct_masked_full_rebuild_pending = false;
                if (rank == 0) {
                    std::cout << "Jacobian sector reduction: completed pending "
                                 "masked-full rebuild\n";
                }
            }
            if (env_flag_enabled("RESIDUAL_FORWARD", true))
                store_forwarded_residual(std::move(trial_residual));
            report_memory_mapper_phase("sparse.step_exit", MPI_COMM_WORLD);

            if (retained_chord_factor_this_step) {
                if (rank == 0) {
                    std::cout << "Sparse chord reuse: retained factorization error="
                              << error << '\n';
                }
                if (error < precision) {
                    release_sparse_direct_chord_factor();
                }
            }

            return false;
        };

        // The ordinary production step and the optional parity split both use
        // the direct-owned orchestration primitive. Analyze reuse and replay
        // capture retain their concrete-MUMPS path below because each needs a
        // mid-lifecycle symbolic/factor snapshot.
        if (!analyze_reuse_requested && !direct_replay_capture_requested) {
            const bool retain_ordinary_factor =
                chord_reuse_active && factor_role_still_current;
            SparseDirectMumpsSolveOptions solve_options;
            solve_options.ordering = mumps_ordering;
            solve_options.out_of_core_mode = out_of_core_mode;
            solve_options.blr = blr_icntl35;
            solve_options.communicator = MPI_COMM_WORLD;
            solve_options.ranks_per_node = mumps_ranks_per_node;
            solve_options.out_of_core_touch = out_of_core_touch;
            solve_options.out_of_core_safety = out_of_core_safety;
            solve_options.out_of_core_budget_mb = out_of_core_budget_mb;
            solve_options.parity_split_requested =
                config.sparse_parity_split_solve && !step_selection_plan &&
                assembled_jacobian.n == column_count;
            solve_options.ordinary_factor_lifecycle =
                retain_ordinary_factor
                    ? SparseDirectMumpsFactorLifecycle::Retained
                    : SparseDirectMumpsFactorLifecycle::Transient;
            // Direct split stays transient even when chord reuse is enabled:
            // this preserves the one-sector-factor residency policy.
            solve_options.split_factor_lifecycle =
                SparseDirectMumpsFactorLifecycle::Transient;
            solve_options.measure_phases = measure_mumps_phases;
            solve_options.report_apply_timing = true;
            solve_options.memory_phase_prefix = "sparse";
            solve_options.diagnostic = rank == 0 ? &std::cout : nullptr;

            SparseDirectMumpsSolveResult solve_result =
                run_sparse_direct_mumps_solve(
                    assembled_jacobian, delta.set_data(),
                    jacobian_parity_mask_state().get(),
                    mumps_state.icntl14, solve_options);
            if (solve_result.retained_factor) {
                mumps_state.sparse_direct_solver =
                    std::move(solve_result.retained_factor);
                mumps_state.sparse_direct_chord_factor_retained = true;
                mumps_state.sparse_direct_consecutive_chord_steps = 0;
                mumps_state.sparse_direct_selection_plan = step_selection_plan;
                mumps_state.sparse_direct_factor_dimension = linear_dimension;
                mumps_state.sparse_direct_out_of_core_mode = out_of_core_mode;
                mumps_state.sparse_direct_out_of_core_touch = out_of_core_touch;
                mumps_state.sparse_direct_out_of_core_safety = out_of_core_safety;
                mumps_state.sparse_direct_out_of_core_budget_mb =
                    out_of_core_budget_mb;
                retained_chord_factor_this_step = true;
            }

            Array<double> full_delta(column_count);
            if (step_selection_plan) {
                const JacobianSelectedValues scattered =
                    scatter_jacobian_selected_values(
                        std::span<const double>{delta.get_data(), delta.get_nbr()},
                        column_count,
                        step_selection_plan->selected_columns());
                if (!scattered) {
                    KADATH_THROW("reduced correction scatter failed: " +
                                 scattered.failure_reason);
                }
                std::copy(scattered.values.begin(), scattered.values.end(),
                          full_delta.set_data());
            } else {
                full_delta = std::move(delta);
            }
            return apply_correction_and_finish(full_delta);
        }

        if (use_cached_analysis) {
            if (superset_update.pattern_changed) {
                // MUMPS borrows irn/jcn through JOB_END. The helper builds the
                // replacement transactionally, so destroy the old solver
                // before moving or freeing its pattern buffers.
                mumps_state.sparse_direct_solver.reset();
                mumps_state.sparse_direct_pattern_irn.swap(next_pattern_irn);
                mumps_state.sparse_direct_pattern_jcn.swap(next_pattern_jcn);
                mumps_state.sparse_direct_pattern_column_offsets.swap(
                    next_pattern_column_offsets);
            }
            mumps_state.sparse_direct_aligned_values.swap(next_aligned_values);
            if (!mumps_state.sparse_direct_solver) {
                std::unique_ptr<MumpsLinearSolver> cached_solver;
                run_sparse_direct_collective_phase(
                    MPI_COMM_WORLD, "MUMPS construction/setup", [&]() {
                        cached_solver = std::make_unique<MumpsLinearSolver>(
                            linear_dimension,
                            mumps_ordering,
                            out_of_core_mode,
                            blr_icntl35,
                            mumps_state.icntl14,
                            MPI_COMM_WORLD,
                            mumps_ranks_per_node,
                            false,
                            out_of_core_touch,
                            out_of_core_safety,
                            out_of_core_budget_mb);
                        cached_solver->set_auto_out_of_core_diagnostic(
                            nullptr, "");
                        cached_solver->set_pattern(
                            linear_dimension,
                            superset_update.superset_nnz,
                            mumps_state.sparse_direct_pattern_irn.data(),
                            mumps_state.sparse_direct_pattern_jcn.data());
                    });
                const auto analyze_start = measure_mumps_phases
                                               ? std::chrono::system_clock::now()
                                               : std::chrono::time_point<
                                                     std::chrono::system_clock>{};
                run_sparse_direct_collective_phase(
                    MPI_COMM_WORLD, "MUMPS analysis",
                    [&]() { cached_solver->analyze_pattern(); });
                analyze_seconds = measure_mumps_phases
                                      ? elapsed_time(analyze_start)
                                      : 0.0;
                mumps_state.sparse_direct_solver = std::move(cached_solver);
                ++mumps_state.sparse_direct_analyze_count;
                reuse_action = mumps_state.sparse_direct_analyze_count == 1
                                   ? "seed-analyze"
                                   : "support-growth-reanalyze";
            } else {
                analysis_reused = true;
                ++mumps_state.sparse_direct_reuse_count;
                reuse_action = "reuse";
            }
            solver = dynamic_cast<MumpsLinearSolver*>(
                mumps_state.sparse_direct_solver.get());
            if (solver == nullptr) {
                KADATH_THROW(
                    "sparse-direct analyze-reuse state holds a non-MUMPS solver");
            }
            AssembledJacobianCoo::IndexVector{}.swap(assembled_jacobian.irn);
            AssembledJacobianCoo::IndexVector{}.swap(assembled_jacobian.jcn);
            AssembledJacobianCoo::ValueVector{}.swap(assembled_jacobian.a);
        } else {
            run_sparse_direct_collective_phase(
                MPI_COMM_WORLD, "MUMPS construction/setup", [&]() {
                    step_solver = std::make_unique<MumpsLinearSolver>(
                        linear_dimension,
                        mumps_ordering,
                        out_of_core_mode,
                        blr_icntl35,
                        mumps_state.icntl14,
                        MPI_COMM_WORLD,
                        mumps_ranks_per_node,
                        false,
                        out_of_core_touch,
                        out_of_core_safety,
                        out_of_core_budget_mb);
                    step_solver->set_auto_out_of_core_diagnostic(nullptr, "");
                });
            solver = step_solver.get();
            // set_pattern takes nnz as long long; MUMPS carries the 64-bit nnz
            // field and irn/jcn entries are row/col indices <= n (fit int32).
            solver->set_pattern(
                linear_dimension,
                assembled_jacobian.nnz,
                assembled_jacobian.irn.data(),
                assembled_jacobian.jcn.data());
            const auto analyze_start = measure_mumps_phases
                                           ? std::chrono::system_clock::now()
                                           : std::chrono::time_point<
                                                 std::chrono::system_clock>{};
            run_sparse_direct_collective_phase(
                MPI_COMM_WORLD, "MUMPS analysis",
                [&]() { solver->analyze_pattern(); });
            analyze_seconds = measure_mumps_phases
                                  ? elapsed_time(analyze_start)
                                  : 0.0;
            reuse_action = analyze_reuse_requested ? "refused-fallback" : "disabled";
        }
        report_memory_mapper_phase("sparse.post_analyze", MPI_COMM_WORLD);

        std::uint64_t capture_candidate = 0;
        bool capture_this_candidate = false;
        CapturedAnalysisSettings captured_analysis_settings;
        std::vector<int> captured_analysis_permutation;
        CapturedLinearSystemHash pre_factor_ordered_matrix_hash{};
        if (direct_replay_capture_requested) {
            capture_candidate = ++sparse_direct_capture_candidate_count;
            unsigned long long local_candidate = static_cast<unsigned long long>(capture_candidate);
            unsigned long long minimum_candidate = 0;
            unsigned long long maximum_candidate = 0;
            MPI_Allreduce(&local_candidate, &minimum_candidate, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
            MPI_Allreduce(&local_candidate, &maximum_candidate, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            if (minimum_candidate != maximum_candidate) {
                KADATH_THROW("direct replay capture candidate ordinal diverged across MPI ranks");
            }
            capture_this_candidate =
                capture_candidate == static_cast<std::uint64_t>(config.diagnostics.direct_replay_capture_ordinal);
        }
        if (capture_this_candidate) {
            run_sparse_direct_collective_phase(MPI_COMM_WORLD, "sparse-direct replay analysis snapshot", [&]() {
                if (rank != 0)
                    return;
                solver->copy_analysis_controls(captured_analysis_settings.icntl, captured_analysis_settings.cntl);
                const std::string mumps_version = MUMPS_VERSION;
                if (mumps_version.size() >= captured_analysis_settings.mumps_version.size()) {
                    throw std::runtime_error("MUMPS version string is too long for the replay archive");
                }
                std::copy(mumps_version.begin(), mumps_version.end(), captured_analysis_settings.mumps_version.begin());
                captured_analysis_settings.requested_ordering = mumps_ordering;
                captured_analysis_settings.actual_ordering = solver->last_actual_ordering();
                captured_analysis_settings.communicator_size = nproc;
                captured_analysis_settings.analysis_rank_count = solver->analysis_rank_count();
                captured_analysis_settings.factor_ranks_per_node = solver->factor_ranks_per_node();
                solver->copy_symmetric_permutation_1based(captured_analysis_permutation);
                pre_factor_ordered_matrix_hash = captured_ordered_matrix_hash(
                    static_cast<std::uint64_t>(linear_dimension), static_cast<std::uint64_t>(linear_dimension),
                    assembled_jacobian.irn, assembled_jacobian.jcn, assembled_jacobian.a);
            });
        }

        const double rss_before_factor_mb =
            analyze_reuse_requested ? max_resident_mb() : -1.0;
        const auto factorize_start = measure_mumps_phases
                                         ? std::chrono::system_clock::now()
                                         : std::chrono::time_point<
                                               std::chrono::system_clock>{};
        run_sparse_direct_collective_phase(
            MPI_COMM_WORLD, "MUMPS factorization", [&]() {
                solver->factor_analyzed(
                    use_cached_analysis
                        ? mumps_state.sparse_direct_aligned_values.data()
                        : assembled_jacobian.a.data());
            });
        const double factorize_seconds = measure_mumps_phases
                                             ? elapsed_time(factorize_start)
                                             : 0.0;
        report_memory_mapper_phase("sparse.post_factor_pre_coo_release", MPI_COMM_WORLD);

        if (capture_this_candidate) {
            CapturedLinearSystemHashes capture_hashes;
            run_sparse_direct_collective_phase(MPI_COMM_WORLD, "sparse-direct replay capture", [&]() {
                if (rank != 0)
                    return;

                TaggedJacobianMetadata metadata;
                build_tagged_jacobian_metadata(metadata, /*include_row_incidence=*/true);
                const TaggedJacobianMetadataValidation validation = validate_tagged_jacobian_metadata(metadata);
                if (!validation.ok) {
                    std::string reason = "tagged Jacobian metadata validation failed";
                    if (!validation.errors.empty())
                        reason += ": " + validation.errors.front();
                    throw std::runtime_error(reason);
                }
                if (metadata.nrows != row_count || metadata.ncols != column_count) {
                    throw std::runtime_error("tagged Jacobian metadata dimensions do not "
                                             "match the full sparse-direct system");
                }

                std::vector<std::uint32_t> row_full_indices(static_cast<std::size_t>(linear_dimension));
                std::vector<std::uint32_t> column_full_indices(static_cast<std::size_t>(linear_dimension));
                if (step_selection_plan) {
                    if (step_selection_plan->selected_rows().size() != row_full_indices.size() ||
                        step_selection_plan->selected_columns().size() != column_full_indices.size()) {
                        throw std::runtime_error("frozen Jacobian selection maps do not match "
                                                 "the factor dimension");
                    }
                    for (int i = 0; i < linear_dimension; ++i) {
                        const int full_row = step_selection_plan->selected_rows()[static_cast<std::size_t>(i)];
                        const int full_column = step_selection_plan->selected_columns()[static_cast<std::size_t>(i)];
                        if (full_row < 0 || full_row >= row_count || full_column < 0 || full_column >= column_count) {
                            throw std::runtime_error("frozen Jacobian selection map contains "
                                                     "an out-of-range full index");
                        }
                        row_full_indices[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(full_row);
                        column_full_indices[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(full_column);
                    }
                } else {
                    for (int i = 0; i < linear_dimension; ++i) {
                        row_full_indices[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(i);
                        column_full_indices[static_cast<std::size_t>(i)] = static_cast<std::uint32_t>(i);
                    }
                }

                std::vector<unsigned char> has_same_domain_volume(static_cast<std::size_t>(linear_dimension), 0);
                std::vector<unsigned char> has_cross_domain_volume(static_cast<std::size_t>(linear_dimension), 0);
                if (assembled_jacobian.nnz < 0 ||
                    static_cast<std::uint64_t>(assembled_jacobian.nnz) != assembled_jacobian.irn.size() ||
                    assembled_jacobian.irn.size() != assembled_jacobian.jcn.size() ||
                    assembled_jacobian.irn.size() != assembled_jacobian.a.size()) {
                    throw std::runtime_error("assembled COO lengths do not match the exact "
                                             "factor input");
                }
                for (std::size_t entry = 0; entry < assembled_jacobian.irn.size(); ++entry) {
                    const int reduced_row = assembled_jacobian.irn[entry] - 1;
                    const int reduced_column = assembled_jacobian.jcn[entry] - 1;
                    if (reduced_row < 0 || reduced_row >= linear_dimension || reduced_column < 0 ||
                        reduced_column >= linear_dimension) {
                        throw std::runtime_error("assembled COO contains an index outside the "
                                                 "factor dimension");
                    }
                    const int full_row = static_cast<int>(row_full_indices[static_cast<std::size_t>(reduced_row)]);
                    const int full_column =
                        static_cast<int>(column_full_indices[static_cast<std::size_t>(reduced_column)]);
                    const RowMetadata& row_metadata = metadata.rows[static_cast<std::size_t>(full_row)];
                    const ColumnMetadata& column_metadata = metadata.columns[static_cast<std::size_t>(full_column)];
                    if (column_metadata.column_class != ColumnClass::FieldInteriorVol || column_metadata.domain < 0 ||
                        row_metadata.taxonomy != RowTaxonomy::Vol) {
                        continue;
                    }
                    if (row_metadata.dom == column_metadata.domain) {
                        has_same_domain_volume[static_cast<std::size_t>(reduced_column)] = 1;
                    } else {
                        has_cross_domain_volume[static_cast<std::size_t>(reduced_column)] = 1;
                    }
                }

                std::vector<CapturedRowTag> row_tags(static_cast<std::size_t>(linear_dimension));
                std::vector<CapturedColumnTag> column_tags(static_cast<std::size_t>(linear_dimension));
                for (int i = 0; i < linear_dimension; ++i) {
                    const int full_row = static_cast<int>(row_full_indices[static_cast<std::size_t>(i)]);
                    const RowMetadata& source = metadata.rows[static_cast<std::size_t>(full_row)];
                    CapturedRowTag& target = row_tags[static_cast<std::size_t>(i)];
                    target.original_row = full_row;
                    target.taxonomy = capture_row_taxonomy(source.taxonomy);
                    target.domain = source.dom;
                    target.domain_pair = source.dom_pair;

                    const int full_column = static_cast<int>(column_full_indices[static_cast<std::size_t>(i)]);
                    const ColumnMetadata& column_source = metadata.columns[static_cast<std::size_t>(full_column)];
                    CapturedColumnTag& column_target = column_tags[static_cast<std::size_t>(i)];
                    column_target.original_column = full_column;
                    column_target.column_class = capture_column_class(column_source.column_class);
                    column_target.incidence_role = column_source.column_class == ColumnClass::FieldInteriorVol &&
                                                           has_same_domain_volume[static_cast<std::size_t>(i)] != 0 &&
                                                           has_cross_domain_volume[static_cast<std::size_t>(i)] == 0
                                                       ? 0
                                                       : 1;
                    column_target.domain = column_source.domain;
                    column_target.term_idx = column_source.term_idx;
                    column_target.var_idx = column_source.var_idx;
                    column_target.var_double_idx = column_source.var_double_idx;
                    column_target.vardom_param = column_source.vardom_param;
                    column_target.basis_mode = column_source.basis_mode;
                    column_target.var_name_hash = stable_capture_name_hash(column_source.var_name);
                    column_target.domain_type_id = column_source.domain_type_id;
                    column_target.tensor_component = column_source.tensor_component;
                    column_target.coefficient_i = column_source.coefficient_i;
                    column_target.coefficient_j = column_source.coefficient_j;
                    column_target.coefficient_k = column_source.coefficient_k;
                    column_target.coefficient_nr = column_source.coefficient_nr;
                    column_target.coefficient_nt = column_source.coefficient_nt;
                    column_target.coefficient_np = column_source.coefficient_np;

                    const bool field_column =
                        column_source.term_idx >= 0 &&
                        column_source.domain >= 0;
                    if (field_column &&
                        (column_target.domain_type_id < 0 ||
                         column_target.tensor_component < 0 ||
                         column_target.coefficient_i < 0 ||
                         column_target.coefficient_j < 0 ||
                         column_target.coefficient_k < 0 ||
                         column_target.coefficient_nr <= 0 ||
                         column_target.coefficient_nt <= 0 ||
                         column_target.coefficient_np <= 0)) {
                        throw std::runtime_error(
                            "direct replay capture cannot emit v2 semantic "
                            "coordinates for a field column");
                    }
                }

                const CapturedLinearSystemHash post_factor_ordered_matrix_hash = captured_ordered_matrix_hash(
                    static_cast<std::uint64_t>(linear_dimension), static_cast<std::uint64_t>(linear_dimension),
                    assembled_jacobian.irn, assembled_jacobian.jcn, assembled_jacobian.a);
                if (post_factor_ordered_matrix_hash != pre_factor_ordered_matrix_hash) {
                    throw std::runtime_error("MUMPS JOB=2 mutated the borrowed COO factor input");
                }
                captured_analysis_settings.successful_factor_icntl14 = solver->successful_factor_icntl14();
                captured_analysis_settings.factor_retry_count = solver->factor_retry_count();

                CapturedLinearSystemView capture_view;
                capture_view.rows = static_cast<std::uint64_t>(linear_dimension);
                capture_view.columns = static_cast<std::uint64_t>(linear_dimension);
                capture_view.full_rows = static_cast<std::uint64_t>(row_count);
                capture_view.full_columns = static_cast<std::uint64_t>(column_count);
                capture_view.row_indices_1based = assembled_jacobian.irn;
                capture_view.column_indices_1based = assembled_jacobian.jcn;
                capture_view.values = assembled_jacobian.a;
                capture_view.rhs =
                    std::span<const double>(delta.get_data(), static_cast<std::size_t>(linear_dimension));
                capture_view.row_full_indices_zero_based = row_full_indices;
                capture_view.column_full_indices_zero_based = column_full_indices;
                capture_view.row_tags = row_tags;
                capture_view.column_tags = column_tags;
                capture_view.analysis_permutation_1based = captured_analysis_permutation;
                capture_view.analysis = captured_analysis_settings;
                capture_hashes =
                    write_captured_linear_system(config.diagnostics.direct_replay_capture_path, capture_view);
            });
            if (rank == 0) {
                std::cout << "Sparse-direct replay capture: path=" << config.diagnostics.direct_replay_capture_path
                          << " ordinal=" << capture_candidate
                          << " archive=" << captured_linear_system_hash_hex(capture_hashes.archive)
                          << " matrix=" << captured_linear_system_hash_hex(capture_hashes.ordered_matrix)
                          << " pattern=" << captured_linear_system_hash_hex(capture_hashes.canonical_pattern)
                          << " canonical_values=" << captured_linear_system_hash_hex(capture_hashes.canonical_values)
                          << " rhs=" << captured_linear_system_hash_hex(capture_hashes.rhs) << std::endl;
            }
        }

        // MUMPS borrows centralized COO through every JOB=2 retry. The LU is
        // self-contained afterwards. In the cache path retain only irn/jcn and
        // symbolic state; values are detached and freed. The ordinary path
        // detaches all three arrays as before.
        if (use_cached_analysis) {
            solver->release_factor_values_input();
            std::vector<double>{}.swap(
                mumps_state.sparse_direct_aligned_values);
        } else {
            solver->release_centralized_coo_input();
            AssembledJacobianCoo::IndexVector{}.swap(assembled_jacobian.irn);
            AssembledJacobianCoo::IndexVector{}.swap(assembled_jacobian.jcn);
            AssembledJacobianCoo::ValueVector{}.swap(assembled_jacobian.a);
            if (chord_reuse_active && step_solver &&
                factor_role_still_current) {
                mumps_state.sparse_direct_solver = std::move(step_solver);
                mumps_state.sparse_direct_chord_factor_retained = true;
                mumps_state.sparse_direct_consecutive_chord_steps = 0;
                mumps_state.sparse_direct_selection_plan =
                    step_selection_plan;
                mumps_state.sparse_direct_factor_dimension =
                    linear_dimension;
                mumps_state.sparse_direct_out_of_core_mode =
                    out_of_core_mode;
                mumps_state.sparse_direct_out_of_core_touch =
                    out_of_core_touch;
                mumps_state.sparse_direct_out_of_core_safety =
                    out_of_core_safety;
                mumps_state.sparse_direct_out_of_core_budget_mb =
                    out_of_core_budget_mb;
                retained_chord_factor_this_step = true;
            }
        }
        report_memory_mapper_phase("sparse.post_coo_release", MPI_COMM_WORLD);
        const double rss_after_factor_mb =
            analyze_reuse_requested ? max_resident_mb() : -1.0;

        if (rank == 0) {
            mumps_state.icntl14 = solver->last_icntl14();
            if (analyze_reuse_requested) {
                const double cache_mb = use_cached_analysis
                                            ? static_cast<double>(
                                                  mumps_state.sparse_direct_pattern_irn.size() *
                                                  (sizeof(int) + sizeof(int))) /
                                                  (1024.0 * 1024.0)
                                            : 0.0;
                std::cout << "MUMPS analyze-reuse: action=" << reuse_action
                          << " analysis_reused=" << (analysis_reused ? 1 : 0)
                          << " pattern_drop_tol=" << pattern_drop_tol
                          << " numerical_drop_tol=" << drop_tol
                          << " candidate_nnz=" << superset_update.candidate_nnz
                          << " numerical_nnz=" << superset_update.numerical_nnz
                          << " superset_nnz=" << superset_update.superset_nnz
                          << " new_entries=" << superset_update.new_pattern_entries
                          << " explicit_zeros=" << superset_update.explicit_zero_entries
                          << " pattern_cache_MB=" << cache_mb
                          << " analyze_s=" << analyze_seconds
                          << " factor_s=" << factorize_seconds
                          << " factor_memory_MB=" << solver->factor_memory_mb()
                          << " factor_flops_G=" << solver->factor_flops_gflop()
                          << " rss_after_assembly_MB=" << rss_after_assembly_mb
                          << " rss_before_factor_MB=" << rss_before_factor_mb
                          << " rss_after_factor_MB=" << rss_after_factor_mb
                          << " analyze_count="
                          << mumps_state.sparse_direct_analyze_count
                          << " reuse_count=" << mumps_state.sparse_direct_reuse_count
                          << std::endl;
            }
        }
        MPI_Bcast(&mumps_state.icntl14, 1, MPI_INT, 0, MPI_COMM_WORLD);
        report_sparse_direct_mumps_factorization(
            MPI_COMM_WORLD, rank == 0 ? &std::cout : nullptr,
            "MUMPS analyze+factorize", analyze_seconds, factorize_seconds,
            mumps_ordering_name(solver->last_actual_ordering()),
            solver->out_of_core_enabled(), solver->factor_memory_mb(),
            solver->factor_allocated_memory_mb());

        // Do not put solve() in the outer collective exception wrapper:
        // MumpsLinearSolver::solve() itself broadcasts on MPI_COMM_WORLD after
        // dmumps_c. If a future implementation throws before that broadcast,
        // excluded ranks could already be blocked inside it. The current solve
        // path performs no C++ or INFOG error check before its internal broadcast,
        // so it has no catchable pre-broadcast solver failure to propagate here.
        const auto solve_start = std::chrono::system_clock::now();
        solver->solve(delta.set_data());
        report_sparse_direct_mumps_apply_timing(
            MPI_COMM_WORLD, rank == 0 ? &std::cout : nullptr,
            elapsed_time(solve_start));
        Array<double> full_delta(column_count);
        if (step_selection_plan) {
            const JacobianSelectedValues scattered =
                scatter_jacobian_selected_values(
                    std::span<const double>{delta.get_data(), delta.get_nbr()},
                    column_count,
                    step_selection_plan->selected_columns());
            if (!scattered)
                KADATH_THROW("reduced correction scatter failed: " +
                             scattered.failure_reason);
            std::copy(scattered.values.begin(), scattered.values.end(),
                      full_delta.set_data());
        } else {
            full_delta = std::move(delta);
        }
        return apply_correction_and_finish(full_delta);
#else
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            std::cerr << "do_newton_sparse: built without MUMPS.\n";
        KADATH_THROW("CELEPHAIS_SOLVER=mumps requires a binary built with CELEPHAIS_USE_MUMPS");
#endif
    }
} // namespace Kadath
