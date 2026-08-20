#pragma once
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <mpi.h>

#include <map>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace Kadath
{

struct JacobianEmissionPlan;

/// Exact communicator-wide comparison of the complete value-owned plan
/// fingerprint. Every rank throws on a divergent plan before it can choose a
/// different assembly or preconditioner-reuse path.
void require_collective_jacobian_emission_plan_agreement(
    const JacobianEmissionPlan& plan, MPI_Comm communicator);

template <typename T>
class UninitializedResizeAllocator : public std::allocator<T>
{
  public:
    using value_type = T;

    UninitializedResizeAllocator() noexcept = default;

    template <typename U>
    UninitializedResizeAllocator(const UninitializedResizeAllocator<U>&) noexcept
    {
    }

    template <typename U>
    struct rebind {
        using other = UninitializedResizeAllocator<U>;
    };

    template <typename U, typename... Args>
    void construct(U* pointer, Args&&... args)
    {
        if constexpr (sizeof...(Args) == 0 && std::is_trivially_default_constructible_v<U>) {
            ::new (static_cast<void*>(pointer)) U;
        } else {
            ::new (static_cast<void*>(pointer)) U(std::forward<Args>(args)...);
        }
    }
};

template <typename T, typename U>
bool operator==(const UninitializedResizeAllocator<T>&, const UninitializedResizeAllocator<U>&) noexcept
{
    return true;
}

template <typename T, typename U>
bool operator!=(const UninitializedResizeAllocator<T>&, const UninitializedResizeAllocator<U>&) noexcept
{
    return false;
}

// One independently consumable parity-sector COO on rank 0. Other ranks see
// empty vectors plus the broadcast dimension and nnz. The immutable selection
// plan maps this sector's local row/column numbering back to the full system.
struct AssembledJacobianCooBlock
{
    using IndexVector = std::vector<int, UninitializedResizeAllocator<int>>;
    using ValueVector = std::vector<double, UninitializedResizeAllocator<double>>;

    int parity_label = 0;
    int n = 0;
    long long nnz = 0;
    std::shared_ptr<const JacobianSelectionPlan> selection_plan;
    IndexVector irn;       // rank 0 only; sector-local, 1-based
    IndexVector jcn;       // rank 0 only; sector-local, 1-based
    ValueVector a;         // rank 0 only
};

// Assembled COO Jacobian on rank 0. Other ranks see empty vectors + the
// global nnz value (broadcast). Uses long long for nnz to bypass the
// 32-bit MPI count limit on large problems.
struct AssembledJacobianCoo
{
    using IndexVector = AssembledJacobianCooBlock::IndexVector;
    using ValueVector = AssembledJacobianCooBlock::ValueVector;

    int n = 0;
    long long nnz = 0;
    double drop_tol_used = 0.0;
    // True when the emitted matrix contains no cross-sector entries and the
    // parity state's row_sector/column_sector tables validly address the two
    // sector blocks.
    bool parity_sector_block_diagonal = false;
    IndexVector irn;       // rank 0 only
    IndexVector jcn;       // rank 0 only
    ValueVector a;         // rank 0 only
    // Empty for the legacy centralized COO. A requested selected pre-J1 path
    // returns one + block independently of fused emission; accepted non-verify
    // fused full emission returns the ordered pair + then -. In block form the
    // top-level arrays stay empty and nnz is the sum of the block nnz values.
    std::vector<AssembledJacobianCooBlock> parity_blocks;
};

namespace jacobian_assembler_detail
{
    using VariableDomainOwnerBuckets = std::map<int, std::vector<int>>;

    struct LocalNnzReserveSamplePlan {
        int provisional_columns = 0;
        int final_columns = 0;
    };

    // The cold first Jacobian has no retained nnz count. Sample a small prefix
    // to establish a provisional capacity, then refine at the existing
    // one-eighth-column checkpoint. This bounds early geometric growth without
    // trusting a single clustered column group as a final density estimate.
    LocalNnzReserveSamplePlan make_local_nnz_reserve_sample_plan(
        int local_column_count);

    std::size_t extrapolate_local_nnz_reserve(
        std::size_t observed_nnz,
        int observed_columns,
        int total_columns,
        std::size_t dense_upper_bound);

    std::size_t provisional_local_nnz_reserve(
        std::size_t observed_nnz,
        std::size_t extrapolated_nnz);

    // Groups rank-local variable-domain columns by the adapted geometry that
    // owns their coefficient. A zero first-owner size retains one generic
    // bucket for spaces without an owner-aware packed implementation.
    VariableDomainOwnerBuckets bucket_variable_domain_columns_by_owner(
        const std::vector<std::pair<int, int>>& local_and_global_columns,
        const std::vector<ColumnMetadata>& metadata,
        int first_owner_parameter_count);

    // Mirrors the variable-domain scheduler's width cascade and returns the
    // global columns belonging to each actual packed or scalar sweep.
    std::vector<std::vector<int>> build_variable_domain_sweep_groups(
        const std::vector<std::pair<int, int>>& local_and_global_columns,
        const std::vector<ColumnMetadata>& metadata,
        int first_owner_parameter_count,
        const std::vector<int>& enabled_widths);
}

// Emit an opt-in, communicator-wide MemoryMapper/RSS snapshot at a solver
// phase boundary. No-op unless MEMORY_MAPPER_PHASE_PROFILE=1.
void report_memory_mapper_phase(const char* phase, MPI_Comm communicator);

// Encapsulates column distribution + chunked MPI gather of the sparse
// Jacobian. The production default is round-robin; the default-off global
// W-lane group planner assigns complete packed groups to ranks. Emits gathered
// COO on rank 0 and broadcasts the global nnz.
class JacobianAssembler
{
  public:
    JacobianAssembler(System_of_eqs& system, MPI_Comm comm);

    // Assemble the sparse Jacobian: each rank builds its owned column block's
    // triplets, which are gathered to rank 0 and returned as a centralized COO
    // (ICNTL(18)=0).
    // Sparse-direct Newton and JFNK-MUMPS build one emission plan per step and
    // share it with the assembler. Diagnostic and other backend callers use the
    // one-argument route, which builds a full-system-capability plan internally.
    AssembledJacobianCoo assemble(double drop_tol);
    AssembledJacobianCoo assemble(double drop_tol,
                                  JacobianEmissionPlan& emission_plan);

    // Process-local invocation counter for assemble(). Phase 4A of the
    // DDM-Schur backend asserts this counter is unchanged across its run to
    // prove no full-J materialization happened on its path. Thread-safe via
    // a relaxed atomic; only read on rank 0 between user-facing steps.
    static long long call_count();

    // Print and reset the profiling totals owned by this assembler's local
    // Jacobian column workspace.
    void dump_column_profile();

    // Diagonal-stats summary line on rank 0. No-op on other ranks.
    void diagonal_stats(const AssembledJacobianCoo& coo, double drop_tol_reported) const;

  private:
    System_of_eqs& system_;
    JacobianColumnEngine::Workspace column_workspace_;
    MPI_Comm comm_;
    int rank_ = 0;
    int nproc_ = 1;
};

} // namespace Kadath
