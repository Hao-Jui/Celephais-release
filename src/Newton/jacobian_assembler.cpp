#include "Linear_algebra/jacobian_assembler.hpp"
#include "Linear_algebra/jacobian_coo_hash.hpp"
#include "Linear_algebra/jacobian_group_planner.hpp"
#include "Linear_algebra/jacobian_parity_mask.hpp"
#include "Linear_algebra/jacobian_parity_mass.hpp"
#include "Linear_algebra/jacobian_task_claims.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Diagnostics/kernel_profile.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <utility>
#include <string>
#include <vector>

namespace Kadath
{
namespace
{

void require_collective_flag_agreement(bool local_value, const char* name,
                                       MPI_Comm comm, int nproc)
{
    const int local = local_value ? 1 : 0;
    int enabled_count = 0;
    MPI_Allreduce(&local, &enabled_count, 1, MPI_INT, MPI_SUM,
                  comm); // NOLINT(bugprone-casting-through-void)
    if (enabled_count != 0 && enabled_count != nproc) {
        std::ostringstream message;
        message << "JacobianAssembler requires rank-consistent " << name;
        KADATH_THROW(message.str());
    }
}

void append_emission_block_fingerprint(
    std::vector<long long>& serialized,
    const JacobianEmissionBlockFingerprint& block)
{
    serialized.push_back(block.parity_label);
    serialized.push_back(static_cast<long long>(block.selected_rows.size()));
    serialized.insert(serialized.end(), block.selected_rows.begin(),
                      block.selected_rows.end());
    serialized.push_back(
        static_cast<long long>(block.selected_columns.size()));
    serialized.insert(serialized.end(), block.selected_columns.begin(),
                      block.selected_columns.end());
}

std::vector<long long> serialize_emission_fingerprint(
    const JacobianEmissionFingerprint& fingerprint)
{
    std::vector<long long> serialized{
        static_cast<long long>(fingerprint.kind),
        fingerprint.full_dimension,
        fingerprint.assembled_dimension,
        fingerprint.selection_plan_requested,
        fingerprint.parity_mask_requested,
        fingerprint.fused_parity_mask_requested,
        fingerprint.fused_emission_active,
        fingerprint.fused_verify_active,
        fingerprint.speculative_j1_fusion,
        fingerprint.local_coo_blocks_requested,
        fingerprint.physical_block_emission_requested,
        fingerprint.parity_split_requested,
        fingerprint.parity_split_ready,
    };
    append_emission_block_fingerprint(serialized,
                                      fingerprint.assembly_block);
    for (const JacobianEmissionBlockFingerprint& block :
         fingerprint.payload_blocks) {
        append_emission_block_fingerprint(serialized, block);
    }
    serialized.push_back(static_cast<long long>(fingerprint.row_sector.size()));
    for (signed char sector : fingerprint.row_sector)
        serialized.push_back(sector);
    serialized.push_back(
        static_cast<long long>(fingerprint.column_sector.size()));
    for (signed char sector : fingerprint.column_sector)
        serialized.push_back(sector);
    return serialized;
}

void require_collective_emission_plan_agreement(
    const JacobianEmissionPlan& plan, MPI_Comm comm, int nproc)
{
    const std::vector<long long> local =
        serialize_emission_fingerprint(plan.fingerprint);
    if (local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        KADATH_THROW("Jacobian emission plan fingerprint exceeds MPI count range");
    const int local_count = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(nproc));
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    std::vector<int> displacements(static_cast<std::size_t>(nproc));
    long long gathered_count = 0;
    for (int rank = 0; rank < nproc; ++rank) {
        if (gathered_count > std::numeric_limits<int>::max() -
                                 counts[static_cast<std::size_t>(rank)]) {
            KADATH_THROW(
                "collective Jacobian emission fingerprints exceed MPI count range");
        }
        displacements[static_cast<std::size_t>(rank)] =
            static_cast<int>(gathered_count);
        gathered_count += counts[static_cast<std::size_t>(rank)];
    }
    std::vector<long long> gathered(static_cast<std::size_t>(gathered_count));
    MPI_Allgatherv(local.data(), local_count, MPI_LONG_LONG, gathered.data(),
                   counts.data(), displacements.data(), MPI_LONG_LONG, comm);
    for (int rank = 0; rank < nproc; ++rank) {
        const int count = counts[static_cast<std::size_t>(rank)];
        const int displacement = displacements[static_cast<std::size_t>(rank)];
        if (count != local_count ||
            !std::equal(local.begin(), local.end(),
                        gathered.begin() + displacement)) {
            KADATH_THROW(
                "JacobianAssembler requires a rank-consistent emission plan fingerprint");
        }
    }
}

std::uint64_t jacobian_group_plan_signature(const JacobianGlobalGroupPlan& plan)
{
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };
    mix(static_cast<std::uint64_t>(plan.groups.size()));
    for (const JacobianColumnGroup& group : plan.groups) {
        mix(static_cast<std::uint64_t>(group.owner_rank));
        mix(static_cast<std::uint64_t>(group.estimated_cost));
        mix(group.direct ? 1ULL : 0ULL);
        mix(static_cast<std::uint64_t>(group.columns.size()));
        for (int column : group.columns)
            mix(static_cast<std::uint64_t>(column));
    }
    return hash;
}


long long count_direct_singleton_columns(const std::vector<DirectJacobianColumn>& columns)
{
    long long count = 0;
    for (const DirectJacobianColumn& column : columns) {
        if (column.has_entries())
            ++count;
    }
    return count;
}

std::vector<int> env_int_list_value(const char* name)
{
    std::vector<int> result;
    const char* cursor = std::getenv(name);
    if (cursor == nullptr || cursor[0] == '\0')
        return result;

    while (*cursor != '\0') {
        char* end = nullptr;
        const long parsed = std::strtol(cursor, &end, 10);
        if (end != cursor)
            result.push_back(static_cast<int>(parsed));
        cursor = end;
        while (*cursor == ',' || *cursor == ':' ||
               std::isspace(static_cast<unsigned char>(*cursor)))
            ++cursor;
        if (end == cursor && *cursor != '\0')
            ++cursor;
    }
    return result;
}


struct LocalNnzReserveCache {
    int n = -1;
    int nproc = -1;
    int rank = -1;
    long long local_nnz = 0;
};

struct LocalCooStorageStats {
    long long capacity_growth_events = 0;
    long long reallocations = 0;
    long long geometric_copy_bytes = 0;
    long long block_allocations = 0;
    long long allocated_capacity_bytes = 0;
    long long gather_copy_bytes = 0;
};

// Optional first-J storage for the rank-local row/value streams. The legacy
// std::vectors repeatedly relocate all accumulated entries while discovering
// the first Jacobian's nnz. Fixed-capacity blocks make growth append-only; the
// blocks are copied/sent directly into phase 1 of the existing rank-0 gather,
// so no flattened rank-local copy is introduced and the phase-2 jcn boundary
// remains unchanged.
class LocalCooStorage {
  public:
    static constexpr std::size_t block_entries = 256 * 1024;

    LocalCooStorage(bool use_blocks, bool profile)
        : use_blocks_{use_blocks}, profile_{profile}
    {
    }

    void reserve_entries(std::size_t entries)
    {
        if (entries == 0)
            return;
        if (use_blocks_) {
            blocks_.reserve((entries + block_entries - 1) / block_entries);
            return;
        }
        const std::size_t old_row_capacity = rows_.capacity();
        const std::size_t old_value_capacity = values_.capacity();
        rows_.reserve(entries);
        values_.reserve(entries);
        record_vector_growth(old_row_capacity, old_value_capacity, rows_.size());
    }

    void append(int row, double value)
    {
        if (use_blocks_) {
            ensure_append_block();
            Block& block = blocks_.back();
            block.rows[block.size] = row;
            block.values[block.size] = value;
            ++block.size;
            ++size_;
            return;
        }

        if (!profile_) {
            rows_.push_back(row);
            values_.push_back(value);
            return;
        }

        const std::size_t old_size = rows_.size();
        const std::size_t old_row_capacity = rows_.capacity();
        const std::size_t old_value_capacity = values_.capacity();
        rows_.push_back(row);
        values_.push_back(value);
        record_vector_growth(old_row_capacity, old_value_capacity, old_size);
    }

    std::size_t size() const
    {
        return use_blocks_ ? size_ : rows_.size();
    }

    std::size_t capacity_entries() const
    {
        return use_blocks_ ? blocks_.size() * block_entries
                           : std::min(rows_.capacity(), values_.capacity());
    }

    template <typename Consumer>
    void for_each_chunk(std::size_t maximum_entries, Consumer&& consume) const
    {
        if (use_blocks_) {
            for (const Block& block : blocks_) {
                std::size_t offset = 0;
                while (offset < block.size) {
                    const std::size_t count =
                        std::min(maximum_entries, block.size - offset);
                    consume(std::span<const int>(block.rows.get() + offset, count),
                            std::span<const double>(block.values.get() + offset,
                                                    count));
                    offset += count;
                }
            }
            return;
        }

        std::size_t offset = 0;
        while (offset < rows_.size()) {
            const std::size_t count =
                std::min(maximum_entries, rows_.size() - offset);
            consume(std::span<const int>(rows_.data() + offset, count),
                    std::span<const double>(values_.data() + offset, count));
            offset += count;
        }
    }

    void copy_to(int* rows, double* values)
    {
        std::size_t offset = 0;
        for_each_chunk(block_entries,
                       [&](std::span<const int> source_rows,
                           std::span<const double> source_values) {
                           std::memcpy(rows + offset, source_rows.data(),
                                       source_rows.size_bytes());
                           std::memcpy(values + offset, source_values.data(),
                                       source_values.size_bytes());
                           offset += source_rows.size();
                       });
        if (profile_)
            stats_.gather_copy_bytes +=
                static_cast<long long>(offset) *
                static_cast<long long>(sizeof(int) + sizeof(double));
    }

    // Drop the entries rejected by keep(row, value), which is called exactly
    // once per stored entry in storage order so the caller can track which
    // column owns it. Returns the retained count. Only the (default) flat
    // vector storage compacts; the blocked layout would need its per-block
    // sizes rewritten too, so the parity mask declines to engage under
    // JACOBIAN_LOCAL_COO_BLOCKS instead of carrying untested code.
    template <typename Predicate>
    std::size_t retain_entries(Predicate&& keep)
    {
        if (use_blocks_)
            KADATH_THROW("LocalCooStorage::retain_entries: blocked storage");
        std::size_t retained = 0;
        for (std::size_t read = 0; read < rows_.size(); ++read) {
            if (!keep(rows_[read], values_[read]))
                continue;
            rows_[retained] = rows_[read];
            values_[retained] = values_[read];
            ++retained;
        }
        rows_.resize(retained);
        values_.resize(retained);
        return retained;
    }

    // Return the capacity stranded by retain_entries. Copies into a fresh
    // allocation, so the caller decides when the transient old+new residency
    // is affordable. Flat storage only, like retain_entries.
    void shrink_to_fit()
    {
        if (use_blocks_)
            return;
        rows_.shrink_to_fit();
        values_.shrink_to_fit();
    }

    void release()
    {
        if (use_blocks_) {
            std::vector<Block>().swap(blocks_);
            size_ = 0;
        } else {
            std::vector<int>().swap(rows_);
            std::vector<double>().swap(values_);
        }
    }

    const LocalCooStorageStats& stats() const
    {
        return stats_;
    }

  private:
    struct Block {
        Block()
            : rows{std::make_unique_for_overwrite<int[]>(block_entries)},
              values{std::make_unique_for_overwrite<double[]>(block_entries)}
        {
        }

        std::unique_ptr<int[]> rows;
        std::unique_ptr<double[]> values;
        std::size_t size = 0;
    };

    void ensure_append_block()
    {
        if (!blocks_.empty() && blocks_.back().size < block_entries)
            return;
        blocks_.emplace_back();
        if (profile_) {
            ++stats_.capacity_growth_events;
            ++stats_.block_allocations;
            stats_.allocated_capacity_bytes +=
                static_cast<long long>(block_entries) *
                static_cast<long long>(sizeof(int) + sizeof(double));
        }
    }

    void record_vector_growth(std::size_t old_row_capacity,
                              std::size_t old_value_capacity,
                              std::size_t old_size)
    {
        if (!profile_)
            return;
        const bool row_grew = rows_.capacity() != old_row_capacity;
        const bool value_grew = values_.capacity() != old_value_capacity;
        if (!row_grew && !value_grew)
            return;
        ++stats_.capacity_growth_events;
        stats_.allocated_capacity_bytes +=
            static_cast<long long>(rows_.capacity() - old_row_capacity) *
                static_cast<long long>(sizeof(int)) +
            static_cast<long long>(values_.capacity() - old_value_capacity) *
                static_cast<long long>(sizeof(double));
        if (old_size == 0)
            return;
        ++stats_.reallocations;
        stats_.geometric_copy_bytes +=
            static_cast<long long>(old_size) *
            static_cast<long long>((row_grew ? sizeof(int) : 0) +
                                   (value_grew ? sizeof(double) : 0));
    }

    bool use_blocks_ = false;
    bool profile_ = false;
    std::vector<int> rows_;
    std::vector<double> values_;
    std::vector<Block> blocks_;
    std::size_t size_ = 0;
    LocalCooStorageStats stats_;
};


struct WLane2PairingBucket {
    ColumnClass column_class = ColumnClass::Unknown;
    int domain = -1;
    std::string variable_name;

    bool operator<(const WLane2PairingBucket& other) const
    {
        const int class_id = static_cast<int>(column_class);
        const int other_class_id = static_cast<int>(other.column_class);
        if (class_id != other_class_id)
            return class_id < other_class_id;
        if (domain != other.domain)
            return domain < other.domain;
        return variable_name < other.variable_name;
    }
};

// Human-readable ColumnClass label for the JACOBIAN_COLMETA_CSV viz dump.
const char* column_class_name(ColumnClass column_class)
{
    switch (column_class) {
        case ColumnClass::Unknown:            return "Unknown";
        case ColumnClass::FieldUnknown:       return "FieldUnknown";
        case ColumnClass::FieldInterior:      return "FieldInterior";
        case ColumnClass::FieldBoundary:      return "FieldBoundary";
        case ColumnClass::FieldInteriorVol:   return "FieldInteriorVol";
        case ColumnClass::FieldBoundaryTau:   return "FieldBoundaryTau";
        case ColumnClass::FieldOuterShellTau: return "FieldOuterShellTau";
        case ColumnClass::FieldMatching:      return "FieldMatching";
        case ColumnClass::FieldGauge:         return "FieldGauge";
        case ColumnClass::VarDomain:          return "VarDomain";
        case ColumnClass::ScalarGlobal:       return "ScalarGlobal";
    }
    return "Unknown";
}

bool column_can_use_wlane2_bucket(const ColumnMetadata& column)
{
    return column.column_class != ColumnClass::VarDomain &&
           column.column_class != ColumnClass::Unknown;
}

WLane2PairingBucket make_wlane2_pairing_bucket(const ColumnMetadata& column)
{
    WLane2PairingBucket bucket;
    bucket.column_class = column.column_class;
    bucket.domain = column.domain;
    bucket.variable_name = column.var_name;
    return bucket;
}

LocalNnzReserveCache& local_nnz_reserve_cache()
{
    static thread_local LocalNnzReserveCache cache;
    return cache;
}

std::size_t cached_local_nnz_reserve(int n, int nproc, int rank)
{
    const LocalNnzReserveCache& cache = local_nnz_reserve_cache();
    if (cache.n != n || cache.nproc != nproc || cache.rank != rank || cache.local_nnz <= 0)
        return 0;
    return static_cast<std::size_t>(cache.local_nnz + cache.local_nnz / 8);
}

void remember_local_nnz_reserve(int n, int nproc, int rank, long long local_nnz)
{
    LocalNnzReserveCache& cache = local_nnz_reserve_cache();
    cache.n = n;
    cache.nproc = nproc;
    cache.rank = rank;
    cache.local_nnz = local_nnz;
}



} // namespace

void require_collective_jacobian_emission_plan_agreement(
    const JacobianEmissionPlan& plan, MPI_Comm communicator)
{
    int nproc = 1;
    MPI_Comm_size(communicator, &nproc);
    require_collective_emission_plan_agreement(plan, communicator, nproc);
}

namespace jacobian_assembler_detail
{
LocalNnzReserveSamplePlan make_local_nnz_reserve_sample_plan(
    int local_column_count)
{
    LocalNnzReserveSamplePlan plan;
    if (local_column_count <= 0)
        return plan;

    plan.final_columns =
        std::min(local_column_count, std::max(1, local_column_count / 8));
    plan.provisional_columns =
        std::min(plan.final_columns, std::max(1, plan.final_columns / 4));
    return plan;
}

std::size_t extrapolate_local_nnz_reserve(std::size_t observed_nnz,
                                          int observed_columns,
                                          int total_columns,
                                          std::size_t dense_upper_bound)
{
    if (observed_nnz == 0 || observed_columns <= 0 ||
        observed_columns >= total_columns) {
        return 0;
    }

    const std::size_t observed_column_count =
        static_cast<std::size_t>(observed_columns);
    const std::size_t remaining_column_count =
        static_cast<std::size_t>(total_columns - observed_columns);
    const std::size_t entries_per_column =
        (observed_nnz + observed_column_count - 1) / observed_column_count;
    if (entries_per_column >
        (dense_upper_bound - std::min(observed_nnz, dense_upper_bound)) /
            remaining_column_count) {
        return dense_upper_bound;
    }

    const std::size_t extrapolated =
        observed_nnz + entries_per_column * remaining_column_count;
    const std::size_t headroom = extrapolated / 8;
    return extrapolated +
           std::min(headroom, dense_upper_bound - extrapolated);
}

std::size_t provisional_local_nnz_reserve(std::size_t observed_nnz,
                                          std::size_t extrapolated_nnz)
{
    if (extrapolated_nnz <= observed_nnz)
        return extrapolated_nnz;
    const std::size_t remaining = extrapolated_nnz - observed_nnz;
    return observed_nnz + remaining / 4 + (remaining % 4 != 0 ? 1 : 0);
}

VariableDomainOwnerBuckets bucket_variable_domain_columns_by_owner(
    const std::vector<std::pair<int, int>>& local_and_global_columns,
    const std::vector<ColumnMetadata>& metadata,
    int first_owner_parameter_count)
{
    VariableDomainOwnerBuckets result;
    int next_invalid_owner = -1;
    for (const auto& local_and_global : local_and_global_columns) {
        const int local_index = local_and_global.first;
        const int global_column = local_and_global.second;
        int owner = 0;
        if (first_owner_parameter_count > 0) {
            if (global_column < 0 ||
                global_column >= static_cast<int>(metadata.size()) ||
                metadata[static_cast<std::size_t>(global_column)].column_class !=
                    ColumnClass::VarDomain ||
                metadata[static_cast<std::size_t>(global_column)].vardom_param < 0) {
                // Invalid owner metadata must remain scalar, so give the
                // column a unique bucket that cannot form a packed group.
                owner = next_invalid_owner--;
            } else {
                owner = metadata[static_cast<std::size_t>(global_column)].vardom_param <
                                first_owner_parameter_count
                    ? 0
                    : 1;
            }
        }
        result[owner].push_back(local_index);
    }
    return result;
}

std::vector<std::vector<int>> build_variable_domain_sweep_groups(
    const std::vector<std::pair<int, int>>& local_and_global_columns,
    const std::vector<ColumnMetadata>& metadata,
    int first_owner_parameter_count,
    const std::vector<int>& enabled_widths)
{
    std::map<int, int> global_column_by_local_index;
    for (const auto& local_and_global : local_and_global_columns)
        global_column_by_local_index[local_and_global.first] = local_and_global.second;

    const VariableDomainOwnerBuckets buckets =
        bucket_variable_domain_columns_by_owner(
            local_and_global_columns, metadata, first_owner_parameter_count);
    std::vector<std::vector<int>> groups;
    for (const auto& owner_columns : buckets) {
        const std::vector<int>& local_indices = owner_columns.second;
        std::size_t index = 0;
        for (int width : enabled_widths) {
            if (width <= 0)
                continue;
            const std::size_t group_width = static_cast<std::size_t>(width);
            while (index + group_width <= local_indices.size()) {
                std::vector<int> group;
                group.reserve(group_width);
                for (std::size_t lane = 0; lane < group_width; ++lane) {
                    group.push_back(global_column_by_local_index.at(
                        local_indices[index + lane]));
                }
                groups.push_back(std::move(group));
                index += group_width;
            }
        }
        while (index < local_indices.size()) {
            groups.push_back({global_column_by_local_index.at(local_indices[index])});
            ++index;
        }
    }
    return groups;
}
} // namespace jacobian_assembler_detail

JacobianAssembler::JacobianAssembler(System_of_eqs& system, MPI_Comm comm)
    : system_{system}, comm_{comm}
{
    MPI_Comm_rank(comm_, &rank_); // NOLINT(bugprone-casting-through-void)
    MPI_Comm_size(comm_, &nproc_); // NOLINT(bugprone-casting-through-void)
}

namespace
{
std::atomic<long long>& jacobian_assembler_call_counter()
{
    static std::atomic<long long> counter{0};
    return counter;
}
} // namespace

long long JacobianAssembler::call_count()
{
    return jacobian_assembler_call_counter().load(std::memory_order_relaxed);
}

void JacobianAssembler::dump_column_profile()
{
    system_.dump_do_col_J_profile(column_workspace_);
}

AssembledJacobianCoo JacobianAssembler::assemble(double drop_tol)
{
    JacobianEmissionCaps caps;
    JacobianEmissionPlan emission_plan = plan_jacobian_emission(
        system_, system_.get_nbr_unknowns(), nullptr, caps);
    require_collective_jacobian_emission_plan_agreement(emission_plan, comm_);
    return assemble(drop_tol, emission_plan);
}

AssembledJacobianCoo JacobianAssembler::assemble(
    double drop_tol, JacobianEmissionPlan& emission_plan)
{
    // Per-kernel timing probe. Enabled by KERNEL_PROFILE=1; default-off
    // path is a single pointer compare on each kernel entry.
    KernelProfileScope kernel_profile_first_j(KernelContext::FirstJ);
    const bool timing = system_.get_solver_runtime_config().diagnostics.timing;
    MemoryMapperJacobianTrafficScope mapper_traffic_scope(timing);
    Transform1dTrafficScope transform_traffic_scope(timing);
    jacobian_assembler_call_counter().fetch_add(1, std::memory_order_relaxed);
    const bool profile = env_flag_enabled("JACOBIAN_ASSEMBLER_PROFILE");
    const bool use_wlane32 = env_flag_enabled("JACOBIAN_WLANE32", true);
    const bool use_wlane16 = env_flag_enabled("JACOBIAN_WLANE16", true);
    const bool use_wlane8 = env_flag_enabled("JACOBIAN_WLANE8", true);
    const bool use_wlane4 = env_flag_enabled("JACOBIAN_WLANE4", true);
    const bool use_wlane2 = env_flag_enabled("JACOBIAN_WLANE2", true);
    const bool use_variable_domain_wlane2 =
        use_wlane2 && env_flag_enabled("JACOBIAN_VARDOM_WLANE2", true);
    const int variable_domain_max_width = use_variable_domain_wlane2
        ? std::clamp(env_int_value("JACOBIAN_VARDOM_MAX_WIDTH", 2), 2, 32)
        : 0;
    const bool request_global_group_plan =
        env_flag_enabled("JACOBIAN_GLOBAL_GROUP_PLAN", false);
    const bool use_structural_plan_cache =
        env_flag_enabled("JACOBIAN_STRUCTURAL_PLAN_CACHE", true);
    const bool use_local_coo_blocks =
        emission_plan.local_coo_blocks_requested;
    const bool parity_mask_requested = emission_plan.parity_mask_requested;
    const bool selection_plan_requested =
        emission_plan.selection_plan_requested;
    const auto output_path_active = [](const char* path) {
        return path != nullptr && path[0] != '\0' && std::string(path) != "0";
    };
    const char* selected_row_entries_path =
        std::getenv("JACOBIAN_SELECTED_ROW_ENTRIES_CSV");
    const bool selected_row_entries_active =
        output_path_active(selected_row_entries_path);
    const char* coo_hash_path = std::getenv("JACOBIAN_COO_BYTE_HASH");
    const bool coo_hash_active = output_path_active(coo_hash_path);
    const char* canonical_coo_hash_path =
        std::getenv("JACOBIAN_CANONICAL_COO_BYTE_HASH");
    const bool canonical_coo_hash_active =
        output_path_active(canonical_coo_hash_path);
    const char* parity_mass_path = std::getenv("JACOBIAN_PARITY_MASS");
    const bool parity_mass_active = output_path_active(parity_mass_path);
    const bool physical_block_diagnostic_conflict =
        selected_row_entries_active || coo_hash_active ||
        canonical_coo_hash_active || parity_mass_active;
    const int local_physical_block_diagnostic_conflict =
        physical_block_diagnostic_conflict ? 1 : 0;
    int any_physical_block_diagnostic_conflict = 0;
    MPI_Allreduce(&local_physical_block_diagnostic_conflict,
                  &any_physical_block_diagnostic_conflict, 1, MPI_INT,
                  MPI_MAX, comm_);
    emission_plan.centralized_coo_diagnostic_requested =
        any_physical_block_diagnostic_conflict != 0;
    if (emission_plan.physical_block_emission_requested &&
        emission_plan.centralized_coo_diagnostic_requested) {
        if (rank_ == 0) {
            std::cout
                << "Jacobian physical parity blocks: disabled for centralized COO diagnostics"
                << std::endl;
        }
        emission_plan.route_payload_to_combined(
            "centralized COO diagnostic requires Combined payload");
    }
    require_collective_flag_agreement(request_global_group_plan,
                                      "JACOBIAN_GLOBAL_GROUP_PLAN",
                                      comm_, nproc_);
    require_collective_flag_agreement(use_structural_plan_cache,
                                      "JACOBIAN_STRUCTURAL_PLAN_CACHE",
                                      comm_, nproc_);
    if (request_global_group_plan) {
        require_collective_flag_agreement(use_wlane32, "JACOBIAN_WLANE32",
                                          comm_, nproc_);
        require_collective_flag_agreement(use_wlane16, "JACOBIAN_WLANE16",
                                          comm_, nproc_);
        require_collective_flag_agreement(use_wlane8, "JACOBIAN_WLANE8",
                                          comm_, nproc_);
        require_collective_flag_agreement(use_wlane4, "JACOBIAN_WLANE4",
                                          comm_, nproc_);
        require_collective_flag_agreement(use_wlane2, "JACOBIAN_WLANE2",
                                          comm_, nproc_);
    }
    const double t_total = MPI_Wtime();
    const int n = system_.get_nbr_unknowns();
    const std::shared_ptr<const JacobianSelectionPlan>& selection_plan =
        emission_plan.assembly_selection_plan;
    const bool use_selection_plan = static_cast<bool>(selection_plan);
    // The plan builder owns first-J structural grading and may install the
    // parity state slot before this call. Certification below can still demote
    // the plan and retry the emission unmasked.
    const bool fuse_j1_attempt = emission_plan.speculative_j1_fusion;
    const std::shared_ptr<JacobianParityMaskState>& parity_state_at_entry =
        system_.jacobian_parity_mask_state();
    // Mutable copies: a mispredicted J1 attempt turns fusion off for the
    // re-assembly pass.
    bool fused_emission_active = emission_plan.fused_emission_active;
    bool fused_verify_active = emission_plan.fused_verify_active;
    bool fuse_j1_mispredicted = false;
    std::vector<unsigned char> fused_parity_verdicts;
    const int assembled_dimension = use_selection_plan
        ? static_cast<int>(selection_plan->selected_columns().size())
        : n;
    std::vector<int> row_to_reduced;
    std::vector<int> column_to_reduced;
    JacobianSelectedRows selected_rows = std::nullopt;
    if (use_selection_plan) {
        const std::vector<int>& rows = selection_plan->selected_rows();
        const std::vector<int>& columns = selection_plan->selected_columns();
        if (rows.size() != columns.size() || rows.empty())
            KADATH_THROW("Jacobian selection plan is not a nonempty square block");
        row_to_reduced.assign(static_cast<std::size_t>(n), -1);
        column_to_reduced.assign(static_cast<std::size_t>(n), -1);
        for (std::size_t reduced = 0; reduced < rows.size(); ++reduced) {
            const int row = rows[reduced];
            const int column = columns[reduced];
            if (row < 0 || row >= n || column < 0 || column >= n ||
                (reduced > 0 &&
                 (row <= rows[reduced - 1] ||
                  column <= columns[reduced - 1])) ||
                row_to_reduced[static_cast<std::size_t>(row)] >= 0 ||
                column_to_reduced[static_cast<std::size_t>(column)] >= 0) {
                KADATH_THROW("Jacobian selection plan contains an invalid index");
            }
            row_to_reduced[static_cast<std::size_t>(row)] =
                static_cast<int>(reduced);
            column_to_reduced[static_cast<std::size_t>(column)] =
                static_cast<int>(reduced);
        }
        selected_rows = std::span<const int>{rows};
    }
    const std::array<std::shared_ptr<const JacobianSelectionPlan>, 2>&
        physical_block_plans = emission_plan.block_plans;
    std::array<std::vector<int>, 2> physical_row_to_reduced;
    std::array<std::vector<int>, 2> physical_column_to_reduced;
    const bool physical_selection_block =
        emission_plan.kind == JacobianEmissionPlan::Kind::ReducedPlus;
    const bool physical_fused_blocks_configured =
        emission_plan.kind == JacobianEmissionPlan::Kind::FusedPair;
    if (physical_selection_block) {
        if (physical_block_plans[0].get() != selection_plan.get())
            KADATH_THROW("ReducedPlus payload does not match its assembly plan");
    } else if (physical_fused_blocks_configured) {
        for (std::size_t block = 0; block < physical_block_plans.size();
             ++block) {
            if (!physical_block_plans[block])
                KADATH_THROW("FusedPair payload is missing a block plan");
            physical_row_to_reduced[block].assign(
                static_cast<std::size_t>(n), -1);
            physical_column_to_reduced[block].assign(
                static_cast<std::size_t>(n), -1);
            const std::vector<int>& rows =
                physical_block_plans[block]->selected_rows();
            const std::vector<int>& columns =
                physical_block_plans[block]->selected_columns();
            for (std::size_t reduced = 0; reduced < rows.size(); ++reduced) {
                physical_row_to_reduced[block][static_cast<std::size_t>(
                    rows[reduced])] = static_cast<int>(reduced);
                physical_column_to_reduced[block][static_cast<std::size_t>(
                    columns[reduced])] = static_cast<int>(reduced);
            }
        }
    }
    report_memory_mapper_phase("jacobian.entry", comm_);

    const double t_direct_singleton_plan = MPI_Wtime();
    const bool need_column_metadata =
        request_global_group_plan || use_wlane2 || use_wlane4 || use_wlane8 ||
        use_wlane16 || use_wlane32 || use_variable_domain_wlane2;
    // System_of_eqs retains only the immutable direct plan and decoded column
    // metadata across assembler reconstruction. The probe-heavy ColumnInfo map
    // remains a miss-only transient. Direct entries include numerical export
    // values, so a hit requires exact equality of system topology, term tensor
    // layout, and all spectral-basis integer codes; primal coefficient values
    // are irrelevant because the builder exports a fresh one-hot derivative.
    JacobianAssemblerStructuralPlanAccess structural_plan_access;
    const JacobianAssemblerStructuralPlan& structural_plan =
        system_.get_jacobian_assembler_structural_plan(
            need_column_metadata, use_structural_plan_cache,
            structural_plan_access);
    const DirectJacobianColumnPlan& direct_singleton_plan =
        structural_plan.direct_singleton_plan;
    const std::vector<ColumnMetadata>& wlane2_column_metadata =
        structural_plan.column_metadata;
    const double direct_singleton_plan_seconds =
        MPI_Wtime() - t_direct_singleton_plan;
    const bool direct_singleton_ready =
        direct_singleton_plan.columns.size() == static_cast<std::size_t>(n);
    long long planned_direct_singletons = 0;
    if (direct_singleton_ready) {
        if (use_selection_plan) {
            for (int column : selection_plan->selected_columns()) {
                if (direct_singleton_plan.columns[static_cast<std::size_t>(column)]
                        .has_entries()) {
                    ++planned_direct_singletons;
                }
            }
        } else {
            planned_direct_singletons =
                count_direct_singleton_columns(direct_singleton_plan.columns);
        }
    }

    const bool wlane2_metadata_ready =
        wlane2_column_metadata.size() == static_cast<std::size_t>(n);
    int first_variable_domain_owner_parameter_count = 0;
    if (use_variable_domain_wlane2) {
        if (const auto* binary_space =
                dynamic_cast<const Space_bin_ns_nosym*>(&system_.get_space())) {
            first_variable_domain_owner_parameter_count =
                system_.get_space().get_domain(binary_space->ADAPTED1)
                    ->nbr_unknowns_from_adapted();
        }
    }
    const bool use_global_group_plan =
        request_global_group_plan && wlane2_metadata_ready &&
        !use_selection_plan;
    require_collective_flag_agreement(use_global_group_plan,
                                      "global-group metadata readiness",
                                      comm_, nproc_);


    // ---- Per-rank build ----
    LocalCooStorage local_coo{use_local_coo_blocks, profile};
    LocalCooStorage local_minus_coo{use_local_coo_blocks, profile};
    std::vector<long long> loc_col_nnz;
    std::vector<long long> minus_loc_col_nnz;
    long long loc_nnz = 0;
    struct FusedParityEmissionStats {
        std::vector<unsigned char>* verify_verdicts = nullptr;
        long long pre_mask_entries = 0;
        long long dropped_entries = 0;
        double maximum_entry = 0.0;
        double maximum_cross = 0.0;

        bool retain(int row, signed char column_sector, double value,
                    const std::vector<signed char>& row_sector)
        {
            ++pre_mask_entries;
            const double magnitude = std::abs(value);
            maximum_entry = std::max(maximum_entry, magnitude);
            if (jacobian_parity_entry_retained(
                    column_sector,
                    row_sector[static_cast<std::size_t>(row)])) {
                if (verify_verdicts != nullptr)
                    verify_verdicts->push_back(1);
                return true;
            }
            ++dropped_entries;
            maximum_cross = std::max(maximum_cross, magnitude);
            if (verify_verdicts != nullptr) {
                verify_verdicts->push_back(0);
                return true;
            }
            return false;
        }
    } fused_parity_stats;
    if (fused_verify_active)
        fused_parity_stats.verify_verdicts = &fused_parity_verdicts;
    const std::vector<signed char>* fused_row_sector =
        emission_plan.fused_emission_active
            ? &parity_state_at_entry->row_sector
            : nullptr;
    const auto fused_column_sector = [&](int column) {
        return parity_state_at_entry->column_sector[static_cast<std::size_t>(column)];
    };
    const auto physical_fused_emission_active = [&]() {
        return physical_fused_blocks_configured && fused_emission_active;
    };
    const auto physical_block_emission_active = [&]() {
        return physical_selection_block || physical_fused_emission_active();
    };
    const auto physical_block_index_for_column = [&](int column) {
        if (physical_selection_block)
            return 0;
        if (physical_fused_emission_active())
            return fused_column_sector(column) > 0 ? 0 : 1;
        return -1;
    };
    const auto local_storage_for_column = [&](int column)
        -> LocalCooStorage& {
        return physical_block_index_for_column(column) == 1
                   ? local_minus_coo
                   : local_coo;
    };
    const auto row_map_for_column = [&](int column)
        -> std::optional<std::span<const int>> {
        if (use_selection_plan)
            return std::span<const int>{row_to_reduced};
        const int block = physical_block_index_for_column(column);
        return block >= 0
                   ? std::optional<std::span<const int>>{
                         physical_row_to_reduced[static_cast<std::size_t>(block)]}
                   : std::nullopt;
    };
    std::vector<long long> all_nnz64(rank_ == 0 ? static_cast<size_t>(nproc_) : 0u);
    long long nnz = 0;
    std::vector<long long> displs64(rank_ == 0 ? static_cast<size_t>(nproc_ + 1) : 0u, 0LL);

    const double t_reset = MPI_Wtime();
    system_.reset_do_col_J_cache(column_workspace_);
    auto ope_der_cache_jacobian_scope = std::make_unique<OpeDerCacheJacobianScope>();
    auto val_domain_der_abs_cache_scope = std::make_unique<ValDomainDerAbsAssemblyCacheScope>();
    auto ope_der_1d_workspace_scope =
        std::make_unique<OpeDer1dAssemblyWorkspaceScope>();
    const double reset_seconds = MPI_Wtime() - t_reset;



    JacobianGlobalGroupPlan global_group_plan;
    std::vector<std::size_t> local_group_indices;
    std::vector<int> owned_global_columns;
    if (use_selection_plan) {
        const std::vector<int>& columns = selection_plan->selected_columns();
        for (std::size_t selected_index = static_cast<std::size_t>(rank_);
             selected_index < columns.size();
             selected_index += static_cast<std::size_t>(nproc_)) {
            owned_global_columns.push_back(columns[selected_index]);
        }
    } else if (use_global_group_plan) {
        std::vector<bool> direct_columns(static_cast<std::size_t>(n), false);
        if (direct_singleton_ready) {
            for (int column = 0; column < n; ++column) {
                direct_columns[static_cast<std::size_t>(column)] =
                    direct_singleton_plan.columns[static_cast<std::size_t>(column)]
                        .has_entries();
            }
        }
        JacobianGroupPlannerOptions planner_options;
        planner_options.nproc = nproc_;
        planner_options.use_wlane32 = use_wlane32;
        planner_options.use_wlane16 = use_wlane16;
        planner_options.use_wlane8 = use_wlane8;
        planner_options.use_wlane4 = use_wlane4;
        planner_options.use_wlane2 = use_wlane2;
        global_group_plan = build_global_jacobian_group_plan(
            wlane2_column_metadata, direct_columns, planner_options);
        const std::uint64_t local_plan_signature =
            jacobian_group_plan_signature(global_group_plan);
        std::uint64_t min_plan_signature = 0;
        std::uint64_t max_plan_signature = 0;
        MPI_Allreduce(&local_plan_signature, &min_plan_signature, 1, MPI_UINT64_T,
                      MPI_MIN, comm_); // NOLINT(bugprone-casting-through-void)
        MPI_Allreduce(&local_plan_signature, &max_plan_signature, 1, MPI_UINT64_T,
                      MPI_MAX, comm_); // NOLINT(bugprone-casting-through-void)
        if (min_plan_signature != max_plan_signature)
            KADATH_THROW("JacobianAssembler global group plans differ across ranks");
        local_group_indices = global_group_plan.group_indices_by_rank
            [static_cast<std::size_t>(rank_)];
        for (std::size_t group_index : local_group_indices) {
            const JacobianColumnGroup& group =
                global_group_plan.groups[group_index];
            owned_global_columns.insert(owned_global_columns.end(),
                                        group.columns.begin(),
                                        group.columns.end());
        }
    }

    const bool use_explicit_column_ownership =
        use_selection_plan || use_global_group_plan;
    const int local_ncols = use_explicit_column_ownership
        ? static_cast<int>(owned_global_columns.size())
        : ((rank_ < n) ? ((n - rank_ + nproc_ - 1) / nproc_) : 0);
    auto global_column_for_local = [&](int local_index) -> int {
        if (use_explicit_column_ownership) {
            return owned_global_columns[static_cast<std::size_t>(local_index)];
        }
        return rank_ + local_index * nproc_;
    };
    int local_plus_columns = local_ncols;
    if (physical_fused_blocks_configured) {
        local_plus_columns = 0;
        for (int local_index = 0; local_index < local_ncols; ++local_index) {
            if (fused_column_sector(global_column_for_local(local_index)) > 0)
                ++local_plus_columns;
        }
    }
    const auto set_local_column_nnz = [&](int local_index, long long count) {
        const int column = global_column_for_local(local_index);
        if (physical_block_index_for_column(column) == 1) {
            minus_loc_col_nnz[static_cast<std::size_t>(local_index)] = count;
        } else {
            loc_col_nnz[static_cast<std::size_t>(local_index)] = count;
        }
    };
    const auto aggregate_local_coo_size = [&]() {
        return local_coo.size() +
               (physical_fused_emission_active() ? local_minus_coo.size() : 0u);
    };
    const auto reserve_local_coo_entries = [&](std::size_t total_entries) {
        if (!physical_fused_emission_active() || local_ncols == 0) {
            local_coo.reserve_entries(total_entries);
            return;
        }
        const std::size_t plus_entries =
            total_entries * static_cast<std::size_t>(local_plus_columns) /
            static_cast<std::size_t>(local_ncols);
        local_coo.reserve_entries(plus_entries);
        local_minus_coo.reserve_entries(total_entries - plus_entries);
    };
    double columns_seconds = 0.0;
    // Hoisted above the emission retry loop: the post-assembly mask walk and
    // jcn reconstruction replay the per-column visit order.
    std::vector<int> visit_order;
    std::size_t reserved_local_nnz = 0;
    auto direct_column_has_entries = [&](int column) -> bool {
        return direct_singleton_ready &&
               direct_singleton_plan.columns[static_cast<std::size_t>(column)].has_entries();
    };
    // Hoisted lane counters: consumed by the profiling reductions after the
    // emission retry loop; reset at each attempt's start.
    int local_col = 0;
    long long local_direct_singleton_columns = 0;
    long long local_wlane2_pairs = 0;
    long long local_wlane2_columns = 0;
    long long local_wlane2_fallbacks = 0;
    long long local_wlane4_quartets = 0;
    long long local_wlane4_columns = 0;
    long long local_wlane4_fallbacks = 0;
    long long local_wlane32_triacontadyads = 0;
    long long local_wlane32_columns = 0;
    long long local_wlane32_fallbacks = 0;
    long long local_wlane16_hexadectets = 0;
    long long local_wlane16_columns = 0;
    long long local_wlane16_fallbacks = 0;
    long long local_wlane8_octets = 0;
    long long local_wlane8_columns = 0;
    long long local_wlane8_fallbacks = 0;
    long long local_wlane2_eligible_columns = 0;
    long long local_wlane2_ideal_pairable_columns = 0;
    long long local_wlane4_ideal_quartetable_columns = 0;
    long long local_wlane8_ideal_octetable_columns = 0;
    long long local_wlane16_ideal_hexadectetable_columns = 0;
    long long local_wlane32_ideal_triacontadyadable_columns = 0;
    // J1-fusion retry: attempt 0 may emit fused; a mispredicted certification
    // resets the containers and re-runs the emission unmasked. The loop body
    // is the pre-existing emission region, deliberately not re-indented.
    while (true) {
    loc_col_nnz.assign(static_cast<size_t>(local_ncols), 0);
    minus_loc_col_nnz.assign(static_cast<size_t>(local_ncols), 0);
    reserved_local_nnz =
        use_selection_plan ? 0 : cached_local_nnz_reserve(n, nproc_, rank_);
    reserve_local_coo_entries(reserved_local_nnz);
    const jacobian_assembler_detail::LocalNnzReserveSamplePlan reserve_sample_plan =
        jacobian_assembler_detail::make_local_nnz_reserve_sample_plan(local_ncols);
    bool provisional_reserve_done = reserved_local_nnz != 0;
    bool final_sample_reserve_done = reserved_local_nnz != 0;
    const std::size_t dense_local_nnz_upper_bound =
        static_cast<std::size_t>(assembled_dimension) *
        static_cast<std::size_t>(local_ncols);
    int local_columns_accounted = 0;

    const double t_columns = MPI_Wtime();
    // Per-domain der_abs cache flush state. Resets the assembly-scoped
    // Val_domain::compute_der_abs memo when the AD path crosses into a
    // new domain so cross-domain entries do not stay resident.
    // Default-ON after spinning-BNS res=9 / res=11 / res=13 stack A/B
    // (2026-05-27) confirmed the regression at res=13 that previously
    // forced default-OFF was caused by round-robin column-visit order
    // interleaving domain visits; the new clustered visit order below
    // (steps 2 + 3 of the stack) makes the flush trigger fire once per
    // (column_class, domain) bucket transition instead of once per
    // round-robin shuffle. Combined with explicit chunked-jcn emission
    // the full stack is Pareto-improving over default-OFF at every
    // measured resolution. Opt out via PER_BUCKET_DER_ABS_FLUSH=0
    // for diagnostic isolation.
    int prev_bucket_domain = -1;
    bool prev_bucket_valid = false;
    static const bool per_bucket_der_abs_flush_enabled =
        env_flag_enabled("PER_BUCKET_DER_ABS_FLUSH", true);
    local_col = 0;
    local_direct_singleton_columns = 0;
    local_wlane2_pairs = 0;
    local_wlane2_columns = 0;
    local_wlane2_fallbacks = 0;
    local_wlane4_quartets = 0;
    local_wlane4_columns = 0;
    local_wlane4_fallbacks = 0;
    local_wlane32_triacontadyads = 0;
    local_wlane32_columns = 0;
    local_wlane32_fallbacks = 0;
    local_wlane16_hexadectets = 0;
    local_wlane16_columns = 0;
    local_wlane16_fallbacks = 0;
    local_wlane8_octets = 0;
    local_wlane8_columns = 0;
    local_wlane8_fallbacks = 0;

    auto emit_direct_column = [&](int column) {
        LocalCooStorage& storage = local_storage_for_column(column);
        const std::optional<std::span<const int>> row_map =
            row_map_for_column(column);
        const DirectJacobianColumn& direct_column =
            direct_singleton_plan.columns[static_cast<std::size_t>(column)];
        const std::size_t end_entry_index =
            direct_column.first_entry_index + direct_column.entry_count;
        for (std::size_t entry_index = direct_column.first_entry_index;
             entry_index < end_entry_index; ++entry_index) {
            const DirectJacobianEntry& direct_entry =
                direct_singleton_plan.entries[entry_index];
            if (std::abs(direct_entry.value) > drop_tol) {
                if (fused_emission_active &&
                    !fused_parity_stats.retain(
                        direct_entry.row, fused_column_sector(column),
                        direct_entry.value, *fused_row_sector)) {
                    continue;
                }
                if (row_map) {
                    const int reduced_row = (*row_map)[static_cast<std::size_t>(
                        direct_entry.row)];
                    if (reduced_row < 0) {
                        if (use_selection_plan)
                            continue;
                        KADATH_THROW(
                            "fused direct Jacobian entry is outside its parity block");
                    }
                    storage.append(reduced_row + 1, direct_entry.value);
                } else {
                    storage.append(direct_entry.row + 1, direct_entry.value);
                }
            }
        }
    };

    struct DirectSparseColumnEmitter {
        LocalCooStorage* storage = nullptr;
        std::size_t* entry_count = nullptr;
        std::optional<std::span<const int>> row_to_reduced;
        FusedParityEmissionStats* parity_stats = nullptr;
        const std::vector<signed char>* row_sector = nullptr;
        signed char column_sector = 0;

        void operator()(int row, double value) const
        {
            if (parity_stats != nullptr &&
                !parity_stats->retain(row, column_sector, value, *row_sector)) {
                return;
            }
            if (row_to_reduced) {
                const int reduced_row =
                    (*row_to_reduced)[static_cast<std::size_t>(row)];
                if (reduced_row < 0)
                    KADATH_THROW("Jacobian column engine emitted an unselected row");
                storage->append(reduced_row + 1, value);
            } else {
                storage->append(row + 1, value);
            }
            ++*entry_count;
        }
    };

    std::vector<int> wlane2_partner_local_index(
        static_cast<std::size_t>(local_ncols), -1);
    std::array<std::vector<int>, 3> wlane4_partner_local_index;
    if (use_wlane4) {
        for (std::vector<int>& slot : wlane4_partner_local_index)
            slot.assign(static_cast<std::size_t>(local_ncols), -1);
    }
    std::array<std::vector<int>, 7> wlane8_partner_local_index;
    if (use_wlane8) {
        for (std::vector<int>& slot : wlane8_partner_local_index)
            slot.assign(static_cast<std::size_t>(local_ncols), -1);
    }
    std::array<std::vector<int>, 31> wlane32_partner_local_index;
    if (use_wlane32) {
        for (std::vector<int>& slot : wlane32_partner_local_index)
            slot.assign(static_cast<std::size_t>(local_ncols), -1);
    }
    std::array<std::vector<int>, 15> wlane16_partner_local_index;
    if (use_wlane16) {
        for (std::vector<int>& slot : wlane16_partner_local_index)
            slot.assign(static_cast<std::size_t>(local_ncols), -1);
    }
    local_wlane2_eligible_columns = 0;
    local_wlane2_ideal_pairable_columns = 0;
    local_wlane4_ideal_quartetable_columns = 0;
    local_wlane8_ideal_octetable_columns = 0;
    local_wlane16_ideal_hexadectetable_columns = 0;
    local_wlane32_ideal_triacontadyadable_columns = 0;
    // Lifted to outer scope so the clustered-visit-order builder below can
    // re-use it. When the W-lane setup is skipped (metadata not ready or
    // all W-lanes disabled), this stays empty -> visit_order falls back
    // to ascending local_col.
    std::map<WLane2PairingBucket, std::vector<int>> local_columns_by_bucket;
    std::vector<std::pair<int, int>> local_variable_domain_columns;
    jacobian_assembler_detail::VariableDomainOwnerBuckets
        local_variable_domain_columns_by_owner;
    if ((use_wlane2 || use_wlane4 || use_wlane8 || use_wlane16 || use_wlane32 ||
         use_variable_domain_wlane2) && wlane2_metadata_ready) {
        for (int local_index = 0; local_index < local_ncols; ++local_index) {
            const int column = global_column_for_local(local_index);
            if (direct_column_has_entries(column))
                continue;
            const ColumnMetadata& metadata =
                wlane2_column_metadata[static_cast<std::size_t>(column)];
            if (use_variable_domain_wlane2 &&
                metadata.column_class == ColumnClass::VarDomain) {
                local_variable_domain_columns.emplace_back(local_index, column);
                continue;
            }
            if (!column_can_use_wlane2_bucket(metadata))
                continue;
            ++local_wlane2_eligible_columns;
            local_columns_by_bucket[make_wlane2_pairing_bucket(metadata)].push_back(local_index);
        }
        for (const auto& bucket_columns : local_columns_by_bucket) {
            const std::vector<int>& local_indices = bucket_columns.second;
            local_wlane2_ideal_pairable_columns +=
                (static_cast<long long>(local_indices.size()) / 2LL) * 2LL;
            std::size_t index = 0;
            if (use_wlane32) {
                local_wlane32_ideal_triacontadyadable_columns +=
                    (static_cast<long long>(local_indices.size()) / 32LL) * 32LL;
                while (index + 31 < local_indices.size()) {
                    const int leader_local_index = local_indices[index];
                    for (int lane = 1; lane < 32; ++lane) {
                        wlane32_partner_local_index[static_cast<std::size_t>(lane - 1)]
                            [static_cast<std::size_t>(leader_local_index)] =
                            local_indices[index + static_cast<std::size_t>(lane)];
                    }
                    index += 32;
                }
            }
            if (use_wlane16) {
                local_wlane16_ideal_hexadectetable_columns +=
                    (static_cast<long long>(local_indices.size()) / 16LL) * 16LL;
                while (index + 15 < local_indices.size()) {
                    const int leader_local_index = local_indices[index];
                    for (int lane = 1; lane < 16; ++lane) {
                        wlane16_partner_local_index[static_cast<std::size_t>(lane - 1)]
                            [static_cast<std::size_t>(leader_local_index)] =
                            local_indices[index + static_cast<std::size_t>(lane)];
                    }
                    index += 16;
                }
            }
            if (use_wlane8) {
                local_wlane8_ideal_octetable_columns +=
                    (static_cast<long long>(local_indices.size()) / 8LL) * 8LL;
                while (index + 7 < local_indices.size()) {
                    const int leader_local_index = local_indices[index];
                    for (int lane = 1; lane < 8; ++lane) {
                        wlane8_partner_local_index[static_cast<std::size_t>(lane - 1)]
                            [static_cast<std::size_t>(leader_local_index)] =
                            local_indices[index + static_cast<std::size_t>(lane)];
                    }
                    index += 8;
                }
            }
            if (use_wlane4) {
                const std::size_t quartet_remainder = local_indices.size() - index;
                local_wlane4_ideal_quartetable_columns +=
                    (static_cast<long long>(quartet_remainder) / 4LL) * 4LL;
                while (index + 3 < local_indices.size()) {
                    const int leader_local_index = local_indices[index];
                    wlane4_partner_local_index[0]
                        [static_cast<std::size_t>(leader_local_index)] =
                        local_indices[index + 1];
                    wlane4_partner_local_index[1]
                        [static_cast<std::size_t>(leader_local_index)] =
                        local_indices[index + 2];
                    wlane4_partner_local_index[2]
                        [static_cast<std::size_t>(leader_local_index)] =
                        local_indices[index + 3];
                    index += 4;
                }
            }
            if (use_wlane2) {
                for (; index + 1 < local_indices.size(); index += 2) {
                    const int first_local_index = local_indices[index];
                    const int second_local_index = local_indices[index + 1];
                    wlane2_partner_local_index[static_cast<std::size_t>(first_local_index)] =
                        second_local_index;
                    wlane2_partner_local_index[static_cast<std::size_t>(second_local_index)] =
                        first_local_index;
                }
            }
        }
        if (use_variable_domain_wlane2) {
            local_variable_domain_columns_by_owner =
                jacobian_assembler_detail::bucket_variable_domain_columns_by_owner(
                    local_variable_domain_columns, wlane2_column_metadata,
                    first_variable_domain_owner_parameter_count);
        }
        for (const auto& owner_columns : local_variable_domain_columns_by_owner) {
            const std::vector<int>& local_indices = owner_columns.second;
            local_wlane2_eligible_columns += static_cast<long long>(local_indices.size());
            local_wlane2_ideal_pairable_columns +=
                (static_cast<long long>(local_indices.size()) / 2LL) * 2LL;
            std::size_t index = 0;
            if (use_wlane32 && variable_domain_max_width >= 32) {
                local_wlane32_ideal_triacontadyadable_columns +=
                    (static_cast<long long>(local_indices.size()) / 32LL) * 32LL;
                while (index + 31 < local_indices.size()) {
                    const int leader = local_indices[index];
                    for (int lane = 1; lane < 32; ++lane) {
                        wlane32_partner_local_index[static_cast<std::size_t>(lane - 1)]
                            [static_cast<std::size_t>(leader)] =
                            local_indices[index + static_cast<std::size_t>(lane)];
                    }
                    index += 32;
                }
            }
            if (use_wlane16 && variable_domain_max_width >= 16) {
                const std::size_t remainder = local_indices.size() - index;
                local_wlane16_ideal_hexadectetable_columns +=
                    (static_cast<long long>(remainder) / 16LL) * 16LL;
                while (index + 15 < local_indices.size()) {
                    const int leader = local_indices[index];
                    for (int lane = 1; lane < 16; ++lane) {
                        wlane16_partner_local_index[static_cast<std::size_t>(lane - 1)]
                            [static_cast<std::size_t>(leader)] =
                            local_indices[index + static_cast<std::size_t>(lane)];
                    }
                    index += 16;
                }
            }
            if (use_wlane8 && variable_domain_max_width >= 8) {
                const std::size_t remainder = local_indices.size() - index;
                local_wlane8_ideal_octetable_columns +=
                    (static_cast<long long>(remainder) / 8LL) * 8LL;
                while (index + 7 < local_indices.size()) {
                    const int leader = local_indices[index];
                    for (int lane = 1; lane < 8; ++lane) {
                        wlane8_partner_local_index[static_cast<std::size_t>(lane - 1)]
                            [static_cast<std::size_t>(leader)] =
                            local_indices[index + static_cast<std::size_t>(lane)];
                    }
                    index += 8;
                }
            }
            if (use_wlane4 && variable_domain_max_width >= 4) {
                const std::size_t remainder = local_indices.size() - index;
                local_wlane4_ideal_quartetable_columns +=
                    (static_cast<long long>(remainder) / 4LL) * 4LL;
                while (index + 3 < local_indices.size()) {
                    const int leader = local_indices[index];
                    for (int lane = 1; lane < 4; ++lane) {
                        wlane4_partner_local_index[static_cast<std::size_t>(lane - 1)]
                            [static_cast<std::size_t>(leader)] =
                            local_indices[index + static_cast<std::size_t>(lane)];
                    }
                    index += 4;
                }
            }
            if (use_wlane2) {
                for (; index + 1 < local_indices.size(); index += 2) {
                    const int first = local_indices[index];
                    const int second = local_indices[index + 1];
                    wlane2_partner_local_index[static_cast<std::size_t>(first)] = second;
                    wlane2_partner_local_index[static_cast<std::size_t>(second)] = first;
                }
            }
        }
    }

    // Step 3 of the clustered-visit-order stack: build visit_order so the
    // column loop walks ordinary buckets and variable-domain owner buckets,
    // then non-bucketable columns in ascending local-col. Combined with step 2's
    // explicit loc_jcn emission, the col-ascending loc_irn invariant is no
    // longer required, so the loop is free to visit in any order.
    visit_order.clear();
    visit_order.reserve(static_cast<std::size_t>(local_ncols));
    {
        std::vector<bool> queued(static_cast<std::size_t>(local_ncols), false);
        for (const auto& bucket_columns : local_columns_by_bucket) {
            for (int idx : bucket_columns.second) {
                visit_order.push_back(idx);
                queued[static_cast<std::size_t>(idx)] = true;
            }
        }
        for (const auto& owner_columns : local_variable_domain_columns_by_owner) {
            for (int idx : owner_columns.second) {
                visit_order.push_back(idx);
                queued[static_cast<std::size_t>(idx)] = true;
            }
        }
        for (int idx = 0; idx < local_ncols; ++idx) {
            if (!queued[static_cast<std::size_t>(idx)])
                visit_order.push_back(idx);
        }

        // Diagnostic: per-column W-lane sweep assignment for the sparsity/packing
        // viz (matlab/Jacobian.m, wlane_real_csv). Reproduces the packing the solver
        // ACTUALLY executes: each bucket chunked W=32 -> 16 -> 8 -> 4 -> 2 (the runtime
        // try_packed_wlane cascade, mirroring the partner-array setup loop above), with
        // bucket remainders + non-bucketable + direct-entry columns scalar (one sweep
        // each). Cross-checked against the assembler census (JACOBIAN_ASSEMBLER_
        // PROFILE): BNS res7 iter-0 -> W32=464/W16=39/W8=32/W4=54/W2=20 groups + 2371
        // scalar = 2980 sweeps, fallbacks=0 (so scheduled == executed). A flat W=32
        // model undercounts by lumping each remainder into one sweep. The legacy
        // branch is an np=1 diagnostic; the global-group branch can emit its complete
        // replicated plan at any rank count. Composes with the COO dump in one run
        // (no early exit). Dump-only -- the hot assembly path is untouched.
        const char* wlane_csv_path = std::getenv("JACOBIAN_WLANE_CSV");
        if (rank_ == 0 && wlane_csv_path != nullptr && wlane_csv_path[0] != '\0' &&
            std::string(wlane_csv_path) != "0") {
            std::vector<int> column_sweep(static_cast<std::size_t>(n), -1);
            long long next_sweep = 0;
            std::vector<int> variable_domain_widths;
            if (use_variable_domain_wlane2) {
                if (use_wlane32 && variable_domain_max_width >= 32)
                    variable_domain_widths.push_back(32);
                if (use_wlane16 && variable_domain_max_width >= 16)
                    variable_domain_widths.push_back(16);
                if (use_wlane8 && variable_domain_max_width >= 8)
                    variable_domain_widths.push_back(8);
                if (use_wlane4 && variable_domain_max_width >= 4)
                    variable_domain_widths.push_back(4);
                if (use_wlane2)
                    variable_domain_widths.push_back(2);
            }
            auto emit_variable_domain_sweeps =
                [&](const std::vector<std::pair<int, int>>& local_and_global_columns) {
                    const std::vector<std::vector<int>> groups =
                        jacobian_assembler_detail::build_variable_domain_sweep_groups(
                            local_and_global_columns, wlane2_column_metadata,
                            first_variable_domain_owner_parameter_count,
                            variable_domain_widths);
                    for (const std::vector<int>& group : groups) {
                        const int sweep = static_cast<int>(next_sweep++);
                        for (int column : group)
                            column_sweep[static_cast<std::size_t>(column)] = sweep;
                    }
                };
            if (use_global_group_plan) {
                for (const JacobianColumnGroup& group : global_group_plan.groups) {
                    std::vector<int> non_variable_domain_columns;
                    for (int column : group.columns) {
                        if (!use_variable_domain_wlane2 ||
                            wlane2_column_metadata[static_cast<std::size_t>(column)]
                                    .column_class != ColumnClass::VarDomain) {
                            non_variable_domain_columns.push_back(column);
                        }
                    }
                    if (non_variable_domain_columns.empty())
                        continue;
                    const int sweep = static_cast<int>(next_sweep++);
                    for (int column : non_variable_domain_columns)
                        column_sweep[static_cast<std::size_t>(column)] = sweep;
                }
                for (int owner_rank = 0;
                     use_variable_domain_wlane2 && owner_rank < nproc_; ++owner_rank) {
                    std::vector<std::pair<int, int>> rank_variable_domain_columns;
                    int local_index = 0;
                    for (std::size_t group_index :
                         global_group_plan.group_indices_by_rank[
                             static_cast<std::size_t>(owner_rank)]) {
                        for (int column : global_group_plan.groups[group_index].columns) {
                            if (wlane2_column_metadata[static_cast<std::size_t>(column)]
                                    .column_class == ColumnClass::VarDomain) {
                                rank_variable_domain_columns.emplace_back(
                                    local_index++, column);
                            }
                        }
                    }
                    emit_variable_domain_sweeps(rank_variable_domain_columns);
                }
            } else {
                for (const auto& bucket_columns : local_columns_by_bucket) {
                    const std::vector<int>& local_indices = bucket_columns.second;
                    std::size_t index = 0;
                    auto emit_chunk = [&](std::size_t width) {
                        const int sweep = static_cast<int>(next_sweep++);
                        for (std::size_t lane = 0; lane < width; ++lane) {
                            const int column = global_column_for_local(
                                local_indices[index + lane]);
                            column_sweep[static_cast<std::size_t>(column)] = sweep;
                        }
                        index += width;
                    };
                    if (use_wlane32) while (index + 31 < local_indices.size()) emit_chunk(32);
                    if (use_wlane16) while (index + 15 < local_indices.size()) emit_chunk(16);
                    if (use_wlane8)  while (index + 7  < local_indices.size()) emit_chunk(8);
                    if (use_wlane4)  while (index + 3  < local_indices.size()) emit_chunk(4);
                    if (use_wlane2)  while (index + 1  < local_indices.size()) emit_chunk(2);
                    for (; index < local_indices.size(); ++index) {
                        const int column =
                            global_column_for_local(local_indices[index]);
                        column_sweep[static_cast<std::size_t>(column)] =
                            static_cast<int>(next_sweep++);
                    }
                }
                emit_variable_domain_sweeps(local_variable_domain_columns);
            }
            for (int column = 0; column < n; ++column)
                if (column_sweep[static_cast<std::size_t>(column)] < 0)
                    column_sweep[static_cast<std::size_t>(column)] =
                        static_cast<int>(next_sweep++);
            std::ofstream csv(wlane_csv_path);
            if (!csv)
                KADATH_THROW("Could not open JACOBIAN_WLANE_CSV");
            csv << "column,sweep\n";
            for (int column = 0; column < n; ++column)
                csv << column << ',' << column_sweep[static_cast<std::size_t>(column)] << '\n';
            csv.close();
        }

        // Diagnostic: per-column metadata (domain / variable / class) for the
        // block-structure figure (matlab/Jacobian.m, fig 5) -- identifies what a
        // column block is: the variable blocks, the phi corner self-block, the two
        // stars. JACOBIAN_COLMETA_CSV=<path> writes column,domain,var_name,
        // class. Metadata is globally replicated, so this remains complete
        // under either cyclic or global-group ownership.
        const char* colmeta_csv_path = std::getenv("JACOBIAN_COLMETA_CSV");
        if (rank_ == 0 && colmeta_csv_path != nullptr && colmeta_csv_path[0] != '\0' &&
            std::string(colmeta_csv_path) != "0" && wlane2_metadata_ready) {
            std::ofstream csv(colmeta_csv_path);
            if (!csv)
                KADATH_THROW("Could not open JACOBIAN_COLMETA_CSV");
            csv << "column,domain,var_name,class\n";
            for (int column = 0; column < n; ++column) {
                const ColumnMetadata& metadata =
                    wlane2_column_metadata[static_cast<std::size_t>(column)];
                csv << column << ',' << metadata.domain << ',' << metadata.var_name
                    << ',' << column_class_name(metadata.column_class) << '\n';
            }
            csv.close();
        }
    }

    std::vector<int> visit_position(static_cast<std::size_t>(local_ncols), -1);
    for (int order_idx = 0; order_idx < local_ncols; ++order_idx) {
        visit_position[static_cast<std::size_t>(
            visit_order[static_cast<std::size_t>(order_idx)])] = order_idx;
    }
    JacobianTaskClaims column_claims(static_cast<std::size_t>(local_ncols));

    // Width-parameterized packed-column dispatch. One body for all W-lanes:
    //   * gather the N-1 partner local indices (caller-supplied);
    //   * eligibility-check every partner against (local_col, local_ncols);
    //   * require the group to be contiguous in visit_order, then emit every
    //     lane directly into the rank-local COO store in engine lane order;
    //   * call the packed engine via the caller-supplied callback, which
    //     forwards the std::array signature for W8/W4 and adapts the distinct
    //     (j, partner, drop_tol, emit_leader, emit_partner) signature for W2;
    //   * on success, retain only each lane's nnz count for the later jcn
    //     stream, bump the width-specific counters, and mark kind=2.
    // Returns true when the packed group was emitted, false when ineligible
    // or the engine declined (the caller's
    // cascade then falls through to the next-narrower width / scalar path).
    auto try_packed_wlane =
        [&]<int N>(int packed_local_col, int packed_j,
                   const std::array<int, N - 1>& partner_local_indices,
                   long long& width_group_counter,
                   long long& width_column_counter,
                   long long& width_fallback_counter,
                   int& packed_emitter_kind,
                   auto&& invoke_engine) -> bool {
            static_assert(N >= 2);
            // Eligibility check FIRST. The cascade dispatches W8/W4 unconditionally
            // (a quartetable or pairable leader is still routed through the W8 then
            // W4 attempts), so the wider-width partner slots arrive at the -1
            // sentinel for any leader that is not octetable/quartetable. Bail
            // cleanly to the next-narrower width before the structural asserts
            // below, which assume real (non-sentinel) partners; evaluating
            // `partner_local_indices[0] > packed_local_col` on -1 would abort an
            // assertions-enabled build (debug / asan-ubsan presets keep asserts
            // live -- NDEBUG is appended only to the Release flags).
            for (int lane = 1; lane < N; ++lane) {
                const int partner =
                    partner_local_indices[static_cast<std::size_t>(lane - 1)];
                if (!(partner > packed_local_col && partner < local_ncols))
                    return false;
            }
            const int packed_visit_position =
                visit_position[static_cast<std::size_t>(packed_local_col)];
            for (int lane = 1; lane < N; ++lane) {
                const int partner =
                    partner_local_indices[static_cast<std::size_t>(lane - 1)];
                if (packed_visit_position + lane >= local_ncols ||
                    visit_order[static_cast<std::size_t>(
                        packed_visit_position + lane)] != partner ||
                    column_claims.state(static_cast<std::size_t>(partner)) !=
                        JacobianTaskState::Pending) {
                    return false;
                }
            }
            // Invariant on the surviving (real) partners: strictly ascending and
            // every partner > the packed leader column. Direct lane emission
            // and owned-global-column lookup assume this monotone ordering.
            assert(partner_local_indices[0] > packed_local_col &&
                   "try_packed_wlane: first partner must exceed leader column");
            for (int lane = 2; lane < N; ++lane)
                assert(partner_local_indices[static_cast<std::size_t>(lane - 1)] >
                           partner_local_indices[static_cast<std::size_t>(lane - 2)] &&
                       "try_packed_wlane: partner local indices must be strictly ascending");
            std::string failure_reason;
            std::array<std::size_t, N> lane_entry_counts{};
            std::array<DirectSparseColumnEmitter, N> lane_storage_emitters{};
            std::array<int, N> columns{};
            std::array<int, N> claimed_local_columns{};
            columns[0] = packed_j;
            claimed_local_columns[0] = packed_local_col;
            for (int lane = 1; lane < N; ++lane)
            {
                const int partner_local =
                    partner_local_indices[static_cast<std::size_t>(lane - 1)];
                columns[static_cast<std::size_t>(lane)] =
                    global_column_for_local(partner_local);
                claimed_local_columns[static_cast<std::size_t>(lane)] =
                    partner_local;
            }
            const std::span<const int, N> claimed_group{claimed_local_columns};
            if (!column_claims.try_claim(claimed_group))
                return false;
            for (int lane = 0; lane < N; ++lane) {
                const int lane_column = columns[static_cast<std::size_t>(lane)];
                lane_storage_emitters[static_cast<std::size_t>(lane)].storage =
                    &local_storage_for_column(lane_column);
                lane_storage_emitters[static_cast<std::size_t>(lane)].entry_count =
                    &lane_entry_counts[static_cast<std::size_t>(lane)];
                lane_storage_emitters[static_cast<std::size_t>(lane)].row_to_reduced =
                    row_map_for_column(lane_column);
                lane_storage_emitters[static_cast<std::size_t>(lane)].parity_stats =
                    fused_emission_active ? &fused_parity_stats : nullptr;
                lane_storage_emitters[static_cast<std::size_t>(lane)].row_sector =
                    fused_row_sector;
                lane_storage_emitters[static_cast<std::size_t>(lane)].column_sector =
                    fused_emission_active
                        ? fused_column_sector(
                              lane_column)
                        : 0;
            }
            // SparseColumnEmitter has no default ctor, so the lane array must
            // be brace-initialized with all N elements at once. Expand the
            // index pack over stable lane_storage_emitters. The packed engine
            // scans lanes 0..N-1, so their direct appends preserve visit_order
            // without a deferred partner copy.
            auto make_lane_emitter = [&](std::size_t lane) -> SparseColumnEmitter {
                return SparseColumnEmitter{lane_storage_emitters[lane]};
            };
            std::array<SparseColumnEmitter, N> emitters =
                [&]<std::size_t... Lane>(std::index_sequence<Lane...>) {
                    return std::array<SparseColumnEmitter, N>{
                        make_lane_emitter(Lane)...};
                }(std::make_index_sequence<N>{});
            const FusedParityEmissionStats parity_stats_before = fused_parity_stats;
            bool emitted = false;
            try {
                emitted = invoke_engine(columns, emitters, failure_reason);
            } catch (...) {
                const bool released = column_claims.release(claimed_group);
                assert(released &&
                       "packed Jacobian claim must roll back after an exception");
                (void)released;
                throw;
            }
            if (!emitted) {
                const bool released = column_claims.release(claimed_group);
                if (jacobian_packed_retained_entry_count(
                        lane_entry_counts) != 0) {
                    KADATH_THROW(
                        "packed Jacobian engine emitted COO entries before declining");
                }
                fused_parity_stats = parity_stats_before;
                if (!released)
                    KADATH_THROW(
                        "packed Jacobian claim rollback lost group ownership");
                ++width_fallback_counter;
                return false;
            }
            set_local_column_nnz(
                packed_local_col,
                static_cast<long long>(lane_entry_counts[0]));
            for (int lane = 1; lane < N; ++lane) {
                const int partner =
                    partner_local_indices[static_cast<std::size_t>(lane - 1)];
                set_local_column_nnz(
                    partner,
                    static_cast<long long>(
                        lane_entry_counts[static_cast<std::size_t>(lane)]));
            }
            if (!column_claims.complete(claimed_group))
                KADATH_THROW(
                    "packed Jacobian claim completion lost group ownership");
            local_columns_accounted += N;
            ++width_group_counter;
            width_column_counter += N;
            packed_emitter_kind = 2;
            return true;
        };

    for (int order_idx = 0; order_idx < local_ncols; ++order_idx) {
        local_col = visit_order[static_cast<std::size_t>(order_idx)];
        const int j = global_column_for_local(local_col);
        LocalCooStorage& column_storage = local_storage_for_column(j);
        const std::optional<std::span<const int>> column_row_map =
            row_map_for_column(j);
        const std::size_t before = column_storage.size();
        const std::array<int, 1> scalar_group{local_col};
        const std::span<const int, 1> scalar_group_view{scalar_group};
        bool scalar_claimed = false;
        bool emitted_direct_singleton = false;
        int column_profile_emitter_kind = 0;
        if (column_claims.state(static_cast<std::size_t>(local_col)) !=
            JacobianTaskState::Pending) {
            column_profile_emitter_kind = 2;
        } else if (direct_column_has_entries(j)) {
            if (!column_claims.try_claim(scalar_group_view)) {
                column_profile_emitter_kind = 2;
            } else {
                scalar_claimed = true;
                emitted_direct_singleton = true;
                column_profile_emitter_kind = 1;
                ++local_direct_singleton_columns;
                emit_direct_column(j);
            }
        }
        if (!emitted_direct_singleton && column_profile_emitter_kind == 0) {
            if (wlane2_metadata_ready) {
                const ColumnMetadata& cur_meta =
                    wlane2_column_metadata[static_cast<std::size_t>(j)];
                if (per_bucket_der_abs_flush_enabled &&
                    prev_bucket_valid &&
                    cur_meta.domain != prev_bucket_domain) {
                    reset_val_domain_der_abs_cache();
                }
                prev_bucket_domain = cur_meta.domain;
                prev_bucket_valid = true;
            }
            if (column_profile_emitter_kind == 0 && use_wlane32) {
                std::array<int, 31> partners{};
                for (int lane = 1; lane < 32; ++lane)
                    partners[static_cast<std::size_t>(lane - 1)] =
                        wlane32_partner_local_index[static_cast<std::size_t>(lane - 1)]
                            [static_cast<std::size_t>(local_col)];
                try_packed_wlane.template operator()<32>(
                    local_col, j, partners, local_wlane32_triacontadyads,
                    local_wlane32_columns, local_wlane32_fallbacks,
                    column_profile_emitter_kind,
                    [&](std::array<int, 32>& cols,
                        std::array<SparseColumnEmitter, 32>& emitters,
                        std::string& fr) {
                        return system_.do_cols_J_wlane32_sparse(
                            column_workspace_, cols, drop_tol, emitters, fr,
                            selected_rows);
                    });
            }

            if (column_profile_emitter_kind == 0 && use_wlane16) {
                std::array<int, 15> partners{};
                for (int lane = 1; lane < 16; ++lane)
                    partners[static_cast<std::size_t>(lane - 1)] =
                        wlane16_partner_local_index[static_cast<std::size_t>(lane - 1)]
                            [static_cast<std::size_t>(local_col)];
                try_packed_wlane.template operator()<16>(
                    local_col, j, partners, local_wlane16_hexadectets,
                    local_wlane16_columns, local_wlane16_fallbacks,
                    column_profile_emitter_kind,
                    [&](std::array<int, 16>& cols,
                        std::array<SparseColumnEmitter, 16>& emitters,
                        std::string& fr) {
                        return system_.do_cols_J_wlane16_sparse(
                            column_workspace_, cols, drop_tol, emitters, fr,
                            selected_rows);
                    });
            }

            if (column_profile_emitter_kind == 0 && use_wlane8) {
                std::array<int, 7> partners{};
                for (int lane = 1; lane < 8; ++lane)
                    partners[static_cast<std::size_t>(lane - 1)] =
                        wlane8_partner_local_index[static_cast<std::size_t>(lane - 1)]
                            [static_cast<std::size_t>(local_col)];
                try_packed_wlane.template operator()<8>(
                    local_col, j, partners, local_wlane8_octets,
                    local_wlane8_columns, local_wlane8_fallbacks,
                    column_profile_emitter_kind,
                    [&](std::array<int, 8>& cols,
                        std::array<SparseColumnEmitter, 8>& emitters,
                        std::string& fr) {
                        return system_.do_cols_J_wlane8_sparse(
                            column_workspace_, cols, drop_tol, emitters, fr,
                            selected_rows);
                    });
            }

            if (column_profile_emitter_kind == 0 && use_wlane4) {
                const std::array<int, 3> partners = {
                    wlane4_partner_local_index[0][static_cast<std::size_t>(local_col)],
                    wlane4_partner_local_index[1][static_cast<std::size_t>(local_col)],
                    wlane4_partner_local_index[2][static_cast<std::size_t>(local_col)],
                };
                try_packed_wlane.template operator()<4>(
                    local_col, j, partners, local_wlane4_quartets,
                    local_wlane4_columns, local_wlane4_fallbacks,
                    column_profile_emitter_kind,
                    [&](std::array<int, 4>& cols,
                        std::array<SparseColumnEmitter, 4>& emitters,
                        std::string& fr) {
                        return system_.do_cols_J_wlane4_sparse(
                            column_workspace_, cols, drop_tol, emitters, fr,
                            selected_rows);
                    });
            }

            if (column_profile_emitter_kind == 0) {
                const std::array<int, 1> partners = {
                    wlane2_partner_local_index[static_cast<std::size_t>(local_col)]};
                try_packed_wlane.template operator()<2>(
                    local_col, j, partners, local_wlane2_pairs,
                    local_wlane2_columns, local_wlane2_fallbacks,
                    column_profile_emitter_kind,
                    [&](std::array<int, 2>& cols,
                        std::array<SparseColumnEmitter, 2>& emitters,
                        std::string& fr) {
                        return system_.do_cols_J_wlane2_sparse(
                            column_workspace_, cols, drop_tol, emitters, fr,
                            selected_rows);
                    });
            }

            if (column_profile_emitter_kind == 0) {
                if (!column_claims.try_claim(scalar_group_view)) {
                    column_profile_emitter_kind = 2;
                } else {
                    scalar_claimed = true;
                    system_.do_col_J_sparse(
                        column_workspace_, j, drop_tol,
                        [&](int row, double value) {
                            if (fused_emission_active &&
                                !fused_parity_stats.retain(
                                    row, fused_column_sector(j), value,
                                    *fused_row_sector)) {
                                return;
                            }
                            if (column_row_map) {
                                const int reduced_row = (*column_row_map)[
                                    static_cast<std::size_t>(row)];
                                if (reduced_row < 0) {
                                    KADATH_THROW(
                                        "Jacobian column engine emitted an unselected row");
                                }
                                column_storage.append(reduced_row + 1, value);
                            } else {
                                column_storage.append(row + 1, value);
                            }
                        },
                        selected_rows);
                }
            }
        }
        if (scalar_claimed) {
            const long long column_nnz =
                static_cast<long long>(column_storage.size() - before);
            set_local_column_nnz(local_col, column_nnz);
            ++local_columns_accounted;
            if (!column_claims.complete(scalar_group_view)) {
                KADATH_THROW(
                    "scalar Jacobian column claim lost exclusive ownership");
            }
        }
        if (!final_sample_reserve_done &&
            local_columns_accounted >= reserve_sample_plan.final_columns) {
            const std::size_t estimate =
                jacobian_assembler_detail::extrapolate_local_nnz_reserve(
                    aggregate_local_coo_size(), local_columns_accounted,
                    local_ncols,
                    dense_local_nnz_upper_bound);
            reserved_local_nnz = std::max(reserved_local_nnz, estimate);
            reserve_local_coo_entries(reserved_local_nnz);
            provisional_reserve_done = true;
            final_sample_reserve_done = true;
        } else if (!provisional_reserve_done &&
                   local_columns_accounted >= reserve_sample_plan.provisional_columns) {
            const std::size_t estimate =
                jacobian_assembler_detail::extrapolate_local_nnz_reserve(
                    aggregate_local_coo_size(), local_columns_accounted,
                    local_ncols,
                    dense_local_nnz_upper_bound);
            const std::size_t provisional =
                jacobian_assembler_detail::provisional_local_nnz_reserve(
                    aggregate_local_coo_size(), estimate);
            reserved_local_nnz = std::max(reserved_local_nnz, provisional);
            reserve_local_coo_entries(reserved_local_nnz);
            provisional_reserve_done = true;
        }
    }
    if (local_columns_accounted != local_ncols) {
        KADATH_THROW("packed Jacobian column accounting did not cover local columns");
    }
    columns_seconds = MPI_Wtime() - t_columns;
    loc_nnz = static_cast<long long>(aggregate_local_coo_size());

    if (fuse_j1_attempt && fused_emission_active) {
        double certification[2] = {fused_parity_stats.maximum_cross,
                                   fused_parity_stats.maximum_entry};
        MPI_Allreduce(MPI_IN_PLACE, certification, 2, MPI_DOUBLE, MPI_MAX,
                      comm_); // NOLINT(bugprone-casting-through-void)
        const double certified_ratio = certification[1] > 0.0
                                           ? certification[0] / certification[1]
                                           : 0.0;
        // Certify at the approximate-engage tolerance: the classic pipeline
        // itself keeps masks up to that bound (measured r15 ladder: exact-only
        // certification re-assembled 3 QE stages at 1.9-2.0x J-build cost for
        // ratios of 3.5e-12..5.8e-11 that classic would have engaged anyway).
        // The env hook remains a diagnostic override for forcing the
        // mispredict/re-assembly path in validation runs.
        static const double j1_certification_tolerance = [] {
            const char* text =
                std::getenv("JACOBIAN_FUSED_PARITY_J1_TOLERANCE");
            if (text == nullptr || text[0] == '\0')
                return jacobian_parity_approximate_cross_tolerance;
            return std::atof(text);
        }();
        const bool j1_certification_accepted =
            certified_ratio < j1_certification_tolerance;
        require_collective_flag_agreement(
            j1_certification_accepted,
            "JACOBIAN_FUSED_PARITY_J1_TOLERANCE decision", comm_, nproc_);
        if (j1_certification_accepted) {
            JacobianParityMaskState& certified_state =
                *system_.jacobian_parity_mask_state();
            certified_state.decision =
                JacobianParityMaskState::Decision::Engaged;
            certified_state.engaged_cross_ratio = certified_ratio;
            const bool certified_exact =
                certified_ratio < jacobian_parity_cross_tolerance;
            // Preserve the approximate-engagement diagnostic even though the
            // complete structural labels allow later refreshes to fuse.
            certified_state.approximate_engagement = !certified_exact;
            // Certified from a full (structurally masked) emission whose
            // accumulated maxima equal the measurement scans'; later
            // refreshes may fuse through the ordinary readiness gate.
            certified_state.unmasked_full_j_emitted = true;
            emission_plan.accept_speculative_fused_emission();
            break;
        }
        if (rank_ == 0) {
            std::cout << "Jacobian parity J1 fusion: mispredicted (ratio "
                      << certified_ratio
                      << "), re-assembling unmasked" << std::endl;
        }
        fused_emission_active = false;
        fused_verify_active = false;
        emission_plan.demote_fused_emission(
            "speculative fused emission certification failed");
        fuse_j1_mispredicted = true;
        fused_parity_stats = FusedParityEmissionStats{};
        fused_parity_verdicts.clear();
        local_coo.release();
        local_minus_coo.release();
        continue;
    }
    break;
    }
    const long long pre_mask_loc_nnz =
        fused_emission_active ? fused_parity_stats.pre_mask_entries : loc_nnz;
    if (fused_emission_active && !fused_verify_active &&
        pre_mask_loc_nnz != loc_nnz + fused_parity_stats.dropped_entries) {
        KADATH_THROW("fused parity-mask emission accounting is inconsistent");
    }
    const long long local_coo_capacity_entries =
        static_cast<long long>(local_coo.capacity_entries() +
                               local_minus_coo.capacity_entries());
    if (!use_selection_plan)
        remember_local_nnz_reserve(n, nproc_, rank_, pre_mask_loc_nnz);
    ope_der_1d_workspace_scope.reset();
    val_domain_der_abs_cache_scope.reset();
    ope_der_cache_jacobian_scope.reset();
    system_.release_do_col_J_assembly_scratch(column_workspace_);
    report_memory_mapper_phase("jacobian.post_columns_scratch_release", comm_);

    // ---- y -> -y parity mask (SPARSE_PARITY_MASK) ----
    // A y-symmetric configuration splits its unknowns into a symmetric and an
    // antisymmetric sector whose coupling blocks hold nothing but roundoff
    // (Linear_algebra/jacobian_parity_mask.hpp). Dropping those entries here,
    // rank-locally and before the gather, hands MUMPS two independent halves
    // instead of one system, and cuts the gathered COO by the same fraction.
    // The residual is untouched: the antisymmetric sector keeps its own
    // diagonal block and solves its own right-hand side.
    bool parity_mask_engaged = false;
    if ((parity_mask_requested || selection_plan_requested) &&
        !use_selection_plan && !fuse_j1_mispredicted) {
        // Storage order is visit_order, with loc_col_nnz[local column]
        // contiguous entries each -- the walk jcn_filler replays -- so a cursor
        // recovers every entry's column without a local jcn array.
        int walk_order_index = 0;
        long long walk_column_remaining = 0;
        int walk_local_column = -1;
        auto reset_column_walk = [&]() {
            walk_order_index = 0;
            walk_column_remaining = 0;
            walk_local_column = -1;
        };
        auto next_local_column = [&]() -> int {
            while (walk_column_remaining == 0) {
                if (walk_order_index >= local_ncols)
                    KADATH_THROW("parity mask: stored entries outrun the per-column counts");
                walk_local_column =
                    visit_order[static_cast<std::size_t>(walk_order_index++)];
                walk_column_remaining =
                    loc_col_nnz[static_cast<std::size_t>(walk_local_column)];
            }
            --walk_column_remaining;
            return walk_local_column;
        };

        std::shared_ptr<JacobianParityMaskState>& state_slot =
            system_.jacobian_parity_mask_state();
        if (!state_slot || state_slot->n != n) {
            state_slot = std::make_shared<JacobianParityMaskState>();
            state_slot->n = n;
            if (use_local_coo_blocks) {
                disable_jacobian_parity_mask(
                    *state_slot,
                    "JACOBIAN_LOCAL_COO_BLOCKS storage cannot compact",
                    rank_);
            } else {
                JacobianParityColumnGrading grading =
                    grade_jacobian_parity_columns(system_);
                const std::string reason =
                    jacobian_parity_column_grading_disable_reason(grading);
                if (!reason.empty()) {
                    disable_jacobian_parity_mask(*state_slot, reason, rank_);
                } else {
                    state_slot->column_sector = std::move(grading.sector);
                }
            }
        }
        JacobianParityMaskState& parity_state = *state_slot;

        if (parity_state.decision == JacobianParityMaskState::Decision::Undecided) {
            // First Jacobian of this solve: retain matrix sector masses for the
            // descriptor-unavailable fallback. Complete structural rows get
            // the first measurement; if it is not exact, asymmetric operators
            // are re-graded from matrix mass before applying the ordinary mask
            // thresholds. Only reduction eligibility remains exact-only.
            std::vector<double> row_mass_symmetric(static_cast<std::size_t>(n), 0.0);
            std::vector<double> row_mass_antisymmetric(static_cast<std::size_t>(n), 0.0);
            double maximum_entry = 0.0;
            reset_column_walk();
            local_coo.for_each_chunk(
                LocalCooStorage::block_entries,
                [&](std::span<const int> rows, std::span<const double> values) {
                    for (std::size_t entry = 0; entry < rows.size(); ++entry) {
                        const int column = global_column_for_local(next_local_column());
                        const double magnitude = std::abs(values[entry]);
                        if (magnitude > maximum_entry)
                            maximum_entry = magnitude;
                        const std::size_t row =
                            static_cast<std::size_t>(rows[entry] - 1);
                        if (parity_state.column_sector[static_cast<std::size_t>(column)] > 0)
                            row_mass_symmetric[row] += magnitude;
                        else
                            row_mass_antisymmetric[row] += magnitude;
                    }
                });
            MPI_Allreduce(MPI_IN_PLACE, row_mass_symmetric.data(), n, MPI_DOUBLE,
                          MPI_SUM, comm_); // NOLINT(bugprone-casting-through-void)
            MPI_Allreduce(MPI_IN_PLACE, row_mass_antisymmetric.data(), n,
                          MPI_DOUBLE, MPI_SUM,
                          comm_); // NOLINT(bugprone-casting-through-void)
            MPI_Allreduce(MPI_IN_PLACE, &maximum_entry, 1, MPI_DOUBLE, MPI_MAX,
                          comm_); // NOLINT(bugprone-casting-through-void)
            const JacobianParityRowPrediction structural_prediction =
                predict_jacobian_parity_rows(system_);
            JacobianParityRowGradingSelection row_grading =
                select_jacobian_parity_row_grading(
                    structural_prediction, parity_state.column_sector,
                    row_mass_symmetric, row_mass_antisymmetric);
            parity_state.row_sector = std::move(row_grading.sector);
            // Recorded, not printed: the assembler emits one combined
            // engage/drop summary line once the drop statistics exist.
            if (row_grading.source ==
                JacobianParityRowGradingSelection::Source::Structural) {
                parity_state.row_grading_source_label = "structural";
                parity_state.structural_labels_available = true;
            } else {
                parity_state.row_grading_source_label =
                    "matrix-derived fallback, " + row_grading.fallback_reason;
                parity_state.structural_labels_available = false;
            }

            const auto measure_maximum_cross = [&]() {
                double maximum_cross = 0.0;
                reset_column_walk();
                local_coo.for_each_chunk(
                    LocalCooStorage::block_entries,
                    [&](std::span<const int> rows,
                        std::span<const double> values) {
                        for (std::size_t entry = 0; entry < rows.size(); ++entry) {
                            const int column =
                                global_column_for_local(next_local_column());
                            if (jacobian_parity_entry_retained(
                                    parity_state.column_sector[
                                        static_cast<std::size_t>(column)],
                                    parity_state.row_sector[
                                        static_cast<std::size_t>(rows[entry] - 1)]))
                                continue;
                            const double magnitude = std::abs(values[entry]);
                            if (magnitude > maximum_cross)
                                maximum_cross = magnitude;
                        }
                    });
                MPI_Allreduce(MPI_IN_PLACE, &maximum_cross, 1, MPI_DOUBLE,
                              MPI_MAX,
                              comm_); // NOLINT(bugprone-casting-through-void)
                return maximum_cross;
            };
            double maximum_cross = measure_maximum_cross();
            if (regrade_jacobian_parity_rows_after_structural_measurement(
                    row_grading, maximum_cross, maximum_entry,
                    row_mass_symmetric, row_mass_antisymmetric)) {
                parity_state.row_sector = std::move(row_grading.sector);
                maximum_cross = measure_maximum_cross();
                parity_state.row_grading_source_label =
                    "matrix-derived second pass";
                parity_state.structural_labels_available = false;
            }
            decide_jacobian_parity_mask(parity_state, maximum_cross,
                                        maximum_entry, rank_);
            if (selection_plan_requested) {
                decide_jacobian_parity_reduction(
                    parity_state, maximum_cross, maximum_entry, rank_);
            }
        }

        if (!fused_emission_active)
            parity_state.unmasked_full_j_emitted = true;

        parity_mask_engaged =
            parity_mask_requested &&
            parity_state.decision == JacobianParityMaskState::Decision::Engaged;
        if (parity_mask_engaged) {
            if (fused_emission_active && !fused_verify_active) {
                double fused_maximum_cross = fused_parity_stats.maximum_cross;
                double fused_maximum_entry = fused_parity_stats.maximum_entry;
                MPI_Allreduce(MPI_IN_PLACE, &fused_maximum_cross, 1, MPI_DOUBLE,
                              MPI_MAX,
                              comm_); // NOLINT(bugprone-casting-through-void)
                MPI_Allreduce(MPI_IN_PLACE, &fused_maximum_entry, 1, MPI_DOUBLE,
                              MPI_MAX,
                              comm_); // NOLINT(bugprone-casting-through-void)
            } else {
                long long verify_cursor = 0;
                long long verify_mismatches = 0;
                int retained_local_column = -1;
                long long retained_in_column = 0;
                reset_column_walk();
                const std::size_t retained =
                    local_coo.retain_entries([&](int row_plus_one, double) {
                        const int local_column = next_local_column();
                        if (local_column != retained_local_column) {
                            if (retained_local_column >= 0) {
                                loc_col_nnz[static_cast<std::size_t>(
                                    retained_local_column)] = retained_in_column;
                            }
                            retained_local_column = local_column;
                            retained_in_column = 0;
                        }
                        const int column = global_column_for_local(local_column);
                        const bool keep = jacobian_parity_entry_retained(
                            parity_state.column_sector[
                                static_cast<std::size_t>(column)],
                            parity_state.row_sector[static_cast<std::size_t>(
                                row_plus_one - 1)]);
                        if (fused_verify_active) {
                            const std::size_t verdict_index =
                                static_cast<std::size_t>(verify_cursor++);
                            const int fused_keep =
                                verdict_index < fused_parity_verdicts.size()
                                    ? int(fused_parity_verdicts[verdict_index])
                                    : -1;
                            if (fused_keep != int(keep)) {
                                ++verify_mismatches;
                                if (verify_mismatches <= 10) {
                                    std::cerr << "FUSED-VERIFY MISMATCH rank="
                                              << rank_ << " entry=" << verdict_index
                                              << " col=" << column << " row="
                                              << row_plus_one << " retain_keep="
                                              << int(keep) << " fused_keep="
                                              << fused_keep << std::endl;
                                }
                            }
                        }
                        if (!keep)
                            return false;
                        ++retained_in_column;
                        return true;
                    });
                if (retained_local_column >= 0) {
                    loc_col_nnz[static_cast<std::size_t>(retained_local_column)] =
                        retained_in_column;
                }
                loc_nnz = static_cast<long long>(retained);
                if (fused_verify_active) {
                    long long total_mismatches = verify_mismatches;
                    MPI_Allreduce(MPI_IN_PLACE, &total_mismatches, 1,
                                  MPI_LONG_LONG, MPI_SUM, comm_);
                    if (rank_ == 0) {
                        std::cout << "FUSED-VERIFY: entries="
                                  << fused_parity_verdicts.size()
                                  << " mismatches=" << total_mismatches
                                  << std::endl;
                    }
                }
                // retain_entries shrinks the size but resize() never returns the
                // dropped capacity, so the pre-mask allocation would otherwise
                // stay resident through the whole MUMPS phase (~30% of COO RSS at
                // res15). shrink_to_fit copies into a fresh allocation and briefly
                // holds old + new, so ranks take turns instead of spiking the
                // node's memory together.
                for (int stagger = 0; stagger < nproc_; ++stagger) {
                    if (stagger == rank_)
                        local_coo.shrink_to_fit();
                    MPI_Barrier(comm_);
                }
            }
        }
    }
    if (fuse_j1_mispredicted) {
        system_.jacobian_parity_mask_state()->unmasked_full_j_emitted = true;
    }
    // Phase boundary: J-build done, cache scopes dtor'd thousands of
    // small Val_domain entries into libmalloc's heap free-list. Ask
    // the allocator to return those pages to the OS before MUMPS
    // allocates its factor workspace. No-op unless
    // RELEASE_ALLOCATOR_PAGES=1.
    release_allocator_pages();

    // ---- Gathered COO container; rank-0 buffers are allocated block-wise. ----
    AssembledJacobianCoo coo;
    coo.n = assembled_dimension;
    coo.drop_tol_used = drop_tol;
    // Both emission-time masking and the classic retain walk remove every
    // cross-sector entry. Verify mode deliberately emits cross entries.
    const std::shared_ptr<JacobianParityMaskState>& emitted_parity_state =
        system_.jacobian_parity_mask_state();
    coo.parity_sector_block_diagonal = physical_block_emission_active() ||
        (emitted_parity_state &&
         jacobian_parity_mask_emission_is_block_diagonal(
             *emitted_parity_state, n, parity_mask_engaged,
             fused_verify_active));

    // The centralized column-index array (jcn) is rebuilt per rank inside
    // jcn_filler below from each rank's local loc_col_nnz, so the global
    // per-column nnz array that an MPI_Gatherv used to assemble here
    // (all_col_nnz, with its ncol_per_rank / col_displs scaffold) was never
    // read by anyone -- dead output. Dropped. The diagnostic profile slot is
    // retained (reports 0) so the JACOBIAN_ASSEMBLER_PROFILE field
    // layout is unchanged; production behaviour is bit-identical.
    const double col_nnz_gather_seconds = 0.0;

    double nnz_gather_seconds = 0.0;
    double irn_a_gather_seconds = 0.0;
    double reconstruct_jcn_seconds = 0.0;
    constexpr int jcn_chunk_max = 4 * 1024 * 1024;  // 4M ints = 16 MB scratch
    constexpr int chunk_max = jcn_chunk_max;
    const int local_coo_chunk_max =
        use_local_coo_blocks ? static_cast<int>(LocalCooStorage::block_entries)
                             : chunk_max;
    auto gather_one_coo = [&](LocalCooStorage& storage,
                              const std::vector<long long>& column_counts,
                              const std::vector<int>* column_map,
                              int gathered_dimension,
                              AssembledJacobianCoo::IndexVector& gathered_irn,
                              AssembledJacobianCoo::IndexVector& gathered_jcn,
                              AssembledJacobianCoo::ValueVector& gathered_a) {
        const long long local_entries = static_cast<long long>(storage.size());
        long long counted_entries = 0;
        for (long long count : column_counts)
            counted_entries += count;
        bool local_coordinates_valid = gathered_dimension > 0;
        for (int local_column = 0;
             local_coordinates_valid && local_column < local_ncols;
             ++local_column) {
            if (column_counts[static_cast<std::size_t>(local_column)] == 0)
                continue;
            const int full_column = global_column_for_local(local_column);
            const int gathered_column = column_map != nullptr
                ? (*column_map)[static_cast<std::size_t>(full_column)]
                : full_column;
            local_coordinates_valid = gathered_column >= 0 &&
                gathered_column < gathered_dimension;
        }
        const int local_counts_valid =
            counted_entries == local_entries && local_coordinates_valid ? 1 : 0;
        int all_counts_valid = 0;
        MPI_Allreduce(&local_counts_valid, &all_counts_valid, 1, MPI_INT,
                      MPI_MIN, comm_);
        if (all_counts_valid == 0) {
            KADATH_THROW(
                "JacobianAssembler: block storage and per-column nnz disagree");
        }

        const double t_nnz_gather = MPI_Wtime();
        MPI_Gather(&local_entries, 1, MPI_LONG_LONG,
                   rank_ == 0 ? all_nnz64.data() : nullptr, 1,
                   MPI_LONG_LONG, 0,
                   comm_); // NOLINT(bugprone-casting-through-void)
        long long gathered_nnz = 0;
        if (rank_ == 0) {
            std::fill(displs64.begin(), displs64.end(), 0LL);
            for (int r = 0; r < nproc_; ++r) {
                displs64[static_cast<std::size_t>(r + 1)] =
                    displs64[static_cast<std::size_t>(r)] +
                    all_nnz64[static_cast<std::size_t>(r)];
            }
            gathered_nnz = displs64[static_cast<std::size_t>(nproc_)];
        }
        MPI_Bcast(&gathered_nnz, 1, MPI_LONG_LONG, 0,
                  comm_); // NOLINT(bugprone-casting-through-void)
        nnz_gather_seconds += MPI_Wtime() - t_nnz_gather;

        auto fill_columns = [&](std::span<int> destination, int& order_index,
                                long long& column_remaining,
                                int& gathered_column) {
            long long filled = 0;
            while (filled < static_cast<long long>(destination.size())) {
                if (column_remaining == 0) {
                    while (order_index < local_ncols &&
                           column_remaining == 0) {
                        const int local_column = visit_order[
                            static_cast<std::size_t>(order_index++)];
                        column_remaining = column_counts[
                            static_cast<std::size_t>(local_column)];
                        if (column_remaining == 0)
                            continue;
                        const int full_column =
                            global_column_for_local(local_column);
                        gathered_column = column_map != nullptr
                            ? (*column_map)[static_cast<std::size_t>(
                                  full_column)]
                            : full_column;
                        if (gathered_column < 0) {
                            KADATH_THROW(
                                "Jacobian block emitted a column outside its selection plan");
                        }
                    }
                    if (column_remaining == 0)
                        break;
                }
                const long long remaining =
                    static_cast<long long>(destination.size()) - filled;
                const long long take =
                    std::min(column_remaining, remaining);
                std::fill_n(destination.data() + filled,
                            static_cast<std::size_t>(take),
                            gathered_column + 1);
                filled += take;
                column_remaining -= take;
            }
            if (filled != static_cast<long long>(destination.size())) {
                KADATH_THROW(
                    "Jacobian block column counts underrun its jcn chunk");
            }
        };

        gathered_irn.resize(rank_ == 0
                                ? static_cast<std::size_t>(gathered_nnz)
                                : 0u);
        gathered_a.resize(rank_ == 0
                              ? static_cast<std::size_t>(gathered_nnz)
                              : 0u);
        const double t_irn_a_gather = MPI_Wtime();
        if (rank_ == 0) {
            if (local_entries > 0) {
                storage.copy_to(
                    gathered_irn.data() +
                        static_cast<std::size_t>(displs64[0]),
                    gathered_a.data() +
                        static_cast<std::size_t>(displs64[0]));
            }
            storage.release();
            for (int r = 1; r < nproc_; ++r) {
                long long remaining = all_nnz64[static_cast<std::size_t>(r)];
                long long offset = displs64[static_cast<std::size_t>(r)];
                while (remaining > 0) {
                    const int chunk = static_cast<int>(std::min<long long>(
                        remaining, local_coo_chunk_max));
                    MPI_Recv(gathered_irn.data() +
                                 static_cast<std::size_t>(offset),
                             chunk, MPI_INT, r, 701, comm_,
                             MPI_STATUS_IGNORE);
                    MPI_Recv(gathered_a.data() +
                                 static_cast<std::size_t>(offset),
                             chunk, MPI_DOUBLE, r, 702, comm_,
                             MPI_STATUS_IGNORE);
                    offset += chunk;
                    remaining -= chunk;
                }
            }
        } else {
            storage.for_each_chunk(
                static_cast<std::size_t>(local_coo_chunk_max),
                [&](std::span<const int> rows,
                    std::span<const double> values) {
                    const int chunk = static_cast<int>(rows.size());
                    MPI_Send(rows.data(), chunk, MPI_INT, 0, 701, comm_);
                    MPI_Send(values.data(), chunk, MPI_DOUBLE, 0, 702, comm_);
                });
            storage.release();
        }
        MPI_Barrier(comm_);
        irn_a_gather_seconds += MPI_Wtime() - t_irn_a_gather;

        const double t_reconstruct_jcn = MPI_Wtime();
        std::vector<int> jcn_scratch;
        int local_jcn_buffers_ready = 1;
        try {
            if (rank_ == 0) {
                gathered_jcn.resize(static_cast<std::size_t>(gathered_nnz));
            } else {
                jcn_scratch.resize(static_cast<std::size_t>(
                    std::min<long long>(local_entries, jcn_chunk_max)));
            }
        } catch (...) {
            local_jcn_buffers_ready = 0;
        }
        int all_jcn_buffers_ready = 0;
        MPI_Allreduce(&local_jcn_buffers_ready, &all_jcn_buffers_ready, 1,
                      MPI_INT, MPI_MIN, comm_);
        if (all_jcn_buffers_ready == 0) {
            KADATH_THROW(
                "JacobianAssembler: could not allocate block jcn buffers");
        }

        if (rank_ == 0) {
            if (local_entries > 0) {
                int order_index = 0;
                long long column_remaining = 0;
                int gathered_column = 0;
                fill_columns(
                    std::span<int>(gathered_jcn).subspan(
                        static_cast<std::size_t>(displs64[0]),
                        static_cast<std::size_t>(local_entries)),
                    order_index, column_remaining, gathered_column);
            }
            for (int r = 1; r < nproc_; ++r) {
                long long remaining = all_nnz64[static_cast<std::size_t>(r)];
                long long offset = displs64[static_cast<std::size_t>(r)];
                while (remaining > 0) {
                    const int chunk = static_cast<int>(std::min<long long>(
                        remaining, chunk_max));
                    MPI_Recv(gathered_jcn.data() +
                                 static_cast<std::size_t>(offset),
                             chunk, MPI_INT, r, 703, comm_,
                             MPI_STATUS_IGNORE);
                    offset += chunk;
                    remaining -= chunk;
                }
            }
        } else {
            int order_index = 0;
            long long column_remaining = 0;
            int gathered_column = 0;
            long long remaining = local_entries;
            while (remaining > 0) {
                const int chunk = static_cast<int>(std::min<long long>(
                    remaining, chunk_max));
                fill_columns(
                    std::span<int>(jcn_scratch).first(
                        static_cast<std::size_t>(chunk)),
                    order_index, column_remaining, gathered_column);
                MPI_Send(jcn_scratch.data(), chunk, MPI_INT, 0, 703, comm_);
                remaining -= chunk;
            }
        }
        reconstruct_jcn_seconds += MPI_Wtime() - t_reconstruct_jcn;
        return gathered_nnz;
    };

    if (physical_block_emission_active()) {
        const std::size_t block_count =
            physical_selection_block ? 1u : 2u;
        coo.parity_blocks.reserve(block_count);
        for (std::size_t block_index = 0; block_index < block_count;
             ++block_index) {
            AssembledJacobianCooBlock block;
            block.parity_label = block_index == 0 ? +1 : -1;
            block.selection_plan = physical_block_plans[block_index];
            block.n = static_cast<int>(
                block.selection_plan->selected_columns().size());
            LocalCooStorage& storage = block_index == 0
                ? local_coo
                : local_minus_coo;
            const std::vector<long long>& column_counts = block_index == 0
                ? loc_col_nnz
                : minus_loc_col_nnz;
            const std::vector<int>* block_column_map =
                physical_selection_block
                    ? &column_to_reduced
                    : &physical_column_to_reduced[block_index];
            block.nnz = gather_one_coo(
                storage, column_counts, block_column_map, block.n, block.irn,
                block.jcn, block.a);
            coo.nnz += block.nnz;
            coo.parity_blocks.push_back(std::move(block));
        }
    } else {
        const std::vector<int>* legacy_column_map =
            use_selection_plan ? &column_to_reduced : nullptr;
        coo.nnz = gather_one_coo(
            local_coo, loc_col_nnz, legacy_column_map, assembled_dimension,
            coo.irn, coo.jcn, coo.a);
    }
    nnz = coo.nnz;
    report_memory_mapper_phase("jacobian.post_local_coo_release", comm_);
    report_memory_mapper_phase("jacobian.post_jcn_reconstruction", comm_);
    const bool hash_exit_after_first =
        env_flag_enabled("JACOBIAN_COO_BYTE_HASH_EXIT_AFTER_FIRST", true);
    require_collective_flag_agreement(coo_hash_active,
                                      "JACOBIAN_COO_BYTE_HASH activation",
                                      comm_, nproc_);
    require_collective_flag_agreement(
        canonical_coo_hash_active,
        "JACOBIAN_CANONICAL_COO_BYTE_HASH activation", comm_, nproc_);
    if (coo_hash_active || canonical_coo_hash_active) {
        require_collective_flag_agreement(
            hash_exit_after_first,
            "JACOBIAN_COO_BYTE_HASH_EXIT_AFTER_FIRST", comm_, nproc_);
    }

    int diagnostics_ok = 1;
    if (rank_ == 0) {
        try {
            if (selected_row_entries_active) {
                const std::vector<int> selected_rows =
                    env_int_list_value("JACOBIAN_SELECTED_ROWS");
                std::set<int> wanted_rows(selected_rows.begin(), selected_rows.end());
                std::ofstream csv(selected_row_entries_path);
                if (!csv)
                    KADATH_THROW("Could not open JACOBIAN_SELECTED_ROW_ENTRIES_CSV");
                csv << std::setprecision(17);
                csv << "row,column,value\n";
                for (long long entry = 0; entry < coo.nnz; ++entry) {
                    const auto index = static_cast<std::size_t>(entry);
                    const int row = coo.irn[index] - 1;
                    if (!wanted_rows.empty() && wanted_rows.count(row) == 0)
                        continue;
                    csv << row << ',' << (coo.jcn[index] - 1) << ','
                        << coo.a[index] << '\n';
                }
                csv.close();
                if (!csv)
                    KADATH_THROW("Could not finish JACOBIAN_SELECTED_ROW_ENTRIES_CSV");
            }

            // Diagnostic: y=0 reflection-parity mass (JACOBIAN_PARITY_MASS
            // =<report path>).  Default off; reads the centralized COO only.
            if (parity_mass_active) {
                jacobian_parity_mass_report(system_, coo, parity_mass_path);
            }

            if (coo_hash_active && canonical_coo_hash_active &&
                std::string(coo_hash_path) == canonical_coo_hash_path) {
                KADATH_THROW("Raw and canonical Jacobian COO hashes require distinct output paths");
            }
            if (coo_hash_active) {
                const std::uint64_t hash = jacobian_coo_bit_hash(
                    coo.n, coo.nnz, coo.drop_tol_used, coo.irn.data(),
                    coo.jcn.data(), coo.a.data());
                std::ofstream out(coo_hash_path);
                if (!out)
                    KADATH_THROW("Could not open JACOBIAN_COO_BYTE_HASH output");
                out << "n,nnz,drop_tol,hash\n";
                out << coo.n << ',' << coo.nnz << ',' << std::setprecision(17)
                    << coo.drop_tol_used << ',' << hash << '\n';
                out.close();
                if (!out)
                    KADATH_THROW("Could not finish JACOBIAN_COO_BYTE_HASH output");
            }
            if (canonical_coo_hash_active) {
                const std::uint64_t hash = canonical_jacobian_coo_bit_hash(
                    coo.n, coo.nnz, coo.drop_tol_used, coo.irn.data(),
                    coo.jcn.data(), coo.a.data());
                std::ofstream out(canonical_coo_hash_path);
                if (!out)
                    KADATH_THROW(
                        "Could not open JACOBIAN_CANONICAL_COO_BYTE_HASH output");
                out << "n,nnz,drop_tol,hash\n";
                out << coo.n << ',' << coo.nnz << ',' << std::setprecision(17)
                    << coo.drop_tol_used << ',' << hash << '\n';
                out.close();
                if (!out)
                    KADATH_THROW(
                        "Could not finish JACOBIAN_CANONICAL_COO_BYTE_HASH output");
            }
        } catch (const std::exception& error) {
            diagnostics_ok = 0;
            std::cerr << "Jacobian diagnostic failed: " << error.what() << std::endl;
        } catch (...) {
            diagnostics_ok = 0;
            std::cerr << "Jacobian diagnostic failed with an unknown exception" << std::endl;
        }
    }
    MPI_Bcast(&diagnostics_ok, 1, MPI_INT, 0,
              comm_); // NOLINT(bugprone-casting-through-void)
    if (diagnostics_ok == 0)
        KADATH_THROW("Jacobian diagnostic failed on rank 0");
    long long global_direct_singleton_columns = 0;
    long long global_wlane2_pairs = 0;
    long long global_wlane2_columns = 0;
    long long global_wlane2_fallbacks = 0;
    long long global_wlane2_eligible_columns = 0;
    long long global_wlane2_ideal_pairable_columns = 0;
    long long global_wlane4_quartets = 0;
    long long global_wlane4_columns = 0;
    long long global_wlane4_fallbacks = 0;
    long long global_wlane4_ideal_quartetable_columns = 0;
    long long global_wlane32_triacontadyads = 0;
    long long global_wlane32_columns = 0;
    long long global_wlane32_fallbacks = 0;
    long long global_wlane32_ideal_triacontadyadable_columns = 0;
    long long global_wlane16_hexadectets = 0;
    long long global_wlane16_columns = 0;
    long long global_wlane16_fallbacks = 0;
    long long global_wlane16_ideal_hexadectetable_columns = 0;
    long long global_wlane8_octets = 0;
    long long global_wlane8_columns = 0;
    long long global_wlane8_fallbacks = 0;
    long long global_wlane8_ideal_octetable_columns = 0;
    if (profile) {
        MPI_Reduce(&local_direct_singleton_columns,
                   rank_ == 0 ? &global_direct_singleton_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane2_pairs,
                   rank_ == 0 ? &global_wlane2_pairs : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane2_columns,
                   rank_ == 0 ? &global_wlane2_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane2_fallbacks,
                   rank_ == 0 ? &global_wlane2_fallbacks : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane2_eligible_columns,
                   rank_ == 0 ? &global_wlane2_eligible_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane2_ideal_pairable_columns,
                   rank_ == 0 ? &global_wlane2_ideal_pairable_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane4_quartets,
                   rank_ == 0 ? &global_wlane4_quartets : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane4_columns,
                   rank_ == 0 ? &global_wlane4_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane4_fallbacks,
                   rank_ == 0 ? &global_wlane4_fallbacks : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane4_ideal_quartetable_columns,
                   rank_ == 0 ? &global_wlane4_ideal_quartetable_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane32_triacontadyads,
                   rank_ == 0 ? &global_wlane32_triacontadyads : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane32_columns,
                   rank_ == 0 ? &global_wlane32_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane32_fallbacks,
                   rank_ == 0 ? &global_wlane32_fallbacks : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane32_ideal_triacontadyadable_columns,
                   rank_ == 0 ? &global_wlane32_ideal_triacontadyadable_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane16_hexadectets,
                   rank_ == 0 ? &global_wlane16_hexadectets : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane16_columns,
                   rank_ == 0 ? &global_wlane16_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane16_fallbacks,
                   rank_ == 0 ? &global_wlane16_fallbacks : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane16_ideal_hexadectetable_columns,
                   rank_ == 0 ? &global_wlane16_ideal_hexadectetable_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane8_octets,
                   rank_ == 0 ? &global_wlane8_octets : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane8_columns,
                   rank_ == 0 ? &global_wlane8_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane8_fallbacks,
                   rank_ == 0 ? &global_wlane8_fallbacks : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
        MPI_Reduce(&local_wlane8_ideal_octetable_columns,
                   rank_ == 0 ? &global_wlane8_ideal_octetable_columns : nullptr,
                   1, MPI_LONG_LONG, MPI_SUM, 0, comm_);
    }

    if (profile) {
        constexpr int profile_fields = 25;
        const LocalCooStorageStats& local_coo_stats = local_coo.stats();
        const LocalCooStorageStats& local_minus_coo_stats =
            local_minus_coo.stats();
        const double local_profile[profile_fields] = {
            static_cast<double>(rank_),
            static_cast<double>(local_ncols),
            static_cast<double>(loc_nnz),
            direct_singleton_plan_seconds,
            static_cast<double>(reserved_local_nnz),
            reset_seconds,
            columns_seconds,
            nnz_gather_seconds,
            col_nnz_gather_seconds,
            reconstruct_jcn_seconds,
            irn_a_gather_seconds,
            MPI_Wtime() - t_total,
            static_cast<double>(use_local_coo_blocks ? 1 : 0),
            static_cast<double>(local_coo_capacity_entries),
            static_cast<double>(local_coo_stats.capacity_growth_events +
                                local_minus_coo_stats.capacity_growth_events),
            static_cast<double>(local_coo_stats.reallocations +
                                local_minus_coo_stats.reallocations),
            static_cast<double>(local_coo_stats.geometric_copy_bytes +
                                local_minus_coo_stats.geometric_copy_bytes),
            static_cast<double>(local_coo_stats.block_allocations +
                                local_minus_coo_stats.block_allocations),
            static_cast<double>(local_coo_stats.allocated_capacity_bytes +
                                local_minus_coo_stats.allocated_capacity_bytes),
            static_cast<double>(local_coo_stats.gather_copy_bytes +
                                local_minus_coo_stats.gather_copy_bytes),
            static_cast<double>(use_structural_plan_cache ? 1 : 0),
            static_cast<double>(structural_plan_access.cache_hit ? 1 : 0),
            static_cast<double>(structural_plan_access.cache_hit ? 0 : 1),
            structural_plan_access.cache_check_seconds,
            structural_plan_access.cache_miss_build_seconds};
        std::vector<double> gathered_profile;
        if (rank_ == 0) {
            gathered_profile.assign(
                static_cast<std::size_t>(nproc_) *
                    static_cast<std::size_t>(profile_fields),
                0.0);
        }
        MPI_Gather(local_profile, profile_fields, MPI_DOUBLE,
                   rank_ == 0 ? gathered_profile.data() : nullptr,
                   profile_fields, MPI_DOUBLE, 0, comm_);
        if (rank_ == 0) {
            std::cout << "JacobianAssembler: profile"
                      << " n=" << n
                      << " nnz=" << nnz
                      << " ranks=" << nproc_
                      << std::endl;
            if (use_global_group_plan) {
                const auto minmax_cost = std::minmax_element(
                    global_group_plan.estimated_cost_by_rank.begin(),
                    global_group_plan.estimated_cost_by_rank.end());
                std::cout << "JacobianAssembler: global_group_plan"
                          << " enabled=1"
                          << " groups=" << global_group_plan.groups.size()
                          << " estimated_cost_min=" << *minmax_cost.first
                          << " estimated_cost_max=" << *minmax_cost.second
                          << std::endl;
            } else {
                std::cout << "JacobianAssembler: global_group_plan enabled=0"
                          << std::endl;
            }
            std::cout << "JacobianAssembler: direct_singleton"
                      << " planned_columns=" << planned_direct_singletons
                      << " emitted_columns=" << global_direct_singleton_columns
                      << std::endl;
            double structural_cache_hits = 0.0;
            double structural_cache_misses = 0.0;
            double structural_cache_check_seconds_max = 0.0;
            double structural_cache_miss_build_seconds_max = 0.0;
            for (int r = 0; r < nproc_; ++r) {
                const std::size_t off =
                    static_cast<std::size_t>(r) *
                    static_cast<std::size_t>(profile_fields);
                structural_cache_hits += gathered_profile[off + 21];
                structural_cache_misses += gathered_profile[off + 22];
                structural_cache_check_seconds_max = std::max(
                    structural_cache_check_seconds_max,
                    gathered_profile[off + 23]);
                structural_cache_miss_build_seconds_max = std::max(
                    structural_cache_miss_build_seconds_max,
                    gathered_profile[off + 24]);
            }
            std::cout << "JacobianAssembler: structural_plan_cache"
                      << " enabled=" << (use_structural_plan_cache ? 1 : 0)
                      << " hits=" << static_cast<long long>(structural_cache_hits)
                      << " misses=" << static_cast<long long>(structural_cache_misses)
                      << " check_seconds_max="
                      << structural_cache_check_seconds_max
                      << " miss_build_seconds_max="
                      << structural_cache_miss_build_seconds_max
                      << std::endl;
            std::cout << "JacobianAssembler: wlane32"
                      << " enabled=" << (use_wlane32 ? 1 : 0)
                      << " triacontadyads=" << global_wlane32_triacontadyads
                      << " emitted_columns=" << global_wlane32_columns
                      << " fallbacks=" << global_wlane32_fallbacks
                      << " ideal_triacontadyadable_columns="
                      << global_wlane32_ideal_triacontadyadable_columns
                      << std::endl;
            std::cout << "JacobianAssembler: wlane16"
                      << " enabled=" << (use_wlane16 ? 1 : 0)
                      << " hexadectets=" << global_wlane16_hexadectets
                      << " emitted_columns=" << global_wlane16_columns
                      << " fallbacks=" << global_wlane16_fallbacks
                      << " ideal_hexadectetable_columns="
                      << global_wlane16_ideal_hexadectetable_columns
                      << std::endl;
            std::cout << "JacobianAssembler: wlane8"
                      << " enabled=" << (use_wlane8 ? 1 : 0)
                      << " octets=" << global_wlane8_octets
                      << " emitted_columns=" << global_wlane8_columns
                      << " fallbacks=" << global_wlane8_fallbacks
                      << " ideal_octetable_columns="
                      << global_wlane8_ideal_octetable_columns
                      << std::endl;
            std::cout << "JacobianAssembler: wlane4"
                      << " enabled=" << (use_wlane4 ? 1 : 0)
                      << " quartets=" << global_wlane4_quartets
                      << " emitted_columns=" << global_wlane4_columns
                      << " fallbacks=" << global_wlane4_fallbacks
                      << " ideal_quartetable_columns="
                      << global_wlane4_ideal_quartetable_columns
                      << std::endl;
            std::cout << "JacobianAssembler: wlane2"
                      << " enabled=" << (use_wlane2 ? 1 : 0)
                      << " pairs=" << global_wlane2_pairs
                      << " emitted_columns=" << global_wlane2_columns
                      << " fallbacks=" << global_wlane2_fallbacks
                      << std::endl;
            if (use_wlane8) {
                const long long missed_octetable_columns =
                    std::max(0LL, global_wlane8_ideal_octetable_columns -
                                      global_wlane8_columns);
                // True GLOBAL octetable ideal: count every same-bucket column
                // together. The gap from the per-rank ideal measures cyclic
                // ownership fragmentation; the global planner makes that gap
                // zero by assigning whole cascade groups.
                long long global_unfragmented_octetable_columns =
                    global_wlane8_ideal_octetable_columns;
                if (wlane2_metadata_ready) {
                    std::map<WLane2PairingBucket, long long> global_bucket_size;
                    for (int column = 0; column < n; ++column) {
                        const ColumnMetadata& metadata =
                            wlane2_column_metadata[static_cast<std::size_t>(column)];
                        if (!column_can_use_wlane2_bucket(metadata))
                            continue;
                        if (direct_column_has_entries(column))
                            continue;
                        ++global_bucket_size[make_wlane2_pairing_bucket(metadata)];
                    }
                    global_unfragmented_octetable_columns = 0;
                    for (const auto& kv : global_bucket_size)
                        global_unfragmented_octetable_columns += (kv.second / 8) * 8;
                }
                const long long lost_to_rank_split = use_global_group_plan
                    ? 0LL
                    : std::max(0LL, global_unfragmented_octetable_columns -
                                        global_wlane8_ideal_octetable_columns);
                std::cout << "JacobianAssembler: wlane8_coverage"
                          << " scheduled_columns=" << global_wlane8_columns
                          << " ideal_same_bucket_columns="
                          << global_wlane8_ideal_octetable_columns
                          << " missed_by_scheduler=" << missed_octetable_columns
                          << " ideal_unfragmented_columns="
                          << global_unfragmented_octetable_columns
                          << " lost_to_rank_split=" << lost_to_rank_split
                          << std::endl;
            }
            if (use_wlane2) {
                const long long missed_pairable_columns =
                    std::max(0LL, global_wlane2_ideal_pairable_columns -
                                      global_wlane2_columns);
                const long long structurally_stranded_columns =
                    std::max(0LL, global_wlane2_eligible_columns -
                                      global_wlane2_ideal_pairable_columns);
                std::cout << "JacobianAssembler: wlane2_coverage"
                          << " eligible_columns=" << global_wlane2_eligible_columns
                          << " scheduled_columns=" << global_wlane2_columns
                          << " ideal_same_bucket_columns="
                          << global_wlane2_ideal_pairable_columns
                          << " missed_by_scheduler=" << missed_pairable_columns
                          << " stranded_columns=" << structurally_stranded_columns
                          << std::endl;
            }
            for (int r = 0; r < nproc_; ++r) {
                const std::size_t off =
                    static_cast<std::size_t>(r) *
                    static_cast<std::size_t>(profile_fields);
                std::cout << "JacobianAssembler: rank_profile"
                          << " rank=" << static_cast<int>(gathered_profile[off + 0])
                          << " local_cols=" << gathered_profile[off + 1]
                          << " local_nnz=" << gathered_profile[off + 2]
                          << " direct_plan_seconds=" << gathered_profile[off + 3]
                          << " reserve_entries=" << gathered_profile[off + 4]
                          << " reset_seconds=" << gathered_profile[off + 5]
                          << " column_seconds=" << gathered_profile[off + 6]
                          << " nnz_gather_seconds=" << gathered_profile[off + 7]
                          << " col_nnz_gather_seconds=" << gathered_profile[off + 8]
                          << " reconstruct_jcn_seconds=" << gathered_profile[off + 9]
                          << " irn_a_gather_seconds=" << gathered_profile[off + 10]
                          << " total_seconds=" << gathered_profile[off + 11]
                          << std::endl;
                std::cout << "JacobianAssembler: local_coo_storage"
                          << " rank=" << static_cast<int>(gathered_profile[off + 0])
                          << " blocks_enabled="
                          << static_cast<int>(gathered_profile[off + 12])
                          << " final_entries=" << gathered_profile[off + 2]
                          << " capacity_entries=" << gathered_profile[off + 13]
                          << " capacity_growth_events=" << gathered_profile[off + 14]
                          << " reallocations=" << gathered_profile[off + 15]
                          << " geometric_copy_bytes=" << gathered_profile[off + 16]
                          << " block_allocations=" << gathered_profile[off + 17]
                          << " allocated_capacity_bytes=" << gathered_profile[off + 18]
                          << " gather_copy_bytes=" << gathered_profile[off + 19]
                          << std::endl;
                std::cout << "JacobianAssembler: structural_plan_cache_rank"
                          << " rank=" << static_cast<int>(gathered_profile[off + 0])
                          << " enabled=" << static_cast<int>(gathered_profile[off + 20])
                          << " hit=" << static_cast<int>(gathered_profile[off + 21])
                          << " miss=" << static_cast<int>(gathered_profile[off + 22])
                          << " check_seconds=" << gathered_profile[off + 23]
                          << " miss_build_seconds=" << gathered_profile[off + 24]
                          << std::endl;
            }
        }
    }

    const MemoryMapperTrafficSnapshot mapper_traffic = mapper_traffic_scope.finish();
    const Transform1dTrafficSnapshot transform_traffic = transform_traffic_scope.finish();
    if (timing && rank_ == 0) {
        std::cout << "Jacobian mapper stats (rank 0): get_calls="
                  << mapper_traffic.get_calls
                  << " release_calls=" << mapper_traffic.release_calls
                  << " get_bytes=" << mapper_traffic.requested_get_bytes
                  << " release_bytes=" << mapper_traffic.requested_release_bytes
                  << std::endl;
        std::cout << "Jacobian transform stats (rank 0): forward_1d="
                  << transform_traffic.forward
                  << " backward_1d=" << transform_traffic.backward
                  << std::endl;
    }

    // Collective exit after the optional profile reductions: every rank must
    // participate in both the diagnostic collectives and MPI_Finalize.
    if ((coo_hash_active || canonical_coo_hash_active) && hash_exit_after_first) {
        MPI_Barrier(comm_);
        MPI_Finalize();
        std::exit(0);
    }

    report_memory_mapper_phase("jacobian.return", comm_);
    return coo;
}

void JacobianAssembler::diagonal_stats(const AssembledJacobianCoo& coo,
                                       double drop_tol_reported) const
{
    if (rank_ != 0)
        return;
    const int n = coo.n;
    const long long nnz = coo.nnz;
    double min_dii = std::numeric_limits<double>::max();
    double max_dii = 0.0;
    int n_missing_diag = 0;
    const auto scan_block = [&](int block_n, long long block_nnz,
                                const AssembledJacobianCoo::IndexVector& irn,
                                const AssembledJacobianCoo::IndexVector& jcn,
                                const AssembledJacobianCoo::ValueVector& a) {
        std::vector<bool> has_diag(static_cast<std::size_t>(block_n), false);
        for (long long k = 0; k < block_nnz; ++k) {
            const std::size_t index = static_cast<std::size_t>(k);
            if (irn[index] == jcn[index]) {
                has_diag[static_cast<std::size_t>(irn[index] - 1)] = true;
                const double diagonal = std::abs(a[index]);
                min_dii = std::min(min_dii, diagonal);
                max_dii = std::max(max_dii, diagonal);
            }
        }
        for (bool present : has_diag) {
            if (!present)
                ++n_missing_diag;
        }
    };
    if (coo.parity_blocks.empty()) {
        scan_block(n, nnz, coo.irn, coo.jcn, coo.a);
    } else {
        for (const AssembledJacobianCooBlock& block : coo.parity_blocks) {
            scan_block(block.n, block.nnz, block.irn, block.jcn, block.a);
        }
    }
    const double sparsity =
        static_cast<double>(nnz) / (static_cast<double>(n) * static_cast<double>(n)) * 100.0;
    std::cout << "Jacobian: n=" << n << " nnz=" << nnz << std::fixed << std::setprecision(4)
              << " sparsity=" << sparsity << "%" << std::defaultfloat
              << " drop_tol=" << drop_tol_reported << " missing_diag=" << n_missing_diag
              << " |J_diag| in [" << min_dii << ", " << max_dii << "]" << std::endl;
}

} // namespace Kadath
