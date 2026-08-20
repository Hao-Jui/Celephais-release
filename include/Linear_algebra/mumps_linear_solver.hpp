#pragma once
#include "Linear_algebra/linear_solver.hpp"
#include "Linear_algebra/mumps_out_of_core_mode.hpp"

#ifdef CELEPHAIS_USE_MUMPS

#include <mpi.h>

#include "dmumps_c.h"

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace Kadath
{

// Human-readable ICNTL(7)/INFOG(7) ordering label. Unknown values retain their
// numeric ID so diagnostics remain actionable.
std::string mumps_ordering_name(int ordering);

namespace mumps_memory_detail
{

// Internal parsing surface exposed for deterministic fixture tests. Memory is
// reported in MiB (the existing diagnostics call these values MB).
long long parse_mem_available_mb(const std::string& meminfo);

enum class CgroupMemoryStatus
{
    Unreadable,
    Unlimited,
    Limited,
};

struct CgroupMemoryHeadroom
{
    CgroupMemoryStatus status = CgroupMemoryStatus::Unreadable;
    long long available_mb = -1;
};

// Parse cgroup v2 memory.max/current or v1 limit/usage fixture contents.
// "max" and the v1 kernel sentinel are Unlimited; a finite exhausted limit is
// Limited with zero headroom, distinct from unreadable input.
CgroupMemoryHeadroom parse_cgroup_memory_headroom(
    const std::string& limit, const std::string& usage);

// Available memory for the current node/process after applying any effective
// cgroup memory limit. Returns -1 when the required platform data is unreadable.
long long node_available_memory_mb();

// Process-local monotone minimum for valid node-available-memory probes.
// Negative probes retain the unreadable sentinel and do not change the minimum.
long long ratcheted_node_available_memory_mb(long long probed_mb);

// Test-only hook: production code must retain the process-lifetime minimum.
void reset_node_available_memory_ratchet_for_tests();

} // namespace mumps_memory_detail

// Result of aligning one numerical COO matrix to a monotone sparsity-pattern
// superset. `pattern_changed` is the correctness gate for symbolic reuse: a
// true value means at least one current coordinate was absent from the cached
// pattern and MUMPS analysis must be run again before factorization.
struct MumpsPatternSupersetUpdate
{
    bool pattern_changed = false;
    long long candidate_nnz = 0;
    long long numerical_nnz = 0;
    long long superset_nnz = 0;
    long long new_pattern_entries = 0;
    long long explicit_zero_entries = 0;
};

// Build or extend a column-grouped COO pattern and align current values to it.
// The current COO may list columns in any order, but every column must occupy
// at most one contiguous group (the JacobianAssembler ownership/gather
// contract). Rows inside a group may be unsorted. Cached patterns are a
// multiset union, so duplicate COO coordinates retain MUMPS' duplicate-entry
// semantics rather than being silently combined. The cached buffers are never
// mutated: on support growth the replacement pattern is returned through the
// `next_pattern_*` outputs. This lets a caller destroy a MUMPS solver that still
// borrows the cached irn/jcn pointers before committing the replacement.
MumpsPatternSupersetUpdate update_mumps_pattern_superset(
    int n,
    long long nnz,
    const int* irn,
    const int* jcn,
    const double* values,
    double numerical_drop_tol,
    const std::vector<int>& pattern_irn,
    const std::vector<int>& pattern_jcn,
    const std::vector<long long>& pattern_column_offsets,
    std::vector<int>& next_pattern_irn,
    std::vector<int>& next_pattern_jcn,
    std::vector<long long>& next_pattern_column_offsets,
    std::vector<double>& aligned_values);

// MUMPS-backed LinearSolver. Encapsulates DMUMPS_STRUC_C lifecycle, ICNTL
// settings, adaptive workspace retry on -9/-20, RHS gather + bcast. Throws
// LinearSolverError on persistent failure (replaces previous MPI_Abort).
//
// factor() is split into a separate analyze_pattern() + factor_analyzed()
// pair so the caller can detach and free the centralized COO input after a
// successful factorization, before the factor is held through subsequent solves.
// MUMPS still borrows irn/jcn/a during every JOB=2 attempt, including -9/-20
// retries, so none of those arrays may be released after analysis alone.
//
// Lifecycle owned by RAII: ctor calls JOB_INIT + sets ICNTLs; dtor calls
// JOB_END. Caller can call reset() explicitly to release memory between
// Newton iterations.
class MumpsLinearSolver : public LinearSolver
{
  public:
    // MUMPS cannot apply its zero-free-diagonal matching when ICNTL(15)
    // analysis-by-blocks is active. Preserve is therefore the fail-closed
    // default. ExplicitlyDisable is an experimental opt-out from matching; it
    // never changes the independent ICNTL(8) factor scaling control.
    enum class BlockAnalysisMatching
    {
        Preserve,
        ExplicitlyDisable,
    };

    // ranks_per_node > 0 factors/solves MUMPS on only the first `ranks_per_node`
    // MPI ranks of each physical node (a sub-communicator); assembly, do_JX, GMRES
    // and the residual stay on the full `comm`. Fewer factor-ranks = more RAM/rank
    // and less total MUMPS memory (its scaling is flat in speed but grows total
    // memory), which clears the per-rank front/buffer wall at high resolution.
    // 0 (default) = factor on all of `comm` (bit-identical to the original path).
    // memory_capped sets ICNTL(23) (per-rank workspace cap, MB) to
    // (this node's free RAM) x 0.8 / ranks_per_node, so a factor that would exceed
    // the available memory returns a catchable INFOG(1)=-9/-16 instead of a SIGKILL.
    // Requires ranks_per_node > 0; no-op when platform memory data is unreadable.
    // The trailing AUTO policy values control the post-analysis resident-memory
    // decision; the test-only budget replaces probed node-available MB.
    MumpsLinearSolver(int n, int ordering, MumpsOutOfCoreMode out_of_core_mode,
                      int blr,
                      int icntl14_initial, MPI_Comm comm, int ranks_per_node = 0,
                      bool memory_capped = false,
                      double out_of_core_touch = kMumpsOutOfCoreTouchDefault,
                      double out_of_core_safety = kMumpsOutOfCoreSafetyDefault,
                      double out_of_core_budget_mb = kMumpsOutOfCoreBudgetUnset);
    // Compatibility overload for explicit policy callers. Configuration-derived
    // callers use the strong mode overload so Auto survives until factor time.
    MumpsLinearSolver(int n, int ordering, bool out_of_core, int blr,
                      int icntl14_initial, MPI_Comm comm, int ranks_per_node = 0,
                      bool memory_capped = false);
    ~MumpsLinearSolver() override;

    MumpsLinearSolver(const MumpsLinearSolver&) = delete;
    MumpsLinearSolver& operator=(const MumpsLinearSolver&) = delete;

    void set_pattern(int n, long long nnz, const int* irn, const int* jcn) override;
    void factor(const double* values) override;
    // Route the post-analysis AUTO OOC decision to a caller-owned stream and
    // optionally replace its default "MUMPS " prefix. A null stream suppresses
    // only that diagnostic. Configure before analyze_pattern().
    void set_auto_out_of_core_diagnostic(std::ostream* diagnostic,
                                         std::string prefix = "MUMPS ");
    // Every JOB=1 entry performs one fixed-size MPI_Allgather on world_comm_
    // before entering MUMPS. It fails collectively if experimental block,
    // user-permutation, or OOC policy metadata differs across caller ranks.
    void analyze_pattern();
    // Enable MUMPS user-provided analysis by blocks (ICNTL(15)=1). `blkptr`
    // follows the 1-based MUMPS convention, starts at 1, ends at n+1, and
    // partitions all variables into non-empty blocks. An empty `blkvar` asks
    // MUMPS to use the identity permutation (contiguous blocks); otherwise it
    // must be a 1-based permutation of [1,n] and supports noncontiguous blocks.
    // The solver owns copies of both arrays through analysis. Call only before
    // analyze_pattern(); production behavior remains disabled by default.
    void enable_block_analysis(
        const std::vector<int>& blkptr_1based,
        const std::vector<int>& blkvar_1based = {},
        BlockAnalysisMatching matching = BlockAnalysisMatching::Preserve);
    void disable_block_analysis();
    bool block_analysis_enabled() const { return block_analysis_enabled_; }
    int configured_block_count() const
    {
        return block_analysis_enabled_
                   ? static_cast<int>(block_ptr_1based_.size() - 1)
                   : 0;
    }
    // Diagnostic-only user ordering. The vector follows MUMPS' convention:
    // permutation[i] is the 1-based pivot position of original variable i+1.
    // A copy is owned until analysis. If block analysis is active, each block
    // must occupy consecutive pivot positions in this permutation.
    void set_user_permutation_1based(
        const std::vector<int>& permutation_1based);
    void clear_user_permutation();
    // Detach all borrowed centralized COO pointers after the final successful
    // factorization. The factor remains usable for solve(), but another
    // factor_analyzed() requires a fresh set_pattern() + analyze_pattern().
    void release_centralized_coo_input();
    // Detach only the numerical values after a successful JOB=2. The symbolic
    // analysis and its still-owned irn/jcn pattern remain reusable for another
    // factor_analyzed() call with new values.
    void release_factor_values_input();
    // Compatibility name retained for existing callers. Historically this
    // detached only irn/jcn; it now safely detaches a as well.
    void release_pattern_input();
    void factor_analyzed(const double* values);
    // Register the column permutation used to build a composed matrix A*P.
    // Entry i is the 1-based original unknown represented by solution entry i,
    // so solve() returns original-order x via x[permutation[i]-1] = y[i].
    // The solver owns the map and a no-throw solve workspace. Configure it
    // before analysis; solve_transpose() intentionally remains in the composed
    // system's raw ordering.
    void set_solution_column_permutation_1based(
        const std::vector<int>& permutation_1based);
    void clear_solution_column_permutation();
    void solve(double* rhs_inout) override;
    // Diagnostic-only companion for nonsymmetric near-null/coarse-space probes.
    // Uses the existing factorization and solves A^T x = rhs.
    void solve_transpose(double* rhs_inout);
    void reset() override;

    // Diagnostics for the most recent successful analysis/factor. Estimated
    // values are populated by analyze_pattern(); effective values are populated
    // by factor_analyzed(). Values are stale outside those lifecycle points.
    int last_icntl14() const { return last_icntl14_; }
    // Exact ICNTL(14) value on the successful JOB=2 attempt, before the
    // production next-iteration seed is clamped to its retention range.
    int successful_factor_icntl14() const { return successful_factor_icntl14_; }
    int last_actual_ordering() const { return last_actual_ordering_; }
    int analysis_rank_count() const { return analysis_rank_count_; }
    int factor_ranks_per_node() const { return ranks_per_node_; }
    MumpsOutOfCoreMode requested_out_of_core_mode() const
    {
        return out_of_core_mode_;
    }
    bool out_of_core_enabled() const { return ooc_icntl22_ != 0; }
    // Snapshot the controls retained after successful analysis for the upcoming
    // factorization. For AUTO, ICNTL(22) contains the resolved post-analysis
    // value. MUMPS 5.9 exposes 60 ICNTL and 15 CNTL entries. Rank 0 receives the
    // values; ranks outside the factor communicator receive zero-filled arrays.
    void copy_analysis_controls(
        std::array<std::int32_t, 60>& icntl,
        std::array<double, 15>& cntl) const;
    // INFOG(21): memory effectively used by the factorization on the rank that
    // uses the most, in MB. This is MUMPS' effective-use counter, not process
    // RSS or the amount allocated by MUMPS.
    int factor_memory_mb() const { return factor_memory_mb_; }
    // INFOG(18): memory allocated by MUMPS for factorization on the rank that
    // allocated the most, in MB.
    int factor_allocated_memory_mb() const
    {
        return factor_allocated_memory_mb_;
    }
    // INFOG(16): estimated per-rank-max factor memory (MB) from ANALYSIS, available
    // before the factor runs -- the number to size ranks_per_node against.
    int estimated_factor_memory_mb() const { return estimated_factor_memory_mb_; }
    // INFOG(17): analysis estimate summed over all factor ranks (MB).
    int estimated_factor_memory_total_mb() const
    {
        return estimated_factor_memory_total_mb_;
    }
    // INFOG(5): maximum front order estimated by analysis.
    int estimated_max_front_order() const { return estimated_max_front_order_; }
    // Analysis-only structural estimates. INFOG(20)/(3)/(4) use MUMPS'
    // negative-millions overflow encoding and are decoded to 64-bit counts.
    long long estimated_factor_nnz() const { return estimated_factor_nnz_; }
    long long estimated_factor_real_slots() const
    {
        return estimated_factor_real_slots_;
    }
    long long estimated_factor_integer_slots() const
    {
        return estimated_factor_integer_slots_;
    }
    // INFOG(6): number of nodes in the analysis elimination tree.
    int estimated_tree_node_count() const { return estimated_tree_node_count_; }
    // RINFOG(1): estimated elimination work from analysis, in Gflop.
    double estimated_factor_flops_gflop() const
    {
        return estimated_factor_flops_gflop_;
    }
    // INFOG(22): same, summed over all ranks (MB).
    int factor_memory_total_mb() const { return factor_memory_total_mb_; }
    // INFOG(29)/(9): actual factor entries and real factor-storage slots. MUMPS
    // encodes counts that exceed a 32-bit integer as negative millions; these
    // accessors return decoded 64-bit counts.
    long long factor_nnz() const { return factor_nnz_; }
    long long factor_real_slots() const { return factor_real_slots_; }
    // INFOG(11): order of the largest frontal matrix in the completed factor.
    int max_front_order() const { return max_front_order_; }
    // INFOG(13): total number of delayed pivots in the completed factorization.
    int delayed_pivot_count() const { return delayed_pivot_count_; }
    // Number of workspace-relaxation JOB=2 retries used by the most recent
    // successful regular factorization. This excludes the first attempt and
    // does not describe the separate diagnostic Schur path.
    int factor_retry_count() const { return factor_retry_count_; }
    // RINFOG(3): floating-point operations for the elimination (= the actual
    // factorization work), over all ranks.
    double factor_flops_gflop() const { return factor_flops_gflop_; }

    // Diagnostic-only Schur extraction. `listvar_schur_1based` follows the
    // MUMPS Fortran convention; on success rank 0 receives a dense
    // column-major Schur matrix in `schur_col_major`.
    void extract_schur(const double* values,
                       const std::vector<int>& listvar_schur_1based,
                       std::vector<double>& schur_col_major);

    // Enable MUMPS null-pivot detection (ICNTL(24)=1, CNTL(3)=threshold) for
    // the next factor(). Diagnostic-only path: the factor numerically completes
    // even when the matrix is structurally singular; null pivots are flagged
    // for later inspection via last_null_pivot_count() / pivnul_list.
    // Default-OFF; callers must opt in. Call BEFORE factor().
    void enable_null_pivot_detection(bool enable, double threshold = 0.0);

    // After a factor() call performed with null-pivot detection enabled,
    // INFOG(28) null-pivot count and PIVNUL_LIST (1-based variable indices)
    // are captured here. Both are zero / empty when detection was off.
    int last_null_pivot_count() const { return last_null_pivot_count_; }
    const std::vector<int>& last_null_pivot_list_1based() const
    {
        return last_null_pivot_list_1based_;
    }

    // Diagnostic-only MUMPS ordering extraction. On success rank 0 receives
    // SYM_PERM in MUMPS' 1-based convention: sym_perm_1based[i] is the pivot
    // position of original variable i+1.
    void analyze_symmetric_permutation(const double* values,
                                       std::vector<int>& sym_perm_1based);
    // Copy SYM_PERM from the existing successful analysis without another
    // JOB=1. Rank 0 receives MUMPS' 1-based original-variable -> pivot-position
    // map; other ranks receive an empty vector.
    void copy_symmetric_permutation_1based(
        std::vector<int>& sym_perm_1based) const;
    // Copy UNS_PERM (the maximum-transversal column permutation) from the
    // existing successful analysis. Rank 0 receives MUMPS' 1-based map — entry
    // A(i, uns_perm_1based[i-1]) sits on the diagonal of the internally
    // permuted matrix — and matching_applied=true; when the analysis applied
    // no column permutation, rank 0 receives the identity and
    // matching_applied=false. Other ranks receive an empty vector and false.
    void copy_column_permutation_1based(std::vector<int>& uns_perm_1based,
                                        bool& matching_applied) const;

  private:
    DMUMPS_STRUC_C mumps_;
    MPI_Comm comm_;            // the MUMPS communicator (== world_comm_ when ranks_per_node==0)
    MPI_Comm world_comm_;      // the full caller comm; solve()'s solution bcast is over this
    MPI_Comm mumps_comm_ = MPI_COMM_NULL;  // sub-comm (owned) when ranks_per_node>0; else == comm
    bool in_mumps_ = true;     // does this rank participate in MUMPS factor/solve?
    bool owns_mumps_comm_ = false;  // free mumps_comm_ in dtor (only the split sub-comm)
    int rank_ = 0;             // rank within comm_ (the MUMPS comm); host == 0
    bool initialized_ = false;
    int icntl14_;  // seeded from ctor arg icntl14_initial
    int ordering_ = 0;
    MumpsOutOfCoreMode out_of_core_mode_ = MumpsOutOfCoreMode::Off;
    int ooc_icntl22_ = 0;
    double out_of_core_touch_ = kMumpsOutOfCoreTouchDefault;
    double out_of_core_safety_ = kMumpsOutOfCoreSafetyDefault;
    double out_of_core_budget_mb_ = kMumpsOutOfCoreBudgetUnset;
    std::ostream* auto_out_of_core_diagnostic_ = nullptr;
    std::string auto_out_of_core_diagnostic_prefix_ = "MUMPS ";
    int blr_icntl35_ = 0;

    // Latest pattern bookkeeping (rank 0 owns full COO; others zero).
    int n_ = 0;
    long long nnz_ = 0;
    const int* pattern_irn_ = nullptr;
    const int* pattern_jcn_ = nullptr;
    bool analyzed_ = false;

    // Diagnostics from last successful factor().
    int last_icntl14_ = 100;
    int successful_factor_icntl14_ = 100;
    int last_actual_ordering_ = 0;
    int factor_memory_mb_ = 0;        // INFOG(21): effective factor mem, max rank (MB)
    int factor_memory_total_mb_ = 0;  // INFOG(22): effective factor mem, sum of ranks (MB)
    int factor_allocated_memory_mb_ = 0; // INFOG(18): allocated factor mem, max rank (MB)
    int estimated_factor_memory_mb_ = 0; // INFOG(16): est. factor mem/rank-max from analysis
    int estimated_factor_memory_total_mb_ = 0; // INFOG(17): est. factor mem, rank sum (MB)
    int estimated_max_front_order_ = 0; // INFOG(5): analysis-estimated largest front
    long long estimated_factor_nnz_ = 0; // INFOG(20): estimated factor entries
    long long estimated_factor_real_slots_ = 0; // INFOG(3): estimated real slots
    long long estimated_factor_integer_slots_ = 0; // INFOG(4): estimated integer slots
    int estimated_tree_node_count_ = 0; // INFOG(6): elimination-tree nodes
    double estimated_factor_flops_gflop_ = 0.0; // RINFOG(1): estimated factor work
    long long factor_nnz_ = 0;       // INFOG(29): entries in factors, decoded
    long long factor_real_slots_ = 0; // INFOG(9): real factor space, decoded
    int max_front_order_ = 0;        // INFOG(11): actual largest front order
    int delayed_pivot_count_ = 0;    // INFOG(13): total delayed pivots
    int factor_retry_count_ = 0;     // additional JOB=2 attempts after -9/-20
    double factor_flops_gflop_ = 0.0; // RINFOG(3): elimination flops, all ranks
    int analysis_rank_count_ = 0;     // participants in the MUMPS communicator
    int ranks_per_node_ = 0;          // factor-ranks per node (for the ICNTL(23) budget)
    int factor_ranks_on_host_node_ = 0; // actual factor participants colocated with host rank 0
    bool memory_capped_ = false;      // set ICNTL(23) = free-RAM x 0.8 / ranks_per_node_

    // Null-pivot detection (diagnostic-only; ICNTL(24)=1 + CNTL(3) threshold).
    bool null_pivot_detection_ = false;
    double null_pivot_threshold_ = 0.0;
    int last_null_pivot_count_ = 0;
    std::vector<int> last_null_pivot_list_1based_;

    // Owned ICNTL(15)=1 metadata. MUMPS accesses these pointers on its host
    // during analysis; keeping copies avoids borrowing caller vector storage.
    std::vector<MUMPS_INT> block_ptr_1based_;
    std::vector<MUMPS_INT> block_variables_1based_;
    bool block_analysis_enabled_ = false;
    bool block_matching_overridden_ = false;
    int saved_block_icntl6_ = 0;
    std::vector<MUMPS_INT> user_permutation_1based_;
    bool user_permutation_enabled_ = false;

    // Optional composed-matrix solution map. The workspace is allocated with
    // the map so solve() cannot throw between MUMPS JOB=3 and its world bcast.
    std::vector<int> solution_column_permutation_1based_;
    std::vector<double> solution_column_permutation_workspace_;

    void apply_icntls();
    void prepare_auto_out_of_core_for_analysis();
    void resolve_auto_out_of_core_after_analysis();
    static void validate_block_permutation_compatibility(
        int n,
        const std::vector<MUMPS_INT>& blkptr_1based,
        const std::vector<MUMPS_INT>& blkvar_1based,
        const std::vector<MUMPS_INT>& permutation_1based);
    void validate_collective_analysis_metadata() const;
    void run_job(int job);
    void capture_analysis_diagnostics();
    void capture_factor_diagnostics(int successful_icntl14);
    void print_infog_trace() const;
};

} // namespace Kadath

#endif // CELEPHAIS_USE_MUMPS
