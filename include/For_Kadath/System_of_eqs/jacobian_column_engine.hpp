#pragma once

// JacobianColumnEngine — per-System_of_eqs cache-and-compute facade for
// `do_col_J` / `do_col_J_sparse`. See .omc/plans/jacobian-column-engine.md.
//
// Owns Jacobian column timing, caches, scratch workspaces, and the compute
// bodies; `System_of_eqs::do_col_J*`, `reset_do_col_J_cache`, and
// `dump_do_col_J_profile` are thin shims.
//
// `<memory>` / `<type_traits>` / `<vector>` and `array.hpp` are pulled in
// directly so the header is self-contained and can be included independently.

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Space/space.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace Kadath {

class System_of_eqs;

using JacobianSelectedRows = std::optional<std::span<const int>>;

template <typename Sink>
concept SparseColumnEmissionSink =
    std::is_object_v<std::remove_reference_t<Sink>> &&
    std::invocable<Sink&, int, double> &&
    std::same_as<std::invoke_result_t<Sink&, int, double>, void>;

namespace jacobian_column_engine_detail
{
    // Visit the sorted selected indices inside one already-active half-open
    // range. A disengaged selection preserves the full-range legacy scan;
    // an engaged empty span selects no indices.
    template <typename Visitor>
        requires std::invocable<Visitor&, int>
    long long for_each_selected_index_in_range(
        int begin, int end, JacobianSelectedRows selected_indices,
        Visitor&& visitor)
    {
        if (end <= begin)
            return 0;
        if (!selected_indices) {
            for (int index = begin; index < end; ++index)
                visitor(index);
            return static_cast<long long>(end - begin);
        }
        const auto first = std::ranges::lower_bound(*selected_indices, begin);
        const auto last =
            std::ranges::lower_bound(first, selected_indices->end(), end);
        for (auto it = first; it != last; ++it)
            visitor(*it);
        return static_cast<long long>(std::distance(first, last));
    }

    inline bool range_contains_selected_index(
        int begin, int end, JacobianSelectedRows selected_indices)
    {
        if (end <= begin)
            return false;
        if (!selected_indices)
            return true;
        const auto first = std::ranges::lower_bound(*selected_indices, begin);
        return first != selected_indices->end() && *first < end;
    }
} // namespace jacobian_column_engine_detail

class SparseColumnEmitter {
  public:
    template <SparseColumnEmissionSink Callable>
        requires (!std::same_as<std::remove_cvref_t<Callable>, SparseColumnEmitter>)
    SparseColumnEmitter(Callable&& callable)
        : context_(const_cast<void*>(static_cast<const void*>(std::addressof(callable)))),
          invoke_([](void* context, int row, double value) {
              (*static_cast<std::remove_reference_t<Callable>*>(context))(row, value);
          })
    {
    }

    void operator()(int row, double value) const { invoke_(context_, row, value); }

  private:
    void* context_ = nullptr;
    void (*invoke_)(void*, int, double) = nullptr;
};

template <std::size_t Width>
using PackedJacobianColumns = std::span<const int, Width>;

template <std::size_t Width>
using PackedSparseColumnEmitters = std::span<SparseColumnEmitter, Width>;

template <std::size_t Width>
using PackedJacobianOutputs = std::span<Array<double>*, Width>;

class JacobianColumnEngine {
  public:
    class Workspace;

    explicit JacobianColumnEngine(System_of_eqs& sys);

    // Compute Jacobian column `cc`; write into caller-owned `out`.
    void compute_column(int cc, Array<double>& out,
                        JacobianSelectedRows selected_rows = std::nullopt);
    void compute_column(Workspace& workspace, int cc, Array<double>& out,
                        JacobianSelectedRows selected_rows = std::nullopt);

    // Sparse path: emit (row, value) pairs above `drop_tol`.
    void compute_column_sparse(int cc,
                               double drop_tol,
                               SparseColumnEmitter emit,
                               JacobianSelectedRows selected_rows = std::nullopt);
    void compute_column_sparse(Workspace& workspace, int cc, double drop_tol,
                               SparseColumnEmitter emit,
                               JacobianSelectedRows selected_rows = std::nullopt);

    // W=2 fallback sparse path: seed two supported columns into independent
    // derivative lanes, run one residual traversal, and emit both sparse
    // columns. Variable-domain pairs require explicit space opt-in and the
    // runtime gate; unsupported pairs return false for scalar fallback.
    bool compute_packed_wlane2_columns_sparse(int first_column,
                                              int second_column,
                                              double drop_tol,
                                              SparseColumnEmitter emit_first,
                                              SparseColumnEmitter emit_second,
                                              std::string& failure_reason,
                                              JacobianSelectedRows selected_rows = std::nullopt);
    bool compute_packed_wlane2_columns_sparse(Workspace& workspace,
                                              PackedJacobianColumns<2> columns,
                                              double drop_tol,
                                              PackedSparseColumnEmitters<2> emitters,
                                              std::string& failure_reason,
                                              JacobianSelectedRows selected_rows = std::nullopt);

    // Diagnostic W=2 path: seed two columns into independent derivative lanes,
    // run one residual traversal, and export both full columns. This is a
    // correctness oracle for packed-column AD; production assembly stays scalar.
    bool compute_packed_wlane2_columns_for_verification(int first_column,
                                                        int second_column,
                                                        Array<double>& first_output,
                                                        Array<double>& second_output,
                                                        std::string& failure_reason);

    // W=4 sparse path: seed four compatible columns into independent
    // derivative lanes, run one residual traversal, and emit four sparse
    // columns. Supported variable-domain groups require the same space opt-in
    // and runtime gate as W=2. Mirrors compute_packed_wlane2_columns_sparse
    // with lane-count 4.
    // Bitwise-identity validation is covered by validate_packed_wlane4_columns;
    // this entry point itself relies on the per-lane independence guaranteed by
    // the Term_eq lane storage.
    bool compute_packed_wlane4_columns_sparse(const std::array<int, 4>& columns,
                                              double drop_tol,
                                              std::array<SparseColumnEmitter, 4>& emitters,
                                              std::string& failure_reason,
                                              JacobianSelectedRows selected_rows = std::nullopt);

    // W=8 sparse path: seed eight compatible field or supported
    // variable-domain columns into independent derivative lanes, run one
    // residual traversal, and emit eight sparse columns. The assembler
    // cascades W=8 -> W=4 -> W=2 -> scalar.
    bool compute_packed_wlane8_columns_sparse(const std::array<int, 8>& columns,
                                              double drop_tol,
                                              std::array<SparseColumnEmitter, 8>& emitters,
                                              std::string& failure_reason,
                                              JacobianSelectedRows selected_rows = std::nullopt);

    // W=16 sparse path: seed sixteen compatible field or supported
    // variable-domain columns into independent derivative lanes, run one
    // residual traversal, and emit sixteen sparse columns. The assembler
    // cascades W=16 -> W=8 -> W=4 -> W=2 -> scalar.
    bool compute_packed_wlane16_columns_sparse(const std::array<int, 16>& columns,
                                               double drop_tol,
                                               std::array<SparseColumnEmitter, 16>& emitters,
                                               std::string& failure_reason,
                                               JacobianSelectedRows selected_rows = std::nullopt);

    // W=32 sparse path: seed thirty-two compatible field or supported
    // variable-domain columns into independent derivative lanes, run one
    // residual traversal, and emit thirty-two sparse columns. The assembler
    // cascades W=32 -> W=16 -> W=8 -> W=4 -> W=2 -> scalar.
    // Enabled by JACOBIAN_WLANE32 (default on; set 0 for scalar fallback).
    bool compute_packed_wlane32_columns_sparse(const std::array<int, 32>& columns,
                                               double drop_tol,
                                               std::array<SparseColumnEmitter, 32>& emitters,
                                               std::string& failure_reason,
                                               JacobianSelectedRows selected_rows = std::nullopt);

    // Diagnostic W=4 path: seed four columns into independent derivative lanes,
    // run one residual traversal, and export all four full columns into the
    // caller-provided output buffers. Correctness oracle for the W=4 packed
    // assembly.
    bool compute_packed_wlane4_columns_for_verification(const std::array<int, 4>& columns,
                                                        std::array<Array<double>*, 4>& outputs,
                                                        std::string& failure_reason);

    // Diagnostic W=8 path: correctness oracle for the W=8 packed assembly,
    // mirroring the W=2/W=4 oracles. Exposed by templating the shared body
    // (compute_packed_wlaneN_columns_for_verification<W>) so the production W=8
    // sparse path now has a per-lane bit-identity reference, not only the
    // value-tolerance check against scalar do_col_J.
    bool compute_packed_wlane8_columns_for_verification(const std::array<int, 8>& columns,
                                                        std::array<Array<double>*, 8>& outputs,
                                                        std::string& failure_reason);

    // Diagnostic W=16 path: correctness oracle for the W=16 packed assembly.
    bool compute_packed_wlane16_columns_for_verification(const std::array<int, 16>& columns,
                                                         std::array<Array<double>*, 16>& outputs,
                                                         std::string& failure_reason);

    // Diagnostic W=32 path: correctness oracle for the W=32 packed assembly.
    bool compute_packed_wlane32_columns_for_verification(const std::array<int, 32>& columns,
                                                         std::array<Array<double>*, 32>& outputs,
                                                         std::string& failure_reason);

    bool compute_packed_wlane4_columns_sparse(Workspace& workspace,
                                              PackedJacobianColumns<4> columns,
                                              double drop_tol,
                                              PackedSparseColumnEmitters<4> emitters,
                                              std::string& failure_reason,
                                              JacobianSelectedRows selected_rows = std::nullopt);
    bool compute_packed_wlane8_columns_sparse(Workspace& workspace,
                                              PackedJacobianColumns<8> columns,
                                              double drop_tol,
                                              PackedSparseColumnEmitters<8> emitters,
                                              std::string& failure_reason,
                                              JacobianSelectedRows selected_rows = std::nullopt);
    bool compute_packed_wlane16_columns_sparse(Workspace& workspace,
                                               PackedJacobianColumns<16> columns,
                                               double drop_tol,
                                               PackedSparseColumnEmitters<16> emitters,
                                               std::string& failure_reason,
                                               JacobianSelectedRows selected_rows = std::nullopt);
    bool compute_packed_wlane32_columns_sparse(Workspace& workspace,
                                               PackedJacobianColumns<32> columns,
                                               double drop_tol,
                                               PackedSparseColumnEmitters<32> emitters,
                                               std::string& failure_reason,
                                               JacobianSelectedRows selected_rows = std::nullopt);

    // Reset transient caches that depend on the current adapted-domain state.
    void reset_cache();
    void reset_cache(Workspace& workspace);

    // Release large per-assembly scratch buffers once column emission is done.
    void release_assembly_scratch();
    void release_assembly_scratch(Workspace& workspace);

    // Print and reset profiling totals (rank 0 only).
    void dump_profile() const;
    void dump_profile(Workspace& workspace) const;

  private:
    struct DoColJTimingState {
        double update_terms = 0.0;
        double var_double = 0.0;
        double fields = 0.0;
        double eq_affects = 0.0;
        double apply = 0.0;
        double export_der = 0.0;
        int calls = 0;
        int var_domain_calls = 0;
        double var_domain_total = 0.0;
        double metric_update = 0.0;
        double def_update = 0.0;
        double def_clear = 0.0;
        double def_compute = 0.0;
        long long sparse_scan_rows = 0;
        long long sparse_full_scan_rows = 0;
        long long equation_domain_active = 0;
        long long equation_variable_skipped = 0;
        long long equation_active = 0;
        int packed_wlane2_calls = 0;
        double packed_wlane2_total = 0.0;
        double packed_wlane2_prepare = 0.0;
        double packed_wlane2_seed = 0.0;
        double packed_wlane2_metric_update = 0.0;
        double packed_wlane2_def_update = 0.0;
        double packed_wlane2_def_clear = 0.0;
        double packed_wlane2_def_compute = 0.0;
        double packed_wlane2_equation_filter = 0.0;
        double packed_wlane2_apply = 0.0;
        double packed_wlane2_export = 0.0;
        double packed_wlane2_scan = 0.0;
        double packed_wlane2_restore = 0.0;
        int packed_wlane4_calls = 0;
        double packed_wlane4_total = 0.0;
        double packed_wlane4_prepare = 0.0;
        double packed_wlane4_seed = 0.0;
        double packed_wlane4_metric_update = 0.0;
        double packed_wlane4_def_update = 0.0;
        double packed_wlane4_def_clear = 0.0;
        double packed_wlane4_def_compute = 0.0;
        double packed_wlane4_equation_filter = 0.0;
        double packed_wlane4_apply = 0.0;
        double packed_wlane4_export = 0.0;
        double packed_wlane4_scan = 0.0;
        double packed_wlane4_restore = 0.0;
        int packed_wlane8_calls = 0;
        double packed_wlane8_total = 0.0;
        double packed_wlane8_prepare = 0.0;
        double packed_wlane8_seed = 0.0;
        double packed_wlane8_metric_update = 0.0;
        double packed_wlane8_def_update = 0.0;
        double packed_wlane8_def_clear = 0.0;
        double packed_wlane8_def_compute = 0.0;
        double packed_wlane8_equation_filter = 0.0;
        double packed_wlane8_apply = 0.0;
        double packed_wlane8_export = 0.0;
        double packed_wlane8_export_clear = 0.0;
        double packed_wlane8_export_integral = 0.0;
        double packed_wlane8_export_equations = 0.0;
        double packed_wlane8_scan = 0.0;
        double packed_wlane8_restore = 0.0;
        int packed_wlane16_calls = 0;
        double packed_wlane16_total = 0.0;
        double packed_wlane16_prepare = 0.0;
        double packed_wlane16_seed = 0.0;
        double packed_wlane16_metric_update = 0.0;
        double packed_wlane16_def_update = 0.0;
        double packed_wlane16_def_clear = 0.0;
        double packed_wlane16_def_compute = 0.0;
        double packed_wlane16_equation_filter = 0.0;
        double packed_wlane16_apply = 0.0;
        double packed_wlane16_export = 0.0;
        double packed_wlane16_export_clear = 0.0;
        double packed_wlane16_export_integral = 0.0;
        double packed_wlane16_export_equations = 0.0;
        double packed_wlane16_scan = 0.0;
        double packed_wlane16_restore = 0.0;
        std::vector<double> def_eq;
        std::vector<int> def_calls;
        std::vector<double> apply_eq;
        std::vector<int> apply_calls;
        std::vector<double> packed_wlane8_export_eq;
        std::vector<int> packed_wlane8_export_calls;
        std::vector<double> packed_wlane16_export_eq;
        std::vector<int> packed_wlane16_export_calls;
        int packed_wlane32_calls = 0;
        double packed_wlane32_total = 0.0;
        double packed_wlane32_prepare = 0.0;
        double packed_wlane32_seed = 0.0;
        double packed_wlane32_metric_update = 0.0;
        double packed_wlane32_def_update = 0.0;
        double packed_wlane32_def_clear = 0.0;
        double packed_wlane32_def_compute = 0.0;
        double packed_wlane32_equation_filter = 0.0;
        double packed_wlane32_apply = 0.0;
        double packed_wlane32_export = 0.0;
        double packed_wlane32_export_clear = 0.0;
        double packed_wlane32_export_integral = 0.0;
        double packed_wlane32_export_equations = 0.0;
        double packed_wlane32_scan = 0.0;
        double packed_wlane32_restore = 0.0;
        std::vector<double> packed_wlane32_export_eq;
        std::vector<int> packed_wlane32_export_calls;

        void reset_totals();
    };

    struct DoColJDefDomainCache {
        int dom_min = 0;
        int dom_max = -1;
        int ndef = -1;
        std::vector<std::vector<int>> defs_by_dom;
        std::vector<int> defs_global;
    };

    struct DoColJDefDepCache {
        struct CachedTauSeed {
            int term_index = -1;
            int domain = -1;
            int basis_mode = -1;
            bool supported = false;
            TauSeedDescriptor descriptor;
        };

        int ndef = -1;
        int nbr_unknowns = -1;
        int nterm = -1;
        int nvar = -1;
        int nvar_double = -1;
        int neq = -1;
        bool base_defs_ready = false;
        std::vector<std::vector<int>> def_dep_ids;
        std::vector<std::vector<int>> equation_dep_ids;
        std::vector<int> col_var_id;
        std::vector<int> col_term_idx;
        std::vector<int> term_start_col;
        std::vector<unsigned char> col_is_var_domain;
        std::vector<unsigned char> col_is_var_double;
        std::vector<CachedTauSeed> col_tau_seed;
        bool tau_seed_descriptors_ready = false;
        int nvar_domain = -1;
        bool prev_was_var_domain = false;
    };

    struct ActiveEquationRange {
        int equation_index = -1;
        int operator_offset = 0;
        int row_begin = 0;
        int row_end = 0;
    };

    struct DoColJState {
        DoColJTimingState timing;
        DoColJDefDomainCache def_domain_cache;
        DoColJDefDepCache def_dep_cache;
        std::unique_ptr<Array<double>> column_buffer;
        int column_buffer_size = 0;
        std::unique_ptr<Array<double>> packed_first_buffer;
        std::unique_ptr<Array<double>> packed_second_buffer;
        int packed_buffer_size = 0;
        // Extra packed-output buffers for lanes 2..31 (W=4/W=8/W=16/W=32 paths). Lazily
        // allocated to nbr_conditions on the first wide column. Lanes 0 and 1
        // continue to use packed_first_buffer / packed_second_buffer for
        // hot-path layout compatibility with W=2.
        std::array<std::unique_ptr<Array<double>>, 30> packed_extra_buffers;
        int packed_extra_buffer_size = 0;
        std::vector<unsigned char> dom_affected;
        std::vector<unsigned char> eq_affects;
        std::vector<ActiveEquationRange> active_equation_ranges;
    };

  public:
    // Per-assembler cache, scratch, and profiling state. Separate workspaces
    // expose the state that can be made worker-local; they do not by
    // themselves make the mutable System_of_eqs residual graph concurrent.
    class Workspace {
      public:
        Workspace() = default;
        Workspace(Workspace&&) noexcept = default;
        Workspace& operator=(Workspace&&) noexcept = default;
        Workspace(const Workspace&) = delete;
        Workspace& operator=(const Workspace&) = delete;

      private:
        DoColJState state_;

        friend class JacobianColumnEngine;
        friend class JacobianColumnEngineTestHelper;
    };

  private:

    // Internal scratch buffer shared by buffered + sparse paths.
    Array<double>& do_col_J_buffer();
    Array<double>& do_col_J_buffer(Workspace& workspace);
    void refresh_definition_filter_caches(Workspace& workspace);

    // Unified packed-column sparse engine. The public compute_packed_wlane2 /
    // _wlane4 / _wlane8 / _wlane16 / _wlane32 entry points are thin explicit-instantiation
    // forwarders over this single body (W in {2,4,8,16,32}); it produces byte-for-byte
    // identical COO output to the former per-width copies.
    template <int W>
    bool compute_packed_wlaneN_columns_sparse(Workspace& workspace,
                                              PackedJacobianColumns<W> columns,
                                              double drop_tol,
                                              PackedSparseColumnEmitters<W> emitters,
                                              std::string& failure_reason,
                                              JacobianSelectedRows selected_rows);

    // Unified packed-column verification oracle. The public _wlane2 / _wlane4 /
    // _wlane8 / _wlane16 / _wlane32 _for_verification entry points are thin forwarders over
    // this body (W in {2,4,8,16,32}); seeds W columns into independent lanes, runs
    // one full unfiltered residual traversal, and exports all W dense columns.
    // Prepare/restore match the production sparse path exactly so it is a
    // faithful parity reference.
    template <int W>
    bool compute_packed_wlaneN_columns_for_verification(PackedJacobianColumns<W> columns,
                                                        PackedJacobianOutputs<W> outputs,
                                                        std::string& failure_reason);

    System_of_eqs& sys_;
    mutable Workspace default_workspace_;

    friend class JacobianColumnEngineTestHelper;
};

} // namespace Kadath
