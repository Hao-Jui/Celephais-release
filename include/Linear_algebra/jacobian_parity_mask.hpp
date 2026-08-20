/*
    Added 2026.

    y = 0 reflection-parity sectors of the assembled Jacobian.

    The nosym domains store the azimuthal dependence in a unified COSSIN series:
    index k even holds cos((k/2).phi) and index k odd holds sin(((k-1)/2).phi)
    (summation_1d.cpp:176).  Spherical charts map y -> -y to phi -> -phi,
    whereas bispherical charts map it to phi -> pi-phi; each Domain maps its
    coefficient index to the corresponding parity.  Combined with the field
    grading of a y-symmetric configuration (spin axis in the x-z plane) this
    splits every unknown into a symmetric sector S and an antisymmetric sector
    A, and the Jacobian block-decouples: the S-row / A-column blocks hold
    nothing but roundoff.

    The default-on SPARSE_PARITY_MASK control makes the assembler drop
    those cross-sector entries before the COO gather, so MUMPS factors two
    independent halves; the exact value 0 opts out.  The full-J path measures
    the coupling before masking.  Sparse-direct and JFNK sector reduction may
    instead certify the structural plan before J1 when all descriptors are
    available and the full entry residual satisfies its dual guard; otherwise
    they fall back to the measured full-J path.
*/

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace Kadath
{
    class System_of_eqs;

    /// Intrinsic y-reflection grading of a field component or numeric global.
    /// These pure lookups are exposed so additions to the parity tables can be
    /// covered without constructing a complete equation system.
    int jacobian_parity_field_grading(const std::string& variable_name,
                                      int component);
    int jacobian_parity_global_grading(const std::string& variable_name);

    /// Column grading under the y -> -y involution, one entry per Jacobian
    /// column.  \c sector is +1 for the symmetric sector and -1 for the
    /// antisymmetric one; \c phi_index is the azimuthal coefficient index
    /// (-1 when the column carries none, -2 when it straddles two).  An empty
    /// sector table carrying either unsupported marker is a fail-closed refusal,
    /// not a partial grading.
    struct JacobianParityColumnGrading
    {
        std::vector<signed char> sector;
        std::vector<int> phi_index;
        std::vector<int> phi_basis;
        std::vector<int> phi_domain;
        std::vector<int> component_index;
        std::vector<std::string> ungraded_names;
        long long ungraded_columns = 0;
        long long mixed_phi_columns = 0;
        long long unsupported_phi_basis_columns = 0;
        int unsupported_tau_seed_domain = -1;
        bool unsupported_variable_domain_layout = false;
    };

    /// Grade every Jacobian column of \c system.  Deterministic and
    /// communication-free, so every rank builds the identical table.
    JacobianParityColumnGrading
    grade_jacobian_parity_columns(const System_of_eqs& system);

    /// Empty when the grading is safe to use; otherwise the single fail-closed
    /// reason that the assembler and diagnostics must report.
    std::string jacobian_parity_column_grading_disable_reason(
        const JacobianParityColumnGrading& grading);

    /** Partial structural row grading.  Unavailable descriptors retain sector
     * zero; callers may use the table pre-J1 only when all_rows_available. */
    struct JacobianParityRowPrediction
    {
        std::vector<signed char> sector;
        bool all_rows_available = false;
        long long unavailable_rows = 0;
        long long ungraded_rows = 0;
        long long unsupported_phi_basis_rows = 0;
    };

    JacobianParityRowPrediction
    predict_jacobian_parity_rows(System_of_eqs& system);

    /// Source selected for the full-J mask's row grading.  Structural rows
    /// take the first pass whenever their descriptors cover the whole system
    /// and every column has a valid grading.  A non-exact structural
    /// measurement is re-graded from matrix mass; matrix mass also remains the
    /// fail-closed fallback for incomplete or ungraded descriptors.
    struct JacobianParityRowGradingSelection
    {
        enum class Source {
            Structural,
            MatrixDerivedSecondPass,
            MatrixDerivedFallback
        };

        Source source = Source::MatrixDerivedFallback;
        std::vector<signed char> sector;
        std::string fallback_reason;
    };

    JacobianParityRowGradingSelection select_jacobian_parity_row_grading(
        const JacobianParityRowPrediction& structural_prediction,
        const std::vector<signed char>& column_sector,
        const std::vector<double>& row_mass_symmetric,
        const std::vector<double>& row_mass_antisymmetric);

    /// Replace a complete structural first-pass grading with the matrix-derived
    /// grading when its measured cross ratio is not exact (< 1e-12).  Returns
    /// true only when the caller must re-measure the cross ratio.
    bool regrade_jacobian_parity_rows_after_structural_measurement(
        JacobianParityRowGradingSelection& selection,
        double structural_maximum_cross, double maximum_entry,
        const std::vector<double>& row_mass_symmetric,
        const std::vector<double>& row_mass_antisymmetric);

    struct JacobianParityRowOracleComparison
    {
        bool exact_on_covered_rows = false;
        bool whole_fixture_covered = false;
        long long compared_rows = 0;
        long long unavailable_rows = 0;
        long long mismatched_rows = 0;
        int first_mismatch = -1;
        std::string failure_reason;
    };

    JacobianParityRowOracleComparison compare_jacobian_parity_row_prediction(
        const JacobianParityRowPrediction& prediction,
        const std::vector<signed char>& matrix_row_sector);

    /// Symmetry-neutral description of one selected square Jacobian block.
    /// The label is deliberately opaque to consumers: only the adapter that
    /// constructs the plan assigns it physical meaning.  A plan has no
    /// mutators and is shared as const for the lifetime of one solve.
    class JacobianSelectionPlan
    {
      public:
        using BlockLabel = int;

        JacobianSelectionPlan(BlockLabel selected_block,
                              std::vector<int> selected_rows,
                              std::vector<int> selected_columns);

        BlockLabel selected_block() const noexcept { return selected_block_; }
        const std::vector<int>& selected_rows() const noexcept
        {
            return selected_rows_;
        }
        const std::vector<int>& selected_columns() const noexcept
        {
            return selected_columns_;
        }

      private:
        const BlockLabel selected_block_;
        const std::vector<int> selected_rows_;
        const std::vector<int> selected_columns_;
    };

    /// Result of validating a two-block partition.  Failure is a permanent
    /// masked-full fallback for the solve and carries the reason to log.
    struct JacobianSelectionPlanBuild
    {
        std::shared_ptr<const JacobianSelectionPlan> plan;
        std::string fallback_reason;
    };

    /// Build a selection plan from opaque row and column block labels.  Every
    /// label must match exactly one of the two declared labels, and the chosen
    /// block must contain the same nonzero number of rows and columns.
    JacobianSelectionPlanBuild make_jacobian_selection_plan(
        const std::vector<JacobianSelectionPlan::BlockLabel>& row_block_labels,
        const std::vector<JacobianSelectionPlan::BlockLabel>& column_block_labels,
        JacobianSelectionPlan::BlockLabel selected_block,
        JacobianSelectionPlan::BlockLabel excluded_block);

    /// Validated selected values from a full vector.  The pure gather/scatter
    /// helpers below are shared by the sparse-direct RHS and correction paths.
    struct JacobianSelectedValues
    {
        std::vector<double> values;
        std::string failure_reason;

        explicit operator bool() const noexcept
        {
            return failure_reason.empty();
        }
    };

    JacobianSelectedValues gather_jacobian_selected_values(
        std::span<const double> full_values,
        const std::vector<int>& selected_indices);
    JacobianSelectedValues scatter_jacobian_selected_values(
        std::span<const double> selected_values, int full_size,
        const std::vector<int>& selected_indices);

    struct JacobianSelectionNorms
    {
        double active_linf = 0.0;
        double forbidden_linf = 0.0;
        std::string failure_reason;

        explicit operator bool() const noexcept
        {
            return failure_reason.empty();
        }
    };

    /// Infinity norms on a selected index set and its complement.  Non-finite
    /// input is a refusal, not an accidentally passing comparison.
    JacobianSelectionNorms measure_jacobian_selection_norms(
        std::span<const double> full_values,
        const std::vector<int>& selected_indices);

    struct JacobianPreJ1SelectionPlanBuild
    {
        std::shared_ptr<const JacobianSelectionPlan> plan;
        JacobianSelectionNorms entry_norms;
        double entry_limit = 0.0;
        std::string fallback_reason;

        explicit operator bool() const noexcept
        {
            return static_cast<bool>(plan) && fallback_reason.empty();
        }
    };

    inline constexpr double jacobian_pre_j1_forbidden_relative_tolerance =
        1e-10;
    inline constexpr double jacobian_pre_j1_forbidden_absolute_floor = 1e-12;

    /// Build the structural +1 block before any Jacobian is assembled.  Every
    /// row descriptor and column grading must be available, the block must be
    /// square, and the full entry residual must satisfy the dual entry guard.
    ///
    /// A relative-only guard is marginal for certified warm entries: the boost
    /// stage enters at active=4.548e-4 and forbidden about 2.5e-14, a ratio of
    /// 5.5e-11 within a factor two of the 1e-10 line.  The 1e-12 absolute floor
    /// cleanly admits roundoff-scale forbidden entries (about 1e-14) while
    /// refusing the much larger adot-sourced ECC_RED odd entry residual.
    JacobianPreJ1SelectionPlanBuild make_jacobian_pre_j1_selection_plan(
        const JacobianParityRowPrediction& row_prediction,
        const JacobianParityColumnGrading& column_grading,
        const double* full_entry_residual, int full_size);

    struct JacobianForbiddenResidualCheck
    {
        bool allowed = false;
        double limit = 0.0;
    };

    inline constexpr double jacobian_forbidden_residual_baseline_multiplier =
        10.0;
    /// Install the first finite forbidden norm as this solve's baseline.  That
    /// first check passes; later checks admit at most ten times the baseline.
    /// The returned limit is carried into the reduced-step diagnostic log.
    JacobianForbiddenResidualCheck check_jacobian_forbidden_residual(
        const JacobianSelectionNorms& norms, double& forbidden_baseline,
        bool& forbidden_baseline_installed) noexcept;

    inline constexpr double jacobian_inactive_state_drift_tolerance = 1e-14;
    double install_or_measure_jacobian_inactive_state_drift(
        const std::vector<double>& current, std::vector<double>& baseline,
        bool& baseline_installed);
    bool jacobian_inactive_state_drift_allowed(double drift_linf) noexcept;

    /// A retained factor is valid only for the exact immutable plan object and
    /// dimension that produced it.  Null/null represents the ordinary full
    /// system role.
    bool jacobian_selection_factor_compatible(
        const std::shared_ptr<const JacobianSelectionPlan>& retained_plan,
        int retained_dimension,
        const std::shared_ptr<const JacobianSelectionPlan>& current_plan,
        int current_dimension) noexcept;

    /// Parity tables retained for the whole solve: the unknown count is fixed
    /// within one Newton loop, so structural grading can install the plan
    /// before J1 and reuse it for every later Jacobian.  A full-J path first
    /// measures complete structural rows independently of whether sector
    /// reduction was requested, then re-grades non-exact cases from the matrix.
    /// Matrix grading also remains the descriptor-unavailable fallback.
    struct JacobianParityMaskState
    {
        enum class Decision { Undecided, Disabled, Engaged };
        enum class ReductionDecision {
            Undecided,
            Eligible,
            MaskedFullFallback
        };

        Decision decision = Decision::Undecided;
        ReductionDecision reduction_decision = ReductionDecision::Undecided;
        // Measured first-J cross ratio and engagement mode, retained for later
        // emission decisions and opt-in diagnostics. The ordinary runtime log
        // reports only whether the mask is on or off.
        double engaged_cross_ratio = -1.0;
        bool approximate_engagement = false;
        // Fused emission is permitted only after one full Jacobian established
        // the sticky decision from complete structural labels.
        bool structural_labels_available = false;
        bool unmasked_full_j_emitted = false;
        std::string row_grading_source_label;
        int n = 0;
        std::vector<signed char> column_sector;
        std::vector<signed char> row_sector;
        std::shared_ptr<const JacobianSelectionPlan> selection_plan;
        std::vector<double> inactive_state_baseline;
        bool inactive_state_baseline_installed = false;
        double forbidden_baseline = 0.0;
        bool forbidden_baseline_installed = false;
        std::string reduction_fallback_reason;
    };

    /// Shared by post-hoc retention and the optional emission-time mask.
    inline bool jacobian_parity_entry_retained(signed char column_sector,
                                               signed char row_sector) noexcept
    {
        return column_sector == row_sector;
    }

    /// True when \c sector contains exactly \c dimension complete +/-1 labels.
    inline bool jacobian_parity_sector_labels_complete(
        const std::vector<signed char>& sector, int dimension) noexcept
    {
        if (dimension <= 0 ||
            sector.size() != static_cast<std::size_t>(dimension)) {
            return false;
        }
        for (signed char label : sector)
            if (label != 1 && label != -1)
                return false;
        return true;
    }

    /// True when an engaged, non-verification mask emitted a complete pair of
    /// parity blocks addressed by valid +/-1 row and column labels.
    inline bool jacobian_parity_mask_emission_is_block_diagonal(
        const JacobianParityMaskState& state, int dimension,
        bool mask_engaged, bool verify_emission_active) noexcept
    {
        if (!mask_engaged || verify_emission_active ||
            state.n != dimension ||
            state.decision != JacobianParityMaskState::Decision::Engaged ||
            !jacobian_parity_sector_labels_complete(state.column_sector,
                                                     dimension) ||
            !jacobian_parity_sector_labels_complete(state.row_sector,
                                                     dimension)) {
            return false;
        }
        return true;
    }

    /// True only after a full unmasked emission established an engaged decision
    /// from complete structural +/-1 label tables for this stage.
    inline bool jacobian_fused_parity_mask_ready(
        const JacobianParityMaskState& state, int dimension) noexcept
    {
        if (state.n != dimension ||
            state.decision != JacobianParityMaskState::Decision::Engaged ||
            !state.structural_labels_available || !state.unmasked_full_j_emitted ||
            !jacobian_parity_sector_labels_complete(state.column_sector,
                                                     dimension) ||
            !jacobian_parity_sector_labels_complete(state.row_sector,
                                                     dimension)) {
            return false;
        }
        return true;
    }

    /// Predict whether the next full-dimensional assembly will emit two
    /// disconnected parity blocks. A requested fused verification is active
    /// only when fused emission itself is ready; otherwise the ordinary
    /// post-hoc mask still produces a block-diagonal matrix.
    inline bool jacobian_parity_split_ready_for_next_emission(
        const JacobianParityMaskState& state, int dimension,
        bool parity_mask_requested, bool fused_parity_mask_requested,
        bool fused_verify_requested) noexcept
    {
        const bool fused_verify_active =
            fused_parity_mask_requested && fused_verify_requested &&
            jacobian_fused_parity_mask_ready(state, dimension);
        return jacobian_parity_mask_emission_is_block_diagonal(
            state, dimension, parity_mask_requested, fused_verify_active);
    }

    enum class JacobianEmissionKind { ReducedPlus, FusedPair, Combined };

    /// Driver capabilities that can veto a physical parity payload without
    /// changing the selected or fused assembly strategy.  The direct-MUMPS
    /// special paths require the legacy top-level COO, and the parity-mass
    /// oracle requires full-J grading instead of selected-block assembly.
    struct JacobianEmissionCaps
    {
        bool selected_block_supported = false;
        bool physical_payload_supported = false;
        bool analyze_reuse_requested = false;
        bool replay_capture_requested = false;
        bool parity_mass_probe_requested = false;
    };

    struct JacobianEmissionBlockFingerprint
    {
        int parity_label = 0;
        std::vector<int> selected_rows;
        std::vector<int> selected_columns;

        bool operator==(const JacobianEmissionBlockFingerprint&) const = default;
    };

    /// Collision-free retained identity of an emission plan.  Sector tables
    /// remain value-owned because a digest alone could permit stale JFNK
    /// preconditioner reuse after a parity-sector change.
    struct JacobianEmissionFingerprint
    {
        JacobianEmissionKind kind = JacobianEmissionKind::Combined;
        int full_dimension = 0;
        int assembled_dimension = 0;
        bool selection_plan_requested = false;
        bool parity_mask_requested = false;
        bool fused_parity_mask_requested = false;
        bool fused_emission_active = false;
        bool fused_verify_active = false;
        bool speculative_j1_fusion = false;
        bool local_coo_blocks_requested = false;
        bool physical_block_emission_requested = false;
        bool parity_split_requested = false;
        bool parity_split_ready = false;
        JacobianEmissionBlockFingerprint assembly_block;
        std::array<JacobianEmissionBlockFingerprint, 2> payload_blocks;
        std::vector<signed char> row_sector;
        std::vector<signed char> column_sector;

        bool operator==(const JacobianEmissionFingerprint&) const = default;
    };

    /// One per-step decision shared by the Newton driver, assembler, and MUMPS
    /// payload layer.  The builder may install a parity-mask state slot when a
    /// first-J structural grading makes speculative fused emission viable.
    /// A failed post-emission certification demotes this mutable plan to
    /// Combined while the assembler retries the same Jacobian unmasked.
    struct JacobianEmissionPlan
    {
        using Kind = JacobianEmissionKind;

        Kind kind = Kind::Combined;
        std::array<std::shared_ptr<const JacobianSelectionPlan>, 2> block_plans;
        std::shared_ptr<const JacobianSelectionPlan> assembly_selection_plan;
        std::string refusal_reason;
        bool selection_plan_requested = false;
        bool parity_mask_requested = false;
        bool fused_parity_mask_requested = false;
        bool fused_emission_active = false;
        bool fused_verify_active = false;
        bool speculative_j1_fusion = false;
        bool local_coo_blocks_requested = false;
        bool physical_block_emission_requested = false;
        bool centralized_coo_diagnostic_requested = false;
        JacobianEmissionFingerprint fingerprint;

        /// Route a physical payload through the legacy top-level COO while
        /// retaining the selected/fused assembly strategy.
        void route_payload_to_combined(std::string reason);

        /// Demote a mispredicted speculative fused assembly before retrying it
        /// unmasked through the Combined lane.
        void demote_fused_emission(std::string reason);

        /// A certified first-J fusion has the same retained identity as later
        /// ready-state fused emissions; its speculative status is no longer a
        /// preconditioner distinction after certification succeeds.
        void accept_speculative_fused_emission() noexcept;
    };

    JacobianEmissionPlan plan_jacobian_emission(
        System_of_eqs& system, int dimension,
        const std::shared_ptr<const JacobianSelectionPlan>& step_selection_plan,
        const JacobianEmissionCaps& caps);

    /// A declined packed engine is safe to retry scalar when every lane retained
    /// zero entries. Cross-sector candidates skipped by fusion do not count.
    inline std::size_t jacobian_packed_retained_entry_count(
        std::span<const std::size_t> lane_entry_counts) noexcept
    {
        std::size_t total = 0;
        for (const std::size_t count : lane_entry_counts)
            total += count;
        return total;
    }

    /// Largest tolerated ratio of the biggest cross-sector entry to the
    /// biggest entry of the Jacobian.  Above it the two sectors are genuinely
    /// coupled and the mask stays off.
    inline constexpr double jacobian_parity_cross_tolerance = 1e-12;

    /// Largest ratio accepted as approximate y-reflection parity.  This fixed
    /// policy admits noise-scale centre-of-mass drift while refusing the
    /// warm-start asymmetry scale.
    inline constexpr double jacobian_parity_approximate_cross_tolerance = 1e-5;

    /// Disable the mask and report \c reason once on rank 0.
    void disable_jacobian_parity_mask(JacobianParityMaskState& state,
                                      const std::string& reason, int rank);

    /// Grade the rows from the communicator-summed per-row sector masses: a row
    /// belongs to the sector carrying most of its mass.  Derived from the
    /// matrix rather than assumed, so it cannot invent a violation.
    void
    derive_jacobian_parity_row_sectors(JacobianParityMaskState& state,
                                       const std::vector<double>& row_mass_symmetric,
                                       const std::vector<double>& row_mass_antisymmetric);

    /// Engage the mask only when the two sectors are square and the largest
    /// cross-sector entry is at roundoff level, or is no larger than the fixed
    /// approximate-parity allowance.  \c maximum_cross and \c maximum_entry are
    /// the communicator-reduced maxima of the first Jacobian.  Reports one line
    /// on rank 0 either way.
    void decide_jacobian_parity_mask(JacobianParityMaskState& state,
                                     double maximum_cross, double maximum_entry,
                                     int rank);

    /// Adapt an engaged full-J parity measurement into the phase-1 fallback's
    /// neutral selection plan.  Reduction requires strict exact parity (cross
    /// ratio < 1e-12); approximate mask engagement is intentionally ineligible.
    /// The +1 block is selected, so -1-graded globals and their paired rows are
    /// excluded together instead of disqualifying the solve.
    /// The decision is sticky: later calls cannot replace a per-solve plan or
    /// fallback.
    void decide_jacobian_parity_reduction(JacobianParityMaskState& state,
                                          double maximum_cross,
                                          double maximum_entry, int rank);

    /// Permanently abandon reduction for this solve while preserving an
    /// already-engaged parity mask.  Repeated calls are silent.
    void abandon_jacobian_parity_reduction(JacobianParityMaskState& state,
                                           const std::string& reason,
                                           int rank);
} // namespace Kadath
