/*
    Copyright 2017 Philippe Grandclement & Gregoire Martinon

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/Matrice/matrice.hpp"
#include "For_Kadath/List_comp/list_comp.hpp"
#include "For_Kadath/Array/memory.hpp"
#include "solver_runtime_config.hpp"

#include <array>
#include "jacobian_column_engine.hpp"
#include "Jacobian/column_types.hpp"
#include "Jacobian/tagged_metadata.hpp"
#include "Linear_algebra/linear_solver.hpp"

#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <memory>
#include <cstddef>
#include <ostream>

namespace Kadath
{

    class Equation;
    class Eq_int;
    struct AssembledJacobianCoo; // Linear_algebra/jacobian_assembler.hpp
    struct JacobianAssemblerStructuralPlanCache;
    struct JacobianParityMaskState; // Linear_algebra/jacobian_parity_mask.hpp
    struct JacobianEmissionFingerprint;
    class JacobianSelectionPlan;

    using OpeUserFn = Term_eq (*)(const Term_eq&, Param*);
    using OpeUserBinFn = Term_eq (*)(const Term_eq&, const Term_eq&, Param*);

    /// Decoded identity of one Jacobian column for the p-coarse two-level PC
    /// probe (diagnostic-only; src/Newton/pcoarse_probe_maps.cpp). block: 0 =
    /// variable-domain surface, 1 = numeric (var_double), 2 = spectral field.
    /// For field columns (comp,dom,i,j,k) is the primary tau coefficient; for
    /// var-domain columns (comp = star, j, k) is the adapted-surface mode.
    struct PcoarseColumnKey
    {
        int block = -1;
        std::string name;
        int comp = -1;
        int dom = -1;
        int i = -1;
        int j = -1;
        int k = -1;
    };

    /// Decoded identity of one Jacobian row for the p-coarse probe. Integral
    /// (Eq_int) rows carry expr and are matched by expression; field rows carry
    /// (eq_index,comp,dom,i,j,k,tag). decoded=false marks a row the enc replay
    /// could not resolve (excluded from the transfer map, reported honestly).
    struct PcoarseRowKey
    {
        int eq_index = -1;
        int comp = -1;
        int dom = -1;
        int i = -1;
        int j = -1;
        int k = -1;
        std::string tag;      // bulk | boundary | galerkin | eq_int
        std::string expr;     // Eq_int expression (empty for field rows)
        std::string taxonomy; // RowTaxonomy name (report-only)
        bool decoded = false;
    };

    /**
     * Class used to describe and solve a system of equations. It is the central object of Kadath.
     * The equations are solved between the domains \c dom_min and \c dom_max .
     * The various quantities are given names (char*) that are used when passing the equations to the \c System_of_eqs.
     * \ingroup systems
     */

    class System_of_eqs
    {
      protected:
        const Space& espace; ///< Associated \c Space
        int dom_min;         ///< Smallest domain number
        int dom_max;         ///< Highest domain number
        int ndom;            ///< Number of domains used.

        int nvar_double;                    ///< Number of unknowns that are numbers (i.e. not fields)
        std::vector<double*> var_double;     ///< Non-owning pointers on numeric unknowns.
        std::vector<std::string> names_var_double; ///< Names of the numeric unknowns.

        int nvar;                    ///< Number of unknown fields.
        std::vector<Tensor*> var;     ///< Non-owning pointers on unknown fields.
        std::vector<std::string> names_var; ///< Names of the unknown fields.

        int nterm_double;            ///< Number of \c Term_eq corresponding to the unknowns that are numbers.
        std::vector<std::unique_ptr<Term_eq>>
            term_double;             ///< Owning \c Term_eq entries for the unknowns that are numbers.
                                     ///< Indexed by \c pos = which*ndom + (dd - dom_min); grows by ndom per add_var.
        std::vector<int> assoc_var_double; ///< Correspondance with the \c var_double pointers.

        int nterm;            ///< Number of \c Term_eq corresponding to the unknown fields.
        std::vector<std::unique_ptr<Term_eq>>
            term;             ///< Owning \c Term_eq entries for the unknown fields.
                              ///< Indexed by \c pos = which*ndom + (dd - dom_min); grows by ndom per add_var.
        std::vector<int> assoc_var; ///< Correspondance with the \c var pointers.

        int ncst;         ///< Number of constants passed by the user.
        int nterm_cst;    ///< Number of \c Term_eq coming from the constants passed by the user.
        std::vector<std::unique_ptr<Term_eq>>
            cst;          ///< Owning \c Term_eq entries coming from constants passed by the user.
                          ///< Indexed by \c pos = which*ndom + (dd - dom_min); grows by ndom per add_cst.
        std::vector<std::string> names_cst; ///< Names of the constants passed by the user.

        mutable int ncst_hard; ///< Number of constants generated on the fly (when encoutering things like "2.2" etc...)
        mutable std::vector<std::unique_ptr<Term_eq>>
            cst_hard; ///< Owning \c Term_eq entries for constants generated on the fly (e.g. "2.2"). Sequential append.
        mutable std::vector<double>
            val_cst_hard; ///< Values of the constants generated on the fly (when encoutering things like "2.2" etc...)

        // Definitions (liste of operators I guess)
        int ndef;         ///< Number of definitions.
        std::vector<std::unique_ptr<Ope_def>>
            def;          ///< Owning \c Ope_def entries needed to compute the result. Indexed by name-lookup id < ndef.
        std::vector<std::string> names_def; ///< Names of the definitions.

        // Definitions globale (liste of operators I guess)
        int ndef_glob; ///< Number of global definitions (the one that require the knowledge of the whole space to give
                       ///< the result, like integrals).
        std::vector<std::unique_ptr<Ope_def_global>>
            def_glob;              ///< Owning global definitions. Indexed by counter < ndef_glob.
        std::vector<std::string> names_def_glob; ///< Names of the global definitions.

        // user defined operators
        int nopeuser; ///< Number of operators defined by the user (single argument).
        std::vector<OpeUserFn> opeuser; ///< Function pointers for user-defined operators (single argument).
        std::vector<Param*> paruser;    ///< Non-owning parameters used by user defined operators (single argument).
        std::vector<std::string> names_opeuser; ///< Names of the user defined operators (single argument).

        int nopeuser_bin; ///< Number of operators defined by the user (with two arguments).
        std::vector<OpeUserBinFn> opeuser_bin; ///< Function pointers for user-defined operators (binary).
        std::vector<Param*> paruser_bin; ///< Non-owning parameters used by user defined operators (with two arguments).
        std::vector<std::string> names_opeuser_bin; ///< Names of the user defined operators (with two arguments).

        // The metric (only one for now)
        Metric* met;          ///< Non-owning pointer on the associated \c Metric, if defined.
        std::string name_met; ///< Name by which the metric is recognized.

        int neq_int;     ///< Number of integral equations (i.e. which are doubles)
        std::vector<std::unique_ptr<Eq_int>>
            eq_int;      ///< Owning integral equations. Indexed by counter < \c neq_int; grows on each \c add_eq_int_*.

        int neq;       ///< Number of field equations.
        std::vector<std::unique_ptr<Equation>>
            eq;        ///< Owning field equations. Indexed by counter < \c neq; grows on each \c add_eq_*.

        std::vector<std::unique_ptr<Term_eq>>
            results;   ///< Owning residual \c Term_eq entries; parallel to \c eq, grows alongside it.

        /// Ensure \c eq and \c results have a slot at index \c neq (grow by 1 if needed).
        /// Called at the top of each \c add_eq_* implementation; idempotent within a single call.
        void ensure_eq_slot() {
            if (static_cast<std::size_t>(neq) >= eq.size()) {
                eq.emplace_back(nullptr);
                results.emplace_back(nullptr);
            }
        }
        /// Ensure \c eq_int has a slot at index \c neq_int.
        void ensure_eq_int_slot() {
            if (static_cast<std::size_t>(neq_int) >= eq_int.size()) {
                eq_int.emplace_back(nullptr);
            }
        }

        /// Raw-pointer shadow of \c results, rebuilt before calls to \c Equation::apply / \c export_val / \c
        /// export_der which expect a \c Term_eq**. Ownership is transferred back into \c results by
        /// \c sync_results_shadow() when \c apply() allocates a new slot.
        mutable std::vector<Term_eq*> results_shadow;
        mutable int results_shadow_equation_count = -1;
        mutable std::size_t results_shadow_slot_count = 0;

        int nbr_unknowns; ///< Number of unknowns (basically the number of coefficients of all the unknown fields, once
                          ///< regularities are taken into account).
        int nbr_conditions; ///< Total number of conditions (the number of coefficients of all the equations, once
                            ///< regularities are taken into account).

        std::vector<std::tuple<std::string, int, int>>
            eq_list; ///< List of all equations through all domains, to match indices to expressions.
        std::vector<std::tuple<std::string, int, int>>
            eq_int_list; ///< List of all integral equations through all domains, to match indices to expressions.

        struct EquationColumnAttachment {
            ColumnClass column_class = ColumnClass::Unknown;
            int domain = -1;
            int boundary = -1;
            std::string owner_var_name;
        };
        std::vector<EquationColumnAttachment>
            eq_column_attachments; ///< Column-owner metadata aligned with eq_list/eq.

        std::vector<std::unique_ptr<Index>>
            which_coef; ///< Stores the "true" coefficients on some boundaries (probably deprecated).
                        ///< Owning view; no live call site currently populates this.

        struct NewtonColumnCostState {
            std::vector<double> costs;
            std::vector<int> indices;
            bool initialized = false;
            int nbr_unknowns = -1;

            void reset(int n)
            {
                if (nbr_unknowns != n) {
                    costs.clear();
                    indices.clear();
                    initialized = false;
                    nbr_unknowns = n;
                }
            }
        };
        mutable NewtonColumnCostState newton_column_cost_state;

        struct SeededJacobianProfile {
            double total_s = 0.0;
            double seed_s = 0.0;
            double def_s = 0.0;
            double apply_s = 0.0;
            std::size_t calls = 0;
            std::size_t cols = 0;
            std::size_t seeded_groups = 0;
            std::size_t single_calls = 0;
            std::size_t fallback_calls = 0;
        };

        struct SeededJacobianState {
            ColumnColoring coloring_cache;
            SeededJacobianProfile profile;
            std::vector<int> row_owner_pool;
        };
        mutable SeededJacobianState seeded_jacobian_state;

        struct MumpsRuntimeState {
            int icntl14 = 500;  // fallback only: every System_of_eqs ctor overwrites this with the resolution-keyed seed (mumps_icntl14_seed_for_space, 50..2000); the -9 retry ladder still adapts upward and the success value ratchets within the stage
            bool settings_printed = false;
            // Sparse-direct fixed-threshold and symbolic-reuse state. The COO
            // pattern and its column offsets outlive one Newton step; the solver
            // is declared after the borrowed buffers so reverse member
            // destruction runs JOB_END before those buffers are released.
            SparseDirectDropState sparse_direct_drop_state;
            int sparse_direct_dimension = 0;
            int sparse_direct_ordering = -1;
            int sparse_direct_blr = -1;
            MumpsOutOfCoreMode sparse_direct_out_of_core_mode =
                MumpsOutOfCoreMode::Off;
            double sparse_direct_out_of_core_touch =
                kMumpsOutOfCoreTouchDefault;
            double sparse_direct_out_of_core_safety =
                kMumpsOutOfCoreSafetyDefault;
            double sparse_direct_out_of_core_budget_mb =
                kMumpsOutOfCoreBudgetUnset;
            int sparse_direct_ranks_per_node = -1;
            double sparse_direct_pattern_drop_tol = -1.0;
            bool sparse_direct_analyze_reuse_refused = false;
            long long sparse_direct_analyze_count = 0;
            long long sparse_direct_reuse_count = 0;
            bool sparse_direct_chord_factor_retained = false;
            bool sparse_direct_chord_disabled_for_solve = false;
            bool sparse_direct_chord_exclusion_printed = false;
            bool sparse_direct_chord_entry_checked = false;
            int sparse_direct_consecutive_chord_steps = 0;
            // A reduced-step guard may trip after the full residual is already
            // below the ordinary convergence tolerance.  Preserve the
            // required masked-full rebuild across the next Newton-loop call.
            bool sparse_direct_masked_full_rebuild_pending = false;
            // Exact immutable selection role of retained sparse-direct state.
            // A null plan denotes the ordinary full system.
            std::shared_ptr<const JacobianSelectionPlan>
                sparse_direct_selection_plan;
            int sparse_direct_factor_dimension = 0;
            std::vector<int> sparse_direct_pattern_irn;
            std::vector<int> sparse_direct_pattern_jcn;
            std::vector<long long> sparse_direct_pattern_column_offsets;
            std::vector<double> sparse_direct_aligned_values;
            std::unique_ptr<LinearSolver> sparse_direct_solver;
            int jfnk_step_index = 0;
            int jfnk_preconditioner_dimension = 0;
            int jfnk_preconditioner_step_index = 0;
            int jfnk_preconditioner_refresh_steps = 0;
            int jfnk_preconditioner_ordering = -1;
            int jfnk_preconditioner_blr = -1;
            MumpsOutOfCoreMode jfnk_preconditioner_out_of_core_mode =
                MumpsOutOfCoreMode::Off;
            double jfnk_preconditioner_out_of_core_touch =
                kMumpsOutOfCoreTouchDefault;
            double jfnk_preconditioner_out_of_core_safety =
                kMumpsOutOfCoreSafetyDefault;
            double jfnk_preconditioner_out_of_core_budget_mb =
                kMumpsOutOfCoreBudgetUnset;
            int jfnk_preconditioner_ranks_per_node = -1;
            // Exact value-owned emission identity of the retained factor. The
            // pointed-to fingerprint includes the realized split result and
            // full row/column sector tables, not a collision-prone digest.
            std::shared_ptr<const JacobianEmissionFingerprint>
                jfnk_preconditioner_emission_fingerprint;
            std::shared_ptr<const JacobianSelectionPlan>
                jfnk_preconditioner_selection_plan;
            int jfnk_preconditioner_factor_dimension = 0;
            long long jfnk_preconditioner_nnz = 0;
            std::unique_ptr<LinearSolver> jfnk_preconditioner;
            // Eisenstat-Walker forcing-term state. Kept across Newton steps
            // so the controller can drive eta_k from the residual ratio
            // ||F_k|| / ||F_{k-1}||. Both fields are reset whenever the
            // preconditioner is reset (so a regime change does not poison
            // the next step with stale history).
            double jfnk_eisenstat_walker_prev_residual = -1.0;
            double jfnk_eisenstat_walker_prev_eta = -1.0;
            // Start-of-step nonlinear infinity norm. Used to detect a stale
            // preconditioner before applying another Newton correction.
            double jfnk_previous_start_error = -1.0;
            // Inputs and recovery state for the opt-in cost-aware PC cadence.
            // Times are MPI maximum walls so every rank makes the same policy
            // decision. A recovery retry is consumed by the next call before
            // any nonlinear correction has been applied.
            double jfnk_last_preconditioner_refresh_seconds = -1.0;
            double jfnk_last_krylov_seconds = -1.0;
            int jfnk_last_krylov_iterations = 0;
            int jfnk_last_krylov_status = -1;
            bool jfnk_force_preconditioner_recovery = false;
            // Persistent analysis-pair cache lifecycle is scoped to one
            // independent Newton stage. Entry validation runs before even an
            // already-converged return; a rejected replay is never retried in
            // the same stage.
            bool jfnk_tree_cache_stage_entry_checked = false;
            bool jfnk_tree_cache_replay_disabled = false;
            double jfnk_guard_initial_field_norm = -1.0;
        };
        mutable MumpsRuntimeState mumps_runtime_state;
        // Resolution-keyed ICNTL(14) seed computed once at construction;
        // re-applied by reset_solver_runtime_state so every stage starts from it.
        int mumps_icntl14_seed = 500;

        SolverRuntimeConfig solver_runtime_config = SolverRuntimeConfig::from_environment();
        mutable JacobianColumnEngine jac_col_engine_;
        mutable std::shared_ptr<JacobianAssemblerStructuralPlanCache>
            jacobian_assembler_structural_plan_cache_;
        mutable std::shared_ptr<JacobianParityMaskState>
            jacobian_parity_mask_state_;

        /// Rebuild \c results_shadow as raw \c Term_eq* view over \c results. Returns \c Term_eq**
        /// to pass to \c Equation::apply / \c export_val / \c export_der which still take a C pointer array.
        Term_eq** results_shadow_view();
        /// Transfer ownership of any newly-allocated \c Term_eq* from \c results_shadow back into
        /// the owning \c results unique_ptrs. Call after \c Equation::apply() may have allocated new slots.
        void results_shadow_sync();

      public:
        /**
         * Diagnostic: total discretized conditions contributed by each domain,
         * keyed by domain index. Triggers \c sec_member() to populate per-equation
         * \c n_cond_tot, then sums over all field equations. Used to localize
         * square-system (conditions vs unknowns) imbalances per domain.
         */
        std::map<int, int> conditions_per_domain();

        /**
         * Diagnostic: total unknowns contributed by each domain, keyed by domain
         * index, summed over all unknown fields exactly as the global unknown
         * count is built. Pair with \c conditions_per_domain() to localize
         * per-domain square-system imbalances.
         */
        std::map<int, int> unknowns_per_domain();

        /**
         * Solvers implemented in the do_newton routine
         */
        enum class SOLVER { NEWTON_RAPHSON };
        using RowClass = Kadath::RowClass;
        using RowTaxonomy = Kadath::RowTaxonomy;
        using RowMetadata = Kadath::RowMetadata;
        using TaggedJacobianMetadata = Kadath::TaggedJacobianMetadata;
        using TaggedJacobianMetadataValidation = Kadath::TaggedJacobianMetadataValidation;
        using IncidenceColumnPartition = Kadath::IncidenceColumnPartition;

        /**
         * Prints the structure of the system (unknowns and equations per domain).
         * @param os : output stream.
         */
        void print_system_structure(std::ostream& os) const;

        /**
         * Standard constructor nothing is done. The space is affected and the equations are to be solved in all space.
         * @param so [input] : associated space.
         */
        System_of_eqs(const Space& so);
        /**
         * Constructor, nothing is done. The space is affected and the equations are solved only between two domains.
         * @param so [input] : associated space.
         * @param i [input] : smallest domain number.
         * @param j [input] : highest domain number.
         */
        System_of_eqs(const Space& so, int i, int j);
        /**
         * Constructor, nothing is done. The space is affected and the equations are solved only in one domain.
         * @param so [input] : associated space.
         * @param i [input] : the domain number.
         **/
        System_of_eqs(const Space& so, int i);
        System_of_eqs(const System_of_eqs&); ///< Constructor by copy.
        ~System_of_eqs();                    ///< Destuctor

        const Metric* get_met() const; ///< Returns a pointer on the \c Metric.

        /**
         * Sets the typed runtime configuration used by Newton solver entry points and diagnostics.
         */
        void set_solver_runtime_config(const SolverRuntimeConfig& config)
        {
            solver_runtime_config = config;
        }

        /**
         * Resets per-solve runtime state before an independent configured Newton loop.
         */
        void reset_solver_runtime_state() const
        {
            // A reset starts an independent solve. A residual forwarded by the
            // previous solve therefore cannot describe the new solve's entry
            // state, even when its unknown buffers happen to be unchanged.
            forwarded_residual_.reset();
            jacobian_parity_mask_state_.reset();
            // sparse_direct_solver borrows the pattern vector buffers. Default
            // member-wise assignment visits those vectors before the later
            // solver member, so end MUMPS explicitly before replacing state.
            mumps_runtime_state.sparse_direct_solver.reset();
            mumps_runtime_state = MumpsRuntimeState{};
            mumps_runtime_state.icntl14 = mumps_icntl14_seed;
        }

        /**
         * Returns the runtime configuration currently attached to this system.
         */
        const SolverRuntimeConfig& get_solver_runtime_config() const { return solver_runtime_config; }

        // Accessors
        /**
         * Returns the space.
         */
        const Space& get_space() const { return espace; };
        /**
         * Returns the smallest index of the domains.
         */
        int get_dom_min() const { return dom_min; };
        /**
         * Returns the highest index of the domains.
         */
        int get_dom_max() const { return dom_max; };
        /**
         * Returns the number of conditions.
         */
        int get_nbr_conditions() const { return nbr_conditions; };
        void classify_equation_rows(std::vector<RowClass>& out) const;
        void classify_equation_row_metadata(std::vector<RowMetadata>& out) const;
        void classify_columns(std::vector<ColumnMetadata>& out) const;
        /// Overload reusing a column map already built by build_column_map(...,true),
        /// so a caller that needs both the singleton plan and the classification
        /// does not pay for two identical build_column_map passes.
        void classify_columns(std::vector<ColumnMetadata>& out,
                              const std::vector<ColumnInfo>& column_map) const;
        /// Compose existing row, column, and row-incidence classifiers into one block-extraction substrate.
        void build_tagged_jacobian_metadata(TaggedJacobianMetadata& out,
                                            bool include_row_incidence = true) const;
        TaggedJacobianMetadataValidation
        validate_tagged_jacobian_metadata(const TaggedJacobianMetadata& metadata) const;
        void dump_tagged_jacobian_metadata_summary(std::ostream& os) const;
        /// Emit row_metadata.csv and col_metadata.csv-compatible streams without going through do_newton().
        void dump_tagged_jacobian_metadata_csv(std::ostream& row_os,
                                               std::ostream& column_os) const;
        /**
         * Returns the number of unknowns.
         */
        int get_nbr_unknowns() const { return nbr_unknowns; };

        /// First index in \c eq_int_list whose stored expression contains \c needle,
        /// or -1 if none. Lets callers resolve an integral-equation index by its
        /// expression (e.g. "intMb) = Mb2") instead of a hard-coded position, which
        /// shifts with the active constraint set (spin, nosym momentum rows, ...).
        int eq_int_index_of(const std::string& needle) const {
            for (std::size_t i = 0; i < eq_int_list.size(); ++i)
                if (std::get<0>(eq_int_list[i]).find(needle) != std::string::npos)
                    return static_cast<int>(i);
            return -1;
        }

        /// Number of registered numeric (\c double) variables.
        int get_nvar_double() const { return nvar_double; }
        /// Number of registered tensor (field) variables.
        int get_nvar() const { return nvar; }
        /// Number of registered constants.
        int get_ncst() const { return ncst; }
        /// Number of registered definitions.
        int get_ndef() const { return ndef; }
        /// Number of registered global definitions.
        int get_ndef_glob() const { return ndef_glob; }
        /// Number of registered user-defined unary operators.
        int get_nopeuser() const { return nopeuser; }
        /// Number of registered user-defined binary operators.
        int get_nopeuser_bin() const { return nopeuser_bin; }

        /**
         * Declare that every registered user operator (add_ope) reads ONLY
         * its Term_eq argument and that its Param* holds no field, system, or
         * global-mutable references. This widens the DO_JX_TERM_CLOSURE
         * activation guard so the per-rank term/cst preamble filter activates
         * even when user operators are present. The caller is responsible for
         * auditing every registered operator before calling this; an incorrect
         * declaration silently corrupts the partitioned matvec.
         */
        void declare_user_opes_target_only() { user_opes_target_only_ = true; }

        /**
         * Returns a pointer on a \c Term_eq corresponding to an unknown number.
         * @param which : index of the unknown.
         * @param dd : the index of the \c Domain.
         */
        Term_eq* give_term_double(int which, int dd) const;
        /**
         * Returns a pointer on a \c Term_eq corresponding to an unknown field.
         * @param which : index of the unknown.
         * @param dd : the index of the \c Domain.
         */
        Term_eq* give_term(int which, int dd) const;
        /**
         * Returns a pointer on a \c Term_eq corresponding to a constant.
         * @param which : index of the unknown.
         * @param dd : the index of the \c Domain.
         */
        Term_eq* give_cst(int which, int dd) const;
        /**
         * Returns a pointer on a \c Term_eq corresponding to a constant generated on the fly.
         * @param xx : the value of the constant
         * @param dd : the index of the \c Domain.
         */
        Term_eq* give_cst_hard(double xx, int dd) const;
        /**
         * Returns a pointer on a definition (better to use \c give_val_def if one wants to access the result of some
         * definition).
         * @param  i : the index of the definition (one has to manage the different domains properly...)
         */
        Ope_def* give_def(int i) const;
        /**
         * Returns a pointer on a global definition.
         * @param  i : the index of the definition (one has to manage the different domains properly...)
         */
        Ope_def_global* give_def_glob(int i) const;

        /**
         * Gives the result of a definition. It is set to zero on domains where the definition is undefined.
         * @param name : name of the definition.
         */
        Tensor give_val_def(const char* name) const;

        /**
         * Gives an immutable view of a scalar definition on one of its
         * defining domains, without materializing a full-space Tensor.
         * If the same name is registered more than once on the requested
         * domain, the last registration wins, matching \c give_val_def.
         *
         * The returned reference remains valid only until the next operation
         * that evaluates or otherwise changes this system's definitions.
         * @param name : name of the scalar definition.
         * @param domain : domain on which the definition is registered.
         */
        const Val_domain& give_val_def_scalar_domain(const char* name, int domain) const;

        /**
         * Addition of a variable (number case)
         * @param name : name of the variable (used afterwards by \c System_of_eqs)
         * @param var : variable.
         */
        void add_var(std::string_view name, double& var);
        /**
         * Addition of a variable (field case)
         * @param name : name of the variable (used afterwards by \c System_of_eqs)
         * @param var : variable.
         */
        void add_var(std::string_view name, Tensor& var);
        /**
         * Addition of a variable (field case, anonymous placeholder).
         * Distinct overload because std::string_view has no null state.
         */
        void add_var(std::nullptr_t, Tensor& var);
        /**
         * Addition of a constant (number case)
         * @param name : name of the constant (used afterwards by \c System_of_eqs)
         * @param cst : variable.
         */
        void add_cst(std::string_view name, double cst);
        /**
         * Addition of a constant (field case)
         * @param name : name of the constant (used afterwards by \c System_of_eqs)
         * @param cst : constant.
         */
        void add_cst(std::string_view name, const Tensor& cst);
        /**
         * Addition of a constant (field case, anonymous placeholder).
         */
        void add_cst(std::nullptr_t, const Tensor& cst);

        /**
         * Addition of a definition
         * @param name : string describing the definition (like "A=...")
         */
        void add_def(std::string_view name);
        /**
         * Addition of a definition in a single domain.
         * @param dd : number of the \c Domain.
         * @param name : string describing the definition (like "A=...")
         */
        void add_def(int dd, std::string_view name);
        /**
         * Addition of a global definition
         * @param name : string describing the definition (like "A=...")
         */
        void add_def_global(std::string_view name);
        /**
         * Addition of a global definition in a single domain.
         * @param dd : number of the \c Domain.
         * @param name : string describing the definition (like "A=...")
         */
        void add_def_global(int dd, std::string_view name);
        /**
         * Addition of a user defined operator (one argument version)
         * @param name : name of the operator (used afterwards by \c System_of_eqs)
         * @param pope : pointer on the function describing the action of the  operator.
         * @param par : parameters of the operator.
         */
        void add_ope(std::string_view name, Term_eq (*pope)(const Term_eq&, Param*), Param* par);
        /**
         * Addition of a user defined operator (two arguments version)
         * @param name : name of the operator (used afterwards by \c System_of_eqs)
         * @param pope : pointer on the function describing the action of the  operator.
         * @param par : parameters of the operator.
         */
        void add_ope(std::string_view name, Term_eq (*pope)(const Term_eq&, const Term_eq&, Param*), Param* par);

        /**
         * Check if a string is an unknown (number).
         * @param target : the string to be tested.
         * @param which : the index of the found variable (if found).
         */
        bool isvar_double(const char* target, int& which) const;
        /**
         * Check if a string is an unknown field.
         * @param target : the string to be tested.
         * @param which : the index of the found variable (if found).
         */
        bool isvar(const char*, int&, int&, char*&, Array<int>*&) const;
        /**
         * Check if a string is a constant (can required indices manipulation and/or inner contraction).
         * @param target : the string to be tested.
         * @param which : the index of the found constant (if found).
         * @param valence : valence of the result.
         * @param name_ind : name of the indices of the result.
         * @param type_ind : type of the indices of the result (COV or CON).
         */
        bool iscst(const char* target, int& which, int& valence, char*& name_ind, Array<int>*& type_ind) const;
        /**
         * Check if a string is a definition (can required indices manipulation and/or inner contraction).
         * @param dd : index of the \c Domain.
         * @param target : the string to be tested.
         * @param which : the index of the found definition (if found).
         * @param valence : valence of the result.
         * @param name_ind : name of the indices of the result.
         * @param type_ind : type of the indices of the result (COV or CON).
         */
        bool isdef(int dd, const char* target, int& which, int& valence, char*& name_ind, Array<int>*& type_ind) const;
        /**
         * Check if a string is a global definition.
         * @param dd : index of the \c Domain.
         * @param target : the string to be tested.
         * @param which : the index of the found definition (if found).
         */
        bool isdef_glob(int dd, const char* target, int& which) const;

        /**
         * Checks if a string is a double
         * @param input : the string to be tested.
         * @param output : returns the double
         */
        bool isdouble(const char* input, double& output) const;
        /**
         * Checks if a string is a metric
         * @param input : the string to be tested.
         * @param name_ind : name of the indices of the result.
         * @param type_ind : type of the indices of the result (COV or CON).
         */
        bool ismet(const char* input, char*& name_ind, int& type_ind) const;
        /**
         * Checks if a string is a metric (without arguments, probably deprecated)
         * @param input : the string to be tested.
         */
        bool ismet(const char* input) const;
        /**
         * Checks if a string represents the Christoffel symbols.
         * The reserved word is "Gam"
         * @param input : the string to be tested.
         * @param name_ind : name of the indices of the result.
         * @param type_ind : type of the indices of the result (COV or CON).
         */
        bool ischristo(const char* input, char*& name_ind, Array<int>*& type_ind) const;
        /**
         * Checks if a string represents the Riemann tensor.
         * The reserved word is "R"
         * @param input : the string to be tested.
         * @param name_ind : name of the indices of the result.
         * @param type_ind : type of the indices of the result (COV or CON).
         */
        bool isriemann(const char* input, char*& name_ind, Array<int>*& type_ind) const;
        /**
         * Checks if a string represents the Ricci tensor.
         * The reserved word is "R"
         * @param input : the string to be tested.
         * @param name_ind : name of the indices of the result.
         * @param type_ind : type of the indices of the result (COV or CON).
         */
        bool isricci_tensor(const char* input, char*& name_ind, Array<int>*& type_ind) const;
        /**
         * Checks if a string represents the Ricci scalar.
         * The reserved word is "R"
         * @param input : the string to be tested.
         * @param name_ind : name of the indices of the result (not used in this case).
         * @param type_ind : type of the indices of the result (COV or CON) (not used in this case).
         */
        bool isricci_scalar(const char* input, char*& name_ind, Array<int>*& type_ind) const;
        /**
         * Checks if a string represents an operator of the type "a + b".
         * @param input : the string to be tested.
         * @param p1 : the returned first argument.
         * @param p2 : the returned second argument.
         * @param symb : the symbol representing the operator (can be +, -; * and so on...)
         */
        bool is_ope_bin(const char* input, char* p1, char* p2, char symb) const;

        /**
         * Function that reads a string and returns a pointer on the generated \c Ope_eq.
         * It is higly recursive, calling itself unless the full operator is generated. Returns empty pointer if not
         * found
         * @param dom : number of the \c Domain
         * @param name : the sting to be read.
         * @param bb : boundary, if a boundary is needed (depending on the type of equation considered : bulk vs
         * matching for instance).
         */
        Ope_eq* find_ope(int dom, const char* name, int bb = 0) const;

        /**
         * Function that reads a string and returns a pointer on the generated \c Ope_eq.
         * It is higly recursive, calling itslef unless the full operator is generated or an error encoutered.
         * @param dom : number of the \c Domain
         * @param name : the sting to be read.
         * @param bb : boundary, if a boundary is needed (depending on the type of equation considered : bulk vs
         * matching for instance).
         */
        Ope_eq* give_ope(int dom, const char* name, int bb = 0) const;

        /**
         * Parse an equation to be solved inside a domain.
         * Based on give_ope and gets called by the different add_eq routines.
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param bb : boundary, if a boundary is needed (depending on the type of equation considered : bulk vs
         * matching for instance).
         */
        Ope_eq* parse_eq(int dom, const char* eq, int bb = 0) const;

        /**
         * Parse an expression with or without equal sign.
         * Based on give_ope and gets called by the different add_eq routines.
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param bb : boundary, if a boundary is needed (depending on the type of equation considered : bulk vs
         * matching for instance).
         * @param first : first or second part of the expression involving an equal sign.
         */
        Ope_eq* parse_eq_trim(int dom, const char* eq, int bb = 0, bool first = 0) const;

        /**
         * Maps a \c Term_eq pointer back to an unknown variable name.
         * @param term : term pointer to resolve.
         * @param name : output variable name if found.
         * @return true if the term corresponds to an unknown variable.
         */
        bool term_to_var_name(const Term_eq* term, std::string& name) const;
        Ope_def* def_from_term(const Term_eq* term) const;
        Ope_def_global* def_glob_from_term(const Term_eq* term) const;

        /**
         * Collects variable names used by an equation string.
         * @param dom : domain index.
         * @param eq : equation string.
         * @param bb : boundary (if applicable).
         * @param vars : output set of variable names.
         */
        void collect_vars_for_eq(int dom, const char* eq, int bb, std::set<std::string>& vars) const;

        std::string infer_equation_owner_var_name(int dom, const char* owner_expr, int bb,
                                                  const char* fallback_expr = nullptr) const;
        void record_equation_column_attachment(ColumnClass column_class, int dom, int bb,
                                               const char* owner_expr,
                                               const char* fallback_expr = nullptr);

        /**
         * Builds per-equation variable dependency sets for structural sparsity inspection.
         * @param eq_vars : output vector aligned with eq_list.
         * @param eq_int_vars : output vector aligned with eq_int_list.
         */
        void build_eq_dependencies(std::vector<std::set<std::string>>& eq_vars,
                                   std::vector<std::set<std::string>>& eq_int_vars) const;
        /**
         * Greedy graph coloring based on variable dependencies for sparsity analysis.
         * @param os : output stream.
         */
        void dump_eq_dependency_coloring(std::ostream& os) const;

        /**
         * Addition of an equation to be solved inside a domain (assumed to be second order).
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_inside(int dom, const char* eq, int n_cmp = -1, Array<int>** p_cmp = nullptr,
                           const char* owner_expr = nullptr);

        /**
         * Addition of an equation to be solved inside a domain (assumed to be second order).
         * Version with a list of components
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered
         */
        void add_eq_inside(int dom, const char* eq, const List_comp& list);

        /**
         * Addition of an equation to be solved inside a domain (of arbitrary order).
         * @param dom : number of the \c Domain.
         * @param order : order of the equation.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_order(int dom, int order, const char* eq, int n_cmp = -1, Array<int>** p_cmp = nullptr,
                          const char* owner_expr = nullptr);

        /**
         * Addition of an equation to be solved inside a domain (of arbitrary order).
         * @param dom : number of the \c Domain.
         * @param order : order of the equation.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_order(int dom, int order, const char* eq, const List_comp& list);

        /**
         * Addition of an equation describing a boundary condition.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_bc(int dom, int bb, const char* eq, int n_cmp = -1, Array<int>** p_cmp = nullptr,
                       const char* owner_expr = nullptr);

        void add_eq_bc_projected(int dom, int bb, const char* eq, int n_cmp = -1,
                                 Array<int>** p_cmp = nullptr, const char* owner_expr = nullptr);

        /**
         * Addition of an equation describing a boundary condition.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_bc(int dom, int bb, const char* eq, const List_comp& list);

        /**
         * Addition of an equation describing a matching condition between two domains (standard setting)
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_matching(int dom, int bb, const char* eq, int n_cmp = -1, Array<int>** p_cmp = nullptr,
                             const char* owner_expr = nullptr);

        /**
         * Addition of an equation describing a matching condition between two domains (standard setting)
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_matching(int dom, int bb, const char* eq, const List_comp& list);

        /**
         * Addition of an equation describing a matching condition between two domains (specialized function for time
         * evolution).
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_matching_one_side(int dom, int bb, const char* eq, int n_cmp = -1,
                                      Array<int>** p_cmp = nullptr, const char* owner_expr = nullptr);

        /**
         * Addition of an equation describing a matching condition between two domains (specialized function for time
         * evolution).
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_matching_one_side(int dom, int bb, const char* eq, const List_comp& list);

        /**
         * Addition of an equation describing a matching condition between domains.
         * The matching is performed in the configuration space.
         * It is intended where the collocations points are different at each side of the boundary.
         * It can happen when there are more than one touching domain (bispheric vs spheric) and when the number of
         * points is different.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_matching_non_std(int dom, int bb, const char* eq, int n_cmp = -1,
                                     Array<int>** p_cmp = nullptr, const char* owner_expr = nullptr);

        /**
         * Addition of an equation describing a matching condition between domains.
         * The matching is performed in the configuration space.
         * It is intended where the collocations points are different at each side of the boundary.
         * It can happen when there are more than one touching domain (bispheric vs spheric) and when the number of
         * points is different.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_matching_non_std(int dom, int bb, const char* eq, const List_comp& list);

        /**
         * Addition of an equation describing a matching condition between domains using the ("import" setting)
         * The matching is performed in the configuration space.
         * It is intended where the collocations points are different at each side of the boundary.
         * It can happen when there are more than one touching domain (bispheric vs spheric) and when the number of
         * points is different.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         * @param reflection_parity_preserving : explicit contract that the
         * complete import matching operator preserves the y-reflection sector.
         */
        void add_eq_matching_import(int dom, int bb, const char* eq, int n_cmp = -1,
                                    Array<int>** p_cmp = nullptr, const char* owner_expr = nullptr,
                                    bool reflection_parity_preserving = false);

        /**
         * Addition of an equation describing a matching condition between domains using the ("import" setting)
         * The matching is performed in the configuration space.
         * It is intended where the collocations points are different at each side of the boundary.
         * It can happen when there are more than one touching domain (bispheric vs spheric) and when the number of
         * points is different.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_matching_import(int dom, int bb, const char* eq, const List_comp& list);

        /**
         * Addition of an equation to be solved inside a domain (assumed to be zeroth order i.e. with no derivatives).
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_full(int dom, const char* eq, int n_cmp = -1, Array<int>** p_cmp = nullptr,
                         const char* owner_expr = nullptr);

        /**
         * Addition of an equation to be solved inside a domain (assumed to be zeroth order i.e. with no derivatives).
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_full(int dom, const char* eq, const List_comp& list);

        /**
         * Addition of an equation to be solved inside a domain (assumed to be first order).
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_one_side(int dom, const char* eq, int n_cmp = -1, Array<int>** p_cmp = nullptr,
                             const char* owner_expr = nullptr);

        /**
         * Addition of an equation to be solved inside a domain (assumed to be first order).
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_one_side(int dom, const char* eq, const List_comp& list);

        /**
         * Addition of a matching condition, except for one coefficient where an alternative condition is enforced
         * (highly specialized usage).
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param par : parameters for the exceptional condition (i.e. which coefficient is concerned basically).
         * @param eq_exception : the excpetionnal equation used.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_matching_exception(int dom, int bb, const char* eq, const Param& par, const char* eq_exception,
                                       int n_cmp = -1, Array<int>** p_cmp = nullptr,
                                       const char* owner_expr = nullptr);

        /**
         * Addition of a matching condition, except for one coefficient where an alternative condition is enforced
         * (highly specialized usage).
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param par : parameters for the exceptional condition (i.e. which coefficient is concerned basically).
         * @param eq_exception : the excpetionnal equation used.
         * @param list : list of the components to be considered.
         */
        void add_eq_matching_exception(int dom, int bb, const char* eq, const Param& par, const char* eq_exception,
                                       const List_comp& list);

        /**
         * Addition of an equation to be solved inside a domain of arbitrary order.
         * The order can be different for each variable (first order in time and second in \f$r\f$ for instance).
         * @param dom : number of the \c Domain.
         * @param orders : orders of the equation, for each variable.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_order(int dom, const Array<int>& orders, const char* eq, int n_cmp = -1,
                          Array<int>** p_cmp = nullptr, const char* owner_expr = nullptr);

        /**
         * Addition of an equation to be solved inside a domain of arbitrary order.
         * The order can be different for each variable (first order in time and second in \f$r\f$ for instance).
         * @param dom : number of the \c Domain.
         * @param orders : orders of the equation, for each variable.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_order(int dom, const Array<int>& orders, const char* eq, const List_comp& list);

        /**
         * Addition of an equation for the velocity potential of irrotational binaries.
         * @param dom : number of the \c Domain.
         * @param order : order of the equation.
         * @param eq : string defining the equation.
         * @param const_part : constant par
         * @param same_reflection_sector : explicit contract that the ordinary
         * and constant-part substitutions transform in the same y-reflection
         * sector.
         */
        void add_eq_vel_pot(int dom, int order, const char* eq, const char* const_part,
                            const char* owner_expr = nullptr,
                            bool same_reflection_sector = false);

        /**
         * Addition of an boundary equation with an exception for \f$l=m=0\f$
         * @param dom : number of the \c Domain.
         * @param bound : boundary index.
         * @param eq : string defining the equation.
         * @param const_part : constant par
         */
        void add_eq_bc_exception(int dom, int bound, const char* eq, const char* const_part,
                                 const char* owner_expr = nullptr);

        /**
         * Addition of an equation a boundary condition of arbitrary orders.
         * The order can be different for each variable. It is irrelevant for the variable corresponding to the
         * boundary.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param orders : orders of the equation, for each variable.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_bc(int dom, int bb, const Array<int>& orders, const char* eq, int n_cmp = -1,
                       Array<int>** p_cmp = nullptr, const char* owner_expr = nullptr);

        /**
         * Addition of an equation a boundary condition of arbitrary orders.
         * The order can be different for each variable. It is irrelevant for the variable corresponding to the
         * boundary.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param orders : orders of the equation, for each variable.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_bc(int dom, int bb, const Array<int>& orders, const char* eq, const List_comp& list);

        /**
         * Addition of an equation a matching condition of arbitrary orders.
         * The order can be different for each variable. It is irrelevant for the variable corresponding to the
         * boundary.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param orders : orders of the equation, for each variable.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_matching(int dom, int bb, const Array<int>& orders, const char* eq, int n_cmp = -1,
                             Array<int>** p_cmp = nullptr, const char* owner_expr = nullptr);

        /**
         * Addition of an equation a matching condition of arbitrary orders.
         * The order can be different for each variable. It is irrelevant for the variable corresponding to the
         * boundary.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param orders : orders of the equation, for each variable.
         * @param eq : string defining the equation.
         * @param list : list of the components to be considered.
         */
        void add_eq_matching(int dom, int bb, const Array<int>& orders, const char* eq, const List_comp& list);

        /**
         * Addition of an equation representing a first integral.
         * @param dom : number of the \c Domain.
         * @param eq : string defining the equation.
         * @param nused : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param pused : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        void add_eq_first_integral(int dom, const char* eq, int n_cmp = -1, Array<int>** p_cmp = nullptr);

        /**
         * Addition of an equation representing a first integral.
         * @param dommin : index of the first \d Domain
         * @param dommax : index of the last \d Domain
         * @param integ_part : name of the integral quantity
         * @param const_part : equation fixing the value of the integral.
         * @param same_reflection_sector : explicit contract that the integral
         * and constant parts transform in the same y-reflection sector.
         */
        void add_eq_first_integral(int dom_min, int dom_max, const char* integ_part,
                                   const char* const_part,
                                   bool same_reflection_sector = false);

        /**
         * Addition of an equation prescribing the value of one coefficient of a scalar field, on a given boundary.
         * @param dom : number of the \c Domain.
         * @param bb : the boundary.
         * @param eq : string defining the scalar field.
         * @param pos_cf : which coefficient is used.
         * @param val : the value the coefficient must have.
         */
        void add_eq_mode(int dom, int bb, const char* eq, const Index& pos_cf, double val);
        /**
         * Addition of an equation prescribing the value of one coefficient of a scalar field.
         * @param dom : number of the \c Domain.
         * @param eq : string defining the scalar field.
         * @param pos_cf : which coefficient is used.
         * @param val : the value the coefficient must have.
         */
        void add_eq_val_mode(int dom, const char* eq, const Index& pos_cf, double val);
        /**
         * Addition of an equation saying that the value of a field must be zero at one collocation point.
         * @param dom : number of the \c Domain.
         * @param eq : string defining the scalar field that must vanish.
         * @param pos : which collocation point is used.
         */
        void add_eq_val(int dom, const char* eq, const Index& pos);
        /**
         * Assign y-reflection parity to the most recently registered scalar
         * equation. Accepted sectors are -1 and +1. Repeating the same tag is
         * harmless; changing an existing tag is rejected.
         */
        void set_last_eq_int_reflection_sector(int sector);
        /**
         * Addition of an equation saying that the value of a field must be zero at one point (arbitrary).
         * @param dom : number of the \c Domain.
         * @param eq : string defining the scalar field that must vanish.
         * @param MM : which point is used.
         */
        void add_eq_point(int dom, const char* eq, const Point& MM);

        // Various computational stuff :
        /**
         * Computes the residual of all the equations.
         * This is essentially the value of the biggest coefficient.
         * @returns an array of the error, on all the equations.
         */
        Array<double> check_equations();
        /**
         * Copies the various unknowns (doubles and \c Tensors) into their \c Term_eq counterparts.
         */
        void vars_to_terms();
        /**
         * Computes the second member of the Newton-Raphson equations.
         * It is essentially the coefficients of the residual of all the equations (plus some Galerking issues).
         */
        Array<double> sec_member();
        /**
         * Describe residual rows in canonical sec_member order.  Unsupported
         * equation families remain explicit unavailable placeholders, making a
         * complete pre-Jacobian description fail closed.
         */
        bool describe_residual_rows(
            std::vector<ResidualRowDescriptor>& descriptors);
        const std::string& equation_owner_var_name(int equation_index) const;
        /**
         * Computes one integral-equation residual without materializing the
         * complete residual vector. The unknown/metric preamble is identical
         * to \c sec_member(), but only the selected Eq_int's transitive
         * definition closure is evaluated before the unchanged
         * Eq_int::get_val() reduction.
         *
         * This is a local, non-collective operation intended for targeted
         * scalar probes before Newton entry. Definitions outside the selected
         * closure are not made current; callers must run a full value pass
         * before reading unrelated definitions or entering Newton.
         *
         * @param eq_int_index zero-based index in the integral-equation prefix.
         * @returns the selected integral-equation residual.
         */
        double sec_member_eq_int(int eq_int_index);
        /// Row-partitioned Newton-driver residual: same result as
        /// \c sec_member(), but the definition/apply/export tail reuses the
        /// \c do_JX row partition when it matches the current topology (gated
        /// by \c SEC_MEMBER_MPI; never builds the partition itself).
        ///
        /// PUBLIC WITH A COLLECTIVE CONTRACT (same pattern as the \c do_newton*
        /// entry points): when the partition is active this issues MPI
        /// collectives (\c MPI_Iallgatherv / \c MPI_Allreduce), therefore
        ///   (i)   the caller MUST be all-rank collective — a rank-subset call
        ///         deadlocks;
        ///   (ii)  the sanctioned callers are the all-rank Newton drivers and
        ///         the end-of-iteration \c update_fields forwarded-residual
        ///         refresh (collectivity of all its \c &syst call sites verified
        ///         2026-07-05, every site inside \c run_newton_loop);
        ///   (iii) rank-0-only / diagnostic paths MUST keep using the bare
        ///         \c sec_member() (local, fully replicated);
        ///   (iv)  when the partition is absent/inactive it falls back to the
        ///         replicated \c sec_member() on a rank-uniform branch, so the
        ///         fallback is itself collective-safe.
        /// Implemented in \c assembly/solver.cpp.
        Array<double> sec_member_partitioned();
        /**
         * Sets the values the variation of the fields.
         * @param vder : values of all the variations (essentially the coefficients of all the variations of the
         * unknowns).
         * @param term_filter : optional per-\c term[] mask (size \c nterm) from the \c do_JX
         * per-rank term closure (\c DO_JX_TERM_CLOSURE): masked-out terms skip the
         * tensor allocation + \c affecte_tau + \c set_der_t and their xx block is jumped via
         * \c DoJxMpiPartition::term_xx_stride (the term keeps its previous, unread derivative).
         * The adapted-surface head and the numeric \c var_double singles stay unfiltered.
         */
        void xx_to_ders(const Array<double>&, const std::vector<char>* term_filter = nullptr);
        /**
         * Sets the values the  of the fields.
         * @param val : values of all the fields (essentially the coefficients of all the unknowns).
         * @param conte : current position in the \c Array \c val.
         */
        void xx_to_vars(const Array<double>& val, int& conte);
        /**
         * Applies a Newton update in-place: var = var - delta, where delta is encoded in \c xx.
         * @param xx : values of all the variations (coefficients of all unknowns).
         * @param conte : current position in the \c Array \c xx.
         */
        void xx_to_vars_delta(const Array<double>& xx, int& conte);
        /**
         * Computes the product \f$ J \times x\f$ where \f$J\f$ is the Jacobian of the system.
         * @param xx : the vector \f$ x\f$.
         * @return the product.
         */
        Array<double> do_JX(const Array<double>& xx);
        /**
         * Computes \f$J \times x\f$ into caller-owned storage.
         * @param xx : the vector \f$x\f$.
         * @param out : preallocated one-dimensional output of size nbr_conditions.
         *
         * This overload preserves the return-by-value interface above while allowing
         * iterative solvers to reuse the result buffer across matrix-vector products.
         */
        void do_JX(const Array<double>& xx, Array<double>& out);
        /**
         * Computes one column of the Jacobian.
         * @param i : column number.
         * @return the column.
         */
        Array<double> do_col_J(int i);
        /**
         * Computes one column of the Jacobian into a preallocated buffer.
         * @param i : column number.
         * @param out : output buffer (size nbr_conditions).
         */
        void do_col_J(int i, double* out);
        /**
         * Computes one column of the Jacobian and emits nonzeros via callback.
         * @param i : column number.
         * @param drop_tol : absolute threshold for emitting values.
         * @param emit : callback invoked for each (row, value) pair.
         */
        void do_col_J_sparse(int i, double drop_tol, SparseColumnEmitter emit,
                             JacobianSelectedRows selected_rows = std::nullopt);
        void do_col_J_sparse(JacobianColumnEngine::Workspace& workspace,
                             int i, double drop_tol, SparseColumnEmitter emit,
                             JacobianSelectedRows selected_rows = std::nullopt);
        /**
         * Packed sparse Jacobian entries. These compute multiple independent
         * tangent columns in one residual traversal. A false return means the
         * requested column group falls outside the supported envelope; callers
         * should retry with a narrower packed group or scalar \c do_col_J_sparse.
         */
        bool do_cols_J_wlane2_sparse(int first_column, int second_column,
                                     double drop_tol,
                                     SparseColumnEmitter emit_first,
                                     SparseColumnEmitter emit_second,
                                     std::string& failure_reason,
                                     JacobianSelectedRows selected_rows = std::nullopt);
        bool do_cols_J_wlane4_sparse(const std::array<int, 4>& columns,
                                     double drop_tol,
                                     std::array<SparseColumnEmitter, 4>& emitters,
                                     std::string& failure_reason,
                                     JacobianSelectedRows selected_rows = std::nullopt);
        bool do_cols_J_wlane8_sparse(const std::array<int, 8>& columns,
                                     double drop_tol,
                                     std::array<SparseColumnEmitter, 8>& emitters,
                                     std::string& failure_reason,
                                     JacobianSelectedRows selected_rows = std::nullopt);
        bool do_cols_J_wlane16_sparse(const std::array<int, 16>& columns,
                                      double drop_tol,
                                      std::array<SparseColumnEmitter, 16>& emitters,
                                      std::string& failure_reason,
                                      JacobianSelectedRows selected_rows = std::nullopt);
        bool do_cols_J_wlane32_sparse(const std::array<int, 32>& columns,
                                      double drop_tol,
                                      std::array<SparseColumnEmitter, 32>& emitters,
                                      std::string& failure_reason,
                                      JacobianSelectedRows selected_rows = std::nullopt);

        bool do_cols_J_wlane2_sparse(
            JacobianColumnEngine::Workspace& workspace,
            PackedJacobianColumns<2> columns, double drop_tol,
            PackedSparseColumnEmitters<2> emitters,
            std::string& failure_reason,
            JacobianSelectedRows selected_rows = std::nullopt);
        bool do_cols_J_wlane4_sparse(
            JacobianColumnEngine::Workspace& workspace,
            PackedJacobianColumns<4> columns, double drop_tol,
            PackedSparseColumnEmitters<4> emitters,
            std::string& failure_reason,
            JacobianSelectedRows selected_rows = std::nullopt);
        bool do_cols_J_wlane8_sparse(
            JacobianColumnEngine::Workspace& workspace,
            PackedJacobianColumns<8> columns, double drop_tol,
            PackedSparseColumnEmitters<8> emitters,
            std::string& failure_reason,
            JacobianSelectedRows selected_rows = std::nullopt);
        bool do_cols_J_wlane16_sparse(
            JacobianColumnEngine::Workspace& workspace,
            PackedJacobianColumns<16> columns, double drop_tol,
            PackedSparseColumnEmitters<16> emitters,
            std::string& failure_reason,
            JacobianSelectedRows selected_rows = std::nullopt);
        bool do_cols_J_wlane32_sparse(
            JacobianColumnEngine::Workspace& workspace,
            PackedJacobianColumns<32> columns, double drop_tol,
            PackedSparseColumnEmitters<32> emitters,
            std::string& failure_reason,
            JacobianSelectedRows selected_rows = std::nullopt);
        /**
         * Diagnostic W=2 packed-column oracle. Computes two columns with
         * independent tangent lanes and compares them against scalar
         * \c do_col_J references. Returns false and writes details to \p report
         * when the packed path is not exact.
         */
        bool validate_packed_wlane2_columns(int first_column, int second_column,
                                            double abs_tol, double rel_tol,
                                            std::ostream& report);
        /**
         * Diagnostic W=2 packed-column oracle over representative non-VarDomain
         * column pairs selected from the current system taxonomy.
         */
        bool validate_packed_wlane2_representative_columns(int max_pairs_per_class,
                                                           double abs_tol, double rel_tol,
                                                           std::ostream& report);
        /**
         * Diagnostic W=4 packed-column oracle. Computes four columns with
         * independent tangent lanes and compares them against scalar
         * \c do_col_J references plus the W=4 sparse path. Returns false and
         * writes details to \p report when the packed path is not exact.
         */
        bool validate_packed_wlane4_columns(const std::array<int, 4>& columns,
                                            double abs_tol, double rel_tol,
                                            std::ostream& report);
        /**
         * Diagnostic W=4 packed-column oracle over representative non-VarDomain
         * column quartets selected from the current system taxonomy.
         */
        bool validate_packed_wlane4_representative_columns(int max_quartets_per_class,
                                                           double abs_tol, double rel_tol,
                                                           std::ostream& report);
        /**
         * Diagnostic W=8 packed-column oracle. Computes eight columns with
         * independent tangent lanes and compares the W=8 sparse path against
         * scalar \c do_col_J references. Returns false and writes details to
         * \p report when the packed path is not exact.
         */
        bool validate_packed_wlane8_columns(const std::array<int, 8>& columns,
                                            double abs_tol, double rel_tol,
                                            std::ostream& report);
        /**
         * Diagnostic W=8 packed-column oracle over representative non-VarDomain
         * column octets selected from the current system taxonomy.
         */
        bool validate_packed_wlane8_representative_columns(int max_octets_per_class,
                                                           double abs_tol, double rel_tol,
                                                           std::ostream& report);
        /**
         * Resets transient do_col_J caches that depend on the current adapted-domain state.
         */
        void reset_do_col_J_cache();
        void reset_do_col_J_cache(JacobianColumnEngine::Workspace& workspace);
        /**
         * Releases large do_col_J scratch buffers after sparse assembly has
         * finished but before global COO buffers are allocated.
         */
        void release_do_col_J_assembly_scratch();
        void release_do_col_J_assembly_scratch(
            JacobianColumnEngine::Workspace& workspace);
        /**
         * Prints and resets do_col_J profiling totals (rank 0 only).
         */
        void dump_do_col_J_profile() const;
        void dump_do_col_J_profile(
            JacobianColumnEngine::Workspace& workspace) const;
        /**
         * Diagnostic probe: writes per-definition transitive variable
         * dependencies as CSV `(def_idx,vars)` where `vars` is a
         * semicolon-joined sorted list of variable names this definition
         * resolves (including names reached via other definitions). Rank-0
         * only; silently no-ops on other ranks. Intended to pair with the
         * per-bucket variable dump in JacobianAssembler for the per-bucket
         * def-invariance census.
         */
        void write_def_var_census_csv(const std::string& path) const;

        /**
         * Diagnostic probe: measures the primal-vs-tangent cost split of the
         * definition sweep (`for i: def[i]->compute_res()`). Times `reps`
         * sweeps in three derivative regimes by toggling input-term der
         * presence (clear_der -> null -> primal only; set_der_zero(0..W-1) ->
         * W non-null zero lanes -> primal + full W-lane tangent, since the AD
         * gates tangent work on der_t != nullptr). Returns the mean per-sweep
         * wall for each regime via the out-params. Mutates and then restores
         * (clears) input-term derivative state; intended as a one-shot probe
         * that the caller follows with an exit. Quantifies the ceiling of the
         * primal-hoisting lever (reuse the linearization-invariant primal,
         * recompute only the tangent per W-group).
         */
        void measure_primal_tangent_split(int reps,
                                          double& out_primal,
                                          double& out_full_w1,
                                          double& out_full_w8);

        /**
         * Diagnostic sizing for the primal-hoisting lever (VAL_CACHE_RSS_PROBE).
         * Builds the set of transform-heavy intermediate-node vals across every
         * definition tree (the per-node val cache the lever would add) and
         * measures the resident-memory delta it costs. Reports the cache bytes,
         * the resident bytes before/after, and the heavy/total node counts.
         * One-shot; the holder is released before returning.
         */
        void measure_val_cache_rss(long long& out_cache_bytes,
                                   long long& out_rss_before,
                                   long long& out_rss_after,
                                   long long& out_heavy_nodes,
                                   long long& out_total_nodes);

        /**
         * Does one step of the solver iteration.
         * @param prec : required precision.
         * @param error : achieved precision.
         * @param solver: the underlying solver type
         * @return true if the required precision is achieved, false otherwise.
         */
        bool do_newton(double, double&, SOLVER solver = SOLVER::NEWTON_RAPHSON);
        /**
         * Dispatches the Newton step through the explicitly supplied runtime configuration.
         */
        bool do_newton_configured(double precision, double& error, const SolverRuntimeConfig& config);
        /**
         * Sparse Newton core implementation (MUMPS).
         */
        bool do_newton_sparse(double precision, double& error, double drop_tol = -1.0);
        bool do_newton_sparse(double precision, double& error, const SolverRuntimeConfig& config);
        bool do_newton_jfnk_mumps(double precision, double& error, const SolverRuntimeConfig& config);
        bool do_newton_jfnk_dense(double precision, double& error, const SolverRuntimeConfig& config);
        // Diagnostic backend (CELEPHAIS_SOLVER=jfnk-schur): one-shot pre-backend
        // scaling gate for the domain-Schur direction. Assembles the COO,
        // measures MUMPS' selected-variable Schur (ICNTL(19)) on the DDM-Schur
        // interface partition plus a dense LAPACK factor of S, prints
        // timings/RSS, then exits. Does not converge the system.
        bool do_newton_jfnk_schur(double precision, double& error, const SolverRuntimeConfig& config);

        /**
         * In-memory snapshot of the full unknown state for a Newton line search.
         * Captures everything a Newton step mutates: the field unknowns (\c var),
         * the numeric unknowns (\c var_double), the user constants remapped by a
         * moving surface (\c cst entries of type \c TERM_T), and the adapted-domain
         * surface mappings. Restoring is exact, so a trial step can be applied,
         * evaluated, and rolled back to retry at a smaller damping factor. The
         * adapted-mapping remap is nonlinear (see Domain::update_variable), so a
         * trial cannot be undone by re-applying a scaled correction — only a full
         * snapshot/restore is correct.
         */
        struct State_snapshot {
            std::vector<double> scalars;                  ///< var_double values, parallel to \c var_double.
            std::vector<Tensor> fields;                   ///< var values, parallel to \c var.
            std::vector<std::pair<int, Tensor>> cst_terms; ///< (slot, val_t) for each TERM_T entry of \c cst.
            std::vector<Val_domain> radii;                ///< Adapted-domain mapping components, in domain order.
        };
        /// Capture the current unknown state (see \c State_snapshot).
        /**
         * Hand a residual computed after the last unknown-state mutation
         * forward to the next Newton step's entry evaluation, which consumes
         * it instead of recomputing the identical residual
         * (RESIDUAL_FORWARD). Producers are the end-of-iteration
         * update_fields refresh and the direct-sparse trial evaluation. The
         * slot is single-producer/single-consumer, taken exactly once, and
         * defensively cleared by every unknown-state mutation and independent
         * solver-runtime reset.
         */
        void store_forwarded_residual(Array<double>&& residual)
        {
            forwarded_residual_ = std::make_unique<Array<double>>(std::move(residual));
        }
        /**
         * Return the infinity norm of the current forwarded residual without
         * consuming it.  Returns false when no current residual is available.
         * This lets post-update diagnostics observe the exact buffer that the
         * next Newton entry will consume, without a second residual assembly.
         */
        bool forwarded_residual_infinity_norm(double& norm) const;
        /// Take (and clear) the forwarded residual; null when none is current.
        std::unique_ptr<Array<double>> take_forwarded_residual()
        {
            return std::move(forwarded_residual_);
        }

        State_snapshot snapshot_state() const;
        /// Restore a previously captured unknown state, rebuilding adapted-domain mappings.
        void restore_state(const State_snapshot& snapshot);
        /**
         * Apply the Newton correction with a guarded backtracking line search
         * (enabled by SolverRuntimeConfig::jfnk_mumps.line_search). Accepts the full step when it does
         * not increase \f$\|F\|\f$ (guard mode, the default) or satisfies Armijo
         * sufficient-decrease (JFNK_LS_ARMIJO); otherwise halves the
         * damping up to JFNK_LS_MAX_BACKTRACK times (floor
         * JFNK_LS_MIN_LAMBDA) and commits the best finite trial. Relies on
         * the bit-exact \c snapshot_state / \c restore_state primitive. The
         * accepted step is left committed; returns the achieved infinity-norm
         * residual. \c error_before is \f$\|F(x)\|\f$ at the current iterate.
         */
        double apply_jfnk_line_search(Array<double>& delta, double error_before, int rank);

        /**
         * Shared Newton start-of-step diagnostics (rank 0 only), so every solver
         * backend prints the same lines.
         * - print_error_init_diagnostic: "Error init = <error>, Eq: <expr> Dom: <d>
         *   Boundary: <bc>" identifying the equation that owns the largest residual.
         * - print_dof_rank_diagnostic: dense/ScaLAPACK diagnostic with block size.
         * - print_sparse_dof_rank_diagnostic: legacy COO diagnostic retained by
         *   JFNK-dense. MUMPS-backed solvers use the direct-owned system summary.
         */
        void print_error_init_diagnostic(const Array<double>& residual, double error) const;
        void print_dof_rank_diagnostic(int n, int m, int nproc) const;
        void print_sparse_dof_rank_diagnostic(int n, int m, int nproc, long long nnz) const;

        bool do_newton_seq(double, double&);

        /**
         * Parallel Newton solver with cost-balanced column distribution.
         * Uses MPI and ScaLAPACK for distributed linear algebra.
         * @param precision : required precision.
         * @param error : achieved precision.
         * @param solver : solver type (unused, retained for interface compatibility).
         * @return true if the required precision is achieved, false otherwise.
         */
        bool do_newton_color(double precision, double& error, SOLVER solver = SOLVER::NEWTON_RAPHSON);

        /**
         * Get the column coloring for this system (computes if not cached).
         * @return Reference to the cached ColumnColoring structure.
         */
        const ColumnColoring& get_column_coloring();

        /**
         * Reset the column coloring cache (call when system structure changes).
         */
        void reset_column_coloring();

        /**
         * Print column coloring statistics to the given output stream.
         * Shows number of colors, potential speedup, and group size distribution.
         * @param os : output stream to write to.
         */
        void dump_column_coloring_stats(std::ostream& os) const;

        /**
         * Print an opt-in diagnostic comparing metadata coloring with actual sparse row support.
         * This is analysis-only and does not enable seeded Newton assembly.
         * @param os : output stream to write to.
         * @param max_columns : maximum prefix of columns to analyze (-1 for all columns).
         * @param drop_tol : sparse support threshold passed to do_col_J_sparse.
         */
        void dump_column_coloring_analysis(std::ostream& os, int max_columns = -1, double drop_tol = 0.0);

        /**
         * Validate production metadata fallback coloring groups with isolated diagnostic seeded attempts.
         * This is analysis-only and does not enable seeded Newton assembly.
         * @param os : output stream to write the validation report to.
         * @param max_columns : maximum prefix of columns to analyze (-1 for all columns).
         * @param max_group_size : maximum selected bucket size for the diagnostic attempt.
         * @param max_buckets : maximum number of fallback buckets to validate.
         * @param drop_tol : sparse support threshold passed to do_col_J_sparse.
         * @param rtol : relative tolerance for decoded/reference sparse-map comparison.
         * @param atol : absolute tolerance for decoded/reference sparse-map comparison.
         * @return true if the selected buckets pass the sparse-map comparison.
         */
        bool validate_fallback_coloring_bucket(std::ostream& os, int max_columns = -1, int max_group_size = 4,
                                               int max_buckets = 1, double drop_tol = 0.0,
                                               double rtol = 1e-10, double atol = 1e-14);

        /**
         * Validate diagnostic seeded COO entries against standard sparse columns for production metadata groups.
         * This is analysis-only and does not enable seeded Newton assembly.
         * @param os : output stream to write the validation report to.
         * @param max_columns : maximum prefix of columns to analyze (-1 for all columns).
         * @param max_group_size : maximum selected group size for each diagnostic seeded attempt.
         * @param max_groups : maximum number of metadata groups to validate.
         * @param drop_tol : sparse support threshold passed to do_col_J_sparse.
         * @param rtol : relative tolerance for COO value comparison.
         * @param atol : absolute tolerance for COO value comparison.
         * @return true if the selected seeded COO entries match standard sparse-column COO entries.
         */
        bool validate_seeded_coo_equivalence(std::ostream& os, int max_columns = -1, int max_group_size = 4,
                                             int max_groups = 1, double drop_tol = 0.0,
                                             double rtol = 1e-10, double atol = 1e-14);

        /**
         * Build mapping from column indices to variable information.
         * @param column_map : output vector to fill with ColumnInfo structs.
         */
        void build_column_map(std::vector<ColumnInfo>& column_map,
                              bool classify_field_columns = false) const;
        /**
         * Parity grading support (Linear_algebra/jacobian_parity_mask.hpp): for
         * every Jacobian column, the azimuthal (phi) coefficient index, its
         * actual component basis and owning domain, and the tensor component
         * it perturbs.  A COSSIN series stores cos(m.phi) at even index and
         * sin(m.phi) at odd index (summation_1d.cpp:176); the owning domain maps
         * the index and basis to physical y-reflection parity.  The maps are
         * obtained by one-hot probing the production
         * affecte_tau_one_coef / xx_to_vars_from_adapted walks, so they cannot
         * drift from the assembler's own column enumeration.  Columns carrying
         * no single azimuthal index (the numeric globals) report -1.  Returns
         * false when a field block has no non-materializing tau-seed
         * description or when the Space cannot describe its contiguous
         * variable-domain blocks; the final two outputs identify which refusal
         * occurred.
         */
        bool build_column_phi_and_component_indices(
            std::vector<int>& phi_index,
            std::vector<int>& phi_basis,
            std::vector<int>& phi_domain,
            std::vector<int>& component_index,
            int& unsupported_tau_seed_domain,
            bool& unsupported_variable_domain_layout) const;
        /**
         * Parity sector tables retained for this solve.  Sparse-direct may
         * install structurally validated tables before J1; otherwise they stay
         * empty until the first masked Jacobian builds and validates them
         * (Linear_algebra/jacobian_parity_mask.hpp).
         */
        std::shared_ptr<JacobianParityMaskState>& jacobian_parity_mask_state()
        {
            return jacobian_parity_mask_state_;
        }
        /**
         * Read the current unknown coordinates outside a selection plan.
         * Field columns use the primary tau-seed coefficient; production
         * variable-domain blocks use an exclusive one-hot mapping pivot.
         * Unsupported layouts fail closed with a reason. This does not mutate
         * the adapted variable-domain state.
         */
        bool read_inactive_jacobian_state(
            const JacobianSelectionPlan& plan, std::vector<double>& values,
            std::string& failure_reason) const;
        void build_direct_singleton_jacobian_columns(
            DirectJacobianColumnPlan& direct_plan) const;
        /// Overload reusing a column map already built by build_column_map(...,true).
        void build_direct_singleton_jacobian_columns(
            DirectJacobianColumnPlan& direct_plan,
            const std::vector<ColumnInfo>& column_map) const;

        /**
         * Return the immutable structural products used only by
         * JacobianAssembler.  A cache hit requires an exact match of system
         * topology, tensor layout, and every spectral-basis code that can
         * affect direct exported values.  The returned reference remains
         * valid until the next call on this System_of_eqs.
         */
        const JacobianAssemblerStructuralPlan&
        get_jacobian_assembler_structural_plan(
            bool include_column_metadata,
            bool cache_enabled,
            JacobianAssemblerStructuralPlanAccess& access) const;

        /// Returns the set of (column_class, domain, owner_var_name) triples
        /// whose owning equation passes the bare-owner-zero structural predicate
        /// ("X = 0" or "X^i = 0").  Read-only; no probe calls.  Used by the
        /// EQBC_DIRECT_EMIT_CENSUS predicate-filtering pass.
        std::set<std::tuple<ColumnClass, int, std::string>>
        build_bare_owner_zero_attachment_set() const;
        ColumnClass classify_field_column_from_equations(int term_idx, int global_col,
                                                         int term_start_col) const;
        bool field_coefficient_selected_by_volume_basis(int term_idx, int global_col,
                                                        int term_start_col, int eq_idx) const;
        bool field_coefficient_exported_by_equation(int term_idx, int global_col,
                                                    int term_start_col, int eq_idx,
                                                    bool volume_rows_only = false) const;

        /**
         * Build row incidence sets for each column.
         * For each column, determines which equation rows it affects.
         * @param column_map : input column mapping from build_column_map().
         * @param rows_per_column : output vector of row index sets per column.
         */
        void build_column_row_incidence(const std::vector<ColumnInfo>& column_map,
                                        std::vector<std::set<int>>& rows_per_column) const;

        /**
         * Partition columns for bordered block extraction from caller-supplied
         * row support. The support must describe the emitted block operator,
         * not the coloring graph. A column is bulk only when it is a domain
         * FieldInteriorVol column with same-domain Vol-row support and no
         * Vol-row support in another domain.
         * Transfer-row support does not demote the column; it forms the B_tb
         * block in the bordered Schur model. Everything else is a transfer column.
         */
        void build_incidence_column_partition(
            const std::vector<RowMetadata>& row_metadata,
            const std::vector<ColumnMetadata>& column_metadata,
            const std::vector<std::set<int>>& rows_per_column,
            IncidenceColumnPartition& out) const;

        /**
         * Compute Welsh-Powell greedy coloring on the column adjacency graph.
         * Two columns are adjacent if they affect any common row.
         * @param coloring : output ColumnColoring structure to fill.
         */
        void compute_column_coloring(ColumnColoring& coloring) const;

        /**
         * Compute multiple Jacobian columns simultaneously using seeded finite differences.
         * All columns must have the same color (disjoint row sets).
         * @param cols_to_compute : vector of column indices to compute together.
         * @param result_columns : output vector of computed column arrays.
         */
        [[deprecated("Reopen on DOF > 16x BNS crossover")]]
        void do_cols_J_seeded(const std::vector<int>& cols_to_compute, std::vector<Array<double>>& result_columns);

        /**
         * Diagnostic oracle for row-disjoint grouped seeding. Uses explicit scalar
         * row-support ownership supplied by the caller and compares the grouped
         * seeded result back against scalar \c do_col_J_sparse entries.
         *
         * This does not enable production seeded assembly.
         */
        bool validate_row_disjoint_seeded_group(
            const std::vector<int>& cols_to_compute,
            const std::vector<std::set<int>>& rows_per_selected_column,
            double drop_tol,
            std::ostream& report);

        /**
         * Print and reset do_cols_J_seeded profiling totals (rank 0 only).
         * Enabled by \c SolverRuntimeConfig::diagnostics.timing.
         */
        void dump_do_cols_J_seeded_profile() const;

        /**
         * Validate column coloring by comparing seeded vs single-column results.
         * @param max_checks : maximum number of columns to validate (-1 for all).
         * @param rtol : relative tolerance for comparison.
         * @param atol : absolute tolerance for comparison.
         * @param os : optional output stream for diagnostics.
         * @return true if validation passes, false otherwise.
         */
        bool validate_column_coloring(int max_checks = -1, double rtol = 1e-10, double atol = 1e-14,
                                      std::ostream* os = nullptr);

        int project_z_symmetric_diagnostic_correction(Array<double>& correction,
                                                      std::ostream* os = nullptr) const;

        void describe_column_seed(int column, std::ostream& os, int max_coefficients = 8) const;

        // ----- p-coarse two-level PC probe (diagnostic-only) -----------------
        // Defined in src/Newton/pcoarse_probe_maps.cpp. Decode the current
        // system's column/row spectral tuples, dump the coarse-rung artifacts
        // (PCOARSE_DUMP), and run the fine-rung 3-arm GMRES A/B against a
        // previously dumped coarse rung (PCOARSE_PC_PROBE). All rank-0,
        // self-exiting; never touch production solve paths.
        /// Read just the coarse unknown count from a dumped coo.bin header
        /// (first int64), or -1 if the file is missing. Lets the schur harness
        /// fire the probe only on the FINE rung (nbr_unknowns > coarse n).
        long long pcoarse_read_coarse_n(const std::string& directory) const;
        void pcoarse_decode_columns(std::vector<PcoarseColumnKey>& out) const;
        void pcoarse_decode_rows(std::vector<PcoarseRowKey>& out);
        void pcoarse_dump_probe_artifacts(const std::string& directory,
                                          const AssembledJacobianCoo& coo,
                                          double drop_tol_used);
        void pcoarse_run_pc_probe(const std::string& directory,
                                  const AssembledJacobianCoo& coo,
                                  const Array<double>& fine_residual,
                                  const SolverRuntimeConfig& config);

        /**
         * Updates the variations of the \c Term_eq that comes from the fact that some \c Domains are variable (i.e.
         * their shape).
         * @param zedoms : the number of all the variable domains.
         * @param term_filter : optional per-\c term[] mask (size \c nterm) from the \c do_JX
         * per-rank term closure: masked-out terms skip \c update_term_eq (their derivative
         * is never read by this rank's owned rows).
         * @param cst_filter : optional per-\c cst[] mask (size \c nterm_cst) — same closure
         * for the \c TERM_T constants the adapted-domain update remaps.
         */
        void update_terms_from_variable_domains(const Array<int>& zedoms,
                                                const std::vector<char>* term_filter = nullptr,
                                                const std::vector<char>* cst_filter = nullptr);
        /**
         * Adds variable-domain mapping derivatives to already seeded
         * \c term[] derivatives. Tensor constants retain replacement semantics.
         */
        void accumulate_terms_from_variable_domains(const Array<int>& zedoms,
                                                    const std::vector<char>* term_filter = nullptr,
                                                    const std::vector<char>* cst_filter = nullptr);

      private:
        enum class VariableDomainTermUpdate
        {
            replace,
            accumulate
        };
        void update_terms_from_variable_domains_impl(
            const Array<int>& zedoms, const std::vector<char>* term_filter,
            const std::vector<char>* cst_filter, VariableDomainTermUpdate term_update);
        /// Refresh a caller-owned snapshot, reusing its topology-stable field,
        /// constant, and adapted-mapping storage when possible.
        void snapshot_state_into(State_snapshot& snapshot) const;
        /// Persistent capture buffer used only by the non-reentrant JFNK line
        /// search and its one-shot bit-exact self-test.
        State_snapshot jfnk_line_search_snapshot_;

        // Cache of adapted-domain ids touched by any variable-domain coefficient.
        // Lazy-init on first do_JX call; topology is fixed for a Newton solve.
        mutable std::vector<int> variable_domains_cache_;
        mutable bool variable_domains_cache_valid_ = false;
        /// When true, the DO_JX_TERM_CLOSURE activation guard treats
        /// user operators as transparent (their reads are visible to the
        /// Ope_id-target closure). Default false (conservative: user operators
        /// disable the filter).
        bool user_opes_target_only_ = false;

        /**
         * MPI row partition for the matrix-free Jv product (\c do_JX), gated by
         * \c DO_JX_MPI. Each rank applies and exports only a contiguous
         * block of field equations (plus the \c Eq_int integral parts whose
         * domain it owns) and recomputes only the definitions those rows
         * consume (via \c Equation::take_into_account); the full vector is then
         * reassembled with an MPI all-gather. Bit-exact with the replicated
         * path: every row/part value is produced by exactly one rank through
         * the unchanged serial code, so no floating-point reduction occurs.
         * Topology is keyed on (neq, neq_int, nbr_conditions, nproc) and lazily
         * rebuilt; the first partitioned call after each (re)build verifies the
         * result bitwise against the replicated path and falls back permanently
         * (collective verdict, no rank-divergent throw) on mismatch.
         * The same split also drives the row-partitioned nonlinear residual
         * (\c sec_member_partitioned, gated by \c SEC_MEMBER_MPI):
         * identical exchanges and closure, \c export_val / \c get_val_d
         * instead of the derivative accessors, with its own one-shot bitwise
         * self-test and permanent fallback (\c val_active /
         * \c val_selftest_pending).
         * The split additionally drives a per-rank term/cst closure for the
         * \c do_JX preamble (gated by \c DO_JX_TERM_CLOSURE): each
         * rank seeds (\c xx_to_ders) and shape-updates
         * (\c update_terms_from_variable_domains) only the \c term[] /
         * \c TERM_T \c cst[] entries whose derivatives its owned rows read,
         * jumping skipped xx blocks via \c term_xx_stride; the first
         * filtered call per split verifies bitwise against the full
         * preamble and falls back permanently on mismatch
         * (\c term_filter_active / \c term_filter_selftest_pending).
         */
        struct DoJxMpiPartition {
            bool active = false;           ///< partition usable for the current topology
            bool selftest_pending = false; ///< first partitioned call still has to verify
            bool val_active = false;       ///< value (sec_member) path usable for the current topology
            bool val_selftest_pending = false; ///< first partitioned value call still has to verify
            bool rebalanced = false;       ///< measured-cost repartition already applied
            bool term_filter_active = false; ///< per-rank term/cst preamble closure usable
            bool term_filter_selftest_pending = false; ///< first filtered-preamble call still has to verify
            std::vector<double> def_cost;  ///< per-def compute seconds, measured on rank 0
            std::vector<double> eq_cost;   ///< per-eq apply+export seconds, measured on rank 0
            std::vector<double> part_cost; ///< per-Eq_int-part der seconds, measured on rank 0
            int nproc = 1;                 ///< topology key
            int rank = 0;                  ///< this rank within MPI_COMM_WORLD
            int neq = -1;                  ///< topology key
            int neq_int = -1;              ///< topology key
            int nbr_conditions = -1;       ///< topology key
            int eq_begin = 0;              ///< first owned field-equation index
            int eq_end = 0;                ///< one-past-last owned field-equation index
            std::vector<int> eq_operator_offset; ///< prefix sum of n_ope (size neq+1)
            std::vector<int> eq_row_begin;       ///< row offset per equation, starts at neq_int (size neq+1)
            std::vector<char> def_needed;        ///< per-def: this rank must recompute it
            std::vector<char> def_val_current;   ///< per-def: value fresh on this rank after the last value pass
            /// Exact definition closure per field equation: indices of every
            /// def the equation's operator trees reference (Ope_id targets
            /// resolved against def results), expanded through def-to-def
            /// chains; sorted unique.
            std::vector<std::vector<int>> eq_defs;
            /// Same exact closure per flattened Eq_int part (part_offset slots).
            std::vector<std::vector<int>> part_defs;
            /// Transitive def-to-def chains per def (sorted, lower index =
            /// referenced first); consumed by \c ensure_def_values_current to
            /// lazily repair rank-local stale defs after a partitioned value
            /// pass.
            std::vector<std::vector<int>> def_closed;
            /// Per \c term[] entry: some owned row of this rank (field
            /// equation, owned \c Eq_int part, or a def in \c def_needed)
            /// reads its derivative — seed it in \c xx_to_ders and update it
            /// in \c update_terms_from_variable_domains. Split-dependent.
            std::vector<char> term_needed;
            /// Per \c cst[] entry: same read set for the \c TERM_T constants
            /// (only consumed by the adapted-domain update; constants are
            /// never seeded by \c xx_to_ders). Split-dependent.
            std::vector<char> cst_needed;
            /// xx cells consumed per \c term[] entry by \c affecte_tau
            /// (= \c Domain::nbr_unknowns(var, dom), the count \c add_var
            /// used to size the block); lets the filtered \c xx_to_ders jump
            /// over skipped per-term blocks. Topology-constant; the sum is
            /// cross-checked against \c nbr_unknowns at build time and the
            /// filter stays inactive on mismatch.
            std::vector<int> term_xx_stride;
            std::vector<int> row_counts;         ///< Allgatherv recvcounts (field rows per rank)
            std::vector<int> row_displs;         ///< Allgatherv displs into the field-row section
            std::vector<int> part_owner;         ///< owner rank per flattened eq_int part
            std::vector<int> part_offset;        ///< flattened part offset per eq_int (size neq_int+1)
            std::vector<int> part_counts;        ///< Allgatherv recvcounts (eq_int parts per rank)
            std::vector<int> part_displs;        ///< Allgatherv displs for the part exchange
            /// Jv/value exchange and reconstruction storage, resized only
            /// when the partition split is (re)derived. Both partitioned
            /// assembly paths are collective and non-reentrant, so sequential
            /// applications can safely share these buffers.
            std::vector<double> part_send_workspace;
            std::vector<double> part_recv_workspace;
            std::vector<int> owner_cursor_workspace;
        };
        mutable DoJxMpiPartition do_jx_mpi_;

        /// Single-shot forwarded residual (see \c store_forwarded_residual).
        /// Mutable because reset_solver_runtime_state clears logical runtime
        /// cache state through its existing const interface.
        mutable std::unique_ptr<Array<double>> forwarded_residual_;

        /// Consume a current forwarded residual or compute the residual afresh.
        /// With RESIDUAL_FORWARD_SELFTEST, the forwarded and fresh byte
        /// sequences are compared collectively before the forwarded vector is
        /// accepted. Implemented in assembly/solver.cpp.
        Array<double> take_forwarded_residual_or_compute(const char* producer);

        /// Derive the split-dependent part of \c do_jx_mpi_ (row ownership,
        /// Allgatherv layout, Eq_int part owners, definition closure) from a
        /// contiguous field-equation split, identical on every rank.
        /// Implemented in \c assembly/row_partition.cpp.
        void derive_do_jx_partition_from_split(const std::vector<int>& eq_split);
        /// Build (or rebuild) the \c do_JX MPI row partition for the current
        /// equation topology and communicator size; deterministic, so every
        /// rank derives the identical initial (row-count-balanced) partition
        /// without communication. Implemented in \c assembly/row_partition.cpp.
        void build_do_jx_mpi_partition(int nproc, int rank);
        /// One-shot makespan repartition of the \c do_JX row split from the
        /// per-def/per-eq/per-Eq_int-part costs measured on rank 0 by the
        /// self-test reference pass (split broadcast to all ranks).
        /// Implemented in \c assembly/row_partition.cpp.
        void rebalance_do_jx_partition_from_measured_costs();
        /// Row-partitioned value tail: the exact mirror of \c do_JX's
        /// partitioned tail with values instead of derivatives
        /// (\c compute_res closure, owned-equation \c export_val, \c Eq_int
        /// parts via \c get_val_d, identical \c MPI_Iallgatherv pair).
        /// Implemented in \c assembly/solver.cpp.
        Array<double> sec_member_partitioned_tail();
        /// Lazy rank-local repair for \c give_val_def readers after a
        /// partitioned value pass: recompute the named def's stale instances
        /// plus their \c def_closed chains (ascending index = referenced
        /// defs first), updating \c def_val_current. Local computation only —
        /// safe for rank-0-only callers. Implemented in
        /// \c assembly/system_of_eqs.cpp.
        void ensure_def_values_current(const char* name) const;

        bool do_cols_J_seeded_diagnostic_force(const std::vector<int>& cols_to_compute,
                                               const std::vector<std::set<int>>& rows_per_selected_column,
                                               std::vector<Array<double>>& result_columns,
                                               bool& guard_bypassed_for_diagnostic,
                                               std::string& failure_reason);
        void attach_metric(Metric& metric, const char* name);
        bool do_col_J_def_filter_enabled() const;
        friend class JacobianColumnEngine;
        friend class JacobianColumnEngineTestHelper;
        friend class EquationParser;
        friend struct BinaryCoSpaceEquations; // shared binary-CO Space add_eq* bodies (src/Space/Shared)
        friend class Space_spheric;
        friend class Space_bispheric;
        friend class Space_critic;
        friend class Space_polar;
        friend class Space_spheric_adapted;
        friend class Space_spheric_homothetic;
        friend class Space_trumpet;
        friend class Space_spheric_adapted_nosym;
        friend class Space_bin_ns_nosym;
        friend class Space_three_body;
        friend class Space_polar_adapted;
        friend class Space_adapted_bh_polar;
        friend class Space_polar_bilateral_adapted;
        friend class Space_bin_ns;
        friend class Space_bin_ns_nosym;
        friend class Space_bhns;
        friend class Space_bhns_nosym;
        friend class Space_bin_bh;
        friend class Space_bin_fake;
        friend class Space_polar_periodic;
        friend class Space_adapted_bh;
        friend class Space_adapted_bh_nosym;
        friend class Space_KerrSchild_bh;
        friend class Space_Kerr_bbh;
        friend class Space_bbh;
        friend class Metric;
        friend class Metric_general;
        friend class Metric_flat;
        friend class Metric_dirac;
        friend class Metric_dirac_const;
        friend class Metric_conf;
        friend class Metric_relax;
        friend class Metric_ADS;
        friend class Metric_AADS;
        friend class Metric_const;
        friend class Metric_flat_nophi;
        friend class Metric_nophi;
        friend class Metric_nophi_AADS;
        friend class Metric_nophi_const;
        friend class Metric_nophi_AADS_const;
        friend class Metric_conf_factor;
        friend class Metric_conf_factor_const;
        friend class Metric_cfc;
    };

    /**
     * Class implementing an equation. This is a purely abstract class that can not be instanciated.
     * \ingroup systems
     */
    class Equation : public MemoryMappable
    {

      protected:
        const Domain* dom; ///< Pointer on the \c Domain where the equation is defined.
        int ndom;          ///< Number of the domain
        int n_ope;         ///< Number of terms involved in the equation (one for bulk, two or more fot matching...).
        std::vector<std::unique_ptr<Ope_eq>> parts; ///< Array of owned pointers on the various terms.

        /**
         * Indicator checking whther the result has been computed already once.
         * If not the quantities \c n_cond must be computed.
         */
        bool called;
        int n_comp;     ///< Number of components of the residual (1 for a scalar, 6 for a symmetric rank-2 tensor etc).
        int n_cond_tot; ///< Total number of discretized equations (essentially the number of all coefficients of the
                        ///< residual).
        Array<int>* n_cond; ///< Number of discretized equations, component by component.

        int n_cmp_used;          ///< Number of components used (by default the same thing as \c n_comp).
        Array<int>** p_cmp_used; ///< Array of pointer on the indices of the used components

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : index of the \d Domain (consistence is not checked).
         * @param nope : number of operators.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Equation(const Domain*, int nd, int nope, int n_cmp = -1, Array<int>** p_cmp = nullptr);

        /**
         * Shared lane-direct packed export for n_ope=1 equations whose scalar
         * \c export_der is exactly a single bulk
         * <tt>dom->export_tau(*get_p_der_t(), ndom, selector, sec, pos_res,
         * *n_cond, n_cmp_used, p_cmp_used)</tt>. Reads each tangent lane's
         * derivative tensor via \c get_p_der_t(lane) instead of swapping the
         * primary slot. Lanes the AD chain did not populate skip the
         * \c export_tau call and only advance \c pos_res_arr by \c n_cond_tot;
         * the engine zeroes this equation's active rows before it dispatches,
         * so the untouched rows already hold the legacy
         * "zero tensor exported" values. Byte-identical to the per-lane
         * primary-slot-swap fallback while avoiding the per-miss Tensor
         * allocation. \c selector is the per-subclass \c export_tau argument
         * (0 for \c Eq_full, 2 for \c Eq_inside, 1 for \c Eq_one_side,
         * \c order for \c Eq_order). Advances \c conte by one (single operator).
         */
        void export_der_lanes_single_tau(int& conte, Term_eq** residus, int lane_count,
                                         Array<double>* const* secs, int* pos_res_arr,
                                         int selector) const;
        bool describe_residual_rows_single_tau(
            int& conte, Term_eq** residuals, int equation_index, int selector,
            std::vector<ResidualRowDescriptor>& descriptors) const;

      public:
        virtual ~Equation(); ///< Destructor

        /**
         * @return the total number of discretized conditions.
         */
        int get_n_cond_tot() const { return n_cond_tot; };

        /**
         * @return the index of the domain where the equation is defined.
         */
        int get_ndom() const { return ndom; };

        /**
         * Computes the terms involved in computing the residual of the equations.
         * @param conte : current position in the array of terms.
         * @param res : array of pointers on the various terms.
         */
        virtual void apply(int& conte, Term_eq** res);
        /**
         * Generates the discretized errors, from the various \c Term_eq computed by the equation.
         *  Basically used when computing the second member of the Newton-Raphson algorithm.
         * @param conte : current position in the array of terms.
         * @param residuals : array of pointers on the various terms.
         * @param sec : array of the discretized errors.
         * @param pos_sec : current position in \c sec.
         */
        virtual void export_val(int& conte, Term_eq** residuals, Array<double>& sec, int& pos_sec) const = 0;
        /**
         * Generates the discretized variations, from the various \c Term_eq computed by the equation.
         *  Basically used when computing the Jacobian of the Newton-Raphson algorithm.
         * @param conte : current position in the array of terms.
         * @param residuals : array of pointers on the various terms.
         * @param sec : array of the discretized errors.
         * @param pos_sec : current position in \c sec.
         */
        virtual void export_der(int& conte, Term_eq** residuals, Array<double>& sec, int& pos_sec) const = 0;
        /**
         * Packed-export entry point used by the W=2/W=4/W=8 column-AD paths.
         * Walks \c lane_count tangent lanes for this equation, writing the
         * discretized variation for each lane into \c secs[lane] starting at
         * \c pos_res_arr[lane] and advancing that counter by the equation's
         * row span. Subclasses that override this MUST mirror \c export_der's
         * row-position advance so the assembler's row layout stays consistent.
         */
        virtual void export_der_lanes(int& conte, Term_eq** residuals,
                                       int lane_count,
                                       Array<double>* const* secs,
                                       int* pos_res_arr) const;
        virtual bool describe_residual_rows(
            int& conte, Term_eq** residuals, int equation_index,
            std::vector<ResidualRowDescriptor>& descriptors) const;
        /**
         * Computes the number of conditions associated with the equation.
         * @param tt : the residual of the equation.
         */
        virtual Array<int> do_nbr_conditions(const Tensor& tt) const = 0;
        /**
         * Check whether the variation of the residual has to be taken into account when computing a given column.
         * @param target : domain involved in the computation of the given column.
         */
        virtual bool take_into_account(int) const = 0;

        friend class System_of_eqs;
        friend class JacobianColumnEngine;
    };

    /**
     * Class for bulk equations that are solved stricly inside a given domain.
     * By this one means that they have to be given matching/boundary conditions at all their boundaries.
     * Typically used for second order PDE.
     * \ingroup systems
     */
    class Eq_inside : public Equation
    {

      public:
        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param ope : pointer on the operator describing the equation.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Eq_inside(const Domain* dom, int nd, Ope_eq* op, int n_cmp = -1, Array<int>** p_cmp = nullptr);
        ~Eq_inside() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der_lanes(int& conte, Term_eq** residuals, int lane_count,
                              Array<double>* const* secs, int* pos_res_arr) const override;
        bool describe_residual_rows(int&, Term_eq**, int,
                                    std::vector<ResidualRowDescriptor>&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class for the velocity potential in irrotational binray neutron stars.
     * It is solved for all harmonics byt l=m=0 for which an exceptionnal condition is enforced.
     * Typically used for second order PDE.
     * \ingroup systems
     */
    class Eq_vel_pot : public Equation
    {

      public:
        int order; ///< Order of the equation.
        /** Audited contract for the ordinary/constant first-row substitution. */
        bool same_reflection_sector;

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param ord : order of the equation (probably only safe with 0, 1 or 2).
         * @param ope : pointer on the operator describing the equation.
         * @param ope_constant : condition for the constant part
         * @param same_reflection_sector : explicit contract that the ordinary
         * and constant-part substitutions transform in the same y-reflection
         * sector.
         */
        Eq_vel_pot(const Domain* dom, int nd, int ord, Ope_eq* op,
                   Ope_eq* op_constant,
                   bool same_reflection_sector = false);
        ~Eq_vel_pot() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        bool describe_residual_rows(int&, Term_eq**, int,
                                    std::vector<ResidualRowDescriptor>&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class for enforcing boundary condition.
     * It is solved for all harmonics byt l=m=0 for which an exceptionnal condition is enforced.
     * \ingroup systems
     */
    class Eq_bc_exception : public Equation
    {

      public:
        int bound; ///< Order of the equation.

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param bound : boundary where the condition is enforced
         * @param ope : pointer on the operator describing the equation.
         * @param ope_constant : condition for the constant part
         */
        Eq_bc_exception(const Domain* dom, int nd, int bound, Ope_eq* op, Ope_eq* op_constant);
        ~Eq_bc_exception() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class for bulk equation which order is passed as a parameter.
     * \ingroup systems
     */
    class Eq_order : public Equation
    {

      public:
        int order; ///< Order of the equation.

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param ord : order of the equation (probably only safe with 0, 1 or 2).
         * @param ope : pointer on the operator describing the equation.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */

        Eq_order(const Domain* dom, int nd, int ord, Ope_eq* ope, int n_cmp = -1, Array<int>** p_cmp = nullptr);
        ~Eq_order() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der_lanes(int& conte, Term_eq** residuals, int lane_count,
                              Array<double>* const* secs, int* pos_res_arr) const override;
        bool describe_residual_rows(int&, Term_eq**, int,
                                    std::vector<ResidualRowDescriptor>&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class for an equation representing a boundary condition on some surface.
     * \ingroup systems
     */
    class Eq_bc : public Equation
    {

      public:
        int bound; ///< The boundary
                   /**
                    * Constructor
                    * @param dom : Pointer on the \d Domain
                    * @param nd : number of the \d Domain (consistence is not checked).
                    * @param bb : boundary where the equation is enforced.
                    * @param ope : pointer on the operator describing the equation.
                    * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
                    * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
                    */
        Eq_bc(const Domain* dom, int nd, int bb, Ope_eq* ope, int n_cmp = -1, Array<int>** p_cmp = nullptr);
        Eq_bc(const Domain* dom, int nd, int bb, Ope_eq* ope, bool projected_boundary,
              int n_cmp = -1, Array<int>** p_cmp = nullptr);
        ~Eq_bc() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        bool describe_residual_rows(int&, Term_eq**, int,
                                    std::vector<ResidualRowDescriptor>&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
      private:
        bool projected_boundary = false;
    };

    /**
     * Class for an equation representing the matching of quantities accross a boundary.
     * \ingroup systems
     */
    class Eq_matching : public Equation
    {

      public:
        int bound;       ///< Name of the boundary in the domain of the equation.
        int other_dom;   ///< Number of the other domain.
        int other_bound; ///< Name of the boundary in the other domain.

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param bb : name of the boundary in the domain of the equation.
         * @param oz_nd : number of the other domain.
         * @param oz_bb : name of the boundary in the other domain.
         * @param ope : pointer on the operator describing the quantity to be matched in the current domain.
         * @param oz_ope : pointer on the operator describing the quantity to be matched in the other domain.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */

        Eq_matching(const Domain* dom, int nd, int bb, int oz_nd, int oz_bb, Ope_eq* ope, Ope_eq* oz_ope,
                    int n_cmp = -1, Array<int>** p_cmp = nullptr);
        ~Eq_matching() override; ///< Destructor.

        // No apply() override (restored to 74e7f645): inherits Equation::apply().
        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der_lanes(int&, Term_eq**, int, Array<double>* const*, int*) const override;
        bool describe_residual_rows(int&, Term_eq**, int,
                                    std::vector<ResidualRowDescriptor>&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class for an equation representing the matching of quantities accross a boundary.
     * Used when the matching surface also has boundaries.
     * \ingroup systems
     */
    class Eq_matching_one_side : public Equation
    {

      public:
        int bound;       ///< Name of the boundary in the domain of the equation.
        int other_dom;   ///< Number of the other domain.
        int other_bound; ///< Name of the boundary in the other domain.

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param bb : name of the boundary in the domain of the equation.
         * @param oz_nd : number of the other domain.
         * @param oz_bb : name of the boundary in the other domain.
         * @param ope : pointer on the operator describing the quantity to be matched in the current domain.
         * @param oz_ope : pointer on the operator describing the quantity to be matched in the other domain.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Eq_matching_one_side(const Domain* dom, int nd, int bb, int oz_nd, int oz_bb, Ope_eq* ope, Ope_eq* oz_ope,
                             int n_cmp = -1, Array<int>** p_cmp = nullptr);
        ~Eq_matching_one_side() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class for an equation representing the matching of quantities accross a boundary.
     * The matching is performed in the configuration space.
     * It is intended where the collocations points are different at each side of the boundary.
     * It can happen when there are more than one touching domain (bispheric vs spheric) and when the number of points
     * is different.
     * \ingroup systems
     */
    class Eq_matching_non_std : public Equation
    {

      public:
        int bound;               ///< Name of the boundary in the domain of the equation.
        Array<int> other_doms;   ///< Array containing the number of the domains being on the other side of the surface.
        Array<int> other_bounds; ///< Names of the boundary, as seen in the other domains.

        Index** which_points; ///< Lists the collocation points on the boundary (probably...)

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param bb : name of the boundary in the domain of the equation.
         * @param ozers : numbers of the others domains in (0,*) and names of the boundary as seen on the other side in
         * (1,*).
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Eq_matching_non_std(const Domain* dom, int nd, int bb, const Array<int>& ozers, int n_cmp = -1,
                            Array<int>** p_cmp = nullptr);
        ~Eq_matching_non_std() override; ///< Destructor.

        void apply(int&, Term_eq**) override;
        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;

        /**
         * Computes the collocation points used
         * @param base : the spectral base of the residual (in order to get the symmetries).
         * @param start : starting index
         */
        void do_which_points(const Base_spectral& base, int start);
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;

        friend class System_of_eqs;
    };

    /**
     * Class for an equation representing the matching of quantities accross a boundary using the "import" reserved
     * word. The matching is performed in the configuration space. It is intended where the collocations points are
     * different at each side of the boundary. It can happen when there are more than one touching domain (bispheric vs
     * spheric) and when the number of points is different.
     * \ingroup systems
     */
    class Eq_matching_import : public Equation
    {

      public:
        int bound;               ///< Name of the boundary in the domain of the equation.
        Array<int> other_doms;   ///< Array containing the number of the domains being on the other side of the surface.
        Array<int> other_bounds; ///< Names of the boundary, as seen in the other domains.

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param bb : name of the boundary in the domain of the equation.
         * @param eq : the matching condition. Should be of the form "a = import(b)".
         * @param ozers : numbers of the others domains in (0,*) and names of the boundary as seen on the other side in
         * (1,*).
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Eq_matching_import(const Domain* dom, int nd, int bb, Ope_eq* eq, const Array<int>& ozers, int n_cmp = -1,
                           Array<int>** p_cmp = nullptr);
        ~Eq_matching_import() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der_lanes(int&, Term_eq**, int, Array<double>* const*, int*) const override;
        bool describe_residual_rows(int&, Term_eq**, int,
                                    std::vector<ResidualRowDescriptor>&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class implementing an integral equation.
     * \ingroup systems
     */
    class Eq_int : public MemoryMappable
    {

      protected:
        int n_ope;      ///< Number of terms.
        std::vector<std::unique_ptr<Ope_eq>> parts; ///< Array of owned pointers on the various terms.
        signed char reflection_sector = 0;

      public:
        /**
         * Constructor just sets n_ope.
         * @param nop : the number of operators involved.
         */
        Eq_int(int nop);

      public:
        ~Eq_int(); ///< Destructor

        double get_val() const; ///< Return the value of the equation.
        double get_der() const; ///< Return the variation of the equation.
        double get_der(int lane) const; ///< Return the variation in the requested tangent lane.
        /// Fill \p derivatives with the variations for the first \p lane_count tangent lanes.
        void get_der_lanes(int lane_count, double* derivatives) const;
        /**
         * Sets one of the \c Ope_eq needed by the equation.
         * @param i : the part to be set.
         * @param ope : pointer on the operator to be used.
         */
        void set_part(int, Ope_eq*);
        void set_reflection_sector(int sector);
        int get_reflection_sector() const { return reflection_sector; }

        friend class System_of_eqs;
    };

    /**
     * Class for a zeroth order equation in a \c Domain.
     * Should be used for equations without derivatives.
     * \ingroup systems
     */
    class Eq_full : public Equation
    {

      public:
        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param ope : pointer on the operator describing the equation.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Eq_full(const Domain* dom, int nd, Ope_eq* ope, int n_cmp = -1, Array<int>** p_cmp = nullptr);
        ~Eq_full() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der_lanes(int& conte, Term_eq** residuals, int lane_count,
                              Array<double>* const* secs, int* pos_res_arr) const override;
        bool describe_residual_rows(int&, Term_eq**, int,
                                    std::vector<ResidualRowDescriptor>&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class for a first order equation in a \c Domain.
     * \ingroup systems
     */
    class Eq_one_side : public Equation
    {

      public:
        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param ope : pointer on the operator describing the equation.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Eq_one_side(const Domain* dom, int nd, Ope_eq* ope, int n_cmp = -1, Array<int>** p_cmp = nullptr);
        ~Eq_one_side() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der_lanes(int& conte, Term_eq** residuals, int lane_count,
                              Array<double>* const* secs, int* pos_res_arr) const override;
        bool describe_residual_rows(int&, Term_eq**, int,
                                    std::vector<ResidualRowDescriptor>&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class for an equation in a \c Domain which order is passed, for each variable.
     * \ingroup systems
     */
    class Eq_order_array : public Equation
    {

      public:
        const Array<int>& order; ///< Orders of the equation wrt each variable.

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param ord : orders of the equation wrt each variable.
         * @param ope : pointer on the operator describing the equation.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Eq_order_array(const Domain* dom, int nd, const Array<int>& ord, Ope_eq* ope, int n_cmp = -1,
                       Array<int>** p_cmp = nullptr);
        ~Eq_order_array() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class for a boundary condition.
     * The order is specified for each variable. The one corresponding to the boundary is irrelevant.
     * \ingroup systems
     */
    class Eq_bc_order_array : public Equation
    {

      public:
        int bound;               ///< The boundary.
        const Array<int>& order; ///< Orders of the equation wrt each variable.

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param bb : the boundary.
         * @param ord : orders of the equation wrt each variable.
         * @param ope : pointer on the operator describing the equation.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Eq_bc_order_array(const Domain* dom, int nd, int bb, const Array<int>& ord, Ope_eq* ope, int n_cmp = -1,
                          Array<int>** p_cmp = nullptr);
        ~Eq_bc_order_array() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Class for a matching condition.
     * The order is specified for each variable. The one corresponding to the boundary is irrelevant.
     * \ingroup systems
     */
    class Eq_matching_order_array : public Equation
    {

      public:
        int bound;               ///< The boundary.
        int other_dom;           ///< Number of the \c Domain on the other side of the boundary.
        int other_bound;         ///< Name of the boundary as seen from the other domain.
        const Array<int>& order; ///< Orders of the equation wrt each variable.

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param bb : the boundary.
         * @param oz_nd : number of the \c Domain on the other side of the boundary.
         * @param oz_bb : the boundary.
         * @param ord : orders of the equation wrt each variable.
         * @param ope : pointer on the operator.
         * @param oz_ope : pointer on the operator from the other domain.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Eq_matching_order_array(const Domain* dom, int nd, int bb, int oz_nd, int oz_bb, const Array<int>& ord,
                                Ope_eq* ope, Ope_eq* oz_ope, int n_cmp = -1, Array<int>** p_cmp = nullptr);
        ~Eq_matching_order_array() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Equation for a matching condition, except for one coefficient where an alternative condition is enforced (highly
     * specialized usage).
     * \ingroup systems
     */
    class Eq_matching_exception : public Equation
    {

      public:
        int bound;               ///< The boundary.
        int other_dom;           ///< Number of the \c Domain on the other side of the boundary.
        int other_bound;         ///< Name of the boundary as seen from the other domain.
        const Param& parameters; ///< Parameters needed for describing the exception.

        /**
         * Constructor
         * @param dom : Pointer on the \d Domain
         * @param nd : number of the \d Domain (consistence is not checked).
         * @param bb : the boundary.
         * @param oz_nd : number of the \c Domain on the other side of the boundary.
         * @param oz_bb : the boundary.
         * @param ope : pointer on the operator.
         * @param oz_ope : pointer on the operator from the other domain.
         * @param par : parameters needed for describing the exception.
         * @param ope_exc : operator for the exceptionnal condition.
         * @param n_cmp : number of components of \c eq to be considered. All the components are used of it is -1.
         * @param p_cmp : pointer on the indexes of the components to be considered. Not used of nused = -1 .
         */
        Eq_matching_exception(const Domain* dom, int nd, int bb, int oz_nd, int oz_bb, Ope_eq* ope, Ope_eq* oz_ope,
                              const Param& par, Ope_eq* ope_exc, int n_cmp = -1, Array<int>** p_cmp = nullptr);
        ~Eq_matching_exception() override; ///< Destructor.

        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    /**
     * Equation for describing a first integral equation (i.e. a constant quantity in some domains).
     * \ingroup systems
     */
    class Eq_first_integral : public Equation
    {

      public:
        int dom_min; ///< Index of the first \c Domain
        int dom_max; ///< Index of the last \c Domain
        /**
         * Audited contract that integral and constant parts transform in the
         * same reflection sector. Required because origin rows substitute the
         * constant part and cannot be described from the integral tensor alone.
         */
        bool same_reflection_sector;

        /**
         * Constructor
         * @param syst : Pointer on the associated \c System_of_eqs
         * @param dom : Pointer on the first \d Domain (needed for \c Equation constructor)
         * @param dommin : index of the first \d Domain (consistence is not checked).
         * @param dommax : index of the last \d Domain
         * @param integ_part : name of the integral quantity
         * @param const_part : equation fixing the value of the integral.
         * @param same_reflection_sector : explicit contract that the integral
         * and constant parts transform in the same y-reflection sector.
         */
        Eq_first_integral(const System_of_eqs* syst, const Domain* dom, int dommin, int dommax, const char* integ_part,
                          const char* const_part,
                          bool same_reflection_sector = false);
        ~Eq_first_integral() override; ///< Destructor.

        void apply(int&, Term_eq**) override;
        void export_val(int&, Term_eq**, Array<double>&, int&) const override;
        void export_der(int&, Term_eq**, Array<double>&, int&) const override;
        bool describe_residual_rows(int&, Term_eq**, int,
                                    std::vector<ResidualRowDescriptor>&) const override;
        Array<int> do_nbr_conditions(const Tensor& tt) const override;
        bool take_into_account(int) const override;
    };

    inline std::map<int, int> System_of_eqs::conditions_per_domain()
    {
        sec_member();
        std::map<int, int> per_dom;
        for (int i = 0; i < neq; ++i)
            per_dom[eq[i]->get_ndom()] += eq[i]->get_n_cond_tot();
        return per_dom;
    }

    inline std::map<int, int> System_of_eqs::unknowns_per_domain()
    {
        std::map<int, int> per_dom;
        for (int v = 0; v < nvar; ++v)
            for (int dd = dom_min; dd <= dom_max; ++dd)
                per_dom[dd] += var[v]->get_space().get_domain(dd)->nbr_unknowns(*var[v], dd);
        return per_dom;
    }

} // namespace Kadath
