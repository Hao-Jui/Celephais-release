/*
    Copyright 2017 Philippe Grandclement

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

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 *   2026-08-06  RAII/span modernization.
 */

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Array/memory.hpp"

namespace Kadath
{

    class System_of_eqs;

    namespace ope_action_detail
    {
        /**
         * LIFO lease for a thread-local operand scratch slot. Storage is stable
         * across nested evaluation and retained by depth rather than by every
         * node in the operator tree.
         */
        class OperandScratchLease
        {
          public:
            explicit OperandScratchLease(const System_of_eqs* owner = nullptr);
            ~OperandScratchLease();
            OperandScratchLease(const OperandScratchLease&) = delete;
            OperandScratchLease& operator=(const OperandScratchLease&) = delete;

            std::optional<Term_eq>& storage();

          private:
            std::size_t slot_;
        };

        /// Release retained operands while their referenced Space is alive.
        void release_operand_scratch(const System_of_eqs* owner);
    } // namespace ope_action_detail

    /**
     * Abstract class that describes the various operators that can appear in the equations.
     * It can not be instanciated and one must use the derived classes.
     * It works at the Term_eq level (i.e. on a given Domain and using the dual quantities required by the automatic
     * differentiation technique).
     * \ingroup systems
     */
    class Ope_eq : public MemoryMappable
    {

      protected:
        const System_of_eqs* syst; ///< The associated \c System_of_eqs
        int dom;                   ///< Index of the \c Domain where the operator is defined.
        int n_ope;                 ///< Number of terms involved (2 for + for instance, only one for sqrt...)
        std::vector<std::unique_ptr<Ope_eq>> parts; ///< Owned parts of the current operator.
        const Term_eq* borrowed_action_result_; ///< Persistent immutable result for exact identity leaves.
        bool reflection_parity_preserving_ = false;
        /**
         * Constructor. The various parts are null at this point.
         * @param syst : the associated \c System_of_eqs.
         * @param dom : the index of the \c Domain.
         * @param np : number of terms.
         */
        Ope_eq(const System_of_eqs* syst, int dom, int np);
        /**
         * Constructor. The number of terms is undefined.
         * @param syst : the associated \c System_of_eqs.
         * @param dom : the index of the \c Domain.
         */
        Ope_eq(const System_of_eqs* syst, int dom);
        Ope_eq(const Ope_eq&) = delete; ///< Copying Ope_eq is forbidden
        /**
         * Evaluates into internal optional scratch. Derived operators may
         * override this to avoid materializing a temporary when the retained
         * slot has a compatible result layout.
         */
        virtual void action_into_scratch(std::optional<Term_eq>& result) const;
      public:
        virtual ~Ope_eq(); ///< Destructor

        /**
         * @return the index of the \c Domain.
         */
        int get_dom() const { return dom; };

        /**
         * Explicit structural contract used by residual-row provenance.
         * This is registration metadata, not an inference over the operator
         * tree: callers must set it only when the complete operator is known
         * to preserve the y-reflection sector.
         */
        void set_reflection_parity_preserving(bool preserving = true)
        {
            reflection_parity_preserving_ = preserving;
        }
        bool preserves_reflection_parity() const
        {
            return reflection_parity_preserving_;
        }

        /**
         * Computes the action of the current \c Ope_eq using its various parts.
         * @return the \c Term_eq containing the result.
         */
        virtual Term_eq action() const = 0;
        /**
         * Evaluates an operand without copying a persistent exact-identity
         * result when one is available. When @p storage already contains a
         * compatible result, evaluation uses the same guarded reusable-result
         * transfer as action_into(); incompatible internal scratch is rebuilt
         * instead of imposing a stale layout. The returned reference remains
         * valid while @p storage is alive, or for the lifetime of the persistent
         * leaf target when no storage is used.
         */
        const Term_eq& action_operand(std::optional<Term_eq>& storage) const;
        /**
         * Computes this operator into a persistent result slot. The default
         * implementation attempts the reusable-result path and retains the
         * legacy assignment when the Term_eq layout preflight refuses it.
         * @return true when the reusable-result path was used.
         */
        virtual bool action_into(Term_eq& result) const;
        /**
         * Collects the names of unknown variables used by this operator.
         * @param vars : output set of variable names.
         */
        virtual void collect_vars(std::set<std::string>& vars) const;

        /**
         * Collects the \c Term_eq targets referenced by the \c Ope_id leaves of
         * this operator tree (without descending into definition bodies — the
         * caller resolves which targets are definition results and handles
         * def-to-def chains itself). Used by the do_JX MPI row partition to
         * compute the exact per-equation definition closure.
         * @param targets : output set of referenced \c Term_eq pointers.
         */
        virtual void collect_def_targets(std::set<const Term_eq*>& targets) const;

        /**
         * Diagnostic sizing for the primal-hoisting lever (VAL_CACHE_RSS_PROBE).
         * Recursively walks this operator tree; for every transform-heavy
         * intermediate node (Ope_mult / Ope_der / Ope_der_flat / Ope_der_background)
         * pushes a copy of its action() result into @p holder, so the caller can
         * measure the resident bytes a per-node val cache would add. Increments
         * @p heavy_count for cached nodes and @p total_count for every node.
         * Leaf Ope_id targets and Ope_def results already have persistent val and
         * are not counted. Not on any hot path.
         */
        void collect_heavy_node_vals(std::vector<std::unique_ptr<Term_eq>>& holder,
                                     long long& heavy_count,
                                     long long& total_count) const;
    };

    /// Cached process gate used by persistent Ope_def and Equation result sinks.
    bool ope_action_write_into_enabled();

    /**
     * Env-gated profiler scope for \c Ope_eq::action() implementations.
     *
     * Enable with \c OPE_ACTION_PROFILE=1. When disabled, construction is
     * just a cached flag check inside the existing action body.
     */
    class ScopedOpeActionProfile
    {
      public:
        explicit ScopedOpeActionProfile(const Ope_eq& op);
        ~ScopedOpeActionProfile();

      private:
        void add_child_seconds(double seconds);

        bool active_;
        const char* type_name_;
        double child_seconds_;
        std::chrono::steady_clock::time_point start_;
    };

    // Speed-of-light probe for Ope_der cached-primal lever. Returns true if the
    // OPE_DER_CACHE_PROBE environment variable is enabled. Probe semantics:
    // on cache hit, return clone of cached Term_eq with stale derivatives — output
    // is numerically wrong, only timing is meaningful. Cache cleared per J build
    // via ope_der_cache_probe_reset() from JacobianColumnEngine::reset_cache.
    bool ope_der_cache_probe_enabled();
    void ope_der_cache_probe_reset();

    // Scoped guard that gates the correct-cache path (OPE_DER_CACHE=1)
    // for Ope_der::action(). Cache populate/lookup is bypassed unless an
    // OpeDerCacheJacobianScope is active. The Jacobian column engine opens
    // the scope around each column / packed-pair compute so residual-eval
    // (sec_member) calls never engage the cache.
    class OpeDerCacheJacobianScope
    {
      public:
        OpeDerCacheJacobianScope();
        ~OpeDerCacheJacobianScope();
        OpeDerCacheJacobianScope(const OpeDerCacheJacobianScope&) = delete;
        OpeDerCacheJacobianScope& operator=(const OpeDerCacheJacobianScope&) = delete;
    };

    bool ope_der_cache_jacobian_scope_active();

    // Ope_der dispatch/time census. Enabled with OPE_DER_DISPATCH_CENSUS=1.
    // Metric_flat::derive() records per-dispatch-class call count + cumulative
    // wall seconds; JacobianColumnEngine::dump_profile() prints the table and
    // resets the counters. Used to pick the most valuable cache-v2 extension
    // target (largest non-cacheable class by wall time, not call count).
    enum class OpeDerDispatchClass : int {
        Cart_Cov_NoSum     = 0,  // already cached by v1 (the engaged subset)
        Cart_Con_NoSum     = 1,
        Cart_Cov_Sum       = 2,
        Cart_Con_Sum       = 3,
        Cart_AdaptedBypass = 4,  // derive_flat_cart path on adapted shells
        Spher_Cov_NoSum    = 5,
        Spher_Con_NoSum    = 6,
        Spher_Sum          = 7,
        Mtz_Cov_NoSum      = 8,
        Mtz_Con_NoSum      = 9,
        Mtz_Sum            = 10,
        NonFlatMetric      = 11,  // recorded by Ope_der::action only
        UnknownBasis       = 12,
        Count              = 13,
    };

    bool ope_der_dispatch_census_enabled();
    void ope_der_dispatch_census_record(OpeDerDispatchClass cls, double elapsed_seconds);
    void ope_der_dispatch_census_dump();
    void ope_der_dispatch_census_reset();

    /**
     * The operator identity.
     * \ingroup systems.
     */
    class Ope_id : public Ope_eq
    {

      protected:
        const Term_eq* target; ///< The input \c Term_eq
        int valence;           ///< Valence of the result.
        char* name_ind;        ///< The names of the various indices (if a \c Tensor of valence >0)
        Array<int>* type_ind;  ///< The type of the indices.
        bool need_sum;         ///< True if an inner contraction is needed to compute the result.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : The input \c Term_eq
         * @param valence : valence of the result (can differ from the one of so, due to inner contraction).
         * @param names : name of the indices.
         * @param ttype : type of the indices (can differ from so, in whic case a \c Metric is required to do the
         * manipulation).
         */
        Ope_id(const System_of_eqs* syst, const Term_eq* so, int valence, char* names, Array<int>* ttype);
        /**
         * Constructor with mos of the stuff uninitialized.
         * @param syst : the associated \c System_of_eqs.
         * @param so : The input \c Term_eq
         */
        Ope_id(const System_of_eqs*, const Term_eq*);
        ~Ope_id() override; ///< Destructor.
      public:
        Term_eq action() const override;
        void collect_vars(std::set<std::string>& vars) const override;
        void collect_def_targets(std::set<const Term_eq*>& targets) const override;
    };

    /**
     * The operator power-law
     * \ingroup systems.
     */
    class Ope_pow : public Ope_eq
    {

      protected:
        int power; ///< The exponent (an integer, possibly negative).
      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param pow : the exponent.
         * @param so : the source.
         */
        Ope_pow(const System_of_eqs* syst, int pow, Ope_eq* so);
        ~Ope_pow() override; ///< Destructor.

        Term_eq action() const override;
    };

    /**
     * The operator minus
     * \ingroup systems.
     */
    class Ope_minus : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : the source.
         */
        Ope_minus(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_minus() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator addition
     * \ingroup systems.
     */
    class Ope_add : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param aa : first term.
         * @param bb : second term.
         */
        Ope_add(const System_of_eqs*, Ope_eq* aa, Ope_eq* bb);
        ~Ope_add() override; ///< Destructor

        Term_eq action() const override;
        bool action_into(Term_eq& result) const override;

      protected:
        void action_into_scratch(std::optional<Term_eq>& result) const override;
    };

    /**
     * The operator substraction
     * \ingroup systems.
     */
    class Ope_sub : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param aa : first term.
         * @param bb : second term.
         */
        Ope_sub(const System_of_eqs* syst, Ope_eq* aa, Ope_eq* bb);
        ~Ope_sub() override; ///< Destructor

        Term_eq action() const override;
        bool action_into(Term_eq& result) const override;

      protected:
        void action_into_scratch(std::optional<Term_eq>& result) const override;
    };

    /**
     * The operator Multiplication.
     * When dealing with tensors it takes into account the possible contractions.
     * \ingroup systems.
     */
    class Ope_mult : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param aa : first term.
         * @param bb : second term.
         */
        Ope_mult(const System_of_eqs* syst, Ope_eq* aa, Ope_eq* bb);
        ~Ope_mult() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator Division.
     * The second term must be a double or a \c Scalar
     * \ingroup systems.
     */
    class Ope_div : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param aa : first term.
         * @param bb : second term.
         */
        Ope_div(const System_of_eqs* syst, Ope_eq* aa, Ope_eq* bb);
        ~Ope_div() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator Laplacian 3D.
     * Computes the flat 3D Laplacian
     * \ingroup systems.
     */
    class Ope_lap : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : the target.
         */
        Ope_lap(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_lap() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator time derivative.
     * Computes the first time derivative
     * \ingroup systems.
     */
    class Ope_dtime : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : the target.
         */
        Ope_dtime(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_dtime() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * Second time derivative
     * Computes the second time derivative
     * \ingroup systems
     */

    class Ope_ddtime : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs
         * @param so : the target
         */
        Ope_ddtime(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_ddtime() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator Laplacian 2D.
     * Computes the flat 2D Laplacian
     * \ingroup systems.
     */
    class Ope_lap2 : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : the target.
         */
        Ope_lap2(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_lap2() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator normal derivative
     * Computes the derivative in the direction normal to a given boundary.
     * \ingroup systems.
     */
    class Ope_dn : public Ope_eq
    {

      protected:
        int bound; ///< The boundary

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param bb : name of the boundary.
         * @param so : the target.
         */
        Ope_dn(const System_of_eqs* syst, int bb, Ope_eq* so);
        ~Ope_dn() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator flat gradient
     * Intended for systems where no metric has been defined.
     * \ingroup systems.
     */
    class Ope_grad : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : the target.
         */
        Ope_grad(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_grad() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator flat scalar product
     * Intended for systems where no metric has been defined.
     * \ingroup systems.
     */
    class Ope_scal : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param aa : first vector.
         * @param bb : second vector.
         */
        Ope_scal(const System_of_eqs* syst, Ope_eq* aa, Ope_eq* bb);
        ~Ope_scal() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator covariant derivative.
     * Inner summation is performed, if need be.
     * \ingroup systems.
     */
    class Ope_der : public Ope_eq
    {

      protected:
        int type_der; ///< Type of derivative (CON or COV)
        char ind_der; ///< Name of the index of the derivative.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param typeder : type of derivative (CON or COV)
         * @param indder : name of the index of the derivative.
         * @param so : target
         */
        Ope_der(const System_of_eqs* syst, int typeder, char indder, Ope_eq* so);
        ~Ope_der() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator covariant derivative with respect to the flat metric.
     * Inner summation is performed, if need be.
     * \ingroup systems.
     */
    class Ope_der_flat : public Ope_eq
    {

      protected:
        int type_der; ///< Type of derivative (CON or COV)
        char ind_der; ///< Name of the index of the derivative.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param typeder : type of derivative (CON or COV)
         * @param indder : name of the index of the derivative.
         * @param so : target
         */
        Ope_der_flat(const System_of_eqs* syst, int typeder, char indder, Ope_eq* so);
        ~Ope_der_flat() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator covariant derivative with respect to the background metric.
     * Inner summation is performed, if need be.
     * \ingroup systems.
     */
    class Ope_der_background : public Ope_eq
    {

      protected:
        int type_der; ///< Type of derivative (CON or COV)
        char ind_der; ///< Name of the index of the derivative.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param typeder : type of derivative (CON or COV)
         * @param indder : name of the index of the derivative.
         * @param so : target
         */
        Ope_der_background(const System_of_eqs* syst, int typeder, char indder, Ope_eq* so);
        ~Ope_der_background() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator surface integral.
     * \ingroup systems.
     */
    class Ope_int : public Ope_eq
    {

      protected:
        int bound; ///< Boundary where the integral is computed.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param bb : the boundary
         * @param so : target
         */
        Ope_int(const System_of_eqs* syst, int bb, Ope_eq* so);
        ~Ope_int() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator volume integral (in a given \c Domain)
     * \ingroup systems.
     */
    class Ope_int_volume : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_int_volume(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_int_volume() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator definition.
     * It corresponds to expressions defined by the user in the\c System_of_eqs.
     * Indices may have to be renamed and/or summed, depending on their name.
     * \ingroup systems.
     */
    class Ope_def : public Ope_eq
    {
      protected:
        Term_eq* res; ///< Result of the current definition.
      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         * @param val : valence of the result (can be different from so, due to summations)
         * @param name : names of the indices
         * @param ttype : type of the various indices (COV or CON).
         */
        Ope_def(const System_of_eqs* syst, Ope_eq* so, int val, char* name, Array<int>* ttype);
        ~Ope_def() override;
        Term_eq action() const override;
        Term_eq* get_res(); ///< Returns the result.
        void compute_res(); ///< Forces the computation of the result (when things have changed).
    };

    /**
     * The operator multiplication by \f$r\f$.
     * \ingroup systems.
     */
    class Ope_mult_r : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_mult_r(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_mult_r() override; ///< Destructor.

        Term_eq action() const override;
    };

    /**
     * The operator multiplication by \f$x\f$ (what it means depend on the \c Space considered).
     * \ingroup systems.
     */
    class Ope_mult_x : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_mult_x(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_mult_x() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator multiplication by \f$1/r partial_r\f$.
     * \ingroup systems.
     */
    class Ope_srdr : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_srdr(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_srdr() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator second radial derivative
     * \ingroup systems.
     */
    class Ope_ddr : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_ddr(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_ddr() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator first radial derivative
     * \ingroup systems.
     */
    class Ope_dr : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_dr(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_dr() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator second derivative wrt \f$varphi\f$.
     * \ingroup systems.
     */
    class Ope_ddp : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_ddp(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_ddp() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator first derivative wrt \f$theta\f$.
     * \ingroup systems.
     */
    class Ope_dt : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_dt(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_dt() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator second derivative wrt \f$theta\f$.
     * \ingroup systems.
     */
    class Ope_ddt : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_ddt(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_ddt() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator division by \f$rf$.
     * \ingroup systems.
     */
    class Ope_div_r : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_div_r(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_div_r() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator division by \f$rsin(theta)\f$.
     * \ingroup systems.
     */
    class Ope_div_rsint : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_div_rsint(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_div_rsint() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator multiplication by \f$rsin(theta)\f$.
     * \ingroup systems.
     */
    class Ope_mult_rsint : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_mult_rsint(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_mult_rsint() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator division by \f$rcos(theta)\f$.
     * \ingroup systems.
     */
    class Ope_div_cost : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_div_cost(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_div_cost() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator division by \f$x+1\f$.
     * \ingroup systems.
     */
    class Ope_div_xpone : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_div_xpone(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_div_xpone() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator division by \f$1-x^2\f$.
     * \ingroup systems.
     */
    class Ope_div_1mx2 : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_div_1mx2(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_div_1mx2() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator division by \f$1-r/L\f$ (for AADS spacetimes).
     * \ingroup systems.
     */
    class Ope_div_1mrsL : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_div_1mrsL(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_div_1mrsL() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator multiplication by \f$1-r/L\f$ (for AADS spacetimes).
     * \ingroup systems.
     */
    class Ope_mult_1mrsL : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_mult_1mrsL(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_mult_1mrsL() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator multiplication by \f$sin(theta)\f$.
     * \ingroup systems.
     */
    class Ope_mult_sint : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_mult_sint(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_mult_sint() override; ///> Destructor

        Term_eq action() const override;
    };

    class Ope_mult_cosp : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_mult_cosp(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_mult_cosp() override; ///> Destructor

        Term_eq action() const override;
    };
    /**
     * The operator division by \f$sin(theta)\f$.
     * \ingroup systems.
     */
    class Ope_div_sint : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_div_sint(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_div_sint() override; ///> Destructor

        Term_eq action() const override;
    };

    /**
     * The operator partial derivative
     * \ingroup systems.
     */
    class Ope_partial : public Ope_eq
    {
      protected:
        char ind_der; ///< name of the index
      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param name : name of the index corresponding to the derivative
         * @param so : target
         */
        Ope_partial(const System_of_eqs* syst, char name, Ope_eq* so);
        ~Ope_partial() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator determinant
     * \ingroup systems.
     */
    class Ope_determinant : public Ope_eq
    {
      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_determinant(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_determinant() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator inverse (of a \c Metric_tensor ; i.e. rank 2 symmetric tensor).
     * \ingroup systems.
     */
    class Ope_inverse : public Ope_eq
    {
      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_inverse(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_inverse() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator inverse (of a \c Metric_tensor ; i.e. rank 2 symmetric tensor).
     * It does not compute the true inverse in the sens that the cofactors are not divided by the determinant.
     * \ingroup systems.
     */
    class Ope_inverse_nodet : public Ope_eq
    {
      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_inverse_nodet(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_inverse_nodet() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * The operator partial derivative wrt one variable (same thing as Ope_partial ??)
     * \ingroup systems.
     */
    class Ope_partial_var : public Ope_eq
    {
      protected:
        int which_var;

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_partial_var(const System_of_eqs* syst, int, Ope_eq* so);
        ~Ope_partial_var() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * This operator gives the value of one coefficient of a field, on a given boundary.
     * \ingroup systems.
     */
    class Ope_mode : public Ope_eq
    {
      protected:
        int bound; ///< The boundary where the coefficients are read.
        /**
         * The desired coefficient.
         * The index corresponding to the boundary is unused.
         */
        Index pos_cf;
        double value; ///< The result is the coefficient minus value.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param bb : the boundary
         * @param ind : which coefficient.
         * @param val : the value with which the coefficient is "compared"
         * @param so : target
         */
        Ope_mode(const System_of_eqs*, int bb, const Index& ind, double val, Ope_eq* so);
        ~Ope_mode() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * This operator gives the value of one coefficient of a field.
     * \ingroup systems.
     */
    class Ope_val_mode : public Ope_eq
    {
      protected:
        Index pos_cf; ///< The desired coefficient.
        double value; ///< The result is the coefficient minus value.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param ind : which coefficient.
         * @param val : the value with which the coefficient is "compared"
         * @param so : target
         */
        Ope_val_mode(const System_of_eqs* syst, const Index& ind, double val, Ope_eq* so);
        ~Ope_val_mode() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * This operator gives the value of a field at a given collocation point.
     * \ingroup systems.
     */
    class Ope_val : public Ope_eq
    {
      protected:
        Index pos; ///< which collocation point.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param ind : which  collocation point.
         * @param so : target
         */
        Ope_val(const System_of_eqs* syst, const Index& ind, Ope_eq* so);
        ~Ope_val() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * This operator gives the value of a field at a point (arbitrary not necesseraly a collocation one)
     * \ingroup systems.
     */
    class Ope_point : public Ope_eq
    {
      protected:
        Point num; ///< Absolute coordinates of the point

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param pp : which point.
         * @param so : target
         */
        Ope_point(const System_of_eqs* syst, const Point& pp, Ope_eq* so);
        ~Ope_point() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * This operator gives the value of a field at the origin
     * \ingroup systems.
     */
    class Ope_val_ori : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param dd : index of the \c Domain where the origin is (could be different from 0).
         * @param so : target
         */
        Ope_val_ori(const System_of_eqs* syst, int dd, Ope_eq* so);
        ~Ope_val_ori() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * Operator square-root (only defined for a scalar field or a double)
     * \ingroup systems.
     */
    class Ope_sqrt : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_sqrt(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_sqrt() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * Operator square-root (only defined for a scalar field or a double).
     * Version where the output base is not standard but the anti standard base
     */
    class Ope_sqrt_anti : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_sqrt_anti(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_sqrt_anti() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * Operator square-root (only defined for a scalar field or a double).
     * Version where the output base is not standard but the one of rho (polar coordinates).
     * \ingroup systems.
     */
    class Ope_sqrt_nonstd : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_sqrt_nonstd(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_sqrt_nonstd() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * Operator exponential (only defined for a scalar field or a double)
     * \ingroup systems.
     */
    class Ope_exp : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_exp(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_exp() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * Operator logarithm (only defined for a scalar field or a double)
     * \ingroup systems.
     */
    class Ope_log : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_log(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_log() override; ///< Destructor.

        Term_eq action() const override;
    };

    /**
     * Operator  hyperbolic cosine (only defined for a scalar field or a double)
     * \ingroup systems.
     */
    class Ope_cosh : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_cosh(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_cosh() override; ///< Destructor.

        Term_eq action() const override;
    };

    /**
     * Operator  hyperbolic sine (only defined for a scalar field or a double)
     * \ingroup systems.
     */
    class Ope_sinh : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_sinh(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_sinh() override; ///< Destructor.

        Term_eq action() const override;
    };

    /**
     * Operator inverse hyperbolic tangent (only defined for a scalar field or a double)
     * \ingroup systems.
     */
    class Ope_atanh : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_atanh(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_atanh() override; ///< Destructor.

        Term_eq action() const override;
    };

    class Ope_atan : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_atan(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_atan() override; ///< Destructor.

        Term_eq action() const override;
    };

    /**
     * Operator cosine (only defined for a scalar field or a double)
     * \ingroup systems.
     */
    class Ope_cos : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_cos(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_cos() override; ///< Destructor.

        Term_eq action() const override;
    };

    /**
     * Operator sine (only defined for a scalar field or a double)
     * \ingroup systems.
     */
    class Ope_sin : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_sin(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_sin() override; ///< Destructor.

        Term_eq action() const override;
    };

    /**
     * Operator that fits a field to outgoing waves (highly specialized stuff)
     * \ingroup systems.
     */
    class Ope_fit_waves : public Ope_eq
    {
      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target field
         * @param ome : the orbital frequency (assumes an helical Killing vector).
         */
        Ope_fit_waves(const System_of_eqs* syst, Ope_eq* so, Ope_eq* ome);
        ~Ope_fit_waves() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * Operator defined by the user in the \c System_of_eqs
     * This version is intended to work with one argument.
     * \ingroup systems.
     */
    class Ope_user : public Ope_eq
    {
      protected:
        Param* par;                              ///< Parameters required by the function.
        Term_eq (*pope)(const Term_eq&, Param*); ///< The function that implements the action of the operator.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param zeope : pointer on the function that implements the action of the operator.
         * @param par : parameters required by the function.
         * @param so : the argument.
         */
        Ope_user(const System_of_eqs* syst, Term_eq (*zeope)(const Term_eq&, Param*), Param* par, Ope_eq* so);
        ~Ope_user() override; ///< Destructor
        Term_eq action() const override;
    };

    /**
     * Operator defined by the user in the \c System_of_eqs
     * This version is intended to work with two arguments.
     * \ingroup systems.
     */
    class Ope_user_bin : public Ope_eq
    {
      protected:
        Param* par; ///< Parameters required by the function.
        Term_eq (*pope)(const Term_eq&, const Term_eq&,
                        Param*); ///< The function that implements the action of the operator.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param zeope : pointer on the function that implements the action of the operator.
         * @param par : parameters required by the function.
         * @param aa : the first argument.
         * @param bb : the second argument.
         */
        Ope_user_bin(const System_of_eqs* syst, Term_eq (*zeope)(const Term_eq&, const Term_eq&, Param*), Param* par,
                     Ope_eq* aa, Ope_eq* bb);
        ~Ope_user_bin() override; ///< Destructor.
        Term_eq action() const override;
    };

    /**
     * Operator importing the values of a field from a neighborig \c Domain
     * \ingroup systems.
     */
    class Ope_import : public Ope_eq
    {

      protected:
        int bound; ///< The boundary where the field is imported.
        /**
         * 2d array containing.
         * \li in (0,*) the indexes of the domains situated on the other side of the boundary.
         * \li in (1,*) the name of the boundary, as seen by the other domains.
         */
        Array<int> others;

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param dd : index of the domain considered.
         * @param bb : the boundary.
         * @param field : the quantity to be imported.
         */
        Ope_import(const System_of_eqs* syst, int dd, int bb, const char* field);
        ~Ope_import() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * Operator changin the tensorial basis of a field.
     * \ingroup systems.
     */
    class Ope_change_basis : public Ope_eq
    {

      protected:
        int target_basis; ///< The desired tensorial basis.

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param target : the tensorial basis of the result.
         * @param so : the target field.
         */
        Ope_change_basis(const System_of_eqs* syst, int target, Ope_eq* so);
        ~Ope_change_basis() override; ///< Destructor

        Term_eq action() const override;
    };

    /**
     * Operator for a global definition (i.e. involving all the domains ; line an integral in the whole space).
     * The value is a \c Term_eq and so defined in a given domain (even if its value is computed from all the
     * domains...)
     * \ingroup systems
     */
    /**
     * Operator complex conjugation (changes the sign of m_quant).
     * \ingroup systems.
     */
    class Ope_conjug : public Ope_eq
    {

      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param so : target
         */
        Ope_conjug(const System_of_eqs* syst, Ope_eq* so);
        ~Ope_conjug() override; ///< Destructor.

        Term_eq action() const override;
    };

    class Ope_def_global : public Ope_eq
    {
      protected:
        std::unique_ptr<Term_eq> res; ///< Result
        std::vector<std::unique_ptr<Term_eq>>
            auxi; ///< Various parts of the result (i.e. the contributions of the various domains).
      public:
        /**
         * Constructor
         * @param syst : the associated \c System_of_eqs.
         * @param dom : the index of the \c Domain of the result.
         * @param name_ope : the quantity (typically should contain things like integvolume)
         */
        Ope_def_global(const System_of_eqs* syst, int dom, const char* name_ope);
        ~Ope_def_global() override;
        Term_eq action() const override;
        Term_eq* get_res(); ///< Returns the result.
        void compute_res(); ///< Forces the computation of the result (when things have changed).
    };
} // namespace Kadath
