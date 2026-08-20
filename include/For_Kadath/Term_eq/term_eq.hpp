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
 */

#pragma once

#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Tensor/metric_tensor.hpp"
#include "For_Kadath/Array/memory.hpp"

#include <memory>

namespace Kadath
{
    constexpr int TERM_D = 0;
    constexpr int TERM_T = 1;

    class Term_eq;
    class TermEqExtraDerivativeLanes;
    class Space_adapted_bh_polar;

    ostream& operator<<(ostream&, const Term_eq&);
    Term_eq operator+(const Term_eq&, const Term_eq&);
    Term_eq operator-(const Term_eq&, const Term_eq&);
    Term_eq operator*(const Term_eq&, const Term_eq&);
    Term_eq operator/(const Term_eq&, const Term_eq&);

    Term_eq pow(const Term_eq&, int);
    Term_eq operator*(int, const Term_eq&);
    Term_eq operator*(const Term_eq&, int);
    Term_eq operator*(double, const Term_eq&);
    Term_eq operator*(const Term_eq&, double);
    Term_eq operator/(const Term_eq&, double);
    Term_eq partial(const Term_eq&, char);
    Term_eq bessel_jl(const Term_eq&, int);
    Term_eq bessel_yl(const Term_eq&, int);
    Term_eq bessel_djl(const Term_eq&, int);
    Term_eq bessel_dyl(const Term_eq&, int);
    Term_eq sqrt(const Term_eq&);
    Term_eq div_1mx2(const Term_eq&);

    Term_eq scalar_product(const Term_eq&, const Term_eq&);

    /**
     * This class is intended to describe the manage objects appearing in the equations.
     * Basically a \c Term_eq can be either a double or a tensorial field.
     * It is also defined only in a given \c Domain and not in the whole space.
     * ALong with the quantity itself it also contains its variation that is used by the automatic differentiation
     * algorithm to compute the Jacobian of the system.
     * \ingroup systems
     */

    class Term_eq : public MemoryMappable
    {

      protected:
        const int dom; ///< Index of the \c Domain where the \c Term_eq is defined.
        double* val_d; ///< Pointer on the value, if the \c Term_eq is a double.
        double* der_d; ///< Pointer on the variation if the \c Term_eq is a double.
        Tensor* val_t; ///< Pointer on the value, if the \c Term_eq is a \c Tensor.
        Tensor* der_t; ///< Pointer on the variation, if the \c Term_eq is a \c Tensor.
        int n_derivative_lanes; ///< Configured tangent-tile width. Lane 0 is the legacy derivative pointer above.
        /// Lazily allocated storage for packed-column AD lanes 1--31. Lane 0
        /// stays on the legacy `der_d` / `der_t` pointer.
        std::unique_ptr<TermEqExtraDerivativeLanes> extra_derivative_lanes;
        /**
         * Flag describing the type of data :
         * \li TERM_D for a double
         * \li TERM_T for a tensorial field.
         */
        const int type_data;

      public:
        static constexpr int max_derivative_lanes = 32;

        /**
         * Constructor for a double type \c Term_eq. Only the value is initialized.
         * @param dom : the domain.
         * @param val : the value (as an integer).
         */
        Term_eq(int dom, int val);
        /**
         * Constructor for a double type \c Term_eq. Only the value is initialized.
         * @param dom : the domain.
         * @param val : the value.
         */
        Term_eq(int dom, double val);
        /**
         * Constructor for a double type \c Term_eq.
         * @param dom : the domain.
         * @param val : the value.
         * @param der : the variation.
         */
        Term_eq(int dom, double val, double der);
        /**
         * Constructor for a tensorial field \c Term_eq. Only the value is initialized.
         * @param dom : the domain.
         * @param val : the value.
         */
        Term_eq(int dom, const Tensor& val);
        /**
         * Constructor for a tensorial field \c Term_eq. Only the value is initialized.
         * @param dom : the domain.
         * @param val : the value.
         * @param der : the variation.
         */
        Term_eq(int dom, const Tensor& val, const Tensor& der);
        Term_eq(const Term_eq&); ///< Copy constructor.
        /**
         * Move constructor. Transfers the value, variation and packed-lane
         * storage, leaving the source empty and safe to destroy. Only
         * ownership moves, so every transferred field keeps its exact bits.
         *
         * There is no move assignment: \c dom and \c type_data are const, so
         * a moved-from term could never be re-targeted.
         */
        Term_eq(Term_eq&&) noexcept;
        ~Term_eq();              ///< Destructor

        double get_val_d() const; ///< @return the double value.
        double get_der_d() const; ///< @return the double variation.
        double get_der_d(int lane) const; ///< @return the double variation in the requested tangent lane.
        bool has_der_d(int lane) const; ///< @return true when a double derivative lane is logically present.
        const Tensor& get_val_t() const; ///< @return the tensorial value.
        const Tensor& get_der_t() const; ///< @return the tensorial variation.
        const Tensor& get_der_t(int lane) const; ///< @return the tensorial variation in the requested tangent lane.
        bool has_der_t(int lane) const; ///< @return true when a tensor derivative lane is logically present.
        /**
         * @return a pointer on the tensorial value.
         */
        const Tensor* get_p_val_t() const { return val_t; };
        /**
         * @return a pointer on the tensorial variation.
         */
        const Tensor* get_p_der_t() const { return der_t; };
        const Tensor* get_p_der_t(int lane) const;
        int get_derivative_lane_count() const { return n_derivative_lanes; };
        void set_derivative_lane_count(int lane_count);
        /**
         * Zeros lane 0 and deactivates packed lanes while retaining their
         * backing storage for reuse by the next assembly tile.
         */
        void reset_derivative_tile_retain_storage(int lane_count);
        /** Zeros lane 0 and releases all packed-lane backing storage. */
        void reset_derivative_tile(int lane_count);
        /**
         * @return the type of data (TERM_D or TERM_T)
         */
        int get_type_data() const { return type_data; };
        /**
         * @return the index of the \c Domain.
         */
        int get_dom() const { return dom; };

        void operator=(const Term_eq&); ///< Assignment operator.
        /**
         * Reuses the storage of a temporary operator result without changing
         * this term's persistent tensor structure. Returns false, without
         * mutation, when the result cannot be transferred equivalently to the
         * legacy assignment path.
         */
        bool try_write_action_result(Term_eq& result);
        /**
         * Writes a same-type addition or subtraction directly into reusable
         * storage. The fast path is intentionally limited to doubles and
         * scalar-shaped tensors; unsupported layouts and aliases return false
         * without mutation so callers can use the materialized fallback.
         */
        bool try_write_sum_action_result(const Term_eq& lhs, const Term_eq& rhs, bool subtract);
        /**
         * Accumulates a newly materialized derivative contribution with an
         * existing tensor derivative lane, touching only this term's pertinent
         * domain. On success the installed value is exactly
         * \c contribution + existing, preserving the legacy operand order, and
         * the contribution's pertinent-domain storage is consumed. Unsupported
         * layouts return false without mutating either operand so callers can
         * retain a materialized fallback.
         */
        bool try_accumulate_der_t(int lane, Tensor&& contribution);

        void set_val_d(double); ///< Sets the double value.
        void set_der_d(double); ///< Sets the double variation.
        void set_der_d(int lane, double); ///< Sets the double variation for one tangent lane.
        void set_val_t(const Tensor&); ///< Sets the tensorial value (only the values in the pertinent \c Domain are copied).
        void set_der_t(
            const Tensor&);  ///< Sets the tensorial variation (only the values in the pertinent \c Domain are copied).
        void set_der_t(int lane, const Tensor&); ///< Sets the tensorial variation for one tangent lane.
        /**
         * Sets a tangent-lane variation from an expiring tensor, transferring
         * the pertinent domain's storage instead of copying it. The installed
         * values, index names, component layout and tensor parameters are the
         * same as the copying overload would produce; refuses the transfer and
         * falls back to that overload whenever they would differ.
         */
        void set_der_t(int lane, Tensor&&);
        void set_der_zero(); ///< Sets the variation of the approriate type to zero.
        void set_der_zero(int lane); ///< Sets one tangent lane variation of the appropriate type to zero.
        void clear_der();    ///< Releases derivative storage (sets derivative pointer(s) to null).
        void clear_der(int lane); ///< Releases derivative storage for one tangent lane.
        /**
         * Read/write accessor to the tensorial value.
         */
        Tensor* set_val_t() { return val_t; };
        /**
         * Read/write accessor to the tensorial derivative.
         */
        Tensor* set_der_t() { return der_t; };
        Tensor* set_der_t(int lane);

        /**
         * Computes the derivative wrt to an absolute coordinate (i.e. like the Cartesian ones).
         * @param i : the index of the coordinate (from 1 to the dimension).
         */
        Term_eq der_abs(int i) const;

        friend ostream& operator<<(ostream&, const Term_eq&); ///< Display
        friend Term_eq operator+(const Term_eq&, const Term_eq&);
        friend Term_eq operator-(const Term_eq&, const Term_eq&);
        friend Term_eq operator*(const Term_eq&, const Term_eq&);
        friend Term_eq operator/(const Term_eq&, const Term_eq&);
        friend Term_eq scalar_product(const Term_eq&, const Term_eq&);

        friend Term_eq operator*(int, const Term_eq&);
        friend Term_eq operator*(const Term_eq&, int);
        friend Term_eq operator*(double, const Term_eq&);
        friend Term_eq operator*(const Term_eq&, double);
        friend Term_eq operator*(const Scalar&, const Term_eq&);
        friend Term_eq operator/(const Term_eq&, double);
        friend Term_eq pow(const Term_eq&, int);
        friend Term_eq sqrt(const Term_eq&);
        friend Term_eq partial(const Term_eq&, char);

        friend Term_eq div_1mx2(const Term_eq&);

        friend Term_eq bessel_jl(const Term_eq&, int);
        friend Term_eq bessel_yl(const Term_eq&, int);
        friend Term_eq bessel_djl(const Term_eq&, int);
        friend Term_eq bessel_dyl(const Term_eq&, int);
        friend Term_eq fjl(const Space&, int, int, const Term_eq&, const Param&);
        friend Term_eq fyl(const Space&, int, int, const Term_eq&, const Param&);

        friend Term_eq operator+(const Term_eq&, double);
        friend Term_eq operator+(double, const Term_eq&);
        friend Term_eq operator-(const Term_eq&);

        friend class Ope_lap;
        friend class Ope_lap2;
        friend class Ope_dn;
        friend class Ope_int;
        friend class Ope_grad;
        friend class Ope_id;
        friend class Ope_der;
        friend class Ope_der_flat;
        friend class Ope_der_background;
        friend class Ope_mult_r;
        friend class Ope_mult_x;
        friend class Ope_div_rsint;
        friend class Ope_mult_rsint;
        friend class Ope_div_r;
        friend class Ope_div_sint;
        friend class Ope_div_cost;
        friend class Ope_mult_sint;
        friend class Ope_mult_cosp;
        friend class Ope_div_xpone;
        friend class Ope_partial;
        friend class Ope_partial_var;
        friend class Ope_determinant;
        friend class Ope_inverse;
        friend class Ope_inverse_nodet;
        friend class Ope_mode;
        friend class Ope_val_mode;
        friend class Ope_val;
        friend class Ope_point;
        friend class Ope_val_ori;
        friend class Ope_sqrt;
        friend class Ope_sqrt_anti;
        friend class Ope_sqrt_nonstd;
        friend class Ope_def;
        friend class Ope_def_global;
        friend class Ope_srdr;
        friend class Ope_ddp;
        friend class Ope_ddt;
        friend class Ope_dt;
        friend class Ope_ddr;
        friend class Ope_dr;
        friend class Ope_exp;
        friend class Ope_log;
        friend class Ope_atanh;
        friend class Ope_atan;
        friend class Ope_cosh;
        friend class Ope_sinh;
        friend class Ope_cos;
        friend class Ope_sin;
        friend class Ope_int_volume;
        friend class Ope_fit_waves;
        friend class Ope_change_basis;
        friend class Ope_mult_1mrsL;
        friend class Ope_div_1mrsL;
        friend class Ope_div_1mx2;
        friend class Ope_dtime;
        friend class Ope_ddtime;
        friend class Domain;
        friend class Domain_nucleus;
        friend class Domain_shell;
        friend class Domain_compact;
        friend class Domain_shell_outer_adapted;
        friend class Domain_shell_inner_adapted;
        friend class Domain_polar_shell_outer_adapted;
        friend class Domain_polar_shell_inner_adapted;
        friend class Domain_polar_shell_bilateral_adapted;
        friend class Domain_polar_shell_inner_homothetic;
        friend class Domain_polar_shell_outer_homothetic;
        friend class Domain_bispheric_rect;
        friend class Domain_bispheric_eta_first;
        friend class Domain_bispheric_chi_first;
        friend class Domain_bispheric_rect_nosym;
        friend class Domain_bispheric_eta_first_nosym;
        friend class Domain_bispheric_chi_first_nosym;
        friend class Domain_nucleus_symphi;
        friend class Domain_shell_symphi;
        friend class Domain_compact_symphi;
        friend class Domain_nucleus_nosym;
        friend class Domain_shell_nosym;
        friend class Domain_compact_nosym;
        friend class Domain_shell_inner_adapted_nosym;
        friend class Domain_shell_outer_adapted_nosym;
        friend class Space_spheric_adapted_nosym;
        friend class Space_bin_ns_nosym;
        friend class Metric;
        friend class Metric_general;
        friend class Metric_flat;
        friend class Metric_dirac;
        friend class Metric_dirac_const;
        friend class Metric_harmonic;
        friend class Metric_conf;
        friend class Metric_const;
        friend class Metric_ADS;
        friend class Metric_AADS;
        friend class Metric_conf_factor;
        friend class Metric_conf_factor_const;
        friend class System_of_eqs;
        friend class Space_spheric_adapted;
        friend class Space_spheric_homothetic;
        friend class Space_trumpet;
        friend class Space_polar_adapted;
        friend class Space_polar_bilateral_adapted;
        friend class Space_adapted_bh_polar;
        friend class Space_bin_ns;
        friend class Space_bin_ns_nosym;
        friend class Space_bhns;
        friend class Space_bhns_nosym;
        friend class Space_bin_bh;
        friend class Metric_flat_nophi;
        friend class Metric_nophi;
        friend class Metric_nophi_const;
        friend class Metric_nophi_AADS;
        friend class Metric_nophi_AADS_const;
        friend class Metric_cfc;
        friend class Domain_polar_periodic_nucleus;
        friend class Domain_polar_periodic_shell;
        friend class Space_polar_periodic;
        friend class Space_adapted_bh;
        friend class Space_adapted_bh_nosym;
        friend class Space_KerrSchild_bh;
        friend class Space_Kerr_bbh;
        friend class Space_bbh;
        friend class Space_three_body;
    };
} // namespace Kadath
