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

#pragma once

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/memory.hpp"

#define CHEB_TYPE 1
#define LEG_TYPE 2

#define OUTER_BC 1
#define INNER_BC 2
#define CHI_ONE_BC 3
#define ETA_PLUS_BC 4
#define ETA_MINUS_BC 5
#define TIME_INIT 6

#include "For_Kadath/Array/point.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Kadath
{
    class Space;
    class Scalar;
    class System_of_eqs;
    class Val_domain;
    class Term_eq;
    class Base_tensor;
    class Metric;
    class Tensor;
    class DerAbsLaneBatch;

    struct TauSeedWrite
    {
        std::size_t coefficient_offset = 0;
        double value = 0.0;
    };

    /** Allocation-free description of one local tau-basis seed. */
    struct TauSeedDescriptor
    {
        static constexpr int max_writes = 4;

        int component = -1;
        int write_count = 0;
        std::array<TauSeedWrite, max_writes> writes{};
    };

    /** Structural provenance of one contribution to a residual row. */
    struct ResidualRowCoordinate
    {
        int domain = -1;
        int component = -1;
        int phi_basis = 0;
        int phi_index = -1;
    };

    enum class ResidualRowEquationFamily
    {
        Unavailable,
        Field,
        Integral
    };

    /**
     * Structural description of one row in canonical sec_member/export_tau
     * order.  Matching rows can carry an arbitrary number of contributors;
     * volume rows carry exactly one.
     */
    struct ResidualRowDescriptor
    {
        ResidualRowEquationFamily family =
            ResidualRowEquationFamily::Unavailable;
        int equation_index = -1;
        bool available = false;
        /** Explicit parity for scalar/global rows with no coefficient provenance. */
        signed char explicit_sector = 0;
        std::vector<ResidualRowCoordinate> sides;
    };

    class Domain : public MemoryMappable
    {

      protected:
        int num_dom;
        int ndim;
        Dim_array nbr_points;
        mutable Dim_array nbr_coefs;

        int type_base;
        mutable bool spectral_parity_guard_checked = false;

        Array<double>** coloc;
        mutable Val_domain** absol;
        mutable Val_domain** cart;
        mutable Val_domain* radius;
        mutable Val_domain** cart_surr;

        struct BasisMultCache;
        /// Bounded memo for \c multiplied_base, created on first use.
        mutable std::unique_ptr<BasisMultCache> basis_mult_cache;

        struct ImportPlanCache;
        /// Transient, domain-owned interpolation plans; never serialized.
        mutable std::unique_ptr<ImportPlanCache> import_plan_cache_;

        void clear_import_plan_cache() const;
        bool fingerprint_import_shape_component(const Val_domain& component,
                                                std::uint64_t& fingerprint) const;

        explicit Domain(int num, int ttype, const Dim_array& res);
        explicit Domain(int, BinarySource&); ///< Modern API: from BinarySource.
        Domain(const Domain& so);
        Domain(const Domain& so, bool import);

      public:
        virtual ~Domain();
        virtual void save(BinarySink&) const; ///< Modern API.
        int get_num() const { return num_dom; };
        Dim_array get_nbr_points() const { return nbr_points; };
        Dim_array get_nbr_coefs() const { return nbr_coefs; };
        int get_ndim() const { return ndim; };
        int get_type_base() const { return type_base; };
        Array<double> get_coloc(int) const;
        virtual Point get_center() const;
        virtual const Val_domain& get_chi() const;
        virtual const Val_domain& get_eta() const;
        virtual Val_domain get_X() const;
        virtual Val_domain get_T() const;

      public:
        Val_domain get_absol(int i) const;
        Val_domain get_radius() const;
        Val_domain get_cart(int i) const;
        Val_domain get_cart_surr(int i) const;

      protected:
        virtual void del_deriv() const;
        virtual void do_radius() const;
        virtual void do_cart() const;
        virtual void do_cart_surr() const;
        virtual void do_absol() const;

      public:
        void operator=(const Domain&);

      private:
        virtual void set_cheb_base(Base_spectral& so) const;
        virtual void set_legendre_base(Base_spectral& so) const;
        virtual void set_anti_cheb_base(Base_spectral& so) const;
        virtual void set_anti_legendre_base(Base_spectral& so) const;
        virtual void set_cheb_base_with_m(Base_spectral& so, int m) const;
        virtual void set_legendre_base_with_m(Base_spectral& so, int m) const;
        virtual void set_anti_cheb_base_with_m(Base_spectral& so, int m) const;
        virtual void set_anti_legendre_base_with_m(Base_spectral& so, int m) const;

        virtual void set_cheb_base_r_spher(Base_spectral& so) const;

        virtual void set_cheb_base_t_spher(Base_spectral& so) const;
        virtual void set_cheb_base_p_spher(Base_spectral& so) const;
        virtual void set_cheb_base_r_mtz(Base_spectral& so) const;

        virtual void set_cheb_base_t_mtz(Base_spectral& so) const;
        virtual void set_cheb_base_p_mtz(Base_spectral& so) const;

        virtual void set_cheb_base_rt_spher(Base_spectral& so) const;
        virtual void set_cheb_base_rp_spher(Base_spectral& so) const;
        virtual void set_cheb_base_tp_spher(Base_spectral& so) const;

        virtual void set_legendre_base_r_spher(Base_spectral& so) const;
        virtual void set_legendre_base_t_spher(Base_spectral& so) const;
        virtual void set_legendre_base_p_spher(Base_spectral& so) const;
        virtual void set_legendre_base_r_mtz(Base_spectral& so) const;
        virtual void set_legendre_base_t_mtz(Base_spectral& so) const;
        virtual void set_legendre_base_p_mtz(Base_spectral& so) const;

        virtual void set_cheb_base_x_cart(Base_spectral& so) const;
        virtual void set_cheb_base_y_cart(Base_spectral& so) const;
        virtual void set_cheb_base_z_cart(Base_spectral& so) const;
        virtual void set_cheb_base_xy_cart(Base_spectral& so) const;
        virtual void set_cheb_base_xz_cart(Base_spectral& so) const;
        virtual void set_cheb_base_yz_cart(Base_spectral& so) const;
        virtual void set_legendre_base_x_cart(Base_spectral& so) const;
        virtual void set_legendre_base_y_cart(Base_spectral& so) const;
        virtual void set_legendre_base_z_cart(Base_spectral& so) const;

        virtual void set_cheb_xodd_base(Base_spectral& so) const;
        virtual void set_legendre_xodd_base(Base_spectral&) const;
        virtual void set_cheb_todd_base(Base_spectral& so) const;
        virtual void set_legendre_todd_base(Base_spectral&) const;
        virtual void set_cheb_xodd_todd_base(Base_spectral& so) const;
        virtual void set_legendre_xodd_todd_base(Base_spectral& so) const;

        virtual void set_cheb_base_odd(Base_spectral& so) const;
        virtual void set_legendre_base_odd(Base_spectral&) const;

        void validate_spectral_parity(const Base_spectral& base) const;

        virtual void set_cheb_r_base(Base_spectral& so) const;
        virtual void set_legendre_r_base(Base_spectral& so) const;

        virtual void do_coloc();

      public:
        virtual bool is_in(const Point& xx, double prec = 1e-12) const;
        virtual const Point absol_to_num(const Point& xxx) const;
        virtual const Point absol_to_num_bound(const Point& xxx, int bound) const;
        /// Lane-aware seam; the base implementation adapts to the scalar virtual below.
        virtual void do_der_abs_from_der_var_lanes(DerAbsLaneBatch& batch) const;
        virtual void do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const;
        virtual Base_spectral mult(const Base_spectral&, const Base_spectral&) const;
        /**
         * Memoised \c mult. The basis multiplication rules read only the two
         * operands' complete, ordered basis layouts, never any mapping state, so
         * identical operand layouts reuse the previous result instead of
         * replaying the element-wise alternation switch and its three array
         * allocations. Every \c Val_domain product and quotient routes here.
         */
        Base_spectral multiplied_base(const Base_spectral& a, const Base_spectral& b) const;

      public:
        virtual double get_rmin() const;
        virtual double get_rmax() const;

        virtual Val_domain mult_cos_phi(const Val_domain&) const;
        virtual Val_domain mult_sin_phi(const Val_domain&) const;
        virtual Val_domain mult_cos_theta(const Val_domain&) const;
        virtual Val_domain mult_sin_theta(const Val_domain&) const;
        virtual Val_domain div_sin_theta(const Val_domain&) const;
        virtual Val_domain div_cos_theta(const Val_domain&) const;
        virtual Val_domain div_x(const Val_domain&) const;
        virtual Val_domain div_chi(const Val_domain&) const;
        virtual Val_domain div_xm1(const Val_domain&) const;
        virtual Val_domain div_1mx2(const Val_domain&) const;
        virtual Val_domain div_xp1(const Val_domain&) const;
        virtual Val_domain mult_xm1(const Val_domain&) const;
        virtual Val_domain div_sin_chi(const Val_domain&) const;
        virtual Val_domain mult_cos_time(const Val_domain&) const;
        virtual Val_domain mult_sin_time(const Val_domain&) const;

        virtual Tensor change_basis_cart_to_spher(int dd, const Tensor& so) const;

        virtual Tensor change_basis_spher_to_cart(int dd, const Tensor&) const;

        virtual Val_domain laplacian(const Val_domain& so, int m) const;
        virtual Val_domain laplacian2(const Val_domain& so, int m) const;

        virtual Val_domain der_r(const Val_domain&) const;

        virtual Val_domain der_t(const Val_domain&) const;

        virtual Val_domain der_p(const Val_domain&) const;

        virtual Val_domain der_r_rtwo(const Val_domain& so) const;
        virtual Val_domain srdr(const Val_domain& so) const;
        virtual Val_domain ddr(const Val_domain&) const;
        virtual Val_domain ddp(const Val_domain&) const;
        virtual Val_domain ddt(const Val_domain&) const;
        virtual Val_domain dt(const Val_domain&) const;

        virtual Val_domain dtime(const Val_domain&) const;

        virtual Val_domain ddtime(const Val_domain&) const;

        virtual const Term_eq* give_normal(int bound, int tipe) const;

        // Multipoles extraction
        virtual double multipoles_sym(int k, int j, int, const Val_domain& so, const Array<double>& passage) const;

        virtual double multipoles_asym(int, int, int, const Val_domain&, const Array<double>&) const;

        virtual Term_eq multipoles_sym(int k, int j, int bound, const Term_eq& so, const Array<double>& passage) const;
        virtual Term_eq multipoles_asym(int k, int j, int bound, const Term_eq& so, const Array<double>& passage) const;

        virtual Term_eq radial_part_sym(const Space& space, int k, int j, const Term_eq& omega,
                                        Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&),
                                        const Param& param) const;
        virtual Term_eq radial_part_asym(const Space& space, int k, int j, const Term_eq& omega,
                                         Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&),
                                         const Param& param) const;

        virtual Term_eq harmonics_sym(const Term_eq& so, const Term_eq& omega, int bound,
                                      Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&),
                                      const Param& param, const Array<double>& passage) const;

        virtual Term_eq harmonics_asym(const Term_eq&, const Term_eq&, int,
                                       Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&), const Param&,
                                       const Array<double>&) const;

        virtual Term_eq der_multipoles_sym(int k, int j, int bound, const Term_eq& so,
                                           const Array<double>& passage) const;

        virtual Term_eq der_multipoles_asym(int k, int j, int bound, const Term_eq& so,
                                            const Array<double>& passage) const;

        virtual Term_eq der_radial_part_asym(const Space& space, int k, int j, const Term_eq& omega,
                                             Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&),
                                             const Param& param) const;

        virtual Term_eq der_radial_part_sym(const Space& space, int k, int j, const Term_eq& omega,
                                            Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param& param),
                                            const Param& param) const;

        virtual Term_eq der_harmonics_sym(const Term_eq& so, const Term_eq& omega, int bound,
                                          Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&),
                                          const Param& param, const Array<double>& passage) const;

        virtual Term_eq der_harmonics_asym(const Term_eq& so, const Term_eq& omega, int bound,
                                           Term_eq (*f)(const Space&, int, int, const Term_eq&, const Param&),
                                           const Param& param, const Array<double>& passage) const;

        // Abstract stuff
      protected:
        Term_eq do_comp_by_comp(const Term_eq& so, Val_domain (Domain::*pfunc)(const Val_domain&) const) const;
        Term_eq do_comp_by_comp_with_int(const Term_eq& so, int val,
                                         Val_domain (Domain::*pfunc)(const Val_domain&, int) const) const;

      public:
        virtual Term_eq der_normal_term_eq(const Term_eq& so, int bound) const;

        virtual Term_eq div_1mx2_term_eq(const Term_eq&) const;

        virtual Term_eq lap_term_eq(const Term_eq& so, int m) const;

        virtual Term_eq lap2_term_eq(const Term_eq& so, int m) const;

        virtual Term_eq mult_r_term_eq(const Term_eq& so) const;

        virtual Term_eq integ_volume_term_eq(const Term_eq& so) const;

        virtual Term_eq grad_term_eq(const Term_eq& so) const;

        virtual Term_eq div_r_term_eq(const Term_eq&) const;

        virtual Term_eq integ_term_eq(const Term_eq& so, int bound) const;

        virtual Term_eq dr_term_eq(const Term_eq& so) const;

        virtual Term_eq dtime_term_eq(const Term_eq& so) const;

        virtual Term_eq ddtime_term_eq(const Term_eq& so) const;

        virtual Term_eq derive_flat_spher(int tipe, char ind, const Term_eq& so, const Metric* manip) const;

        virtual Term_eq derive_flat_mtz(int tipe, char ind, const Term_eq& so, const Metric* manip) const;

        virtual Term_eq derive_flat_cart(int tipe, char ind, const Term_eq& so, const Metric* manip) const;

        virtual int nbr_unknowns_from_adapted() const { return 0; };
        virtual void vars_to_terms() const {};
        virtual void affecte_coef(int& conte, int cc, bool& found) const {};
        virtual void xx_to_vars_from_adapted(Val_domain& shape, const Array<double>& xx, int& conte) const {};

        virtual void xx_to_vars_from_adapted(double bound, const Array<double>& xx, int& conte) const {};

        virtual void xx_to_ders_from_adapted(const Array<double>& xx, int& conte) const {};

        virtual void update_term_eq(Term_eq* so) const;
        /**
         * Adds the variable-domain mapping derivative to a term whose field
         * derivative is already seeded. The default retains the exact
         * replacement-based fallback; adapted domains may override with a
         * direct pertinent-domain accumulation.
         */
        virtual void accumulate_term_eq_mapping_derivative(Term_eq* so) const;
        virtual void update_variable(const Val_domain& shape, const Scalar& oldval, Scalar& newval) const {};

        virtual void update_variable(double bound, const Scalar& oldval, Scalar& newval) const {};

        virtual void update_constante(const Val_domain& shape, const Scalar& oldval, Scalar& newval) const {};

        virtual void update_constante(double bound, const Scalar& oldval, Scalar& newval) const {};

        virtual void update_mapping(const Val_domain& shape) {};
        virtual void update_mapping(double bound) {};

        /**
         * Snapshot/restore of the mutable surface-mapping state for a Newton
         * line search. Non-adapted domains hold no mutable mapping (default
         * no-op). Adapted domains append their radius component(s) to \c out in
         * \c snapshot_mapping and consume the same number of entries (advancing
         * \c idx) in \c restore_mapping, in matching domain order. Both are
         * \c const so they can be driven through \c Space::get_domain.
         */
        virtual void snapshot_mapping(std::vector<Val_domain>& out) const {};
        virtual void restore_mapping(const std::vector<Val_domain>& in, std::size_t& idx) const {};

        /**
         * Mix mutable shape content used by is_in()/absol_to_num() into a cache
         * fingerprint. Audited immutable domain types may contribute no bytes by
         * overriding this with true. A false result makes the caller bypass caching.
         */
        virtual bool fingerprint_import_shape(std::uint64_t&) const { return false; }

      protected:
        friend class System_of_eqs;
        /// Capture mapping components into reusable storage, advancing \c idx.
        /// The default adapter preserves compatibility with domain types that
        /// only implement the append-only snapshot_mapping overload.
        virtual void snapshot_mapping_into(std::vector<Val_domain>& out, std::size_t& idx) const;
        /// Copy one mapping component into an existing slot when topology is
        /// unchanged, or append the first time a snapshot is captured.
        void snapshot_mapping_component(const Val_domain& component,
                                        std::vector<Val_domain>& out,
                                        std::size_t& idx) const;

      public:

        virtual void set_val_inf(Val_domain& so, double xx) const;
        virtual Val_domain mult_r(const Val_domain& so) const;
        virtual Val_domain mult_x(const Val_domain& so) const;
        virtual Val_domain div_r(const Val_domain& so) const;
        virtual Val_domain div_1mrsL(const Val_domain& so) const;
        virtual Val_domain mult_1mrsL(const Val_domain& so) const;

        virtual double integ_volume(const Val_domain&) const;

        virtual void find_other_dom(int dom, int bound, int& otherdom, int& otherbound) const;
        virtual Val_domain der_normal(const Val_domain& so, int bound) const;
        virtual Val_domain der_partial_var(const Val_domain& so, int ind) const;

        virtual double integ(const Val_domain&, int) const;
        virtual double integmoment(const Val_domain&, int, int) const;
        virtual double integrale(const Val_domain&) const;

        virtual Term_eq partial_spher(const Term_eq& so) const;
        virtual Term_eq partial_cart(const Term_eq& so) const;
        virtual Term_eq partial_mtz(const Term_eq& so) const;
        virtual Term_eq connection_spher(const Term_eq& so) const;
        virtual Term_eq connection_mtz(const Term_eq& so) const;

        // Derivative-only variants for the cached-primal Ope_der path.
        // Skip the value-side spectral derivative work and reuse the caller-
        // provided val_t. Derivative lanes still build from so.der_t(lane).
        // Subclasses that override partial_*/connection_* with a non-default
        // body must also override these to stay consistent with the cache.
        virtual Term_eq partial_spher_with_cached_value(const Term_eq& so, const Tensor& cached_val) const;
        virtual Term_eq partial_cart_with_cached_value(const Term_eq& so, const Tensor& cached_val) const;
        virtual Term_eq partial_mtz_with_cached_value(const Term_eq& so, const Tensor& cached_val) const;
        virtual Term_eq connection_spher_with_cached_value(const Term_eq& so, const Tensor& cached_val) const;
        virtual Term_eq connection_mtz_with_cached_value(const Term_eq& so, const Tensor& cached_val) const;

        virtual double val_boundary(int bound, const Val_domain& so, const Index& ind) const;
        virtual int nbr_points_boundary(int bound, const Base_spectral& base) const;
        virtual void do_which_points_boundary(int bound, const Base_spectral& base, Index** ind, int start) const;

        virtual int nbr_unknowns(const Tensor& so, int dom) const;
        virtual Array<int> nbr_conditions(const Tensor& eq, int dom, int order, int n_cmp = -1,
                                          Array<int>** p_cmp = nullptr) const;
        virtual Array<int> nbr_conditions_boundary(const Tensor& eq, int dom, int bound, int n_cmp = -1,
                                                   Array<int>** p_cmp = nullptr) const;

        virtual void export_tau(const Tensor& eq, int dom, int order, Array<double>& res, int& pos_res,
                                const Array<int>& ncond, int n_cmp = -1, Array<int>** p_cmp = nullptr) const;

        /**
         * Describes volume rows in exactly the order emitted by export_tau.
         * Unsupported domains clear \c descriptors and return false.
         */
        virtual bool describe_volume_residual_rows(
            const Tensor& eq, int dom, int order, const Array<int>& ncond,
            int n_cmp, Array<int>** p_cmp,
            std::vector<ResidualRowDescriptor>& descriptors) const;

        /** Helpers shared by structural row descriptors. */
        bool residual_tensor_components_in_tau_order(
            const Tensor& eq, int dom, int n_cmp, Array<int>** p_cmp,
            std::vector<int>& components) const;
        bool append_volume_residual_row(
            const Val_domain& field, int dom, int component, int phi_index,
            ResidualRowDescriptor& descriptor) const;

        /**
         * Describes boundary rows in exactly the order emitted by
         * export_tau_boundary. Unsupported domains clear \c descriptors and
         * return false.
         */
        virtual bool describe_boundary_residual_rows(
            const Tensor& eq, int dom, int bound, const Array<int>& ncond,
            int n_cmp, Array<int>** p_cmp,
            std::vector<ResidualRowDescriptor>& descriptors) const;

        virtual void export_tau_boundary(const Tensor& eq, int dom, int bound, Array<double>& res, int& pos_res,
                                         const Array<int>& ncond, int n_cmp = -1, Array<int>** p_cmp = nullptr) const;

        virtual void export_tau_boundary_exception(const Tensor& eq, int dom, int bound, Array<double>& res,
                                                   int& pos_res, const Array<int>& ncond, const Param& param,
                                                   int type_exception, const Tensor& exception, int n_cmp = -1,
                                                   Array<int>** p_cmp = nullptr) const;

        virtual void affecte_tau(Tensor& so, int dom, const Array<double>& cf, int& pos_cf) const;

        virtual void affecte_tau_one_coef(Tensor& so, int dom, int cc, int& pos_cf) const;

        /**
         * Parity of azimuthal coefficient \c k under physical y reflection.
         * The default implements phi -> -phi for COSSIN, COS, and SIN bases;
         * domains whose azimuth winds about a different Cartesian axis
         * override it.  Zero means the supplied basis is unsupported.
         */
        virtual int phi_coefficient_parity(int k,
                                           int phi_basis = COSSIN) const;

        /**
         * Describes every local tau seed in affecte_tau_one_coef order.
         * Unsupported domains return false so callers can retain the legacy
         * materializing path.
         */
        virtual bool describe_tau_seed_block(const Tensor& so, int dom,
                                             std::vector<TauSeedDescriptor>& descriptors) const;

        /** Materializes a validated descriptor into one tensor domain. */
        bool materialize_tau_seed(Tensor& target, const Tensor& source, int dom,
                                  const TauSeedDescriptor& descriptor) const;

        virtual Array<int> nbr_conditions_array(const Tensor& eq, int dom, const Array<int>& order, int n_cmp = -1,
                                                Array<int>** p_cmp = nullptr) const;
        virtual Array<int> nbr_conditions_boundary_array(const Tensor& eq, int dom, int bound, const Array<int>& order,
                                                         int n_cmp = -1, Array<int>** p_cmp = nullptr) const;

        virtual void export_tau_array(const Tensor& eq, int dom, const Array<int>& order, Array<double>& res,
                                      int& pos_res, const Array<int>& ncond, int n_cmp = -1,
                                      Array<int>** p_cmp = nullptr) const;
        virtual void export_tau_boundary_array(const Tensor& eq, int dom, int bound, const Array<int>& order,
                                               Array<double>& res, int& pos_res, const Array<int>& ncond,
                                               int n_cmp = -1, Array<int>** p_cmp = nullptr) const;

        virtual Array<int> nbr_conditions_boundary_one_side(const Tensor& eq, int dom, int bound, int n_cmp = -1,
                                                            Array<int>** p_cmp = nullptr) const;

        virtual void export_tau_boundary_one_side(const Tensor& eq, int dom, int bound, Array<double>& res,
                                                  int& pos_res, const Array<int>& ncond, int n_cmp = -1,
                                                  Array<int>** p_cmp = nullptr) const;

        virtual int give_place_var(char* name) const;

        Term_eq import(int numdom, int bound, int n_ope, Term_eq** parts) const;

        /**
         * Optional point-major packed import.  Input tensors are stored as
         * [value, derivative lane 0, ...] x n_ope; null derivative inputs
         * denote an identically-zero lane.  The base implementation refuses
         * the fast path so Domain::import retains its scalar virtual loop.
         */
      protected:
        enum class ImportLanePlanLayout { RadialBoundary, BisphericRectOuter };
        bool import_lanes_point_major(int numdom, int bound, int n_ope, const Array<int>& other_doms,
                                      int tensor_lane_count, Tensor* const* lane_parts, Tensor** lane_results,
                                      ImportLanePlanLayout layout, double lookup_precision,
                                      bool force_cartesian_basis) const;

      public:
        virtual bool import_lanes_native(int numdom, int bound, int n_ope, const Array<int>& other_doms,
                                         int tensor_lane_count, Tensor* const* lane_parts,
                                         Tensor** lane_results) const;

        virtual Tensor import(int numdom, int bound, int n_ope, const Array<int>& other_doms, Tensor** parts) const;

        virtual void filter(Tensor& tt, int dom, double treshold) const;

      public:
        virtual ostream& print(ostream& o) const = 0;

        friend ostream& operator<<(ostream& o, const Domain& so) { return so.print(o); };
        friend class Val_domain;
        friend class Metric_ADS;
        friend class Metric_AADS;
    };

    /**
     * One contiguous block of variable-domain columns, represented by the
     * domain whose adapted-radius basis enumerates that block.
     */
    struct VariableDomainBlock
    {
        int domain = -1;
        int first_column = 0;
        int column_count = 0;
    };

    class Space : public MemoryMappable
    {

      protected:
        int nbr_domains;
        int ndim;
        int type_base;
        Domain** domains;
        Space();
        virtual ~Space();

      public:
        int get_ndim() const { return ndim; };
        int get_nbr_domains() const { return nbr_domains; };
        int get_type_base() const { return type_base; };
        virtual void save(BinarySink&) const; ///< Modern API.

        const Domain* get_domain(int i) const
        {
            assert((i >= 0) && (i < nbr_domains));
            return domains[i];
        }

        Scalar get_cart_field(int const cart) const;
        Scalar get_cart_field_bound(int const cart, int const bound) const;

        // Things for adapted domains
        virtual int nbr_unknowns_from_variable_domains() const { return 0; };
        virtual bool describe_variable_domain_blocks(
            std::vector<VariableDomainBlock>& blocks) const
        {
            blocks.clear();
            const int total = nbr_unknowns_from_variable_domains();
            if (total == 0)
                return true;
            for (int domain = 0; domain < nbr_domains; ++domain) {
                const int count = domains[domain]->nbr_unknowns_from_adapted();
                if (count == total) {
                    blocks.push_back({domain, 0, count});
                    return true;
                }
            }
            return false;
        };
        virtual void affecte_coef_to_variable_domains(int& conte, int cc, Array<int>& doms) const {};
        // Explicit opt-in seam for independently seeded packed geometry tangents.
        // Adapted spaces remain scalar-only unless they override all three hooks.
        virtual bool supports_packed_variable_domain_jacobian() const { return false; };
        virtual bool affecte_coef_to_variable_domains_lane(
            int& conte, int cc, int lane, int lane_count, Array<int>& doms) const
        {
            return false;
        };
        virtual void restore_scalar_variable_domain_derivatives() const {};
        virtual void xx_to_ders_variable_domains(const Array<double>& xx, int& conte) const {};
        virtual void xx_to_vars_variable_domains(System_of_eqs* syst, const Array<double>& xx, int& pos) const {};

        virtual Array<int> get_indices_matching_non_std(int dom, int bound) const;

        friend ostream& operator<<(ostream& o, const Space& so);
    };
} // namespace Kadath
