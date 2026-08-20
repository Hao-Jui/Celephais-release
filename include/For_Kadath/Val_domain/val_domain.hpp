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

#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/memory.hpp"

#include <span>

namespace Kadath
{

    class DerAbsLaneBatch;

    Val_domain sin(const Val_domain&);
    Val_domain cos(const Val_domain&);
    Val_domain operator+(const Val_domain&);
    Val_domain operator-(const Val_domain&);
    Val_domain operator+(const Val_domain&, const Val_domain&);
    Val_domain operator+(Val_domain&&, const Val_domain&);
    Val_domain operator+(const Val_domain&, double);
    Val_domain operator+(double, const Val_domain&);
    Val_domain operator-(const Val_domain&, const Val_domain&);
    Val_domain operator-(Val_domain&&, const Val_domain&);
    Val_domain operator-(const Val_domain&, double);
    Val_domain operator-(double, const Val_domain&);
    Val_domain operator*(const Val_domain&, const Val_domain&);
    Val_domain operator*(Val_domain&&, const Val_domain&);
    Val_domain operator*(const Val_domain&, double);
    Val_domain operator*(double, const Val_domain&);
    Val_domain operator*(const Val_domain&, int);
    Val_domain operator*(int, const Val_domain&);
    Val_domain operator*(const Val_domain&, long int);
    Val_domain operator*(long int, const Val_domain&);
    Val_domain operator/(const Val_domain&, const Val_domain&);
    Val_domain operator/(const Val_domain&, double);
    Val_domain operator/(double, const Val_domain&);
    Val_domain pow(const Val_domain&, int);
    Val_domain pow(const Val_domain&, double);
    Val_domain sqrt(const Val_domain&);
    Val_domain exp(const Val_domain&);
    Val_domain log(const Val_domain&);
    Val_domain atan(const Val_domain&);
    Val_domain atanh(const Val_domain&);
    double diffmax(const Val_domain&, const Val_domain&);
    Val_domain bessel_jl(const Val_domain&, int);
    Val_domain bessel_yl(const Val_domain&, int);
    Val_domain bessel_djl(const Val_domain&, int);
    Val_domain bessel_dyl(const Val_domain&, int);
    double maxval(const Val_domain&);

    // Empty the assembly-scoped Val_domain::compute_der_abs memo table.
    // Called by JacobianColumnEngine::reset_cache when the engine drops its
    // per-column scratch (matrix re-assembly boundary).
    void reset_val_domain_der_abs_cache();

    // RAII guard that arms the assembly-scoped der_abs cache for the lifetime
    // of the scope. Default-on for Jacobian assembly via
    // VAL_DOMAIN_DER_ABS_CACHE (opt out with =0). Exact Jv / GMRES
    // runs uncached unless an explicit diagnostic scope is installed. Cached
    // derivative tensors are released on destruction so RSS does not leak
    // beyond the active solve region.
    class ValDomainDerAbsAssemblyCacheScope
    {
    public:
        ValDomainDerAbsAssemblyCacheScope();
        ~ValDomainDerAbsAssemblyCacheScope();
        ValDomainDerAbsAssemblyCacheScope(const ValDomainDerAbsAssemblyCacheScope&) = delete;
        ValDomainDerAbsAssemblyCacheScope& operator=(const ValDomainDerAbsAssemblyCacheScope&) = delete;
    };

    /**
     * Class for storing the basis of decompositions of a field and its values on both the configuration
     * and coefficients spaces, in a given \c Domain.
     *
     * \ingroup spectral
     **/

    class Val_domain : public MemoryMappable
    {

      protected:
        const Domain* zone; ///< Pointer to the associated \c Domain
        Base_spectral base; ///< Spectral basis of the field.

        bool is_zero; ///< Indicator used for null fields (for speed issues).

        mutable Array<double>* c;  ///< Pointer on the \c Array of the values in the configuration space.
        mutable Array<double>* cf; ///< Pointer on the \c Array of the values in the coefficients space.
        mutable bool in_conf;      ///< Is the field known in the configuration space ?
        mutable bool in_coef;      ///< Is the field known in the coefficient space ?

        // All current production domains fit in four dimensions. Keep their
        // small derivative-pointer caches inline, while retaining a heap
        // fallback so Val_domain remains valid for higher-dimensional Domain
        // implementations.
        static constexpr int inline_derivative_dimensions = 4;
        mutable Val_domain* inline_derivatives[2 * inline_derivative_dimensions];
        mutable Val_domain**
            p_der_var; ///< Derivatives with respect to the numerical coordinates.
        mutable Val_domain**
            p_der_abs; ///< Derivatives with respect to absolute Cartesian coordinates.

      private:
        void allocate_derivative_storage();
        void release_derivative_storage() noexcept;
        void move_derivative_storage_from(Val_domain&) noexcept;
        void swap_derivative_storage(Val_domain&) noexcept;
        void del_deriv() const;       ///< Delete the derived quantities.
        void compute_der_var() const; ///< Computes the derivatives with respect to the numerical coordinates.
        void compute_der_abs() const; ///< Computes the derivatives with respect to the absolute Cartesian coordinates.
        void assign_vals(const Val_domain&); ///< assigns the content of another Val_domain to this one, even if they
                                             ///< are in different domains. Be careful with this.

        friend class DerAbsLaneBatch;

      public:
        /**
         * Constructor from a \c Domain.
         * Nothing is initialized otherwise.
         * @param so [input] : pointer on the \c Domain.
         */
        Val_domain(const Domain* so);
        /**
         * Copy constructor.
         * @param so [input] : \c Val_domain to be copied.
         * @param copy [input] : if \c true then all the \c so is copied otherwise the arrays are not initialized.
         */
        Val_domain(const Val_domain& so, bool copy = true);
        Val_domain(Val_domain&&) noexcept; ///< Move constructor.
        Val_domain(const Domain* dom, const Val_domain& so);
        Val_domain(const Domain* so, BinarySource&); ///< Constructor from a BinarySource (modern API).
        ~Val_domain();                       ///< Destructor

        void save(BinarySink&) const; ///< Save via BinarySink (modern API).

        /**
         * @returns a pointer on the \c Domain.
         */
        const Domain* get_domain() const { return zone; };
        void operator=(const Val_domain&);            ///< Assignement to another \c Val_domain
        void swap(Val_domain&) noexcept;              ///< Swaps the content with the source.
        Val_domain& operator=(Val_domain&&) noexcept; ///< Move assignment operator.
        void operator=(double);                       ///< Assignement to a \c double , in the configuration space.
        void annule_hard();      ///< Sets all the arrays to zero (the logical state is NOT set to zero).
        void annule_hard_coef(); ///< Sets all the arrays to zero in the coefficient space (the logical state is NOT set
                                 ///< to zero).
        /**
         * Returns the basis of decomposition.
         */
        const Base_spectral& get_base() const { return base; };
        /**
         * Sets the basis of decomposition.
         */
        Base_spectral& set_base() { return base; };

        /**
         * @returns the values in the configuration space.
         */
        Array<double> get_conf() const { return *c; };

        /**
         * @returns the values in the coefficient space.
         */
        Array<double> get_coef() const { return *cf; };
        /** Ensures coefficient storage exists and returns it without copying. */
        const Array<double>& get_coef_ref() const
        {
            coef();
            return *cf;
        }

        void set_zero(); ///< Sets the \c Val_domain to zero (logical state to zero and arrays destroyed).
        /**
         * Check whether the logical state is zero or not.
         */
        bool check_if_zero() const { return is_zero; };

      public:
        /**
         * Destroys the values in the coefficient space.
         */
        void set_in_conf();
        /**
         * Destroys the values in the configuration space.
         */
        void set_in_coef();
        /**
         * Allocates the values in the configuration space and destroys the values in the coefficients space.
         */
        void allocate_conf();
        /**
         * Allocates the values in the coefficient space and destroys the values in the configuration space.
         */
        void allocate_coef();
        void std_base();           ///< Sets the standard basis of decomposition
        void std_anti_base();      ///< Sets the standard, anti-symetric, basis of decomposition
        void std_base(int m);      ///< Sets the standard basis of decomposition
        void std_anti_base(int m); ///< Sets the standard, anti-symetric, basis of decomposition
        void std_base_r_spher();   ///< Sets the basis for the radial component of a vector in orthonormal spherical
                                   ///< coordinates.
        void std_base_t_spher(); ///< Sets the basis for the \f$theta\f$ component of a vector in orthonormal spherical
                                 ///< coordinates.
        void std_base_p_spher(); ///< Sets the basis for the \f$varphi\f$ component of a vector in orthonormal spherical
                                 ///< coordinates.
        void std_base_x_cart();  ///< Sets the basis for the X-component of a vector in Cartesian coordinates.
        void std_base_y_cart();  ///< Sets the basis for the Y-component of a vector in Cartesian coordinates.
        void std_base_z_cart();  ///< Sets the basis for the Z-component of a vector in Cartesian coordinates.
        void std_r_base();       ///< Sets the basis for the radius.

        void std_base_rt_spher(); ///< Sets the basis for the \f$(r,theta)\f$ component of a 2-tensor in orthonormal
                                  ///< spherical coordinates.
        void std_base_rp_spher(); ///< Sets the basis for the \f$(r,varphi)\f$ component of a 2-tensor in orthonormal
                                  ///< spherical coordinates.
        void std_base_tp_spher(); ///< Sets the basis for the \f$(theta, varphi)\f$ component of a 2-tensor in
                                  ///< orthonormal spherical coordinates.
        void std_base_xy_cart();  ///< Sets the basis for the XY component of a 2-tensor in Cartesian coordinates.
        void std_base_xz_cart();  ///< Sets the basis for the XZ component of a 2-tensor in Cartesian coordinates.
        void std_base_yz_cart();  ///< Sets the basis for the YZ component of a 2-tensor in Cartesian coordinates.

        void std_xodd_base();      ///< Sets the basis for an odd function in \f$X\f$ (Critic case).
        void std_todd_base();      ///< Sets the basis for an odd function in \f$T\f$ (Critic case).
        void std_xodd_todd_base(); ///< Sets the basis for an odd function in \f$X\f$ and \f$T\f$ (Critic case).

        void std_base_odd(); ///< Sets the basis in odd polynomials.

        void std_base_r_mtz(); ///< Sets the basis for the radial component of a vector in orthonormal coordinates in
                               ///< the MTZ context.
        void std_base_t_mtz(); ///< Sets the basis for the \f$theta\f$ component of a vector in orthonormal coordinates
                               ///< in the MTZ context.
        void std_base_p_mtz(); ///< Sets the basis for the \f$varphi\f$ component of a vector in orthonormal coordinates
                               ///< in the MTZ context.

        /**
         * Read/write the value of the field in the configuration space. The coefficients are destroyed.
         * @param pos [input] : point concerned.
         */
        double& set(const Index& pos);
        /**
         * Read/write the value of the field in the coefficient space. The values at collocation points are destroyed.
         * @param pos [input] : coefficient concerned.
         */
        double& set_coef(const Index& pos);
        /**
         * Read only value of the field in the coefficient space.
         * @param pos [input] : coefficient concerned.
         */
        double get_coef(const Index& pos) const;

        /**
         * Read only value of the field in the configuration space.
         * @param pos [input] : point concerned.
         */
        double operator()(const Index& pos) const;
        void coef() const;   ///< Computes the coefficients.
        void coef_i() const; ///< Computes the values in the configuration space.

        /**
         * Computes the derivative with respect to a numerical coordinate.
         * @param i [input] : the coordinate index.
         * @returns : the result, in the coefficient space.
         */
        Val_domain der_var(int i) const;

        /**
         * Populate numerical-coordinate derivative caches for compatible
         * fields. Inputs are borrowed and must belong to one Domain.
         */
        static bool derivative_lane_tiling_enabled();
        static void prepare_der_var_batch(std::span<const Val_domain* const> fields);
        /**
         * Computes the derivative with respect to an absolute coordinate (typically Cartesian).
         * @param i [input] : the coordinate index.
         * @returns : the result, in the coefficient space.
         */
        Val_domain der_abs(int i) const;

        /**
         * Populate Cartesian derivative caches for compatible fields through
         * the Domain lane seam. Cache ownership remains with each field.
         */
        static void prepare_der_abs_batch(std::span<const Val_domain* const> fields);
        /**
         * Computes the derivative with respect to the spherical coordinates (if defined).
         * @param i [input] : the coordinate index.
         * @returns : the result, in the coefficient space.
         */
        Val_domain der_spher(int i) const;
        Val_domain der_r() const;          ///< @returns the radial derivative
        Val_domain der_t() const;          ///< @returns the derivative wrt \f$theta\f$.
        Val_domain der_p() const;          ///< @returns the derivative wrt \f$varphi\f$.
        Val_domain mult_sin_theta() const; ///< Multiplication by \f$ \sin theta \f$
        Val_domain mult_cos_theta() const; ///< Multiplication by \f$ \cos theta \f$
        Val_domain mult_sin_phi() const;   ///< Multiplication by \f$ \sin varphi \f$
        Val_domain mult_cos_phi() const;   ///< Multiplication by \f$ \cos varphi \f$
        Val_domain div_sin_theta() const;  ///< Division by \f$ \sin theta \f$
        Val_domain div_cos_theta() const;  ///< Division by \f$ \cos theta \f$
        Val_domain div_x() const;          ///< Division by \f$ x \f$
        Val_domain div_xm1() const;        ///< Division by \f$ x-1 \f$
        Val_domain div_xp1() const;        ///< Division by \f$ x-1 \f$
        Val_domain mult_xm1() const;       ///< Multiplication by \f$ x-1 \f$
        Val_domain div_1mx2() const;       ///< Division by \f$ 1-x^2 \f$
        Val_domain div_1mrsL() const;      ///< Division by \f$ 1-x^2 \f$
        Val_domain div_sin_chi() const;    ///< Division by \f$ \sin \chi \f$
        Val_domain div_chi() const;        ///< Division by \f$ \chi^* \f$
        Val_domain div_r() const;          ///< Division by the radius.
        Val_domain mult_r() const;         ///< Multiplication by the radius.
        Val_domain der_r_rtwo()
            const; ///< @returns the radial derivative multiplied by \f$r^2\f$ (defined in a compactified domain).
        Val_domain mult_cos_time() const; ///< @returns the multiplication by \f$\cos omega t\f$.
        Val_domain mult_sin_time() const; ///< @returns the multiplication by \f$\sin omega t\f$.

        double integrale() const;    ///< @returns integral in the whole domain.
        double integ_volume() const; ///< @returns integral in the whole domain.

        void operator+=(const Val_domain&); ///< Operator +=
        void operator-=(const Val_domain&); ///< Operator -=
        void operator*=(const Val_domain&); ///< Operator *=
        void operator/=(const Val_domain&); ///< Operator /=
        void operator+=(double);            ///< Operator +=
        void operator-=(double);            ///< Operator -=
        void operator*=(double);            ///< Operator *=
        void operator/=(double);            ///< Operator /=

        friend class Space;
        friend class Domain;
        friend class Space_spheric;
        friend class Scalar;
        friend class Domain_nucleus;
        friend class Domain_shell;
        friend class Domain_shell_log;
        friend class Domain_shell_surr;
        friend class Domain_compact;
        friend class Domain_bispheric_rect;
        friend class Domain_bispheric_chi_first;
        friend class Domain_bispheric_eta_first;
        friend class Domain_bispheric_rect_nosym;
        friend class Domain_bispheric_chi_first_nosym;
        friend class Domain_bispheric_eta_first_nosym;
        friend class Domain_critic_inner;
        friend class Domain_critic_outer;
        friend class Domain_polar_nucleus;
        friend class Domain_polar_shell;
        friend class Domain_polar_shell_inner_adapted;
        friend class Domain_polar_shell_inner_homothetic;
        friend class Domain_polar_shell_outer_adapted;
        friend class Domain_polar_shell_outer_homothetic;
        friend class Domain_polar_shell_bilateral_adapted;
        friend class Domain_polar_compact;
        friend class Domain_oned_ori;
        friend class Domain_oned_qcq;
        friend class Domain_oned_inf;
        friend class Domain_spheric_periodic_nucleus;
        friend class Domain_spheric_periodic_shell;
        friend class Domain_spheric_periodic_compact;
        friend class Domain_spheric_time_nucleus;
        friend class Domain_spheric_time_shell;
        friend class Domain_spheric_time_compact;
        friend class Domain_shell_outer_adapted;
        friend class Domain_shell_inner_adapted;
        friend class Domain_shell_inner_homothetic;
        friend class Domain_shell_outer_homothetic;
        friend class Domain_nucleus_symphi;
        friend class Domain_shell_symphi;
        friend class Domain_compact_symphi;
        friend class Domain_nucleus_nosym;
        friend class Domain_shell_nosym;
        friend class Domain_compact_nosym;
        friend class Domain_shell_inner_adapted_nosym;
        friend class Domain_shell_outer_adapted_nosym;
        friend class Domain_shell_inner_homothetic_nosym;
        friend class Domain_shell_outer_homothetic_nosym;
        friend class Domain_polar_periodic_nucleus;
        friend class Domain_polar_periodic_shell;
        friend class Domain_fourD_periodic_nucleus;
        friend class Domain_fourD_periodic_shell;

        friend class Eq_matching_non_std;
        friend ostream& operator<<(ostream&, const Val_domain&);
        friend Val_domain sin(const Val_domain&);
        friend Val_domain cos(const Val_domain&);
        friend Val_domain sinh(const Val_domain&);
        friend Val_domain cosh(const Val_domain&);
        friend Val_domain atan(const Val_domain&);
        friend Val_domain atanh(const Val_domain&);
        friend Val_domain operator+(const Val_domain&);
        friend Val_domain operator-(const Val_domain&);
        friend Val_domain operator+(const Val_domain&, const Val_domain&);
        friend Val_domain operator+(Val_domain&&, const Val_domain&);
        friend Val_domain operator+(const Val_domain&, double);
        friend Val_domain operator+(double, const Val_domain&);
        friend Val_domain operator-(const Val_domain&, const Val_domain&);
        friend Val_domain operator-(Val_domain&&, const Val_domain&);
        friend Val_domain operator-(const Val_domain&, double);
        friend Val_domain operator-(double, const Val_domain&);
        friend Val_domain operator*(const Val_domain&, const Val_domain&);
        friend Val_domain operator*(Val_domain&&, const Val_domain&);
        friend Val_domain operator*(const Val_domain&, double);
        friend Val_domain operator*(double, const Val_domain&);
        friend Val_domain operator*(const Val_domain&, int);
        friend Val_domain operator*(int, const Val_domain&);
        friend Val_domain operator*(const Val_domain&, long int);
        friend Val_domain operator*(long int, const Val_domain&);
        friend Val_domain operator/(const Val_domain&, const Val_domain&);
        friend Val_domain operator/(const Val_domain&, double);
        friend Val_domain operator/(double, const Val_domain&);
        friend Val_domain pow(const Val_domain&, int);
        friend Val_domain pow(const Val_domain&, double);
        friend Val_domain sqrt(const Val_domain&);
        friend Val_domain exp(const Val_domain&);
        friend Val_domain log(const Val_domain&);
        friend Val_domain atanh(const Val_domain&);
        friend Val_domain atan(const Val_domain&);
        friend Val_domain bessel_jl(const Val_domain&, int);
        friend Val_domain bessel_yl(const Val_domain&, int);
        friend double diffmax(const Val_domain&, const Val_domain&);
        friend double maxval(const Val_domain&);
    };
} // namespace Kadath
