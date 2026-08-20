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

/**
 * @file val_domain.cpp
 * @brief Implementation of Val_domain class for field values in spectral domains
 * @author Philippe Grandclement
 * @date 2017
 *
 * This file implements Val_domain, the core class representing scalar field values
 * within a single computational domain in Kadath's spectral method framework.
 *
 * Val_domain manages:
 * - Dual representation: physical space (collocation points) and spectral space (coefficients)
 * - Lazy evaluation: transforms between representations only when needed
 * - Derivative caching: stores computed derivatives to avoid recomputation
 * - Basis management: tracks spectral basis types for proper transformations
 * - Memory efficiency: automatic cleanup and zero-value optimization
 *
 * Key design patterns:
 * - Lazy transformation: Only convert between spaces when explicitly requested
 * - Copy-on-write semantics: Can create shallow copies for efficiency
 * - Cached derivatives: Computed once and stored until invalidated
 * - Zero optimization: Special handling for zero fields (no storage needed)
 *
 * Representation states:
 * - is_zero: Field is identically zero (no arrays allocated)
 * - in_conf: Physical space values available at collocation points
 * - in_coef: Spectral coefficients available
 * - Both can be true (redundant storage for efficiency)
 *
 * Typical workflow:
 * 1. Create Val_domain associated with a Domain
 * 2. Set basis type (Chebyshev, Legendre, etc.)
 * 3. Populate with data (physical or spectral)
 * 4. Transform between representations as needed
 * 5. Compute derivatives (cached automatically)
 * 6. Perform operations (addition, multiplication, etc.)
 *
 * Memory management:
 * - Automatic deallocation of arrays when no longer needed
 * - Derivative cache invalidation on field modification
 * - Efficient handling of zero-valued fields
 */

#include "val_domain.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include <sstream>
#include "For_Kadath/Utilities/utilities.hpp"

namespace
{

    // ============================================================================
    // Helper Template Functions (Anonymous Namespace)
    // ============================================================================

    /**
     * @brief Initialize pointer array to null
     * @tparam T Pointer type
     * @param ptrs Array of pointers to initialize
     * @param count Number of pointers in array
     * @post All pointers set to nullptr (nullptr)
     */
    template <typename T> void reset_ptr_array(T** ptrs, int count)
    {
        for (int i = 0; i < count; ++i)
            ptrs[i] = nullptr;
    }

    /**
     * @brief Safely delete pointer and set to null
     * @tparam T Pointer type
     * @param ptr Reference to pointer to delete
     * @post Pointer is deleted and set to nullptr
     * @note Safe to call with nullptr (no-op)
     */
    template <typename T> void delete_and_clear(T*& ptr)
    {
        if (ptr != nullptr) {
            delete ptr;
            ptr = nullptr;
        }
    }

    /**
     * @brief Apply basis-specific method based on type (2-argument version)
     * @tparam Zone Domain type
     * @tparam Base Base_spectral type
     * @param typeb Basis type identifier (CHEB_TYPE or LEG_TYPE)
     * @param zone Domain pointer
     * @param base Base_spectral reference
     * @param cheb Member function pointer for Chebyshev basis
     * @param leg Member function pointer for Legendre basis
     * @param error Error message if type not recognized
     *
     * Template pattern for dispatching to appropriate basis-specific method.
     * Centralizes the switch logic for basis type selection.
     */
    template <typename Zone, typename Base>
    void apply_base_type(int typeb, Zone* zone, Base& base, void (Zone::*cheb)(Base&), void (Zone::*leg)(Base&),
                         const char* error)
    {
        switch (typeb) {
            case CHEB_TYPE:
                (zone->*cheb)(base);
                break;
            case LEG_TYPE:
                (zone->*leg)(base);
                break;
            default:
                std::ostringstream oss;
                oss << error;
                KADATH_THROW(oss.str());
        }
    }

    /**
     * @brief Apply basis-specific method based on type (3-argument version with m)
     * @tparam Zone Domain type
     * @tparam Base Base_spectral type
     * @param typeb Basis type identifier
     * @param zone Domain pointer
     * @param base Base_spectral reference
     * @param m Additional integer parameter (typically azimuthal mode number)
     * @param cheb Member function pointer for Chebyshev basis
     * @param leg Member function pointer for Legendre basis
     * @param error Error message if type not recognized
     *
     * Extended version for basis setup requiring mode number m.
     */
    template <typename Zone, typename Base>
    void apply_base_type(int typeb, Zone* zone, Base& base, int m, void (Zone::*cheb)(Base&, int),
                         void (Zone::*leg)(Base&, int), const char* error)
    {
        switch (typeb) {
            case CHEB_TYPE:
                (zone->*cheb)(base, m);
                break;
            case LEG_TYPE:
                (zone->*leg)(base, m);
                break;
            default:
                std::ostringstream oss;
                oss << error;
                KADATH_THROW(oss.str());
        }
    }

    /**
     * @brief Apply Chebyshev-only method
     * @tparam Zone Domain type
     * @tparam Base Base_spectral type
     * @param typeb Basis type identifier (must be CHEB_TYPE)
     * @param zone Domain pointer
     * @param base Base_spectral reference
     * @param cheb Member function pointer for Chebyshev basis
     * @param error Error message if type not CHEB_TYPE
     *
     * For operations only defined for Chebyshev basis.
     */
    template <typename Zone, typename Base>
    void apply_base_cheb(int typeb, Zone* zone, Base& base, void (Zone::*cheb)(Base&), const char* error)
    {
        switch (typeb) {
            case CHEB_TYPE:
                (zone->*cheb)(base);
                break;
            default:
                std::ostringstream oss;
                oss << error;
                KADATH_THROW(oss.str());
        }
    }

} // anonymous namespace

namespace Kadath
{

    // ============================================================================
    // Constructors
    // ============================================================================

    /**
     * @brief Construct Val_domain associated with a Domain
     * @param dom Pointer to the computational domain
     * @post Field initialized as zero (is_zero=true)
     * @post No arrays allocated (deferred until needed)
     * @post Derivative cache initialized to null
     *
     * Creates an empty Val_domain ready to store field values. The field starts
     * in zero state for memory efficiency - arrays are allocated only when data
     * is assigned.
     */
    Val_domain::Val_domain(const Domain* dom)
        : zone(dom), base(zone->get_ndim()), is_zero(false), c(nullptr), cf(nullptr), in_conf(false), in_coef(false)
    {

        allocate_derivative_storage();
    }

    /**
     * @brief Copy constructor with optional deep copy
     * @param so Source Val_domain to copy from
     * @param copie If true, deep copy arrays; if false, copy structure only
     * @post New object shares same domain and basis as source
     * @post If copie=true, all arrays and derivatives are deeply copied
     * @post If copie=false, array pointers null, flags reset
     *
     * Flexible copy constructor supporting both shallow (metadata only) and
     * deep (full data) copies. Shallow copies useful for creating result
     * containers before computation.
     *
     * Example:
     * @code
     * Val_domain original(dom);
     * // ... populate original ...
     *
     * Val_domain deep(original, true);    // Full copy
     * Val_domain shallow(original, false); // Metadata only
     * @endcode
     */
    Val_domain::Val_domain(const Val_domain& so, bool copie)
        : zone(so.zone), base(so.base), is_zero(so.is_zero), in_conf(so.in_conf), in_coef(so.in_coef)
    {

        c = ((so.c != nullptr) && copie) ? new Array<double>(*so.c) : nullptr;
        cf = ((so.cf != nullptr) && copie) ? new Array<double>(*so.cf) : nullptr;

        if (!copie) {
            in_conf = false;
            in_coef = false;
        }

        allocate_derivative_storage();
        for (int i = 0; i < zone->get_ndim(); i++) {
            p_der_var[i] = ((so.p_der_var[i] != nullptr) && copie) ? new Val_domain(*so.p_der_var[i]) : nullptr;
            p_der_abs[i] = ((so.p_der_abs[i] != nullptr) && copie) ? new Val_domain(*so.p_der_abs[i]) : nullptr;
        }
    }

    // ============================================================================
    // Destructor
    // ============================================================================

    /**
     * @brief Destructor - releases all allocated memory
     * @post All arrays and derivative cache deallocated
     * @post Inline or heap-backed derivative storage released appropriately
     *
     * Cleans up all dynamically allocated resources including data arrays
     * and cached derivatives.
     */
    Val_domain::~Val_domain()
    {
        del_deriv();
        release_derivative_storage();

        delete_and_clear(c);
        delete_and_clear(cf);
    }

    // ============================================================================
    // I/O Operations
    // ============================================================================


    /**
     * @brief Delete cached derivatives
     * @post All derivative cache entries deallocated and set to null
     *
     * Invalidates and frees cached derivative data. Called when field is
     * modified to ensure derivatives are recomputed with correct values.
     */
    void Val_domain::del_deriv() const
    {
        for (int i = 0; i < zone->get_ndim(); i++) {
            if (p_der_var[i] != nullptr) {
                delete p_der_var[i];
                p_der_var[i] = nullptr;
            }
            if (p_der_abs[i] != nullptr) {
                delete p_der_abs[i];
                p_der_abs[i] = nullptr;
            }
        }
    }

    // ============================================================================
    // Assignment Operators
    // ============================================================================

    /**
     * @brief Assignment operator - copy from another Val_domain
     * @param so Source Val_domain
     * @pre Both Val_domains must be associated with same Domain
     * @post All data and metadata copied from source
     * @post Derivative cache copied
     */
    void Val_domain::operator=(const Val_domain& so)
    {
        assert(zone == so.zone);
        this->assign_vals(so);
    }

    /**
     * @brief Assign values and metadata from another Val_domain
     * @param so Source Val_domain
     * @post Data arrays deeply copied
     * @post Basis information copied
     * @post Old derivative cache deleted, new cache copied
     *
     * Internal assignment implementation. Handles all aspects of copying
     * including data, basis, and derivatives.
     */
    void Val_domain::assign_vals(const Val_domain& so)
    {
        is_zero = so.is_zero;
        in_conf = so.in_conf;
        in_coef = so.in_coef;

        delete_and_clear(c);
        c = (in_conf) ? new Array<double>(*so.c) : nullptr;

        delete_and_clear(cf);
        cf = (in_coef) ? new Array<double>(*so.cf) : nullptr;

        if (so.base.is_def())
            base = so.base;
        else
            base.set_non_def();

        del_deriv();
        for (int i = 0; i < zone->get_ndim(); i++) {
            p_der_var[i] = (so.p_der_var[i] == nullptr) ? nullptr : new Val_domain(*so.p_der_var[i]);
            p_der_abs[i] = (so.p_der_abs[i] == nullptr) ? nullptr : new Val_domain(*so.p_der_abs[i]);
        }
    }

    /**
     * @brief Assign constant value to entire field
     * @param xx Value to assign
     * @post If xx==0, field set to zero state (optimized)
     * @post If xx!=0, physical space allocated and filled with xx
     * @post Derivative cache invalidated
     *
     * Special handling for zero assignment (no array allocated).
     * Non-zero values allocate physical space and fill uniformly.
     *
     * Example:
     * @code
     * Val_domain field(dom);
     * field = 0.0;    // Optimized zero state
     * field = 1.5;    // All collocation points set to 1.5
     * @endcode
     */
    void Val_domain::operator=(double xx)
    {
        if (xx == 0)
            set_zero();
        else {
            is_zero = false;
            allocate_conf();
            del_deriv();
            *c = xx;
        }
    }

    /**
     * @brief Set all values to zero (hard zero, allocates array)
     * @post Physical space allocated and filled with zeros
     * @post is_zero remains false (array explicitly allocated)
     * @post Derivative cache invalidated
     *
     * Unlike set_zero(), this allocates the array and fills it with zeros.
     * Used when you need an actual zero array rather than optimized empty state.
     */
    void Val_domain::annule_hard()
    {
        allocate_conf();
        del_deriv();
        *c = 0.;
    }

    // ============================================================================
    // Element Access
    // ============================================================================

    /**
     * @brief Get reference to value at collocation point (read/write)
     * @param index Multi-dimensional index of collocation point
     * @return Reference to value at specified point
     * @post Ensures field is in physical space representation
     *
     * Provides write access to collocation point values. Automatically
     * ensures physical space is active.
     */
    double& Val_domain::set(const Index& index)
    {
        set_in_conf();
        return c->set(index);
    }

    /**
     * @brief Get reference to spectral coefficient (read/write)
     * @param index Multi-dimensional index of coefficient
     * @return Reference to coefficient at specified index
     * @post Ensures field is in spectral space representation
     *
     * Provides write access to spectral coefficients. Automatically
     * ensures spectral space is active.
     */
    double& Val_domain::set_coef(const Index& index)
    {
        set_in_coef();
        return cf->set(index);
    }

    /**
     * @brief Get spectral coefficient value (read-only)
     * @param index Multi-dimensional index of coefficient
     * @return Coefficient value (0 if field is zero)
     * @pre Field must be in spectral space or be zero
     *
     * Read-only access to spectral coefficients.
     */
    double Val_domain::get_coef(const Index& index) const
    {
        if (is_zero)
            return 0.;
        else {
            assert(in_coef);
            return cf->set(index);
        }
    }

    /**
     * @brief Get value at collocation point (read-only)
     * @param index Multi-dimensional index of collocation point
     * @return Value at specified point
     * @post Ensures physical space representation exists (lazy transform)
     *
     * Read-only access with automatic transformation to physical space if needed.
     */
    double Val_domain::operator()(const Index& index) const
    {
        coef_i();
        return (*c)(index);
    }

    // ============================================================================
    // State Management
    // ============================================================================

    /**
     * @brief Set field to physical space representation
     * @post in_conf=true, in_coef=false
     * @post Spectral array deleted if it existed
     *
     * Switches representation to physical space only. Any existing spectral
     * data is discarded.
     */
    void Val_domain::set_in_conf()
    {
        if (cf != nullptr) {
            delete_and_clear(cf);
        }
        in_conf = true;
        in_coef = false;
    }

    /**
     * @brief Set field to spectral space representation
     * @post in_conf=false, in_coef=true
     * @post Physical array deleted if it existed
     *
     * Switches representation to spectral space only. Any existing physical
     * data is discarded.
     */
    void Val_domain::set_in_coef()
    {
        if (c != nullptr) {
            delete_and_clear(c);
        }
        in_conf = false;
        in_coef = true;
    }

    /**
     * @brief Allocate physical space array
     * @post Physical array allocated with domain's collocation point dimensions
     * @post in_conf=true, is_zero=false
     * @post Any existing physical array replaced
     *
     * Prepares field for storing values at collocation points.
     */
    void Val_domain::allocate_conf()
    {
        set_in_conf();
        is_zero = false;
        delete_and_clear(c);
        c = new Array<double>(zone->get_nbr_points());
    }

    /**
     * @brief Allocate spectral space array
     * @post Spectral array allocated with domain's coefficient dimensions
     * @post in_coef=true, is_zero=false
     * @post Any existing spectral array replaced
     *
     * Prepares field for storing spectral coefficients.
     */
    void Val_domain::allocate_coef()
    {
        set_in_coef();
        is_zero = false;
        delete_and_clear(cf);
        cf = new Array<double>(zone->get_nbr_coefs());
    }

    /**
     * @brief Set field to optimized zero state
     * @post is_zero=true
     * @post All arrays deallocated
     * @post Derivative cache cleared
     * @post Basis set to undefined
     *
     * Puts field in special zero state where no arrays are allocated.
     * This is more memory-efficient than storing an array of zeros.
     *
     * @note Operations with zero fields are optimized (no transformation needed)
     */
    void Val_domain::set_zero()
    {
        if (!is_zero) {
            del_deriv();

            if (in_conf) {
                delete_and_clear(c);
                in_conf = false;
            }

            if (in_coef) {
                delete_and_clear(cf);
                in_coef = false;
            }

            base.set_non_def();
        }
        is_zero = true;
    }

    // ============================================================================
    // Basis Setup Methods
    // ============================================================================

    /**
     * @brief Set standard spectral basis for the domain
     * @post Basis configured according to domain type (Chebyshev or Legendre)
     *
     * Configures the spectral basis using domain-specific defaults.
     */
    void Val_domain::std_base()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base, &Domain::set_legendre_base,
                        "Unknown type of base in Val_domain::std_base");
    }

    /**
     * @brief Set standard radial basis
     * @post Radial basis configured for the domain
     */
    void Val_domain::std_r_base()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_r_base, &Domain::set_legendre_r_base,
                        "Unknown type of base in Val_domain::std_r_base");
    }

    /**
     * @brief Set antisymmetric (odd-parity) basis
     * @post Basis configured for odd-parity functions
     *
     * For functions that satisfy f(-x) = -f(x) in appropriate coordinates.
     */
    void Val_domain::std_anti_base()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_anti_cheb_base, &Domain::set_anti_legendre_base,
                        "Unknown type of base in Val_domain::std_anti_base");
    }

    /**
     * @brief Set standard basis with azimuthal mode number
     * @param m Azimuthal mode number
     * @post Basis configured for specific m-mode
     *
     * Used in axisymmetric or azimuthally-decomposed problems where different
     * azimuthal modes (m) may require different basis setups.
     */
    void Val_domain::std_base(int m)
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, m, &Domain::set_cheb_base_with_m, &Domain::set_legendre_base_with_m,
                        "Unknown type of base in Val_domain::std_base");
    }

    /**
     * @brief Set antisymmetric basis with azimuthal mode number
     * @param m Azimuthal mode number
     * @post Antisymmetric basis configured for specific m-mode
     */
    void Val_domain::std_anti_base(int m)
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, m, &Domain::set_anti_cheb_base_with_m,
                        &Domain::set_anti_legendre_base_with_m, "Unknown type of base in Val_domain::std_anti_base");
    }

    // Coordinate-specific basis setup methods
    // (Spherical coordinates: r, θ, φ)

    /**
     * @brief Set basis for r-θ components in spherical coordinates
     */
    void Val_domain::std_base_rt_spher()
    {
        int typeb(zone->get_type_base());
        apply_base_cheb(typeb, zone, base, &Domain::set_cheb_base_rt_spher,
                        "Unknown type of base in Val_domain::std_base_rt_spher");
    }

    /**
     * @brief Set basis for r-φ components in spherical coordinates
     */
    void Val_domain::std_base_rp_spher()
    {
        int typeb(zone->get_type_base());
        apply_base_cheb(typeb, zone, base, &Domain::set_cheb_base_rp_spher,
                        "Unknown type of base in Val_domain::std_base_rp_spher");
    }

    /**
     * @brief Set basis for θ-φ components in spherical coordinates
     */
    void Val_domain::std_base_tp_spher()
    {
        int typeb(zone->get_type_base());
        apply_base_cheb(typeb, zone, base, &Domain::set_cheb_base_tp_spher,
                        "Unknown type of base in Val_domain::std_base_tp_spher");
    }

    // Cartesian coordinate-specific methods

    /**
     * @brief Set basis for x-y plane components in Cartesian coordinates
     */
    void Val_domain::std_base_xy_cart()
    {
        int typeb(zone->get_type_base());
        apply_base_cheb(typeb, zone, base, &Domain::set_cheb_base_xy_cart,
                        "Unknown type of base in Val_domain::std_base_xy_cart");
    }

    /**
     * @brief Set basis for x-z plane components in Cartesian coordinates
     */
    void Val_domain::std_base_xz_cart()
    {
        int typeb(zone->get_type_base());
        apply_base_cheb(typeb, zone, base, &Domain::set_cheb_base_xz_cart,
                        "Unknown type of base in Val_domain::std_base_xz_cart");
    }

    /**
     * @brief Set basis for y-z plane components in Cartesian coordinates
     */
    void Val_domain::std_base_yz_cart()
    {
        int typeb(zone->get_type_base());
        apply_base_cheb(typeb, zone, base, &Domain::set_cheb_base_yz_cart,
                        "Unknown type of base in Val_domain::std_base_yz_cart");
    }

    // Individual coordinate basis setup

    /**
     * @brief Set basis for radial direction only (spherical)
     */
    void Val_domain::std_base_r_spher()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base_r_spher, &Domain::set_legendre_base_r_spher,
                        "Unknown type of base in Val_domain::std_base_r_spher");
    }

    /**
     * @brief Set basis for polar angle θ direction (spherical)
     */
    void Val_domain::std_base_t_spher()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base_t_spher, &Domain::set_legendre_base_t_spher,
                        "Unknown type of base in Val_domain::std_base_t_spher");
    }

    /**
     * @brief Set basis for azimuthal angle φ direction (spherical)
     */
    void Val_domain::std_base_p_spher()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base_p_spher, &Domain::set_legendre_base_p_spher,
                        "Unknown type of base in Val_domain::std_base_p_spher");
    }

    /**
     * @brief Set basis for x direction (Cartesian)
     */
    void Val_domain::std_base_x_cart()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base_x_cart, &Domain::set_legendre_base_x_cart,
                        "Unknown type of base in Val_domain::std_base_x_cart");
    }

    /**
     * @brief Set basis for y direction (Cartesian)
     */
    void Val_domain::std_base_y_cart()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base_y_cart, &Domain::set_legendre_base_y_cart,
                        "Unknown type of base in Val_domain::std_base_y_cart");
    }

    /**
     * @brief Set basis for z direction (Cartesian)
     */
    void Val_domain::std_base_z_cart()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base_z_cart, &Domain::set_legendre_base_z_cart,
                        "Unknown type of base in Val_domain::std_base_z_cart");
    }

    // Modified isotropic coordinates (MTZ)

    /**
     * @brief Set basis for radial direction in MTZ coordinates
     */
    void Val_domain::std_base_r_mtz()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base_r_mtz, &Domain::set_legendre_base_r_mtz,
                        "Unknown type of base in Val_domain::std_base_r_mtz");
    }

    /**
     * @brief Set basis for θ direction in MTZ coordinates
     */
    void Val_domain::std_base_t_mtz()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base_t_mtz, &Domain::set_legendre_base_t_mtz,
                        "Unknown type of base in Val_domain::std_base_t_mtz");
    }

    /**
     * @brief Set basis for φ direction in MTZ coordinates
     */
    void Val_domain::std_base_p_mtz()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base_p_mtz, &Domain::set_legendre_base_p_mtz,
                        "Unknown type of base in Val_domain::std_base_p_mtz");
    }

    // Parity-specific bases

    /**
     * @brief Set basis with odd parity in x direction
     * @post Basis configured for functions with f(-x,y,z) = -f(x,y,z)
     */
    void Val_domain::std_xodd_base()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_xodd_base, &Domain::set_legendre_xodd_base,
                        "Unknown type of base in Val_domain::std_xodd_base");
    }

    /**
     * @brief Set basis with odd parity in θ direction
     * @post Basis configured for functions with odd parity in polar angle
     */
    void Val_domain::std_todd_base()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_todd_base, &Domain::set_legendre_todd_base,
                        "Unknown type of base in Val_domain::std_todd_base");
    }

    /**
     * @brief Set basis with odd parity in both x and θ directions
     * @post Basis configured for double-odd parity
     */
    void Val_domain::std_xodd_todd_base()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_xodd_todd_base, &Domain::set_legendre_xodd_todd_base,
                        "Unknown type of base in Val_domain::std_xodd_todd_base");
    }

    /**
     * @brief Set fully odd-parity basis (all dimensions)
     * @post Basis configured for globally odd functions
     */
    void Val_domain::std_base_odd()
    {
        int typeb = zone->get_type_base();
        apply_base_type(typeb, zone, base, &Domain::set_cheb_base_odd, &Domain::set_legendre_base_odd,
                        "Unknown type of base in Val_domain::std_base_odd");
    }

    // ============================================================================
    // Spectral Transformations
    // ============================================================================

    /**
     * @brief Transform from physical to spectral space (lazy evaluation)
     * @post Spectral coefficients available (in_coef=true)
     * @post Physical values preserved if already present
     *
     * Converts collocation point values to spectral coefficients using
     * appropriate basis. If already in spectral space or field is zero,
     * this is a no-op (lazy evaluation).
     *
     * @pre Basis must be defined
     * @pre Field must be in physical space or zero
     *
     * Example:
     * @code
     * Val_domain field(dom);
     * field.std_base();
     * field.allocate_conf();
     * // ... fill physical values ...
     * field.coef();  // Transform to spectral space
     * @endcode
     */
    void Val_domain::coef() const
    {
        if ((in_coef) || (is_zero)) {
            return; // Already in spectral space or zero - nothing to do
        } else {
            if (base.is_def() == false) {
                KADATH_THROW("Base not defined in Val_domain::coef");
            } else {
                assert(in_conf);
                if (cf != nullptr)
                    delete cf;
                cf = new Array<double>(base.coef(zone->get_nbr_coefs(), *c));
            }
            in_coef = true;
        }
    }

    /**
     * @brief Transform from spectral to physical space (lazy evaluation)
     * @post Physical space values available (in_conf=true)
     * @post Spectral coefficients preserved if already present
     *
     * Reconstructs collocation point values from spectral coefficients.
     * If already in physical space or field is zero, this is a no-op
     * (lazy evaluation).
     *
     * @pre Basis must be defined
     * @pre Field must be in spectral space or zero
     *
     * Example:
     * @code
     * Val_domain field(dom);
     * field.std_base();
     * field.allocate_coef();
     * // ... fill spectral coefficients ...
     * field.coef_i();  // Transform to physical space
     * @endcode
     */
    void Val_domain::coef_i() const
    {
        if ((in_conf) || (is_zero)) {
            return; // Already in physical space or zero - nothing to do
        } else {
            if (base.is_def() == false) {
                KADATH_THROW("Base not defined in Val_domain::coef_i");
            } else {
                assert(in_coef);
                if (c != nullptr)
                    delete c;
                c = new Array<double>(base.coef_i(zone->get_nbr_points(), *cf));
            }
            in_conf = true;
        }
    }

    // ============================================================================
    // Stream Output
    // ============================================================================

    /**
     * @brief Stream output operator for Val_domain
     * @param o Output stream
     * @param so Val_domain to output
     * @return Reference to output stream
     *
     * Prints field information including zero status and available representations.
     * Shows both physical and spectral arrays if present.
     */
    ostream& operator<<(ostream& o, const Val_domain& so)
    {
        if (so.is_zero)
            o << "Null Val_domain" << endl;

        if (so.in_conf) {
            o << "Configuration space : " << endl;
            o << *so.c << endl;
        }

        if (so.in_coef) {
            o << "Coefficient space : " << endl;
            o << *so.cf << endl;
        }
        return o;
    }

    // ============================================================================
    // Derivative Operations
    // ============================================================================

    /**
     * @brief Get derivative with respect to coordinate variable (cached)
     * @param var Variable index (1-based: 1, 2, 3 for r, θ, φ or x, y, z)
     * @return Derivative field ∂f/∂xᵥₐᵣ
     * @pre 1 <= var <= ndim
     * @post Derivative computed and cached on first call
     *
     * Computes derivative in coordinate basis (covariant derivative for
     * curved coordinates). Result is cached for efficiency.
     *
     * @warning Uses 1-BASED INDEXING for consistency with Point class
     *
     * Example:
     * @code
     * Val_domain field(dom);
     * // ... populate field ...
     * Val_domain df_dr = field.der_var(1);  // ∂f/∂r
     * Val_domain df_dtheta = field.der_var(2);  // ∂f/∂θ
     * @endcode
     */
    Val_domain Val_domain::der_var(int var) const
    {
        assert((var > 0) && (var <= zone->get_ndim()));
        if (is_zero)
            return *this;
        else {
            if (p_der_var[var - 1] == nullptr)
                compute_der_var();
            return *p_der_var[var - 1];
        }
    }

    /**
     * @brief Get derivative with respect to absolute coordinate (cached)
     * @param var Variable index (0-based: 0, 1, 2)
     * @return Derivative in absolute coordinate system
     * @pre 0 <= var <= ndim
     * @post Derivative computed and cached on first call
     *
     * Computes derivative in absolute (often Cartesian) coordinates.
     * For var=0, returns zero field.
     *
     * Example (spherical → Cartesian):
     * @code
     * Val_domain field(dom);
     * Val_domain df_dx = field.der_abs(1);  // ∂f/∂x
     * Val_domain df_dy = field.der_abs(2);  // ∂f/∂y
     * Val_domain df_dz = field.der_abs(3);  // ∂f/∂z
     * @endcode
     */
    Val_domain Val_domain::der_abs(int var) const
    {
        assert((var >= 0) && (var <= zone->get_ndim()));

        if (var == 0) {
            Val_domain zero(get_domain());
            zero = 0.0;
            return zero;
        }

        if (is_zero)
            return *this;
        else {
            if (p_der_abs[var - 1] == nullptr)
                compute_der_abs();
            return *p_der_abs[var - 1];
        }
    }

    /**
     * @brief Get derivative in spherical coordinates
     * @param var Coordinate index (1=r, 2=θ, 3=φ)
     * @return Derivative ∂f/∂xᵥₐᵣ in spherical system
     * @pre 0 < var <= ndim
     *
     * Convenience wrapper for spherical coordinate derivatives.
     */
    Val_domain Val_domain::der_spher(int var) const
    {
        assert((var > 0) and (var <= zone->get_ndim()));

        if (is_zero)
            return *this;
        else {
            Val_domain zero(zone);
            zero = 0.0;

            if (var == 0)
                return zero;
            else if (var == 1)
                return this->der_r();
            else if (var == 2)
                return this->der_t();
            else if (var == 3)
                return this->der_p();
            else {
                KADATH_THROW("bad index in der_spher");
            }
        }
    }

    /**
     * @brief Radial derivative ∂f/∂r
     * @return Radial derivative field
     *
     * Domain-specific radial derivative computation.
     */
    Val_domain Val_domain::der_r() const
    {
        return zone->der_r(*this);
    }

    /**
     * @brief Polar angle derivative ∂f/∂θ
     * @return Polar derivative field
     */
    Val_domain Val_domain::der_t() const
    {
        return zone->der_t(*this);
    }

    /**
     * @brief Azimuthal angle derivative ∂f/∂φ
     * @return Azimuthal derivative field
     */
    Val_domain Val_domain::der_p() const
    {
        return zone->der_p(*this);
    }

    /**
     * @brief Radial derivative divided by r²: (1/r²)∂f/∂r
     * @return Modified radial derivative
     *
     * Useful for certain coordinate transformations and operators.
     */
    Val_domain Val_domain::der_r_rtwo() const
    {
        return zone->der_r_rtwo(*this);
    }

    /**
     * @brief Divide field by radial coordinate: f/r
     * @return Field divided by r
     *
     * Common operation in spherical coordinate calculations.
     */
    Val_domain Val_domain::div_r() const
    {
        if (is_zero)
            return *this;
        else
            return zone->div_r(*this);
    }

    /**
     * @brief Multiply field by radial coordinate: f*r
     * @return Field multiplied by r
     *
     * Common operation in spherical coordinate calculations.
     */
    Val_domain Val_domain::mult_r() const
    {
        if (is_zero)
            return *this;
        else
            return zone->mult_r(*this);
    }

    // ============================================================================
    // Integration Operations
    // ============================================================================

    /**
     * @brief Integrate field over domain (with spectral weight function)
     * @return Integral value ∫f w(x) dx where w is spectral weight
     *
     * Computes weighted integral using spectral quadrature.
     * Returns 0 for zero fields (optimization).
     */
    double Val_domain::integrale() const
    {
        if (is_zero)
            return 0.;
        else
            return zone->integrale(*this);
    }

    /**
     * @brief Integrate field over domain volume
     * @return Volume integral ∫f dV
     *
     * Computes actual volume integral including geometric factors.
     * Returns 0 for zero fields (optimization).
     */
    double Val_domain::integ_volume() const
    {
        if (is_zero)
            return 0.;
        else
            return zone->integ_volume(*this);
    }

    // ============================================================================
    // Private Derivative Computation
    // ============================================================================

    // Forward declaration of 1D derivative function
    int der_1d(int, Array<double>&);

    /**
     * @brief Compute all coordinate derivatives (internal)
     * @post All p_der_var entries populated with derivative fields
     *
     * Computes derivatives with respect to all coordinate variables
     * by applying spectral derivative operators in each dimension.
     * Results are cached in p_der_var array.
     *
     * Uses ope_1d framework to apply der_1d operator along each dimension.
     */
    void Val_domain::compute_der_var() const
    {
        coef(); // Ensure spectral representation exists

        for (int var = 0; var < zone->get_ndim(); var++) {
            Val_domain res(zone);
            res.base = base;
            res.cf = new Array<double>(base.ope_1d(der_1d, var, *cf, res.base));
            res.in_coef = true;

            if (p_der_var[var] != nullptr)
                delete p_der_var[var];
            p_der_var[var] = new Val_domain(res);
        }
    }

    /**
     * @brief Compute absolute coordinate derivatives (internal)
     * @post All p_der_abs entries populated with derivative fields
     *
     * Computes derivatives with respect to absolute (typically Cartesian)
     * coordinates by transforming from coordinate derivatives.
     * Results are cached in p_der_abs array.
     *
     * @pre Coordinate derivatives (p_der_var) must be computed first
     */
    void Val_domain::compute_der_abs() const
    {
        // Ensure coordinate derivatives exist
        for (int i = 0; i < zone->get_ndim(); i++)
            if (p_der_var[i] == nullptr)
                compute_der_var();

        // Transform to absolute coordinates (domain-specific)
        zone->do_der_abs_from_der_var(p_der_var, p_der_abs);
    }

} // namespace Kadath
