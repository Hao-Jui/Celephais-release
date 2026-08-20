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
 * @file base_spectral.cpp
 * @brief Implementation of the Base_spectral class for spectral basis management
 * @author Philippe Grandclement
 * @date 2017
 *
 * This file implements the Base_spectral class which manages multi-dimensional
 * spectral bases used in spectral method computations. Each dimension can have
 * its own basis representation stored as arrays of integers.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "base_spectral.hpp"
#include "For_Kadath/Array/array.hpp"

namespace Kadath
{

    // ============================================================================
    // Constructors
    // ============================================================================

    /**
     * @brief Construct uninitialized spectral base with given dimensions
     * @param dimensions Number of dimensions for the spectral base
     * @post Base is marked as undefined (def=false)
     * @post All 1D basis arrays are initialized to nullptr
     */
    Base_spectral::Base_spectral(int dimensions) : def(false), ndim(dimensions)
    {
        bases_1d = MemoryMapper::get_memory<Array<int>*>(ndim);
        for (int i = 0; i < ndim; i++)
            bases_1d[i] = nullptr;
    }

    /**
     * @brief Copy constructor - creates deep copy of spectral base
     * @param so Source Base_spectral object to copy from
     * @post New object has independent copies of all basis arrays
     * @post Definition state and dimensionality match source
     */
    Base_spectral::Base_spectral(const Base_spectral& so) : def(so.def), ndim(so.ndim)
    {
        bases_1d = MemoryMapper::get_memory<Array<int>*>(ndim);
        for (int l = 0; l < ndim; l++)
            bases_1d[l] = (so.bases_1d[l] == nullptr) ? nullptr : new Array<int>(*so.bases_1d[l]);
    }

    // ============================================================================
    // Destructor
    // ============================================================================

    /**
     * @brief Destructor - releases all allocated memory
     * @post All basis arrays are deallocated
     * @post Memory for the bases_1d pointer array is released
     */
    Base_spectral::~Base_spectral()
    {
        for (int l = 0; l < ndim; l++)
            if (bases_1d[l] != nullptr)
                delete bases_1d[l];
        MemoryMapper::release_memory<Array<int>*>(bases_1d, ndim);
    }

    // ============================================================================
    // Configuration Methods
    // ============================================================================

    /**
     * @brief Mark base as undefined and deallocate all basis arrays
     * @post def flag is set to false
     * @post All basis arrays are deleted and pointers set to nullptr
     * @note Used to reset the spectral base to uninitialized state
     */
    void Base_spectral::set_non_def()
    {
        for (int l = 0; l < ndim; l++)
            if (bases_1d[l] != nullptr) {
                delete bases_1d[l];
                bases_1d[l] = nullptr;
            }
        def = false;
    }

    /**
     * @brief Allocate basis arrays based on coefficient dimensions
     * @param nbr_coef Dimension array specifying number of coefficients per dimension
     * @post Previous basis arrays are deallocated
     * @post New basis arrays are allocated with appropriate sizes
     * @note Last dimension gets a 1D array, earlier dimensions get progressively
     *       higher-dimensional arrays to handle varying basis per coefficient
     *
     * Memory layout:
     * - bases_1d[ndim-1]: 1D array (single basis for last dimension)
     * - bases_1d[i]: (ndim-1-i)-dimensional array for dimension i
     */
    void Base_spectral::allocate(const Dim_array& nbr_coef)
    {
        // Deallocate existing arrays
        for (int i = 0; i < ndim; i++)
            if (bases_1d[i] != nullptr)
                delete bases_1d[i];

        // Allocate last dimension (simplest case)
        bases_1d[ndim - 1] = new Array<int>(1);

        // Allocate remaining dimensions with increasing complexity
        for (int i = ndim - 2; i >= 0; i--) {
            Dim_array taille(ndim - 1 - i);
            for (int k = 0; k < ndim - 1 - i; k++)
                taille.set(k) = nbr_coef(i + k + 1);
            bases_1d[i] = new Array<int>(taille);
        }
    }

    /**
     * @brief Set 3D spectral base with specific basis types
     * @param nbr_coefs 3D dimension array with coefficient counts
     * @param BASEPHI Basis type for phi (azimuthal) dimension
     * @param BASETHETA Basis type for theta (polar) dimension
     * @param BASER Basis type for radial dimension
     * @pre nbr_coefs must be 3-dimensional
     * @post Base is allocated and marked as defined
     * @post All basis types are set according to parameters
     *
     * This method is specifically designed for 3D spherical-like coordinate systems
     * commonly used in spectral methods for astrophysics and general relativity.
     */
    void Base_spectral::set(const Dim_array& nbr_coefs, int BASEPHI, int BASETHETA, int BASER)
    {
        assert(nbr_coefs.get_ndim() == 3);

        allocate(nbr_coefs);
        def = true;

        // Set phi basis (outermost dimension)
        bases_1d[2]->set(0) = BASEPHI;

        // Set theta and radial bases for all coefficients
        Index index(bases_1d[0]->get_dimensions());
        for (int k = 0; k < nbr_coefs(2); ++k) {
            bases_1d[1]->set(k) = BASETHETA;
            for (int j = 0; j < nbr_coefs(1); ++j) {
                index.set(0) = j;
                index.set(1) = k;
                bases_1d[0]->set(index) = BASER;
            }
        }
    }

    // ============================================================================
    // I/O Operations
    // ============================================================================

    // ============================================================================
    // Assignment Operator
    // ============================================================================

    /**
     * @brief Assignment operator - deep copy from another Base_spectral
     * @param so Source Base_spectral to copy from
     * @pre Dimensions must match (ndim == so.ndim)
     * @post All basis arrays are replaced with copies from source
     * @post Definition state is copied from source
     */
    void Base_spectral::operator=(const Base_spectral& so)
    {
        assert(ndim == so.ndim);
        def = so.def;

        for (int i = 0; i < ndim; i++) {
            // Delete existing array if present
            if (bases_1d[i] != nullptr)
                delete bases_1d[i];

            // Copy from source if defined and non-null
            if ((so.def) && (so.bases_1d[i] != nullptr))
                bases_1d[i] = new Array<int>(*so.bases_1d[i]);
            else
                bases_1d[i] = nullptr;
        }
    }

    // ============================================================================
    // Comparison Operators
    // ============================================================================

    /**
     * @brief Equality comparison operator for spectral bases
     * @param a First Base_spectral to compare
     * @param b Second Base_spectral to compare
     * @return true if bases are identical, false otherwise
     *
     * Bases are equal if:
     * - Both are defined
     * - Same dimensionality
     * - All corresponding basis values match element-wise
     *
     * @note Returns false if either base is undefined or dimensions differ
     */
    bool operator==(const Base_spectral& a, const Base_spectral& b)
    {
        bool res = true;

        // Check basic compatibility
        if ((!a.def) || (!b.def) || (a.ndim != b.ndim))
            res = false;

        // Compare all basis arrays element by element
        for (int i = 0; i < a.ndim; i++) {
            Index index(a.bases_1d[i]->get_dimensions());
            do {
                if ((*a.bases_1d[i])(index) != (*b.bases_1d[i])(index))
                    res = false;
            } while (index.inc());
        }

        return res;
    }

    // ============================================================================
    // Stream Operators
    // ============================================================================

    /**
     * @brief Stream insertion operator for formatted output
     * @param o Output stream
     * @param so Base_spectral object to output
     * @return Reference to output stream for chaining
     *
     * Output format:
     * - Header line with dimensionality
     * - If defined: basis arrays for each variable dimension
     * - If undefined: "Base not defined" message
     */
    ostream& operator<<(ostream& o, const Base_spectral& so)
    {
        o << so.ndim << "-dimensional spectral base" << endl;

        if (so.def) {
            for (int l = 0; l < so.ndim; l++) {
                o << "Variable " << l << endl;
                o << *so.bases_1d[l] << endl;
            }
        } else {
            o << "Base not defined" << endl;
        }

        return o;
    }

} // namespace Kadath