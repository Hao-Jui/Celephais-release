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
 *   2026-08-10  Added capacity-3 inline storage with heap fallback.
 */

/**
 * @file dim_array.hpp
 * @brief Inline implementation of the Dim_array class
 * @author Philippe Grandclement
 * @date 2017
 *
 * This file contains inline implementations of Dim_array member functions.
 * Dim_array is a lightweight container for storing dimensional information
 * of multi-dimensional arrays, particularly for spectral method computations.
 *
 * The class provides:
 * - Storage for array dimensions (sizes in each direction)
 * - Efficient inline operations for performance-critical code
 * - Serialization support (binary file I/O)
 * - Safe bounds-checked element access
 *
 * Typical usage:
 * - Specify dimensions of spectral coefficient arrays
 * - Define sizes of collocation point grids
 * - Track dimension transformations during spectral transforms
 *
 * Design philosophy:
 * - Simple, focused responsibility (dimension storage only)
 * - Inline functions for zero-overhead abstraction
 * - Managed memory via MemoryMapper for tracking
 * - Value semantics with explicit copy operations
 */

#pragma once

#include "assert.h"
#include "For_Kadath/Utilities/utilities.hpp"

namespace Kadath
{

    // ============================================================================
    // Constructors
    // ============================================================================

    /**
     * @brief Construct Dim_array with specified number of dimensions
     * @param dim Number of dimensions
     * @post ndim is set to dim
     * @post Storage is available for dim integers (uninitialized)
     *
     * Creates a Dim_array to hold dimension sizes for a dim-dimensional array.
     * The dimension values are NOT initialized - caller must set them explicitly.
     *
     * Example:
     * @code
     * Dim_array dims(3);      // Create 3D dimension array
     * dims.set(0) = 32;       // Set first dimension
     * dims.set(1) = 64;       // Set second dimension
     * dims.set(2) = 16;       // Set third dimension
     * @endcode
     */
    inline Dim_array::Dim_array(int dim)
        : ndim(dim), nbr(allocate_storage(dim))
    {
    }

    /**
     * @brief Copy constructor - creates deep copy
     * @param so Source Dim_array to copy from
     * @post New object has same ndim and dimension values as source
     * @post Independent storage (not shared with source)
     *
     * Creates a complete copy with its own memory. Modifications to the
     * copy do not affect the original and vice versa.
     */
    inline Dim_array::Dim_array(const Dim_array& so)
        : ndim(so.ndim), nbr(allocate_storage(so.ndim))
    {
        for (int i = 0; i < ndim; i++)
            nbr[i] = so.nbr[i];
    }

    // ============================================================================
    // Destructor
    // ============================================================================

    /**
     * @brief Destructor - releases allocated memory
     * @post Heap fallback storage, if any, is released
     *
     * Inline storage belongs to the object; larger ranks use MemoryMapper.
     */
    inline Dim_array::~Dim_array()
    {
        if (!uses_inline_storage(ndim))
            MemoryMapper::release_memory<int>(nbr, ndim);
    }

    // ============================================================================
    // Element Access
    // ============================================================================

    /**
     * @brief Get mutable reference to dimension at index i (read/write)
     * @param i Dimension index (0-based)
     * @return Reference to dimension value at index i
     * @pre 0 <= i < ndim
     *
     * Provides write access to modify dimension values. Use for setting or
     * modifying dimensions.
     *
     * Example:
     * @code
     * Dim_array dims(3);
     * dims.set(0) = 10;       // Set first dimension to 10
     * dims.set(1) += 5;       // Increment second dimension by 5
     * @endcode
     */
    inline int& Dim_array::set(int i)
    {
        assert(i >= 0);
        assert(i < ndim);
        return nbr[i];
    }

    /**
     * @brief Get dimension value at index i (read-only)
     * @param i Dimension index (0-based)
     * @return Copy of dimension value at index i
     * @pre 0 <= i < ndim
     *
     * Provides read-only access via function call syntax. Used in const contexts
     * or when read-only access is semantically appropriate.
     *
     * Example:
     * @code
     * const Dim_array dims(3);
     * int first_dim = dims(0);    // Read first dimension
     * int total = dims(0) * dims(1) * dims(2);  // Compute total size
     * @endcode
     */
    inline int Dim_array::operator()(int i) const
    {
        assert(i >= 0);
        assert(i < ndim);
        return nbr[i];
    }

    // ============================================================================
    // Assignment Operator
    // ============================================================================

    /**
     * @brief Assignment operator - copy dimension values from another Dim_array
     * @param so Source Dim_array to copy from
     * @pre ndim must equal so.ndim (dimensions must match)
     * @post All dimension values are copied from source
     *
     * Copies only the dimension values, not the structure. Both objects must
     * have the same number of dimensions. Memory is NOT reallocated.
     *
     * Example:
     * @code
     * Dim_array dims1(3), dims2(3);
     * dims2.set(0) = 10;
     * dims2.set(1) = 20;
     * dims2.set(2) = 30;
     * dims1 = dims2;  // Copy all dimension values
     * @endcode
     */
    inline void Dim_array::operator=(const Dim_array& so)
    {
        assert(ndim == so.ndim);
        for (int i = 0; i < ndim; i++)
            nbr[i] = so.nbr[i];
    }

    // ============================================================================
    // I/O Operations
    // ============================================================================

    inline Dim_array::Dim_array(BinarySource& source)
    {
        ndim = source.read<int>();
        nbr = allocate_storage(ndim);
        source.read_array(nbr, static_cast<std::size_t>(ndim));
    }

    inline void Dim_array::save(BinarySink& sink) const
    {
        sink.write<int>(ndim);
        sink.write_array(nbr, static_cast<std::size_t>(ndim));
    }

} // namespace Kadath
