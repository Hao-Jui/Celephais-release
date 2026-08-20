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
 *   2026-08-10  Added capacity-3 inline coordinate storage with heap fallback.
 */

/**
 * @file point.hpp
 * @brief Inline implementation of the Point class
 * @author Philippe Grandclement
 * @date 2017
 *
 * This file contains inline implementations of Point member functions.
 * Point represents a geometric position in multi-dimensional space using
 * floating-point coordinates.
 *
 * The Point class provides:
 * - Storage for coordinates in n-dimensional space
 * - Efficient inline operations for performance-critical code
 * - Serialization support (binary file I/O)
 * - Safe 1-based indexing convention for compatibility
 *
 * Key characteristics:
 * - 1-BASED INDEXING: Coordinates accessed as (1, 2, 3, ..., ndim)
 * - Double precision coordinates for accuracy
 * - Value semantics with explicit copy operations
 * - Inline storage through rank 3, with MemoryMapper heap fallback
 *
 * Typical usage:
 * - Represent physical positions in space
 * - Store collocation points for spectral methods
 * - Define boundary positions or evaluation points
 * - Track spatial coordinates in numerical simulations
 *
 * Important indexing convention:
 * Unlike most C/C++ arrays, Point uses 1-based indexing for mathematical
 * convenience and compatibility with physics conventions where coordinates
 * are typically labeled (x₁, x₂, x₃) or (r, θ, φ).
 *
 * Example:
 * @code
 * Point p(3);           // 3D point
 * p.set(1) = 1.0;       // x₁ = 1.0 (first coordinate)
 * p.set(2) = 2.0;       // x₂ = 2.0 (second coordinate)
 * p.set(3) = 3.0;       // x₃ = 3.0 (third coordinate)
 *
 * double x = p(1);      // Read first coordinate
 * double y = p(2);      // Read second coordinate
 * double z = p(3);      // Read third coordinate
 * @endcode
 *
 * Design philosophy:
 * - Lightweight geometric primitive
 * - Inline for zero-overhead abstraction
 * - 1-based indexing matches mathematical conventions
 * - Independent from array structures (Index uses 0-based)
 */

#include "headcpp.hpp"
#include <assert.h>
#include "For_Kadath/Utilities/utilities.hpp"

namespace Kadath
{

    // ============================================================================
    // Constructors
    // ============================================================================

    /**
     * @brief Construct Point with specified number of dimensions
     * @param dim Number of spatial dimensions
     * @post All coordinates initialized to 0.0 (origin)
     * @post ndim is set to dim
     *
     * Creates a Point at the origin of dim-dimensional space. All coordinates
     * are zero-initialized and must be set explicitly if other values are needed.
     *
     * Example:
     * @code
     * Point origin_2d(2);           // 2D origin at (0, 0)
     * Point origin_3d(3);           // 3D origin at (0, 0, 0)
     *
     * Point position(3);
     * position.set(1) = 1.5;        // Set x₁ = 1.5
     * position.set(2) = -2.3;       // Set x₂ = -2.3
     * position.set(3) = 0.7;        // Set x₃ = 0.7
     * @endcode
     */
    inline Point::Point(int dim)
        : ndim(dim), coord(allocate_storage(dim))
    {
        for (int i = 0; i < ndim; i++)
            coord[i] = 0;
    }

    /**
     * @brief Copy constructor - creates deep copy
     * @param so Source Point to copy from
     * @post New object has same ndim and coordinate values as source
     * @post Independent storage (not shared with source)
     *
     * Creates a complete copy with its own coordinate array. Modifications
     * to the copy do not affect the original.
     *
     * Example:
     * @code
     * Point p1(3);
     * p1.set(1) = 1.0; p1.set(2) = 2.0; p1.set(3) = 3.0;
     *
     * Point p2(p1);  // p2 is independent copy of p1
     * p2.set(1) = 5.0;  // p1 unchanged
     * @endcode
     */
    inline Point::Point(const Point& so)
        : ndim(so.ndim), coord(allocate_storage(so.ndim))
    {
        for (int i = 0; i < ndim; i++)
            coord[i] = so.coord[i];
    }

    inline Point::Point(BinarySource& source)
    {
        ndim = source.read<int>();
        coord = allocate_storage(ndim);
        source.read_array(coord, static_cast<std::size_t>(ndim));
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
    inline Point::~Point()
    {
        if (!uses_inline_storage(ndim))
            MemoryMapper::release_memory<double>(coord, ndim);
    }

    // ============================================================================
    // I/O Operations
    // ============================================================================

    inline void Point::save(BinarySink& sink) const
    {
        sink.write<int>(ndim);
        sink.write_array(coord, static_cast<std::size_t>(ndim));
    }

    // ============================================================================
    // Assignment Operator
    // ============================================================================

    /**
     * @brief Assignment operator - copy coordinates from another Point
     * @param so Source Point to copy from
     * @pre ndim must equal so.ndim (dimensions must match)
     * @post All coordinate values are copied from source
     *
     * Copies only the coordinate values, not the structure. Both Point objects
     * must have the same dimensionality. Memory is NOT reallocated.
     *
     * Example:
     * @code
     * Point p1(3), p2(3);
     * p2.set(1) = 1.0; p2.set(2) = 2.0; p2.set(3) = 3.0;
     * p1 = p2;  // Copy all coordinates
     * @endcode
     */
    inline void Point::operator=(const Point& so)
    {
        assert(ndim == so.ndim);
        for (int i = 0; i < ndim; i++)
            coord[i] = so.coord[i];
    }

    // ============================================================================
    // Coordinate Access
    // ============================================================================

    /**
     * @brief Get mutable reference to coordinate (read/write, 1-based indexing)
     * @param i Coordinate index (1-based: 1, 2, 3, ..., ndim)
     * @return Reference to coordinate value at index i
     * @pre 1 <= i <= ndim
     *
     * IMPORTANT: Uses 1-based indexing for mathematical convention.
     * - i=1 accesses the first coordinate (x₁)
     * - i=2 accesses the second coordinate (x₂)
     * - i=3 accesses the third coordinate (x₃)
     * - etc.
     *
     * This differs from typical C++ 0-based indexing but matches physics and
     * mathematics conventions where coordinates are labeled starting from 1.
     *
     * Provides write access to modify coordinate values.
     *
     * Example:
     * @code
     * Point p(3);
     * p.set(1) = 1.0;     // Set x₁ (first coordinate)
     * p.set(2) = 2.0;     // Set x₂ (second coordinate)
     * p.set(3) = 3.0;     // Set x₃ (third coordinate)
     *
     * p.set(1) *= 2.0;    // Multiply first coordinate by 2
     * @endcode
     *
     * @warning Uses 1-based indexing! Valid range: [1, ndim], NOT [0, ndim-1]
     */
    inline double& Point::set(int i)
    {
        assert(i > 0);
        assert(i <= ndim);
        return coord[i - 1]; // Convert 1-based to 0-based for internal storage
    }

    /**
     * @brief Get coordinate value (read-only, 1-based indexing)
     * @param i Coordinate index (1-based: 1, 2, 3, ..., ndim)
     * @return Copy of coordinate value at index i
     * @pre 1 <= i <= ndim
     *
     * IMPORTANT: Uses 1-based indexing for mathematical convention.
     * - i=1 accesses the first coordinate (x₁)
     * - i=2 accesses the second coordinate (x₂)
     * - i=3 accesses the third coordinate (x₃)
     * - etc.
     *
     * Provides read-only access via function call syntax. Used in const contexts
     * or when read-only access is semantically appropriate.
     *
     * Example:
     * @code
     * const Point p(3);
     * double x = p(1);    // Read x₁ (first coordinate)
     * double y = p(2);    // Read x₂ (second coordinate)
     * double z = p(3);    // Read x₃ (third coordinate)
     *
     * // Compute distance from origin
     * double r = sqrt(p(1)*p(1) + p(2)*p(2) + p(3)*p(3));
     * @endcode
     *
     * @warning Uses 1-based indexing! Valid range: [1, ndim], NOT [0, ndim-1]
     */
    inline double Point::operator()(int i) const
    {
        assert(i > 0);
        assert(i <= ndim);
        return coord[i - 1]; // Convert 1-based to 0-based for internal storage
    }

} // namespace Kadath
