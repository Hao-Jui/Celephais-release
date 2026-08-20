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
 * @file index_impl.hpp
 * @brief Inline implementation of the Index class
 * @author Philippe Grandclement
 * @date 2017
 *
 * This file contains inline implementations of Index member functions.
 * Index is a multi-dimensional coordinate/iterator class used extensively
 * throughout Kadath for navigating multi-dimensional arrays and tensors.
 *
 * The Index class provides:
 * - Multi-dimensional coordinate storage and manipulation
 * - Iterator-like incrementing with automatic wrapping
 * - Bounds-aware navigation through multi-dimensional spaces
 * - Efficient inline operations for performance-critical loops
 *
 * Key capabilities:
 * - Standard iteration: increment through all points in lexicographic order
 * - Partial increment: increment specific dimensions with carry propagation
 * - Coordinate access: read and modify individual dimension indices
 * - Bounds checking: maintains coordinates within valid ranges
 *
 * Typical usage patterns:
 * @code
 * Dim_array dims(3);
 * dims.set(0) = 10; dims.set(1) = 20; dims.set(2) = 5;
 *
 * // Iterate through all points
 * Index pos(dims);
 * do {
 *     // Process point at pos(0), pos(1), pos(2)
 * } while (pos.inc());
 *
 * // Manual coordinate manipulation
 * Index idx(dims);
 * idx.set(0) = 5;
 * idx.set(1) = 10;
 * @endcode
 *
 * Design philosophy:
 * - Lightweight wrapper around coordinate array
 * - Inline for zero-overhead in tight loops
 * - Automatic bounds management via associated Dim_array
 * - Iterator semantics for natural loop integration
 */

#include "assert.h"
#include <stdexcept>
#include "dim_array.hpp"

namespace Kadath
{

    // ============================================================================
    // Constructors
    // ============================================================================

    /**
     * @brief Construct Index with specified dimensions
     * @param res Dim_array specifying the bounds in each dimension
     * @post All coordinates initialized to 0 (starting position)
     * @post sizes is set to res
     *
     * Creates an Index positioned at the origin (0,0,...,0) with bounds
     * defined by res. The Index can then be incremented to iterate through
     * all valid coordinate positions.
     *
     * Example:
     * @code
     * Dim_array dims(2);
     * dims.set(0) = 5; dims.set(1) = 3;
     * Index idx(dims);  // Creates index at (0,0) with bounds [5,3]
     * @endcode
     */
    inline Index::Index(const Dim_array& res)
        : sizes(res), coord(allocate_storage(get_ndim()))
    {
        for (int i = 0; i < get_ndim(); i++)
            coord[i] = 0;
    }

    /**
     * @brief Copy constructor - creates deep copy
     * @param so Source Index to copy from
     * @post New object has same sizes and coordinate values as source
     * @post Independent storage (not shared with source)
     *
     * Creates a complete copy with its own coordinate array. Modifications
     * to the copy do not affect the original.
     */
    inline Index::Index(const Index& so)
        : sizes(so.sizes), coord(allocate_storage(get_ndim()))
    {
        for (int i = 0; i < get_ndim(); i++)
            coord[i] = so.coord[i];
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
    inline Index::~Index()
    {
        if (!uses_inline_storage(get_ndim()))
            MemoryMapper::release_memory<int>(coord, get_ndim());
    }

    // ============================================================================
    // Coordinate Access
    // ============================================================================

    /**
     * @brief Get mutable reference to coordinate at dimension i (read/write)
     * @param i Dimension index (0-based)
     * @return Reference to coordinate value at dimension i
     * @pre 0 <= i < get_ndim()
     *
     * Provides write access to modify coordinate values directly. Use when
     * you need to set or adjust specific coordinates manually.
     *
     * Example:
     * @code
     * Index idx(dims);
     * idx.set(0) = 5;     // Set first coordinate to 5
     * idx.set(1) += 2;    // Increment second coordinate by 2
     * @endcode
     *
     * @warning Direct coordinate modification bypasses bounds checking.
     *          Ensure values remain within [0, sizes(i)) to maintain validity.
     */
    inline int& Index::set(int i)
    {
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
        if (i < 0 || i >= get_ndim())
            throw std::out_of_range("Index::set(int): dimension index out of range");
#endif
        return coord[i];
    }

    /**
     * @brief Get coordinate value at dimension i (read-only)
     * @param i Dimension index (0-based)
     * @return Copy of coordinate value at dimension i
     * @pre 0 <= i < get_ndim()
     *
     * Provides read-only access via function call syntax. Used in const
     * contexts or when read-only access is semantically appropriate.
     *
     * Example:
     * @code
     * const Index idx(dims);
     * int x = idx(0);     // Read first coordinate
     * int y = idx(1);     // Read second coordinate
     * @endcode
     */
    inline int Index::operator()(int i) const
    {
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
        if (i < 0 || i >= get_ndim())
            throw std::out_of_range("Index::operator()(int): dimension index out of range");
#endif
        return coord[i];
    }

    // ============================================================================
    // Assignment and Initialization
    // ============================================================================

    /**
     * @brief Assignment operator - copy coordinates from another Index
     * @param so Source Index to copy from
     * @pre sizes must equal so.sizes (dimensions must match)
     * @post All coordinate values are copied from source
     *
     * Copies only the coordinate values, not the structure. Both Index objects
     * must have the same dimensional bounds.
     *
     * Example:
     * @code
     * Index idx1(dims), idx2(dims);
     * idx2.set(0) = 5; idx2.set(1) = 10;
     * idx1 = idx2;  // Copy coordinates: idx1 now at (5, 10)
     * @endcode
     */
    inline void Index::operator=(const Index& so)
    {
        assert(sizes == so.sizes);
        for (int i = 0; i < get_ndim(); i++)
            coord[i] = so.coord[i];
    }

    /**
     * @brief Reset all coordinates to zero (starting position)
     * @post All coord[i] = 0
     *
     * Resets the Index to the origin, allowing iteration to restart from
     * the beginning.
     *
     * Example:
     * @code
     * Index idx(dims);
     * // ... iterate through some positions ...
     * idx.set_start();  // Back to (0, 0, ..., 0)
     * @endcode
     */
    inline void Index::set_start()
    {
        for (int i = 0; i < get_ndim(); i++)
            coord[i] = 0;
    }

    // ============================================================================
    // Increment Operations
    // ============================================================================

    /**
     * @brief Increment specific dimension with carry propagation
     * @param increm Amount to increment by
     * @param var Dimension to increment (0-based)
     * @return true if result is within bounds, false if overflow occurred
     *
     * Increments the coordinate in dimension var by increm, with automatic
     * carry propagation to higher dimensions (like multi-digit addition).
     * This enables partial iteration along specific dimensions.
     *
     * Algorithm:
     * 1. Add increm to coord[var]
     * 2. For each dimension from var to ndim-2:
     *    - Compute quotient and remainder via division by sizes(i)
     *    - Set coord[i] to remainder (wrapped value)
     *    - Add quotient to coord[i+1] (carry)
     * 3. Check if highest dimension overflowed
     *
     * Example (2D array with sizes [3, 4]):
     * @code
     * Index idx(dims);          // Start at (0, 0)
     * idx.inc(1, 0);           // (1, 0) - increment dimension 0
     * idx.inc(1, 0);           // (2, 0)
     * idx.inc(1, 0);           // (0, 1) - wrapped and carried to dimension 1
     * idx.inc(2, 0);           // (2, 1)
     * @endcode
     *
     * Use cases:
     * - Iterate through lower dimensions while keeping higher ones fixed
     * - Skip ahead by multiple positions in a dimension
     * - Implement custom iteration patterns
     *
     * @return false when the highest dimension exceeds its bound (end of range)
     */
    inline bool Index::inc(int increm, int var)
    {
        bool res = ((var >= 0) && (var < get_ndim())) ? true : false;

        if (res) {
            coord[var] += increm;

            // Propagate carries from dimension var upward. Native / and %
            // (here as one division + multiply-subtract) inline to a single
            // hardware UDIV+MSUB; the libc div() this replaced was a real
            // out-of-line call and the hottest leaf in the Jacobian assembly
            // (~10% of the first-J build, invoked per element by the strided
            // transform-loop iterators). Truncation toward zero matches div()
            // exactly, so this is bit-identical for any sign.
            for (int i = var; i < get_ndim() - 1; i++) {
                const int dim = sizes(i);
                const int quot = coord[i] / dim;
                coord[i] -= quot * dim;        // Wrapped coordinate (== coord[i] % dim)
                coord[i + 1] += quot;          // Carry to next dimension
            }

            // Check if we've overflowed the highest dimension
            res = (coord[get_ndim() - 1] >= sizes(get_ndim() - 1)) ? false : true;
        }

        return res;
    }

    /**
     * @brief Increment to next position in lexicographic order
     * @return true if increment succeeded, false if reached end
     *
     * Standard iterator increment: moves to the next coordinate position in
     * lexicographic (dictionary) order. This is the most common iteration pattern
     * for traversing all points in a multi-dimensional space.
     *
     * Lexicographic ordering (dimension 0 varies fastest):
     * (0,0,0) → (1,0,0) → (2,0,0) → ... → (n₀-1,0,0) →
     * (0,1,0) → (1,1,0) → ... → (n₀-1,n₁-1,n₂-1)
     *
     * Algorithm:
     * 1. Try to increment dimension 0
     * 2. If it would overflow, wrap to 0 and try dimension 1
     * 3. Continue until an increment succeeds or all dimensions overflow
     * 4. Return false only when no dimension can be incremented (end reached)
     *
     * Typical usage pattern:
     * @code
     * Dim_array dims(3);
     * dims.set(0) = nx; dims.set(1) = ny; dims.set(2) = nz;
     *
     * Index pos(dims);
     * do {
     *     // Process position (pos(0), pos(1), pos(2))
     *     double value = array(pos);
     * } while (pos.inc());
     * @endcode
     *
     * Performance note: This is the most cache-efficient iteration pattern
     * since dimension 0 (innermost) varies fastest, matching typical array
     * memory layout.
     *
     * @return false when all positions have been visited (iterator exhausted)
     */
    inline bool Index::inc()
    {
        for (int i = 0; i < get_ndim(); ++i) {
            if (coord[i] + 1 < sizes(i)) {
                coord[i]++;
                return true; // Successfully incremented dimension i
            } else {
                coord[i] = 0; // Wrap and try next dimension
            }
        }
        return false; // All dimensions overflowed - iteration complete
    }

    // ============================================================================
    // Comparison Operations
    // ============================================================================

    /**
     * @brief Equality comparison operator
     * @param xx Index to compare with
     * @return true if dimensions match and all coordinates are equal
     *
     * Two Index objects are equal if:
     * 1. They have the same number of dimensions
     * 2. All corresponding coordinates are equal
     *
     * Note: The sizes (bounds) don't need to match, only the actual coordinate
     * values. This allows comparing positions from different Index objects.
     *
     * Example:
     * @code
     * Index idx1(dims1), idx2(dims2);
     * idx1.set(0) = 5; idx1.set(1) = 10;
     * idx2.set(0) = 5; idx2.set(1) = 10;
     * if (idx1 == idx2) {
     *     // Positions match
     * }
     * @endcode
     */
    inline bool Index::operator==(const Index& xx) const
    {
        bool res = (get_ndim() == xx.get_ndim()) ? true : false;

        if (res) {
            for (int i = 0; i < get_ndim(); i++) {
                if (xx.coord[i] != coord[i]) {
                    res = false;
                    break;
                }
            }
        }

        return res;
    }

} // namespace Kadath
