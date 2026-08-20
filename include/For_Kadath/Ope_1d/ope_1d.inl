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
 * @file ope_1d.cpp
 * @brief Generic framework for applying 1D operations to multi-dimensional arrays
 * @author Philippe Grandclement
 * @date 2017
 *
 * This file implements a flexible framework for applying 1D operations along
 * specific dimensions of multi-dimensional arrays while potentially changing
 * the spectral basis. It generalizes the transform operations by allowing
 * arbitrary user-defined functions to be applied.
 *
 * The framework provides:
 * - Application of 1D operations along any dimension
 * - Batch processing of independent 1D slices
 * - Dynamic basis type transformation
 * - Separation of data transformation from basis metadata updates
 *
 * Key design features:
 * - Generic function pointer interface for flexibility
 * - Basis-aware operations (function receives current basis type)
 * - Simultaneous data and metadata transformation
 * - Efficient memory access patterns (stride optimization)
 *
 * Common use cases:
 * - Derivative computation along specific dimensions
 * - Integration along one coordinate
 * - Filtering or smoothing operations
 * - Basis conversion (e.g., Chebyshev → Legendre)
 * - Custom spectral operations (multiplication by coordinate, etc.)
 *
 * The function signature pattern:
 * @code
 * int operation(int base_in, Array<double>& data) {
 *     // Modify data in-place based on input basis type
 *     // Return new basis type for this data
 *     return base_out;
 * }
 * @endcode
 *
 * Example usage:
 * @code
 * // Define a derivative operation
 * int derivative_r(int base, Array<double>& coef) {
 *     // Apply r-derivative in spectral space
 *     apply_derivative_rules(base, coef);
 *     return base;  // Basis type unchanged
 * }
 *
 * // Apply derivative along radial dimension (dimension 2)
 * Base_spectral base_out(3);
 * Array<double> result = base_in.ope_1d(derivative_r, 2, coef_array, base_out);
 * @endcode
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

namespace Kadath
{

    /**
     * @brief Apply 1D operation along specified dimension with basis tracking
     * @param func Function pointer to 1D operation: int (*func)(int base, Array<double>& data)
     * @param var Dimension along which to apply the operation (0-based)
     * @param in Input multi-dimensional array
     * @param base_out Output Base_spectral object to store resulting basis information
     * @return Transformed array with same dimensions as input
     *
     * This is a generic framework that applies a user-defined 1D operation along
     * one dimension of a multi-dimensional array. The operation is applied independently
     * to each 1D slice along the specified dimension, making it suitable for separable
     * operations in spectral methods.
     *
     * Function parameter details:
     *
     * @param func - User-defined transformation function with signature:
     *   - Input: int base - current basis type for this slice
     *   - Input/Output: Array<double>& data - 1D array to transform (modified in-place)
     *   - Returns: int - new basis type after transformation
     *
     *   The function modifies the data array in-place and returns the basis type
     *   that describes the transformed data. This allows operations to change both
     *   the data representation and its basis metadata.
     *
     * @param var - Dimension index along which to apply operation:
     *   - Must satisfy: 0 <= var < ndim
     *   - All 1D slices along this dimension are processed
     *   - Other dimensions are treated as batch indices
     *
     * @param in - Input array:
     *   - Can have any dimensionality (ndim)
     *   - Data should be in spectral space if operation is spectral
     *   - Remains unchanged (const)
     *
     * @param base_out - Output basis specification:
     *   - Must be pre-allocated with same ndim as 'this'
     *   - Will be populated with new basis types for each position
     *   - Allows basis tracking through operations
     *
     * Algorithm overview:
     * 1. Create output array matching input dimensions
     * 2. Compute stride patterns (before/after the target dimension)
     * 3. Loop over all "before" dimensions (higher indices)
     * 4. For each position, retrieve current basis type
     * 5. Loop over all "after" dimensions (lower indices)
     * 6. Extract 1D slice along dimension var
     * 7. Apply user-defined operation and get new basis type
     * 8. Store new basis type in base_out
     * 9. Store transformed data in result array
     *
     * Memory access pattern:
     * - "after" dimensions: innermost (stride = 1)
     * - Target dimension: middle (stride = after)
     * - "before" dimensions: outermost
     * This ordering provides good cache locality for typical operations.
     *
     * Example applications:
     *
     * 1. Radial derivative in 3D spherical coordinates:
     * @code
     * int deriv_r(int base, Array<double>& c) {
     *     apply_chebyshev_derivative(c);
     *     return base;  // Still Chebyshev after derivative
     * }
     * Base_spectral base_dr(3);
     * Array<double> df_dr = base.ope_1d(deriv_r, 2, f_coef, base_dr);
     * @endcode
     *
     * 2. Integration along one dimension:
     * @code
     * int integrate_theta(int base, Array<double>& c) {
     *     apply_legendre_integration(c);
     *     return LEG;  // Integrated Legendre is still Legendre
     * }
     * Base_spectral base_int(3);
     * Array<double> integral = base.ope_1d(integrate_theta, 1, f_coef, base_int);
     * @endcode
     *
     * 3. Basis conversion:
     * @code
     * int cheb_to_leg(int base, Array<double>& c) {
     *     if (base == CHEB) {
     *         convert_chebyshev_to_legendre(c);
     *         return LEG;
     *     }
     *     return base;
     * }
     * Base_spectral base_leg(3);
     * Array<double> leg_coef = base.ope_1d(cheb_to_leg, 0, cheb_coef, base_leg);
     * @endcode
     *
     * @pre 0 <= var < ndim
     * @pre func must be a valid function pointer
     * @pre base_out must be pre-allocated with same ndim as 'this'
     * @pre Input array must have compatible dimensions
     *
     * @post Result has same dimensions as input
     * @post base_out is populated with new basis types for all positions
     * @post Input array is unchanged
     *
     * Performance considerations:
     * - Time complexity: O(N^d) where N is typical size and d is dimensionality
     * - Each 1D slice is processed independently (parallelizable in principle)
     * - Cache-friendly access pattern (stride-1 in innermost loop)
     * - Memory overhead: one working 1D array of size nbr
     *
     * Thread safety:
     * - Not thread-safe (modifies base_out and uses shared Index objects)
     * - Each thread would need its own base_out and Index objects
     *
     * @return Transformed array with same dimensions as input
     */
    Array<double> Base_spectral::ope_1d(int (*func)(int, Array<double>&), int var, const Array<double>& in,
                                        Base_spectral& base_out) const
    {
        // Allocate result array with same dimensions as input
        Array<double> res(in.get_dimensions());

        // Compute strides for memory access patterns
        // "after": number of elements in dimensions 0 to var-1 (innermost)
        int after = 1;
        for (int i = 0; i < var; i++)
            after *= in.get_size(i);

        // "before": number of elements in dimensions var+1 to ndim-1 (outermost)
        int before = 1;
        for (int i = var + 1; i < ndim; i++)
            before *= in.get_size(i);

        // Size of the target dimension
        int nbr = in.get_size(var);

        // Index objects for navigating multi-dimensional arrays
        Index index_base(bases_1d[var]->get_dimensions()); // Position in basis array
        Index demarre(in.get_dimensions());                // Starting position for current slice
        Index loop_before(in.get_dimensions());            // Loop counter for "before" dimensions
        Index lit(in.get_dimensions());                    // Read position
        Index put(in.get_dimensions());                    // Write position

        // Working buffer for 1D operations
        Array<double> tab_1d(nbr);

        // Loop over all "before" dimensions (outermost loop)
        // Each iteration processes a different set of 1D slices
        for (int i = 0; i < before; i++) {
            demarre = loop_before;

            // Retrieve the current basis type for this position
            // Different positions may have different basis types
            int base = (*bases_1d[var])(index_base);

            // Loop over all "after" dimensions (inner loop)
            // Each iteration processes one 1D slice
            for (int j = 0; j < after; j++) {
                // Extract 1D slice along dimension var
                lit = demarre;
                for (int k = 0; k < nbr; k++) {
                    tab_1d.set(k) = in(lit);
                    lit.inc(after); // Move to next element along var
                }

                // Apply the user-defined transformation
                // Function modifies tab_1d in-place and returns new basis type
                base_out.bases_1d[var]->set(index_base) = func(base, tab_1d);

                // Store transformed data in result array
                put = demarre;
                for (int k = 0; k < nbr; k++) {
                    res.set(put) = tab_1d(k);
                    put.inc(after); // Move to next element along var
                }

                // Move to next position in "after" dimensions
                demarre.inc();
            }

            // Move to next position in "before" dimensions
            index_base.inc();
            loop_before.inc(1, var + 1);
        }

        return res;
    }

} // namespace Kadath