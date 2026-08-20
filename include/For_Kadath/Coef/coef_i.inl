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
 * @file coef_i.cpp
 * @brief Multi-dimensional inverse spectral transformations
 * @author Philippe Grandclement
 * @date 2017
 *
 * This file implements multi-dimensional inverse spectral transformations by
 * applying 1D inverse transforms sequentially along each dimension. This approach
 * extends the 1D transforms (from coef_i_fftw.cpp) to handle tensorial spectral
 * expansions in 2D, 3D, or higher-dimensional spaces.
 *
 * The core strategy is dimensional separation: transform along one dimension at
 * a time while treating other dimensions as batch operations. This is efficient
 * and leverages the tensor product structure of multi-dimensional spectral bases.
 *
 * Key features:
 * - Supports different basis types per dimension (mixed spectral bases)
 * - Handles dimension size changes during transformation
 * - Maintains proper data layout and striding for cache efficiency
 * - Batch processes all independent 1D transforms in parallel conceptually
 *
 * Mathematical background: A multi-dimensional spectral expansion can be written as
 * f(x₁,...,xₙ) = Σ_{k₁,...,kₙ} c_{k₁,...,kₙ} φ₁_{k₁}(x₁)...φₙ_{kₙ}(xₙ)
 * where φᵢ are the 1D basis functions for dimension i. The inverse transform
 * reconstructs f from the coefficients c by evaluating this sum at collocation points.
 */

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/array.hpp"

namespace Kadath
{

    // Forward declaration of 1D inverse transform dispatcher
    void coef_i_1d(int, Array<double>&);

    /**
     * @brief Perform inverse spectral transform along a single dimension
     * @param dim Dimension index to transform (0-based)
     * @param nbr_conf Number of collocation points desired in output along dimension dim
     * @param inout Input/output array (modified in-place, replaced with new array)
     *
     * This function transforms spectral coefficients to physical space values along
     * one specified dimension while preserving all other dimensions. It handles:
     * - Variable basis types (different bases for different positions)
     * - Dimension size changes (nbr_coef → nbr_conf)
     * - Proper memory layout and striding for efficient access
     *
     * Algorithm overview:
     * 1. Create output array with new size in dimension dim
     * 2. Compute stride patterns (before/after the transformed dimension)
     * 3. Loop over all "before" dimensions (higher indices)
     * 4. For each position, determine the appropriate 1D basis type
     * 5. Loop over all "after" dimensions (lower indices)
     * 6. Extract 1D slice along dimension dim
     * 7. Apply 1D inverse transform using appropriate basis
     * 8. Store result in output array
     * 9. Replace input pointer with new output array
     *
     * Memory layout:
     * - "after" dimensions: stride = 1 (innermost, varying fastest)
     * - Transformed dimension: stride = after
     * - "before" dimensions: stride = after * size(dim) (outermost)
     *
     * @pre inout must point to a valid Array<double> with ndim dimensions
     * @pre 0 <= dim < ndim
     * @pre nbr_conf > 0
     * @pre Basis functions must be defined for dimension dim
     *
     * @post Input array is left unchanged
     * @post Output dimension sizes: same except size(dim) = nbr_conf
     * @post Spectral coefficients along dim are converted to collocation values
     *
     * @return a fresh array holding the configuration-space values along dimension dim
     * @note Basis type can vary with position in "before" dimensions
     */
    Array<double> Base_spectral::coef_i_dim(int dim, int nbr_conf, const Array<double>& inout) const
    {
        // Setup output dimensions (same as input except for dimension dim)
        Dim_array res_out(inout.get_dimensions());
        res_out.set(dim) = nbr_conf;
        Array<double> res(res_out);

        // Compute strides for memory access
        // "after": number of elements in dimensions 0 to dim-1 (innermost)
        int after = 1;
        for (int i = 0; i < dim; i++)
            after *= inout.get_size(i);

        // "before": number of elements in dimensions dim+1 to ndim-1 (outermost)
        int before = 1;
        for (int i = dim + 1; i < ndim; i++)
            before *= inout.get_size(i);

        // Determine working buffer size (max of input and output sizes)
        int nbr_coef = inout.get_size(dim);
        int nbr = (nbr_coef > nbr_conf) ? nbr_coef : nbr_conf;

        // Index objects for navigating multi-dimensional arrays
        Index index_base(bases_1d[dim]->get_dimensions()); // Position in basis array
        Index demarre_coef(inout.get_dimensions());        // Starting position in input
        Index demarre_conf(res_out);                       // Starting position in output
        Index loop_before_coef(inout.get_dimensions());    // Loop counter for "before" dims (input)
        Index loop_before_out(res_out);                    // Loop counter for "before" dims (output)
        Index lit_in(inout.get_dimensions());              // Read position in input
        Index put_out(res_out);                            // Write position in output

        // Working buffer for 1D transforms
        Array<double> tab_1d(nbr);

        // Loop over all "before" dimensions (outermost loop)
        for (int i = 0; i < before; i++) {
            demarre_coef = loop_before_coef;
            demarre_conf = loop_before_out;

            // Determine which 1D basis to use for this position
            int base = (*bases_1d[dim])(index_base);

            // Loop over all "after" dimensions (inner loop - batch of 1D transforms)
            for (int j = 0; j < after; j++) {
                // Extract 1D slice of coefficients along dimension dim
                lit_in = demarre_coef;
                for (int k = 0; k < nbr_coef; k++) {
                    tab_1d.set(k) = inout(lit_in);
                    lit_in.inc(after); // Move to next element along dim
                }

                // Apply 1D inverse spectral transform
                coef_i_1d(base, tab_1d);

                // Store transformed values in output array
                put_out = demarre_conf;
                for (int k = 0; k < nbr_conf; k++) {
                    res.set(put_out) = tab_1d(k);
                    put_out.inc(after); // Move to next element along dim
                }

                // Move to next position in "after" dimensions
                demarre_conf.inc();
                demarre_coef.inc();
            }

            // Move to next position in "before" dimensions
            index_base.inc();
            loop_before_coef.inc(1, dim + 1);
            loop_before_out.inc(1, dim + 1);
        }

        // Return the transformed array (input left unchanged)
        return res;
    }

    /**
     * @brief Perform complete multi-dimensional inverse spectral transform
     * @param in_conf Desired dimensions of output in physical space (collocation points)
     * @param cof Input array of spectral coefficients
     * @return Array of function values at collocation points
     *
     * This is the main entry point for multi-dimensional inverse transforms. It applies
     * 1D inverse transforms sequentially along each dimension, converting a full
     * multi-dimensional spectral representation back to physical space values.
     *
     * The transformation order (dimension 0, then 1, then 2, ...) is arbitrary since
     * the transforms commute for separable bases. The function handles:
     * - Arbitrary dimensionality (2D, 3D, or higher)
     * - Different basis types in each dimension
     * - Dimension size changes in each direction
     *
     * Mathematical operation: Given spectral coefficients c_{k₁,...,kₙ}, compute
     * f(x₁^{i₁},...,xₙ^{iₙ}) = Σ_{k₁,...,kₙ} c_{k₁,...,kₙ} Π_j φⱼ_{kⱼ}(xⱼ^{iⱼ})
     * where xⱼ^{iⱼ} are the collocation points in dimension j.
     *
     * Example (3D Chebyshev):
     * - Input: cof with dimensions [nr_coef, nθ_coef, nφ_coef]
     * - Output: values at [nr_conf, nθ_conf, nφ_conf] collocation points
     * - Process: Transform r-direction, then θ-direction, then φ-direction
     *
     * @pre cof.get_ndim() == in_conf.get_ndim() == this->ndim
     * @pre All dimensions in in_conf are positive
     * @pre Basis functions are properly defined for all dimensions
     *
     * @post Output has dimensions matching in_conf
     * @post Output contains function values at collocation points
     *
     * @note Creates temporary arrays during transformation - original cof unchanged
     * @note Memory efficient: only one working copy exists at a time
     *
     * Performance considerations:
     * - Time complexity: O(N^{d+1}) for d dimensions with N points per dimension
     * - Transform order doesn't affect result but may affect cache efficiency
     * - Each dimension's transform is cache-friendly (stride-1 access in innermost loop)
     */
    Array<double> Base_spectral::coef_i(const Dim_array& in_conf, const Array<double>& cof) const
    {
        // Apply the inverse transform one dimension at a time. The first pass
        // reads `cof` directly (no workspace copy of the input); each later
        // pass move-assigns its fresh result back into `cur`.
        Array<double> cur = coef_i_dim(0, in_conf(0), cof);
        for (int d = 1; d < ndim; d++)
            cur = coef_i_dim(d, in_conf(d), cur);

        return cur;
    }

} // namespace Kadath
