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
 * @file coef.cpp
 * @brief Multi-dimensional forward spectral transformations
 * @author Philippe Grandclement
 * @date 2017
 *
 * This file implements multi-dimensional forward spectral transformations by
 * applying 1D forward transforms sequentially along each dimension. This is the
 * complementary operation to the inverse transforms in base_spectral_coef_i.cpp,
 * converting physical space values (at collocation points) to spectral space
 * coefficients.
 *
 * The implementation uses dimensional separation: transform one dimension at a
 * time while treating other dimensions as independent batch operations. This
 * approach exploits the tensor product structure of multi-dimensional spectral
 * bases and allows efficient use of optimized 1D transforms.
 *
 * Key features:
 * - Supports mixed spectral bases (different basis per dimension)
 * - Handles dimension size changes (collocation points → coefficients)
 * - Efficient memory layout with stride optimization
 * - Reverse dimension ordering (high to low) for optimal accuracy
 *
 * Mathematical background: Given function values f(x₁^{i₁},...,xₙ^{iₙ}) at
 * collocation points, compute spectral coefficients c_{k₁,...,kₙ} via:
 *
 * c_{k₁,...,kₙ} = ∫...∫ f(x₁,...,xₙ) φ₁_{k₁}(x₁)...φₙ_{kₙ}(xₙ) w(x₁)...w(xₙ) dx₁...dxₙ
 *
 * where φᵢ_{kᵢ} are orthogonal basis functions and w are weight functions. The
 * quadrature is performed via collocation point evaluation and appropriate weights.
 *
 * Transform ordering: Dimensions are processed from highest index to lowest
 * (ndim-1, ndim-2, ..., 1, 0). This reverse ordering often provides better
 * numerical accuracy in practice, particularly for problems with special structure
 * in the radial (outermost) dimension.
 */

#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/array.hpp"

namespace Kadath
{

    // Forward declaration of 1D forward transform dispatcher
    void coef_1d(int, Array<double>&);

    /**
     * @brief Perform forward spectral transform along a single dimension
     * @param dim Dimension index to transform (0-based)
     * @param nbr_coef Number of spectral coefficients desired in output along dimension dim
     * @param inout Input/output array (modified in-place, replaced with new array)
     *
     * This function transforms physical space values (at collocation points) to
     * spectral coefficients along one specified dimension while preserving all other
     * dimensions. It is the forward counterpart to coef_i_dim().
     *
     * Algorithm overview:
     * 1. Create output array with new size in dimension dim
     * 2. Compute stride patterns (before/after the transformed dimension)
     * 3. Loop over all "before" dimensions (higher indices)
     * 4. For each position, determine the appropriate 1D basis type
     * 5. Loop over all "after" dimensions (lower indices)
     * 6. Extract 1D slice of collocation values along dimension dim
     * 7. Apply 1D forward transform to obtain spectral coefficients
     * 8. Store coefficients in output array
     * 9. Replace input pointer with new output array
     *
     * Memory layout:
     * - "after" dimensions: stride = 1 (innermost, varying fastest)
     * - Transformed dimension: stride = after
     * - "before" dimensions: stride = after * size(dim) (outermost)
     *
     * The transformation converts nbr_conf collocation point values to nbr_coef
     * spectral coefficients. Typically nbr_coef ≤ nbr_conf (coefficient compression),
     * though the code handles any relationship.
     *
     * @pre inout must point to a valid Array<double> with ndim dimensions
     * @pre 0 <= dim < ndim
     * @pre nbr_coef > 0
     * @pre Basis functions must be properly defined for dimension dim
     * @pre Input contains function values at appropriate collocation points
     *
     * @post Input array is left unchanged
     * @post Output dimension sizes: same except size(dim) = nbr_coef
     * @post Collocation values along dim are converted to spectral coefficients
     *
     * @return a fresh array holding the coefficients along dimension dim
     * @note Basis type can vary with position in "before" dimensions
     * @note This is the mathematical inverse of coef_i_dim() (up to numerical precision)
     */
    Array<double> Base_spectral::coef_dim(int dim, int nbr_coef, const Array<double>& inout) const
    {
        // Setup output dimensions (same as input except for dimension dim)
        Dim_array res_out(inout.get_dimensions());
        res_out.set(dim) = nbr_coef;
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

        // Current number of collocation points in dimension dim
        int nbr_conf = inout.get_size(dim);

        // Determine working buffer size (max of input and output sizes)
        int nbr = (nbr_coef > nbr_conf) ? nbr_coef : nbr_conf;

        // Index objects for navigating multi-dimensional arrays
        Index index_base(bases_1d[dim]->get_dimensions()); // Position in basis array
        Index demarre_conf(inout.get_dimensions());        // Starting position in input (conf)
        Index demarre_coef(res_out);                       // Starting position in output (coef)
        Index loop_before_conf(inout.get_dimensions());    // Loop counter for "before" dims (input)
        Index loop_before_out(res_out);                    // Loop counter for "before" dims (output)
        Index lit_in(inout.get_dimensions());              // Read position in input
        Index put_out(res_out);                            // Write position in output

        // Working buffer for 1D transforms
        Array<double> tab_1d(nbr);

        // Loop over all "before" dimensions (outermost loop)
        for (int i = 0; i < before; i++) {
            demarre_conf = loop_before_conf;
            demarre_coef = loop_before_out;

            // Determine which 1D basis to use for this position
            int base = (*bases_1d[dim])(index_base);

            // Loop over all "after" dimensions (inner loop - batch of 1D transforms)
            for (int j = 0; j < after; j++) {
                // Extract 1D slice of collocation values along dimension dim
                lit_in = demarre_conf;
                for (int k = 0; k < nbr_conf; k++) {
                    tab_1d.set(k) = inout(lit_in);
                    lit_in.inc(after); // Move to next collocation point along dim
                }

                // Apply 1D forward spectral transform
                // Converts collocation values → spectral coefficients
                coef_1d(base, tab_1d);

                // Store computed coefficients in output array
                put_out = demarre_coef;
                for (int k = 0; k < nbr_coef; k++) {
                    res.set(put_out) = tab_1d(k);
                    put_out.inc(after); // Move to next coefficient along dim
                }

                // Move to next position in "after" dimensions
                demarre_conf.inc();
                demarre_coef.inc();
            }

            // Move to next position in "before" dimensions
            index_base.inc();
            loop_before_conf.inc(1, dim + 1);
            loop_before_out.inc(1, dim + 1);
        }

        // Return the transformed array (input left unchanged)
        return res;
    }

    /**
     * @brief Perform complete multi-dimensional forward spectral transform
     * @param in_coef Desired dimensions of output in spectral space (number of coefficients)
     * @param coloc Input array of function values at collocation points
     * @return Array of spectral coefficients
     *
     * This is the main entry point for multi-dimensional forward transforms. It applies
     * 1D forward transforms sequentially along each dimension in reverse order
     * (from highest dimension to lowest), converting function values at collocation
     * points to a complete multi-dimensional spectral representation.
     *
     * Mathematical operation: Given function values f(x₁^{i₁},...,xₙ^{iₙ}), compute
     * spectral coefficients c_{k₁,...,kₙ} such that:
     *
     * f(x₁^{i₁},...,xₙ^{iₙ}) ≈ Σ_{k₁,...,kₙ} c_{k₁,...,kₙ} Π_j φⱼ_{kⱼ}(xⱼ^{iⱼ})
     *
     * The coefficients are computed via weighted inner products (Gauss quadrature)
     * using the collocation points and basis-specific weights.
     *
     * Transform ordering: Dimensions are processed in reverse order (ndim-1 down to 0)
     * rather than forward order. This choice can improve numerical accuracy for
     * problems where:
     * - The outermost (radial) dimension has special boundary conditions
     * - Higher dimensions have smoother variation
     * - Accumulated roundoff errors are minimized
     *
     * Example (3D spherical coordinates):
     * - Input: coloc with function values at [nr_conf, nθ_conf, nφ_conf] points
     * - Output: coefficients with dimensions [nr_coef, nθ_coef, nφ_coef]
     * - Process: Transform φ-direction first, then θ, then r (reverse order)
     *
     * @pre coloc.get_ndim() == in_coef.get_ndim() == this->ndim
     * @pre All dimensions in in_coef are positive
     * @pre Basis functions are properly defined for all dimensions
     * @pre coloc contains function values at appropriate collocation points
     *
     * @post Output has dimensions matching in_coef
     * @post Output contains spectral expansion coefficients
     * @post Applying coef_i(in_coef, result) should recover original coloc (roundtrip)
     *
     * @note Creates temporary arrays during transformation - original coloc unchanged
     * @note Memory efficient: only one working copy exists at a time
     * @note This is the mathematical inverse of coef_i() (up to numerical precision)
     *
     * Performance considerations:
     * - Time complexity: O(N^{d+1}) for d dimensions with N points per dimension
     * - Reverse ordering (high to low dimensions) is intentional for accuracy
     * - Each dimension's transform is cache-friendly (stride-1 access in innermost loop)
     * - Parallelization possible: all 1D transforms at same level are independent
     *
     * Numerical properties:
     * - For well-resolved smooth functions: ‖f - f_reconstructed‖ ≈ machine epsilon
     * - Spectral accuracy: exponential convergence for smooth functions
     * - Aliasing errors possible if function not properly resolved
     */
    Array<double> Base_spectral::coef(const Dim_array& in_coef, const Array<double>& coloc) const
    {
        // Apply the forward transform one dimension at a time. The first pass
        // reads `coloc` directly (no workspace copy of the input); each later
        // pass move-assigns its fresh result back into `cur`.
        // NOTE: Reverse order (high to low) for improved numerical accuracy.
        Array<double> cur = coef_dim(ndim - 1, in_coef(ndim - 1), coloc);
        for (int d = ndim - 2; d >= 0; d--)
            cur = coef_dim(d, in_coef(d), cur);

        return cur;
    }

} // namespace Kadath
