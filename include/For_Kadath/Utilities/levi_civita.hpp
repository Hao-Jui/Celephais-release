/*
 * This file is part of the KADATH library.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "For_Kadath/Kadath_point_h/kadath.hpp"

/**
 * \addtogroup fields
 * @{
 */

namespace Kadath
{

/**
 * @brief Builds the flat Cartesian Levi-Civita permutation SYMBOL \f$\hat\epsilon_{ijk}\f$.
 *
 * The returned rank-3 covariant tensor holds the totally antisymmetric permutation
 * symbol on the 3D Cartesian basis:
 * \f[
 *   \hat\epsilon_{ijk} =
 *     \begin{cases}
 *       +1 & (i,j,k) \text{ an even permutation of } (1,2,3) \\
 *       -1 & (i,j,k) \text{ an odd permutation of } (1,2,3) \\
 *        0 & \text{any repeated index}
 *     \end{cases}
 * \f]
 * Its value is constant across every domain and collocation point. The tensor is
 * decomposed on the standard spectral bases (`std_base`) so it can be injected as a
 * DSL constant and used directly inside `add_def` strings, e.g.
 * @code
 *   syst.add_cst("eps", build_levi_civita_symbol(space));
 *   syst.add_def("curl^i = eps^ijk * D_j V_k");   // flat curl of V
 * @endcode
 *
 * @warning This is the coordinate SYMBOL \f$\hat\epsilon\f$, not the physical
 *          Levi-Civita TENSOR. For the conformally-flat XCTS metric
 *          \f$\gamma_{ij} = P^4 f_{ij}\f$ the physical tensor carries the metric
 *          volume weight \f$\sqrt{\det\gamma} = P^6\f$:
 *          \f[
 *            \epsilon_{ijk} = P^6\,\hat\epsilon_{ijk}, \qquad
 *            \epsilon^{ijk} = P^{-6}\,\hat\epsilon^{ijk}.
 *          \f]
 *          Apps must supply the \f$P^{\pm 6}\f$ factor explicitly; the builder
 *          provides only the bare symbol \f$\hat\epsilon\f$.
 *
 * @param space the computational \c Space (must be 3-dimensional).
 * @return covariant rank-3 \c Tensor holding \f$\hat\epsilon_{ijk}\f$ on the
 *         Cartesian basis.
 */
inline Tensor build_levi_civita_symbol(const Space& space)
{
    assert(space.get_ndim() == 3);

    Base_tensor basis(space, CARTESIAN_BASIS);
    Tensor eps(space, 3, COV, basis);

    // Totally antisymmetric: nonzero only for permutations of (1,2,3).
    // Setting all 27 components from the permutation sign guarantees exact
    // antisymmetry (eps_ijk = -eps_jik = ...), with repeated indices -> 0.
    auto permutation_sign = [](int i, int j, int k) -> double {
        if (i == j || j == k || i == k)
            return 0.;
        // (i-j)(j-k)(k-i)/2 yields +1 for even and -1 for odd permutations of
        // any three distinct values drawn from {1,2,3}.
        return 0.5 * (i - j) * (j - k) * (k - i);
    };

    for (int i = 1; i <= 3; ++i)
        for (int j = 1; j <= 3; ++j)
            for (int k = 1; k <= 3; ++k) {
                eps.set(i, j, k) = permutation_sign(i, j, k);
                // Decompose each component on the SCALAR standard base. The
                // components are genuine constants (+1/-1/0); the rank-3
                // Tensor::std_base() would instead build a product of Cartesian
                // *vector-component* bases (x-odd, y-odd, z-odd symmetries),
                // which cannot represent a constant and distorts it into a
                // spatially varying field. A constant on the scalar base is
                // exact at every collocation point.
                eps.set(i, j, k).std_base();
            }

    return eps;
}

} // namespace Kadath

/**
 * @}
 */
