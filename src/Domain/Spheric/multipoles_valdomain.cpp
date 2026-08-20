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

#include "For_Kadath/Array/headcpp.hpp"

#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
namespace Kadath
{
    void coef_1d(int, Array<double>&);
    double integral_1d(int, const Array<double>&);

    // Computation norm Legendre associated
    Array<double> legendre_norme(int, int);
    Array<double> mat_leg_even(int, int);
    Array<double> mat_leg_odd(int, int);

    double Domain_shell::multipoles_sym(int k, int j, int bound, const Val_domain& so,
                                        const Array<double>& passage) const
    {

        assert((bound == INNER_BC) || (bound == OUTER_BC));

        Index pos(passage.dimensions);
        Index pos_bound(nbr_coefs);

        double alm = 0;

        int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
        pos.set(0) = mm;
        pos_bound.set(2) = k;
        if (mm % 2 == 0) {
            // Case m even
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1); i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, so, pos_bound);
            }
        } else {
            // Case m odd
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1) - 1; i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, so, pos_bound);
            }
        }
        return alm;
    }

    double Domain_shell::multipoles_asym(int k, int j, int bound, const Val_domain& so,
                                         const Array<double>& passage) const
    {

        assert((bound == INNER_BC) || (bound == OUTER_BC));

        Index pos(passage.dimensions);
        Index pos_bound(nbr_coefs);

        double alm = 0;

        int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
        pos.set(0) = mm;
        pos_bound.set(2) = k;
        if (mm % 2 == 0) {
            // Case m even
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1) - 1; i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, so, pos_bound);
            }
        } else {
            // Case m odd
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1) - 1; i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, so, pos_bound);
            }
        }
        return alm;
    }
} // namespace Kadath
