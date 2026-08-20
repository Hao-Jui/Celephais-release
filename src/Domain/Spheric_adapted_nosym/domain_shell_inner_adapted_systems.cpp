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
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

namespace Kadath
{
    void Domain_shell_inner_adapted_nosym::find_other_dom(int dom, int bound, int& other_dom, int& other_bound) const
    {

        switch (bound) {
            case INNER_BC:
                other_dom = dom - 1;
                other_bound = OUTER_BC;
                break;
            case OUTER_BC:
                other_dom = dom + 1;
                other_bound = INNER_BC;
                break;
            default:
                KADATH_THROW("Unknown boundary case in Domain_shell_inner_adapted_nosym::find_other_dom");
        }
    }

    double Domain_shell_inner_adapted_nosym::val_boundary(int bound, const Val_domain& so, const Index& pos_cf) const
    {

        if (so.check_if_zero())
            return 0.;

        else {
            so.coef();
            double res = 0;
            Index copie_pos(pos_cf);
            switch (bound) {
                case INNER_BC:
                    for (int i = 0; i < nbr_coefs(0); i++) {
                        copie_pos.set(0) = i;
                        if (i % 2 == 0)
                            res += so.get_coef(copie_pos);
                        else
                            res -= so.get_coef(copie_pos);
                    }
                    break;
                case OUTER_BC:
                    for (int i = 0; i < nbr_coefs(0); i++) {
                        copie_pos.set(0) = i;
                        res += so.get_coef(copie_pos);
                    }
                    break;
                default:
                    KADATH_THROW("Unknown boundary type in Domain_shell_inner_adapted_nosym::val_boundary");
            }
            return res;
        }
    }

    int Domain_shell_inner_adapted_nosym::nbr_points_boundary(int bound, const Base_spectral& bb) const
    {

        if ((bound != INNER_BC) && (bound != OUTER_BC)) {
            KADATH_THROW("Unknown boundary in Domain_shell_inner_adapted_nosym::nbr_points_boundary");
        }

        // NONSYM unified COS / SIN: no parity sector distinction. Pick the
        // convention matching the SYM SIN_ODD branch (which is the dominant
        // scalar dispatch under SYM; SIN is its NONSYM analog under
        // l_quant = j vs 2j+1 collapse).
        int nbrj = nbr_points(1) - 1;

        return (nbrj * nbr_points(2) + 1);
    }

    void Domain_shell_inner_adapted_nosym::do_which_points_boundary(int bound, const Base_spectral& bb, Index** which_coef,
                                                              int start) const
    {

        int pos_which = start;
        Index pos(nbr_points);

        switch (bound) {
            case INNER_BC:
                pos.set(0) = 0;
                break;
            case OUTER_BC:
                pos.set(0) = nbr_points(0) - 1;
                break;
            default:
                KADATH_THROW("Unknown boundary in Domain_shell_inner_adapted_nosym::do_which_points_inside");
        }

        // NONSYM unified COS / SIN: no parity sector distinction. Mirror
        // SYM SIN_ODD branch (NONSYM SIN analog).
        int maxj = nbr_points(1);

        for (int k = 0; k < nbr_points(2); k++) {
            pos.set(2) = k;
            for (int j = 0; j < maxj; j++) {
                pos.set(1) = j;
                if ((k == 0) || (j != 0)) {
                    which_coef[pos_which] = new Index(pos);
                    pos_which++;
                }
            }
        }
    }
} // namespace Kadath
