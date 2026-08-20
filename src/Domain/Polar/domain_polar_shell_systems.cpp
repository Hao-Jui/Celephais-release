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
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
namespace Kadath
{
    void Domain_polar_shell::find_other_dom(int dom, int bound, int& other_dom, int& other_bound) const
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
                KADATH_THROW("Unknown boundary case in Domain_polar_shell::find_other_dom");
        }
    }

    double Domain_polar_shell::val_boundary(int bound, const Val_domain& so, const Index& pos_cf) const
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
                    KADATH_THROW("Unknown boundary type in Domain_polar_shell::val_boundary");
            }
            return res;
        }
    }
} // namespace Kadath
