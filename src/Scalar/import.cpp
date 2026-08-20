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
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Array/point.hpp"

namespace Kadath
{
    void Scalar::import(const Scalar& so)
    {

        set_in_conf();
        allocate_conf();

        // Loop on the domains :
        for (int dd = 0; dd < get_nbr_domains(); dd++) {
            Index pos(get_space().get_domain(dd)->get_nbr_points());
            Point xx(ndim);
            do {
                for (int i = 1; i <= ndim; i++)
                    xx.set(i) = (get_space().get_domain(dd)->get_cart(i))(pos);

                // Provisory :

                if ((dd != get_nbr_domains() - 1) || (pos(0) != get_space().get_domain(dd)->get_nbr_points()(0) - 1))
                    set_domain(dd).set(pos) = so.val_point(xx);
            } while (pos.inc());
        }
    }
} // namespace Kadath
