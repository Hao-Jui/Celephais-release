/*
    Copyright 2018 Philippe Grandclement

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

#include <sstream>
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/homothetic_polar.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

namespace Kadath
{
    // Tensorial parts :
    Tensor Domain_polar_shell_outer_homothetic::import(int numdom, int bound, int n_ope, const Array<int>& zedoms,
                                                       Tensor** parts) const
    {
        if (parts[0]->get_valence() != 0) {
            for (int i = 0; i < n_ope; i++) {
                if (parts[i]->get_basis().get_basis(zedoms(i)) != CARTESIAN_BASIS) {
                    KADATH_THROW("Import must be called with a Cartesian tensorial basis");
                }
            }
        }

        Tensor res(*parts[0], false);

        if (res.get_valence() == 0) {
            res.set().set_domain(numdom).allocate_conf();
        } else {
            for (int nc = 0; nc < res.get_n_comp(); nc++)
                res.set(nc).set_domain(numdom).allocate_conf();
        }

        // Loop on the points of the boundary:
        Val_domain xx(get_cart(1));
        Val_domain yy(get_cart(2));

        int index_r;
        switch (bound) {
            case INNER_BC:
                index_r = 0;
                break;
            case OUTER_BC:
                index_r = nbr_points(0) - 1;
                break;
            default:
                KADATH_THROW("Unknown boundary in Domain_polar_shell_outer_homothetic::import");
        }

        Index pos(get_nbr_points());
        Index pos_bound(get_nbr_points());

        for (int j = 0; j < nbr_points(1); j++) {
            // Indices on the shell boundary
            pos_bound.set(0) = index_r;
            pos_bound.set(1) = j;

            // Absolute coordinates
            Point MM(2);
            MM.set(1) = xx(pos_bound);
            MM.set(2) = yy(pos_bound);

            // In which other domain is it?
            bool found = false;
            int current = 0;
            while ((current < n_ope) && (!found)) {
                if (parts[0]->get_space().get_domain(zedoms(current))->is_in(MM))
                    found = true;
                else
                    current++;
            }
            if (!found) {
                std::ostringstream oss;
                oss << "Point " << MM
                     << " not found in other domains, for "
                        "Domain_polar_shell_outer_homothetic::import"
                     << endl;
                KADATH_THROW(oss.str());
            }

            // Convert to numerical coordinates of the other domain
            Point num(parts[0]->get_space().get_domain(zedoms(current))->absol_to_num(MM));

            // Now loop on the components:
            for (int nc = 0; nc < res.get_n_comp(); nc++) {
                const Scalar& cur = (res.get_valence() == 0) ? (*parts[current])() : (*parts[current])(nc);
                double val = cur(zedoms(current)).check_if_zero() ? 0 : cur.val_point(MM, -1);
                // Loop on radius:
                for (int i = 0; i < nbr_points(0); i++) {
                    pos.set(0) = i;
                    pos.set(1) = j;
                    if (res.get_valence() == 0) {
                        res.set().set_domain(numdom).set(pos) = val;
                    } else {
                        res.set(nc).set_domain(numdom).set(pos) = val;
                    }
                }
            }
        }

        // Assert a std_base:
        res.set_basis(numdom) = CARTESIAN_BASIS; // Output in cartesian basis
        res.std_base();

        return res;
    }
} // namespace Kadath
