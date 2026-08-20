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
#include "For_Kadath/Domain/spheric_periodic.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{
    void Space_spheric_periodic::add_outer_bc(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(sys.get_dom_max(), OUTER_BC, name, nused, pused);
    }

    void Space_spheric_periodic::add_eq(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {

        // Nucleus
        for (int dd = sys.get_dom_min(); dd <= sys.get_dom_max(); dd++)
            if (dd == 0)
                sys.add_eq_order(dd, 1, name, nused, pused);
            else
                sys.add_eq_order(dd, 2, name, nused, pused);
    }

    void Space_spheric_periodic::add_eq_full(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd <= sys.get_dom_max(); dd++)
            sys.add_eq_order(dd, 0, name, nused, pused);
    }

    void Space_spheric_periodic::add_matching(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd < sys.get_dom_max(); dd++)
            sys.add_eq_matching(dd, OUTER_BC, name, nused, pused);
    }

    void Space_spheric_periodic::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                        int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd < sys.get_dom_max(); dd++) {
            if (dd == 0)
                sys.add_eq_order(dd, 1, eq, nused, pused);
            else
                sys.add_eq_order(dd, 2, eq, nused, pused);
            sys.add_eq_matching(dd, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(dd, OUTER_BC, rac_der, nused, pused, rac);
        }
        sys.add_eq_order(sys.get_dom_max(), 2, eq, nused, pused);
    }

    void Space_spheric_periodic::add_eq_inverted(System_of_eqs& sys, const char* eq, const char* rac,
                                                 const char* rac_der, int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd < sys.get_dom_max(); dd++) {
            sys.add_eq_order(dd, 2, eq, nused, pused);
            sys.add_eq_matching(dd, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(dd, OUTER_BC, rac_der, nused, pused, rac);
        }
        if (sys.get_dom_max() == nbr_domains - 1)
            sys.add_eq_order(sys.get_dom_max(), 1, eq, nused, pused);
        else
            sys.add_eq_order(sys.get_dom_max(), 2, eq, nused, pused);
    }

    void Space_spheric_periodic::add_eq_ori(System_of_eqs& sys, const char* name)
    {

        Index pos(domains[0]->get_nbr_points());
        char auxi[LMAX];
        trim_spaces(auxi, name);
        sys.add_eq_val(0, auxi, pos);
    }
} // namespace Kadath
