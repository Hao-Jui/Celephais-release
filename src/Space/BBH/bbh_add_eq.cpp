/*
    Copyright 2020 Philippe Grandclement

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

#include "For_Kadath/Space/bbh.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{

    void Space_bbh::add_matching(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_matching(4, CHI_ONE_BC, name, nused, pused);
        sys.add_eq_matching(5, ETA_PLUS_BC, name, nused, pused);
        sys.add_eq_matching(6, ETA_PLUS_BC, name, nused, pused);
        sys.add_eq_matching(7, CHI_ONE_BC, name, nused, pused);
    }

    void Space_bbh::add_matching(System_of_eqs& sys, const char* name, const List_comp& list)
    {
        add_matching(sys, name, list.get_ncomp(), list.get_pcomp());
    }

} // namespace Kadath
