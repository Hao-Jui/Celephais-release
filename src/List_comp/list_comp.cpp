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
 *   2026-08-06  RAII/span modernization.
 */

/*
 * Purpose:
 * `List_comp` stores the tensor-component index selections used when adding
 * equations to `System_of_eqs`.
 *
 * It owns `ncomp` integer arrays, each of length `valence`, where each array
 * represents one component index tuple. `System_of_eqs::add_eq_*` overloads
 * accept a `List_comp` and forward `get_ncomp()` and `get_pcomp()` to the
 * equation objects (`Eq_inside`, `Eq_bc`, `Eq_matching`, ...).
 */

#include "assert.h"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/List_comp/list_comp.hpp"

namespace Kadath
{

    // Standard onsructor
    List_comp::List_comp(int nc, int val) : ncomp(nc), valence(val)
    {
        owned_pcomp.reserve(ncomp);
        pcomp.reserve(ncomp);
        for (int i = 0; i < ncomp; i++) {
            owned_pcomp.push_back(std::make_unique<Array<int>>(valence));
            pcomp.push_back(owned_pcomp.back().get());
        }
    }

    // Constructor by copy
    List_comp::List_comp(const List_comp& so) : ncomp(so.ncomp), valence(so.valence)
    {
        owned_pcomp.reserve(ncomp);
        pcomp.reserve(ncomp);
        for (int i = 0; i < ncomp; i++) {
            owned_pcomp.push_back(std::make_unique<Array<int>>(*so.pcomp[i]));
            pcomp.push_back(owned_pcomp.back().get());
        }
    }

    // Destructor
    List_comp::~List_comp() = default;

    // Read/write
    Array<int>* List_comp::set(int i)
    {
        assert(i >= 0);
        assert(i < ncomp);
        return pcomp[i];
    }

    // Read only
    Array<int>* List_comp::operator()(int i) const
    {
        assert(i >= 0);
        assert(i < ncomp);
        return pcomp[i];
    }
} // namespace Kadath
