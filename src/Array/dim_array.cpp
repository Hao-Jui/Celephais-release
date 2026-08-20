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

#include "assert.h"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/dim_array.hpp"
namespace Kadath
{

    // Output
    ostream& operator<<(ostream& o, const Dim_array& so)
    {
        o << "(";
        for (int i = 0; i < so.ndim - 1; i++)
            o << so.nbr[i] << ", ";
        o << so.nbr[so.ndim - 1] << ")";
        return o;
    }

    // Comparison operator
    bool operator==(const Dim_array& a, const Dim_array& b)
    {
        bool res = (a.ndim == b.ndim) ? true : false;
        if (res)
            for (int i = 0; i < a.ndim; i++)
                if (a.nbr[i] != b.nbr[i])
                    res = false;
        return res;
    }

    // Anti-comparison operator
    bool operator!=(const Dim_array& a, const Dim_array& b)
    {

        return !(a == b);
    }
} // namespace Kadath
