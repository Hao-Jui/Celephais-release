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

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
namespace Kadath
{
    int mult_x_1d(int, Array<double>&);

    Val_domain Domain_shell_surr::mult_r(const Val_domain& so) const
    {
        Val_domain res(so * get_radius());
        res.base = so.base;
        return res;
    }

    Val_domain Domain_shell_surr::div_r(const Val_domain& so) const
    {
        Val_domain res(so / get_radius());
        res.base = so.base;
        return (res);
    }

    Val_domain Domain_shell_surr::der_r(const Val_domain& so) const
    {
        return (-so.der_var(1) / alpha / get_radius() / get_radius());
    }
} // namespace Kadath
