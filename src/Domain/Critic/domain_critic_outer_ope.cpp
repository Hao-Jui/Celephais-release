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
#include "For_Kadath/Domain/critic.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
namespace Kadath
{
    int div_x_1d(int, Array<double>&);

    Val_domain Domain_critic_outer::der_partial_var(const Val_domain& so, int which_var) const
    {

        switch (which_var) {
            case 0:
                return so.der_var(1) / (1 - xlim) * 2.;
                break;
            case 1:
                return so.der_var(2);
                break;

            default:
                KADATH_THROW("Unknown variable in Domain_critic_outer::der_partial_var");
        }
    }
} // namespace Kadath
