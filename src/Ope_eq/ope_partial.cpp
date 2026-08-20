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
 *   2026-08-06  RAII/span modernization.
 */

#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    Ope_partial::Ope_partial(const System_of_eqs* zesys, char ind, Ope_eq* target)
        : Ope_eq(zesys, target->get_dom(), 1), ind_der(ind)
    {
        parts[0].reset(target);
    }

    Ope_partial::~Ope_partial() {}

    Term_eq Ope_partial::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);
        Term_eq target(parts[0]->action());
        // Check it is a tensor
        if (target.type_data != TERM_T) {
            KADATH_THROW("Ope_partial only defined with respect for a tensor");
        }

        return (partial(target, ind_der));
    }
} // namespace Kadath
