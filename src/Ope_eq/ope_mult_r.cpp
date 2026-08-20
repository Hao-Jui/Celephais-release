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
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Ope_mult_r::Ope_mult_r(const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1)
    {
        parts[0].reset(target);
    }

    Ope_mult_r::~Ope_mult_r() {}

    Term_eq Ope_mult_r::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        Term_eq target(parts[0]->action());
        return target.val_t->get_space().get_domain(dom)->mult_r_term_eq(target);
    }
} // namespace Kadath
