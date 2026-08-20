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
#include "For_Kadath/Param/param.hpp"
#include <optional>
namespace Kadath
{
    Ope_user_bin::Ope_user_bin(const System_of_eqs* zesys, Term_eq (*zeope)(const Term_eq&, const Term_eq&, Param*),
                               Param* parso, Ope_eq* p1, Ope_eq* p2)
        : Ope_eq(zesys, p1->get_dom(), 2), pope(zeope)
    {
        par = parso;
        parts[0].reset(p1);
        parts[1].reset(p2);
    }

    Ope_user_bin::~Ope_user_bin() {}

    Term_eq Ope_user_bin::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        std::optional<Term_eq> p1_storage;
        std::optional<Term_eq> p2_storage;
        const Term_eq& p1 = parts[0]->action_operand(p1_storage);
        const Term_eq& p2 = parts[1]->action_operand(p2_storage);
        return pope(p1, p2, par);
    }
} // namespace Kadath
