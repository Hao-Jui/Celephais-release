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
#include "For_Kadath/Scalar/scalar.hpp"
#include "ope_tensor_unary_lane_helpers.hpp"
#include <optional>
namespace Kadath
{
    Ope_mult_rsint::Ope_mult_rsint(const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1)
    {
        parts[0].reset(target);
    }

    Ope_mult_rsint::~Ope_mult_rsint() {}

    Term_eq Ope_mult_rsint::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        std::optional<Term_eq> part_storage;
        const Term_eq& part = parts[0]->action_operand(part_storage);
        // Check it is a tensor
        if (part.type_data != TERM_T) {
            KADATH_THROW("Ope_mult_rsint only defined with respect to a tensor");
        }

        Term_eq target(part.val_t->get_space().get_domain(dom)->mult_r_term_eq(part));

        return ope_tensor_unary_lane_detail::apply_tensor_unary_operator(
            dom, target, "Ope_mult_rsint",
            [](const Val_domain& value) { return value.get_domain()->mult_sin_theta(value); });
    }
} // namespace Kadath
