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
#include "For_Kadath/Term_eq/term_eq.hpp"
namespace Kadath
{
    Ope_add::Ope_add(const System_of_eqs* zesys, Ope_eq* aa, Ope_eq* bb) : Ope_eq(zesys, aa->get_dom(), 2)
    {

        assert(aa->get_dom() == bb->get_dom());
        parts[0].reset(aa);
        parts[1].reset(bb);
    }

    Ope_add::~Ope_add() {}

    Term_eq Ope_add::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);
        ope_action_detail::OperandScratchLease lhs_scratch(syst);
        const Term_eq& lhs = parts[0]->action_operand(lhs_scratch.storage());
        ope_action_detail::OperandScratchLease rhs_scratch(syst);
        const Term_eq& rhs = parts[1]->action_operand(rhs_scratch.storage());
        return lhs + rhs;
    }

    bool Ope_add::action_into(Term_eq& result) const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);
        ope_action_detail::OperandScratchLease lhs_scratch(syst);
        const Term_eq& lhs = parts[0]->action_operand(lhs_scratch.storage());
        ope_action_detail::OperandScratchLease rhs_scratch(syst);
        const Term_eq& rhs = parts[1]->action_operand(rhs_scratch.storage());
        if (result.try_write_sum_action_result(lhs, rhs, false))
            return true;

        Term_eq computed(lhs + rhs);
        if (result.try_write_action_result(computed))
            return true;
        result = computed;
        return false;
    }

    void Ope_add::action_into_scratch(std::optional<Term_eq>& result) const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);
        ope_action_detail::OperandScratchLease lhs_scratch(syst);
        const Term_eq& lhs = parts[0]->action_operand(lhs_scratch.storage());
        ope_action_detail::OperandScratchLease rhs_scratch(syst);
        const Term_eq& rhs = parts[1]->action_operand(rhs_scratch.storage());
        if (result.has_value() && result->try_write_sum_action_result(lhs, rhs, false))
            return;
        if (!result.has_value()) {
            result.emplace(lhs + rhs);
            return;
        }
        Term_eq computed(lhs + rhs);
        if (!result->try_write_action_result(computed)) {
            result.reset();
            result.emplace(computed);
        }
    }
} // namespace Kadath
