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
#include <optional>
namespace Kadath
{
    Ope_pow::Ope_pow(const System_of_eqs* zesys, int nn, Ope_eq* target)
        : Ope_eq(zesys, target->get_dom(), 1), power(nn)
    {
        parts[0].reset(target);
    }

    Ope_pow::~Ope_pow() {}

    Term_eq Ope_pow::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);
        if (power == 0)
            return Term_eq(dom, 1., 0.);
        else {
            std::optional<Term_eq> operand_storage;
            const Term_eq& res_p0 = parts[0]->action_operand(operand_storage);
            // Check of type and valence :
            // int valence = res_p0.get_val_t().get_valence() ;
            // if (valence !=0) {
            if (res_p0.get_type_data() == TERM_T and res_p0.get_p_val_t()->get_valence() != 0) {
                KADATH_THROW("Ope_pow only defined for scalars");
            }
            return pow(res_p0, power);
        }
    }
} // namespace Kadath
