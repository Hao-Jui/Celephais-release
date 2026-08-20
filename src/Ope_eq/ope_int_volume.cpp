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
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
namespace Kadath
{
    Ope_int_volume::Ope_int_volume(const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1)
    {
        parts[0].reset(target);
    }

    Ope_int_volume::~Ope_int_volume() {}

    Term_eq Ope_int_volume::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        Term_eq target(parts[0]->action());
        if (target.get_type_data() == TERM_T) {
            return target.val_t->get_space().get_domain(dom)->integ_volume_term_eq(target);
        }
        else {
            assert(target.get_type_data() == TERM_D);
            Scalar auxival(syst->get_space());
            auxival.set_domain(dom) = target.get_val_d();
            auxival.set_domain(dom).std_base();
            Term_eq auxi(dom, auxival);
            if (target.has_der_d(0)) {
                Scalar auxider(syst->get_space());
                auxider.set_domain(dom) = target.get_der_d();
                auxider.set_domain(dom).std_base();
                auxi.set_der_t(auxider);
            }
            auxi.set_derivative_lane_count(target.get_derivative_lane_count());
            for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane) {
                if (!target.has_der_d(lane))
                    continue;
                Scalar lane_derivative(syst->get_space());
                lane_derivative.set_domain(dom) = target.get_der_d(lane);
                lane_derivative.set_domain(dom).std_base();
                auxi.set_der_t(lane, lane_derivative);
            }
            return auxi.val_t->get_space().get_domain(dom)->integ_volume_term_eq(auxi);
        }
    }
} // namespace Kadath
