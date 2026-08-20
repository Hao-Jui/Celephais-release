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
    Ope_dr::Ope_dr(const System_of_eqs* zesys, Ope_eq* target) : Ope_eq(zesys, target->get_dom(), 1)
    {
        parts[0].reset(target);
    }

    Ope_dr::~Ope_dr() {}

    Term_eq Ope_dr::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);
        Term_eq target(parts[0]->action());
        return target.val_t->get_space().get_domain(dom)->dr_term_eq(target);

        /*Term_eq target (parts[0]->action()) ;
        // Check it is a tensor
        if (target.type_data != TERM_T) {
            KADATH_THROW("Ope_dr only defined with respect for a tensor");
        }

        // The value
        Tensor resval (*target.val_t, false) ;

        for (int i=0 ; i<target.val_t->get_n_comp() ; i++) {
            Array<int> ind (target.val_t->indices(i)) ;
            const Val_domain& value ((*target.val_t)(ind)(dom)) ;
            if (value.check_if_zero())
                resval.set(ind).set_domain(dom).set_zero() ;
            else
                resval.set(ind).set_domain(dom) = value.get_domain()->der_r(value) ;
        }

        if (target.der_t!=nullptr) {
            Tensor resder (*target.der_t, false) ;
            for (int i=0 ; i<target.der_t->get_n_comp() ; i++) {
                Array<int> ind (target.der_t->indices(i)) ;
                const Val_domain& value ((*target.der_t)(ind)(dom)) ;
                if (value.check_if_zero())
                    resder.set(ind).set_domain(dom).set_zero() ;
                else
                    resder.set(ind).set_domain(dom) = value.get_domain()->der_r(value) ;
                }
            Term_eq res (dom, resval, resder) ;
            return res ;
        }
        else {
            Term_eq res (dom, resval) ;
            return res ;
        }*/
    }
} // namespace Kadath
