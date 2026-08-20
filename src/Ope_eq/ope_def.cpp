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
namespace Kadath
{
    Ope_def::Ope_def(const System_of_eqs* zesys, Ope_eq* so, int valence, char* ind, Array<int>* ttype)
        : Ope_eq(zesys, so->get_dom(), 1)
    {
        parts[0].reset(so);

        Term_eq auxi(parts[0]->action());

        res = new Term_eq(auxi);
        if (res->val_t->get_valence() != valence) {
            KADATH_THROW("bad valence in definition");
        }

        // need that to enforce affectation of the indices and types
        if (res->der_t == nullptr)
            res->set_der_zero();

        // Now put the indices and their right type
        if (valence != 0) {
            res->val_t->set_name_affected();
            for (int i = 0; i < valence; i++)
                res->val_t->set_name_ind(i, ind[i]);
            res->val_t->set_index_type() = *ttype;

            res->der_t->set_name_affected();
            for (int i = 0; i < valence; i++)
                res->der_t->set_name_ind(i, ind[i]);
            res->der_t->set_index_type() = *ttype;
        }

        // Put back things to get the right order
        *res = auxi;
    }

    Ope_def::~Ope_def()
    {
        delete res;
    }

    Term_eq Ope_def::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        /*if (!called) {
            res = new Term_eq(parts[0]->action()) ;
            called = true ;
        }
        return *res ;*/
        KADATH_THROW("Should not call Ope_def::action");
    }

    Term_eq* Ope_def::get_res()
    {
        return res;
    }

    void Ope_def::compute_res()
    {
        if (ope_action_write_into_enabled())
            parts[0]->action_into(*res);
        else
            *res = parts[0]->action();
    }
} // namespace Kadath
