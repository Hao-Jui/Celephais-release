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
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/polar_periodic.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"

namespace Kadath
{

    Term_eq Domain_polar_periodic_shell::dtime_term_eq(const Term_eq& so) const
    {

        Tensor val_res(*so.val_t, false);

        for (int cmp = 0; cmp < so.val_t->get_n_comp(); cmp++) {
            Array<int> ind(val_res.indices(cmp));
            val_res.set(ind).set_domain(num_dom) = (*so.val_t)(ind)(num_dom).der_var(3) / (*ome_term_eq->val_d);
        }

        int valence = so.val_t->get_valence();
        // Put name indices :
        if (so.val_t->is_name_affected()) {
            val_res.set_name_affected();
            for (int ncmp = 0; ncmp < valence; ncmp++)
                val_res.set_name_ind(ncmp, so.val_t->get_name_ind()[ncmp]);
        }

        bool doder = ((so.der_t == nullptr) || (ome_term_eq->der_d == nullptr)) ? false : true;

        if (doder) {
            Tensor der_res(*so.der_t, false);
            for (int cmp = 0; cmp < so.val_t->get_n_comp(); cmp++) {
                Array<int> ind(val_res.indices(cmp));
                der_res.set(ind).set_domain(num_dom) = (*so.der_t)(ind)(num_dom).der_var(3) / (*ome_term_eq->val_d) -
                                                       (*ome_term_eq->der_d) * (*so.val_t)(ind)(num_dom).der_var(3) /
                                                           (*ome_term_eq->val_d) / (*ome_term_eq->val_d);
            }

            // Put name indices :
            if (so.der_t->is_name_affected()) {
                der_res.set_name_affected();
                for (int ncmp = 0; ncmp < valence; ncmp++)
                    der_res.set_name_ind(ncmp, so.der_t->get_name_ind()[ncmp]);
            }

            return Term_eq(num_dom, val_res, der_res);
        } else
            return Term_eq(num_dom, val_res);
    }

    Term_eq Domain_polar_periodic_shell::ddtime_term_eq(const Term_eq& so) const
    {

        return dtime_term_eq(dtime_term_eq(so));
    }

} // namespace Kadath
