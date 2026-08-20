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

#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

namespace Kadath
{
    Term_eq Domain_bispheric_chi_first::derive_flat_cart(int type_der, char ind_der, const Term_eq& so,
                                                         const Metric* manipulator) const
    {

        bool doder = (so.der_t == nullptr) ? false : true;
        if ((type_der == CON) && (manipulator->p_met_con[num_dom]->der_t == nullptr))
            doder = false;

        assert((type_der == COV) || (type_der == CON));
        int val_res = so.val_t->get_valence() + 1;

        bool doname = true;
        if (so.val_t->get_valence() > 0)
            if (!so.val_t->is_name_affected())
                doname = false;

        Array<int> type_ind(val_res);
        type_ind.set(0) = COV;
        for (int i = 1; i < val_res; i++)
            type_ind.set(i) = so.val_t->get_index_type(i - 1);

        // Need for summation ?
        bool need_sum = false;
        if (doname)
            for (int i = 1; i < val_res; i++)
                if (ind_der == so.val_t->get_name_ind()[i - 1])
                    need_sum = true;

        // Tensor for val
        Base_tensor basis(so.val_t->get_space(), CARTESIAN_BASIS);
        Tensor auxi_val(so.val_t->get_space(), val_res, type_ind, basis);

        // Set the names of the indices :
        if (doname) {
            auxi_val.set_name_affected();
            auxi_val.set_name_ind(0, ind_der);
            for (int i = 1; i < val_res; i++)
                auxi_val.set_name_ind(i, so.val_t->get_name_ind()[i - 1]);
        }

        // Loop on the components :
        Index pos_auxi(auxi_val);
        Index pos_so(*so.val_t);
        do {
            for (int i = 0; i < val_res - 1; i++)
                pos_so.set(i) = pos_auxi(i + 1);
            auxi_val.set(pos_auxi).set_domain(num_dom) = (*so.val_t)(pos_so)(num_dom).der_abs(pos_auxi(0) + 1);
        } while (pos_auxi.inc());

        if (!doder) {
            // No need for derivative :
            Term_eq auxi(num_dom, auxi_val);
            // If derive contravariant : manipulate first indice :
            if (type_der == CON)
                manipulator->manipulate_ind(auxi, 0);

            if (!need_sum)
                return auxi;
            else
                return Term_eq(num_dom, auxi.val_t->do_summation_one_dom(num_dom));
        } else {
            // Need to compute the derivative :
            // Tensor for der
            Tensor auxi_der(so.val_t->get_space(), val_res, type_ind, basis);
            // Set the names of the indices :
            auxi_der.set_name_affected();
            auxi_der.set_name_ind(0, ind_der);
            for (int i = 1; i < val_res; i++)
                auxi_der.set_name_ind(i, so.der_t->get_name_ind()[i - 1]);

            // Loop on the components :
            Index pos_auxi_der(auxi_der);
            do {
                for (int i = 0; i < val_res - 1; i++)
                    pos_so.set(i) = pos_auxi_der(i + 1);
                auxi_der.set(pos_auxi_der).set_domain(num_dom) =
                    (*so.der_t)(pos_so)(num_dom).der_abs(pos_auxi_der(0) + 1);
            } while (pos_auxi_der.inc());

            // Need for derivative :
            Term_eq auxi(num_dom, auxi_val, auxi_der);

            // If derive contravariant : manipulate first indice :
            if (type_der == CON)
                manipulator->manipulate_ind(auxi, 0);

            if (!need_sum)
                return auxi;
            else
                return Term_eq(num_dom, auxi.val_t->do_summation_one_dom(num_dom),
                               auxi.der_t->do_summation_one_dom(num_dom));
        }
    }
} // namespace Kadath
