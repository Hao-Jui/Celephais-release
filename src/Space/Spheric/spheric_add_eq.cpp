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
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "assert.h"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{
    void Space_spheric::add_inner_bc(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        // BC in domain 5 and 6
        sys.add_eq_bc(sys.get_dom_min(), INNER_BC, name, nused, pused);
    }

    void Space_spheric::add_outer_bc(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(sys.get_dom_max(), OUTER_BC, name, nused, pused);
    }

    void Space_spheric::add_eq(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd <= sys.get_dom_max(); dd++)
            sys.add_eq_inside(dd, name, nused, pused);
    }

    void Space_spheric::add_matching(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd < sys.get_dom_max(); dd++)
            sys.add_eq_matching(dd, OUTER_BC, name, nused, pused);
    }

    void Space_spheric::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der, int nused,
                               Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd < sys.get_dom_max(); dd++) {
            sys.add_eq_inside(dd, eq, nused, pused, rac);
            sys.add_eq_matching(dd, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(dd, OUTER_BC, rac_der, nused, pused, rac);
        }
        sys.add_eq_inside(sys.get_dom_max(), eq, nused, pused, rac);
    }

    void Space_spheric::add_eq_full(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd <= sys.get_dom_max(); dd++)
            sys.add_eq_full(dd, name, nused, pused);
    }

    void Space_spheric::add_eq_one_side(System_of_eqs& sys, const char* name, const char* rac, int nused,
                                        Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd < sys.get_dom_max(); dd++) {
            sys.add_eq_one_side(dd, name, nused, pused);
            sys.add_eq_matching(dd, OUTER_BC, rac, nused, pused, rac);
        }
        sys.add_eq_one_side(sys.get_dom_max(), name, nused, pused);
    }

    void Space_spheric::add_eq_one_side(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd <= sys.get_dom_max(); dd++)
            sys.add_eq_one_side(dd, name, nused, pused);
    }

    void Space_spheric::add_eq_ori(System_of_eqs& sys, const char* name)
    {

        Index pos(domains[0]->get_nbr_points());
        char auxi[LMAX];
        trim_spaces(auxi, name);
        sys.add_eq_val(0, auxi, pos);
    }

    void Space_spheric::add_eq_int_volume(System_of_eqs& sys, const char* nom)
    {

        // Get the lhs and rhs
        char p1[LMAX];
        char p2[LMAX];
        bool indic = sys.is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {
            // Construction of the equation
            sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(nbr_domains + 1));

            // Affectation of the intregrale parts
            for (int d = 0; d < nbr_domains; d++)
                sys.eq_int[sys.neq_int]->set_part(d, sys.give_ope(d, p1));
            // Affectation of the second member (constant value)
            sys.eq_int[sys.neq_int]->set_part(nbr_domains, new Ope_minus(&sys, sys.give_ope(0, p2)));
            sys.neq_int++;
        }
        sys.nbr_conditions = -1;
    }

    void Space_spheric::add_eq_mode_mid(System_of_eqs& sys, const char* name, int itarget, int jtarget, int ktarget)
    {

        Index pos_cf(domains[0]->get_nbr_coefs());
        pos_cf.set(0) = itarget;
        pos_cf.set(1) = jtarget;
        pos_cf.set(2) = ktarget;
        double value = 1.;
        char auxi[LMAX];
        trim_spaces(auxi, name);
        sys.add_eq_val_mode(1, auxi, pos_cf, value);
    }
} // namespace Kadath
