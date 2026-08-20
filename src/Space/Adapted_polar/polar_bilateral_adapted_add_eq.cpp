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

#include "For_Kadath/Domain/adapted_polar.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{

    void Space_polar_bilateral_adapted::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                               int nused, Array<int>** pused) const
    {
        for (int dd = sys.get_dom_min(); dd < sys.get_dom_max(); dd++) {
            sys.add_eq_inside(dd, eq, nused, pused, rac);
            sys.add_eq_matching(dd, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(dd, OUTER_BC, rac_der, nused, pused, rac);
        }
        sys.add_eq_inside(sys.get_dom_max(), eq, nused, pused, rac);
    }

    void Space_polar_bilateral_adapted::add_eq_int_inf(System_of_eqs& sys, const char* nom)
    {

        // Check the last domain is of the right type :
        const Domain_polar_compact* pcomp = dynamic_cast<const Domain_polar_compact*>(domains[nbr_domains - 1]);
        if (pcomp == nullptr) {
            KADATH_THROW("add_eq_int_inf requires a compactified domain");
        }
        int dom = nbr_domains - 1;
        sys.eq_int_list.push_back(std::make_tuple(nom, dom, OUTER_BC));
        // Get the lhs and rhs
        char p1[LMAX];
        char p2[LMAX];
        bool indic = sys.is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {
            // Construction of the equation
            sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(1));

            // Affectation :
            sys.eq_int[sys.neq_int]->set_part(
                0, new Ope_sub(&sys, sys.give_ope(dom, p1, OUTER_BC), sys.give_ope(dom, p2, OUTER_BC)));
            sys.neq_int++;
        }
        sys.nbr_conditions = -1;
    }

    void Space_polar_bilateral_adapted::add_eq_int_volume(System_of_eqs& sys, int nz, const char* nom)
    {

        // Get the lhs and rhs
        char p1[LMAX];
        char p2[LMAX];
        bool indic = sys.is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {
            sys.eq_int_list.push_back(std::make_tuple(nom, 0, -1));
            // Construction of the equation
            sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(nz + 1));

            // Affectation of the intregrale parts
            for (int d = 0; d < nz; d++)
                sys.eq_int[sys.neq_int]->set_part(d, sys.give_ope(d, p1));
            // Affectation of the second member (constant value)
            sys.eq_int[sys.neq_int]->set_part(nz, new Ope_minus(&sys, sys.give_ope(0, p2)));
            sys.neq_int++;
        }
        sys.nbr_conditions = -1;
    }

} // namespace Kadath
