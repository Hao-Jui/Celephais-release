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
#include "For_Kadath/Space/bin_fake.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"

namespace Kadath
{
    void Space_bin_fake::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der, int nused,
                                Array<int>** pused)
    {

        // Stars
        sys.add_eq_inside(0, eq, nused, pused, rac);
        sys.add_eq_inside(1, eq, nused, pused, rac);

        // Matching with bispheric
        sys.add_eq_matching_import(0, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(2, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(3, INNER_BC, rac_der, nused, pused, rac);

        sys.add_eq_matching_import(1, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(5, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(6, INNER_BC, rac_der, nused, pused, rac);

        // Chi first
        sys.add_eq_inside(2, eq, nused, pused, rac);
        sys.add_eq_matching(2, CHI_ONE_BC, rac, nused, pused, rac);
        sys.add_eq_matching(2, CHI_ONE_BC, rac_der, nused, pused, rac);

        // Rect :
        sys.add_eq_inside(3, eq, nused, pused, rac);
        sys.add_eq_matching(3, ETA_PLUS_BC, rac, nused, pused, rac);
        sys.add_eq_matching(3, ETA_PLUS_BC, rac_der, nused, pused, rac);

        // Eta first
        sys.add_eq_inside(4, eq, nused, pused, rac);
        sys.add_eq_matching(4, ETA_PLUS_BC, rac, nused, pused, rac);
        sys.add_eq_matching(4, ETA_PLUS_BC, rac_der, nused, pused, rac);

        // Rect
        sys.add_eq_inside(5, eq, nused, pused, rac);
        sys.add_eq_matching(5, CHI_ONE_BC, rac, nused, pused, rac);
        sys.add_eq_matching(5, CHI_ONE_BC, rac_der, nused, pused, rac);

        // chi first :
        sys.add_eq_inside(6, eq, nused, pused, rac);

        // Matching outer domain :
        for (int d = 2; d <= 6; d++)
            sys.add_eq_matching_import(d, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(7, INNER_BC, rac_der, nused, pused, rac);

        // Shell :
        sys.add_eq_inside(7, eq, nused, pused, rac);
        sys.add_eq_matching(7, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(7, OUTER_BC, rac_der, nused, pused, rac);

        // Compactified domain
        sys.add_eq_inside(8, eq, nused, pused, rac);
    }

    void Space_bin_fake::add_eq_nozec(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                      int nused, Array<int>** pused)
    {

        // Stars
        sys.add_eq_inside(0, eq, nused, pused, rac);
        sys.add_eq_inside(1, eq, nused, pused, rac);

        // Matching with bispheric
        sys.add_eq_matching_import(0, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(2, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(3, INNER_BC, rac_der, nused, pused, rac);

        sys.add_eq_matching_import(1, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(5, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(6, INNER_BC, rac_der, nused, pused, rac);

        // Chi first
        sys.add_eq_inside(2, eq, nused, pused, rac);
        sys.add_eq_matching(2, CHI_ONE_BC, rac, nused, pused, rac);
        sys.add_eq_matching(2, CHI_ONE_BC, rac_der, nused, pused, rac);

        // Rect :
        sys.add_eq_inside(3, eq, nused, pused, rac);
        sys.add_eq_matching(3, ETA_PLUS_BC, rac, nused, pused, rac);
        sys.add_eq_matching(3, ETA_PLUS_BC, rac_der, nused, pused, rac);

        // Eta first
        sys.add_eq_inside(4, eq, nused, pused, rac);
        sys.add_eq_matching(4, ETA_PLUS_BC, rac, nused, pused, rac);
        sys.add_eq_matching(4, ETA_PLUS_BC, rac_der, nused, pused, rac);

        // Rect
        sys.add_eq_inside(5, eq, nused, pused, rac);
        sys.add_eq_matching(5, CHI_ONE_BC, rac, nused, pused, rac);
        sys.add_eq_matching(5, CHI_ONE_BC, rac_der, nused, pused, rac);

        // chi first :
        sys.add_eq_inside(6, eq, nused, pused, rac);

        // Matching outer domain :
        for (int d = 2; d <= 6; d++)
            sys.add_eq_matching_import(d, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(7, INNER_BC, rac_der, nused, pused, rac);

        // Shell :
        sys.add_eq_inside(7, eq, nused, pused, rac);
    }

    void Space_bin_fake::add_eq_int_inf(System_of_eqs& sys, const char* nom)
    {

        // Check the last domain is of the right type :
        const Domain_compact* pcomp = dynamic_cast<const Domain_compact*>(domains[nbr_domains - 1]);
        if (pcomp == nullptr) {
            KADATH_THROW("add_eq_int_inf requires a compactified domain");
        }
        int dom = nbr_domains - 1;

        // Get the lhs and rhs
        char p1[LMAX];
        char p2[LMAX];
        bool indic = sys.is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {
            // Verif lhs = 0 ?
            indic = ((p2[0] == '0') && (p2[1] == ' ') && (p2[2] == '\0')) ? true : false;

            // Construction of the equation
            sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(1));

            // Affectation :
            // no lhs :
            if (indic)
                sys.eq_int[sys.neq_int]->set_part(0, sys.give_ope(dom, p1, OUTER_BC));

            else
                sys.eq_int[sys.neq_int]->set_part(
                    0, new Ope_sub(&sys, sys.give_ope(dom, p1, OUTER_BC), sys.give_ope(dom, p2, OUTER_BC)));
            sys.neq_int++;
        }
        sys.nbr_conditions = -1;
    }
} // namespace Kadath
