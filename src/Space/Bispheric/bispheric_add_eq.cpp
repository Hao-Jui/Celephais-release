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

#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{
    void Space_bispheric::add_bc_sphere_one(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {

        if (ndom_minus == 1) {
            // BC in domain 2 and 3
            sys.add_eq_bc(ndom_minus + ndom_plus, INNER_BC, name, nused, pused);
            sys.add_eq_bc(ndom_minus + ndom_plus + 1, INNER_BC, name, nused, pused);
        } else {
            sys.add_eq_bc(2, INNER_BC, name, nused, pused);
        }
    }

    void Space_bispheric::add_bc_sphere_two(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        if (ndom_plus == 1) {
            // BC in domain 5 and 6
            sys.add_eq_bc(ndom_minus + ndom_plus + 3, INNER_BC, name, nused, pused);
            sys.add_eq_bc(ndom_minus + ndom_plus + 4, INNER_BC, name, nused, pused);
        } else {
            sys.add_eq_bc(ndom_minus + 1, INNER_BC, name, nused, pused);
        }
    }

    void Space_bispheric::add_bc_outer_sphere(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        // All domains :
        for (int dd = ndom_minus + ndom_plus; dd < ndom_minus + ndom_plus + 5; dd++)
            sys.add_eq_bc(dd, OUTER_BC, name, nused, pused);
    }

    void Space_bispheric::add_bc_outer(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(nbr_domains - 1, OUTER_BC, name, nused, pused);
    }

    void Space_bispheric::add_eq(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd <= sys.get_dom_max(); dd++)
            sys.add_eq_inside(dd, name, nused, pused);
    }

    void Space_bispheric::add_eq_full(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd <= sys.get_dom_max(); dd++)
            sys.add_eq_full(dd, name, nused, pused);
    }

    void Space_bispheric::add_eq_one_side(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        for (int dd = sys.get_dom_min(); dd <= sys.get_dom_max(); dd++)
            sys.add_eq_one_side(dd, name, nused, pused);
    }

    void Space_bispheric::add_matching(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_matching(ndom_minus + ndom_plus, CHI_ONE_BC, name, nused, pused);
        sys.add_eq_matching(ndom_minus + ndom_plus + 1, ETA_PLUS_BC, name, nused, pused);
        sys.add_eq_matching(ndom_minus + ndom_plus + 2, ETA_PLUS_BC, name, nused, pused);
        sys.add_eq_matching(ndom_minus + ndom_plus + 3, CHI_ONE_BC, name, nused, pused);
    }

    void Space_bispheric::add_matching(System_of_eqs& sys, const char* name, const List_comp& list)
    {
        add_matching(sys, name, list.get_ncomp(), list.get_pcomp());
    }

    void Space_bispheric::add_matching_one_side(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_matching_one_side(ndom_minus + ndom_plus, CHI_ONE_BC, name, nused, pused);
        sys.add_eq_matching_one_side(ndom_minus + ndom_plus + 1, ETA_PLUS_BC, name, nused, pused);
        sys.add_eq_matching_one_side(ndom_minus + ndom_plus + 2, ETA_PLUS_BC, name, nused, pused);
        sys.add_eq_matching_one_side(ndom_minus + ndom_plus + 3, CHI_ONE_BC, name, nused, pused);
    }

    void Space_bispheric::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der, int nused,
                                 Array<int>** pused)
    {

        // First NS
        sys.add_eq_inside(0, eq, nused, pused, rac);
        sys.add_eq_matching(0, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(0, OUTER_BC, rac_der, nused, pused, rac);
        sys.add_eq_inside(1, eq, nused, pused, rac);
        sys.add_eq_matching(1, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(1, OUTER_BC, rac_der, nused, pused, rac);
        sys.add_eq_inside(2, eq, nused, pused, rac);

        // Matching with bispheric :
        sys.add_eq_matching_import(2, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(6, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(7, INNER_BC, rac_der, nused, pused, rac);

        // Second NS :
        sys.add_eq_inside(3, eq, nused, pused, rac);
        sys.add_eq_matching(3, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(3, OUTER_BC, rac_der, nused, pused, rac);
        sys.add_eq_inside(4, eq, nused, pused, rac);
        sys.add_eq_matching(4, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(4, OUTER_BC, rac_der, nused, pused, rac);
        sys.add_eq_inside(5, eq, nused, pused, rac);

        // Matching with bispheric :
        sys.add_eq_matching_import(5, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(9, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(10, INNER_BC, rac_der, nused, pused, rac);

        // Chi first
        sys.add_eq_inside(6, eq, nused, pused, rac);
        sys.add_eq_matching(6, CHI_ONE_BC, rac, nused, pused, rac);
        sys.add_eq_matching(6, CHI_ONE_BC, rac_der, nused, pused, rac);

        // Rect :
        sys.add_eq_inside(7, eq, nused, pused, rac);
        sys.add_eq_matching(7, ETA_PLUS_BC, rac, nused, pused, rac);
        sys.add_eq_matching(7, ETA_PLUS_BC, rac_der, nused, pused, rac);

        // Eta first
        sys.add_eq_inside(8, eq, nused, pused, rac);
        sys.add_eq_matching(8, ETA_PLUS_BC, rac, nused, pused, rac);
        sys.add_eq_matching(8, ETA_PLUS_BC, rac_der, nused, pused, rac);

        // Rect
        sys.add_eq_inside(9, eq, nused, pused, rac);
        sys.add_eq_matching(9, CHI_ONE_BC, rac, nused, pused, rac);
        sys.add_eq_matching(9, CHI_ONE_BC, rac_der, nused, pused, rac);

        // chi first :
        sys.add_eq_inside(10, eq, nused, pused, rac);

        // Matching outer domain :
        for (int d = 6; d <= 10; d++)
            sys.add_eq_matching_import(d, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(11, INNER_BC, rac_der, nused, pused, rac);

        // Compactified domain
        sys.add_eq_inside(11, eq, nused, pused, rac);
    }

    void Space_bispheric::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                 const List_comp& list)
    {
        add_eq(sys, eq, rac, rac_der, list.get_ncomp(), list.get_pcomp());
    }

    void Space_bispheric::add_eq_no_nucleus(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                            int nused, Array<int>** pused)
    {

        // Shell 1
        sys.add_eq_inside(0, eq, nused, pused, rac);

        // Matching with bispheric :
        sys.add_eq_matching_import(0, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(2, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(3, INNER_BC, rac_der, nused, pused, rac);

        // Shell 2
        sys.add_eq_inside(1, eq, nused, pused, rac);

        // Matching with bispheric :
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

        // Compactified domain
        sys.add_eq_inside(7, eq, nused, pused, rac);
    }

    void Space_bispheric::add_eq_no_nucleus(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                            const List_comp& list)
    {
        add_eq(sys, eq, rac, rac_der, list.get_ncomp(), list.get_pcomp());
    }
    void Space_bispheric::add_eq_int_inf(System_of_eqs& sys, const char* nom)
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

    void Space_bispheric::add_eq_int_sphere_one(System_of_eqs& sys, const char* nom)
    {

        // Get the lhs and rhs
        char p1[LMAX];
        char p2[LMAX];
        bool indic = sys.is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {

            // Verif lhs = 0 ?
            indic = ((p2[0] == '0') && (p2[1] == ' ') && (p2[2] == '\0')) ? true : false;

            if (ndom_minus == 1) {
                // Construction of the equation
                sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(2));

                // Affectation :
                // no lhs :
                if (indic) {
                    sys.eq_int[sys.neq_int]->set_part(0, sys.give_ope(ndom_minus + ndom_plus, p1, INNER_BC));
                    sys.eq_int[sys.neq_int]->set_part(1, sys.give_ope(ndom_minus + ndom_plus + 1, p1, INNER_BC));
                } else {
                    sys.eq_int[sys.neq_int]->set_part(
                        0, new Ope_sub(&sys, sys.give_ope(ndom_minus + ndom_plus, p1, INNER_BC),
                                       sys.give_ope(ndom_minus + ndom_plus, p2, INNER_BC)));
                    sys.eq_int[sys.neq_int]->set_part(
                        1, new Ope_sub(&sys, sys.give_ope(ndom_minus + ndom_plus + 1, p1, INNER_BC),
                                       sys.give_ope(ndom_minus + ndom_plus + 1, p2, INNER_BC)));
                }
            } else {
                // Construction of the equation
                sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(1));

                // Affectation :
                // no lhs :
                if (indic) {
                    sys.eq_int[sys.neq_int]->set_part(0, sys.give_ope(1, p1, INNER_BC));
                } else {
                    sys.eq_int[sys.neq_int]->set_part(
                        0, new Ope_sub(&sys, sys.give_ope(1, p1, INNER_BC), sys.give_ope(1, p2, INNER_BC)));
                }
            }
            sys.neq_int++;
        }
        sys.nbr_conditions = -1;
    }

    void Space_bispheric::add_eq_int_sphere_two(System_of_eqs& sys, const char* nom)
    {

        // Get the lhs and rhs
        char p1[LMAX];
        char p2[LMAX];
        bool indic = sys.is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {
            // Verif lhs = 0 ?
            indic = ((p2[0] == '0') && (p2[1] == ' ') && (p2[2] == '\0')) ? true : false;

            if (ndom_plus == 1) {
                // Construction of the equation
                sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(2));

                // Affectation :
                // no lhs :
                if (indic) {
                    sys.eq_int[sys.neq_int]->set_part(0, sys.give_ope(ndom_minus + ndom_plus + 3, p1, INNER_BC));
                    sys.eq_int[sys.neq_int]->set_part(1, sys.give_ope(ndom_minus + ndom_plus + 4, p1, INNER_BC));
                } else {
                    sys.eq_int[sys.neq_int]->set_part(
                        0, new Ope_sub(&sys, sys.give_ope(ndom_minus + ndom_plus + 3, p1, INNER_BC),
                                       sys.give_ope(ndom_minus + ndom_plus + 3, p2, INNER_BC)));
                    sys.eq_int[sys.neq_int]->set_part(
                        1, new Ope_sub(&sys, sys.give_ope(ndom_minus + ndom_plus + 4, p1, INNER_BC),
                                       sys.give_ope(ndom_minus + ndom_plus + 4, p2, INNER_BC)));
                }
            } else {
                // Construction of the equation
                sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(1));

                // Affectation :
                // no lhs :
                if (indic) {
                    sys.eq_int[sys.neq_int]->set_part(0, sys.give_ope(ndom_minus + 1, p1, INNER_BC));
                } else {
                    sys.eq_int[sys.neq_int]->set_part(0, new Ope_sub(&sys, sys.give_ope(ndom_minus + 1, p1, INNER_BC),
                                                                     sys.give_ope(ndom_minus + 1, p2, INNER_BC)));
                }
            }

            sys.neq_int++;
        }
        sys.nbr_conditions = -1;
    }

    void Space_bispheric::add_eq_zero_mode_inf(System_of_eqs& sys, const char* name, int j, int k)
    {

        Index pos_cf(domains[nbr_domains - 1]->get_nbr_coefs());
        pos_cf.set(1) = j;
        pos_cf.set(2) = k;
        double value = 0.;
        char auxi[LMAX];
        trim_spaces(auxi, name);
        sys.add_eq_mode(nbr_domains - 1, OUTER_BC, auxi, pos_cf, value);
    }
} // namespace Kadath
