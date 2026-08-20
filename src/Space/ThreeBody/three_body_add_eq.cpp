/*
    Copyright 2026 Philippe Grandclement

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

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Space/three_body.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"

namespace Kadath
{
    namespace
    {
        Array<int> one_match(int dom, int bound)
        {
            Array<int> res(2, 1);
            res.set(0, 0) = dom;
            res.set(1, 0) = bound;
            return res;
        }

        Array<int> two_matches(int dom0, int bound0, int dom1, int bound1)
        {
            Array<int> res(2, 2);
            res.set(0, 0) = dom0;
            res.set(1, 0) = bound0;
            res.set(0, 1) = dom1;
            res.set(1, 1) = bound1;
            return res;
        }

        Array<int> five_bispheric_outer_matches(int first_bispheric)
        {
            Array<int> res(2, 5);
            for (int i = 0; i < 5; ++i) {
                res.set(0, i) = first_bispheric + i;
                res.set(1, i) = OUTER_BC;
            }
            return res;
        }

        void add_star_equations(System_of_eqs& sys, int nucleus, int adapted, int n_shells, const char* eq,
                                const char* rac, const char* rac_der, int nused, Array<int>** pused)
        {
            sys.add_eq_inside(nucleus, eq, nused, pused, rac);
            sys.add_eq_matching(nucleus, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(nucleus, OUTER_BC, rac_der, nused, pused, rac);

            sys.add_eq_inside(adapted, eq, nused, pused, rac);
            sys.add_eq_matching(adapted, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(adapted, OUTER_BC, rac_der, nused, pused, rac);
            sys.add_eq_inside(adapted + 1, eq, nused, pused, rac);

            if (n_shells > 0) {
                sys.add_eq_matching(adapted + 1, OUTER_BC, rac, nused, pused, rac);
                sys.add_eq_matching(adapted + 1, OUTER_BC, rac_der, nused, pused, rac);
            }

            for (int i = 0; i < n_shells; ++i) {
                const int shell = adapted + 2 + i;
                sys.add_eq_inside(shell, eq, nused, pused, rac);
                if (i != n_shells - 1) {
                    sys.add_eq_matching(shell, OUTER_BC, rac, nused, pused, rac);
                    sys.add_eq_matching(shell, OUTER_BC, rac_der, nused, pused, rac);
                }
            }
        }

        void add_bispheric_interiors(System_of_eqs& sys, int first, const char* eq, const char* rac,
                                     const char* rac_der, int nused, Array<int>** pused)
        {
            sys.add_eq_inside(first, eq, nused, pused, rac);
            sys.add_eq_matching(first, CHI_ONE_BC, rac, nused, pused, rac);
            sys.add_eq_matching(first, CHI_ONE_BC, rac_der, nused, pused, rac);

            sys.add_eq_inside(first + 1, eq, nused, pused, rac);
            sys.add_eq_matching(first + 1, ETA_PLUS_BC, rac, nused, pused, rac);
            sys.add_eq_matching(first + 1, ETA_PLUS_BC, rac_der, nused, pused, rac);

            sys.add_eq_inside(first + 2, eq, nused, pused, rac);
            sys.add_eq_matching(first + 2, ETA_PLUS_BC, rac, nused, pused, rac);
            sys.add_eq_matching(first + 2, ETA_PLUS_BC, rac_der, nused, pused, rac);

            sys.add_eq_inside(first + 3, eq, nused, pused, rac);
            sys.add_eq_matching(first + 3, CHI_ONE_BC, rac, nused, pused, rac);
            sys.add_eq_matching(first + 3, CHI_ONE_BC, rac_der, nused, pused, rac);

            sys.add_eq_inside(first + 4, eq, nused, pused, rac);
        }
    } // namespace

    void Space_three_body::add_bc_body(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(ADAPTED_BODY + 1, INNER_BC, name, nused, pused);
    }

    void Space_three_body::add_bc_child_one(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(ADAPTED_CHILD1 + 1, INNER_BC, name, nused, pused);
    }

    void Space_three_body::add_bc_child_two(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(ADAPTED_CHILD2 + 1, INNER_BC, name, nused, pused);
    }

    void Space_three_body::add_bc_child_outer(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        for (int dom = CHILD_BI; dom <= CHILD_BI + 4; ++dom)
            sys.add_eq_bc(dom, OUTER_BC, name, nused, pused);
    }

    void Space_three_body::add_bc_outer(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(compact_domain(), OUTER_BC, name, nused, pused);
    }

    void Space_three_body::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der, int nused,
                                  Array<int>** pused)
    {
        add_star_equations(sys, BODY, ADAPTED_BODY, n_shells_body, eq, rac, rac_der, nused, pused);
        sys.add_eq_matching_import(body_outer_domain(), OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(PARENT_BI, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(PARENT_BI + 1, INNER_BC, rac_der, nused, pused, rac);

        add_star_equations(sys, CHILD1, ADAPTED_CHILD1, n_shells_child1, eq, rac, rac_der, nused, pused);
        sys.add_eq_matching_import(child1_outer_domain(), OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(CHILD_BI, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(CHILD_BI + 1, INNER_BC, rac_der, nused, pused, rac);

        add_star_equations(sys, CHILD2, ADAPTED_CHILD2, n_shells_child2, eq, rac, rac_der, nused, pused);
        sys.add_eq_matching_import(child2_outer_domain(), OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(CHILD_BI + 3, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(CHILD_BI + 4, INNER_BC, rac_der, nused, pused, rac);

        add_bispheric_interiors(sys, CHILD_BI, eq, rac, rac_der, nused, pused);
        for (int dom = CHILD_BI; dom <= CHILD_BI + 4; ++dom)
            sys.add_eq_matching_import(dom, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(PARENT_BI + 3, INNER_BC, rac_der, nused, pused, rac);
        sys.add_eq_matching_import(PARENT_BI + 4, INNER_BC, rac_der, nused, pused, rac);

        add_bispheric_interiors(sys, PARENT_BI, eq, rac, rac_der, nused, pused);
        for (int dom = PARENT_BI; dom <= PARENT_BI + 4; ++dom)
            sys.add_eq_matching_import(dom, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching_import(outer_shell_or_compact_domain(), INNER_BC, rac_der, nused, pused, rac);

        for (int i = 0; i < n_shells_outer; ++i) {
            const int shell = OUTER + i;
            sys.add_eq_inside(shell, eq, nused, pused, rac);
            sys.add_eq_matching(shell, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(shell, OUTER_BC, rac_der, nused, pused, rac);
        }

        sys.add_eq_inside(compact_domain(), eq, nused, pused, rac);
    }

    void Space_three_body::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                  const List_comp& used)
    {
        add_eq(sys, eq, rac, rac_der, used.get_ncomp(), used.get_pcomp());
    }

    void Space_three_body::add_eq_int_inf(System_of_eqs& sys, const char* nom)
    {
        sys.eq_int_list.push_back(std::make_tuple(nom, nbr_domains - 1, OUTER_BC));

        // Check the last domain is of the right type :
        const Domain_compact_nosym* pcomp = dynamic_cast<const Domain_compact_nosym*>(domains[nbr_domains - 1]);
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

    void Space_three_body::add_eq_int_volume(System_of_eqs& sys, int dmin, int dmax, const char* nom)
    {
        sys.eq_int_list.push_back(std::make_tuple(nom, dmin, -1));

        // Get the lhs and rhs
        char p1[LMAX];
        char p2[LMAX];

        bool indic = sys.is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {
            // Construction of the equation
            int nz = dmax - dmin + 1;
            sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(nz + 1));

            // Affectation of the intregrale parts
            for (int d = 0; d < nz; d++)
                sys.eq_int[sys.neq_int]->set_part(d, sys.give_ope(d + dmin, p1));
            // Affectation of the second member (constant value)
            sys.eq_int[sys.neq_int]->set_part(nz, new Ope_minus(&sys, sys.give_ope(dmin, p2)));
            sys.neq_int++;
        }
        sys.nbr_conditions = -1;
    }

    Array<int> Space_three_body::get_indices_matching_non_std(int dom, int bound) const
    {
        if (dom == body_outer_domain()) {
            if (bound == OUTER_BC)
                return two_matches(PARENT_BI, INNER_BC, PARENT_BI + 1, INNER_BC);
            KADATH_THROW("Bad body bound in Space_three_body::get_indices_matching_non_std");
        }

        if (dom == child1_outer_domain()) {
            if (bound == OUTER_BC)
                return two_matches(CHILD_BI, INNER_BC, CHILD_BI + 1, INNER_BC);
            KADATH_THROW("Bad child1 bound in Space_three_body::get_indices_matching_non_std");
        }

        if (dom == child2_outer_domain()) {
            if (bound == OUTER_BC)
                return two_matches(CHILD_BI + 3, INNER_BC, CHILD_BI + 4, INNER_BC);
            KADATH_THROW("Bad child2 bound in Space_three_body::get_indices_matching_non_std");
        }

        if (dom == CHILD_BI || dom == CHILD_BI + 1) {
            if (bound == INNER_BC)
                return one_match(child1_outer_domain(), OUTER_BC);
            if (bound == OUTER_BC)
                return two_matches(PARENT_BI + 3, INNER_BC, PARENT_BI + 4, INNER_BC);
            KADATH_THROW("Bad child first-side bispheric bound in Space_three_body::get_indices_matching_non_std");
        }

        if (dom == CHILD_BI + 2) {
            if (bound == OUTER_BC)
                return two_matches(PARENT_BI + 3, INNER_BC, PARENT_BI + 4, INNER_BC);
            KADATH_THROW("Bad child middle bispheric bound in Space_three_body::get_indices_matching_non_std");
        }

        if (dom == CHILD_BI + 3 || dom == CHILD_BI + 4) {
            if (bound == INNER_BC)
                return one_match(child2_outer_domain(), OUTER_BC);
            if (bound == OUTER_BC)
                return two_matches(PARENT_BI + 3, INNER_BC, PARENT_BI + 4, INNER_BC);
            KADATH_THROW("Bad child second-side bispheric bound in Space_three_body::get_indices_matching_non_std");
        }

        if (dom == PARENT_BI || dom == PARENT_BI + 1) {
            if (bound == INNER_BC)
                return one_match(body_outer_domain(), OUTER_BC);
            if (bound == OUTER_BC)
                return one_match(outer_shell_or_compact_domain(), INNER_BC);
            KADATH_THROW("Bad parent first-side bispheric bound in Space_three_body::get_indices_matching_non_std");
        }

        if (dom == PARENT_BI + 2) {
            if (bound == OUTER_BC)
                return one_match(outer_shell_or_compact_domain(), INNER_BC);
            KADATH_THROW("Bad parent middle bispheric bound in Space_three_body::get_indices_matching_non_std");
        }

        if (dom == PARENT_BI + 3 || dom == PARENT_BI + 4) {
            if (bound == INNER_BC)
                return five_bispheric_outer_matches(CHILD_BI);
            if (bound == OUTER_BC)
                return one_match(outer_shell_or_compact_domain(), INNER_BC);
            KADATH_THROW("Bad parent second-side bispheric bound in Space_three_body::get_indices_matching_non_std");
        }

        if (dom == outer_shell_or_compact_domain()) {
            if (bound == INNER_BC)
                return five_bispheric_outer_matches(PARENT_BI);
            KADATH_THROW("Bad outer bound in Space_three_body::get_indices_matching_non_std");
        }

        KADATH_THROW("Bad domain in Space_three_body::get_indices_matching_non_std");
    }
} // namespace Kadath
