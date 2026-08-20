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
#include "For_Kadath/Space/adapted_bh_polar.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"

namespace Kadath
{
    void Space_adapted_bh_polar::add_bc_bh(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(HOMOTHETIC_INNER, INNER_BC, name, nused, pused);
    }

    void Space_adapted_bh_polar::add_bc_inf(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(nbr_domains - 1, OUTER_BC, name, nused, pused);
    }

    void Space_adapted_bh_polar::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                        int nused, Array<int>** pused) const
    {
        for (int dom = HOMOTHETIC_INNER; dom < sys.get_dom_max(); ++dom) {
            sys.add_eq_inside(dom, eq, nused, pused, rac);
            sys.add_eq_matching(dom, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(dom, OUTER_BC, rac_der, nused, pused, rac);
        }
        sys.add_eq_inside(sys.get_dom_max(), eq, nused, pused, rac);
    }

    void Space_adapted_bh_polar::add_eq_free_horizon(System_of_eqs& sys, const char* eq, const char* rac,
                                                     const char* rac_der, int nused, Array<int>** pused) const
    {

        sys.add_eq_one_side(HOMOTHETIC_INNER, eq);
        sys.add_eq_matching(HOMOTHETIC_INNER, OUTER_BC, rac, nused, pused, rac);
        sys.add_eq_matching(HOMOTHETIC_INNER, OUTER_BC, rac_der, nused, pused, rac);

        for (int dom = HOMOTHETIC_INNER + 1; dom < sys.get_dom_max(); ++dom) {
            sys.add_eq_inside(dom, eq, nused, pused, rac);
            sys.add_eq_matching(dom, OUTER_BC, rac, nused, pused, rac);
            sys.add_eq_matching(dom, OUTER_BC, rac_der, nused, pused, rac);
        }
        sys.add_eq_inside(sys.get_dom_max(), eq, nused, pused, rac);
    }

    void Space_adapted_bh_polar::add_eq_inside(System_of_eqs& sys, const char* eq, int nused, Array<int>** pused) const
    {
        for (int dom = HOMOTHETIC_INNER; dom <= sys.get_dom_max(); ++dom) {
            sys.add_eq_inside(dom, eq, nused, pused);
        }
    }

    void Space_adapted_bh_polar::add_eq_one_side(System_of_eqs& sys, const char* eq, int nused,
                                                 Array<int>** pused) const
    {
        for (int dom = HOMOTHETIC_INNER; dom <= sys.get_dom_max(); ++dom) {
            sys.add_eq_one_side(dom, eq, nused, pused);
        }
    }

    void Space_adapted_bh_polar::add_eq_one_side(System_of_eqs& sys, const char* eq, const char* rac, int nused,
                                                 Array<int>** pused) const
    {
        for (int dom = HOMOTHETIC_INNER; dom < sys.get_dom_max(); ++dom) {
            sys.add_eq_one_side(dom, eq, nused, pused);
            sys.add_eq_matching(dom, OUTER_BC, rac, nused, pused, rac);
        }
        sys.add_eq_one_side(sys.get_dom_max(), eq, nused, pused);
    }

    void Space_adapted_bh_polar::add_eq_int_inf(System_of_eqs& sys, const char* nom)
    {
        const Domain_polar_compact* pcomp = dynamic_cast<const Domain_polar_compact*>(domains[nbr_domains - 1]);
        if (pcomp == nullptr) {
            KADATH_THROW("add_eq_int_inf requires a compactified domain");
        }
        int dom = nbr_domains - 1;
        sys.eq_int_list.push_back(std::make_tuple(nom, dom, OUTER_BC));

        char p1[LMAX];
        char p2[LMAX];
        bool indic = sys.is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {
            indic = ((p2[0] == '0') && (p2[1] == ' ') && (p2[2] == '\0')) ? true : false;

            sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(1));

            if (indic)
                sys.eq_int[sys.neq_int]->set_part(0, sys.give_ope(dom, p1, OUTER_BC));
            else
                sys.eq_int[sys.neq_int]->set_part(
                    0, new Ope_sub(&sys, sys.give_ope(dom, p1, OUTER_BC), sys.give_ope(dom, p2, OUTER_BC)));
            sys.neq_int++;
        }
        sys.nbr_conditions = -1;
    }

    void Space_adapted_bh_polar::add_eq_int_volume(System_of_eqs& sys, int nz, const char* nom)
    {
        char p1[LMAX];
        char p2[LMAX];
        bool indic = sys.is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {
            sys.eq_int_list.push_back(std::make_tuple(nom, 0, -1));

            sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(nz + 1));

            for (int d = 0; d < nz; d++)
                sys.eq_int[sys.neq_int]->set_part(d, sys.give_ope(d, p1));
            sys.eq_int[sys.neq_int]->set_part(nz, new Ope_minus(&sys, sys.give_ope(0, p2)));
            sys.neq_int++;
        }
        sys.nbr_conditions = -1;
    }

    void Space_adapted_bh_polar::add_eq_int_horizon(System_of_eqs& sys, const char* nom)
    {
        const Domain_polar_shell_inner_homothetic* pshell =
            dynamic_cast<const Domain_polar_shell_inner_homothetic*>(domains[HOMOTHETIC_INNER]);
        if (pshell == nullptr) {
            KADATH_THROW("add_eq_int_horizon requires that the inner domain is homothetic");
        }
        int dom = HOMOTHETIC_INNER;
        sys.eq_int_list.push_back(std::make_tuple(nom, dom, INNER_BC));

        char p1[LMAX];
        char p2[LMAX];
        bool indic = sys.is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {
            indic = ((p2[0] == '0') && (p2[1] == ' ') && (p2[2] == '\0')) ? true : false;

            sys.ensure_eq_int_slot(); sys.eq_int[sys.neq_int].reset(new Eq_int(1));

            if (indic)
                sys.eq_int[sys.neq_int]->set_part(0, sys.give_ope(dom, p1, INNER_BC));
            else
                sys.eq_int[sys.neq_int]->set_part(
                    0, new Ope_sub(&sys, sys.give_ope(dom, p1, INNER_BC), sys.give_ope(dom, p2, INNER_BC)));
            sys.neq_int++;
        }
        sys.nbr_conditions = -1;
    }

    void Space_adapted_bh_polar::add_eq_zero_mode_inf(System_of_eqs& sys, const char* name, int j, int k)
    {
        Index pos_cf(domains[nbr_domains - 1]->get_nbr_coefs());
        pos_cf.set(1) = j;
        if (pos_cf.get_ndim() > 2) {
            pos_cf.set(2) = k;
        }
        double value = 0.;
        char auxi[LMAX];
        trim_spaces(auxi, name);
        sys.add_eq_mode(nbr_domains - 1, OUTER_BC, auxi, pos_cf, value);
    }

} // namespace Kadath
