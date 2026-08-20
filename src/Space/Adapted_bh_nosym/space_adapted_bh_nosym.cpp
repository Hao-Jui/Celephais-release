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

#include <sstream>
#include "For_Kadath/Space/adapted_bh_nosym.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include <vector>
namespace Kadath
{
    Space_adapted_bh_nosym::Space_adapted_bh_nosym(int ttype, const Point& center, const Dim_array& res,
                                                   const std::vector<double>& BH_bounds)
    {
        enum bound_index { RIN = 0, RMID = 1, NUM_IND };

        ndim = 3;

        nbr_domains = BH_bounds.size() + 1;
        type_base = ttype;
        domains = new Domain*[nbr_domains];
        auto gen_bh_domains = [&](const int nuc_i, auto& bounds) {
            int shells = bounds.size() - 3;
            Point center(ndim);
            domains[nuc_i] = new Domain_nucleus_nosym(nuc_i, ttype, bounds[RIN], center, res);
            domains[nuc_i + 1] =
                new Domain_shell_outer_homothetic_nosym(*this, nuc_i + 1, ttype, bounds[RIN], bounds[RMID], center, res);
            domains[nuc_i + 2] = new Domain_shell_inner_homothetic_nosym(*this, nuc_i + 2, ttype, bounds[RMID],
                                                                         bounds[RMID + 1], center, res);

            for (int i = 0; i < shells; ++i)
                domains[nuc_i + 3 + i] = new Domain_shell_nosym(nuc_i + 3 + i, ttype, bounds[RMID + 1 + i],
                                                                bounds[RMID + 1 + i + 1], center, res);
        };
        gen_bh_domains(0, BH_bounds);

        domains[nbr_domains - 1] = new Domain_compact_nosym(nbr_domains - 1, ttype, BH_bounds.back(), center, res);

        const Domain_shell_outer_homothetic_nosym* pouter_1 =
            dynamic_cast<const Domain_shell_outer_homothetic_nosym*>(domains[1]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_homothetic_nosym* pinner_1 =
            dynamic_cast<const Domain_shell_inner_homothetic_nosym*>(domains[2]);
        pinner_1->vars_to_terms();
        pinner_1->update();
    }

    Space_adapted_bh_nosym::Space_adapted_bh_nosym(BinarySource& source)
    {
        nbr_domains = source.read<int>();
        ndim = source.read<int>();
        type_base = source.read<int>();
        domains = new Domain*[nbr_domains];

        domains[0] = new Domain_nucleus_nosym(0, source);
        domains[1] = new Domain_shell_outer_homothetic_nosym(*this, 1, source);
        domains[2] = new Domain_shell_inner_homothetic_nosym(*this, 2, source);

        for (int i = 3; i < nbr_domains - 1; i++)
            domains[i] = new Domain_shell_nosym(i, source);
        domains[nbr_domains - 1] = new Domain_compact_nosym(nbr_domains - 1, source);

        const Domain_shell_outer_homothetic_nosym* pouter_1 =
            dynamic_cast<const Domain_shell_outer_homothetic_nosym*>(domains[1]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_homothetic_nosym* pinner_1 =
            dynamic_cast<const Domain_shell_inner_homothetic_nosym*>(domains[2]);
        pinner_1->vars_to_terms();
        pinner_1->update();
    }

    Space_adapted_bh_nosym::~Space_adapted_bh_nosym()
    {
        const Domain_shell_outer_homothetic_nosym* pouter_1 =
            dynamic_cast<const Domain_shell_outer_homothetic_nosym*>(domains[1]);
        pouter_1->del_deriv();
        const Domain_shell_inner_homothetic_nosym* pinner_1 =
            dynamic_cast<const Domain_shell_inner_homothetic_nosym*>(domains[2]);
        pinner_1->del_deriv();
    }

    void Space_adapted_bh_nosym::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }

    int Space_adapted_bh_nosym::nbr_unknowns_from_variable_domains() const
    {
        return domains[2]->nbr_unknowns_from_adapted();
    }

    void Space_adapted_bh_nosym::affecte_coef_to_variable_domains(int& pos, int cc, Array<int>& zedoms) const
    {
        bool found_outer = false;
        int old_pos = pos;
        domains[1]->affecte_coef(pos, cc, found_outer);
        pos = old_pos;
        bool found_inner = false;
        domains[2]->affecte_coef(pos, cc, found_inner);
        assert(found_outer == found_inner);
        if (found_outer) {
            zedoms.set(0) = 1;
            zedoms.set(1) = 2;
        } else {
            zedoms.set(0) = -1;
            zedoms.set(1) = -1;
        }
    }

    void Space_adapted_bh_nosym::xx_to_ders_variable_domains(const Array<double>& xx, int& conte) const
    {
        int old_conte = conte;
        domains[1]->xx_to_ders_from_adapted(xx, conte);
        conte = old_conte;
        domains[2]->xx_to_ders_from_adapted(xx, conte);
    }

    void Space_adapted_bh_nosym::xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& xx, int& pos) const
    {

        // First get the corrections :
        int old_pos = pos;
        Val_domain cor_outer(domains[1]);
        domains[1]->xx_to_vars_from_adapted(cor_outer, xx, pos);
        pos = old_pos;
        Val_domain cor_inner(domains[2]);
        domains[2]->xx_to_vars_from_adapted(cor_inner, xx, pos);

        // Now update the variables :
        for (int i = 0; i < sys->nvar; i++) {
            for (int n = 0; n < sys->var[i]->get_n_comp(); n++) {
                Scalar res(*this);
                domains[1]->update_variable(cor_outer, *sys->var[i]->cmp[n], res);
                domains[2]->update_variable(cor_inner, *sys->var[i]->cmp[n], res);

                sys->var[i]->cmp[n]->set_domain(1) = res(1);
                sys->var[i]->cmp[n]->set_domain(2) = res(2);
            }
        }

        // Now the constants :
        for (int i = 0; i < sys->ncst; i++)
            if (sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->get_type_data() == TERM_T)
                for (int n = 0; n < sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->val_t->get_n_comp(); n++) {

                    Scalar so(*this);
                    so = 0;
                    so.set_domain(1) = (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + 1]->val_t->cmp[n])(1);
                    so.set_domain(2) = (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + 2]->val_t->cmp[n])(2);

                    Scalar res(*this);
                    res = 0;
                    domains[1]->update_constante(cor_outer, so, res);
                    domains[2]->update_constante(cor_inner, so, res);

                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + 1]->val_t->cmp[n]->set_domain(1) = res(1);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + 2]->val_t->cmp[n]->set_domain(2) = res(2);
                }

        // Update the mapping :
        domains[1]->update_mapping(cor_outer);
        domains[2]->update_mapping(cor_inner);
    }
    Array<int> Space_adapted_bh_nosym::get_indices_matching_non_std(int dom, int bound) const
    {

        assert((dom >= 0) && (dom < nbr_domains));
        Array<int> res(2, 1);
        switch (bound) {
            case OUTER_BC:
                res.set(0, 0) = dom + 1;
                res.set(1, 0) = INNER_BC;
                break;
            case INNER_BC:
                res.set(0, 0) = dom - 1;
                res.set(1, 0) = OUTER_BC;
                break;
            default:
                cerr << "Unknown boundary in " << endl;
                std::ostringstream oss;
                oss << *this << endl;
                KADATH_THROW(oss.str());
        }
        return res;
    }

} // namespace Kadath
