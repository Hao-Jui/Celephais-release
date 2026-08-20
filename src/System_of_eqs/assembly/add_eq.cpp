/*
    Copyright 2017 Philippe Grandclement
    Copyright 2020 Ludwig Jens Papenfort

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

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "For_Kadath/List_comp/list_comp.hpp"
#include <set>

namespace Kadath
{
    std::string System_of_eqs::infer_equation_owner_var_name(int dom, const char* owner_expr, int bb,
                                                             const char* fallback_expr) const
    {
        auto resolve = [&](const char* expr) -> std::string {
            if (expr == nullptr || expr[0] == '\0')
                return "";
            std::set<std::string> vars;
            collect_vars_for_eq(dom, expr, bb, vars);
            for (const std::string& name : names_var) {
                if (vars.find(name) != vars.end())
                    return name;
            }
            return vars.size() == 1 ? *vars.begin() : "";
        };

        std::string owner = resolve(owner_expr);
        if (!owner.empty())
            return owner;
        return resolve(fallback_expr);
    }

    void System_of_eqs::record_equation_column_attachment(ColumnClass column_class, int dom, int bb,
                                                          const char* owner_expr,
                                                          const char* fallback_expr)
    {
        EquationColumnAttachment attachment;
        attachment.column_class = column_class;
        attachment.domain = dom;
        attachment.boundary = bb;
        attachment.owner_var_name =
            infer_equation_owner_var_name(dom, owner_expr, bb, fallback_expr);
        eq_column_attachments.push_back(attachment);
    }

    Ope_eq* System_of_eqs::parse_eq(int dom, const char* nom, int boundary) const
    {
        // Is the eq written like "... = 0" ?
        char p1[LMAX];
        char p2[LMAX];
        bool indic = is_ope_bin(nom, p1, p2, '=');
        if (!indic) {
            KADATH_THROW("= needed for equations");
        } else {
            // lhs == 0 ?
            indic = ((p2[0] == '0') && (p2[1] == ' ') && (p2[2] == '\0'));

            // no lhs
            if (indic)
                return give_ope(dom, p1, boundary);
            else
                return new Ope_sub(this, give_ope(dom, p1, boundary), give_ope(dom, p2, boundary));
        }
    }

    Ope_eq* System_of_eqs::parse_eq_trim(int dom, const char* nom, int boundary, bool first) const
    {
        // Is the eq written like "... = ..." ?
        char p1[LMAX];
        char p2[LMAX];
        bool indic = is_ope_bin(nom, p1, p2, '=');

        if (!indic) {
            char auxi[LMAX];
            trim_spaces(auxi, nom);

            // Version without =
            return give_ope(dom, auxi, boundary);
        } else {
            return first ? give_ope(dom, p1, boundary) : give_ope(dom, p2, boundary);
        }
    }

    void System_of_eqs::add_eq_inside(int dom, const char* nom, int n_cmp, Array<int>** p_cmp,
                                      const char* owner_expr)
    {
        eq_list.push_back(std::make_tuple(nom, dom, -1));
        record_equation_column_attachment(ColumnClass::FieldInteriorVol, dom, -1, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_inside(espace.get_domain(dom), dom, parse_eq(dom, nom), n_cmp, p_cmp));

        neq++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_inside(int dom, const char* nom, const List_comp& list)
    {
        add_eq_inside(dom, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_order(int dom, int order, const char* nom, int n_cmp, Array<int>** p_cmp,
                                     const char* owner_expr)
    {
        eq_list.push_back(std::make_tuple(nom, dom, -1));
        record_equation_column_attachment(ColumnClass::FieldInteriorVol, dom, -1, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_order(espace.get_domain(dom), dom, order, parse_eq(dom, nom), n_cmp, p_cmp));

        neq++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_vel_pot(int dom, int order, const char* nom, const char* const_part,
                                       const char* owner_expr,
                                       bool same_reflection_sector)
    {
        eq_list.push_back(std::make_tuple(nom, dom, -1));
        record_equation_column_attachment(ColumnClass::FieldInteriorVol, dom, -1,
                                          owner_expr != nullptr ? owner_expr : const_part, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_vel_pot(
            espace.get_domain(dom), dom, order, parse_eq(dom, nom),
            parse_eq(dom, const_part), same_reflection_sector));

        neq++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_bc_exception(int dom, int bound, const char* nom, const char* const_part,
                                            const char* owner_expr)
    {
        eq_list.push_back(std::make_tuple(nom, dom, bound));
        record_equation_column_attachment(ColumnClass::FieldBoundaryTau, dom, bound,
                                          owner_expr != nullptr ? owner_expr : const_part, nom);

        // Is it written like =0 ?
        char p1[LMAX];
        char p2[LMAX];
        bool indic1 = is_ope_bin(nom, p1, p2, '=');

        char p3[LMAX];
        char p4[LMAX];
        bool indic2 = is_ope_bin(const_part, p3, p4, '=');

        if ((!indic1) || (!indic2)) {
            KADATH_THROW("= needed for equations");
        } else {
            // Verif lhs1 = 0 ?
            indic1 = ((p2[0] == '0') && (p2[1] == ' ') && (p2[2] == '\0')) ? true : false;

            indic2 = ((p4[0] == '0') && (p4[1] == ' ') && (p4[2] == '\0')) ? true : false;

            // no lhs :
            if ((indic1) && (indic2)) {
                ensure_eq_slot();
                eq[neq].reset(new Eq_bc_exception(espace.get_domain(dom), dom, bound, give_ope(dom, p1), give_ope(dom, p3)));
            }
            // lhs in 1
            if ((!indic1) && (indic2)) {
                ensure_eq_slot();
                eq[neq].reset(new Eq_bc_exception(espace.get_domain(dom), dom, bound,
                                        new Ope_sub(this, give_ope(dom, p1), give_ope(dom, p2)), give_ope(dom, p3)));
            }
            // lhs in 2
            if ((indic1) && (!indic2)) {
                ensure_eq_slot();
                eq[neq].reset(new Eq_bc_exception(espace.get_domain(dom), dom, bound, give_ope(dom, p1),
                                              new Ope_sub(this, give_ope(dom, p3), give_ope(dom, p4))));
            }
            // both lhs
            if ((!indic1) && (!indic2)) {
                ensure_eq_slot();
                eq[neq].reset(new Eq_bc_exception(espace.get_domain(dom), dom, bound,
                                              new Ope_sub(this, give_ope(dom, p1), give_ope(dom, p2)),
                                              new Ope_sub(this, give_ope(dom, p3), give_ope(dom, p4))));
            }

            neq++;
        }
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_order(int dom, int order, const char* nom, const List_comp& list)
    {
        add_eq_order(dom, order, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_bc(int dom, int bound, const char* nom, int n_cmp, Array<int>** p_cmp,
                                  const char* owner_expr)
    {
        eq_list.push_back(std::make_tuple(nom, dom, bound));
        record_equation_column_attachment(ColumnClass::FieldBoundaryTau, dom, bound, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_bc(espace.get_domain(dom), dom, bound, parse_eq(dom, nom, bound), n_cmp, p_cmp));

        neq++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_bc_projected(int dom, int bound, const char* nom, int n_cmp, Array<int>** p_cmp,
                                            const char* owner_expr)
    {
        eq_list.push_back(std::make_tuple(nom, dom, bound));
        record_equation_column_attachment(ColumnClass::FieldBoundaryTau, dom, bound, owner_expr, nom);

        ensure_eq_slot();
        eq[neq].reset(new Eq_bc(espace.get_domain(dom), dom, bound, parse_eq(dom, nom, bound), true, n_cmp, p_cmp));

        neq++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_bc(int dom, int bound, const char* nom, const List_comp& list)
    {
        add_eq_bc(dom, bound, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_matching(int dom, int bound, const char* nom, int n_cmp, Array<int>** p_cmp,
                                        const char* owner_expr)
    {
        int other_dom;
        int other_bound;
        espace.get_domain(dom)->find_other_dom(dom, bound, other_dom, other_bound);
        assert(other_dom >= dom_min);
        assert(other_dom <= dom_max);

        eq_list.push_back(std::make_tuple(nom, dom, bound));
        record_equation_column_attachment(ColumnClass::FieldMatching, dom, bound, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_matching(espace.get_domain(dom), dom, bound, other_dom, other_bound,
                                  parse_eq_trim(dom, nom, bound, true),
                                  parse_eq_trim(other_dom, nom, other_bound, false), n_cmp, p_cmp));

        neq++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_matching(int dom, int bound, const char* nom, const List_comp& list)
    {
        add_eq_matching(dom, bound, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_matching_exception(int dom, int bound, const char* nom, const Param& par,
                                                  const char* nom_exception, int n_cmp, Array<int>** p_cmp,
                                                  const char* owner_expr)
    {
        int other_dom;
        int other_bound;
        espace.get_domain(dom)->find_other_dom(dom, bound, other_dom, other_bound);
        assert(other_dom >= dom_min);
        assert(other_dom <= dom_max);

        eq_list.push_back(std::make_tuple(nom, dom, bound));
        record_equation_column_attachment(ColumnClass::FieldMatching, dom, bound, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_matching_exception(espace.get_domain(dom), dom, bound, other_dom, other_bound,
                                            parse_eq_trim(dom, nom, bound, true),
                                            parse_eq_trim(other_dom, nom, other_bound, false), par,
                                            parse_eq_trim(dom, nom_exception, bound), n_cmp, p_cmp));
        neq++;

        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_matching_exception(int dom, int bound, const char* nom, const Param& par,
                                                  const char* nom_exception, const List_comp& list)
    {
        add_eq_matching_exception(dom, bound, nom, par, nom_exception, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_matching_one_side(int dom, int bound, const char* nom, int n_cmp,
                                                 Array<int>** p_cmp, const char* owner_expr)
    {
        int other_dom;
        int other_bound;
        espace.get_domain(dom)->find_other_dom(dom, bound, other_dom, other_bound);
        assert(other_dom >= dom_min);
        assert(other_dom <= dom_max);

        eq_list.push_back(std::make_tuple(nom, dom, bound));
        record_equation_column_attachment(ColumnClass::FieldMatching, dom, bound, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_matching_one_side(espace.get_domain(dom), dom, bound, other_dom, other_bound,
                                     parse_eq(dom, nom, bound), parse_eq(other_dom, nom, other_bound), n_cmp, p_cmp));
        neq++;

        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_matching_one_side(int dom, int bound, const char* nom, const List_comp& list)
    {
        add_eq_matching_one_side(dom, bound, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_matching_non_std(int dom, int bound, const char* nom, int n_cmp,
                                                Array<int>** p_cmp, const char* owner_expr)
    {

        // First get the number, the indices and associated boundaries of the other domains (member of espace) :
        Array<int> other_props(espace.get_indices_matching_non_std(dom, bound));

        eq_list.push_back(std::make_tuple(nom, dom, bound));
        record_equation_column_attachment(ColumnClass::FieldMatching, dom, bound, owner_expr, nom);

        // The equation
        ensure_eq_slot(); eq[neq].reset(new Eq_matching_non_std(espace.get_domain(dom), dom, bound, other_props, n_cmp, p_cmp));

        // Affectation of the operator in each concerned domain :
        // Current one :bool indic = is_ope_bin(nom, p1, p2, '=') ;
        // Is it written with =  ?
        char p1[LMAX];
        char p2[LMAX];
        bool indic = is_ope_bin(nom, p1, p2, '=');

        if (!indic) {
            char auxi[LMAX];
            trim_spaces(auxi, nom);
            // Version without =
            eq[neq]->parts[0].reset(give_ope(dom, auxi, bound));
            // The associated ones :
            for (int i = 0; i < eq[neq]->n_ope - 1; i++)
                eq[neq]->parts[i + 1].reset(give_ope(other_props(0, i), auxi, other_props(1, i)));
            neq++;
        } else {
            // Version without =
            eq[neq]->parts[0].reset(give_ope(dom, p1, bound));
            // The associated ones :
            for (int i = 0; i < eq[neq]->n_ope - 1; i++)
                eq[neq]->parts[i + 1].reset(give_ope(other_props(0, i), p2, other_props(1, i)));
            neq++;
        }

        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_matching_non_std(int dom, int bound, const char* nom, const List_comp& list)
    {
        add_eq_matching_non_std(dom, bound, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_matching_import(int dom, int bound, const char* nom, int n_cmp,
                                               Array<int>** p_cmp, const char* owner_expr,
                                               bool reflection_parity_preserving)
    {

        eq_list.push_back(std::make_tuple(nom, dom, bound));
        record_equation_column_attachment(ColumnClass::FieldMatching, dom, bound, owner_expr, nom);

        // First get the number, the indices and associated boundaries of the other domains (member of espace) :
        Array<int> others(espace.get_indices_matching_non_std(dom, bound));

        // Is it written with =  ?
        char p1[LMAX];
        char p2[LMAX];
        bool indic = is_ope_bin(nom, p1, p2, '=');

        if (!indic) {
            char auxi[LMAX];
            trim_spaces(auxi, nom);
            // Version without = ; assumes p2 = import(p1)
            Ope_eq* matching_operator = new Ope_sub(
                this, give_ope(dom, auxi, bound),
                new Ope_import(this, dom, bound, auxi));
            if (reflection_parity_preserving)
                matching_operator->set_reflection_parity_preserving();
            ensure_eq_slot(); eq[neq].reset(new Eq_matching_import(
                espace.get_domain(dom), dom, bound, matching_operator, others,
                n_cmp, p_cmp));
            neq++;
        } else {
            // Version with =
            Ope_eq* matching_operator = new Ope_sub(
                this, give_ope(dom, p1, bound), give_ope(dom, p2, bound));
            if (reflection_parity_preserving)
                matching_operator->set_reflection_parity_preserving();
            ensure_eq_slot(); eq[neq].reset(new Eq_matching_import(
                espace.get_domain(dom), dom, bound, matching_operator, others,
                n_cmp, p_cmp));
            neq++;
        }
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_matching_import(int dom, int bound, const char* nom, const List_comp& list)
    {
        add_eq_matching_import(dom, bound, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_full(int dom, const char* nom, int n_cmp, Array<int>** p_cmp,
                                    const char* owner_expr)
    {
        eq_list.push_back(std::make_tuple(nom, dom, -1));
        record_equation_column_attachment(ColumnClass::FieldInteriorVol, dom, -1, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_full(espace.get_domain(dom), dom, parse_eq(dom, nom), n_cmp, p_cmp));

        neq++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_full(int dom, const char* nom, const List_comp& list)
    {
        add_eq_full(dom, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_one_side(int dom, const char* nom, int n_cmp, Array<int>** p_cmp,
                                        const char* owner_expr)
    {
        eq_list.push_back(std::make_tuple(nom, dom, -1));
        record_equation_column_attachment(ColumnClass::FieldInteriorVol, dom, -1, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_one_side(espace.get_domain(dom), dom, parse_eq(dom, nom), n_cmp, p_cmp));
        neq++;

        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_one_side(int dom, const char* nom, const List_comp& list)
    {
        add_eq_one_side(dom, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_mode(int dom, int bound, const char* nom, const Index& pos_cf, double value)
    {
        eq_int_list.push_back(std::make_tuple(nom, dom, -1));

        char auxi[LMAX];
        trim_spaces(auxi, nom);

        ensure_eq_int_slot(); eq_int[neq_int].reset(new Eq_int(1));
        eq_int[neq_int]->set_part(0, new Ope_mode(this, bound, pos_cf, value, give_ope(dom, auxi, bound)));

        neq_int++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_val_mode(int dom, const char* nom, const Index& pos_cf, double value)
    {
        eq_int_list.push_back(std::make_tuple(nom, dom, -1));

        char auxi[LMAX];
        trim_spaces(auxi, nom);

        ensure_eq_int_slot(); eq_int[neq_int].reset(new Eq_int(1));
        eq_int[neq_int]->set_part(0, new Ope_val_mode(this, pos_cf, value, give_ope(dom, auxi)));

        neq_int++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_val(int dom, const char* nom, const Index& pos)
    {
        eq_int_list.push_back(std::make_tuple(nom, dom, -1));

        char auxi[LMAX];
        trim_spaces(auxi, nom);

        ensure_eq_int_slot(); eq_int[neq_int].reset(new Eq_int(1));
        eq_int[neq_int]->set_part(0, new Ope_val(this, pos, give_ope(dom, auxi)));

        neq_int++;
        nbr_conditions = -1;
    }

    void System_of_eqs::set_last_eq_int_reflection_sector(int sector)
    {
        if (neq_int <= 0 || static_cast<std::size_t>(neq_int) > eq_int.size() ||
            eq_int[static_cast<std::size_t>(neq_int - 1)] == nullptr) {
            KADATH_THROW("No integral equation is available for reflection-sector tagging");
        }
        eq_int[static_cast<std::size_t>(neq_int - 1)]->set_reflection_sector(sector);
    }

    void System_of_eqs::add_eq_point(int dom, const char* nom, const Point& num)
    {
        eq_int_list.push_back(std::make_tuple(nom, dom, -1));

        char auxi[LMAX];
        trim_spaces(auxi, nom);

        ensure_eq_int_slot(); eq_int[neq_int].reset(new Eq_int(1));
        eq_int[neq_int]->set_part(0, new Ope_point(this, num, give_ope(dom, auxi)));

        neq_int++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_order(int dom, const Array<int>& order, const char* nom, int n_cmp,
                                     Array<int>** p_cmp, const char* owner_expr)
    {
        eq_list.push_back(std::make_tuple(nom, dom, -1));
        record_equation_column_attachment(ColumnClass::FieldInteriorVol, dom, -1, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_order_array(espace.get_domain(dom), dom, order, parse_eq(dom, nom), n_cmp, p_cmp));

        neq++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_order(int dom, const Array<int>& order, const char* nom, const List_comp& list)
    {
        add_eq_order(dom, order, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_bc(int dom, int bound, const Array<int>& order, const char* nom, int n_cmp,
                                  Array<int>** p_cmp, const char* owner_expr)
    {
        eq_list.push_back(std::make_tuple(nom, dom, bound));
        record_equation_column_attachment(ColumnClass::FieldBoundaryTau, dom, bound, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_bc_order_array(espace.get_domain(dom), dom, bound, order, parse_eq(dom, nom, bound), n_cmp, p_cmp));

        neq++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_bc(int dom, int bound, const Array<int>& order, const char* nom, const List_comp& list)
    {
        add_eq_bc(dom, bound, order, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_matching(int dom, int bound, const Array<int>& order, const char* nom, int n_cmp,
                                        Array<int>** p_cmp, const char* owner_expr)
    {
        int other_dom;
        int other_bound;
        espace.get_domain(dom)->find_other_dom(dom, bound, other_dom, other_bound);
        assert(other_dom >= dom_min);
        assert(other_dom <= dom_max);

        eq_list.push_back(std::make_tuple(nom, dom, bound));
        record_equation_column_attachment(ColumnClass::FieldMatching, dom, bound, owner_expr, nom);

        ensure_eq_slot(); eq[neq].reset(new Eq_matching_order_array(espace.get_domain(dom), dom, bound, other_dom, other_bound, order,
                                              parse_eq_trim(dom, nom, bound, true),
                                              parse_eq_trim(other_dom, nom, other_bound, false), n_cmp, p_cmp));

        neq++;
        nbr_conditions = -1;
    }

    void System_of_eqs::add_eq_matching(int dom, int bound, const Array<int>& order, const char* nom,
                                        const List_comp& list)
    {
        add_eq_matching(dom, bound, order, nom, list.get_ncomp(), list.get_pcomp());
    }

    void System_of_eqs::add_eq_first_integral(int dom_min, int dom_max,
                                              const char* integ_part,
                                              const char* cst_part,
                                              bool same_reflection_sector)
    {
        eq_list.push_back(std::make_tuple(integ_part, dom_min, -1));
        record_equation_column_attachment(ColumnClass::FieldGauge, dom_min, -1, cst_part, integ_part);

        ensure_eq_slot(); eq[neq].reset(new Eq_first_integral(
            this, espace.get_domain(dom_min), dom_min, dom_max, integ_part,
            cst_part, same_reflection_sector));
        neq++;

        nbr_conditions = -1;
    }

} // namespace Kadath
