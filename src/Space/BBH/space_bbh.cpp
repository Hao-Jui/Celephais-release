/*
    Copyright 2020 Philippe Grandclement

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
#include "For_Kadath/Space/bbh.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
namespace Kadath
{
    double eta_lim_chi(double chi, double rext, double a, double eta_c);
    double chi_lim_eta(double chi, double rext, double a, double chi_c);

    double zerosec(double (*f)(double, const Param&), const Param& parf, double x1, double x2, double precis,
                   int nitermax, int& niter);

    double func_ab2(double aa, const Param& par)
    {
        double r1 = par.get_double(0);
        double r2 = par.get_double(1);
        double d = par.get_double(2);
        return (sqrt(aa * aa + r1 * r1) + sqrt(aa * aa + r2 * r2) - d);
    }

    Space_bbh::Space_bbh(int ttype, double dist, double rbh1, double rbh2, double rbi, double rext, int nr)
    {

        ndim = 3;

        nbr_domains = 10;
        type_base = ttype;
        domains = new Domain*[nbr_domains];

        Dim_array res(ndim);
        res.set(0) = nr;
        res.set(1) = nr;
        res.set(2) = nr - 1;
        Dim_array res_bi(ndim);
        res_bi.set(0) = nr;
        res_bi.set(1) = nr;
        res_bi.set(2) = nr;

        // Bispheric :
        // Computation of aa
        Param par_a;
        par_a.add_double(2 * rbh1, 0);
        par_a.add_double(2 * rbh2, 1);
        par_a.add_double(dist, 2);
        double a_min = 0;
        double a_max = dist / 2.;
        double precis = PRECISION;
        int nitermax = 500;
        int niter;
        double aa = zerosec(func_ab2, par_a, a_min, a_max, precis, nitermax, niter);
        double eta_plus = asinh(aa / 2 / rbh2);
        double eta_minus = -asinh(aa / 2 / rbh1);

        double chi_c = 2 * atan(aa / rbi);
        double eta_c = log((1 + rbi / aa) / (rbi / aa - 1));
        double eta_lim = eta_c / 2.;
        double chi_lim = chi_lim_eta(eta_lim, rbi, aa, chi_c);

        // First BH
        Point center_minus(ndim);
        center_minus.set(1) = aa * cosh(eta_minus) / sinh(eta_minus);
        domains[0] = new Domain_shell_outer_adapted(*this, 0, ttype, rbh1 / 2., rbh1, center_minus, res);
        domains[1] = new Domain_shell_inner_adapted(*this, 1, ttype, rbh1, rbh1 * 2., center_minus, res);

        // Second bh
        Point center_plus(ndim);
        center_plus.set(1) = aa * cosh(eta_plus) / sinh(eta_plus);
        domains[2] = new Domain_shell_outer_adapted(*this, 2, ttype, rbh2 / 2., rbh2, center_plus, res);
        domains[3] = new Domain_shell_inner_adapted(*this, 3, ttype, rbh2, rbh2 * 2., center_plus, res);

        // Bispheric part
        domains[4] = new Domain_bispheric_chi_first(4, ttype, aa, eta_minus, rbi, chi_lim, res_bi);
        domains[5] = new Domain_bispheric_rect(5, ttype, aa, rbi, eta_minus, -eta_lim, chi_lim, res_bi);
        domains[6] = new Domain_bispheric_eta_first(6, ttype, aa, rbi, -eta_lim, eta_lim, res_bi);
        domains[7] = new Domain_bispheric_rect(7, ttype, aa, rbi, eta_plus, eta_lim, chi_lim, res_bi);
        domains[8] = new Domain_bispheric_chi_first(8, ttype, aa, eta_plus, rbi, chi_lim, res_bi);

        // Compactified domain
        Point center(3);
        domains[9] = new Domain_shell(9, ttype, rbi, rext, center, res);

        const Domain_shell_outer_adapted* pouter_1 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[0]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_adapted* pinner_1 = dynamic_cast<const Domain_shell_inner_adapted*>(domains[1]);
        pinner_1->vars_to_terms();
        pinner_1->update();
        const Domain_shell_outer_adapted* pouter_2 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[2]);
        pouter_2->vars_to_terms();
        pouter_2->update();
        const Domain_shell_inner_adapted* pinner_2 = dynamic_cast<const Domain_shell_inner_adapted*>(domains[3]);
        pinner_2->vars_to_terms();
        pinner_2->update();
    }

    Space_bbh::Space_bbh(BinarySource& source)
    {
        nbr_domains = source.read<int>();
        ndim = source.read<int>();
        type_base = source.read<int>();
        domains = new Domain*[nbr_domains];
        // First BH :
        domains[0] = new Domain_shell_outer_adapted(*this, 0, source);
        domains[1] = new Domain_shell_inner_adapted(*this, 1, source);

        // second BH :
        domains[2] = new Domain_shell_outer_adapted(*this, 2, source);
        domains[3] = new Domain_shell_inner_adapted(*this, 3, source);

        // Bispheric
        domains[4] = new Domain_bispheric_chi_first(4, source);
        domains[5] = new Domain_bispheric_rect(5, source);
        domains[6] = new Domain_bispheric_eta_first(6, source);
        domains[7] = new Domain_bispheric_rect(7, source);
        domains[8] = new Domain_bispheric_chi_first(8, source);

        // Compactified
        domains[9] = new Domain_shell(9, source);

        const Domain_shell_outer_adapted* pouter_1 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[0]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_adapted* pinner_1 = dynamic_cast<const Domain_shell_inner_adapted*>(domains[1]);
        pinner_1->vars_to_terms();
        pinner_1->update();
        const Domain_shell_outer_adapted* pouter_2 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[2]);
        pouter_2->vars_to_terms();
        pouter_2->update();
        const Domain_shell_inner_adapted* pinner_2 = dynamic_cast<const Domain_shell_inner_adapted*>(domains[3]);
        pinner_2->vars_to_terms();
        pinner_2->update();
    }

    Space_bbh::~Space_bbh()
    {
        const Domain_shell_outer_adapted* pouter_1 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[0]);
        pouter_1->del_deriv();
        const Domain_shell_inner_adapted* pinner_1 = dynamic_cast<const Domain_shell_inner_adapted*>(domains[1]);
        pinner_1->del_deriv();
        const Domain_shell_outer_adapted* pouter_2 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[2]);
        pouter_2->del_deriv();
        const Domain_shell_inner_adapted* pinner_2 = dynamic_cast<const Domain_shell_inner_adapted*>(domains[3]);
        pinner_2->del_deriv();
    }

    void Space_bbh::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }

    int Space_bbh::nbr_unknowns_from_variable_domains() const
    {

        return (domains[1]->nbr_unknowns_from_adapted() + domains[3]->nbr_unknowns_from_adapted());
    }

    void Space_bbh::affecte_coef_to_variable_domains(int& pos, int cc, Array<int>& zedoms) const
    {

        // In BH 1 ?
        bool found_outer_1 = false;
        int old_pos = pos;
        domains[0]->affecte_coef(pos, cc, found_outer_1);
        pos = old_pos;
        bool found_inner_1 = false;
        domains[1]->affecte_coef(pos, cc, found_inner_1);
        assert(found_outer_1 == found_inner_1);
        if (found_outer_1) {
            zedoms.set(0) = 0;
            zedoms.set(1) = 1;
        } else {
            zedoms.set(0) = -1;
            zedoms.set(1) = -1;
        }

        // In BH 2 ?
        bool found_outer_2 = false;
        old_pos = pos;
        domains[2]->affecte_coef(pos, cc, found_outer_2);
        pos = old_pos;
        bool found_inner_2 = false;
        domains[3]->affecte_coef(pos, cc, found_inner_2);
        assert(found_outer_2 == found_inner_2);
        if (found_outer_2) {
            zedoms.set(0) = 2;
            zedoms.set(1) = 3;
        }
    }

    void Space_bbh::xx_to_ders_variable_domains(const Array<double>& xx, int& conte) const
    {
        // BH 1
        int old_conte = conte;
        domains[0]->xx_to_ders_from_adapted(xx, conte);
        conte = old_conte;
        domains[1]->xx_to_ders_from_adapted(xx, conte);
        // bh2 2
        old_conte = conte;
        domains[2]->xx_to_ders_from_adapted(xx, conte);
        conte = old_conte;
        domains[3]->xx_to_ders_from_adapted(xx, conte);
    }

    void Space_bbh::xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& xx, int& pos) const
    {

        // First get the corrections :
        // BH 1
        int old_pos = pos;
        Val_domain cor_outer_1(domains[0]);
        domains[0]->xx_to_vars_from_adapted(cor_outer_1, xx, pos);
        pos = old_pos;
        Val_domain cor_inner_1(domains[1]);
        domains[1]->xx_to_vars_from_adapted(cor_inner_1, xx, pos);

        // BH 2
        old_pos = pos;
        Val_domain cor_outer_2(domains[2]);
        domains[2]->xx_to_vars_from_adapted(cor_outer_2, xx, pos);
        pos = old_pos;
        Val_domain cor_inner_2(domains[3]);
        domains[3]->xx_to_vars_from_adapted(cor_inner_2, xx, pos);

        // Now update the variables :
        for (int i = 0; i < sys->nvar; i++) {
            for (int n = 0; n < sys->var[i]->get_n_comp(); n++) {
                Scalar res(*this);
                domains[0]->update_variable(cor_outer_1, *sys->var[i]->cmp[n], res);
                domains[1]->update_variable(cor_inner_1, *sys->var[i]->cmp[n], res);
                domains[2]->update_variable(cor_outer_2, *sys->var[i]->cmp[n], res);
                domains[3]->update_variable(cor_inner_2, *sys->var[i]->cmp[n], res);

                sys->var[i]->cmp[n]->set_domain(0) = res(0);
                sys->var[i]->cmp[n]->set_domain(1) = res(1);
                sys->var[i]->cmp[n]->set_domain(2) = res(2);
                sys->var[i]->cmp[n]->set_domain(3) = res(3);
            }
        }

        // Now the constants :
        for (int i = 0; i < sys->ncst; i++)
            if (sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->get_type_data() == TERM_T)
                for (int n = 0; n < sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->val_t->get_n_comp(); n++) {

                    Scalar so(*this);
                    so = 0;
                    for (int d = 0; d <= 3; d++)
                        so.set_domain(d) = (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + d]->val_t->cmp[n])(d);

                    Scalar res(*this);
                    res = 0;
                    domains[0]->update_constante(cor_outer_1, so, res);
                    domains[1]->update_constante(cor_inner_1, so, res);
                    domains[2]->update_constante(cor_outer_2, so, res);
                    domains[3]->update_constante(cor_inner_2, so, res);

                    sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->val_t->cmp[n]->set_domain(0) = res(0);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + 1]->val_t->cmp[n]->set_domain(1) = res(1);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + 2]->val_t->cmp[n]->set_domain(2) = res(2);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + 3]->val_t->cmp[n]->set_domain(3) = res(3);
                }

        // Update the mapping :
        domains[0]->update_mapping(cor_outer_1);
        domains[1]->update_mapping(cor_inner_1);
        domains[2]->update_mapping(cor_outer_2);
        domains[3]->update_mapping(cor_inner_2);
    }

    Array<int> Space_bbh::get_indices_matching_non_std(int dom, int bound) const
    {

        if (dom == 1) {
            // First BH ;
            Array<int> res(2, 2);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = 4; // Matching with chi first
                    res.set(1, 0) = INNER_BC;
                    res.set(0, 1) = 5; // Matching with rect
                    res.set(1, 1) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bbh::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 3) {
            // second bh ;
            Array<int> res(2, 2);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = 7; // Matching with rect
                    res.set(1, 0) = INNER_BC;
                    res.set(0, 1) = 8; // Matching with chi_first
                    res.set(1, 1) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bbh::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 4) {
            // first chi first :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = 1; // First bh
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = 9; // Shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bbh::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 5) {
            // first rect :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = 1; // First bh
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = 9; // Shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bbh::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 6) {
            // eta first
            Array<int> res(2, 1);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = 9; // Shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bbh::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 7) {
            // second rect :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = 3; // Second bh
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = 9; // Shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bbh::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 8) {
            // second chi first :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = 3; // second nucleus
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = 9; // Shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bbh::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == 9) {
            // Shell :
            Array<int> res(2, 5);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = 4; // first chi first
                    res.set(0, 1) = 5; // first rect
                    res.set(0, 2) = 6; // eta first
                    res.set(0, 3) = 7; // second rect
                    res.set(0, 4) = 8; // second chi first
                    // Outer BC for all :
                    for (int i = 0; i < 5; i++)
                        res.set(1, i) = OUTER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bbh::get_indices_matching_non_std");
            }
            return res;
        }

        KADATH_THROW("Bad domain in Space_bbh::get_indices_matching_non_std");
    }

} // namespace Kadath
