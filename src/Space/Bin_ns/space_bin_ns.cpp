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
#include "For_Kadath/Space/bin_ns.hpp"
#include "For_Kadath/Space/binary_co_domains.hpp"
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

    double func_abns(double aa, const Param& par)
    {
        double r1 = par.get_double(0);
        double r2 = par.get_double(1);
        double d = par.get_double(2);
        return (sqrt(aa * aa + r1 * r1) + sqrt(aa * aa + r2 * r2) - d);
    }

    Space_bin_ns::Space_bin_ns(int ttype, double dist, double rinstar1, double rstar1, double routstar1,
                               double rinstar2, double rstar2, double routstar2, double rext, int nr, int nshells1,
                               int nshells2)
    {

        ndim = 3;

        // The legacy scalar-radius constructor places every requested shell
        // outside the inner adapted shell, i.e. they are outer (post-adapted)
        // shells in the BHNS-aligned ordering. No outer (exterior) shells.
        n_shells1 = nshells1;
        n_shells2 = nshells2;

        NS1 = 0;
        ADAPTED1 = NS1 + 1;
        NS2 = ADAPTED1 + 2 + n_shells1;
        ADAPTED2 = NS2 + 1;
        OUTER = ADAPTED2 + 2 + n_shells2;

        n_shells_outer = 0;
        nbr_domains = 12 + n_shells1 + n_shells2;
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
        par_a.add_double(routstar1, 0);
        par_a.add_double(routstar2, 1);
        par_a.add_double(dist, 2);
        double a_min = 0;
        double a_max = dist / 2.;
        double precis = PRECISION;
        int nitermax = 500;
        int niter;
        double aa = zerosec(func_abns, par_a, a_min, a_max, precis, nitermax, niter);
        double eta_plus = asinh(aa / routstar2);
        double eta_minus = -asinh(aa / routstar1);

        double chi_c = 2 * atan(aa / rext);
        double eta_c = log((1 + rext / aa) / (rext / aa - 1));
        double eta_lim = eta_c / 2.;
        double chi_lim = chi_lim_eta(eta_lim, rext, aa, chi_c);

        // Assemble per-star radial boundaries in the BHNS layout
        // [rin, rmid, rout, outer shells..., r_bisph]. This legacy constructor
        // treats routstar as the fixed outer target and subdivides the shell band.
        auto build_star_bounds = [](double rin, double rmid, double rout, int outer_shells) {
            std::vector<double> bounds(outer_shells + 3);
            bounds[0] = rin;
            bounds[1] = rmid;
            const double shell_width = (rout - rmid) / (outer_shells + 1.);
            for (int i = 0; i <= outer_shells; ++i)
                bounds[2 + i] = rmid + shell_width * (i + 1);
            return bounds;
        };

        // First NS
        Point center_minus(ndim);
        center_minus.set(1) = aa * cosh(eta_minus) / sinh(eta_minus);
        std::vector<double> bounds1 = build_star_bounds(rinstar1, rstar1, routstar1, n_shells1);
        build_adapted_star_domains(*this, domains, NS1, ADAPTED1, ttype, bounds1, n_shells1, center_minus, res);

        // Second NS
        Point center_plus(ndim);
        center_plus.set(1) = aa * cosh(eta_plus) / sinh(eta_plus);
        std::vector<double> bounds2 = build_star_bounds(rinstar2, rstar2, routstar2, n_shells2);
        build_adapted_star_domains(*this, domains, NS2, ADAPTED2, ttype, bounds2, n_shells2, center_plus, res);

        // Bispheric part
        domains[OUTER] = new Domain_bispheric_chi_first(OUTER, ttype, aa, eta_minus, rext, chi_lim, res_bi);
        domains[OUTER + 1] =
            new Domain_bispheric_rect(OUTER + 1, ttype, aa, rext, eta_minus, -eta_lim, chi_lim, res_bi);
        domains[OUTER + 2] = new Domain_bispheric_eta_first(OUTER + 2, ttype, aa, rext, -eta_lim, eta_lim, res_bi);
        domains[OUTER + 3] = new Domain_bispheric_rect(OUTER + 3, ttype, aa, rext, eta_plus, eta_lim, chi_lim, res_bi);
        domains[OUTER + 4] = new Domain_bispheric_chi_first(OUTER + 4, ttype, aa, eta_plus, rext, chi_lim, res_bi);

        // Compactified domain
        Point center(3);
        domains[OUTER + 5] = new Domain_compact(OUTER + 5, ttype, rext, center, res);

        const Domain_shell_outer_adapted* pouter_1 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTED1]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_adapted* pinner_1 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTED1 + 1]);
        pinner_1->vars_to_terms();
        pinner_1->update();
        const Domain_shell_outer_adapted* pouter_2 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTED2]);
        pouter_2->vars_to_terms();
        pouter_2->update();
        const Domain_shell_inner_adapted* pinner_2 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTED2 + 1]);
        pinner_2->vars_to_terms();
        pinner_2->update();
    }

    Space_bin_ns::Space_bin_ns(int ttype, double dist, double rinstar1, double rstar1, double routstar1,
                               double rinstar2, double rstar2, double routstar2, double rext, double rshell, int nr)
    {

        ndim = 3;

        n_shells1 = 0;
        n_shells2 = 0;

        NS1 = 0;
        ADAPTED1 = NS1 + 1;
        NS2 = ADAPTED1 + 2 + n_shells1;
        ADAPTED2 = NS2 + 1;
        OUTER = ADAPTED2 + 2 + n_shells2;

        n_shells_outer = 1;
        nbr_domains = 13;
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
        par_a.add_double(routstar1, 0);
        par_a.add_double(routstar2, 1);
        par_a.add_double(dist, 2);
        double a_min = 0;
        double a_max = dist / 2.;
        double precis = PRECISION;
        int nitermax = 500;
        int niter;
        double aa = zerosec(func_abns, par_a, a_min, a_max, precis, nitermax, niter);
        double eta_plus = asinh(aa / routstar2);
        double eta_minus = -asinh(aa / routstar1);

        double chi_c = 2 * atan(aa / rext);
        double eta_c = log((1 + rext / aa) / (rext / aa - 1));
        double eta_lim = eta_c / 2.;
        double chi_lim = chi_lim_eta(eta_lim, rext, aa, chi_c);

        // First NS
        Point center_minus(ndim);
        center_minus.set(1) = aa * cosh(eta_minus) / sinh(eta_minus);
        std::vector<double> bounds1{rinstar1, rstar1, routstar1};
        build_adapted_star_domains(*this, domains, NS1, ADAPTED1, ttype, bounds1, n_shells1, center_minus, res);

        // Second NS
        Point center_plus(ndim);
        center_plus.set(1) = aa * cosh(eta_plus) / sinh(eta_plus);
        std::vector<double> bounds2{rinstar2, rstar2, routstar2};
        build_adapted_star_domains(*this, domains, NS2, ADAPTED2, ttype, bounds2, n_shells2, center_plus, res);

        // Bispheric part
        domains[6] = new Domain_bispheric_chi_first(6, ttype, aa, eta_minus, rext, chi_lim, res_bi);
        domains[7] = new Domain_bispheric_rect(7, ttype, aa, rext, eta_minus, -eta_lim, chi_lim, res_bi);
        domains[8] = new Domain_bispheric_eta_first(8, ttype, aa, rext, -eta_lim, eta_lim, res_bi);
        domains[9] = new Domain_bispheric_rect(9, ttype, aa, rext, eta_plus, eta_lim, chi_lim, res_bi);
        domains[10] = new Domain_bispheric_chi_first(10, ttype, aa, eta_plus, rext, chi_lim, res_bi);

        // Shell in 1/R
        Point center(3);
        domains[11] = new Domain_shell(11, ttype, rext, rshell, center, res);

        // Compactified domain
        domains[12] = new Domain_compact(12, ttype, rshell, center, res);

        const Domain_shell_outer_adapted* pouter_1 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[1]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_adapted* pinner_1 = dynamic_cast<const Domain_shell_inner_adapted*>(domains[2]);
        pinner_1->vars_to_terms();
        pinner_1->update();
        const Domain_shell_outer_adapted* pouter_2 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[4]);
        pouter_2->vars_to_terms();
        pouter_2->update();
        const Domain_shell_inner_adapted* pinner_2 = dynamic_cast<const Domain_shell_inner_adapted*>(domains[5]);
        pinner_2->vars_to_terms();
        pinner_2->update();
    }

    Space_bin_ns::Space_bin_ns(BinarySource& source, bool old)
    {
        nbr_domains = source.read<int>();
        // Two per-star int slots. They once held the (now-retired) inner-shell
        // counts and were always 0; they now persist the per-star outer
        // (post-adapted) shell counts n_shells1/2 so a space with NSHELLS>0
        // round-trips. Legacy files (old) and every pre-NSHELLS file carry 0
        // here, which is the correct n_shells for them.
        if (!old) {
            n_shells1 = source.read<int>();
            n_shells2 = source.read<int>();
        } else {
            n_shells1 = 0;
            n_shells2 = 0;
        }
        ndim = source.read<int>();
        type_base = source.read<int>();

        // Total = 12 base domains (2 nuclei + 2 adapted pairs + 5 bispheric +
        // 1 compact) + per-star outer shells + exterior (binary-surrounding) shells.
        n_shells_outer = nbr_domains - 12 - n_shells1 - n_shells2;

        NS1 = 0;
        ADAPTED1 = NS1 + 1;
        NS2 = ADAPTED1 + 2 + n_shells1;
        ADAPTED2 = NS2 + 1;
        OUTER = ADAPTED2 + 2 + n_shells2;

        domains = new Domain*[nbr_domains];

        // First and second NS (shared adapted-CO read path):
        read_adapted_star_domains<Domain_nucleus, Domain_shell_outer_adapted, Domain_shell_inner_adapted,
                                  Domain_shell>(*this, domains, NS1, ADAPTED1, n_shells1, source);
        read_adapted_star_domains<Domain_nucleus, Domain_shell_outer_adapted, Domain_shell_inner_adapted,
                                  Domain_shell>(*this, domains, NS2, ADAPTED2, n_shells2, source);

        // Bispheric
        domains[OUTER] = new Domain_bispheric_chi_first(OUTER, source);
        domains[OUTER + 1] = new Domain_bispheric_rect(OUTER + 1, source);
        domains[OUTER + 2] = new Domain_bispheric_eta_first(OUTER + 2, source);
        domains[OUTER + 3] = new Domain_bispheric_rect(OUTER + 3, source);
        domains[OUTER + 4] = new Domain_bispheric_chi_first(OUTER + 4, source);

        for (int i = 0; i < n_shells_outer; i++)
            domains[OUTER + 5 + i] = new Domain_shell(OUTER + 5 + i, source);

        // Compactified
        domains[OUTER + 5 + n_shells_outer] = new Domain_compact(OUTER + 5 + n_shells_outer, source);

        const Domain_shell_outer_adapted* pouter_1 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTED1]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_adapted* pinner_1 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTED1 + 1]);
        pinner_1->vars_to_terms();
        pinner_1->update();
        const Domain_shell_outer_adapted* pouter_2 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTED2]);
        pouter_2->vars_to_terms();
        pouter_2->update();
        const Domain_shell_inner_adapted* pinner_2 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTED2 + 1]);
        pinner_2->vars_to_terms();
        pinner_2->update();
    }

    Space_bin_ns::~Space_bin_ns()
    {
        const Domain_shell_outer_adapted* pouter_1 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTED1]);
        pouter_1->del_deriv();
        const Domain_shell_inner_adapted* pinner_1 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTED1 + 1]);
        pinner_1->del_deriv();
        const Domain_shell_outer_adapted* pouter_2 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTED2]);
        pouter_2->del_deriv();
        const Domain_shell_inner_adapted* pinner_2 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTED2 + 1]);
        pinner_2->del_deriv();
    }

    void Space_bin_ns::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        // Persist the per-star outer (post-adapted) shell counts in the two slots
        // that formerly held the retired inner-shell counts (always 0). This lets
        // the deserializer rebuild the per-star outer shells and tell them apart
        // from exterior shells; pre-NSHELLS files carry 0 here, as before.
        sink.write<int>(n_shells1);
        sink.write<int>(n_shells2);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }

    int Space_bin_ns::nbr_unknowns_from_variable_domains() const
    {

        return (domains[ADAPTED1]->nbr_unknowns_from_adapted() + domains[ADAPTED2]->nbr_unknowns_from_adapted());
    }

    bool Space_bin_ns::describe_variable_domain_blocks(
        std::vector<VariableDomainBlock>& blocks) const
    {
        blocks.clear();
        int first_column = 0;
        for (const int adapted : {ADAPTED1, ADAPTED2}) {
            const int count = domains[adapted]->nbr_unknowns_from_adapted();
            if (count <= 0)
                return false;
            blocks.push_back({adapted, first_column, count});
            first_column += count;
        }
        return first_column == nbr_unknowns_from_variable_domains();
    }

    void Space_bin_ns::affecte_coef_to_variable_domains(int& pos, int cc, Array<int>& zedoms) const
    {

        // In star 1 ?
        bool found_outer_1 = false;
        int old_pos = pos;
        domains[ADAPTED1]->affecte_coef(pos, cc, found_outer_1);
        pos = old_pos;
        bool found_inner_1 = false;
        domains[ADAPTED1 + 1]->affecte_coef(pos, cc, found_inner_1);
        assert(found_outer_1 == found_inner_1);
        if (found_outer_1) {
            zedoms.set(0) = ADAPTED1;
            zedoms.set(1) = ADAPTED1 + 1;
        } else {
            zedoms.set(0) = -1;
            zedoms.set(1) = -1;
        }

        // In star 2 ?
        bool found_outer_2 = false;
        old_pos = pos;
        domains[ADAPTED2]->affecte_coef(pos, cc, found_outer_2);
        pos = old_pos;
        bool found_inner_2 = false;
        domains[ADAPTED2 + 1]->affecte_coef(pos, cc, found_inner_2);
        assert(found_outer_2 == found_inner_2);
        if (found_outer_2) {
            zedoms.set(0) = ADAPTED2;
            zedoms.set(1) = ADAPTED2 + 1;
        }
    }

    void Space_bin_ns::xx_to_ders_variable_domains(const Array<double>& xx, int& conte) const
    {

        // Star 1
        int old_conte = conte;
        domains[ADAPTED1]->xx_to_ders_from_adapted(xx, conte);
        conte = old_conte;
        domains[ADAPTED1 + 1]->xx_to_ders_from_adapted(xx, conte);
        // Star 2
        old_conte = conte;
        domains[ADAPTED2]->xx_to_ders_from_adapted(xx, conte);
        conte = old_conte;
        domains[ADAPTED2 + 1]->xx_to_ders_from_adapted(xx, conte);
    }

    void Space_bin_ns::xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& xx, int& pos) const
    {

        // First get the corrections :
        // Star 1
        int old_pos = pos;
        Val_domain cor_outer_1(domains[ADAPTED1]);
        domains[ADAPTED1]->xx_to_vars_from_adapted(cor_outer_1, xx, pos);
        pos = old_pos;
        Val_domain cor_inner_1(domains[ADAPTED1 + 1]);
        domains[ADAPTED1 + 1]->xx_to_vars_from_adapted(cor_inner_1, xx, pos);

        // Star 2
        old_pos = pos;
        Val_domain cor_outer_2(domains[ADAPTED2]);
        domains[ADAPTED2]->xx_to_vars_from_adapted(cor_outer_2, xx, pos);
        pos = old_pos;
        Val_domain cor_inner_2(domains[ADAPTED2 + 1]);
        domains[ADAPTED2 + 1]->xx_to_vars_from_adapted(cor_inner_2, xx, pos);

        // Now update the variables :
        for (int i = 0; i < sys->nvar; i++) {
            for (int n = 0; n < sys->var[i]->get_n_comp(); n++) {
                Scalar res(*this);
                domains[ADAPTED1]->update_variable(cor_outer_1, *sys->var[i]->cmp[n], res);
                domains[ADAPTED1 + 1]->update_variable(cor_inner_1, *sys->var[i]->cmp[n], res);
                domains[ADAPTED2]->update_variable(cor_outer_2, *sys->var[i]->cmp[n], res);
                domains[ADAPTED2 + 1]->update_variable(cor_inner_2, *sys->var[i]->cmp[n], res);

                sys->var[i]->cmp[n]->set_domain(ADAPTED1) = res(ADAPTED1);
                sys->var[i]->cmp[n]->set_domain(ADAPTED1 + 1) = res(ADAPTED1 + 1);
                sys->var[i]->cmp[n]->set_domain(ADAPTED2) = res(ADAPTED2);
                sys->var[i]->cmp[n]->set_domain(ADAPTED2 + 1) = res(ADAPTED2 + 1);
            }
        }

        // Now the constants :
        for (int i = 0; i < sys->ncst; i++)
            if (sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->get_type_data() == TERM_T)
                for (int n = 0; n < sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->val_t->get_n_comp(); n++) {

                    Scalar so(*this);
                    so = 0;
                    so.set_domain(ADAPTED1) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTED1]->val_t->cmp[n])(ADAPTED1);
                    so.set_domain(ADAPTED1 + 1) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTED1 + 1]->val_t->cmp[n])(ADAPTED1 + 1);
                    so.set_domain(ADAPTED2) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTED2]->val_t->cmp[n])(ADAPTED2);
                    so.set_domain(ADAPTED2 + 1) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTED2 + 1]->val_t->cmp[n])(ADAPTED2 + 1);

                    Scalar res(*this);
                    res = 0;
                    domains[ADAPTED1]->update_constante(cor_outer_1, so, res);
                    domains[ADAPTED1 + 1]->update_constante(cor_inner_1, so, res);
                    domains[ADAPTED2]->update_constante(cor_outer_2, so, res);
                    domains[ADAPTED2 + 1]->update_constante(cor_inner_2, so, res);

                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTED1]->val_t->cmp[n]->set_domain(ADAPTED1) =
                        res(ADAPTED1);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTED1 + 1]->val_t->cmp[n]->set_domain(
                        ADAPTED1 + 1) = res(ADAPTED1 + 1);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTED2]->val_t->cmp[n]->set_domain(ADAPTED2) =
                        res(ADAPTED2);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTED2 + 1]->val_t->cmp[n]->set_domain(
                        ADAPTED2 + 1) = res(ADAPTED2 + 1);
                }

        // Update the mapping :
        domains[ADAPTED1]->update_mapping(cor_outer_1);
        domains[ADAPTED1 + 1]->update_mapping(cor_inner_1);
        domains[ADAPTED2]->update_mapping(cor_outer_2);
        domains[ADAPTED2 + 1]->update_mapping(cor_inner_2);
    }

    Array<int> Space_bin_ns::get_indices_matching_non_std(int dom, int bound) const
    {
        // Star-internal fixed-radius seams: nucleus outer face <-> ADAPTED outer
        // shell inner face, and shell-to-shell seams between post-adapted shells.
        // These seams are traversed with add_eq_matching_import, which queries
        // get_indices_matching_non_std on the domain that holds the value condition
        // to discover the neighbour that holds the derivative condition.
        //
        // Star 1: NS1.OUTER <-> ADAPTED1.INNER, then for each shell i in
        // [0, n_shells1): (ADAPTED1+1+i).OUTER <-> (ADAPTED1+2+i).INNER.
        if (dom == NS1 && bound == OUTER_BC) {
            Array<int> res(2, 1);
            res.set(0, 0) = ADAPTED1;
            res.set(1, 0) = INNER_BC;
            return res;
        }
        if (dom == ADAPTED1 && bound == INNER_BC) {
            Array<int> res(2, 1);
            res.set(0, 0) = NS1;
            res.set(1, 0) = OUTER_BC;
            return res;
        }
        for (int i = 0; i < n_shells1; ++i) {
            if (dom == ADAPTED1 + 1 + i && bound == OUTER_BC) {
                Array<int> res(2, 1);
                res.set(0, 0) = ADAPTED1 + 2 + i;
                res.set(1, 0) = INNER_BC;
                return res;
            }
            if (dom == ADAPTED1 + 2 + i && bound == INNER_BC) {
                Array<int> res(2, 1);
                res.set(0, 0) = ADAPTED1 + 1 + i;
                res.set(1, 0) = OUTER_BC;
                return res;
            }
        }

        // Star 2: NS2.OUTER <-> ADAPTED2.INNER, then for each shell i in
        // [0, n_shells2): (ADAPTED2+1+i).OUTER <-> (ADAPTED2+2+i).INNER.
        if (dom == NS2 && bound == OUTER_BC) {
            Array<int> res(2, 1);
            res.set(0, 0) = ADAPTED2;
            res.set(1, 0) = INNER_BC;
            return res;
        }
        if (dom == ADAPTED2 && bound == INNER_BC) {
            Array<int> res(2, 1);
            res.set(0, 0) = NS2;
            res.set(1, 0) = OUTER_BC;
            return res;
        }
        for (int i = 0; i < n_shells2; ++i) {
            if (dom == ADAPTED2 + 1 + i && bound == OUTER_BC) {
                Array<int> res(2, 1);
                res.set(0, 0) = ADAPTED2 + 2 + i;
                res.set(1, 0) = INNER_BC;
                return res;
            }
            if (dom == ADAPTED2 + 2 + i && bound == INNER_BC) {
                Array<int> res(2, 1);
                res.set(0, 0) = ADAPTED2 + 1 + i;
                res.set(1, 0) = OUTER_BC;
                return res;
            }
        }

        // The star-to-bispheric matching happens at the last domain of each star
        // before the bispheric region: the inner adapted shell plus any outer
        // (post-adapted) shells.
        if (dom == ADAPTED1 + 1 + n_shells1) {
            // First star ;
            Array<int> res(2, 2);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = OUTER; // Matching with chi first
                    res.set(1, 0) = INNER_BC;
                    res.set(0, 1) = OUTER + 1; // Matching with rect
                    res.set(1, 1) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_ns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == ADAPTED2 + 1 + n_shells2) {
            // second star ;
            Array<int> res(2, 2);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 3; // Matching with rect
                    res.set(1, 0) = INNER_BC;
                    res.set(0, 1) = OUTER + 4; // Matching with chi_first
                    res.set(1, 1) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_ns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == OUTER) {
            // first chi first :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = ADAPTED1 + 1 + n_shells1; // First star
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 5; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_ns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == OUTER + 1) {
            // first rect :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = ADAPTED1 + 1 + n_shells1; // First star
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 5; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_ns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == OUTER + 2) {
            // eta first
            Array<int> res(2, 1);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 5; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_ns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == OUTER + 3) {
            // second rect :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = ADAPTED2 + 1 + n_shells2; // Second star
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 5; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_ns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == OUTER + 4) {
            // second chi first :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = ADAPTED2 + 1 + n_shells2; // second nucleus
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 5; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_ns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == OUTER + 5) {
            // compactified domain or first shell :
            Array<int> res(2, 5);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = OUTER;     // first chi first
                    res.set(0, 1) = OUTER + 1; // first rect
                    res.set(0, 2) = OUTER + 2; // eta first
                    res.set(0, 3) = OUTER + 3; // second rect
                    res.set(0, 4) = OUTER + 4; // second chi first
                    // Outer BC for all :
                    for (int i = 0; i < 5; i++)
                        res.set(1, i) = OUTER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bin_ns::get_indices_matching_non_std");
            }
            return res;
        }

        KADATH_THROW("Bad domain in Space_bin_ns::get_indices_matching_non_std");
    }

    namespace {
        // Domain count implied by the vector-bounds layout: 12 fixed domains
        // (2 nuclei, 2 adapted pairs, 5 bispheric, 1 compact) + per-star outer
        // shells + exterior shells. Must stay in sync with the constructor body.
        int bin_ns_domain_count(const std::vector<double>& NS1_bounds, const std::vector<double>& NS2_bounds,
                                const std::vector<double>& outer_bounds)
        {
            return 12 + (int(NS1_bounds.size()) - 3) + (int(NS2_bounds.size()) - 3) + (int(outer_bounds.size()) - 1);
        }
    } // namespace

    Space_bin_ns::Space_bin_ns(int ttype, double dist, const std::vector<double>& NS1_bounds,
                               const std::vector<double>& NS2_bounds, const std::vector<double>& outer_bounds, int nr)
        : Space_bin_ns(ttype, dist, NS1_bounds, NS2_bounds, outer_bounds,
                       std::vector<int>(bin_ns_domain_count(NS1_bounds, NS2_bounds, outer_bounds), nr))
    {
    }

    Space_bin_ns::Space_bin_ns(int ttype, double dist, const std::vector<double>& NS1_bounds,
                               const std::vector<double>& NS2_bounds, const std::vector<double>& outer_bounds,
                               const std::vector<int>& nr_per_domain)
        : Space_bin_ns(ttype, dist, NS1_bounds, NS2_bounds, outer_bounds,
                       [&] {
                           // Radial-only path: pin the angular counts to the shared
                           // base everywhere (theta = base, phi = base-1 spherical,
                           // base in all three dims bispheric) exactly as the legacy
                           // spherical_res/bispheric_res lambdas did, then delegate to
                           // the full per-domain Dim_array constructor. The base is
                           // the bispheric radial count; the bispheric-share check
                           // below would otherwise read an undefined base.
                           const int count = bin_ns_domain_count(NS1_bounds, NS2_bounds, outer_bounds);
                           if (int(nr_per_domain.size()) != count)
                               KADATH_THROW("Space_bin_ns: nr_per_domain must have one entry per domain");
                           const int n_sh1 = int(NS1_bounds.size()) - 3;
                           const int n_sh2 = int(NS2_bounds.size()) - 3;
                           const int outer_index = (n_sh1 + 1) + (n_sh2 + 1) + 4; // == OUTER
                           const int base_nr = nr_per_domain[outer_index];
                           std::vector<Dim_array> res_per_domain;
                           res_per_domain.reserve(count);
                           for (int d = 0; d < count; ++d) {
                               const bool bispheric = (d >= outer_index) && (d < outer_index + 5);
                               Dim_array r(3);
                               r.set(0) = nr_per_domain[d];
                               r.set(1) = base_nr;
                               r.set(2) = bispheric ? base_nr : base_nr - 1;
                               res_per_domain.push_back(r);
                           }
                           return res_per_domain;
                       }())
    {
    }

    Space_bin_ns::Space_bin_ns(int ttype, double dist, const std::vector<double>& NS1_bounds,
                               const std::vector<double>& NS2_bounds, const std::vector<double>& outer_bounds,
                               const std::vector<Dim_array>& res_per_domain)
    {

        ndim = 3;

        double rext = outer_bounds[0];
        // Bounds layout per star: [rin, rmid, rout, outer shells..., r_bisph].
        // Every entry past rout is an outer (post-adapted) shell.
        n_shells1 = int(NS1_bounds.size()) - 3;
        n_shells2 = int(NS2_bounds.size()) - 3;

        NS1 = 0;
        ADAPTED1 = NS1 + 1;
        NS2 = ADAPTED1 + 2 + n_shells1;
        ADAPTED2 = NS2 + 1;
        OUTER = ADAPTED2 + 2 + n_shells2;

        n_shells_outer = outer_bounds.size() - 1;
        nbr_domains = 12 + n_shells1 + n_shells2 + n_shells_outer;
        type_base = ttype;
        domains = new Domain*[nbr_domains]();

        if (int(res_per_domain.size()) != nbr_domains)
            KADATH_THROW("Space_bin_ns: res_per_domain must have one entry per domain");
        for (int d = 0; d < nbr_domains; ++d) {
            if (res_per_domain[d].get_ndim() != ndim)
                KADATH_THROW("Space_bin_ns: each per-domain Dim_array must be 3-dimensional");
            if (res_per_domain[d](0) < 5)
                KADATH_THROW("Space_bin_ns: per-domain radial resolution must be at least 5");
            if (res_per_domain[d](1) < 5)
                KADATH_THROW("Space_bin_ns: per-domain theta resolution must be at least 5");
        }

        // Per-domain ANGULAR (theta/phi) refinement is allowed: fixed-radius
        // star-internal seams are coupled with split Eq_matching_import, so a
        // theta/phi jump across them stays square in coefficient space. Two seams
        // stay conforming and therefore constrain the angular counts:
        // (a) the five bispheric domains share an identical Dim_array — their
        //     mutual seams couple every dimension;
        // (b) each adapted pair shares the one deformable surface object, so the
        //     pair must agree in theta AND phi (the surface seam is conforming).
        const Dim_array& base = res_per_domain[OUTER];
        for (int d = OUTER; d < OUTER + 5; ++d)
            if (res_per_domain[d](0) != base(0) || res_per_domain[d](1) != base(1) ||
                res_per_domain[d](2) != base(2))
                KADATH_THROW("Space_bin_ns: the five bispheric domains must share an identical resolution");
        auto require_pair_angular_match = [&](int adapted) {
            if (res_per_domain[adapted](1) != res_per_domain[adapted + 1](1) ||
                res_per_domain[adapted](2) != res_per_domain[adapted + 1](2))
                KADATH_THROW("Space_bin_ns: each adapted pair must share theta and phi (it shares the surface)");
        };
        require_pair_angular_match(ADAPTED1);
        require_pair_angular_match(ADAPTED2);

        const std::vector<Dim_array>& star_res = res_per_domain;

        // Bispheric :
        // Computation of aa
        Param par_a;
        par_a.add_double(NS1_bounds.back(), 0);
        par_a.add_double(NS2_bounds.back(), 1);
        par_a.add_double(dist, 2);
        double a_min = 0;
        double a_max = dist / 2.;
        double precis = PRECISION;
        int nitermax = 500;
        int niter;
        double aa = zerosec(func_abns, par_a, a_min, a_max, precis, nitermax, niter);
        double eta_plus = asinh(aa / NS2_bounds.back());
        double eta_minus = -asinh(aa / NS1_bounds.back());

        double chi_c = 2 * atan(aa / rext);
        double eta_c = log((1 + rext / aa) / (rext / aa - 1));
        double eta_lim = eta_c / 2.;
        double chi_lim = chi_lim_eta(eta_lim, rext, aa, chi_c);

        Point center_minus(ndim);
        center_minus.set(1) = aa * cosh(eta_minus) / sinh(eta_minus);
        build_adapted_star_domains(*this, domains, NS1, ADAPTED1, ttype, NS1_bounds, n_shells1, center_minus,
                                   star_res);

        Point center_plus(ndim);
        center_plus.set(1) = aa * cosh(eta_plus) / sinh(eta_plus);
        build_adapted_star_domains(*this, domains, NS2, ADAPTED2, ttype, NS2_bounds, n_shells2, center_plus,
                                   star_res);

        // Bispheric part
        domains[OUTER] =
            new Domain_bispheric_chi_first(OUTER, ttype, aa, eta_minus, rext, chi_lim, res_per_domain[OUTER]);
        domains[OUTER + 1] = new Domain_bispheric_rect(OUTER + 1, ttype, aa, rext, eta_minus, -eta_lim, chi_lim,
                                                       res_per_domain[OUTER + 1]);
        domains[OUTER + 2] =
            new Domain_bispheric_eta_first(OUTER + 2, ttype, aa, rext, -eta_lim, eta_lim, res_per_domain[OUTER + 2]);
        domains[OUTER + 3] = new Domain_bispheric_rect(OUTER + 3, ttype, aa, rext, eta_plus, eta_lim, chi_lim,
                                                       res_per_domain[OUTER + 3]);
        domains[OUTER + 4] =
            new Domain_bispheric_chi_first(OUTER + 4, ttype, aa, eta_plus, rext, chi_lim, res_per_domain[OUTER + 4]);

        Point center(3);
        for (int i = 0; i < n_shells_outer; i++)
            domains[OUTER + 5 + i] = new Domain_shell(OUTER + 5 + i, ttype, outer_bounds[i], outer_bounds[i + 1],
                                                      center, res_per_domain[OUTER + 5 + i]);

        // Compactified
        domains[OUTER + 5 + n_shells_outer] =
            new Domain_compact(OUTER + 5 + n_shells_outer, ttype, outer_bounds[n_shells_outer], center,
                               res_per_domain[OUTER + 5 + n_shells_outer]);

        const Domain_shell_outer_adapted* pouter_1 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTED1]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_adapted* pinner_1 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTED1 + 1]);
        pinner_1->vars_to_terms();
        pinner_1->update();
        const Domain_shell_outer_adapted* pouter_2 = dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTED2]);
        pouter_2->vars_to_terms();
        pouter_2->update();
        const Domain_shell_inner_adapted* pinner_2 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTED2 + 1]);
        pinner_2->vars_to_terms();
        pinner_2->update();
    }
} // namespace Kadath
