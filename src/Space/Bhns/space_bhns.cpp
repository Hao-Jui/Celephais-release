
/*
    Copyright 2020 Samuel Tootle

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
#include "For_Kadath/Space/bhns.hpp"
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

    double func_abhns(double aa, const Param& par)
    {
        double r1 = par.get_double(0);
        double r2 = par.get_double(1);
        double d = par.get_double(2);
        return (sqrt(aa * aa + r1 * r1) + sqrt(aa * aa + r2 * r2) - d);
    }

    Space_bhns::Space_bhns(int ttype, double dist, const std::vector<double>& NS_bounds,
                           const std::vector<double>& BH_bounds, const std::vector<double>& outer_bounds, int nr)
    {

        ndim = 3;
        double rext = outer_bounds[0];

        n_shells1 = NS_bounds.size() - 3;
        n_shells2 = BH_bounds.size() - 3;

        NS = 0;
        BH = 3 + n_shells1;
        ADAPTEDNS = NS + 1;
        ADAPTEDBH = BH + 1;
        OUTER = 6 + n_shells1 + n_shells2;

        n_shells_outer = outer_bounds.size() - 1;
        nbr_domains = 12 + n_shells1 + n_shells2 + n_shells_outer;
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
        par_a.add_double(NS_bounds.back(), 0);
        par_a.add_double(BH_bounds.back(), 1);
        par_a.add_double(dist, 2);
        double a_min = 0;
        double a_max = dist / 2.;
        double precis = PRECISION;
        int nitermax = 500;
        int niter;
        double aa = zerosec(func_abhns, par_a, a_min, a_max, precis, nitermax, niter);
        double eta_plus = asinh(aa / BH_bounds.back());
        double eta_minus = -asinh(aa / NS_bounds.back());

        double chi_c = 2 * atan(aa / rext);
        double eta_c = log((1 + rext / aa) / (rext / aa - 1));
        double eta_lim = eta_c / 2.;
        double chi_lim = chi_lim_eta(eta_lim, rext, aa, chi_c);

        // NS
        Point center_ns(ndim);
        center_ns.set(1) = aa * cosh(eta_minus) / sinh(eta_minus);
        build_adapted_star_domains(*this, domains, NS, ADAPTEDNS, ttype, NS_bounds, n_shells1, center_ns, res);

        // BH
        auto gen_bh_domains = [&](const int nuc_i, auto& bounds, const double& eta) {
            const int RMID = 1;
            const int RIN = 0;
            int shells = bounds.size() - 3;
            Point center(ndim);
            center.set(1) = aa * cosh(eta) / sinh(eta);
            domains[nuc_i] = new Domain_nucleus(nuc_i, ttype, bounds[RIN], center, res);
            domains[nuc_i + 1] =
                new Domain_shell_outer_homothetic(*this, nuc_i + 1, ttype, bounds[RIN], bounds[RMID], center, res);
            domains[nuc_i + 2] =
                new Domain_shell_inner_homothetic(*this, nuc_i + 2, ttype, bounds[RMID], bounds[RMID + 1], center, res);

            for (int i = 0; i < shells; ++i)
                domains[nuc_i + 3 + i] =
                    new Domain_shell(nuc_i + 3 + i, ttype, bounds[RMID + 1 + i], bounds[RMID + 1 + i + 1], center, res);
        };
        gen_bh_domains(BH, BH_bounds, eta_plus);

        // Bispheric part
        domains[OUTER] = new Domain_bispheric_chi_first(OUTER, ttype, aa, eta_minus, rext, chi_lim, res_bi);
        domains[OUTER + 1] =
            new Domain_bispheric_rect(OUTER + 1, ttype, aa, rext, eta_minus, -eta_lim, chi_lim, res_bi);
        domains[OUTER + 2] = new Domain_bispheric_eta_first(OUTER + 2, ttype, aa, rext, -eta_lim, eta_lim, res_bi);
        domains[OUTER + 3] = new Domain_bispheric_rect(OUTER + 3, ttype, aa, rext, eta_plus, eta_lim, chi_lim, res_bi);
        domains[OUTER + 4] = new Domain_bispheric_chi_first(OUTER + 4, ttype, aa, eta_plus, rext, chi_lim, res_bi);

        Point center(3);
        for (int i = 0; i < n_shells_outer; i++)
            domains[OUTER + 5 + i] =
                new Domain_shell(OUTER + 5 + i, ttype, outer_bounds[i], outer_bounds[i + 1], center, res);

        // Compactified
        domains[OUTER + 5 + n_shells_outer] =
            new Domain_compact(OUTER + 5 + n_shells_outer, ttype, outer_bounds[n_shells_outer], center, res);

        const Domain_shell_outer_adapted* pouter_1 =
            dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTEDNS]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_adapted* pinner_1 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTEDNS + 1]);
        pinner_1->vars_to_terms();
        pinner_1->update();
        const Domain_shell_outer_homothetic* pouter_2 =
            dynamic_cast<const Domain_shell_outer_homothetic*>(domains[ADAPTEDBH]);
        pouter_2->vars_to_terms();
        pouter_2->update();
        const Domain_shell_inner_homothetic* pinner_2 =
            dynamic_cast<const Domain_shell_inner_homothetic*>(domains[ADAPTEDBH + 1]);
        pinner_2->vars_to_terms();
        pinner_2->update();
    }

    // Per-domain-resolution variant of the uniform constructor: each domain carries
    // its own Dim_array{nr, ntheta, nphi}. Used by per-domain AMR p-refinement
    // (including the bispheric phi block). Mirrors the uniform ctor exactly,
    // substituting res_per_domain[d] for the shared res/res_bi. Constraints, as for
    // Space_bin_ns: the five bispheric domains (OUTER..OUTER+4) must share one
    // Dim_array (their mutual seams couple every dimension); the NS adapted pair and
    // the BH homothetic pair must each agree in theta/phi (each shares one surface
    // object, so that seam stays conforming).
    Space_bhns::Space_bhns(int ttype, double dist, const std::vector<double>& NS_bounds,
                           const std::vector<double>& BH_bounds, const std::vector<double>& outer_bounds,
                           const std::vector<Dim_array>& res_per_domain)
    {

        ndim = 3;
        double rext = outer_bounds[0];

        n_shells1 = NS_bounds.size() - 3;
        n_shells2 = BH_bounds.size() - 3;

        NS = 0;
        BH = 3 + n_shells1;
        ADAPTEDNS = NS + 1;
        ADAPTEDBH = BH + 1;
        OUTER = 6 + n_shells1 + n_shells2;

        n_shells_outer = outer_bounds.size() - 1;
        nbr_domains = 12 + n_shells1 + n_shells2 + n_shells_outer;
        type_base = ttype;
        domains = new Domain*[nbr_domains]();

        if (int(res_per_domain.size()) != nbr_domains)
            KADATH_THROW("Space_bhns: res_per_domain must have one entry per domain");
        for (int d = 0; d < nbr_domains; ++d) {
            if (res_per_domain[d].get_ndim() != ndim)
                KADATH_THROW("Space_bhns: each per-domain Dim_array must be 3-dimensional");
            if (res_per_domain[d](0) < 5)
                KADATH_THROW("Space_bhns: per-domain radial resolution must be at least 5");
            if (res_per_domain[d](1) < 5)
                KADATH_THROW("Space_bhns: per-domain theta resolution must be at least 5");
        }
        // The five bispheric domains share an identical Dim_array.
        const Dim_array& base = res_per_domain[OUTER];
        for (int d = OUTER; d < OUTER + 5; ++d)
            if (res_per_domain[d](0) != base(0) || res_per_domain[d](1) != base(1) ||
                res_per_domain[d](2) != base(2))
                KADATH_THROW("Space_bhns: the five bispheric domains must share an identical resolution");
        auto require_pair_angular_match = [&](int adapted, const char* what) {
            if (res_per_domain[adapted](1) != res_per_domain[adapted + 1](1) ||
                res_per_domain[adapted](2) != res_per_domain[adapted + 1](2))
                KADATH_THROW(std::string("Space_bhns: the ") + what + " pair must share theta and phi");
        };
        require_pair_angular_match(ADAPTEDNS, "NS adapted");
        require_pair_angular_match(ADAPTEDBH, "BH homothetic");

        // Bispheric : computation of aa (identical to the uniform ctor).
        Param par_a;
        par_a.add_double(NS_bounds.back(), 0);
        par_a.add_double(BH_bounds.back(), 1);
        par_a.add_double(dist, 2);
        double a_min = 0;
        double a_max = dist / 2.;
        double precis = PRECISION;
        int nitermax = 500;
        int niter;
        double aa = zerosec(func_abhns, par_a, a_min, a_max, precis, nitermax, niter);
        double eta_plus = asinh(aa / BH_bounds.back());
        double eta_minus = -asinh(aa / NS_bounds.back());

        double chi_c = 2 * atan(aa / rext);
        double eta_c = log((1 + rext / aa) / (rext / aa - 1));
        double eta_lim = eta_c / 2.;
        double chi_lim = chi_lim_eta(eta_lim, rext, aa, chi_c);

        // NS — per-domain (vector overload of build_adapted_star_domains).
        Point center_ns(ndim);
        center_ns.set(1) = aa * cosh(eta_minus) / sinh(eta_minus);
        build_adapted_star_domains(*this, domains, NS, ADAPTEDNS, ttype, NS_bounds, n_shells1, center_ns,
                                   res_per_domain);

        // BH — per-domain (nucleus + homothetic pair + outer shells).
        {
            const int RMID = 1;
            const int RIN = 0;
            Point center(ndim);
            center.set(1) = aa * cosh(eta_plus) / sinh(eta_plus);
            domains[BH] = new Domain_nucleus(BH, ttype, BH_bounds[RIN], center, res_per_domain[BH]);
            domains[BH + 1] = new Domain_shell_outer_homothetic(*this, BH + 1, ttype, BH_bounds[RIN], BH_bounds[RMID],
                                                                center, res_per_domain[BH + 1]);
            domains[BH + 2] = new Domain_shell_inner_homothetic(*this, BH + 2, ttype, BH_bounds[RMID],
                                                                BH_bounds[RMID + 1], center, res_per_domain[BH + 2]);
            for (int i = 0; i < n_shells2; ++i)
                domains[BH + 3 + i] = new Domain_shell(BH + 3 + i, ttype, BH_bounds[RMID + 1 + i],
                                                       BH_bounds[RMID + 1 + i + 1], center, res_per_domain[BH + 3 + i]);
        }

        // Bispheric part — per-domain (the five share one Dim_array, enforced above).
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

        const Domain_shell_outer_adapted* pouter_1 =
            dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTEDNS]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_adapted* pinner_1 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTEDNS + 1]);
        pinner_1->vars_to_terms();
        pinner_1->update();
        const Domain_shell_outer_homothetic* pouter_bh =
            dynamic_cast<const Domain_shell_outer_homothetic*>(domains[ADAPTEDBH]);
        pouter_bh->vars_to_terms();
        pouter_bh->update();
        const Domain_shell_inner_homothetic* pinner_bh =
            dynamic_cast<const Domain_shell_inner_homothetic*>(domains[ADAPTEDBH + 1]);
        pinner_bh->vars_to_terms();
        pinner_bh->update();
    }

    Space_bhns::Space_bhns(BinarySource& source, bool oldspace)
    {
        nbr_domains = source.read<int>();
        assert(nbr_domains >= 12);

        // Retired inner-shell slot, still read (and discarded) for .dat back-compat.
        int /*retired n_inner_shells*/ ignore = source.read<int>();
        (void)ignore;

        // in the original BHNS, n_shells referred to interior shells
        // whereas it now refers to exterior shells like for the BH
        if (!oldspace)
            n_shells1 = source.read<int>();

        n_shells2 = source.read<int>();

        ndim = source.read<int>();
        type_base = source.read<int>();

        n_shells_outer = nbr_domains - 12 - n_shells1 - n_shells2;
        assert(n_shells_outer >= 0);

        NS = 0;
        BH = 3 + n_shells1;
        ADAPTEDNS = NS + 1;
        ADAPTEDBH = BH + 1;
        OUTER = 6 + n_shells1 + n_shells2;

        domains = new Domain*[nbr_domains];

        // NS (shared adapted-CO read path):
        read_adapted_star_domains<Domain_nucleus, Domain_shell_outer_adapted, Domain_shell_inner_adapted,
                                  Domain_shell>(*this, domains, NS, ADAPTEDNS, n_shells1, source);

        // BH :
        domains[BH] = new Domain_nucleus(BH, source);
        domains[ADAPTEDBH] = new Domain_shell_outer_homothetic(*this, ADAPTEDBH, source);
        domains[ADAPTEDBH + 1] = new Domain_shell_inner_homothetic(*this, ADAPTEDBH + 1, source);

        for (int i = 0; i < n_shells2; ++i)
            domains[ADAPTEDBH + 2 + i] = new Domain_shell(ADAPTEDBH + 2 + i, source);

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

        const Domain_shell_outer_adapted* pouter_1 =
            dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTEDNS]);
        pouter_1->vars_to_terms();
        pouter_1->update();
        const Domain_shell_inner_adapted* pinner_1 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTEDNS + 1]);
        pinner_1->vars_to_terms();
        pinner_1->update();
        const Domain_shell_outer_homothetic* pouter_2 =
            dynamic_cast<const Domain_shell_outer_homothetic*>(domains[ADAPTEDBH]);
        pouter_2->vars_to_terms();
        pouter_2->update();
        const Domain_shell_inner_homothetic* pinner_2 =
            dynamic_cast<const Domain_shell_inner_homothetic*>(domains[ADAPTEDBH + 1]);
        pinner_2->vars_to_terms();
        pinner_2->update();
    }

    Space_bhns::~Space_bhns()
    {
        const Domain_shell_outer_adapted* pouter_1 =
            dynamic_cast<const Domain_shell_outer_adapted*>(domains[ADAPTEDNS]);
        const Domain_shell_inner_adapted* pinner_1 =
            dynamic_cast<const Domain_shell_inner_adapted*>(domains[ADAPTEDNS + 1]);
        const Domain_shell_outer_homothetic* pouter_2 =
            dynamic_cast<const Domain_shell_outer_homothetic*>(domains[ADAPTEDBH]);
        const Domain_shell_inner_homothetic* pinner_2 =
            dynamic_cast<const Domain_shell_inner_homothetic*>(domains[ADAPTEDBH + 1]);
        pouter_1->del_deriv();
        pinner_1->del_deriv();
        pouter_2->del_deriv();
        pinner_2->del_deriv();
    }

    void Space_bhns::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        // Retired inner-shell slot, kept 0 for .dat back-compat.
        sink.write<int>(0);
        sink.write<int>(n_shells1);
        sink.write<int>(n_shells2);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }

    int Space_bhns::nbr_unknowns_from_variable_domains() const
    {
        return (domains[ADAPTEDNS]->nbr_unknowns_from_adapted() + domains[ADAPTEDBH]->nbr_unknowns_from_adapted());
    }

    void Space_bhns::affecte_coef_to_variable_domains(int& pos, int cc, Array<int>& zedoms) const
    {

        // In star 1 ?
        bool found_outer_1 = false;
        int old_pos = pos;
        domains[ADAPTEDNS]->affecte_coef(pos, cc, found_outer_1);
        pos = old_pos;
        bool found_inner_1 = false;
        domains[ADAPTEDNS + 1]->affecte_coef(pos, cc, found_inner_1);
        assert(found_outer_1 == found_inner_1);
        if (found_outer_1) {
            zedoms.set(0) = ADAPTEDNS;
            zedoms.set(1) = ADAPTEDNS + 1;
        } else {
            zedoms.set(0) = -1;
            zedoms.set(1) = -1;
        }

        // In star 2 ?
        bool found_outer_2 = false;
        old_pos = pos;
        domains[ADAPTEDBH]->affecte_coef(pos, cc, found_outer_2);
        pos = old_pos;
        bool found_inner_2 = false;
        domains[ADAPTEDBH + 1]->affecte_coef(pos, cc, found_inner_2);
        assert(found_outer_2 == found_inner_2);
        if (found_outer_2) {
            zedoms.set(0) = ADAPTEDBH;
            zedoms.set(1) = ADAPTEDBH + 1;
        }
    }

    void Space_bhns::xx_to_ders_variable_domains(const Array<double>& xx, int& conte) const
    {

        // Star 1
        int old_conte = conte;
        domains[ADAPTEDNS]->xx_to_ders_from_adapted(xx, conte);
        conte = old_conte;
        domains[ADAPTEDNS + 1]->xx_to_ders_from_adapted(xx, conte);
        // Star 2
        old_conte = conte;
        domains[ADAPTEDBH]->xx_to_ders_from_adapted(xx, conte);
        conte = old_conte;
        domains[ADAPTEDBH + 1]->xx_to_ders_from_adapted(xx, conte);
    }

    void Space_bhns::xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& xx, int& pos) const
    {

        // First get the corrections :
        // Star 1
        int old_pos = pos;
        Val_domain cor_outer_1(domains[ADAPTEDNS]);
        domains[ADAPTEDNS]->xx_to_vars_from_adapted(cor_outer_1, xx, pos);
        pos = old_pos;
        Val_domain cor_inner_1(domains[ADAPTEDNS + 1]);
        domains[ADAPTEDNS + 1]->xx_to_vars_from_adapted(cor_inner_1, xx, pos);

        // Star 2
        old_pos = pos;
        Val_domain cor_outer_2(domains[ADAPTEDBH]);
        domains[ADAPTEDBH]->xx_to_vars_from_adapted(cor_outer_2, xx, pos);
        pos = old_pos;
        Val_domain cor_inner_2(domains[ADAPTEDBH + 1]);
        domains[ADAPTEDBH + 1]->xx_to_vars_from_adapted(cor_inner_2, xx, pos);

        // Now update the variables :
        for (int i = 0; i < sys->nvar; i++) {
            for (int n = 0; n < sys->var[i]->get_n_comp(); n++) {
                Scalar res(*this);
                domains[ADAPTEDNS]->update_variable(cor_outer_1, *sys->var[i]->cmp[n], res);
                domains[ADAPTEDNS + 1]->update_variable(cor_inner_1, *sys->var[i]->cmp[n], res);
                domains[ADAPTEDBH]->update_variable(cor_outer_2, *sys->var[i]->cmp[n], res);
                domains[ADAPTEDBH + 1]->update_variable(cor_inner_2, *sys->var[i]->cmp[n], res);

                sys->var[i]->cmp[n]->set_domain(ADAPTEDNS) = res(ADAPTEDNS);
                sys->var[i]->cmp[n]->set_domain(ADAPTEDNS + 1) = res(ADAPTEDNS + 1);
                sys->var[i]->cmp[n]->set_domain(ADAPTEDBH) = res(ADAPTEDBH);
                sys->var[i]->cmp[n]->set_domain(ADAPTEDBH + 1) = res(ADAPTEDBH + 1);
            }
        }

        // Now the constants :
        for (int i = 0; i < sys->ncst; i++)
            if (sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->get_type_data() == TERM_T)
                for (int n = 0; n < sys->cst[i * (sys->dom_max - sys->dom_min + 1)]->val_t->get_n_comp(); n++) {

                    Scalar so(*this);
                    so = 0;
                    so.set_domain(ADAPTEDNS) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTEDNS]->val_t->cmp[n])(ADAPTEDNS);
                    so.set_domain(ADAPTEDNS + 1) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTEDNS + 1]->val_t->cmp[n])(ADAPTEDNS +
                                                                                                          1);
                    so.set_domain(ADAPTEDBH) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTEDBH]->val_t->cmp[n])(ADAPTEDBH);
                    so.set_domain(ADAPTEDBH + 1) =
                        (*sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTEDBH + 1]->val_t->cmp[n])(ADAPTEDBH +
                                                                                                          1);

                    Scalar res(*this);
                    res = 0;
                    domains[ADAPTEDNS]->update_constante(cor_outer_1, so, res);
                    domains[ADAPTEDNS + 1]->update_constante(cor_inner_1, so, res);
                    domains[ADAPTEDBH]->update_constante(cor_outer_2, so, res);
                    domains[ADAPTEDBH + 1]->update_constante(cor_inner_2, so, res);

                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTEDNS]->val_t->cmp[n]->set_domain(ADAPTEDNS) =
                        res(ADAPTEDNS);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTEDNS + 1]->val_t->cmp[n]->set_domain(
                        ADAPTEDNS + 1) = res(ADAPTEDNS + 1);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTEDBH]->val_t->cmp[n]->set_domain(ADAPTEDBH) =
                        res(ADAPTEDBH);
                    sys->cst[i * (sys->dom_max - sys->dom_min + 1) + ADAPTEDBH + 1]->val_t->cmp[n]->set_domain(
                        ADAPTEDBH + 1) = res(ADAPTEDBH + 1);
                }

        // Update the mapping :
        domains[ADAPTEDNS]->update_mapping(cor_outer_1);
        domains[ADAPTEDNS + 1]->update_mapping(cor_inner_1);
        domains[ADAPTEDBH]->update_mapping(cor_outer_2);
        domains[ADAPTEDBH + 1]->update_mapping(cor_inner_2);
    }

    Array<int> Space_bhns::get_indices_matching_non_std(int dom, int bound) const
    {
        // NS fixed-radius seams: nucleus.OUTER <-> adapted-outer.INNER, then
        // post-adapted shell-to-shell seams. Mixed angular p-refinement uses
        // add_eq_matching_import on these seams.
        if (dom == NS && bound == OUTER_BC) {
            Array<int> res(2, 1);
            res.set(0, 0) = ADAPTEDNS;
            res.set(1, 0) = INNER_BC;
            return res;
        }
        if (dom == ADAPTEDNS && bound == INNER_BC) {
            Array<int> res(2, 1);
            res.set(0, 0) = NS;
            res.set(1, 0) = OUTER_BC;
            return res;
        }
        for (int i = 0; i < n_shells1; ++i) {
            if (dom == ADAPTEDNS + 1 + i && bound == OUTER_BC) {
                Array<int> res(2, 1);
                res.set(0, 0) = ADAPTEDNS + 2 + i;
                res.set(1, 0) = INNER_BC;
                return res;
            }
            if (dom == ADAPTEDNS + 2 + i && bound == INNER_BC) {
                Array<int> res(2, 1);
                res.set(0, 0) = ADAPTEDNS + 1 + i;
                res.set(1, 0) = OUTER_BC;
                return res;
            }
        }

        // BH post-homothetic shell-to-shell fixed-radius seams. The deformable
        // homothetic pair itself stays conforming and is not handled here.
        for (int i = 0; i < n_shells2; ++i) {
            if (dom == ADAPTEDBH + 1 + i && bound == OUTER_BC) {
                Array<int> res(2, 1);
                res.set(0, 0) = ADAPTEDBH + 2 + i;
                res.set(1, 0) = INNER_BC;
                return res;
            }
            if (dom == ADAPTEDBH + 2 + i && bound == INNER_BC) {
                Array<int> res(2, 1);
                res.set(0, 0) = ADAPTEDBH + 1 + i;
                res.set(1, 0) = OUTER_BC;
                return res;
            }
        }

        if (dom == ADAPTEDNS + 1 + n_shells1) {
            // NS ;
            Array<int> res(2, 2);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = OUTER; // Matching with chi first
                    res.set(1, 0) = INNER_BC;
                    res.set(0, 1) = OUTER + 1; // Matching with rect
                    res.set(1, 1) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bhns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == ADAPTEDBH + 1 + n_shells2) {
            // BH ;
            Array<int> res(2, 2);
            switch (bound) {
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 3; // Matching with rect
                    res.set(1, 0) = INNER_BC;
                    res.set(0, 1) = OUTER + 4; // Matching with chi_first
                    res.set(1, 1) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bhns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == OUTER) {
            // first chi first :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = ADAPTEDNS + 1 + n_shells1; // First star
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 5; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bhns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == OUTER + 1) {
            // first rect :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = ADAPTEDNS + 1 + n_shells1; // First star
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 5; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bhns::get_indices_matching_non_std");
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
                    KADATH_THROW("Bad bound in Space_bhns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == OUTER + 3) {
            // second rect :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = ADAPTEDBH + 1 + n_shells2; // BH
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 5; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bhns::get_indices_matching_non_std");
            }
            return res;
        }

        if (dom == OUTER + 4) {
            // second chi first :
            Array<int> res(2, 1);
            switch (bound) {
                case INNER_BC:
                    res.set(0, 0) = ADAPTEDBH + 1 + n_shells2; // BH
                    res.set(1, 0) = OUTER_BC;
                    break;
                case OUTER_BC:
                    res.set(0, 0) = OUTER + 5; // Compactified domain or first shell
                    res.set(1, 0) = INNER_BC;
                    break;
                default:
                    KADATH_THROW("Bad bound in Space_bhns::get_indices_matching_non_std");
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
                    KADATH_THROW("Bad bound in Space_bhns::get_indices_matching_non_std");
            }
            return res;
        }

        KADATH_THROW("Bad domain in Space_bhns::get_indices_matching_non_std");
    }

} // namespace Kadath
