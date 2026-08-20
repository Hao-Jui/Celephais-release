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
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/binary_co_domains.hpp"
#include "For_Kadath/Space/three_body.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include <array>
#include <memory>

namespace Kadath
{
    double chi_lim_eta(double eta, double rext, double a, double chi_c);
    double zerosec(double (*f)(double, const Param&), const Param& parf, double x1, double x2, double precis,
                   int nitermax, int& niter);

    namespace
    {
        constexpr int bispheric_domain_count = 5;

        double three_body_bispheric_root(double aa, const Param& par)
        {
            const double r_minus = par.get_double(0);
            const double r_plus = par.get_double(1);
            const double distance = par.get_double(2);
            return sqrt(aa * aa + r_minus * r_minus) + sqrt(aa * aa + r_plus * r_plus) - distance;
        }

        struct BisphericMap
        {
            double aa{0.};
            double eta_minus{0.};
            double eta_plus{0.};
            double eta_lim{0.};
            double chi_lim{0.};
            double minus_center{0.};
            double plus_center{0.};
        };

        Point x_center(double x)
        {
            Point center(3);
            center.set(1) = x;
            center.set(2) = 0.;
            center.set(3) = 0.;
            return center;
        }

        int outer_shell_count(const std::vector<double>& bounds, const char* label)
        {
            if (bounds.size() < 3)
                KADATH_THROW(std::string(label) + " bounds must be [rin, rmid, rout] or include outer shell bounds");
            return int(bounds.size()) - 3;
        }

        BisphericMap make_bispheric_map(double r_minus, double r_plus, double distance, double outer_radius)
        {
            if (r_minus <= 0. || r_plus <= 0. || outer_radius <= 0.)
                KADATH_THROW("Bispheric radii must be positive");
            if (distance <= r_minus + r_plus)
                KADATH_THROW("Bispheric centers must be separated by more than the two exposed radii");

            Param par_a;
            par_a.add_double(r_minus, 0);
            par_a.add_double(r_plus, 1);
            par_a.add_double(distance, 2);

            int niter = 0;
            const double aa = zerosec(three_body_bispheric_root, par_a, 0., distance / 2., PRECISION, 500, niter);
            if (outer_radius <= aa)
                KADATH_THROW("Bispheric outer radius must be larger than the coordinate scale");

            BisphericMap map;
            map.aa = aa;
            map.eta_minus = -asinh(aa / r_minus);
            map.eta_plus = asinh(aa / r_plus);
            map.minus_center = aa * cosh(map.eta_minus) / sinh(map.eta_minus);
            map.plus_center = aa * cosh(map.eta_plus) / sinh(map.eta_plus);

            const double chi_c = 2. * atan(aa / outer_radius);
            const double eta_c = log((1. + outer_radius / aa) / (outer_radius / aa - 1.));
            map.eta_lim = eta_c / 2.;
            map.chi_lim = chi_lim_eta(map.eta_lim, outer_radius, aa, chi_c);

            if (fabs(map.minus_center) + r_minus >= outer_radius ||
                fabs(map.plus_center) + r_plus >= outer_radius)
                KADATH_THROW("Bispheric outer sphere must enclose both exposed inner spheres");

            return map;
        }

        void build_adapted_star_domains_nosym(const Space& space, Domain** domains, int nucleus_index,
                                              int adapted_index, int ttype, const std::vector<double>& bounds,
                                              int n_outer_shells, const Point& center, const Dim_array& res)
        {
            domains[nucleus_index] = new Domain_nucleus_nosym(nucleus_index, ttype, bounds[0], center, res);
            domains[adapted_index] =
                new Domain_shell_outer_adapted_nosym(space, adapted_index, ttype, bounds[0], bounds[1], center, res);
            domains[adapted_index + 1] = new Domain_shell_inner_adapted_nosym(
                space, adapted_index + 1, ttype, bounds[1], bounds[2], center, res);
            for (int i = 0; i < n_outer_shells; ++i)
                domains[adapted_index + 2 + i] = new Domain_shell_nosym(
                    adapted_index + 2 + i, ttype, bounds[2 + i], bounds[3 + i], center, res);
        }

    } // namespace

    void Space_three_body::set_domain_indices()
    {
        BODY = 0;
        ADAPTED_BODY = BODY + 1;
        CHILD1 = ADAPTED_BODY + 2 + n_shells_body;
        ADAPTED_CHILD1 = CHILD1 + 1;
        CHILD2 = ADAPTED_CHILD1 + 2 + n_shells_child1;
        ADAPTED_CHILD2 = CHILD2 + 1;
        CHILD_BI = ADAPTED_CHILD2 + 2 + n_shells_child2;
        PARENT_BI = CHILD_BI + bispheric_domain_count;
        OUTER = PARENT_BI + bispheric_domain_count;
    }

    Space_three_body::Space_three_body(int ttype, double parent_dist, double child_dist,
                                       const std::vector<double>& body_bounds,
                                       const std::vector<double>& child1_bounds,
                                       const std::vector<double>& child2_bounds, double child_outer_radius,
                                       const std::vector<double>& outer_bounds, int nr)
    {
        if (outer_bounds.empty())
            KADATH_THROW("ThreeBody outer_bounds must include the parent bispheric outer radius");

        ndim = 3;
        type_base = ttype;
        n_shells_body = outer_shell_count(body_bounds, "body");
        n_shells_child1 = outer_shell_count(child1_bounds, "child1");
        n_shells_child2 = outer_shell_count(child2_bounds, "child2");
        n_shells_outer = int(outer_bounds.size()) - 1;
        set_domain_indices();
        nbr_domains = OUTER + n_shells_outer + 1;
        domains = new Domain*[nbr_domains];

        Dim_array res(ndim);
        res.set(0) = nr;
        res.set(1) = nr;
        res.set(2) = nr - 1;
        Dim_array res_bi(ndim);
        res_bi.set(0) = nr;
        res_bi.set(1) = nr;
        res_bi.set(2) = nr - 1;

        const BisphericMap parent_map =
            make_bispheric_map(body_bounds.back(), child_outer_radius, parent_dist, outer_bounds[0]);
        const BisphericMap child_map =
            make_bispheric_map(child1_bounds.back(), child2_bounds.back(), child_dist, child_outer_radius);
        child_origin_x = parent_map.plus_center;

        build_adapted_star_domains_nosym(*this, domains, BODY, ADAPTED_BODY, ttype, body_bounds, n_shells_body,
                                         x_center(parent_map.minus_center), res);
        build_adapted_star_domains_nosym(*this, domains, CHILD1, ADAPTED_CHILD1, ttype, child1_bounds,
                                         n_shells_child1, x_center(child_origin_x + child_map.minus_center), res);
        build_adapted_star_domains_nosym(*this, domains, CHILD2, ADAPTED_CHILD2, ttype, child2_bounds,
                                         n_shells_child2, x_center(child_origin_x + child_map.plus_center), res);

        domains[CHILD_BI] = new Domain_bispheric_chi_first_nosym(CHILD_BI, ttype, child_map.aa, child_map.eta_minus,
                                                                  child_outer_radius, child_map.chi_lim, res_bi);
        domains[CHILD_BI + 1] = new Domain_bispheric_rect_nosym(
            CHILD_BI + 1, ttype, child_map.aa, child_outer_radius, child_map.eta_minus, -child_map.eta_lim,
            child_map.chi_lim, res_bi);
        domains[CHILD_BI + 2] = new Domain_bispheric_eta_first_nosym(
            CHILD_BI + 2, ttype, child_map.aa, child_outer_radius, -child_map.eta_lim, child_map.eta_lim, res_bi);
        domains[CHILD_BI + 3] = new Domain_bispheric_rect_nosym(
            CHILD_BI + 3, ttype, child_map.aa, child_outer_radius, child_map.eta_plus, child_map.eta_lim,
            child_map.chi_lim, res_bi);
        domains[CHILD_BI + 4] = new Domain_bispheric_chi_first_nosym(CHILD_BI + 4, ttype, child_map.aa,
                                                                      child_map.eta_plus, child_outer_radius,
                                                                      child_map.chi_lim, res_bi);
        set_child_bispheric_origin();

        domains[PARENT_BI] = new Domain_bispheric_chi_first_nosym(PARENT_BI, ttype, parent_map.aa,
                                                                   parent_map.eta_minus, outer_bounds[0],
                                                                   parent_map.chi_lim, res_bi);
        domains[PARENT_BI + 1] = new Domain_bispheric_rect_nosym(
            PARENT_BI + 1, ttype, parent_map.aa, outer_bounds[0], parent_map.eta_minus, -parent_map.eta_lim,
            parent_map.chi_lim, res_bi);
        domains[PARENT_BI + 2] = new Domain_bispheric_eta_first_nosym(
            PARENT_BI + 2, ttype, parent_map.aa, outer_bounds[0], -parent_map.eta_lim, parent_map.eta_lim, res_bi);
        domains[PARENT_BI + 3] = new Domain_bispheric_rect_nosym(
            PARENT_BI + 3, ttype, parent_map.aa, outer_bounds[0], parent_map.eta_plus, parent_map.eta_lim,
            parent_map.chi_lim, res_bi);
        domains[PARENT_BI + 4] = new Domain_bispheric_chi_first_nosym(PARENT_BI + 4, ttype, parent_map.aa,
                                                                       parent_map.eta_plus, outer_bounds[0],
                                                                       parent_map.chi_lim, res_bi);

        Point origin(3);
        for (int i = 0; i < n_shells_outer; ++i)
            domains[OUTER + i] =
                new Domain_shell_nosym(OUTER + i, ttype, outer_bounds[i], outer_bounds[i + 1], origin, res);
        domains[OUTER + n_shells_outer] =
            new Domain_compact_nosym(OUTER + n_shells_outer, ttype, outer_bounds[n_shells_outer], origin, res);

        update_adapted_mappings();
    }

    Space_three_body::Space_three_body(BinarySource& source)
    {
        nbr_domains = source.read<int>();
        n_shells_body = source.read<int>();
        n_shells_child1 = source.read<int>();
        n_shells_child2 = source.read<int>();
        n_shells_outer = source.read<int>();
        child_origin_x = source.read<double>();
        ndim = source.read<int>();
        type_base = source.read<int>();
        set_domain_indices();
        if (nbr_domains != OUTER + n_shells_outer + 1)
            KADATH_THROW("Inconsistent Space_three_body domain count in binary source");

        domains = new Domain*[nbr_domains];
        read_adapted_star_domains<Domain_nucleus_nosym, Domain_shell_outer_adapted_nosym,
                                  Domain_shell_inner_adapted_nosym, Domain_shell_nosym>(
            *this, domains, BODY, ADAPTED_BODY, n_shells_body, source);
        read_adapted_star_domains<Domain_nucleus_nosym, Domain_shell_outer_adapted_nosym,
                                  Domain_shell_inner_adapted_nosym, Domain_shell_nosym>(
            *this, domains, CHILD1, ADAPTED_CHILD1, n_shells_child1, source);
        read_adapted_star_domains<Domain_nucleus_nosym, Domain_shell_outer_adapted_nosym,
                                  Domain_shell_inner_adapted_nosym, Domain_shell_nosym>(
            *this, domains, CHILD2, ADAPTED_CHILD2, n_shells_child2, source);

        domains[CHILD_BI] = new Domain_bispheric_chi_first_nosym(CHILD_BI, source);
        domains[CHILD_BI + 1] = new Domain_bispheric_rect_nosym(CHILD_BI + 1, source);
        domains[CHILD_BI + 2] = new Domain_bispheric_eta_first_nosym(CHILD_BI + 2, source);
        domains[CHILD_BI + 3] = new Domain_bispheric_rect_nosym(CHILD_BI + 3, source);
        domains[CHILD_BI + 4] = new Domain_bispheric_chi_first_nosym(CHILD_BI + 4, source);
        set_child_bispheric_origin();

        domains[PARENT_BI] = new Domain_bispheric_chi_first_nosym(PARENT_BI, source);
        domains[PARENT_BI + 1] = new Domain_bispheric_rect_nosym(PARENT_BI + 1, source);
        domains[PARENT_BI + 2] = new Domain_bispheric_eta_first_nosym(PARENT_BI + 2, source);
        domains[PARENT_BI + 3] = new Domain_bispheric_rect_nosym(PARENT_BI + 3, source);
        domains[PARENT_BI + 4] = new Domain_bispheric_chi_first_nosym(PARENT_BI + 4, source);

        for (int i = 0; i < n_shells_outer; ++i)
            domains[OUTER + i] = new Domain_shell_nosym(OUTER + i, source);
        domains[OUTER + n_shells_outer] = new Domain_compact_nosym(OUTER + n_shells_outer, source);

        update_adapted_mappings();
    }

    Space_three_body::~Space_three_body()
    {
        delete_adapted_derivatives();
    }

    void Space_three_body::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        sink.write<int>(n_shells_body);
        sink.write<int>(n_shells_child1);
        sink.write<int>(n_shells_child2);
        sink.write<int>(n_shells_outer);
        sink.write<double>(child_origin_x);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; ++i)
            domains[i]->save(sink);
    }

    void Space_three_body::set_child_bispheric_origin() const
    {
        dynamic_cast<Domain_bispheric_chi_first_nosym*>(domains[CHILD_BI])->set_origin_x(child_origin_x);
        dynamic_cast<Domain_bispheric_rect_nosym*>(domains[CHILD_BI + 1])->set_origin_x(child_origin_x);
        dynamic_cast<Domain_bispheric_eta_first_nosym*>(domains[CHILD_BI + 2])->set_origin_x(child_origin_x);
        dynamic_cast<Domain_bispheric_rect_nosym*>(domains[CHILD_BI + 3])->set_origin_x(child_origin_x);
        dynamic_cast<Domain_bispheric_chi_first_nosym*>(domains[CHILD_BI + 4])->set_origin_x(child_origin_x);
    }

    void Space_three_body::update_adapted_mappings() const
    {
        const std::array<int, 3> adapted_domains = {ADAPTED_BODY, ADAPTED_CHILD1, ADAPTED_CHILD2};
        for (const int adapted : adapted_domains) {
            const Domain_shell_outer_adapted_nosym* outer =
                dynamic_cast<const Domain_shell_outer_adapted_nosym*>(domains[adapted]);
            const Domain_shell_inner_adapted_nosym* inner =
                dynamic_cast<const Domain_shell_inner_adapted_nosym*>(domains[adapted + 1]);
            outer->vars_to_terms();
            outer->update();
            inner->vars_to_terms();
            inner->update();
        }
    }

    void Space_three_body::delete_adapted_derivatives() const
    {
        const std::array<int, 3> adapted_domains = {ADAPTED_BODY, ADAPTED_CHILD1, ADAPTED_CHILD2};
        for (const int adapted : adapted_domains) {
            const Domain_shell_outer_adapted_nosym* outer =
                dynamic_cast<const Domain_shell_outer_adapted_nosym*>(domains[adapted]);
            const Domain_shell_inner_adapted_nosym* inner =
                dynamic_cast<const Domain_shell_inner_adapted_nosym*>(domains[adapted + 1]);
            outer->del_deriv();
            inner->del_deriv();
        }
    }

    int Space_three_body::nbr_unknowns_from_variable_domains() const
    {
        return domains[ADAPTED_BODY]->nbr_unknowns_from_adapted() +
               domains[ADAPTED_CHILD1]->nbr_unknowns_from_adapted() +
               domains[ADAPTED_CHILD2]->nbr_unknowns_from_adapted();
    }

    void Space_three_body::affecte_coef_to_variable_domains(int& pos, int cc, Array<int>& zedoms) const
    {
        zedoms.set(0) = -1;
        zedoms.set(1) = -1;
        const std::array<int, 3> adapted_domains = {ADAPTED_BODY, ADAPTED_CHILD1, ADAPTED_CHILD2};
        for (const int adapted : adapted_domains) {
            bool found_outer = false;
            const int old_pos = pos;
            domains[adapted]->affecte_coef(pos, cc, found_outer);
            pos = old_pos;
            bool found_inner = false;
            domains[adapted + 1]->affecte_coef(pos, cc, found_inner);
            assert(found_outer == found_inner);
            if (found_outer) {
                zedoms.set(0) = adapted;
                zedoms.set(1) = adapted + 1;
            }
        }
    }

    void Space_three_body::xx_to_ders_variable_domains(const Array<double>& xx, int& conte) const
    {
        const std::array<int, 3> adapted_domains = {ADAPTED_BODY, ADAPTED_CHILD1, ADAPTED_CHILD2};
        for (const int adapted : adapted_domains) {
            const int old_conte = conte;
            domains[adapted]->xx_to_ders_from_adapted(xx, conte);
            conte = old_conte;
            domains[adapted + 1]->xx_to_ders_from_adapted(xx, conte);
        }
    }

    void Space_three_body::xx_to_vars_variable_domains(System_of_eqs* sys, const Array<double>& xx, int& pos) const
    {
        const std::array<int, 3> adapted_domains = {ADAPTED_BODY, ADAPTED_CHILD1, ADAPTED_CHILD2};
        std::array<std::unique_ptr<Val_domain>, 3> cor_outer;
        std::array<std::unique_ptr<Val_domain>, 3> cor_inner;

        for (int i = 0; i < 3; ++i) {
            const int adapted = adapted_domains[i];
            const int old_pos = pos;
            cor_outer[i] = std::make_unique<Val_domain>(domains[adapted]);
            domains[adapted]->xx_to_vars_from_adapted(*cor_outer[i], xx, pos);
            pos = old_pos;
            cor_inner[i] = std::make_unique<Val_domain>(domains[adapted + 1]);
            domains[adapted + 1]->xx_to_vars_from_adapted(*cor_inner[i], xx, pos);
        }

        for (int i = 0; i < sys->nvar; i++) {
            for (int n = 0; n < sys->var[i]->get_n_comp(); n++) {
                Scalar res(*this);
                for (int pair = 0; pair < 3; ++pair) {
                    const int adapted = adapted_domains[pair];
                    domains[adapted]->update_variable(*cor_outer[pair], *sys->var[i]->cmp[n], res);
                    domains[adapted + 1]->update_variable(*cor_inner[pair], *sys->var[i]->cmp[n], res);
                }
                for (const int adapted : adapted_domains) {
                    sys->var[i]->cmp[n]->set_domain(adapted) = res(adapted);
                    sys->var[i]->cmp[n]->set_domain(adapted + 1) = res(adapted + 1);
                }
            }
        }

        const int dom_count = sys->dom_max - sys->dom_min + 1;
        for (int i = 0; i < sys->ncst; i++) {
            if (sys->cst[i * dom_count]->get_type_data() != TERM_T)
                continue;
            for (int n = 0; n < sys->cst[i * dom_count]->val_t->get_n_comp(); n++) {
                Scalar so(*this);
                so = 0;
                for (const int adapted : adapted_domains) {
                    so.set_domain(adapted) = (*sys->cst[i * dom_count + adapted]->val_t->cmp[n])(adapted);
                    so.set_domain(adapted + 1) =
                        (*sys->cst[i * dom_count + adapted + 1]->val_t->cmp[n])(adapted + 1);
                }

                Scalar res(*this);
                res = 0;
                for (int pair = 0; pair < 3; ++pair) {
                    const int adapted = adapted_domains[pair];
                    domains[adapted]->update_constante(*cor_outer[pair], so, res);
                    domains[adapted + 1]->update_constante(*cor_inner[pair], so, res);
                }
                for (const int adapted : adapted_domains) {
                    sys->cst[i * dom_count + adapted]->val_t->cmp[n]->set_domain(adapted) = res(adapted);
                    sys->cst[i * dom_count + adapted + 1]->val_t->cmp[n]->set_domain(adapted + 1) = res(adapted + 1);
                }
            }
        }

        for (int pair = 0; pair < 3; ++pair) {
            const int adapted = adapted_domains[pair];
            domains[adapted]->update_mapping(*cor_outer[pair]);
            domains[adapted + 1]->update_mapping(*cor_inner[pair]);
        }
    }
} // namespace Kadath
