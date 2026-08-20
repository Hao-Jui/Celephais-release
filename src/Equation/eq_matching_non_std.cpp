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

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Eq_matching_non_std::Eq_matching_non_std(const Domain* zedom, int dd, int bb, const Array<int>& ozers, int nused,
                                             Array<int>** pused)
        : Equation(zedom, dd, ozers.get_size(1) + 1, nused, pused), bound(bb), other_doms(ozers.get_size(1)),
          other_bounds(ozers.get_size(1)), which_points(nullptr)
    {

        for (int i = 0; i < n_ope - 1; i++) {
            other_doms.set(i) = ozers(0, i);
            other_bounds.set(i) = ozers(1, i);
        }
    }

    Eq_matching_non_std::~Eq_matching_non_std()
    {
        if (called) {
            for (int i = 0; i < n_cond_tot; i++)
                delete which_points[i];
            delete[] which_points;
        }
    }

    Array<int> Eq_matching_non_std::do_nbr_conditions(const Tensor& tt) const
    {

        int size = (n_cmp_used == -1) ? tt.get_n_comp() : n_cmp_used;
        Array<int> res(size);
        if (n_cmp_used == -1) {
            for (int i = 0; i < tt.get_n_comp(); i++)
                res.set(i) = dom->nbr_points_boundary(bound, (*tt.cmp[i])(ndom).get_base());
        } else {
            for (int i = 0; i < n_cmp_used; i++)
                res.set(i) = dom->nbr_points_boundary(bound, tt(*p_cmp_used[i])(ndom).get_base());
        }
        return res;
    }

    void Eq_matching_non_std::do_which_points(const Base_spectral& bb, int start)
    {
        dom->do_which_points_boundary(bound, bb, which_points, start);
    }

    void Eq_matching_non_std::export_val(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {
        for (int i = 0; i < n_ope; i++)
            assert(residus[conte + i]->get_type_data() == TERM_T);

        Tensor copie(*residus[conte]->get_p_val_t());

        // Loop on the components :
        int start = pos_res;
        // Case for all the cmps :
        if (n_cmp_used == -1) {
            for (int comp = 0; comp < n_comp; comp++) {
                Array<int> ind(copie.indices(comp));
                copie.set(ind).set_domain(ndom).coef_i(); // Configuration space

                for (int j = 0; j < (*n_cond)(comp); j++) {
                    // Value at the colocation point
                    sec.set(pos_res) =
                        (copie(ind)(ndom).check_if_zero()) ? 0. : (*copie(ind)(ndom).c)(*which_points[pos_res - start]);

                    // Get the absolute coordinates :
                    Point absol(dom->get_ndim());
                    for (int i = 1; i <= dom->get_ndim(); i++)
                        absol.set(i) = (*dom->get_absol(i).c)(*which_points[pos_res - start]);

                    bool loop = true;
                    // Loop on the other domains :
                    int num_other = 0;
                    bool indic = false;
                    while (loop) {
                        const Tensor& other_value = *residus[conte + num_other + 1]->get_p_val_t();
                        const Domain* zedom = other_value.espace.get_domain(other_doms(num_other));
                        int ozerbound = other_bounds(num_other);
                        bool found = zedom->is_in(absol, 1e-5);
                        if (found) {
                            Point num(zedom->absol_to_num_bound(absol, ozerbound));
                            Val_domain other_copie(other_value(ind)(other_doms(num_other)));
                            if (!other_copie.check_if_zero()) {
                                other_copie.coef();
                                sec.set(pos_res) -= other_copie.base.summation(num, *other_copie.cf);
                            }
                            loop = false;
                            indic = true;
                        }
                        num_other++;
                        if (num_other >= n_ope - 1)
                            loop = false;
                    }

                    if (!indic) {
                        cerr << absol << endl;
                        KADATH_THROW("not found in Eq_matching_non_std::export_val");
                    }

                    pos_res++;
                }
            }
        } else {
            for (int comp = 0; comp < n_cmp_used; comp++) {
                copie.set(*p_cmp_used[comp]).set_domain(ndom).coef_i(); // Configuration space

                for (int j = 0; j < (*n_cond)(comp); j++) {
                    // Value at the colocation point
                    sec.set(pos_res) = (copie(*p_cmp_used[comp])(ndom).check_if_zero())
                                           ? 0.
                                           : (*copie(*p_cmp_used[comp])(ndom).c)(*which_points[pos_res - start]);

                    // Get the absolute coordinates :
                    Point absol(dom->get_ndim());
                    for (int i = 1; i <= dom->get_ndim(); i++)
                        absol.set(i) = (*dom->get_absol(i).c)(*which_points[pos_res - start]);

                    bool loop = true;
                    // Loop on the other domains :
                    int num_other = 0;
                    bool indic = false;
                    while (loop) {
                        const Tensor& other_value = *residus[conte + num_other + 1]->get_p_val_t();
                        const Domain* zedom = other_value.espace.get_domain(other_doms(num_other));
                        int ozerbound = other_bounds(num_other);
                        bool found = zedom->is_in(absol, 1e-5);
                        if (found) {
                            Point num(zedom->absol_to_num_bound(absol, ozerbound));
                            Val_domain other_copie(other_value(*p_cmp_used[comp])(other_doms(num_other)));
                            if (!other_copie.check_if_zero()) {
                                other_copie.coef();
                                sec.set(pos_res) -= other_copie.base.summation(num, *other_copie.cf);
                            }
                            loop = false;
                            indic = true;
                        }
                        num_other++;
                        if (num_other >= n_ope - 1)
                            loop = false;
                    }
                    if (!indic) {
                        cerr << absol << endl;
                        KADATH_THROW("not found in Eq_matching_non_std::export_val");
                    }

                    pos_res++;
                }
            }
        }
        conte += n_ope;
    }

    void Eq_matching_non_std::export_der(int& conte, Term_eq** residus, Array<double>& sec, int& pos_res) const
    {

        for (int i = 0; i < n_ope; i++)
            assert(residus[conte + i]->get_type_data() == TERM_T);

        Tensor copie(*residus[conte]->get_p_der_t());

        // Loop on the components :
        int start = pos_res;
        // Case for all the cmps :
        if (n_cmp_used == -1) {
            for (int comp = 0; comp < n_comp; comp++) {
                Array<int> ind(copie.indices(comp));
                copie.set(ind).set_domain(ndom).coef_i(); // Configuration space

                for (int j = 0; j < (*n_cond)(comp); j++) {
                    // Value at the colocation point
                    sec.set(pos_res) =
                        (copie(ind)(ndom).check_if_zero()) ? 0. : (*copie(ind)(ndom).c)(*which_points[pos_res - start]);

                    // Get the absolute coordinates :
                    Point absol(dom->get_ndim());
                    for (int i = 1; i <= dom->get_ndim(); i++)
                        absol.set(i) = (*dom->get_absol(i).c)(*which_points[pos_res - start]);

                    bool loop = true;
                    // Loop on the other domains :
                    int num_other = 0;
                    bool indic = false;
                    while (loop) {
                        const Tensor& other_derivative = *residus[conte + num_other + 1]->get_p_der_t();
                        const Domain* zedom = other_derivative.espace.get_domain(other_doms(num_other));
                        int ozerbound = other_bounds(num_other);
                        bool found = zedom->is_in(absol, 1e-3);
                        if (found) {
                            Point num(zedom->absol_to_num_bound(absol, ozerbound));
                            Val_domain other_copie(other_derivative(ind)(other_doms(num_other)));
                            if (!other_copie.check_if_zero()) {
                                other_copie.coef();
                                sec.set(pos_res) -= other_copie.base.summation(num, *other_copie.cf);
                            }
                            loop = false;
                            indic = true;
                        }
                        num_other++;
                        if (num_other >= n_ope - 1)
                            loop = false;
                    }
                    if (!indic) {
                        cerr << absol << endl;
                        KADATH_THROW("not found in Eq_matching_non_std::export_der");
                    }

                    pos_res++;
                }
            }
        } else {
            for (int comp = 0; comp < n_cmp_used; comp++) {
                copie.set(*p_cmp_used[comp]).set_domain(ndom).coef_i(); // Configuration space

                for (int j = 0; j < (*n_cond)(comp); j++) {
                    // Value at the colocation point
                    sec.set(pos_res) = (copie(*p_cmp_used[comp])(ndom).check_if_zero())
                                           ? 0.
                                           : (*copie(*p_cmp_used[comp])(ndom).c)(*which_points[pos_res - start]);

                    // Get the absolute coordinates :
                    Point absol(dom->get_ndim());
                    for (int i = 1; i <= dom->get_ndim(); i++)
                        absol.set(i) = (*dom->get_absol(i).c)(*which_points[pos_res - start]);

                    bool loop = true;
                    // Loop on the other domains :
                    int num_other = 0;
                    bool indic = false;
                    while (loop) {
                        const Tensor& other_derivative = *residus[conte + num_other + 1]->get_p_der_t();
                        const Domain* zedom = other_derivative.espace.get_domain(other_doms(num_other));
                        int ozerbound = other_bounds(num_other);
                        bool found = zedom->is_in(absol, 1e-3);
                        if (found) {
                            Point num(zedom->absol_to_num_bound(absol, ozerbound));
                            Val_domain other_copie(other_derivative(*p_cmp_used[comp])(other_doms(num_other)));
                            if (!other_copie.check_if_zero()) {
                                other_copie.coef();
                                sec.set(pos_res) -= other_copie.base.summation(num, *other_copie.cf);
                            }
                            loop = false;
                            indic = true;
                        }
                        num_other++;
                        if (num_other >= n_ope - 1)
                            loop = false;
                    }
                    if (!indic) {
                        cerr << absol << endl;
                        KADATH_THROW("not found in Eq_matching_non_std::export_der");
                    }

                    pos_res++;
                }
            }
        }
        conte += n_ope;
    }

    bool Eq_matching_non_std::take_into_account(int target) const
    {
        bool res = (target == ndom) ? true : false;
        for (int i = 0; i < n_ope - 1; i++)
            if (target == other_doms(i))
                res = true;
        return res;
    }

    void Eq_matching_non_std::apply(int& conte, Term_eq** res)
    {

        int old_conte = conte;
        for (int i = 0; i < n_ope; i++) {
            if (res[conte] != nullptr)
                *res[conte] = parts[i]->action();
            else
                res[conte] = new Term_eq(parts[i]->action());

            assert(res[conte]->get_type_data() == TERM_T);
            conte++;
        }

        if (!called) {
            Tensor copie(*res[old_conte]->get_p_val_t());
            n_cond = new Array<int>(do_nbr_conditions(copie));
            n_comp = n_cond->get_size(0);
            n_cond_tot = 0;
            if (n_cmp_used == -1)
                for (int i = 0; i < n_cond->get_size(0); i++)
                    n_cond_tot += (*n_cond)(i);
            else
                for (int i = 0; i < n_cmp_used; i++)
                    n_cond_tot += (*n_cond)(i);

            which_points = new Index*[n_cond_tot];
            int start = 0;
            if (n_cmp_used == -1)
                for (int comp = 0; comp < n_comp; comp++) {
                    Array<int> ind(copie.indices(comp));
                    Base_spectral bb(copie(ind)(ndom).get_base());
                    do_which_points(bb, start);
                    start += (*n_cond)(comp);
                }
            else
                for (int comp = 0; comp < n_cmp_used; comp++) {
                    Base_spectral bb(copie(*p_cmp_used[comp])(ndom).get_base());
                    do_which_points(bb, start);
                    start += (*n_cond)(comp);
                }
            called = true;
        }
    }
} // namespace Kadath
