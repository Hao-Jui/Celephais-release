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

#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    void Tensor::operator=(const Tensor& t)
    {

        assert(valence == t.valence);
        assert(&espace == &t.espace);

        basis = t.basis;

        // Case when one does not care about the indices :
        if ((!t.name_affected) || (!name_affected)) {

            for (int i = 0; i < valence; i++)
                assert(t.type_indice(i) == type_indice(i));

            for (int i = 0; i < n_comp; i++) {
                int place_t = t.position(indices(i));
                *cmp[i] = *t.cmp[place_t];
            }
        } else {
            // Find the permutation :
            Array<int> perm(valence);
            bool same_ind = find_indices(t, perm);
            if (!same_ind) {
                KADATH_THROW("Indices do not match in Tensor::operator=");
            } else
                for (int i = 0; i < n_comp; i++) {
                    Array<int> ind(indices(i));
                    Array<int> ind_t(valence);
                    for (int j = 0; j < valence; j++)
                        ind_t.set(perm(j)) = ind(j);
                    set(ind) = t(ind_t);
                }
        }
    }
    void Tensor::operator=(double xx)
    {
        for (int i = 0; i < n_comp; i++)
            *cmp[i] = xx;
    }
    void Tensor::operator+=(const Tensor& t)
    {

        assert(&espace == &t.espace);
        assert(valence == t.valence);
        assert(basis == t.basis);

        // Case when one does not care about the indices :
        if ((!t.name_affected) || (!name_affected)) {

            for (int i = 0; i < valence; i++)
                assert(t.type_indice(i) == type_indice(i));

            for (int i = 0; i < n_comp; i++) {
                int place_t = t.position(indices(i));
                *cmp[i] += *t.cmp[place_t];
            }
        } else {
            // Find the permutation :
            Array<int> perm(valence);
            bool same_ind = find_indices(t, perm);
            if (!same_ind) {
                KADATH_THROW("Indices do not match in Tensor::operator+=");
            } else
                for (int i = 0; i < n_comp; i++) {
                    Array<int> ind(indices(i));
                    Array<int> ind_t(valence);
                    for (int j = 0; j < valence; j++)
                        ind_t.set(perm(j)) = ind(j);
                    set(ind) += t(ind_t);
                }
        }
    }

    void Tensor::operator-=(const Tensor& t)
    {
        assert(&espace == &t.espace);
        assert(valence == t.valence);
        assert(basis == t.basis);

        // Case when one does not care about the indices :
        if ((!t.name_affected) || (!name_affected)) {

            for (int i = 0; i < valence; i++)
                assert(t.type_indice(i) == type_indice(i));

            for (int i = 0; i < n_comp; i++) {
                int place_t = t.position(indices(i));
                *cmp[i] -= *t.cmp[place_t];
            }
        } else {
            // Find the permutation :
            Array<int> perm(valence);
            bool same_ind = find_indices(t, perm);
            if (!same_ind) {
                KADATH_THROW("Indices do not match in Tensor::operator-=");
            } else
                for (int i = 0; i < n_comp; i++) {
                    Array<int> ind(indices(i));
                    Array<int> ind_t(valence);
                    for (int j = 0; j < valence; j++)
                        ind_t.set(perm(j)) = ind(j);
                    set(ind) -= t(ind_t);
                }
        }
    }

    //********************//
    // OPERATEURS UNAIRES //
    //********************//

    Tensor operator+(const Tensor& t)
    {

        return t;
    }

    Tensor operator-(const Tensor& t)
    {

        Tensor res(t.espace, t.valence, t.type_indice, t.basis);

        res.name_affected = t.name_affected;
        if (res.name_affected) {
            for (int i = 0; i < res.valence; i++)
                res.name_indice[i] = t.name_indice[i];
        }

        for (int i = 0; i < res.n_comp; i++) {
            Array<int> ind(res.indices(i));
            res.set(ind) = -(t(ind));
        }
        return res;
    }

    //**********
    // ADDITION
    //*********

    Tensor operator+(const Tensor& t1, const Tensor& t2)
    {

        Tensor res(t1);
        res += t2;
        return res;
    }

    Scalar operator+(const Tensor& t1, const Scalar& t2)
    {
        assert(t1.valence == 0);
        Scalar res(t1);
        res += t2;
        return res;
    }

    Scalar operator+(const Scalar& t1, const Tensor& t2)
    {
        assert(t2.valence == 0);
        Scalar res(t2);
        res += t1;
        return res;
    }

    Tensor operator+(const Tensor& t, double xx)
    {
        Tensor res(t);
        for (int i = 0; i < res.n_comp; i++) {
            Array<int> ind(res.indices(i));
            res.set(ind) = res(ind) + xx;
        }
        return res;
    }

    Tensor operator+(double xx, const Tensor& t)
    {
        Tensor res(t);
        for (int i = 0; i < res.n_comp; i++) {
            Array<int> ind(res.indices(i));
            res.set(ind) = res(ind) + xx;
        }
        return res;
    }

    //*************
    // SOUSTRACTION
    //*************

    Tensor operator-(const Tensor& t1, const Tensor& t2)
    {
        Tensor res(t1);
        res -= t2;
        return res;
    }

    Scalar operator-(const Tensor& t1, const Scalar& t2)
    {
        assert(t1.valence == 0);
        Scalar res(t1);
        res -= t2;
        return res;
    }

    Scalar operator-(const Scalar& t1, const Tensor& t2)
    {
        assert(t2.valence == 0);
        Scalar res(-t2);
        res += t1;
        return res;
    }

    Tensor operator-(const Tensor& t, double xx)
    {
        Tensor res(t);
        for (int i = 0; i < res.n_comp; i++) {
            Array<int> ind(res.indices(i));
            res.set(ind) = res(ind) - xx;
        }
        return res;
    }

    Tensor operator-(double xx, const Tensor& t)
    {
        Tensor res(t);
        for (int i = 0; i < res.n_comp; i++) {
            Array<int> ind(res.indices(i));
            res.set(ind) = -res(ind) + xx;
        }
        return res;
    }

    //***************
    // MULTIPLICATION
    //***************

    Tensor operator*(const Scalar& t1, const Tensor& t2)
    {
        assert(&t1.espace == &t2.espace);
        Tensor res(t2);

        for (int ic = 0; ic < res.n_comp; ic++) {
            Array<int> ind = res.indices(ic);
            res.set(ind) *= t1;
        }
        return res;
    }

    Tensor operator*(const Tensor& t2, const Scalar& t1)
    {
        return t1 * t2;
    }

    Tensor operator*(double x, const Tensor& t)
    {

        Tensor res(t);

        for (int i = 0; i < res.n_comp; i++) {
            Array<int> ind(res.indices(i));
            res.set(ind) *= x;
        }

        return res;
    }

    Tensor operator*(const Tensor& t, double x)
    {
        return x * t;
    }

    Tensor operator*(int m, const Tensor& t)
    {
        Tensor res(t);

        for (int i = 0; i < res.n_comp; i++) {
            Array<int> ind(res.indices(i));
            res.set(ind) *= m;
        }

        return res;
    }

    Tensor operator*(const Tensor& t, int m)
    {
        return m * t;
    }

    Tensor operator*(const Tensor& t1, const Tensor& t2)
    {

        assert(&t1.espace == &t2.espace);

        if (t1.valence == 0) {
            Tensor res(Scalar(t1) * t2);
            return res;
        } else if (t2.valence == 0) {
            Tensor res(t1 * Scalar(t2));
            return res;
        } else {
            assert(t1.basis == t2.basis);

            if ((!t1.name_affected) || (!t2.name_affected)) {
                // Standard product
                int val_res = t1.valence + t2.valence;
                Array<int> ind_res(val_res);
                for (int i = 0; i < t1.valence; i++)
                    ind_res.set(i) = t1.type_indice(i);
                for (int i = 0; i < t2.valence; i++)
                    ind_res.set(i + t1.valence) = t2.type_indice(i);

                const Base_tensor& base = (t1.valence != 0) ? t1.basis : t2.basis;
                Tensor res(t1.espace, val_res, ind_res, base);

                Array<int> ind1(t1.valence);
                Array<int> ind2(t2.valence);
                for (int i = 0; i < res.n_comp; i++) {
                    ind_res = res.indices(i);
                    for (int j = 0; j < t1.valence; j++)
                        ind1.set(j) = ind_res(j);
                    for (int j = 0; j < t2.valence; j++)
                        ind2.set(j) = ind_res(j + t1.valence);
                    res.set(ind_res) = t1(ind1) * t2(ind2);
                }

                return res;
            }

            else {
                // Case with indices name

                // Check if one needs to sum on some indices :
                Array<int> sum_1(t1.valence);
                Array<int> sum_2(t2.valence);
                for (int i = 0; i < t2.valence; i++)
                    sum_2.set(i) = -1;

                for (int i = 0; i < t1.valence; i++) {
                    sum_1.set(i) = -1;
                    for (int j = 0; j < t2.valence; j++)
                        if (t1.name_indice[i] == t2.name_indice[j]) {
                            sum_1.set(i) = j;
                            sum_2.set(j) = i;
                        }
                    if ((sum_1(i) != -1) && (t1.type_indice(i) == t2.type_indice(sum_1(i)))) {
                        KADATH_THROW("Can not sum on indices of the same type in operator*");
                    }
                }

                // Valence of the result
                int val_res = 0;
                for (int i = 0; i < t1.valence; i++)
                    if (sum_1(i) == -1)
                        val_res++;
                for (int i = 0; i < t2.valence; i++)
                    if (sum_2(i) == -1)
                        val_res++;

                if (val_res > 0) {
                    // Type on indices
                    Array<int> ind_res(val_res);
                    int conte = 0;
                    for (int i = 0; i < t1.valence; i++)
                        if (sum_1(i) == -1) {
                            ind_res.set(conte) = t1.type_indice(i);
                            conte++;
                        }
                    for (int i = 0; i < t2.valence; i++)
                        if (sum_2(i) == -1) {
                            ind_res.set(conte) = t2.type_indice(i);
                            conte++;
                        }

                    // Le tenseur
                    const Base_tensor& base = (t1.valence != 0) ? t1.basis : t2.basis;
                    Tensor res(t1.espace, val_res, ind_res, base);

                    // Name of the remaining variables :
                    res.name_affected = true;
                    conte = 0;
                    for (int i = 0; i < t1.valence; i++)
                        if (sum_1(i) == -1) {
                            res.name_indice[conte] = t1.name_indice[i];
                            conte++;
                        }
                    for (int i = 0; i < t2.valence; i++)
                        if (sum_2(i) == -1) {
                            res.name_indice[conte] = t2.name_indice[i];
                            conte++;
                        }

                    Array<bool> first(res.get_n_comp());
                    first = true;

                    Index pos_t1(t1);
                    Index pos_t2(t2);
                    Index pos_res(res);

                    do {
                        pos_t2.set_start();
                        do {
                            // Check if the summation indices are the same
                            bool same = true;
                            for (int i = 0; i < t1.valence; i++)
                                if (sum_1(i) != -1)
                                    if (pos_t1(i) != pos_t2(sum_1(i)))
                                        same = false;
                            // if the same :
                            if (same) {
                                conte = 0;
                                for (int i = 0; i < t1.valence; i++)
                                    if (sum_1(i) == -1) {
                                        pos_res.set(conte) = pos_t1(i);
                                        conte++;
                                    }
                                for (int i = 0; i < t2.valence; i++)
                                    if (sum_2(i) == -1) {
                                        pos_res.set(conte) = pos_t2(i);
                                        conte++;
                                    }

                                int ind(res.position(pos_res));
                                if (first(ind)) {
                                    res.set(pos_res) = t1(pos_t1) * t2(pos_t2);
                                    first.set(ind) = false;
                                } else
                                    res.set(pos_res) += t1(pos_t1) * t2(pos_t2);
                            }
                        } while (pos_t2.inc());
                    } while (pos_t1.inc());

                    return res;
                } else {
                    // Result is a scalar :
                    Scalar res(t1.espace);
                    Index pos_t1(t1);
                    Index pos_t2(t2);
                    int first = true;

                    do {
                        pos_t2.set_start();
                        do {
                            // Check if the summation indices are the same
                            bool same = true;
                            for (int i = 0; i < t1.valence; i++)
                                if (sum_1(i) != -1)
                                    if (pos_t1(i) != pos_t2(sum_1(i)))
                                        same = false;
                            // if the same :
                            if (same) {
                                if (first) {
                                    res = t1(pos_t1) * t2(pos_t2);
                                    first = false;
                                } else
                                    res += t1(pos_t1) * t2(pos_t2);
                            }
                        } while (pos_t2.inc());
                    } while (pos_t1.inc());
                    return res;
                }
            }
        }
    }

    //*********
    // DIVISION
    //*********

    Tensor operator/(const Tensor& t1, const Scalar& s2)
    {

        // Protections
        assert(&t1.espace == &s2.espace);

        Tensor res(t1);

        for (int i = 0; i < res.n_comp; i++) {
            Array<int> ind(res.indices(i));
            res.set(ind) /= s2;
        }
        return res;
    }

    Tensor operator/(const Tensor& t, double x)
    {

        if (x == double(0)) {
            KADATH_THROW("Division by 0 in Tensor / double !");
        }

        if (x == double(1))
            return t;
        else {
            Tensor res(t);

            for (int i = 0; i < res.n_comp; i++) {
                Array<int> ind(res.indices(i));
                res.set(ind) /= x;
            }
            return res;
        }
    }

    Tensor operator/(const Tensor& t, int m)
    {

        return t / double(m);
    }

    double maxval(const Tensor& target)
    {

        Tensor so(target);

        double res = 0;
        int ndom = so.get_space().get_nbr_domains();

        for (int c = 0; c < so.get_n_comp(); c++) {
            Array<int> indices(so.indices(c));

            for (int d = 0; d < ndom; d++)
                if (!so(indices)(d).check_if_zero()) {

                    so.set(indices).coef_i();
                    Index pos(so.get_space().get_domain(d)->get_nbr_points());
                    do {

                        double value = so(indices)(d)(pos);

                        if (fabs(value) > res)
                            res = fabs(value);
                    } while (pos.inc());
                }
        }
        return res;
    }

    double minval(const Tensor& target)
    {

        Tensor so(target);

        double res = -1;
        int ndom = so.get_space().get_nbr_domains();

        for (int c = 0; c < so.get_n_comp(); c++) {
            Array<int> indices(so.indices(c));

            for (int d = 0; d < ndom; d++)
                if (!so(indices)(d).check_if_zero()) {

                    so.set(indices).coef_i();
                    Index pos(so.get_space().get_domain(d)->get_nbr_points());
                    do {

                        double value = so(indices)(d)(pos);
                        if (res < 0)
                            res = fabs(value);

                        if (fabs(value) < res)
                            res = fabs(value);
                    } while (pos.inc());
                }
        }
        return res;
    }

} // namespace Kadath
