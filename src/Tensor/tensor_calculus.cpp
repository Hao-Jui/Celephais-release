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
 *   2026-08-06  RAII/span modernization.
 */

#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

#include <memory>
namespace Kadath
{
    Tensor Tensor::do_summation() const
    {
        if (!name_affected) {
            KADATH_THROW("Names of indices must be affected in Tensor::do_summation");
        }

        Array<int> sum_ind(valence);
        Array<int> nsame(valence);
        sum_ind = -1;
        nsame = 0;
        for (int i = 0; i < valence; i++)
            for (int j = i + 1; j < valence; j++) {
                if (name_indice[i] == name_indice[j]) {
                    nsame.set(i)++;
                    nsame.set(j)++;
                    if ((nsame(i) > 1) || (nsame(j) > 1)) {
                        KADATH_THROW("Too many identical indices in Tensor::do_summation");
                    }
                    sum_ind.set(i) = j;
                    sum_ind.set(j) = i;
                    if (type_indice(i) == type_indice(j)) {
                        KADATH_THROW("Can not sum on indices of the same type in Tensor::do_summation");
                    }
                }
            }

        int valence_res = 0;
        for (int i = 0; i < valence; i++)
            if (nsame(i) == 0)
                valence_res++;

        // difference Scalar or not :
        if (valence_res > 0) {
            Array<int> type_ind_res(valence_res);
            auto name_ind_res = std::make_unique_for_overwrite<char[]>(valence_res);
            int conte = 0;
            for (int i = 0; i < valence; i++)
                if (nsame(i) == 0) {
                    type_ind_res.set(conte) = type_indice(i);
                    name_ind_res[conte] = name_indice[i];
                    conte++;
                }

            Tensor res(espace, valence_res, type_ind_res, basis);
            res.name_affected = true;
            for (int i = 0; i < valence_res; i++)
                res.name_indice[i] = name_ind_res[i];

            res = 0;
            Array<bool> first(res.get_n_comp());
            first = true;

            // Loop on the various components :
            Index pos(*this);
            Index pos_res(res);
            do {
                bool take = true;
                for (int i = 0; i < valence; i++)
                    if (pos(sum_ind(i)) != pos(i))
                        take = false;

                if (take) {
                    conte = 0;
                    for (int i = 0; i < valence; i++)
                        if (nsame(i) == 0) {
                            pos_res.set(conte) = pos(i);
                            conte++;
                        }
                    int ind(res.position(pos_res));
                    if (first(ind)) {
                        for (int dd = 0; dd < espace.get_nbr_domains(); dd++)
                            res.set(pos_res).set_domain(dd).set_base() = (*this)(pos)(dd).get_base();
                        first.set(ind) = false;
                    }
                    res.set(pos_res) += (*this)(pos);
                }
            } while (pos.inc());
            return res;
        } else {
            // Scalar case :
            Scalar res(espace);
            res = 0;
            bool first = true;
            Index pos(*this);
            do {
                bool take = true;
                for (int i = 0; i < valence; i++)
                    if (pos(sum_ind(i)) != pos(i))
                        take = false;

                if (take) {
                    if (first) {
                        for (int dd = 0; dd < espace.get_nbr_domains(); dd++)
                            res.set_domain(dd).set_base() = (*this)(pos)(dd).get_base();
                        first = false;
                    }
                    res += (*this)(pos);
                }
            } while (pos.inc());
            return res;
        }
    }

    Tensor Tensor::grad() const
    {

        // Verif triad (only implemented for cartesian coordinates so far)
        for (int d = 0; d < ndom; d++)
            assert(basis.get_basis(d) == CARTESIAN_BASIS);

        Array<int> indices_res(valence + 1);
        indices_res.set(0) = COV;
        for (int i = 0; i < valence; i++)
            indices_res.set(i + 1) = type_indice(i);
        Tensor res(espace, valence + 1, indices_res, basis);

        for (int i = 0; i < ndim; i++) {
            indices_res.set(0) = i + 1;
            for (int j = 0; j < n_comp; j++) {
                Array<int> cible(indices(j));
                for (int k = 0; k < valence; k++)
                    indices_res.set(k + 1) = cible(k);
                res.set(indices_res) = operator()(cible).der_abs(i + 1);
            }
        }
        return res;
    }
} // namespace Kadath
