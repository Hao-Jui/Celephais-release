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
 *   2026-08-07  Replaced GSL permutation handling with the C++ standard library.
 */

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Ope_inverse_nodet::Ope_inverse_nodet(const System_of_eqs* zesys, Ope_eq* target)
        : Ope_eq(zesys, target->get_dom(), 1)
    {
        parts[0].reset(target);
    }

    Ope_inverse_nodet::~Ope_inverse_nodet() {}

    Term_eq Ope_inverse_nodet::action() const
    {
        ScopedOpeActionProfile ope_action_profile_scope(*this);

        Term_eq target(parts[0]->action());

        // Various checks
        if (target.type_data != TERM_T) {
            KADATH_THROW("Ope_inverse only defined with respect for a tensor");
        }
        if (target.val_t->get_valence() != 2) {
            KADATH_THROW("Ope_inverse only defined with respect to a second order tensor");
        }
        if (target.val_t->get_index_type(0) != target.val_t->get_index_type(1)) {
            KADATH_THROW("Ope_inverse only defined with respect to a tensor which indices are of the same type");
        }
        int typeresult = -target.val_t->get_index_type(0);
        int dim = target.val_t->get_space().get_ndim();
        if (dim != 3) {
            KADATH_THROW("Ope_inverse_nodet currently supports only 3D tensors");
        }

        bool m_doder = (target.der_t == nullptr) ? false : true;

        std::vector<std::unique_ptr<Term_eq>> res(dim * (dim + 1) / 2); // 6 components to compute...
        Scalar val(
            target.val_t
                ->get_space()); // val and cmpval are more or less the same intermediate, but with different types
        Scalar der(target.val_t->get_space());
        Val_domain cmpval(target.val_t->get_space().get_domain(dom));
        Val_domain cmpder(target.val_t->get_space().get_domain(dom));

        // compute the transposed comatrix
        double signature;
        std::vector<int> permutation(dim - 1);
        for (int i(1); i <= dim; ++i) // sum over components, only the upper triangular part
        {
            for (int j(i); j <= dim; ++j) {
                cmpval = 0.0; // initialise to 0 before each computation
                cmpder = 0.0; // initialise to 0 before each computation
                const double parity_sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
                std::iota(permutation.begin(), permutation.end(), 0);
                do {
                    signature = sign(permutation);
                    int p1(permutation[0] + 1);
                    int p2(permutation[1] + 1);
                    vector<int> index1, index2;    // manage the indices of the transposed comatrix
                    index1 = ind_com(i, j, p1, 1); // we need to play a bit with indices to get transpose(COM(metric))
                    index2 = ind_com(i, j, p2, 2);
                    cmpval += parity_sign * signature * (*target.val_t)(index1[0], index1[1])(dom) *
                              (*target.val_t)(index2[0], index2[1])(dom); // sum on permutations
                    if (m_doder)
                        cmpder += parity_sign * signature * (*target.der_t)(index1[0], index1[1])(dom) *
                                      (*target.val_t)(index2[0], index2[1])(dom) +
                                  parity_sign * signature * (*target.val_t)(index1[0], index1[1])(dom) *
                                      (*target.der_t)(index2[0], index2[1])(dom);
                } while (std::next_permutation(permutation.begin(), permutation.end()));
                val.set_domain(dom) = cmpval; // now we have the component of the inverse matrix
                if (m_doder) {
                    der.set_domain(dom) = cmpder;
                    res[j - 1 + dim * (i - 1) - i * (i - 1) / 2] =
                        std::make_unique<Term_eq>(
                            dom, val, der); // just save upper triangular components in a 1D array of Term_eq
                                            // (the indice will run throug 0, 1, 2, 3, 4, 5 with i, j)
                } else
                    res[j - 1 + dim * (i - 1) - i * (i - 1) / 2] = std::make_unique<Term_eq>(dom, val);
            }
        }
        // Value field :
        Metric_tensor resval(target.val_t->get_space(), typeresult, target.val_t->get_basis());
        Metric_tensor resder(target.val_t->get_space(), typeresult, target.val_t->get_basis());
        for (int i(1); i <= dim; ++i)
            for (int j(i); j <= dim; ++j)
                resval.set(i, j) = res[j - 1 + dim * (i - 1) - i * (i - 1) / 2]->get_val_t();

        // Der field
        if (m_doder) {
            for (int i(1); i <= dim; ++i)
                for (int j(i); j <= dim; ++j)
                    resder.set(i, j) = res[j - 1 + dim * (i - 1) - i * (i - 1) / 2]->get_der_t();
        }

        if (!m_doder)
            return Term_eq(dom, resval);
        else
            return Term_eq(dom, resval, resder);
    }
} // namespace Kadath
