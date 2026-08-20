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
#include "For_Kadath/Domain/spheric_symphi.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    int Domain_shell_symphi::nbr_conditions_val_domain_boundary(const Val_domain& so) const
    {

        int res = 0;

        for (int k = 0; k < nbr_coefs(2); k++)
            for (int j = 0; j < nbr_coefs(1); j++) {
                bool indic = true;

                int mquant;
                // Base in phi
                int basep = (*so.get_base().bases_1d[2])(0);
                switch (basep) {
                    case COS_EVEN:
                        mquant = 2 * k;
                        break;
                    case COS_ODD:
                        if (k == nbr_coefs(2) - 1)
                            indic = false;
                        mquant = 2 * k + 1;
                        break;
                    case SIN_EVEN:
                        if ((k == 0) || (k == nbr_coefs(2) - 1))
                            indic = false;
                        mquant = 2 * k;
                        break;
                    case SIN_ODD:
                        if (k == nbr_coefs(2) - 1)
                            indic = false;
                        mquant = 2 * k + 1;
                        break;
                    default:
                        KADATH_THROW("Unknow phi basis in Domain_shell_symphi::nbr_unknowns_val_domain");
                }

                // Get base in theta :
                int baset = (*so.get_base().bases_1d[1])(k);
                if (indic) {
                    switch (baset) {
                        case COS_EVEN:
                            if ((j == 0) && (mquant > 0))
                                indic = false;
                            break;
                        case COS_ODD:
                            if ((j == nbr_coefs(1) - 1) || ((j == 0) && (mquant > 0)))
                                indic = false;
                            break;
                        case SIN_EVEN:
                            if (((j == 1) && (mquant > 1)) || (j == 0) || (j == nbr_coefs(1) - 1))
                                indic = false;
                            break;
                        case SIN_ODD:
                            if (((j == 0) && (mquant > 1)) || (j == nbr_coefs(1) - 1))
                                indic = false;
                            break;
                        default:
                            KADATH_THROW("Unknow theta basis in Domain_shell_symphi::nbr_unknowns_val_domain");
                    }
                }

                if (indic)
                    res++;
            }
        return res;
    }

    Array<int> Domain_shell_symphi::nbr_conditions_boundary(const Tensor& tt, int dom, int bound, int n_cmp,
                                                            Array<int>** p_cmp) const
    {

        // Check boundary
        if ((bound != OUTER_BC) && (bound != INNER_BC)) {
            KADATH_THROW("Unknown boundary in Domain_shell_symphi::nbr_conditions_boundary");
        }

        int size = (n_cmp == -1) ? tt.get_n_comp() : n_cmp;
        Array<int> res(size);
        int val = tt.get_valence();
        switch (val) {
            case 0:
                res.set(0) = nbr_conditions_val_domain_boundary(tt()(dom));
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain_boundary(tt(1)(dom));
                        res.set(1) = nbr_conditions_val_domain_boundary(tt(2)(dom));
                        res.set(2) = nbr_conditions_val_domain_boundary(tt(3)(dom));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1)(dom));
                            if ((*p_cmp[i])(0) == 2)
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2)(dom));
                            if ((*p_cmp[i])(0) == 3)
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3)(dom));
                        }
                    found = true;
                }

                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_shell_symphi::nbr_conditions_boundary");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom));
                        res.set(1) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom));
                        res.set(2) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom));
                        res.set(3) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom));
                        res.set(4) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom));
                        res.set(5) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom));
                        }
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom));
                        res.set(1) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom));
                        res.set(2) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom));
                        res.set(3) = nbr_conditions_val_domain_boundary(tt(2, 1)(dom));
                        res.set(4) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom));
                        res.set(5) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom));
                        res.set(6) = nbr_conditions_val_domain_boundary(tt(3, 1)(dom));
                        res.set(7) = nbr_conditions_val_domain_boundary(tt(3, 2)(dom));
                        res.set(8) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 1)(dom));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 1)(dom));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 2)(dom));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom));
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_shell_symphi::nbr_conditions_boundary");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_shell_symphi::nbr_conditions_boundary" << endl;
                break;
        }
        return res;
    }
} // namespace Kadath
