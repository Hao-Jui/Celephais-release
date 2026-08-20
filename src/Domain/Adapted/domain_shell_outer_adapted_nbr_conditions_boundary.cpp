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
#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    int Domain_shell_outer_adapted::nbr_conditions_val_domain_boundary(const Val_domain& so, int mlim) const
    {

        int res = 0;
        int kmin = 2 * mlim + 2;

        for (int k = 0; k < nbr_coefs(2); k++)
            for (int j = 0; j < nbr_coefs(1); j++) {
                bool indic = true;
                // True coef in phi ?
                if ((k == 1) || (k == nbr_coefs(2) - 1))
                    indic = false;
                // Get base in theta :
                int baset = (*so.get_base().bases_1d[1])(k);
                switch (baset) {
                    case COS_EVEN:
                        if ((j == 0) && (k >= kmin))
                            indic = false;
                        break;
                    case COS_ODD:
                        if ((j == nbr_coefs(1) - 1) || ((j == 0) && (k >= kmin)))
                            indic = false;
                        break;
                    case SIN_EVEN:
                        if (((j == 1) && (k >= kmin + 2)) || (j == 0) || (j == nbr_coefs(1) - 1))
                            indic = false;
                        break;
                    case SIN_ODD:
                        if (((j == 0) && (k >= kmin + 2)) || (j == nbr_coefs(1) - 1))
                            indic = false;
                        break;
                    default:
                        KADATH_THROW("Unknow theta basis in Domain_shell_outer_adapted::nbr_conditions_val_boundary");
                }

                if (indic)
                    res++;
            }
        return res;
    }

    Array<int> Domain_shell_outer_adapted::nbr_conditions_boundary(const Tensor& tt, int dom, int bound, int n_cmp,
                                                                   Array<int>** p_cmp) const
    {

        // Check boundary
        if ((bound != INNER_BC) && (bound != OUTER_BC)) {
            KADATH_THROW("Unknown boundary in Domain_shell_outer_adapted::nbr_conditions_boundary");
        }

        int size = (n_cmp == -1) ? tt.get_n_comp() : n_cmp;
        Array<int> res(size);
        int val = tt.get_valence();
        switch (val) {
            case 0:
                if (!tt.is_m_order_affected())
                    res.set(0) = nbr_conditions_val_domain_boundary(tt()(dom), 0);
                else
                    res.set(0) = nbr_conditions_val_domain_boundary(tt()(dom), tt.get_parameters()->get_m_order());
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain_boundary(tt(1)(dom), 0);
                        res.set(1) = nbr_conditions_val_domain_boundary(tt(2)(dom), 0);
                        res.set(2) = nbr_conditions_val_domain_boundary(tt(3)(dom), 0);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1)(dom), 0);
                            if ((*p_cmp[i])(0) == 2)
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2)(dom), 0);
                            if ((*p_cmp[i])(0) == 3)
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3)(dom), 0);
                        }
                    found = true;
                }
                // Spherical coordinates
                if (tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain_boundary(tt(1)(dom), 0);
                        res.set(1) = nbr_conditions_val_domain_boundary(tt(2)(dom), 1);
                        res.set(2) = nbr_conditions_val_domain_boundary(tt(3)(dom), 1);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1)(dom), 0);
                            if ((*p_cmp[i])(0) == 2)
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2)(dom), 1);
                            if ((*p_cmp[i])(0) == 3)
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3)(dom), 1);
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_shell_outer_adapted::nbr_conditions_boundary");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom), 0);
                        res.set(1) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom), 0);
                        res.set(2) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom), 0);
                        res.set(3) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom), 0);
                        res.set(4) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom), 0);
                        res.set(5) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom), 0);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom), 0);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom), 0);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom), 0);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom), 0);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom), 0);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom), 0);
                        }
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom), 0);
                        res.set(1) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom), 0);
                        res.set(2) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom), 0);
                        res.set(3) = nbr_conditions_val_domain_boundary(tt(2, 1)(dom), 0);
                        res.set(4) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom), 0);
                        res.set(5) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom), 0);
                        res.set(6) = nbr_conditions_val_domain_boundary(tt(3, 1)(dom), 0);
                        res.set(7) = nbr_conditions_val_domain_boundary(tt(3, 2)(dom), 0);
                        res.set(8) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom), 0);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom), 0);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom), 0);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom), 0);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 1)(dom), 0);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom), 0);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom), 0);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 1)(dom), 0);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 2)(dom), 0);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom), 0);
                        }
                    found = true;
                }
                // Spherical coordinates and symetric
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom), 0);
                        res.set(1) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom), 1);
                        res.set(2) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom), 1);
                        res.set(3) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom), 2);
                        res.set(4) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom), 2);
                        res.set(5) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom), 2);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom), 0);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom), 1);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom), 1);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom), 2);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom), 2);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom), 2);
                        }
                    found = true;
                }
                // Spherical coordinates and not symetric
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom), 0);
                        res.set(1) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom), 1);
                        res.set(2) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom), 1);
                        res.set(3) = nbr_conditions_val_domain_boundary(tt(2, 1)(dom), 1);
                        res.set(4) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom), 2);
                        res.set(5) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom), 2);
                        res.set(6) = nbr_conditions_val_domain_boundary(tt(3, 1)(dom), 1);
                        res.set(7) = nbr_conditions_val_domain_boundary(tt(3, 2)(dom), 2);
                        res.set(8) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom), 2);

                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 1)(dom), 0);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 2)(dom), 1);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(1, 3)(dom), 1);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 1)(dom), 1);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 2)(dom), 2);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(2, 3)(dom), 2);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 1)(dom), 1);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 2)(dom), 2);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain_boundary(tt(3, 3)(dom), 2);
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_shell_outer_adapted::nbr_conditions_boundary");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_shell_outer_adapted::nbr_conditions_boundary"
                     << endl;
                break;
        }
        return res;
    }

    bool Domain_shell_outer_adapted::describe_boundary_residual_rows(
        const Tensor& tt, int dom, int bound, const Array<int>& ncond,
        int n_cmp, Array<int>** p_cmp,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        std::vector<int> components;
        if (tt.get_valence() > 2 ||
            (tt.get_valence() == 1 && tt.get_n_comp() != 3) ||
            (tt.get_valence() == 2 && tt.get_n_comp() != 6 &&
             tt.get_n_comp() != 9) ||
            (bound != INNER_BC && bound != OUTER_BC) ||
            !residual_tensor_components_in_tau_order(
                tt, dom, n_cmp, p_cmp, components) ||
            ncond.get_nbr() != components.size() ||
            (tt.get_valence() > 0 &&
             tt.get_basis().get_basis(dom) != CARTESIAN_BASIS)) {
            return false;
        }

        for (std::size_t slot = 0; slot < components.size(); ++slot) {
            const int component = components[slot];
            const Array<int> indices = tt.indices(component);
            const Val_domain& field = tt.get_valence() == 0
                                          ? tt()(dom)
                                          : tt(indices)(dom);
            if (!field.get_base().is_def() ||
                field.get_base().get_base_1d(1) == nullptr) {
                descriptors.clear();
                return false;
            }
            const int mlim = tt.get_valence() == 0 &&
                                     tt.is_m_order_affected()
                                 ? tt.get_parameters()->get_m_order()
                                 : 0;
            const int kmin = 2 * mlim + 2;
            const std::size_t before = descriptors.size();
            for (int k = 0; k < nbr_coefs(2) - 1; ++k) {
                if (k == 1)
                    continue;
                const int theta_basis =
                    (*field.get_base().get_base_1d(1))(k);
                for (int j = 0; j < nbr_coefs(1); ++j) {
                    bool emitted = false;
                    switch (theta_basis) {
                        case COS_EVEN:
                            emitted = k < kmin || j != 0;
                            break;
                        case COS_ODD:
                            emitted = j != nbr_coefs(1) - 1 &&
                                      (k < kmin || j != 0);
                            break;
                        case SIN_EVEN:
                            emitted = j != 0 &&
                                      j != nbr_coefs(1) - 1 &&
                                      (k < kmin + 2 || j != 1);
                            break;
                        case SIN_ODD:
                            emitted = j != nbr_coefs(1) - 1 &&
                                      (k < kmin + 2 || j != 0);
                            break;
                        default:
                            descriptors.clear();
                            return false;
                    }
                    if (!emitted)
                        continue;
                    ResidualRowDescriptor descriptor;
                    if (!append_volume_residual_row(
                            field, dom, component, k, descriptor)) {
                        descriptors.clear();
                        return false;
                    }
                    descriptors.push_back(std::move(descriptor));
                }
            }
            if (descriptors.size() - before !=
                static_cast<std::size_t>(ncond(static_cast<int>(slot)))) {
                descriptors.clear();
                return false;
            }
        }
        return true;
    }
} // namespace Kadath
