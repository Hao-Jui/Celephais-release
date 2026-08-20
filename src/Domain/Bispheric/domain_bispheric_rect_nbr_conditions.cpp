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
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    int Domain_bispheric_rect::nbr_conditions_val_domain(const Val_domain& so, int order) const
    {

        int forgot_chi = 0;
        switch (order) {
            case 0:
                forgot_chi = 0;
                break;
            case 1:
                forgot_chi = 1;
                break;
            case 2:
                forgot_chi = 1;
                break;
            default:
                KADATH_THROW("Unknown order in Domain_bispheric_rect::nbr_conditions_val_domain");
        }

        int res = 0;
        int basep = (*so.get_base().bases_1d[2])(0);

        // Loop on phi :
        for (int k = 0; k < nbr_coefs(2); k++)
            // Loop on chi ;
            for (int j = 0; j < nbr_coefs(1); j++) {
                bool true_other = true;

                switch (basep) {
                    case COS:
                        // Last odd ones
                        if ((k % 2 == 1) && (j == nbr_coefs(1) - 1 - forgot_chi))
                            true_other = false;
                        // Regularity for even ones :
                        if ((k != 0) && (k % 2 == 0) && (j == 0))
                            true_other = false;
                        if (j == nbr_coefs(1) - forgot_chi)
                            true_other = false;
                        break;
                    case SIN:
                        // sin(0)
                        if ((k == 0) || (k == nbr_coefs(2) - 1))
                            true_other = false;
                        // Last odd ones :
                        if ((k % 2 == 1) && (j == nbr_coefs(1) - 1 - forgot_chi))
                            true_other = false;
                        // Regularity for even ones :
                        if ((k % 2 == 0) && (j == 0))
                            true_other = false;
                        if (j == nbr_coefs(1) - forgot_chi)
                            true_other = false;
                        break;
                    default:
                        KADATH_THROW("Unknwon phi basis in Domain_bispheric_rect:nbr_conditions");
                }

                if (true_other)
                    res += nbr_coefs(0) - order;
            }
        return res;
    }

    Array<int> Domain_bispheric_rect::nbr_conditions(const Tensor& tt, int dom, int order, int n_cmp,
                                                     Array<int>** p_cmp) const
    {

        int size = (n_cmp == -1) ? tt.get_n_comp() : n_cmp;
        Array<int> res(size);
        int val = tt.get_valence();
        switch (val) {
            case 0:
                res.set(0) = nbr_conditions_val_domain(tt()(dom), order);
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1)(dom), order);
                        res.set(1) = nbr_conditions_val_domain(tt(2)(dom), order);
                        res.set(2) = nbr_conditions_val_domain(tt(3)(dom), order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                res.set(i) = nbr_conditions_val_domain(tt(1)(dom), order);
                            if ((*p_cmp[i])(0) == 2)
                                res.set(i) = nbr_conditions_val_domain(tt(2)(dom), order);
                            if ((*p_cmp[i])(0) == 3)
                                res.set(i) = nbr_conditions_val_domain(tt(3)(dom), order);
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_bispheric_rect::nbr_conditions");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1, 1)(dom), order);
                        res.set(1) = nbr_conditions_val_domain(tt(1, 2)(dom), order);
                        res.set(2) = nbr_conditions_val_domain(tt(1, 3)(dom), order);
                        res.set(3) = nbr_conditions_val_domain(tt(2, 2)(dom), order);
                        res.set(4) = nbr_conditions_val_domain(tt(2, 3)(dom), order);
                        res.set(5) = nbr_conditions_val_domain(tt(3, 3)(dom), order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 1)(dom), order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 2)(dom), order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 3)(dom), order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 2)(dom), order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 3)(dom), order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 3)(dom), order);
                        }
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1, 1)(dom), order);
                        res.set(1) = nbr_conditions_val_domain(tt(1, 2)(dom), order);
                        res.set(2) = nbr_conditions_val_domain(tt(1, 3)(dom), order);
                        res.set(3) = nbr_conditions_val_domain(tt(2, 1)(dom), order);
                        res.set(4) = nbr_conditions_val_domain(tt(2, 2)(dom), order);
                        res.set(5) = nbr_conditions_val_domain(tt(2, 3)(dom), order);
                        res.set(6) = nbr_conditions_val_domain(tt(3, 1)(dom), order);
                        res.set(7) = nbr_conditions_val_domain(tt(3, 2)(dom), order);
                        res.set(8) = nbr_conditions_val_domain(tt(3, 3)(dom), order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 1)(dom), order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 2)(dom), order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 3)(dom), order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 1)(dom), order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 2)(dom), order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 3)(dom), order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 1)(dom), order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 2)(dom), order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 3)(dom), order);
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_bispheric_rect::nbr_conditions");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_bispheric_rect::nbr_conditions" << endl;
                break;
        }
        return res;
    }

    bool Domain_bispheric_rect::describe_volume_residual_rows(
        const Tensor& tt, int dom, int order, const Array<int>& ncond,
        int n_cmp, Array<int>** p_cmp,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        std::vector<int> components;
        if (order < 0 || order > 2 ||
            !residual_tensor_components_in_tau_order(
                tt, dom, n_cmp, p_cmp, components) ||
            ncond.get_nbr() != components.size() ||
            (tt.get_valence() > 0 &&
             tt.get_basis().get_basis(dom) != CARTESIAN_BASIS)) {
            return false;
        }

        const int forgot_chi = order == 0 ? 0 : 1;
        const int forgot_eta = order;
        for (std::size_t slot = 0; slot < components.size(); ++slot) {
            const int component = components[slot];
            const Array<int> indices = tt.indices(component);
            const Val_domain& field = tt.get_valence() == 0
                                          ? tt()(dom)
                                          : tt(indices)(dom);
            if (!field.get_base().is_def() ||
                field.get_base().get_base_1d(2) == nullptr) {
                descriptors.clear();
                return false;
            }
            const int phi_basis = (*field.get_base().get_base_1d(2))(0);
            if (phi_basis != COS && phi_basis != SIN) {
                descriptors.clear();
                return false;
            }

            const std::size_t before = descriptors.size();
            for (int k = 0; k < nbr_coefs(2); ++k) {
                for (int j = 0; j < nbr_coefs(1) - forgot_chi; ++j) {
                    bool emitted = true;
                    if (phi_basis == COS) {
                        emitted = !((k % 2 == 1) &&
                                    (j == nbr_coefs(1) - 1 - forgot_chi));
                        emitted = emitted &&
                                  (k == 0 || k % 2 == 1 || j != 0);
                    } else {
                        emitted = k != 0 && k != nbr_coefs(2) - 1;
                        emitted = emitted &&
                                  !((k % 2 == 1) &&
                                    (j == nbr_coefs(1) - 1 - forgot_chi));
                        emitted = emitted && (k % 2 == 1 || j != 0);
                    }
                    if (!emitted)
                        continue;
                    for (int i = 0; i < nbr_coefs(0) - forgot_eta; ++i) {
                        ResidualRowDescriptor descriptor;
                        if (!append_volume_residual_row(
                                field, dom, component, k, descriptor)) {
                            descriptors.clear();
                            return false;
                        }
                        descriptors.push_back(std::move(descriptor));
                    }
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
