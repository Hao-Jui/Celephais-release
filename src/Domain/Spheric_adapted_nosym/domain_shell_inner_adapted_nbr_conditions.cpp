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
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Domain/spheric_nosym_regularization.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    int Domain_shell_inner_adapted_nosym::nbr_conditions_val_domain(const Val_domain& so, int mlim, int order) const
    {
        Base_spectral fallback_base(3);
        const Base_spectral* count_base = &so.get_base();
        if (!count_base->is_def() || count_base->get_base_1d(1) == nullptr) {
            switch (get_type_base()) {
                case CHEB_TYPE:
                    set_cheb_base(fallback_base);
                    break;
                case LEG_TYPE:
                    set_legendre_base(fallback_base);
                    break;
                default:
                    KADATH_THROW("Unknown type of base in Domain_shell_inner_adapted_nosym::nbr_conditions_val_domain");
            }
            count_base = &fallback_base;
        }

        int res = 0;
        int kmin = 2 * mlim + 2;

        Index pos(nbr_coefs);
        do {
            bool indic = true;
            // True coef in phi ?
            if ((pos(2) == 1) || (pos(2) == nbr_coefs(2) - 1))
                indic = false;
            // Get base in theta :
            int baset = (*count_base->get_base_1d(1))(pos(2));
            indic = indic && detail::spheric_nosym_true_theta_coef(baset, pos(1), pos(2), kmin, nbr_coefs(1));
            // Order with respect to r :
            if (pos(0) > nbr_coefs(0) - order - 1)
                indic = false;

            if (indic)
                res++;
        } while (pos.inc());

        return res;
    }

    Array<int> Domain_shell_inner_adapted_nosym::nbr_conditions(const Tensor& tt, int dom, int order, int n_cmp,
                                                          Array<int>** p_cmp) const
    {

        int size = (n_cmp == -1) ? tt.get_n_comp() : n_cmp;
        Array<int> res(size);
        int val = tt.get_valence();
        switch (val) {
            case 0:
                if (!tt.is_m_order_affected())
                    res.set(0) = nbr_conditions_val_domain(tt()(dom), 0, order);
                else
                    res.set(0) = nbr_conditions_val_domain(tt()(dom), tt.get_parameters()->get_m_order(), order);
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1)(dom), 0, order);
                        res.set(1) = nbr_conditions_val_domain(tt(2)(dom), 0, order);
                        res.set(2) = nbr_conditions_val_domain(tt(3)(dom), 0, order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                res.set(i) = nbr_conditions_val_domain(tt(1)(dom), 0, order);
                            if ((*p_cmp[i])(0) == 2)
                                res.set(i) = nbr_conditions_val_domain(tt(2)(dom), 0, order);
                            if ((*p_cmp[i])(0) == 3)
                                res.set(i) = nbr_conditions_val_domain(tt(3)(dom), 0, order);
                        }
                    found = true;
                }
                // Spherical coordinates
                if (tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1)(dom), 0, order);
                        res.set(1) = nbr_conditions_val_domain(tt(2)(dom), 1, order);
                        res.set(2) = nbr_conditions_val_domain(tt(3)(dom), 1, order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                res.set(i) = nbr_conditions_val_domain(tt(1)(dom), 0, order);
                            if ((*p_cmp[i])(0) == 2)
                                res.set(i) = nbr_conditions_val_domain(tt(2)(dom), 1, order);
                            if ((*p_cmp[i])(0) == 3)
                                res.set(i) = nbr_conditions_val_domain(tt(3)(dom), 1, order);
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_shell_inner_adapted_nosym::nbr_conditions");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, order);
                        res.set(1) = nbr_conditions_val_domain(tt(1, 2)(dom), 0, order);
                        res.set(2) = nbr_conditions_val_domain(tt(1, 3)(dom), 0, order);
                        res.set(3) = nbr_conditions_val_domain(tt(2, 2)(dom), 0, order);
                        res.set(4) = nbr_conditions_val_domain(tt(2, 3)(dom), 0, order);
                        res.set(5) = nbr_conditions_val_domain(tt(3, 3)(dom), 0, order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 2)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 3)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 2)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 3)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 3)(dom), 0, order);
                        }
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, order);
                        res.set(1) = nbr_conditions_val_domain(tt(1, 2)(dom), 0, order);
                        res.set(2) = nbr_conditions_val_domain(tt(1, 3)(dom), 0, order);
                        res.set(3) = nbr_conditions_val_domain(tt(2, 1)(dom), 0, order);
                        res.set(4) = nbr_conditions_val_domain(tt(2, 2)(dom), 0, order);
                        res.set(5) = nbr_conditions_val_domain(tt(2, 3)(dom), 0, order);
                        res.set(6) = nbr_conditions_val_domain(tt(3, 1)(dom), 0, order);
                        res.set(7) = nbr_conditions_val_domain(tt(3, 2)(dom), 0, order);
                        res.set(8) = nbr_conditions_val_domain(tt(3, 3)(dom), 0, order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 2)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 3)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 1)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 2)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 3)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 1)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 2)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 3)(dom), 0, order);
                        }
                    found = true;
                }
                // Spherical coordinates and symetric
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, order);
                        res.set(1) = nbr_conditions_val_domain(tt(1, 2)(dom), 1, order);
                        res.set(2) = nbr_conditions_val_domain(tt(1, 3)(dom), 1, order);
                        res.set(3) = nbr_conditions_val_domain(tt(2, 2)(dom), 2, order);
                        res.set(4) = nbr_conditions_val_domain(tt(2, 3)(dom), 2, order);
                        res.set(5) = nbr_conditions_val_domain(tt(3, 3)(dom), 2, order);
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 2)(dom), 1, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 3)(dom), 1, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 2)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 3)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 3)(dom), 2, order);
                        }
                    found = true;
                }
                // Spherical coordinates and not symetric
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        res.set(0) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, order);
                        res.set(1) = nbr_conditions_val_domain(tt(1, 2)(dom), 1, order);
                        res.set(2) = nbr_conditions_val_domain(tt(1, 3)(dom), 1, order);
                        res.set(3) = nbr_conditions_val_domain(tt(2, 1)(dom), 1, order);
                        res.set(4) = nbr_conditions_val_domain(tt(2, 2)(dom), 2, order);
                        res.set(5) = nbr_conditions_val_domain(tt(2, 3)(dom), 2, order);
                        res.set(6) = nbr_conditions_val_domain(tt(3, 1)(dom), 1, order);
                        res.set(7) = nbr_conditions_val_domain(tt(3, 2)(dom), 2, order);
                        res.set(8) = nbr_conditions_val_domain(tt(3, 3)(dom), 2, order);

                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 1)(dom), 0, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 2)(dom), 1, order);
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(1, 3)(dom), 1, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 1)(dom), 1, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 2)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(2, 3)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 1)(dom), 1, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 2)(dom), 2, order);
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                res.set(i) = nbr_conditions_val_domain(tt(3, 3)(dom), 2, order);
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_shell_inner_adapted_nosym::nbr_conditions");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_shell_inner_adapted_nosym::nbr_conditions" << endl;
                break;
        }
        return res;
    }

    bool Domain_shell_inner_adapted_nosym::describe_volume_residual_rows(
        const Tensor& tt, int dom, int order, const Array<int>& ncond,
        int n_cmp, Array<int>** p_cmp,
        std::vector<ResidualRowDescriptor>& descriptors) const
    {
        descriptors.clear();
        std::vector<int> components;
        if (!residual_tensor_components_in_tau_order(
                tt, dom, n_cmp, p_cmp, components) ||
            ncond.get_nbr() != components.size()) {
            return false;
        }

        const int tensor_basis = tt.get_basis().get_basis(dom);
        if (tt.get_valence() > 0 && tensor_basis != CARTESIAN_BASIS &&
            tensor_basis != SPHERICAL_BASIS) {
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
            int mlim = 0;
            if (tt.get_valence() == 0 && tt.is_m_order_affected()) {
                mlim = tt.get_parameters()->get_m_order();
            } else if (tensor_basis == SPHERICAL_BASIS) {
                for (std::size_t index = 0; index < indices.get_nbr(); ++index)
                    if (indices(index) != 1)
                        ++mlim;
            }

            const int kmin = 2 * mlim + 2;
            const std::size_t before = descriptors.size();
            Index position(nbr_coefs);
            do {
                bool emitted = position(2) != 1 &&
                               position(2) != nbr_coefs(2) - 1;
                const int theta_basis =
                    (*field.get_base().get_base_1d(1))(position(2));
                emitted = emitted && detail::spheric_nosym_true_theta_coef(
                                         theta_basis, position(1), position(2),
                                         kmin, nbr_coefs(1));
                emitted = emitted &&
                          position(0) <= nbr_coefs(0) - order - 1;
                if (emitted) {
                    ResidualRowDescriptor descriptor;
                    if (!append_volume_residual_row(
                            field, dom, component, position(2), descriptor)) {
                        descriptors.clear();
                        return false;
                    }
                    descriptors.push_back(std::move(descriptor));
                }
            } while (position.inc());
            if (descriptors.size() - before !=
                static_cast<std::size_t>(ncond(static_cast<int>(slot)))) {
                descriptors.clear();
                return false;
            }
        }
        return true;
    }
} // namespace Kadath
