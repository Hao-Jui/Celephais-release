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
    void Domain_shell_outer_adapted_nosym::export_tau_val_domain_boundary(const Val_domain& so, int mlim, int bound,
                                                                    Array<double>& sec, int& pos_sec, int ncond) const
    {

        if (so.check_if_zero())
            pos_sec += ncond;
        else {
            so.coef();
            int kmin = 2 * mlim + 2;
            Index pos_cf(nbr_coefs);
            Index pos_galerkin(nbr_coefs);

            // Loop on phi :
            for (int k = 0; k < nbr_coefs(2) - 1; k++)
                if (k != 1) {
                    pos_cf.set(2) = k;
                    // Loop on theta
                    int baset = (*so.get_base().bases_1d[1])(k);
                    const bool uses_theta_galerkin =
                        detail::spheric_nosym_uses_theta_galerkin(baset, k, kmin);
                    double anchor_boundary[2] = {0., 0.};
                    if (uses_theta_galerkin) {
                        pos_galerkin.set(2) = k;
                        const int first_even_theta = (baset == COS) ? 2 : 4;
                        if (first_even_theta < nbr_coefs(1) - 1) {
                            pos_galerkin.set(1) = detail::spheric_nosym_theta_anchor(baset, first_even_theta);
                            anchor_boundary[0] = val_boundary(bound, so, pos_galerkin);
                        }
                        if (3 < nbr_coefs(1) - 1) {
                            pos_galerkin.set(1) = detail::spheric_nosym_theta_anchor(baset, 3);
                            anchor_boundary[1] = val_boundary(bound, so, pos_galerkin);
                        }
                    }
                    for (int j = 0; j < nbr_coefs(1); j++) {
                        pos_cf.set(1) = j;
                        switch (baset) {
                            case COS:
                                if (detail::spheric_nosym_true_theta_coef(baset, j, k, kmin, nbr_coefs(1))) {
                                    sec.set(pos_sec) = val_boundary(bound, so, pos_cf);
                                    if (uses_theta_galerkin) {
                                        sec.set(pos_sec) -= detail::spheric_nosym_export_anchor_weight(baset, j) *
                                                            anchor_boundary[j % 2];
                                    }
                                    pos_sec++;
                                }
                                break;
                            case SIN:
                                if (detail::spheric_nosym_true_theta_coef(baset, j, k, kmin, nbr_coefs(1))) {
                                    sec.set(pos_sec) = val_boundary(bound, so, pos_cf);
                                    if (uses_theta_galerkin) {
                                        sec.set(pos_sec) -= detail::spheric_nosym_export_anchor_weight(baset, j) *
                                                            anchor_boundary[j % 2];
                                    }
                                    pos_sec++;
                                }
                                break;
                            default:
                                KADATH_THROW("Unknow theta basis in Domain_shell_outer_adapted_nosym::export_tau_val_domain_boundary");
                        }
                    }
                }
        }
    }

    void Domain_shell_outer_adapted_nosym::export_tau_boundary(const Tensor& tt, int dom, int bound, Array<double>& res,
                                                         int& pos_res, const Array<int>& ncond, int n_cmp,
                                                         Array<int>** p_cmp) const
    {

        // Check boundary
        if ((bound != INNER_BC) && (bound != OUTER_BC)) {
            KADATH_THROW("Unknown boundary in Domain_shell_outer_adapted_nosym::export_tau_boundary");
        }

        int val = tt.get_valence();
        switch (val) {
            case 0:
                if (!tt.is_m_order_affected())
                    export_tau_val_domain_boundary(tt()(dom), 0, bound, res, pos_res, ncond(0));
                else
                    export_tau_val_domain_boundary(tt()(dom), tt.get_parameters()->get_m_order(), bound, res, pos_res,
                                                   ncond(0));
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    if (n_cmp == -1) {
                        export_tau_val_domain_boundary(tt(1)(dom), 0, bound, res, pos_res, ncond(0));
                        export_tau_val_domain_boundary(tt(2)(dom), 0, bound, res, pos_res, ncond(1));
                        export_tau_val_domain_boundary(tt(3)(dom), 0, bound, res, pos_res, ncond(2));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                export_tau_val_domain_boundary(tt(1)(dom), 0, bound, res, pos_res, ncond(i));
                            if ((*p_cmp[i])(0) == 2)
                                export_tau_val_domain_boundary(tt(2)(dom), 0, bound, res, pos_res, ncond(i));
                            if ((*p_cmp[i])(0) == 3)
                                export_tau_val_domain_boundary(tt(3)(dom), 0, bound, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                // Spherical coordinates
                if (tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) {
                    if (n_cmp == -1) {
                        export_tau_val_domain_boundary(tt(1)(dom), 0, bound, res, pos_res, ncond(0));
                        export_tau_val_domain_boundary(tt(2)(dom), 1, bound, res, pos_res, ncond(1));
                        export_tau_val_domain_boundary(tt(3)(dom), 1, bound, res, pos_res, ncond(2));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                export_tau_val_domain_boundary(tt(1)(dom), 0, bound, res, pos_res, ncond(i));
                            if ((*p_cmp[i])(0) == 2)
                                export_tau_val_domain_boundary(tt(2)(dom), 1, bound, res, pos_res, ncond(i));
                            if ((*p_cmp[i])(0) == 3)
                                export_tau_val_domain_boundary(tt(3)(dom), 1, bound, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_shell_outer_adapted_nosym::export_tau_boundary");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        export_tau_val_domain_boundary(tt(1, 1)(dom), 0, bound, res, pos_res, ncond(0));
                        export_tau_val_domain_boundary(tt(1, 2)(dom), 0, bound, res, pos_res, ncond(1));
                        export_tau_val_domain_boundary(tt(1, 3)(dom), 0, bound, res, pos_res, ncond(2));
                        export_tau_val_domain_boundary(tt(2, 2)(dom), 0, bound, res, pos_res, ncond(3));
                        export_tau_val_domain_boundary(tt(2, 3)(dom), 0, bound, res, pos_res, ncond(4));
                        export_tau_val_domain_boundary(tt(3, 3)(dom), 0, bound, res, pos_res, ncond(5));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain_boundary(tt(1, 1)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain_boundary(tt(1, 2)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(1, 3)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain_boundary(tt(2, 2)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(2, 3)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(3, 3)(dom), 0, bound, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        export_tau_val_domain_boundary(tt(1, 1)(dom), 0, bound, res, pos_res, ncond(0));
                        export_tau_val_domain_boundary(tt(1, 2)(dom), 0, bound, res, pos_res, ncond(1));
                        export_tau_val_domain_boundary(tt(1, 3)(dom), 0, bound, res, pos_res, ncond(2));
                        export_tau_val_domain_boundary(tt(2, 1)(dom), 0, bound, res, pos_res, ncond(3));
                        export_tau_val_domain_boundary(tt(2, 2)(dom), 0, bound, res, pos_res, ncond(4));
                        export_tau_val_domain_boundary(tt(2, 3)(dom), 0, bound, res, pos_res, ncond(5));
                        export_tau_val_domain_boundary(tt(3, 1)(dom), 0, bound, res, pos_res, ncond(6));
                        export_tau_val_domain_boundary(tt(3, 2)(dom), 0, bound, res, pos_res, ncond(7));
                        export_tau_val_domain_boundary(tt(3, 3)(dom), 0, bound, res, pos_res, ncond(8));

                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain_boundary(tt(1, 1)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain_boundary(tt(1, 2)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(1, 3)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain_boundary(tt(2, 1)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain_boundary(tt(2, 2)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(2, 3)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain_boundary(tt(3, 1)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain_boundary(tt(3, 2)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(3, 3)(dom), 0, bound, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                // Spherical coordinates and symetric
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        export_tau_val_domain_boundary(tt(1, 1)(dom), 0, bound, res, pos_res, ncond(0));
                        export_tau_val_domain_boundary(tt(1, 2)(dom), 1, bound, res, pos_res, ncond(1));
                        export_tau_val_domain_boundary(tt(1, 3)(dom), 1, bound, res, pos_res, ncond(2));
                        export_tau_val_domain_boundary(tt(2, 2)(dom), 2, bound, res, pos_res, ncond(3));
                        export_tau_val_domain_boundary(tt(2, 3)(dom), 2, bound, res, pos_res, ncond(4));
                        export_tau_val_domain_boundary(tt(3, 3)(dom), 2, bound, res, pos_res, ncond(5));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain_boundary(tt(1, 1)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain_boundary(tt(1, 2)(dom), 1, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(1, 3)(dom), 1, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain_boundary(tt(2, 2)(dom), 2, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(2, 3)(dom), 2, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(3, 3)(dom), 2, bound, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                // Spherical coordinates and not symetric
                if ((tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        export_tau_val_domain_boundary(tt(1, 1)(dom), 0, bound, res, pos_res, ncond(0));
                        export_tau_val_domain_boundary(tt(1, 2)(dom), 1, bound, res, pos_res, ncond(1));
                        export_tau_val_domain_boundary(tt(1, 3)(dom), 1, bound, res, pos_res, ncond(2));
                        export_tau_val_domain_boundary(tt(2, 1)(dom), 1, bound, res, pos_res, ncond(3));
                        export_tau_val_domain_boundary(tt(2, 2)(dom), 2, bound, res, pos_res, ncond(4));
                        export_tau_val_domain_boundary(tt(2, 3)(dom), 2, bound, res, pos_res, ncond(5));
                        export_tau_val_domain_boundary(tt(3, 1)(dom), 1, bound, res, pos_res, ncond(6));
                        export_tau_val_domain_boundary(tt(3, 2)(dom), 2, bound, res, pos_res, ncond(7));
                        export_tau_val_domain_boundary(tt(3, 3)(dom), 2, bound, res, pos_res, ncond(8));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain_boundary(tt(1, 1)(dom), 0, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain_boundary(tt(1, 2)(dom), 1, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(1, 3)(dom), 1, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain_boundary(tt(2, 1)(dom), 1, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain_boundary(tt(2, 2)(dom), 2, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(2, 3)(dom), 2, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain_boundary(tt(3, 1)(dom), 1, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain_boundary(tt(3, 2)(dom), 2, bound, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain_boundary(tt(3, 3)(dom), 2, bound, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_shell_outer_adapted_nosym::export_tau_boundary");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_shell_outer_adapted_nosym::export_tau_boundary"
                     << endl;
                break;
        }
    }

    // Matching export for the bispheric/binary coupling (added post-74e7f645,
    // required by Eq_bc/Eq_matching_import). Delegates to the A.5-validated
    // boundary export so count and emit stay consistent.
    void Domain_shell_outer_adapted_nosym::export_tau_val_domain_boundary_matching(const Val_domain& so, int mlim,
                                                                    int bound, Array<double>& sec, int& pos_sec,
                                                                    int ncond) const
    {
        export_tau_val_domain_boundary(so, mlim, bound, sec, pos_sec, ncond);
    }
} // namespace Kadath
