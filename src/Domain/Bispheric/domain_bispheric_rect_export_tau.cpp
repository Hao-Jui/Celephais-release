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
    void Domain_bispheric_rect::export_tau_val_domain(const Val_domain& so, int order, Array<double>& sec, int& pos_sec,
                                                      int ncond) const
    {

        if (so.check_if_zero())
            pos_sec += ncond;
        else {
            int forgot_chi = 0;
            int forgot_eta = 0;
            switch (order) {
                case 0:
                    forgot_chi = 0;
                    forgot_eta = 0;
                    break;
                case 1:
                    forgot_chi = 1;
                    forgot_eta = 1;
                    break;
                case 2:
                    forgot_chi = 1;
                    forgot_eta = 2;
                    break;
                default:
                    cerr << "Unknown order in Domain_bispheric_rect::export_tau_val_domain" << endl;
                    break;
            }
            so.coef();
            Index pos_cf(nbr_coefs);
            Index pos_galerkin(nbr_coefs);
            int basep = (*so.get_base().bases_1d[2])(0);

            // Loop on phi :
            for (int k = 0; k < nbr_coefs(2); k++) {
                pos_cf.set(2) = k;
                // Loop on chi :
                for (int j = 0; j < nbr_coefs(1) - forgot_chi; j++) {
                    pos_cf.set(1) = j;
                    // Loop on eta ;
                    for (int i = 0; i < nbr_coefs(0) - forgot_eta; i++) {
                        pos_cf.set(0) = i;
                        switch (basep) {
                            case COS:
                                // Avoid last odd ones
                                if ((k % 2 != 1) || (j != nbr_coefs(1) - 1 - forgot_chi)) {
                                    if ((k == 0) || (k % 2 == 1)) {
                                        // The ones without regularity issues
                                        sec.set(pos_sec) = (*so.cf)(pos_cf);
                                        pos_sec++;
                                    } else if (j != 0) {
                                        // Regularity on the axis thanks to Galerkin
                                        // Factor fo Galerkin different for Legendre or Chebyshev:
                                        double factor_galerkin;
                                        switch (type_base) {
                                            case CHEB_TYPE:
                                                factor_galerkin = (j % 2 == 1) ? -2. : 2.;
                                                break;
                                            case LEG_TYPE:
                                                factor_galerkin = -double(4 * j + 1);
                                                for (int jj = 1; jj <= j; jj++)
                                                    factor_galerkin *= -double(2 * jj - 1) / double(2 * jj);
                                                break;
                                            default:
                                                KADATH_THROW("Unknown type of basis in Domain_bispheric_rect::export_tau_val_domain");
                                        }
                                        pos_galerkin = pos_cf;
                                        pos_galerkin.set(1) = 0;
                                        sec.set(pos_sec) = (*so.cf)(pos_cf) + factor_galerkin * (*so.cf)(pos_galerkin);
                                        pos_sec++;
                                    }
                                }
                                break;
                            case SIN:
                                // Avoid sin(0)
                                if ((k != 0) && (k != nbr_coefs(2) - 1))
                                    // Avoid last odd ones
                                    if ((k % 2 != 1) || (j != nbr_coefs(1) - 1 - forgot_chi)) {
                                        if (k % 2 == 1) {
                                            // The ones without regularity issues
                                            sec.set(pos_sec) = (*so.cf)(pos_cf);
                                            pos_sec++;
                                        } else if (j != 0) {
                                            // Regularity on the axis thanks to Galerkin
                                            // Factor fo Galerkin different for Legendre or Chebyshev:
                                            double factor_galerkin;
                                            switch (type_base) {
                                                case CHEB_TYPE:
                                                    factor_galerkin = (j % 2 == 1) ? -2. : 2.;
                                                    break;
                                                case LEG_TYPE:
                                                    factor_galerkin = -double(4 * j + 1);
                                                    for (int jj = 1; jj <= j; jj++)
                                                        factor_galerkin *= -double(2 * jj - 1) / double(2 * jj);
                                                    break;
                                                default:
                                                    KADATH_THROW("Unknown type of basis in Domain_bispheric_rect::export_tau_val_domain");
                                            }
                                            pos_galerkin = pos_cf;
                                            pos_galerkin.set(1) = 0;
                                            sec.set(pos_sec) =
                                                (*so.cf)(pos_cf) + factor_galerkin * (*so.cf)(pos_galerkin);
                                            pos_sec++;
                                        }
                                    }
                                break;
                            default:
                                cerr << "Unknown base in Domain_bispheric_rect::export_tau_val_domain" << endl;
                                break;
                        }
                    }
                }
            }
        }
    }

    void Domain_bispheric_rect::export_tau(const Tensor& tt, int dom, int order, Array<double>& res, int& pos_res,
                                           const Array<int>& ncond, int n_cmp, Array<int>** p_cmp) const
    {
        int val = tt.get_valence();
        switch (val) {
            case 0:
                export_tau_val_domain(tt()(dom), order, res, pos_res, ncond(0));

                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    if (n_cmp == -1) {
                        export_tau_val_domain(tt(1)(dom), order, res, pos_res, ncond(0));
                        export_tau_val_domain(tt(2)(dom), order, res, pos_res, ncond(1));
                        export_tau_val_domain(tt(3)(dom), order, res, pos_res, ncond(2));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                export_tau_val_domain(tt(1)(dom), order, res, pos_res, ncond(i));
                            if ((*p_cmp[i])(0) == 2)
                                export_tau_val_domain(tt(2)(dom), order, res, pos_res, ncond(i));
                            if ((*p_cmp[i])(0) == 3)
                                export_tau_val_domain(tt(3)(dom), order, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_bispheric_rect::export_tau");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        export_tau_val_domain(tt(1, 1)(dom), order, res, pos_res, ncond(0));
                        export_tau_val_domain(tt(1, 2)(dom), order, res, pos_res, ncond(1));
                        export_tau_val_domain(tt(1, 3)(dom), order, res, pos_res, ncond(2));
                        export_tau_val_domain(tt(2, 2)(dom), order, res, pos_res, ncond(3));
                        export_tau_val_domain(tt(2, 3)(dom), order, res, pos_res, ncond(4));
                        export_tau_val_domain(tt(3, 3)(dom), order, res, pos_res, ncond(5));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(1, 1)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(1, 2)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(1, 3)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(2, 2)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(2, 3)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(3, 3)(dom), order, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        export_tau_val_domain(tt(1, 1)(dom), order, res, pos_res, ncond(0));
                        export_tau_val_domain(tt(1, 2)(dom), order, res, pos_res, ncond(1));
                        export_tau_val_domain(tt(1, 3)(dom), order, res, pos_res, ncond(2));
                        export_tau_val_domain(tt(2, 1)(dom), order, res, pos_res, ncond(3));
                        export_tau_val_domain(tt(2, 2)(dom), order, res, pos_res, ncond(4));
                        export_tau_val_domain(tt(2, 3)(dom), order, res, pos_res, ncond(5));
                        export_tau_val_domain(tt(3, 1)(dom), order, res, pos_res, ncond(6));
                        export_tau_val_domain(tt(3, 2)(dom), order, res, pos_res, ncond(7));
                        export_tau_val_domain(tt(3, 3)(dom), order, res, pos_res, ncond(8));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(1, 1)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(1, 2)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(1, 3)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(2, 1)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(2, 2)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(2, 3)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(3, 1)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(3, 2)(dom), order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(3, 3)(dom), order, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_bispheric_rect::export_tau");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_bispheric_rect::export_tau" << endl;
                break;
        }
    }
} // namespace Kadath
