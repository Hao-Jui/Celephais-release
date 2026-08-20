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
#include "For_Kadath/Domain/polar_periodic.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    void Domain_polar_periodic_nucleus::export_tau_val_domain(const Val_domain& so, int llim, int order,
                                                              Array<double>& sec, int& pos_sec, int ncond) const
    {

        if (so.check_if_zero())
            pos_sec += ncond;
        else {

            so.coef();
            int rlim = 0;
            switch (order) {
                case 2:
                    rlim = 1;
                    break;
                case 0:
                    rlim = 0;
                    break;
                default:
                    KADATH_THROW("Unknown order in Domain_polar_periodic_nucleus_export_tau_val_domain");
            }

            Index pos_cf(nbr_coefs);
            // Positions of the Galerkin basis
            Index pos_gal_r(nbr_coefs);
            double fact_r;

            int lquant;
            int basetime = (*so.get_base().bases_1d[2])(0);
            int mink, maxk;
            switch (basetime) {
                case COS:
                    mink = 0;
                    maxk = nbr_coefs(2);
                    break;
                case SIN:
                    mink = 1;
                    maxk = nbr_coefs(2) - 1;
                    break;
                default:
                    KADATH_THROW("Unknown time basis in Domain_polar_periodic_nucleus_export_tau_val_domain");
            }

            for (int k = mink; k < maxk; k++) {
                pos_cf.set(2) = k;
                // Loop on theta
                int baset = (*so.get_base().bases_1d[1])(k);
                int minj, maxj;
                switch (baset) {
                    case COS_EVEN:
                        minj = 0;
                        maxj = nbr_coefs(1);
                        break;
                    case COS_ODD:
                        minj = 0;
                        maxj = nbr_coefs(1) - 1;
                        break;
                    case SIN_EVEN:
                        minj = 1;
                        maxj = nbr_coefs(1) - 1;
                        break;
                    case SIN_ODD:
                        minj = 0;
                        maxj = nbr_coefs(1) - 1;
                        break;

                    default:
                        KADATH_THROW("Unknown theta basis in Domain_polar_periodic_nucleus_export_tau_val_domain");
                }

                for (int j = minj; j < maxj; j++) {
                    int baser = (*so.get_base().bases_1d[0])(j, k);

                    // Compute lquant :
                    switch (baset) {
                        case COS_EVEN:
                            lquant = 2 * j;
                            break;
                        case COS_ODD:
                            lquant = 2 * j + 1;
                            break;
                        case SIN_EVEN:
                            lquant = 2 * j;
                            break;
                        case SIN_ODD:
                            lquant = 2 * j + 1;
                            break;

                        default:
                            KADATH_THROW("Unknown theta basis in Domain_polar_periodic_nucleus_export_tau_val_domain");
                    }

                    pos_cf.set(1) = j;

                    int minr, maxr;
                    switch (baser) {
                        case CHEB_EVEN:
                            minr = (lquant <= llim) ? 0 : 1;
                            maxr = nbr_coefs(0) - rlim;
                            break;
                        case CHEB_ODD:
                            minr = (lquant <= llim) ? 0 : 1;
                            maxr = nbr_coefs(0) - 1 - rlim;
                            break;
                        case LEG_EVEN:
                            minr = (lquant <= llim) ? 0 : 1;
                            maxr = nbr_coefs(0) - rlim;
                            break;
                        case LEG_ODD:
                            minr = (lquant <= llim) ? 0 : 1;
                            maxr = nbr_coefs(0) - 1 - rlim;
                            break;
                        default:
                            KADATH_THROW("Unknown r basis in Domain_polar_periodic_nucleus_export_tau_val_domain");
                    }

                    // Loop on r :
                    for (int i = minr; i < maxr; i++) {
                        pos_cf.set(0) = i;

                        // No garlekin
                        if (lquant <= llim) {
                            sec.set(pos_sec) = (*so.cf)(pos_cf);
                            pos_sec++;
                        } else {
                            // Galerkin en r
                            pos_gal_r = pos_cf;
                            pos_gal_r.set(0) = 0;
                            switch (baser) {
                                case CHEB_EVEN:
                                    fact_r = -2 * pow(-1, i);
                                    break;
                                case LEG_EVEN: {
                                    fact_r = -double(4 * i + 1);
                                    for (int t = 0; t < i; t++)
                                        fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                                } break;
                                case CHEB_ODD:
                                    fact_r = -(2 * i + 1) * pow(-1, i);
                                    break;
                                case LEG_ODD: {
                                    fact_r = -double(4 * i + 3) / 3.;
                                    for (int t = 0; t < i; t++)
                                        fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                                } break;

                                default:
                                    KADATH_THROW("Strange base in Domain_polar_periodic_nucleus:export_tau_inside");
                            }
                            sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                            pos_sec++;
                        }
                    } // end loop i
                } // end loop j
            } // end loop k

        } // end case nul residual
    }

    void Domain_polar_periodic_nucleus::export_tau(const Tensor& tt, int dom, int order, Array<double>& res,
                                                   int& pos_res, const Array<int>& ncond, int n_cmp,
                                                   Array<int>** p_cmp) const
    {
        int val = tt.get_valence();
        switch (val) {
            case 0:
                export_tau_val_domain(tt()(dom), 0, order, res, pos_res, ncond(0));
                break;
            case 1:
                if (n_cmp == -1) {
                    export_tau_val_domain(tt(1)(dom), 2, order, res, pos_res, ncond(0));
                    export_tau_val_domain(tt(2)(dom), 2, order, res, pos_res, ncond(1));
                    export_tau_val_domain(tt(3)(dom), 2, order, res, pos_res, ncond(2));
                } else
                    for (int i = 0; i < n_cmp; i++) {
                        if ((*p_cmp[i])(0) == 1)
                            export_tau_val_domain(tt(1)(dom), 2, order, res, pos_res, ncond(i));
                        if ((*p_cmp[i])(0) == 2)
                            export_tau_val_domain(tt(2)(dom), 2, order, res, pos_res, ncond(i));
                        if ((*p_cmp[i])(0) == 3)
                            export_tau_val_domain(tt(3)(dom), 2, order, res, pos_res, ncond(i));
                    }
                break;
            case 2:
                if (tt.get_n_comp() == 6) {
                    if (n_cmp == -1) {
                        export_tau_val_domain(tt(1, 1)(dom), 2, order, res, pos_res, ncond(0));
                        export_tau_val_domain(tt(1, 2)(dom), 2, order, res, pos_res, ncond(1));
                        export_tau_val_domain(tt(1, 3)(dom), 2, order, res, pos_res, ncond(2));
                        export_tau_val_domain(tt(2, 2)(dom), 2, order, res, pos_res, ncond(3));
                        export_tau_val_domain(tt(2, 3)(dom), 2, order, res, pos_res, ncond(4));
                        export_tau_val_domain(tt(3, 3)(dom), 2, order, res, pos_res, ncond(5));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(1, 1)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(1, 2)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(1, 3)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(2, 2)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(2, 3)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(3, 3)(dom), 2, order, res, pos_res, ncond(i));
                        }
                }

                if (tt.get_n_comp() == 9) {
                    if (n_cmp == -1) {
                        export_tau_val_domain(tt(1, 1)(dom), 2, order, res, pos_res, ncond(0));
                        export_tau_val_domain(tt(1, 2)(dom), 2, order, res, pos_res, ncond(1));
                        export_tau_val_domain(tt(1, 3)(dom), 2, order, res, pos_res, ncond(2));
                        export_tau_val_domain(tt(2, 1)(dom), 2, order, res, pos_res, ncond(3));
                        export_tau_val_domain(tt(2, 2)(dom), 2, order, res, pos_res, ncond(4));
                        export_tau_val_domain(tt(2, 3)(dom), 2, order, res, pos_res, ncond(5));
                        export_tau_val_domain(tt(3, 1)(dom), 2, order, res, pos_res, ncond(6));
                        export_tau_val_domain(tt(3, 2)(dom), 2, order, res, pos_res, ncond(7));
                        export_tau_val_domain(tt(3, 3)(dom), 2, order, res, pos_res, ncond(8));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(1, 1)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(1, 2)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(1, 3)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(2, 1)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(2, 2)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(2, 3)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(3, 1)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(3, 2)(dom), 2, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(3, 3)(dom), 2, order, res, pos_res, ncond(i));
                        }
                }
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_polar_periodicnucleus::export_tau" << endl;
                break;
        }
    }
} // namespace Kadath
