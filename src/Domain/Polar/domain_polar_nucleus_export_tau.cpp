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
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    void Domain_polar_nucleus::export_tau_val_domain(const Val_domain& so, int mquant, int llim, int order,
                                                     Array<double>& sec, int& pos_sec, int ncond) const
    {

        if (so.check_if_zero())
            pos_sec += ncond;
        else {

            so.coef();
            int rlim = 0;
            switch (order) {
                case 2:
                    rlim = order;
                    break;
                case 1:
                    rlim = 2;
                    break;
                case 0:
                    rlim = 1;
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_polar_nucleus_export_tau_val_domain");
            }

            Index pos_cf(nbr_coefs);
            // Positions of the Galerkin basis
            Index pos_gal_t(nbr_coefs);
            Index pos_gal_r(nbr_coefs);
            Index pos_gal_rt(nbr_coefs);
            double fact_t, fact_r, fact_rt;

            int lquant;
            // Loop on theta
            int baset = (*so.get_base().bases_1d[1])(0);
            for (int j = 0; j < nbr_coefs(1); j++) {
                int baser = (*so.get_base().bases_1d[0])(j);
                pos_cf.set(1) = j;
                // Loop on r :
                for (int i = 0; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    switch (baset) {
                        case COS_EVEN:
                            assert((baser == LEG_EVEN) || (baser == CHEB_EVEN));
                            lquant = 2 * j;
                            // No galerkin :
                            if ((mquant == 0) && (lquant <= llim)) {
                                if (i != nbr_coefs(0) - rlim + 1) {
                                    sec.set(pos_sec) = (*so.cf)(pos_cf);
                                    pos_sec++;
                                }
                            } else if (mquant == 0) {
                                if ((i != 0) && (i != nbr_coefs(0) - rlim + 1)) {
                                    // Galerkin base in r only
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
                                        default:
                                            KADATH_THROW("Strange base in Domain_polar_nucleus:export_tau_inside");
                                    }

                                    sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                                    pos_sec++;
                                }
                            } else if ((j != 0) && (i != 0) && (i != nbr_coefs(0) - rlim + 1)) {
                                // Need to use two_dimensional Galerkin basis (aouch !)
                                pos_gal_r = pos_cf;
                                pos_gal_r.set(0) = 0;
                                pos_gal_t = pos_cf;
                                pos_gal_t.set(1) = 0;
                                pos_gal_rt = pos_cf;
                                pos_gal_rt.set(0) = 0;
                                pos_gal_rt.set(1) = 0;
                                switch (baser) {
                                    case CHEB_EVEN:
                                        fact_r = -2 * pow(-1, i);
                                        fact_t = -2;
                                        fact_rt = 4 * pow(-1, i);
                                        break;
                                    case LEG_EVEN: {
                                        double l0 = 1;
                                        for (int t = 0; t < i; t++)
                                            l0 *= -double(2 * t + 1) / double(2 * t + 2);
                                        fact_r = -l0 * double(4 * i + 1);
                                        fact_t = -2;
                                        fact_rt = 2 * double(4 * i + 1) * l0;
                                    } break;
                                    default:
                                        KADATH_THROW("Strange base in Domain_polar_nucleus:export_tau_inside");
                                }
                                sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r) +
                                                   fact_t * (*so.cf)(pos_gal_t) + fact_rt * (*so.cf)(pos_gal_rt);
                                pos_sec++;
                            }
                            break;
                        case COS_ODD:
                            assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                            lquant = 2 * j + 1;
                            // True coefs ?
                            if ((j != nbr_coefs(1) - 1) && (i != nbr_coefs(0) - rlim + 1) &&
                                (i != nbr_coefs(0) - rlim)) {
                                if ((mquant == 0) && (lquant <= llim + 1)) {
                                    // No Galerkin :
                                    sec.set(pos_sec) = (*so.cf)(pos_cf);
                                    pos_sec++;
                                } else {
                                    if ((mquant == 0) && (i != 0)) {
                                        // Galerkin in r only :
                                        pos_gal_r = pos_cf;
                                        pos_gal_r.set(0) = 0;
                                        switch (baser) {
                                            case CHEB_ODD:
                                                fact_r = -(2 * i + 1) * pow(-1, i);
                                                break;
                                            case LEG_ODD: {
                                                fact_r = -double(4 * i + 3) / 3.;
                                                for (int t = 0; t < i; t++)
                                                    fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                                            } break;
                                            default:
                                                KADATH_THROW("Strange base in Domain_polar_nucleus:export_tau_inside");
                                        }
                                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                                        pos_sec++;
                                    } else if ((j != 0) && (i != 0)) {
                                        // Need to use two_dimensional Galerkin basis (aouch !)
                                        pos_gal_r = pos_cf;
                                        pos_gal_r.set(0) = 0;
                                        pos_gal_t = pos_cf;
                                        pos_gal_t.set(1) = 0;
                                        pos_gal_rt = pos_cf;
                                        pos_gal_rt.set(0) = 0;
                                        pos_gal_rt.set(1) = 0;
                                        switch (baser) {
                                            case CHEB_ODD:
                                                fact_r = -pow(-1, i) * (2 * i + 1);
                                                fact_t = -1.;
                                                fact_rt = pow(-1, i) * (2 * i + 1);
                                                break;
                                            case LEG_ODD: {
                                                double l0 = 1;
                                                for (int t = 0; t < i; t++)
                                                    l0 *= -double(2 * t + 3) / double(2 * t + 2);
                                                fact_r = -l0 * double(4 * i + 3) / 3.;
                                                fact_t = -1.;
                                                fact_rt = l0 * double(4 * i + 3) / 3.;
                                            } break;
                                            default:
                                                KADATH_THROW("Strange base in Domain_polar_nucleus:export_tau_inside");
                                        }
                                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r) +
                                                           fact_t * (*so.cf)(pos_gal_t) +
                                                           fact_rt * (*so.cf)(pos_gal_rt);
                                        pos_sec++;
                                    }
                                }
                            }
                            break;
                        case SIN_EVEN:
                            lquant = 2 * j;
                            if (j != 0) {
                                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                                if ((j != 0) && (j != nbr_coefs(1) - 1) && (i != nbr_coefs(0) - rlim + 1)) {
                                    if ((mquant <= 1) && (lquant <= llim)) {
                                        // No Galerkin
                                        sec.set(pos_sec) = (*so.cf)(pos_cf);
                                        pos_sec++;
                                    } else {
                                        if ((mquant <= 1) && (i != 0)) {
                                            // Galerkin base in r only
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
                                                default:
                                                    KADATH_THROW("Strange base in Domain_polar_nucleus::export_tau_val_domain");
                                            }
                                            sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                                            pos_sec++;
                                        } else {
                                            if ((j != 1) && (i != 0)) {
                                                // Double Galerkin
                                                pos_gal_r = pos_cf;
                                                pos_gal_r.set(0) = 0;
                                                pos_gal_t = pos_cf;
                                                pos_gal_t.set(1) = 1;
                                                pos_gal_rt = pos_cf;
                                                pos_gal_rt.set(0) = 0;
                                                pos_gal_rt.set(1) = 1;
                                                switch (baser) {
                                                    case CHEB_EVEN:
                                                        fact_r = -pow(-1, i);
                                                        fact_t = -j;
                                                        fact_rt = pow(-1, i) * j;
                                                        break;
                                                    case LEG_EVEN: {
                                                        double l0 = 1;
                                                        for (int t = 0; t < i; t++)
                                                            l0 *= -double(2 * t + 1) / double(2 * t + 2);
                                                        fact_r = -l0;
                                                        fact_t = -j;
                                                        fact_rt = l0 * j;
                                                    } break;
                                                    default:
                                                        KADATH_THROW("Strange base in Domain_polar_nucleus::affecte_tau_val_domain");
                                                }
                                                sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r) +
                                                                   fact_t * (*so.cf)(pos_gal_t) +
                                                                   fact_rt * (*so.cf)(pos_gal_rt);
                                                pos_sec++;
                                            }
                                        }
                                    }
                                }
                            }
                            break;
                        case SIN_ODD:
                            lquant = 2 * j + 1;
                            assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                            // True coefs ?
                            if ((j != nbr_coefs(1) - 1) && (i != nbr_coefs(0) - rlim + 1) &&
                                (i != nbr_coefs(0) - rlim)) {
                                if ((mquant <= 1) && (lquant <= llim + 1)) {
                                    // No Galerkin :
                                    sec.set(pos_sec) = (*so.cf)(pos_cf);
                                    pos_sec++;
                                } else {
                                    if ((mquant <= 1) && (i != 0)) {
                                        // Galerkin in r only :
                                        pos_gal_r = pos_cf;
                                        pos_gal_r.set(0) = 0;
                                        switch (baser) {
                                            case CHEB_ODD:
                                                fact_r = -(2 * i + 1) * pow(-1, i);
                                                break;
                                            case LEG_ODD: {
                                                fact_r = -double(4 * i + 3) / 3.;
                                                for (int t = 0; t < i; t++)
                                                    fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                                            } break;
                                            default:
                                                KADATH_THROW("Strange base in Domain_polar_nucleus::export_tau_val_domain");
                                        }
                                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                                        pos_sec++;
                                    } else if ((j != 0) && (i != 0)) {
                                        // Need to use two_dimensional Galerkin basis (aouch !)
                                        pos_gal_r = pos_cf;
                                        pos_gal_r.set(0) = 0;
                                        pos_gal_t = pos_cf;
                                        pos_gal_t.set(1) = 0;
                                        pos_gal_rt = pos_cf;
                                        pos_gal_rt.set(0) = 0;
                                        pos_gal_rt.set(1) = 0;
                                        switch (baser) {
                                            case CHEB_ODD:
                                                fact_r = -pow(-1, i) * (2 * i + 1);
                                                fact_t = -(2 * j + 1);
                                                fact_rt = pow(-1, i) * (2 * i + 1) * (2 * j + 1);
                                                break;
                                            case LEG_ODD: {
                                                double l0 = 1;
                                                for (int t = 0; t < i; t++)
                                                    l0 *= -double(2 * t + 3) / double(2 * t + 2);
                                                fact_r = -l0 * double(4 * i + 3) / 3.;
                                                fact_t = -(2 * j + 1);
                                                fact_rt = l0 * double(4 * i + 3) / 3. * (2 * j + 1);
                                            } break;
                                            default:
                                                KADATH_THROW("Strange base in Domain_polar_nucleus::export_tau_val_domain");
                                        }
                                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r) +
                                                           fact_t * (*so.cf)(pos_gal_t) +
                                                           fact_rt * (*so.cf)(pos_gal_rt);
                                        pos_sec++;
                                    }
                                }
                            }
                            break;
                        default:
                            KADATH_THROW("Unknow theta basis in Domain_polar_nucleus::export_tau_inside");
                    }
                }
                if (order == 1)
                    rlim = 1;
            }
        }
    }

    void Domain_polar_nucleus::export_tau(const Tensor& tt, int dom, int order, Array<double>& res, int& pos_res,
                                          const Array<int>& ncond, int, Array<int>**) const
    {
        int val = tt.get_valence();
        switch (val) {
            case 0:
                if (!tt.is_m_quant_affected())
                    export_tau_val_domain(tt()(dom), 0, 0, order, res, pos_res, ncond(0));
                else
                    export_tau_val_domain(tt()(dom), tt.get_parameters()->get_m_quant(), 0, order, res, pos_res,
                                          ncond(0));
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_polar_nucleus::export_tau" << endl;
                break;
        }
    }
} // namespace Kadath
