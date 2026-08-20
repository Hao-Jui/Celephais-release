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

#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    void Domain_nucleus::affecte_tau_val_domain_vr(Val_domain& so, const Array<double>& values, int& conte) const
    {

        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        // Positions of the Galerkin basis
        Index pos_gal_t(nbr_coefs);
        Index pos_gal_r(nbr_coefs);
        Index pos_gal_rt(nbr_coefs);
        double fact_t, fact_r, fact_rt;

        // Case k=0 ; j<=1
        {
            pos_cf.set(2) = 0;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
            assert(baset == COS_EVEN);
            for (int j = 0; j < 2; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    so.cf->set(pos_cf) += values(conte);
                    conte++;
                }
            }
        }

        // Case k=0 ; j!=0
        {
            pos_cf.set(2) = 0;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
            assert(baset == COS_EVEN);
            for (int j = 2; j < nbr_coefs(1); j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_ODD:
                            fact_r = -(2 * i + 1) * pow(-1, i);
                            break;
                        case LEG_ODD: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_cart");
                    }
                    so.cf->set(pos_cf) += values(conte);
                    so.cf->set(pos_gal_r) += fact_r * values(conte);
                    conte++;
                }
            }
        }

        // Next ones
        for (int k = 2; k < nbr_coefs(2) - 1; k++) {

            int mquant = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
            pos_cf.set(2) = k;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);

            if (mquant % 2 == 0) {
                assert(baset = COS_EVEN);
                for (int j = 1; j < nbr_coefs(1); j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                    for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                        pos_cf.set(0) = i;
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
                                fact_r = -l0;
                                fact_t = -1.;
                                fact_rt = l0;
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_cart");
                        }

                        so.cf->set(pos_cf) += values(conte);
                        so.cf->set(pos_gal_r) += fact_r * values(conte);
                        so.cf->set(pos_gal_t) += fact_t * values(conte);
                        so.cf->set(pos_gal_rt) += fact_rt * values(conte);
                        conte++;
                    }
                }
            }

            if (mquant % 2 == 1) {
                assert(baset = SIN_ODD);
                for (int j = 0; j < nbr_coefs(1) - 1; j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                    for (int i = 1; i < nbr_coefs(0); i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        switch (baser) {
                            case CHEB_EVEN:
                                fact_r = -pow(-1, i);
                                break;
                            case LEG_EVEN: {
                                fact_r = -1.;
                                for (int t = 0; t < i; t++)
                                    fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vr");
                        }

                        so.cf->set(pos_cf) += values(conte);
                        so.cf->set(pos_gal_r) += fact_r * values(conte);
                        conte++;
                    }
                }
            }
        }
    }

    void Domain_nucleus::affecte_tau_val_domain_vt(Val_domain& so, const Array<double>& values, int& conte) const
    {

        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        // Positions of the Galerkin basis
        Index pos_gal_t(nbr_coefs);
        Index pos_gal_r(nbr_coefs);
        Index pos_gal_rt(nbr_coefs);
        double fact_t, fact_r, fact_rt;

        // Case k=0 ; j>1
        {
            pos_cf.set(2) = 0;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
            assert(baset == SIN_EVEN);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_ODD:
                            fact_r = -(2 * i + 1) * pow(-1, i);
                            break;
                        case LEG_ODD: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vt");
                    }
                    so.cf->set(pos_cf) += values(conte);
                    so.cf->set(pos_gal_r) += fact_r * values(conte);
                    conte++;
                }
            }
        }

        // Case k==2 ; j==0
        if (nbr_coefs(2) - 1 > 2) {
            pos_cf.set(2) = 2;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
            assert(baset == COS_ODD);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 2);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                so.cf->set(pos_cf) += values(conte);
                conte++;
            }
        }

        // Case k==2 ; j!=0
        if (nbr_coefs(2) - 1 > 2) {
            pos_cf.set(2) = 2;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
            assert(baset = COS_ODD);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 2);
                pos_cf.set(1) = j;
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 1; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_EVEN:
                            fact_r = -pow(-1, i);
                            break;
                        case LEG_EVEN: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vt");
                    }

                    so.cf->set(pos_cf) += values(conte);
                    so.cf->set(pos_gal_r) += fact_r * values(conte);
                    conte++;
                }
            }
        }

        // Case k==3 ; j==0
        if (nbr_coefs(2) - 1 > 3) {
            pos_cf.set(2) = 3;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
            assert(baset == COS_ODD);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 3);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                so.cf->set(pos_cf) += values(conte);
                conte++;
            }
        }

        // Case k==3 ; j!=0
        if (nbr_coefs(2) - 1 > 3) {
            pos_cf.set(2) = 3;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
            assert(baset = COS_ODD);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 3);
                pos_cf.set(1) = j;
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 1; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_EVEN:
                            fact_r = -pow(-1, i);
                            break;
                        case LEG_EVEN: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vt");
                    }

                    so.cf->set(pos_cf) += values(conte);
                    so.cf->set(pos_gal_r) += fact_r * values(conte);
                    conte++;
                }
            }
        }

        // Next ones
        for (int k = 4; k < nbr_coefs(2) - 1; k++) {

            int mquant = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
            pos_cf.set(2) = k;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);

            if (mquant % 2 == 0) {
                assert(baset = SIN_EVEN);
                for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                    for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        switch (baser) {
                            case CHEB_ODD:
                                fact_r = -(2 * i + 1) * pow(-1, i);
                                break;
                            case LEG_ODD: {
                                fact_r = -1.;
                                for (int t = 0; t < i; t++)
                                    fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vt");
                        }
                        so.cf->set(pos_cf) += values(conte);
                        so.cf->set(pos_gal_r) += fact_r * values(conte);
                        conte++;
                    }
                }
            }

            if (mquant % 2 == 1) {
                assert(baset = COS_ODD);
                for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                    for (int i = 1; i < nbr_coefs(0); i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        pos_gal_t = pos_cf;
                        pos_gal_t.set(1) = 0;
                        pos_gal_rt = pos_cf;
                        pos_gal_rt.set(0) = 0;
                        pos_gal_rt.set(1) = 0;
                        switch (baser) {
                            case CHEB_EVEN:
                                fact_r = -pow(-1, i);
                                fact_t = -1.;
                                fact_rt = pow(-1, i);
                                break;
                            case LEG_EVEN: {
                                double l0 = 1;
                                for (int t = 0; t < i; t++)
                                    l0 *= -double(2 * t + 1) / double(2 * t + 2);
                                fact_r = -l0;
                                fact_t = -1.;
                                fact_rt = l0;
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vt");
                        }
                        so.cf->set(pos_cf) += values(conte);
                        so.cf->set(pos_gal_r) += fact_r * values(conte);
                        so.cf->set(pos_gal_t) += fact_t * values(conte);
                        so.cf->set(pos_gal_rt) += fact_rt * values(conte);
                        conte++;
                    }
                }
            }
        }
    }

    void Domain_nucleus::affecte_tau_val_domain_vp(Val_domain& so, const Array<double>& values, int& conte) const
    {

        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        // Positions of the Galerkin basis
        Index pos_gal_t(nbr_coefs);
        Index pos_gal_r(nbr_coefs);
        Index pos_gal_rt(nbr_coefs);
        double fact_t, fact_r, fact_rt;

        // k = 0 ; j==0
        {
            pos_cf.set(2) = 0;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
            assert(baset == SIN_ODD);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 0);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_ODD) || (baser == LEG_ODD));
            for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                pos_cf.set(0) = i;
                so.cf->set(pos_cf) += values(conte);
                conte++;
            }
        }

        // Case k==0 ; j!=0
        {
            pos_cf.set(2) = 0;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
            assert(baset = SIN_ODD);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_ODD:
                            fact_r = -(2 * i + 1) * pow(-1, i);
                            break;
                        case LEG_ODD: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                    }
                    so.cf->set(pos_cf) += values(conte);
                    so.cf->set(pos_gal_r) += fact_r * values(conte);
                    conte++;
                }
            }
        }

        // Case k==2 ; j==0
        if (nbr_coefs(2) - 1 > 2) {
            pos_cf.set(2) = 2;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
            assert(baset == COS_EVEN);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 2);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                so.cf->set(pos_cf) += values(conte);
                conte++;
            }
        }

        // Case k==2 ; j!=0
        if (nbr_coefs(2) - 1 > 2) {
            pos_cf.set(2) = 2;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
            assert(baset = COS_EVEN);
            for (int j = 1; j < nbr_coefs(1); j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 2);
                pos_cf.set(1) = j;
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 1; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_EVEN:
                            fact_r = -pow(-1, i);
                            break;
                        case LEG_EVEN: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                    }

                    so.cf->set(pos_cf) += values(conte);
                    so.cf->set(pos_gal_r) += fact_r * values(conte);
                    conte++;
                }
            }
        }

        // Case k==3 ; j==0
        if (nbr_coefs(2) - 1 > 3) {
            pos_cf.set(2) = 3;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
            assert(baset == COS_EVEN);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 3);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
            for (int i = 0; i < nbr_coefs(0); i++) {
                pos_cf.set(0) = i;
                so.cf->set(pos_cf) += values(conte);
                conte++;
            }
        }

        // Case k==3 ; j!=0
        if (nbr_coefs(2) - 1 > 3) {
            pos_cf.set(2) = 3;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
            assert(baset = COS_EVEN);
            for (int j = 1; j < nbr_coefs(1); j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 3);
                pos_cf.set(1) = j;
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 1; i < nbr_coefs(0); i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_EVEN:
                            fact_r = -pow(-1, i);
                            break;
                        case LEG_EVEN: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                    }

                    so.cf->set(pos_cf) += values(conte);
                    so.cf->set(pos_gal_r) += fact_r * values(conte);
                    conte++;
                }
            }
        }
        // k = 4; j==0
        if (nbr_coefs(2) - 1 > 4) {
            pos_cf.set(2) = 4;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(4);
            assert(baset == SIN_ODD);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 4);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_ODD) || (baser == LEG_ODD));
            for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                pos_cf.set(0) = i;
                so.cf->set(pos_cf) += values(conte);
                conte++;
            }
        }

        // Case k==4 ; j!=0
        if (nbr_coefs(2) - 1 > 4) {
            pos_cf.set(2) = 4;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(4);
            assert(baset = SIN_ODD);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 4);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_ODD:
                            fact_r = -(2 * i + 1) * pow(-1, i);
                            break;
                        case LEG_ODD: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                    }
                    so.cf->set(pos_cf) += values(conte);
                    so.cf->set(pos_gal_r) += fact_r * values(conte);
                    conte++;
                }
            }
        }

        // k = 5; j==0
        if (nbr_coefs(2) - 1 > 5) {
            pos_cf.set(2) = 5;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(5);
            assert(baset == SIN_ODD);
            [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 5);
            pos_cf.set(1) = 0;
            assert((baser == CHEB_ODD) || (baser == LEG_ODD));
            for (int i = 0; i < nbr_coefs(0) - 1; i++) {
                pos_cf.set(0) = i;
                so.cf->set(pos_cf) += values(conte);
                conte++;
            }
        }

        // Case k==5 ; j!=0
        if (nbr_coefs(2) - 1 > 5) {
            pos_cf.set(2) = 5;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(5);
            assert(baset = SIN_ODD);
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 5);
                pos_cf.set(1) = j;
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                    pos_cf.set(0) = i;
                    pos_gal_r = pos_cf;
                    pos_gal_r.set(0) = 0;
                    switch (baser) {
                        case CHEB_ODD:
                            fact_r = -(2 * i + 1) * pow(-1, i);
                            break;
                        case LEG_ODD: {
                            fact_r = -1.;
                            for (int t = 0; t < i; t++)
                                fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                        } break;
                        default:
                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                    }
                    so.cf->set(pos_cf) += values(conte);
                    so.cf->set(pos_gal_r) += fact_r * values(conte);
                    conte++;
                }
            }
        }

        // Next ones
        for (int k = 6; k < nbr_coefs(2) - 1; k++) {

            int mquant = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
            pos_cf.set(2) = k;
            [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);

            if (mquant % 2 == 0) {
                assert(baset = SIN_ODD);
                for (int j = 0; j < nbr_coefs(1) - 1; j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                    for (int i = 1; i < nbr_coefs(0) - 1; i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        switch (baser) {
                            case CHEB_ODD:
                                fact_r = -(2 * i + 1) * pow(-1, i);
                                break;
                            case LEG_ODD: {
                                fact_r = -1.;
                                for (int t = 0; t < i; t++)
                                    fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                        }
                        so.cf->set(pos_cf) += values(conte);
                        so.cf->set(pos_gal_r) += fact_r * values(conte);
                        conte++;
                    }
                }
            }

            if (mquant % 2 == 1) {
                assert(baset = COS_EVEN);
                for (int j = 1; j < nbr_coefs(1); j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                    for (int i = 1; i < nbr_coefs(0); i++) {
                        pos_cf.set(0) = i;
                        pos_gal_r = pos_cf;
                        pos_gal_r.set(0) = 0;
                        pos_gal_t = pos_cf;
                        pos_gal_t.set(1) = 0;
                        pos_gal_rt = pos_cf;
                        pos_gal_rt.set(0) = 0;
                        pos_gal_rt.set(1) = 0;
                        switch (baser) {
                            case CHEB_EVEN:
                                fact_r = -pow(-1, i);
                                fact_t = -1.;
                                fact_rt = pow(-1, i);
                                break;
                            case LEG_EVEN: {
                                double l0 = 1;
                                for (int t = 0; t < i; t++)
                                    l0 *= -double(2 * t + 1) / double(2 * t + 2);
                                fact_r = -l0;
                                fact_t = -1.;
                                fact_rt = l0;
                            } break;
                            default:
                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain_vp");
                        }
                        so.cf->set(pos_cf) += values(conte);
                        so.cf->set(pos_gal_r) += fact_r * values(conte);
                        so.cf->set(pos_gal_t) += fact_t * values(conte);
                        so.cf->set(pos_gal_rt) += fact_rt * values(conte);
                        conte++;
                    }
                }
            }
        }
    }

    void Domain_nucleus::affecte_tau_val_domain(Val_domain& so, int mlim, int llim, const Array<double>& values,
                                                int& conte) const
    {

        int kmin = 2 * mlim + 2;
        int lquant;

        so.allocate_coef();
        *so.cf = 0.;
        Index pos_cf(nbr_coefs);

        // Positions of the Galerkin basis
        Index pos_gal_t(nbr_coefs);
        Index pos_gal_r(nbr_coefs);
        Index pos_gal_rt(nbr_coefs);
        double fact_t, fact_r, fact_rt;

        // Loop on phi :
        for (int k = 0; k < nbr_coefs(2) - 1; k++)
            if (k != 1) {
                pos_cf.set(2) = k;
                // Loop on theta
                [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);
                for (int j = 0; j < nbr_coefs(1); j++) {
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                    pos_cf.set(1) = j;
                    // Loop on r :
                    for (int i = 0; i < nbr_coefs(0); i++) {
                        pos_cf.set(0) = i;
                        switch (baset) {
                            case COS_EVEN:
                                lquant = 2 * j;
                                // No galerkin :
                                if ((k < kmin) && (lquant <= llim)) {
                                    so.cf->set(pos_cf) += values(conte);
                                    conte++;
                                } else if (k < kmin) {
                                    if (i != 0) {
                                        // Galerkin base in r only
                                        pos_gal_r = pos_cf;
                                        pos_gal_r.set(0) = 0;
                                        switch (baser) {
                                            case CHEB_EVEN:
                                                fact_r = -pow(-1, i);
                                                break;
                                            case LEG_EVEN: {
                                                fact_r = -1.;
                                                for (int t = 0; t < i; t++)
                                                    fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                                            } break;
                                            default:
                                                KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain");
                                        }

                                        so.cf->set(pos_cf) += values(conte);
                                        so.cf->set(pos_gal_r) += fact_r * values(conte);
                                        conte++;
                                    }
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
                                        case CHEB_EVEN:
                                            fact_r = -pow(-1, i);
                                            fact_t = -1.;
                                            fact_rt = pow(-1, i);
                                            break;
                                        case LEG_EVEN: {
                                            double l0 = 1;
                                            for (int t = 0; t < i; t++)
                                                l0 *= -double(2 * t + 1) / double(2 * t + 2);
                                            fact_r = -l0;
                                            fact_t = -1.;
                                            fact_rt = l0;
                                        } break;
                                        default:
                                            KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain");
                                    }
                                    so.cf->set(pos_cf) += values(conte);
                                    so.cf->set(pos_gal_r) += fact_r * values(conte);
                                    so.cf->set(pos_gal_t) += fact_t * values(conte);
                                    so.cf->set(pos_gal_rt) += fact_rt * values(conte);
                                    conte++;
                                }
                                break;
                            case COS_ODD:
                                lquant = 2 * j + 1;
                                if ((j != nbr_coefs(1) - 1) && (i != nbr_coefs(0) - 1)) {
                                    if ((k < kmin) && (lquant <= llim + 1)) {
                                        so.cf->set(pos_cf) += values(conte);
                                        conte++;
                                    } else {
                                        if ((k < kmin) && (i != 0)) {
                                            pos_gal_r = pos_cf;
                                            pos_gal_r.set(0) = 0;
                                            switch (baser) {
                                                case CHEB_ODD:
                                                    fact_r = -(2 * i + 1) * pow(-1, i);
                                                    break;
                                                case LEG_ODD: {
                                                    fact_r = -1.;
                                                    for (int t = 0; t < i; t++)
                                                        fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                                                } break;
                                                default:
                                                    KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain");
                                            }
                                            so.cf->set(pos_cf) += values(conte);
                                            so.cf->set(pos_gal_r) += fact_r * values(conte);
                                            conte++;
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
                                                    fact_r = -l0;
                                                    fact_t = -1.;
                                                    fact_rt = l0;
                                                } break;
                                                default:
                                                    KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain");
                                            }
                                            so.cf->set(pos_cf) += values(conte);
                                            so.cf->set(pos_gal_r) += fact_r * values(conte);
                                            so.cf->set(pos_gal_t) += fact_t * values(conte);
                                            so.cf->set(pos_gal_rt) += fact_rt * values(conte);
                                            conte++;
                                        }
                                    }
                                }
                                break;
                            case SIN_EVEN:
                                lquant = 2 * j;
                                if ((j != 0) && (j != nbr_coefs(1) - 1)) {
                                    if ((k < kmin + 2) && (lquant <= llim)) {
                                        so.cf->set(pos_cf) += values(conte);
                                        conte++;
                                    } else {
                                        if ((k < kmin + 2) && (i != 0)) {
                                            // Galerkin base in r only
                                            pos_gal_r = pos_cf;
                                            pos_gal_r.set(0) = 0;
                                            switch (baser) {
                                                case CHEB_EVEN:
                                                    fact_r = -pow(-1, i);
                                                    break;
                                                case LEG_EVEN: {
                                                    fact_r = -1.;
                                                    for (int t = 0; t < i; t++)
                                                        fact_r *= -double(2 * t + 1) / double(2 * t + 2);
                                                } break;
                                                default:
                                                    KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain");
                                            }
                                            so.cf->set(pos_cf) += values(conte);
                                            so.cf->set(pos_gal_r) += fact_r * values(conte);
                                            conte++;
                                        }

                                        else // Double Galerkin
                                            if ((j != 1) && (i != 0)) {
                                                // Need to use two_dimensional Galerkin basis (aouch !)
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
                                                        KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain");
                                                }
                                                so.cf->set(pos_cf) += values(conte);
                                                so.cf->set(pos_gal_r) += fact_r * values(conte);
                                                so.cf->set(pos_gal_t) += fact_t * values(conte);
                                                so.cf->set(pos_gal_rt) += fact_rt * values(conte);
                                                conte++;
                                            }
                                    }
                                }
                                break;
                            case SIN_ODD:
                                lquant = 2 * j + 1;
                                if ((j != nbr_coefs(1) - 1) && (i != nbr_coefs(0) - 1)) {
                                    if ((k < kmin + 2) && (lquant <= llim + 1)) {
                                        so.cf->set(pos_cf) += values(conte);
                                        conte++;
                                    } else {
                                        if ((k < kmin + 2) && (i != 0)) {
                                            pos_gal_r = pos_cf;
                                            pos_gal_r.set(0) = 0;
                                            switch (baser) {
                                                case CHEB_ODD:
                                                    fact_r = -(2 * i + 1) * pow(-1, i);
                                                    break;
                                                case LEG_ODD: {
                                                    fact_r = -1.;
                                                    for (int t = 0; t < i; t++)
                                                        fact_r *= -double(2 * t + 3) / double(2 * t + 2);
                                                } break;
                                                default:
                                                    KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain");
                                            }
                                            so.cf->set(pos_cf) += values(conte);
                                            so.cf->set(pos_gal_r) += fact_r * values(conte);
                                            conte++;
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
                                                    fact_r = -l0;
                                                    fact_t = -(2 * j + 1);
                                                    fact_rt = l0 * (2 * j + 1);
                                                } break;
                                                default:
                                                    KADATH_THROW("Strange base in Domain_nucleus::affecte_tau_val_domain");
                                            }
                                            so.cf->set(pos_cf) += values(conte);
                                            so.cf->set(pos_gal_r) += fact_r * values(conte);
                                            so.cf->set(pos_gal_t) += fact_t * values(conte);
                                            so.cf->set(pos_gal_rt) += fact_rt * values(conte);
                                            conte++;
                                        }
                                    }
                                }
                                break;
                            default:
                                KADATH_THROW("Unknow theta basis in Domain_nucleus::affecte_tau_val_domain");
                        }
                    }
                }
            }
    }

    void Domain_nucleus::affecte_tau(Tensor& tt, int dom, const Array<double>& cf, int& pos_cf) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int val = tt.get_valence();
        switch (val) {
            case 0:
                affecte_tau_val_domain(tt.set().set_domain(dom), 0, 0, cf, pos_cf);
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    affecte_tau_val_domain(tt.set(1).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3).set_domain(dom), 0, 0, cf, pos_cf);
                    found = true;
                }
                // Spherical coordinates
                if (tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) {
                    affecte_tau_val_domain_vr(tt.set(1).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain_vt(tt.set(2).set_domain(dom), cf, pos_cf);
                    affecte_tau_val_domain_vp(tt.set(3).set_domain(dom), cf, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_nucleus::affecte_tau");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    affecte_tau_val_domain(tt.set(1, 1).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 2).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 3).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 2).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 3).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 3).set_domain(dom), 0, 0, cf, pos_cf);
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    affecte_tau_val_domain(tt.set(1, 1).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 2).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(1, 3).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 1).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 2).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(2, 3).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 1).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 2).set_domain(dom), 0, 0, cf, pos_cf);
                    affecte_tau_val_domain(tt.set(3, 3).set_domain(dom), 0, 0, cf, pos_cf);
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_nucleus::affecte_tau");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_nucleus::affecte_tau" << endl;
                break;
        }
    }
} // namespace Kadath
