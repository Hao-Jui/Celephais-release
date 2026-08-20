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
    void Domain_nucleus::export_tau_val_domain_vr(const Val_domain& so, int order, Array<double>& sec, int& pos_sec,
                                                  int ncond) const
    {

        if (so.check_if_zero())
            pos_sec += ncond;
        else {

            so.coef();
            int ilim = 0;
            switch (order) {
                case 2:
                    ilim = nbr_coefs(0) - 1;
                    break;
                case 0:
                    ilim = nbr_coefs(0);
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_adapted::export_tau_val_domain_vr");
            }

            Index pos_cf(nbr_coefs);
            // Positions of the Galerkin basis
            Index pos_gal_t(nbr_coefs);
            Index pos_gal_r(nbr_coefs);
            Index pos_gal_rt(nbr_coefs);
            double fact_t, fact_r, fact_rt;

            // Cas k = 0
            for (int j = 0; j < nbr_coefs(1); j++) {
                pos_cf.set(1) = j;
                [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
                assert(baset == COS_EVEN);
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));

                if ((j == 0) || (j == 1)) {
                    // No Galerkin
                    for (int i = 0; i < ilim - 1; i++) {
                        pos_cf.set(0) = i;
                        sec.set(pos_sec) = (*so.cf)(pos_cf);
                        pos_sec++;
                    }
                }

                else {
                    for (int i = 1; i < ilim - 1; i++) {
                        // Galerkin in
                        pos_cf.set(0) = i;
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
                                KADATH_THROW("Strange base in Domain_adapted:export_tau_val_domain_vr");
                        }
                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                        pos_sec++;
                    }
                }
            }

            // Next ones
            for (int k = 2; k < nbr_coefs(2) - 1; k++) {
                pos_cf.set(2) = k;
                int mquant = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);

                // Garlerkin in theta
                if (mquant % 2 == 0) {
                    for (int j = 1; j < nbr_coefs(1); j++) {
                        pos_cf.set(1) = j;
                        [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);
                        assert(baset == COS_EVEN);
                        [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                        assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                        for (int i = 1; i < ilim - 1; i++) {
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
                                    fact_t = -2.;
                                    fact_rt = 2 * pow(-1, i) * (2 * i + 1);
                                    break;
                                case LEG_ODD: {
                                    double l0 = 1;
                                    for (int t = 0; t < i; t++)
                                        l0 *= -double(2 * t + 3) / double(2 * t + 2);
                                    fact_r = -l0 * double(4 * i + 3) / 3.;
                                    fact_t = -2.;
                                    fact_rt = 2 * l0 * double(4 * i + 3) / 3.;
                                } break;
                                default:
                                    KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vr");
                            }
                            sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r) +
                                               fact_t * (*so.cf)(pos_gal_t) + fact_rt * (*so.cf)(pos_gal_rt);
                            pos_sec++;
                        }
                    }
                }

                // Galerkin in r
                if (mquant % 2 == 1) {
                    for (int j = 0; j < nbr_coefs(1) - 1; j++) {
                        pos_cf.set(1) = j;
                        [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);
                        assert(baset == SIN_ODD);
                        [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                        assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                        for (int i = 1; i < ilim; i++) {
                            pos_cf.set(0) = i;
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
                                    KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vr");
                            }
                            sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                            pos_sec++;
                        }
                    }
                }
            }
        }
    }

    void Domain_nucleus::export_tau_val_domain_vt(const Val_domain& so, int order, Array<double>& sec, int& pos_sec,
                                                  int ncond) const
    {

        if (so.check_if_zero())
            pos_sec += ncond;
        else {
            so.coef();
            int ilim = 0;
            switch (order) {
                case 2:
                    ilim = nbr_coefs(0) - 1;
                    break;
                case 0:
                    ilim = nbr_coefs(0);
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_nucleus::export_tau_val_domain_vt");
            }

            Index pos_cf(nbr_coefs);
            // Positions of the Galerkin basis
            Index pos_gal_t(nbr_coefs);
            Index pos_gal_r(nbr_coefs);
            Index pos_gal_rt(nbr_coefs);
            double fact_t, fact_r, fact_rt;

            // Cas k = 0
            for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                pos_cf.set(1) = j;
                [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
                assert(baset == SIN_EVEN);
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));

                for (int i = 1; i < ilim - 1; i++) {
                    // Galerkin in r
                    pos_cf.set(0) = i;
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
                            KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vt");
                    }
                    sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                    pos_sec++;
                }
            }

            // k=2 and j=0 STANDARD
            if (nbr_coefs(2) - 1 > 2) {
                pos_cf.set(1) = 0;
                pos_cf.set(2) = 2;
                [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
                assert(baset == COS_ODD);
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 2);
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 0; i < ilim; i++) {
                    pos_cf.set(0) = i;
                    sec.set(pos_sec) = (*so.cf)(pos_cf);
                    pos_sec++;
                }
            }

            // k=2 and j!=0 Garlerkin in r
            if (nbr_coefs(2) - 1 > 2)
                for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                    pos_cf.set(1) = j;
                    [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
                    assert(baset == COS_ODD);
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 2);
                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                    for (int i = 1; i < ilim; i++) {
                        pos_cf.set(0) = i;
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
                                KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vt");
                        }
                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                        pos_sec++;
                    }
                }

            // k=3 and j=0 STANDARD
            if (nbr_coefs(2) - 1 > 3) {
                pos_cf.set(1) = 0;
                pos_cf.set(2) = 3;
                [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
                assert(baset == COS_ODD);
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 3);
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 0; i < ilim; i++) {
                    pos_cf.set(0) = i;
                    sec.set(pos_sec) = (*so.cf)(pos_cf);
                    pos_sec++;
                }
            }

            // k=3 and j!=0 Garlerkin in r
            if (nbr_coefs(2) - 1 > 3)
                for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                    pos_cf.set(1) = j;
                    [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
                    assert(baset == COS_ODD);
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 3);
                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                    for (int i = 1; i < ilim; i++) {
                        pos_cf.set(0) = i;
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
                                KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vt");
                        }
                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                        pos_sec++;
                    }
                }

            // Next ones
            for (int k = 4; k < nbr_coefs(2) - 1; k++) {
                pos_cf.set(2) = k;
                int mquant = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);

                // Galerkin in r
                if (mquant % 2 == 0) {
                    for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                        pos_cf.set(1) = j;
                        [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);
                        assert(baset == SIN_EVEN);
                        [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                        assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                        for (int i = 1; i < ilim - 1; i++) {
                            // Galerkin in
                            pos_cf.set(0) = i;
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
                                    KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vt");
                            }
                            sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                            pos_sec++;
                        }
                    }
                }

                // Double Galerkin
                if (mquant % 2 == 1) {
                    for (int j = 1; j < nbr_coefs(1) - 1; j++) {
                        pos_cf.set(1) = j;
                        [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);
                        assert(baset == COS_ODD);
                        [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                        assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                        for (int i = 1; i < ilim; i++) {
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
                                    fact_r = -2 * pow(-1, i);
                                    fact_t = -1;
                                    fact_rt = 2 * pow(-1, i);
                                    break;
                                case LEG_EVEN: {
                                    double l0 = 1;
                                    for (int t = 0; t < i; t++)
                                        l0 *= -double(2 * t + 1) / double(2 * t + 2);
                                    fact_r = -l0 * double(4 * i + 1);
                                    fact_t = -1;
                                    fact_rt = double(4 * i + 1) * l0;
                                } break;
                                default:
                                    KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vt");
                            }
                            sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r) +
                                               fact_t * (*so.cf)(pos_gal_t) + fact_rt * (*so.cf)(pos_gal_rt);
                            pos_sec++;
                        }
                    }
                }
            }
        }
    }

    void Domain_nucleus::export_tau_val_domain_vp(const Val_domain& so, int order, Array<double>& sec, int& pos_sec,
                                                  int ncond) const
    {

        if (so.check_if_zero())
            pos_sec += ncond;
        else {

            so.coef();
            int ilim = 0;
            switch (order) {
                case 2:
                    ilim = nbr_coefs(0) - 1;
                    break;
                case 0:
                    ilim = nbr_coefs(0);
                    break;
                default:
                    KADATH_THROW("Unknown case in Domain_nucleus::export_tau_val_domain_vp");
            }

            Index pos_cf(nbr_coefs);
            // Positions of the Galerkin basis
            Index pos_gal_t(nbr_coefs);
            Index pos_gal_r(nbr_coefs);
            Index pos_gal_rt(nbr_coefs);
            double fact_t, fact_r, fact_rt;

            // Cas k = 0
            for (int j = 0; j < nbr_coefs(1) - 1; j++) {
                pos_cf.set(1) = j;
                [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
                assert(baset == SIN_ODD);
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                assert((baser == CHEB_ODD) || (baser == LEG_ODD));

                if (j == 0) {
                    // No Galerkin
                    for (int i = 0; i < ilim - 1; i++) {
                        pos_cf.set(0) = i;
                        sec.set(pos_sec) = (*so.cf)(pos_cf);
                        pos_sec++;
                    }
                }

                else {
                    for (int i = 1; i < ilim - 1; i++) {
                        // Galerkin in
                        pos_cf.set(0) = i;
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
                                KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vt");
                        }
                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                        pos_sec++;
                    }
                }
            }

            // k=2 and j=0 STANDARD
            if (nbr_coefs(2) - 1 > 2) {
                pos_cf.set(1) = 0;
                pos_cf.set(2) = 2;
                [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
                assert(baset == COS_EVEN);
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 2);
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 0; i < ilim; i++) {
                    pos_cf.set(0) = i;
                    sec.set(pos_sec) = (*so.cf)(pos_cf);
                    pos_sec++;
                }
            }

            // k=2 and j!=0 Garlerkin in r
            if (nbr_coefs(2) - 1 > 2)
                for (int j = 1; j < nbr_coefs(1); j++) {
                    pos_cf.set(1) = j;
                    [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(2);
                    assert(baset == COS_EVEN);
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 2);
                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                    for (int i = 1; i < ilim; i++) {
                        pos_cf.set(0) = i;
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
                                KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vp");
                        }
                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                        pos_sec++;
                    }
                }

            // k=3 and j=0 STANDARD
            if (nbr_coefs(2) - 1 > 3) {
                pos_cf.set(1) = 0;
                pos_cf.set(2) = 3;
                [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
                assert(baset == COS_EVEN);
                [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(0, 3);
                assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                for (int i = 0; i < ilim; i++) {
                    pos_cf.set(0) = i;
                    sec.set(pos_sec) = (*so.cf)(pos_cf);
                    pos_sec++;
                }
            }

            // k=3 and j!=0 Garlerkin in r
            if (nbr_coefs(2) - 1 > 3)
                for (int j = 1; j < nbr_coefs(1); j++) {
                    pos_cf.set(1) = j;
                    [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(3);
                    assert(baset == COS_EVEN);
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 3);
                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                    for (int i = 1; i < ilim; i++) {
                        pos_cf.set(0) = i;
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
                                KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vp");
                        }
                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                        pos_sec++;
                    }
                }

            // Cas k = 4
            if (nbr_coefs(2) - 1 > 4) {
                pos_cf.set(2) = 4;
                for (int j = 0; j < nbr_coefs(1) - 1; j++) {
                    pos_cf.set(1) = j;
                    [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
                    assert(baset == SIN_ODD);
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                    assert((baser == CHEB_ODD) || (baser == LEG_ODD));

                    if (j == 0) {
                        // No Galerkin
                        for (int i = 0; i < ilim - 1; i++) {
                            pos_cf.set(0) = i;
                            sec.set(pos_sec) = (*so.cf)(pos_cf);
                            pos_sec++;
                        }
                    }

                    else {
                        for (int i = 1; i < ilim - 1; i++) {
                            // Galerkin in
                            pos_cf.set(0) = i;
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
                                    KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vt");
                            }
                            sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                            pos_sec++;
                        }
                    }
                }
            }

            // Cas k = 5
            if (nbr_coefs(2) - 1 > 5) {
                pos_cf.set(2) = 5;
                for (int j = 0; j < nbr_coefs(1) - 1; j++) {
                    pos_cf.set(1) = j;
                    [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(0);
                    assert(baset == SIN_ODD);
                    [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, 0);
                    assert((baser == CHEB_ODD) || (baser == LEG_ODD));

                    if (j == 0) {
                        // No Galerkin
                        for (int i = 0; i < ilim - 1; i++) {
                            pos_cf.set(0) = i;
                            sec.set(pos_sec) = (*so.cf)(pos_cf);
                            pos_sec++;
                        }
                    }

                    else {
                        for (int i = 1; i < ilim - 1; i++) {
                            // Galerkin in
                            pos_cf.set(0) = i;
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
                                    KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vt");
                            }
                            sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                            pos_sec++;
                        }
                    }
                }
            }

            // Next ones
            for (int k = 6; k < nbr_coefs(2) - 1; k++) {
                pos_cf.set(2) = k;
                int mquant = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);

                // STD
                if (mquant % 2 == 0) {
                    for (int j = 0; j < nbr_coefs(1) - 1; j++) {
                        pos_cf.set(1) = j;
                        [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);
                        assert(baset == SIN_ODD);
                        [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                        assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                        for (int i = 1; i < ilim - 1; i++) {
                            // Galerkin in
                            pos_cf.set(0) = i;
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
                                    KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vt");
                            }
                            sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r);
                            pos_sec++;
                        }
                    }
                }

                // Galerkin in r
                if (mquant % 2 == 1) {
                    for (int j = 1; j < nbr_coefs(1); j++) {
                        pos_cf.set(1) = j;
                        [[maybe_unused]] int baset = (*so.get_base().bases_1d[1])(k);
                        assert(baset == COS_EVEN);
                        [[maybe_unused]] int baser = (*so.get_base().bases_1d[0])(j, k);
                        assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                        for (int i = 1; i < ilim; i++) {
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
                                    KADATH_THROW("Strange base in Domain_nucleus:export_tau_val_domain_vp");
                            }
                            sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r) +
                                               fact_t * (*so.cf)(pos_gal_t) + fact_rt * (*so.cf)(pos_gal_rt);
                            pos_sec++;
                        }
                    }
                }
            }
        }
    }

    void Domain_nucleus::export_tau_val_domain(const Val_domain& so, int mlim, int llim, int order, Array<double>& sec,
                                               int& pos_sec, int ncond) const
    {

        assert(mlim >= llim);

        if (so.check_if_zero())
            pos_sec += ncond;
        else {

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
                    KADATH_THROW("Unknown case in Domain_nucleus_export_tau_val_domain");
            }

            so.coef();
            int kmin = 2 * mlim + 2;

            Index pos_cf(nbr_coefs);
            // Positions of the Galerkin basis
            Index pos_gal_t(nbr_coefs);
            Index pos_gal_r(nbr_coefs);
            Index pos_gal_rt(nbr_coefs);
            double fact_t, fact_r, fact_rt;

            int lquant;
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
                                    assert((baser == LEG_EVEN) || (baser == CHEB_EVEN));
                                    lquant = 2 * j;
                                    // No galerkin :
                                    if ((k < kmin) && (lquant <= llim)) {
                                        if (i != nbr_coefs(0) - rlim + 1) {
                                            sec.set(pos_sec) = (*so.cf)(pos_cf);
                                            pos_sec++;
                                        }
                                    } else if (k < kmin) {
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
                                                    KADATH_THROW("Strange base in Domain_nucleus:export_tau_inside");
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
                                                KADATH_THROW("Strange base in Domain_nucleus:export_tau_inside");
                                        }
                                        sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r) +
                                                           fact_t * (*so.cf)(pos_gal_t) +
                                                           fact_rt * (*so.cf)(pos_gal_rt);
                                        pos_sec++;
                                    }
                                    break;
                                case COS_ODD:
                                    assert((baser == CHEB_ODD) || (baser == LEG_ODD));
                                    lquant = 2 * j + 1;
                                    // True coefs ?
                                    if ((j != nbr_coefs(1) - 1) && (i != nbr_coefs(0) - rlim + 1) &&
                                        (i != nbr_coefs(0) - rlim)) {
                                        if ((k < kmin) && (lquant <= llim + 1)) {
                                            // No Galerkin :
                                            sec.set(pos_sec) = (*so.cf)(pos_cf);
                                            pos_sec++;
                                        } else {
                                            if ((k < kmin) && (i != 0)) {
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
                                                        KADATH_THROW("Strange base in Domain_nucleus:export_tau_inside");
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
                                                        KADATH_THROW("Strange base in Domain_nucleus:export_tau_inside");
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
                                    assert((baser == CHEB_EVEN) || (baser == LEG_EVEN));
                                    if ((j != 0) && (j != nbr_coefs(1) - 1) && (i != nbr_coefs(0) - rlim + 1)) {
                                        if ((k < kmin + 2) && (lquant <= llim)) {
                                            // No Galerkin
                                            sec.set(pos_sec) = (*so.cf)(pos_cf);
                                            pos_sec++;
                                        } else {
                                            if ((k < kmin + 2) && (i != 0)) {
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
                                                        KADATH_THROW("Strange base in Domain_nucleus::export_tau_val_domain");
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
                                                            KADATH_THROW("Strange base in " "Domain_nucleus::affecte_tau_val_domain");
                                                    }
                                                    sec.set(pos_sec) = (*so.cf)(pos_cf) + fact_r * (*so.cf)(pos_gal_r) +
                                                                       fact_t * (*so.cf)(pos_gal_t) +
                                                                       fact_rt * (*so.cf)(pos_gal_rt);
                                                    pos_sec++;
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
                                        if ((k < kmin + 2) && (lquant <= llim + 1)) {
                                            // No Galerkin :
                                            sec.set(pos_sec) = (*so.cf)(pos_cf);
                                            pos_sec++;
                                        } else {
                                            if ((k < kmin + 2) && (i != 0)) {
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
                                                        KADATH_THROW("Strange base in Domain_nucleus::export_tau_val_domain");
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
                                                        KADATH_THROW("Strange base in Domain_nucleus::export_tau_val_domain");
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
                                    KADATH_THROW("Unknow theta basis in Domain_nucleus::export_tau_inside");
                            }
                        }
                        if (order == 1)
                            rlim = 1;
                    }
                }
        }
    }

    void Domain_nucleus::export_tau(const Tensor& tt, int dom, int order, Array<double>& res, int& pos_res,
                                    const Array<int>& ncond, int n_cmp, Array<int>** p_cmp) const
    {
        int val = tt.get_valence();
        switch (val) {
            case 0:
                if (!tt.is_m_order_affected())
                    export_tau_val_domain(tt()(dom), 0, 0, order, res, pos_res, ncond(0));
                else
                    export_tau_val_domain(tt()(dom), tt.get_parameters()->get_m_order(), 0, order, res, pos_res,
                                          ncond(0));
                break;
            case 1: {
                bool found = false;
                // Cartesian basis
                if (tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) {
                    if (n_cmp == -1) {
                        export_tau_val_domain(tt(1)(dom), 0, 0, order, res, pos_res, ncond(0));
                        export_tau_val_domain(tt(2)(dom), 0, 0, order, res, pos_res, ncond(1));
                        export_tau_val_domain(tt(3)(dom), 0, 0, order, res, pos_res, ncond(2));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                export_tau_val_domain(tt(1)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if ((*p_cmp[i])(0) == 2)
                                export_tau_val_domain(tt(2)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if ((*p_cmp[i])(0) == 3)
                                export_tau_val_domain(tt(3)(dom), 0, 0, order, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                // Spherical coordinates
                if (tt.get_basis().get_basis(dom) == SPHERICAL_BASIS) {
                    if (n_cmp == -1) {
                        export_tau_val_domain_vr(tt(1)(dom), order, res, pos_res, ncond(0));
                        export_tau_val_domain_vt(tt(2)(dom), order, res, pos_res, ncond(1));
                        export_tau_val_domain_vp(tt(3)(dom), order, res, pos_res, ncond(2));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if ((*p_cmp[i])(0) == 1)
                                export_tau_val_domain_vr(tt(1)(dom), order, res, pos_res, ncond(i));
                            if ((*p_cmp[i])(0) == 2)
                                export_tau_val_domain_vt(tt(2)(dom), order, res, pos_res, ncond(i));
                            if ((*p_cmp[i])(0) == 3)
                                export_tau_val_domain_vp(tt(3)(dom), order, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of vector Domain_nucleus::export_tau");
                }
            } break;
            case 2: {
                bool found = false;
                // Cartesian basis and symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 6)) {
                    if (n_cmp == -1) {
                        export_tau_val_domain(tt(1, 1)(dom), 0, 0, order, res, pos_res, ncond(0));
                        export_tau_val_domain(tt(1, 2)(dom), 0, 0, order, res, pos_res, ncond(1));
                        export_tau_val_domain(tt(1, 3)(dom), 0, 0, order, res, pos_res, ncond(2));
                        export_tau_val_domain(tt(2, 2)(dom), 0, 0, order, res, pos_res, ncond(3));
                        export_tau_val_domain(tt(2, 3)(dom), 0, 0, order, res, pos_res, ncond(4));
                        export_tau_val_domain(tt(3, 3)(dom), 0, 0, order, res, pos_res, ncond(5));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(1, 1)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(1, 2)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(1, 3)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(2, 2)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(2, 3)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(3, 3)(dom), 0, 0, order, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                // Cartesian basis and not symetric
                if ((tt.get_basis().get_basis(dom) == CARTESIAN_BASIS) && (tt.get_n_comp() == 9)) {
                    if (n_cmp == -1) {
                        export_tau_val_domain(tt(1, 1)(dom), 0, 0, order, res, pos_res, ncond(0));
                        export_tau_val_domain(tt(1, 2)(dom), 0, 0, order, res, pos_res, ncond(1));
                        export_tau_val_domain(tt(1, 3)(dom), 0, 0, order, res, pos_res, ncond(2));
                        export_tau_val_domain(tt(2, 1)(dom), 0, 0, order, res, pos_res, ncond(3));
                        export_tau_val_domain(tt(2, 2)(dom), 0, 0, order, res, pos_res, ncond(4));
                        export_tau_val_domain(tt(2, 3)(dom), 0, 0, order, res, pos_res, ncond(5));
                        export_tau_val_domain(tt(3, 1)(dom), 0, 0, order, res, pos_res, ncond(6));
                        export_tau_val_domain(tt(3, 2)(dom), 0, 0, order, res, pos_res, ncond(7));
                        export_tau_val_domain(tt(3, 3)(dom), 0, 0, order, res, pos_res, ncond(8));
                    } else
                        for (int i = 0; i < n_cmp; i++) {
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(1, 1)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(1, 2)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 1) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(1, 3)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(2, 1)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(2, 2)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 2) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(2, 3)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 1))
                                export_tau_val_domain(tt(3, 1)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 2))
                                export_tau_val_domain(tt(3, 2)(dom), 0, 0, order, res, pos_res, ncond(i));
                            if (((*p_cmp[i])(0) == 3) && ((*p_cmp[i])(1) == 3))
                                export_tau_val_domain(tt(3, 3)(dom), 0, 0, order, res, pos_res, ncond(i));
                        }
                    found = true;
                }
                if (!found) {
                    KADATH_THROW("Unknown type of 2-tensor Domain_nucleus::export_tau");
                }
            } break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_nucleus::export_tau" << endl;
                break;
        }
    }
} // namespace Kadath
