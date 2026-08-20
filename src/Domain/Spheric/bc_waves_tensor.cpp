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
 *   2026-08-06  RAII/span modernization.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include <memory>
#include <vector>
namespace Kadath
{
    Array<double> mat_inv_leg_even(int, int);
    Array<double> mat_inv_leg_odd(int, int);
    Array<double> mat_leg_even(int, int);
    Array<double> mat_leg_odd(int, int);

    Tensor Domain_shell::bc_waves(int dom, const Tensor& gamma, const double omega, bool toinf) const
    {

        // Free parameters for checking
        // int kkmax = nbr_coefs(2)-1 ;
        // int jjmax = nbr_coefs(1) ;
        int kkmax = 10;
        int jjmax = 5;

        // Check if gamma is a metric
        int valence = gamma.get_valence();
        if (valence != 2) {
            KADATH_THROW("Domain_shell::bc_waves_tensor only defined with respect to second order tensor (i.e. valence must be 2)");
        }
        int ncomp = gamma.get_n_comp();
        if (ncomp != 6) {
            KADATH_THROW("Domain_shell::bc_waves_tensor only defined with respect to a symmetric tensor");
        }
        if (gamma.get_basis().get_basis(dom) != CARTESIAN_BASIS) {
            KADATH_THROW("Domain_shell::bc_waves_tensor only defined with respect to Cartesian tensorial basis");
        }

        // Put rmax in a Term_eq
        Val_domain rr = get_radius();
        rr.std_base();

        // Compute the term_eq for the BC
        int nbrm = int(nbr_coefs(2) / 2) + 2;
        int nbrl = 2 * nbr_coefs(1) - 1;
        // Real parts and Imaginar parts
        std::vector<std::unique_ptr<Val_domain>> Rsh(nbrm * nbrl);
        std::vector<std::unique_ptr<Val_domain>> Ish(nbrm * nbrl);
        std::vector<std::unique_ptr<Val_domain>> dRsh(nbrm * nbrl);
        std::vector<std::unique_ptr<Val_domain>> dIsh(nbrm * nbrl);

        for (int m = 0; m < nbrm; m++)
            for (int l = 0; l < nbrl; l++) {
                Rsh[m * nbrl + l] =
                    (m == 0) ? std::make_unique<Val_domain>(pow(1. / get_radius(), (l + 1)))
                             : std::make_unique<Val_domain>(bessel_jl(m * omega * get_radius(), l));
                Ish[m * nbrl + l] =
                    (m == 0) ? std::make_unique<Val_domain>(pow(1. / get_radius(), (l + 1)))
                             : std::make_unique<Val_domain>(bessel_yl(m * omega * get_radius(), l));
                dRsh[m * nbrl + l] =
                    (m == 0) ? std::make_unique<Val_domain>(-(l + 1) * pow(1. / get_radius(), (l + 2)))
                             : std::make_unique<Val_domain>(m * omega * bessel_djl(m * omega * get_radius(), l));
                dIsh[m * nbrl + l] =
                    (m == 0) ? std::make_unique<Val_domain>(-(l + 1) * pow(1. / get_radius(), (l + 2)))
                             : std::make_unique<Val_domain>(m * omega * bessel_dyl(m * omega * get_radius(), l));
            }

        // To store the result
        Tensor res(gamma, false);
        res.std_base();
        for (int i = 0; i < res.get_n_comp(); i++) {
            res.set(res.indices(i)).set_domain(dom).allocate_coef();
            Index pcf(nbr_coefs);
            do {
                res.set(res.indices(i)).set_domain(dom).set_coef(pcf) = 0;
            } while (pcf.inc());
        }

        // Spherical harmonics -> Standard basis
        // Passage matrices :
        Array<double> inv_even(mat_inv_leg_even(nbr_coefs(1), nbr_coefs(2)));
        Array<double> inv_odd(mat_inv_leg_odd(nbr_coefs(1), nbr_coefs(2)));
        Array<double> passage_even(mat_leg_even(nbr_coefs(1), nbr_coefs(2)));
        Array<double> passage_odd(mat_leg_odd(nbr_coefs(1), nbr_coefs(2)));

        // Symetric part :
        // Loop on phi :
        Index posylm(nbr_coefs);
        Index posradial(nbr_coefs);
        Index ppp(nbr_points);

        Index ppp_last(nbr_points);
        ppp_last.set(0) = nbr_points(0) - 1;

        for (int k = 0; k < kkmax; k += 2) {

            posylm.set(2) = k;

            int mm = int(k / 2);

            // loop on theta
            int jmin = (mm % 2 == 0) ? int(mm / 2) : int((mm - 1) / 2);
            int jmax = (mm % 2 == 0) ? jjmax : jjmax - 1;
            for (int j = jmin; j < jmax; j++) {

                int ll = (mm % 2 == 0) ? 2 * j : 2 * j + 1;

                // Radial derivatives
                double Rxx, Rxy, Ryy, Rzz, Ixx, Ixy, Iyy, Izz;

                if (toinf) {
                    Rxx = multipoles_sym(k, j, OUTER_BC, gamma(1, 1)(dom - 1).der_r(), passage_even);
                    Rxy = multipoles_sym(k, j, OUTER_BC, gamma(1, 2)(dom - 1).der_r(), passage_even);
                    Ryy = multipoles_sym(k, j, OUTER_BC, gamma(2, 2)(dom - 1).der_r(), passage_even);
                    Rzz = multipoles_sym(k, j, OUTER_BC, gamma(3, 3)(dom - 1).der_r(), passage_even);

                    Ixx = ((k == kkmax - 1) || (k == 0))
                              ? 0
                              : multipoles_sym(k + 1, j, OUTER_BC, gamma(1, 1)(dom - 1).der_r(), passage_even);
                    Ixy = ((k == kkmax - 1) || (k == 0))
                              ? 0
                              : multipoles_sym(k + 1, j, OUTER_BC, gamma(1, 2)(dom - 1).der_r(), passage_even);
                    Iyy = ((k == kkmax - 1) || (k == 0))
                              ? 0
                              : multipoles_sym(k + 1, j, OUTER_BC, gamma(2, 2)(dom - 1).der_r(), passage_even);
                    Izz = ((k == kkmax - 1) || (k == 0))
                              ? 0
                              : multipoles_sym(k + 1, j, OUTER_BC, gamma(3, 3)(dom - 1).der_r(), passage_even);
                } else {
                    Rxx = multipoles_sym(k, j, OUTER_BC, gamma(1, 1)(dom).der_r(), passage_even);
                    Rxy = multipoles_sym(k, j, OUTER_BC, gamma(1, 2)(dom).der_r(), passage_even);
                    Ryy = multipoles_sym(k, j, OUTER_BC, gamma(2, 2)(dom).der_r(), passage_even);
                    Rzz = multipoles_sym(k, j, OUTER_BC, gamma(3, 3)(dom).der_r(), passage_even);

                    Ixx = ((k == kkmax - 1) || (k == 0))
                              ? 0
                              : multipoles_sym(k + 1, j, OUTER_BC, gamma(1, 1)(dom).der_r(), passage_even);
                    Ixy = ((k == kkmax - 1) || (k == 0))
                              ? 0
                              : multipoles_sym(k + 1, j, OUTER_BC, gamma(1, 2)(dom).der_r(), passage_even);
                    Iyy = ((k == kkmax - 1) || (k == 0))
                              ? 0
                              : multipoles_sym(k + 1, j, OUTER_BC, gamma(2, 2)(dom).der_r(), passage_even);
                    Izz = ((k == kkmax - 1) || (k == 0))
                              ? 0
                              : multipoles_sym(k + 1, j, OUTER_BC, gamma(3, 3)(dom).der_r(), passage_even);
                }

                double RV1 = Rxx - 2 * Ixy - Ryy;
                double RV2 = Rxx + 2 * Ixy - Ryy;
                double RV3 = Rxx + Ryy;
                double RV6 = Rzz;

                double IV1 = Ixx + 2 * Rxy - Iyy;
                double IV2 = Ixx - 2 * Rxy - Iyy;
                double IV3 = Ixx + Iyy;
                double IV6 = Izz;

                double RA1, RA2, RA3, RA6, IA1, IA2, IA3, IA6;

                if (toinf) {
                    RA1 = RV1 / (*dRsh[(mm + 2) * nbrl + ll])(ppp);
                    RA2 = RV2 / (*dRsh[abs(mm - 2) * nbrl + ll])(ppp);
                    RA3 = RV3 / (*dRsh[mm * nbrl + ll])(ppp);
                    RA6 = RV6 / (*dRsh[mm * nbrl + ll])(ppp);

                    IA1 = IV1 / (*dIsh[(mm + 2) * nbrl + ll])(ppp);
                    IA2 = IV2 / (*dIsh[abs(mm - 2) * nbrl + ll])(ppp);
                    IA3 = IV3 / (*dIsh[mm * nbrl + ll])(ppp);
                    IA6 = IV6 / (*dIsh[mm * nbrl + ll])(ppp);
                } else {
                    RA1 = RV1 / (*dRsh[(mm + 2) * nbrl + ll])(ppp_last);
                    RA2 = RV2 / (*dRsh[abs(mm - 2) * nbrl + ll])(ppp_last);
                    RA3 = RV3 / (*dRsh[mm * nbrl + ll])(ppp_last);
                    RA6 = RV6 / (*dRsh[mm * nbrl + ll])(ppp_last);

                    IA1 = IV1 / (*dIsh[(mm + 2) * nbrl + ll])(ppp_last);
                    IA2 = IV2 / (*dIsh[abs(mm - 2) * nbrl + ll])(ppp_last);
                    IA3 = IV3 / (*dIsh[mm * nbrl + ll])(ppp_last);
                    IA6 = IV6 / (*dIsh[mm * nbrl + ll])(ppp_last);
                }

                // Imaginary and real part of the eighenvectors
                Val_domain Rfxx(0.25 * RA1 * (*Rsh[(mm + 2) * nbrl + ll]) +
                                0.25 * RA2 * (*Rsh[abs(mm - 2) * nbrl + ll]) + 0.5 * RA3 * (*Rsh[mm * nbrl + ll]));

                Val_domain Ifxx(0.25 * IA1 * (*Ish[(mm + 2) * nbrl + ll]) +
                                0.25 * IA2 * (*Ish[abs(mm - 2) * nbrl + ll]) + 0.5 * IA3 * (*Ish[mm * nbrl + ll]));

                Val_domain Rfxy(0.25 * IA1 * (*Ish[(mm + 2) * nbrl + ll]) -
                                0.25 * IA2 * (*Ish[abs(mm - 2) * nbrl + ll]));

                Val_domain Ifxy(-0.25 * RA1 * (*Rsh[(mm + 2) * nbrl + ll]) +
                                0.25 * RA2 * (*Rsh[abs(mm - 2) * nbrl + ll]));

                Val_domain Rfyy(-0.25 * RA1 * (*Rsh[(mm + 2) * nbrl + ll]) -
                                0.25 * RA2 * (*Rsh[abs(mm - 2) * nbrl + ll]) + 0.5 * RA3 * (*Rsh[mm * nbrl + ll]));

                Val_domain Ifyy(-0.25 * IA1 * (*Ish[(mm + 2) * nbrl + ll]) -
                                0.25 * IA2 * (*Ish[abs(mm - 2) * nbrl + ll]) + 0.5 * IA3 * (*Ish[mm * nbrl + ll]));

                Val_domain Rfzz(RA6 * (*Rsh[mm * nbrl + ll]));

                Val_domain Ifzz(IA6 * (*Ish[mm * nbrl + ll]));

                Rfxx.coef();
                Ifxx.coef();
                Rfxy.coef();
                Ifxy.coef();
                Rfyy.coef();
                Ifyy.coef();
                Rfzz.coef();
                Ifzz.coef();

                // Loop on r
                for (int i = 0; i < nbr_coefs(0); i++) {
                    posylm.set(0) = i;
                    posradial.set(0) = i;

                    // Loop on theta for the matrix part :
                    for (int inds = 0; inds < nbr_coefs(1); inds++) {

                        posylm.set(1) = inds;

                        // Parts in cosines
                        res.set(1, 1).set_domain(dom).set_coef(posylm) +=
                            inv_even(mm, j, inds) * Rfxx.get_coef(posradial);
                        res.set(1, 2).set_domain(dom).set_coef(posylm) +=
                            inv_even(mm, j, inds) * Rfxy.get_coef(posradial);
                        res.set(2, 2).set_domain(dom).set_coef(posylm) +=
                            inv_even(mm, j, inds) * Rfyy.get_coef(posradial);
                        res.set(3, 3).set_domain(dom).set_coef(posylm) +=
                            inv_even(mm, j, inds) * Rfzz.get_coef(posradial);

                        // Parts in sines
                        if ((k != kkmax - 1) && (k != 0)) {
                            posylm.set(2)++;
                            res.set(1, 1).set_domain(dom).set_coef(posylm) +=
                                inv_even(mm, j, inds) * Ifxx.get_coef(posradial);
                            res.set(1, 2).set_domain(dom).set_coef(posylm) +=
                                inv_even(mm, j, inds) * Ifxy.get_coef(posradial);
                            res.set(2, 2).set_domain(dom).set_coef(posylm) +=
                                inv_even(mm, j, inds) * Ifyy.get_coef(posradial);
                            res.set(3, 3).set_domain(dom).set_coef(posylm) +=
                                inv_even(mm, j, inds) * Ifzz.get_coef(posradial);
                            posylm.set(2)--;
                        }
                    }
                }
            }
        }

        // Antisymetric part :
        // Loop on phi :
        for (int k = 0; k < kkmax; k += 2)
            if (k != 1) {

                posylm.set(2) = k;

                int mm = int(k / 2);

                // loop on theta
                int jmin = (mm % 2 == 0) ? int(mm / 2) : int((mm + 1) / 2);
                int jmax = jjmax - 1;
                for (int j = jmin; j < jmax; j++) {

                    int ll = (mm % 2 == 0) ? 2 * j + 1 : 2 * j;

                    double Rxz, Ryz, Ixz, Iyz;

                    if (toinf) {
                        Rxz = multipoles_asym(k, j, OUTER_BC, gamma(1, 3)(dom - 1).der_r(), passage_odd);
                        Ryz = multipoles_asym(k, j, OUTER_BC, gamma(2, 3)(dom - 1).der_r(), passage_odd);

                        Ixz = ((k == kkmax - 1) || (k == 0))
                                  ? 0
                                  : multipoles_asym(k + 1, j, OUTER_BC, gamma(1, 3)(dom - 1).der_r(), passage_odd);
                        Iyz = ((k == kkmax - 1) || (k == 0))
                                  ? 0
                                  : multipoles_asym(k + 1, j, OUTER_BC, gamma(2, 3)(dom - 1).der_r(), passage_odd);
                    } else {
                        Rxz = multipoles_asym(k, j, OUTER_BC, gamma(1, 3)(dom).der_r(), passage_odd);
                        Ryz = multipoles_asym(k, j, OUTER_BC, gamma(2, 3)(dom).der_r(), passage_odd);

                        Ixz = ((k == kkmax - 1) || (k == 0))
                                  ? 0
                                  : multipoles_asym(k + 1, j, OUTER_BC, gamma(1, 3)(dom).der_r(), passage_odd);
                        Iyz = ((k == kkmax - 1) || (k == 0))
                                  ? 0
                                  : multipoles_asym(k + 1, j, OUTER_BC, gamma(2, 3)(dom).der_r(), passage_odd);
                    }

                    double RV4 = Rxz - Iyz;
                    double RV5 = Rxz + Iyz;
                    double IV4 = Ixz + Ryz;
                    double IV5 = Ixz - Ryz;

                    double RA4, RA5, IA4, IA5;

                    if (toinf) {
                        RA4 = RV4 / (*dRsh[(mm + 1) * nbrl + ll])(ppp);
                        RA5 = RV5 / (*dRsh[abs(mm - 1) * nbrl + ll])(ppp);
                        IA4 = IV4 / (*dIsh[(mm + 1) * nbrl + ll])(ppp);
                        IA5 = IV5 / (*dIsh[abs(mm - 1) * nbrl + ll])(ppp);
                    } else {
                        RA4 = RV4 / (*dRsh[(mm + 1) * nbrl + ll])(ppp_last);
                        RA5 = RV5 / (*dRsh[abs(mm - 1) * nbrl + ll])(ppp_last);
                        IA4 = IV4 / (*dIsh[(mm + 1) * nbrl + ll])(ppp_last);
                        IA5 = IV5 / (*dIsh[abs(mm - 1) * nbrl + ll])(ppp_last);
                    }

                    // Imaginary and real part of the eighenvectors
                    Val_domain Rfxz(0.5 * RA4 * (*Rsh[(mm + 1) * nbrl + ll]) +
                                    0.5 * RA5 * (*Rsh[abs(mm - 1) * nbrl + ll]));
                    Val_domain Ifxz(0.5 * IA4 * (*Ish[(mm + 1) * nbrl + ll]) +
                                    0.5 * IA5 * (*Ish[abs(mm - 1) * nbrl + ll]));

                    Val_domain Rfyz(0.5 * IA4 * (*Ish[(mm + 1) * nbrl + ll]) -
                                    0.5 * IA5 * (*Ish[abs(mm - 1) * nbrl + ll]));
                    Val_domain Ifyz(-0.5 * RA4 * (*Rsh[(mm + 1) * nbrl + ll]) +
                                    0.5 * RA5 * (*Rsh[abs(mm - 1) * nbrl + ll]));

                    Rfxz.coef();
                    Ifxz.coef();
                    Rfyz.coef();
                    Ifyz.coef();

                    // Put in the right component :
                    for (int i = 0; i < nbr_coefs(0); i++) {
                        posylm.set(0) = i;
                        posradial.set(0) = i;

                        // Loop on theta for the matrix part :

                        for (int inds = 0; inds < nbr_coefs(1); inds++) {

                            posylm.set(1) = inds;

                            // Parts in cosines
                            res.set(1, 3).set_domain(dom).set_coef(posylm) +=
                                inv_odd(mm, j, inds) * Rfxz.get_coef(posradial);
                            res.set(2, 3).set_domain(dom).set_coef(posylm) +=
                                inv_odd(mm, j, inds) * Rfyz.get_coef(posradial);

                            // Parts in sines
                            if ((k != kkmax - 1) && (k != 0)) {
                                posylm.set(2)++;
                                res.set(1, 3).set_domain(dom).set_coef(posylm) +=
                                    inv_odd(mm, j, inds) * Ifxz.get_coef(posradial);
                                res.set(2, 3).set_domain(dom).set_coef(posylm) +=
                                    inv_odd(mm, j, inds) * Ifyz.get_coef(posradial);
                                posylm.set(2)--;
                            }
                        }
                    }
                }
            }

        return res;
    }
} // namespace Kadath
