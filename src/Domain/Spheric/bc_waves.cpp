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
#include "For_Kadath/Term_eq/term_eq.hpp"
#include <memory>
#include <vector>
namespace Kadath
{
    Array<double> mat_inv_leg_even(int, int);
    Array<double> mat_inv_leg_odd(int, int);
    Array<double> mat_leg_even(int, int);
    Array<double> mat_leg_odd(int, int);

    Term_eq Domain_shell::bc_waves(const Term_eq& gamma, const Term_eq& omega) const
    {

        // Free parameters for checking
        // int kkmax = nbr_coefs(2)-1 ;
        // int jjmax = nbr_coefs(1) ;
        int kkmax = 10;
        int jjmax = 5;
        int dom = gamma.get_dom();

        // Check if gamma is a metric
        if (gamma.type_data != TERM_T) {
            KADATH_THROW("Domain_shell::bc_waves only defined for a tensor");
        }
        int valence = gamma.val_t->get_valence();
        if (valence != 2) {
            KADATH_THROW("Domain_shell::bc_waves only defined with respect to second order tensor (i.e. valence must be 2)");
        }
        if (gamma.val_t->get_basis().get_basis(dom) != CARTESIAN_BASIS) {
            KADATH_THROW("Domain_shell::bc_waves only defined with respect to Cartesian tensorial basis");
        }

        bool doder = ((gamma.der_t == nullptr) || (omega.der_d == nullptr)) ? false : true;

        // Put rmax in a Term_eq
        Index posr(nbr_points);
        posr.set(0) = nbr_points(1) - 1;
        double rmax = get_radius()(posr);

        // Compute the term_eq for the BC
        int nbrm = int(nbr_coefs(2) / 2) + 2;
        int nbrl = 2 * nbr_coefs(1) - 1;
        // Real parts and Imaginar parts
        std::vector<std::unique_ptr<Term_eq>> Rsh(nbrm * nbrl);
        std::vector<std::unique_ptr<Term_eq>> Ish(nbrm * nbrl);
        std::vector<std::unique_ptr<Term_eq>> dRsh(nbrm * nbrl);
        std::vector<std::unique_ptr<Term_eq>> dIsh(nbrm * nbrl);

        for (int m = 0; m < nbrm; m++)
            for (int l = 0; l < nbrl; l++) {
                Rsh[m * nbrl + l] = (m == 0) ? std::make_unique<Term_eq>(dom, pow(rmax, -(l + 1)), 0)
                                             : std::make_unique<Term_eq>(bessel_jl(m * omega * rmax, l));
                Ish[m * nbrl + l] = (m == 0) ? std::make_unique<Term_eq>(dom, pow(rmax, -(l + 1)), 0)
                                             : std::make_unique<Term_eq>(bessel_yl(m * omega * rmax, l));
                dRsh[m * nbrl + l] =
                    (m == 0) ? std::make_unique<Term_eq>(dom, -(l + 1) * pow(rmax, -(l + 2)), 0)
                             : std::make_unique<Term_eq>(m * omega * bessel_djl(m * omega * rmax, l));
                dIsh[m * nbrl + l] =
                    (m == 0) ? std::make_unique<Term_eq>(dom, -(l + 1) * pow(rmax, -(l + 2)), 0)
                             : std::make_unique<Term_eq>(m * omega * bessel_dyl(m * omega * rmax, l));
            }

        // To store the result
        Tensor valres(*gamma.val_t, false);
        valres.std_base();
        for (int i = 0; i < valres.get_n_comp(); i++) {
            valres.set(valres.indices(i)).set_domain(dom).allocate_coef();
            Index pcf(nbr_coefs);
            do {
                valres.set(valres.indices(i)).set_domain(dom).set_coef(pcf) = 0;
            } while (pcf.inc());
        }
        Tensor derres(valres);

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

        for (int k = 0; k < kkmax; k += 2) {

            posylm.set(2) = k;

            int mm = int(k / 2);

            // loop on theta
            int jmin = (mm % 2 == 0) ? int(mm / 2) : int((mm - 1) / 2);
            int jmax = (mm % 2 == 0) ? jjmax : jjmax - 1;
            for (int j = jmin; j < jmax; j++) {

                int ll = (mm % 2 == 0) ? 2 * j : 2 * j + 1;

                // Radial derivatives
                double Rvaldrxx = multipoles_sym(k, j, OUTER_BC, (*gamma.val_t)(1, 1)(dom).der_r(), passage_even);
                double Rvaldrxy = multipoles_sym(k, j, OUTER_BC, (*gamma.val_t)(1, 2)(dom).der_r(), passage_even);
                double Rvaldryy = multipoles_sym(k, j, OUTER_BC, (*gamma.val_t)(2, 2)(dom).der_r(), passage_even);
                double Rvaldrzz = multipoles_sym(k, j, OUTER_BC, (*gamma.val_t)(3, 3)(dom).der_r(), passage_even);

                double Ivaldrxx =
                    ((k == kkmax - 1) || (k == 0))
                        ? 0
                        : multipoles_sym(k + 1, j, OUTER_BC, (*gamma.val_t)(1, 1)(dom).der_r(), passage_even);
                double Ivaldrxy =
                    ((k == kkmax - 1) || (k == 0))
                        ? 0
                        : multipoles_sym(k + 1, j, OUTER_BC, (*gamma.val_t)(1, 2)(dom).der_r(), passage_even);
                double Ivaldryy =
                    ((k == kkmax - 1) || (k == 0))
                        ? 0
                        : multipoles_sym(k + 1, j, OUTER_BC, (*gamma.val_t)(2, 2)(dom).der_r(), passage_even);
                double Ivaldrzz =
                    ((k == kkmax - 1) || (k == 0))
                        ? 0
                        : multipoles_sym(k + 1, j, OUTER_BC, (*gamma.val_t)(3, 3)(dom).der_r(), passage_even);

                Term_eq* RdV1;
                Term_eq* RdV2;
                Term_eq* RdV3;
                Term_eq* RdV6;

                Term_eq* IdV1;
                Term_eq* IdV2;
                Term_eq* IdV3;
                Term_eq* IdV6;

                if (doder) {

                    double Rderdrxx = multipoles_sym(k, j, OUTER_BC, (*gamma.der_t)(1, 1)(dom).der_r(), passage_even);
                    double Rderdrxy = multipoles_sym(k, j, OUTER_BC, (*gamma.der_t)(1, 2)(dom).der_r(), passage_even);
                    double Rderdryy = multipoles_sym(k, j, OUTER_BC, (*gamma.der_t)(2, 2)(dom).der_r(), passage_even);
                    double Rderdrzz = multipoles_sym(k, j, OUTER_BC, (*gamma.der_t)(3, 3)(dom).der_r(), passage_even);

                    double Iderdrxx =
                        ((k == kkmax - 1) || (k == 0))
                            ? 0
                            : multipoles_sym(k + 1, j, OUTER_BC, (*gamma.der_t)(1, 1)(dom).der_r(), passage_even);
                    double Iderdrxy =
                        ((k == kkmax - 1) || (k == 0))
                            ? 0
                            : multipoles_sym(k + 1, j, OUTER_BC, (*gamma.der_t)(1, 2)(dom).der_r(), passage_even);
                    double Iderdryy =
                        ((k == kkmax - 1) || (k == 0))
                            ? 0
                            : multipoles_sym(k + 1, j, OUTER_BC, (*gamma.der_t)(2, 2)(dom).der_r(), passage_even);
                    double Iderdrzz =
                        ((k == kkmax - 1) || (k == 0))
                            ? 0
                            : multipoles_sym(k + 1, j, OUTER_BC, (*gamma.der_t)(3, 3)(dom).der_r(), passage_even);

                    RdV1 = new Term_eq(dom, Rvaldrxx - 2 * Ivaldrxy - Rvaldryy, Rderdrxx - 2 * Iderdrxy - Rderdryy);
                    RdV2 = new Term_eq(dom, Rvaldrxx + 2 * Ivaldrxy - Rvaldryy, Rderdrxx + 2 * Iderdrxy - Rderdryy);
                    RdV3 = new Term_eq(dom, Rvaldrxx + Rvaldryy, Rderdrxx + Rderdryy);
                    RdV6 = new Term_eq(dom, Rvaldrzz, Rderdrzz);

                    IdV1 = new Term_eq(dom, Ivaldrxx + 2 * Rvaldrxy - Ivaldryy, Iderdrxx + 2 * Rderdrxy - Iderdryy);
                    IdV2 = new Term_eq(dom, Ivaldrxx - 2 * Rvaldrxy - Ivaldryy, Iderdrxx - 2 * Rderdrxy - Iderdryy);
                    IdV3 = new Term_eq(dom, Ivaldrxx + Ivaldryy, Iderdrxx + Iderdryy);
                    IdV6 = new Term_eq(dom, Ivaldrzz, Iderdrzz);

                } else {

                    RdV1 = new Term_eq(dom, Rvaldrxx - 2 * Ivaldrxy - Rvaldryy);
                    RdV2 = new Term_eq(dom, Rvaldrxx + 2 * Ivaldrxy - Rvaldryy);
                    RdV3 = new Term_eq(dom, Rvaldrxx + Rvaldryy);
                    RdV6 = new Term_eq(dom, Rvaldrzz);

                    IdV1 = new Term_eq(dom, Ivaldrxx + 2 * Rvaldrxy - Ivaldryy);
                    IdV2 = new Term_eq(dom, Ivaldrxx - 2 * Rvaldrxy - Ivaldryy);
                    IdV3 = new Term_eq(dom, Ivaldrxx + Ivaldryy);
                    IdV6 = new Term_eq(dom, Ivaldrzz);
                }

                Term_eq RA1(*RdV1 / (*dRsh[(mm + 2) * nbrl + ll]));
                Term_eq RA2(*RdV2 / (*dRsh[abs(mm - 2) * nbrl + ll]));
                Term_eq RA3(*RdV3 / (*dRsh[mm * nbrl + ll]));
                Term_eq RA6(*RdV6 / (*dRsh[mm * nbrl + ll]));

                Term_eq IA1(*IdV1 / (*dIsh[(mm + 2) * nbrl + ll]));
                Term_eq IA2(*IdV2 / (*dIsh[abs(mm - 2) * nbrl + ll]));
                Term_eq IA3(*IdV3 / (*dIsh[mm * nbrl + ll]));
                Term_eq IA6(*IdV6 / (*dIsh[mm * nbrl + ll]));

                // Imaginary and real part of the eighenvectors
                Term_eq Rxx(0.25 * RA1 * (*Rsh[(mm + 2) * nbrl + ll]) + 0.25 * RA2 * (*Rsh[abs(mm - 2) * nbrl + ll]) +
                            0.5 * RA3 * (*Rsh[mm * nbrl + ll]));

                Term_eq Ixx(0.25 * IA1 * (*Ish[(mm + 2) * nbrl + ll]) + 0.25 * IA2 * (*Ish[abs(mm - 2) * nbrl + ll]) +
                            0.5 * IA3 * (*Ish[mm * nbrl + ll]));

                Term_eq Rxy(0.25 * IA1 * (*Ish[(mm + 2) * nbrl + ll]) - 0.25 * IA2 * (*Ish[abs(mm - 2) * nbrl + ll]));

                Term_eq Ixy(-0.25 * RA1 * (*Rsh[(mm + 2) * nbrl + ll]) + 0.25 * RA2 * (*Rsh[abs(mm - 2) * nbrl + ll]));

                Term_eq Ryy(-0.25 * RA1 * (*Rsh[(mm + 2) * nbrl + ll]) - 0.25 * RA2 * (*Rsh[abs(mm - 2) * nbrl + ll]) +
                            0.5 * RA3 * (*Rsh[mm * nbrl + ll]));

                Term_eq Iyy(-0.25 * IA1 * (*Ish[(mm + 2) * nbrl + ll]) - 0.25 * IA2 * (*Ish[abs(mm - 2) * nbrl + ll]) +
                            0.5 * IA3 * (*Ish[mm * nbrl + ll]));

                Term_eq Rzz(RA6 * (*Rsh[mm * nbrl + ll]));

                Term_eq Izz(IA6 * (*Ish[mm * nbrl + ll]));

                // Put in the right component :
                posylm.set(0) = 0;

                // Loop on theta for the matrix part :
                for (int inds = 0; inds < nbr_coefs(1); inds++) {

                    posylm.set(1) = inds;

                    // Parts in cosines
                    valres.set(1, 1).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Rxx.val_d);
                    valres.set(1, 2).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Rxy.val_d);
                    valres.set(2, 2).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Ryy.val_d);
                    valres.set(3, 3).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Rzz.val_d);

                    if (doder) {
                        derres.set(1, 1).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Rxx.der_d);
                        derres.set(1, 2).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Rxy.der_d);
                        derres.set(2, 2).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Ryy.der_d);
                        derres.set(3, 3).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Rzz.der_d);
                    }

                    // Parts in sines
                    if ((k != kkmax - 1) && (k != 0)) {
                        posylm.set(2)++;
                        valres.set(1, 1).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Ixx.val_d);
                        valres.set(1, 2).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Ixy.val_d);
                        valres.set(2, 2).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Iyy.val_d);
                        valres.set(3, 3).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Izz.val_d);

                        if (doder) {
                            derres.set(1, 1).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Ixx.der_d);
                            derres.set(1, 2).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Ixy.der_d);
                            derres.set(2, 2).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Iyy.der_d);
                            derres.set(3, 3).set_domain(dom).set_coef(posylm) += inv_even(mm, j, inds) * (*Izz.der_d);
                        }
                        posylm.set(2)--;
                    }
                }

                delete RdV1;
                delete RdV2;
                delete RdV3;
                delete RdV6;

                delete IdV1;
                delete IdV2;
                delete IdV3;
                delete IdV6;
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

                    // Radial derivatives
                    double Rvaldrxz = multipoles_asym(k, j, OUTER_BC, (*gamma.val_t)(1, 3)(dom).der_r(), passage_odd);
                    double Rvaldryz = multipoles_asym(k, j, OUTER_BC, (*gamma.val_t)(2, 3)(dom).der_r(), passage_odd);

                    double Ivaldrxz =
                        ((k == kkmax - 1) || (k == 0))
                            ? 0
                            : multipoles_asym(k + 1, j, OUTER_BC, (*gamma.val_t)(1, 3)(dom).der_r(), passage_odd);
                    double Ivaldryz =
                        ((k == kkmax - 1) || (k == 0))
                            ? 0
                            : multipoles_asym(k + 1, j, OUTER_BC, (*gamma.val_t)(2, 3)(dom).der_r(), passage_odd);

                    Term_eq* RdV4;
                    Term_eq* RdV5;
                    Term_eq* IdV4;
                    Term_eq* IdV5;

                    if (doder) {

                        double Rderdrxz =
                            multipoles_asym(k, j, OUTER_BC, (*gamma.der_t)(1, 3)(dom).der_r(), passage_odd);
                        double Rderdryz =
                            multipoles_asym(k, j, OUTER_BC, (*gamma.der_t)(2, 3)(dom).der_r(), passage_odd);

                        double Iderdrxz =
                            ((k == kkmax - 1) || (k == 0))
                                ? 0
                                : multipoles_asym(k + 1, j, OUTER_BC, (*gamma.der_t)(1, 3)(dom).der_r(), passage_odd);
                        double Iderdryz =
                            ((k == kkmax - 1) || (k == 0))
                                ? 0
                                : multipoles_asym(k + 1, j, OUTER_BC, (*gamma.der_t)(2, 3)(dom).der_r(), passage_odd);

                        RdV4 = new Term_eq(dom, Rvaldrxz - Ivaldryz, Rderdrxz - Iderdryz);
                        RdV5 = new Term_eq(dom, Rvaldrxz + Ivaldryz, Rderdrxz + Iderdryz);

                        IdV4 = new Term_eq(dom, Ivaldrxz + Rvaldryz, Iderdrxz + Rderdryz);
                        IdV5 = new Term_eq(dom, Ivaldrxz - Rvaldryz, Iderdrxz - Rderdryz);
                    } else {
                        RdV4 = new Term_eq(dom, Rvaldrxz - Ivaldryz);
                        RdV5 = new Term_eq(dom, Rvaldrxz + Ivaldryz);

                        IdV4 = new Term_eq(dom, Ivaldrxz + Rvaldryz);
                        IdV5 = new Term_eq(dom, Ivaldrxz - Rvaldryz);
                    }

                    Term_eq RA4(*RdV4 / (*dRsh[(mm + 1) * nbrl + ll]));
                    Term_eq RA5(*RdV5 / (*dRsh[abs(mm - 1) * nbrl + ll]));

                    Term_eq IA4(*IdV4 / (*dIsh[(mm + 1) * nbrl + ll]));
                    Term_eq IA5(*IdV5 / (*dIsh[abs(mm - 1) * nbrl + ll]));

                    // Case m==0 is different
                    // int signe = (mm==0) ? -1 : 1 ;

                    // Imaginary and real part of the eighenvectors
                    Term_eq Rxz(0.5 * RA4 * (*Rsh[(mm + 1) * nbrl + ll]) + 0.5 * RA5 * (*Rsh[abs(mm - 1) * nbrl + ll]));
                    Term_eq Ixz(0.5 * IA4 * (*Ish[(mm + 1) * nbrl + ll]) + 0.5 * IA5 * (*Ish[abs(mm - 1) * nbrl + ll]));

                    Term_eq Ryz(0.5 * IA4 * (*Ish[(mm + 1) * nbrl + ll]) - 0.5 * IA5 * (*Ish[abs(mm - 1) * nbrl + ll]));
                    Term_eq Iyz(-0.5 * RA4 * (*Rsh[(mm + 1) * nbrl + ll]) +
                                0.5 * RA5 * (*Rsh[abs(mm - 1) * nbrl + ll]));

                    // Put in the right component :
                    posylm.set(0) = 0;

                    // Loop on theta for the matrix part :

                    for (int inds = 0; inds < nbr_coefs(1); inds++) {

                        posylm.set(1) = inds;

                        // Parts in cosines
                        valres.set(1, 3).set_domain(dom).set_coef(posylm) += inv_odd(mm, j, inds) * (*Rxz.val_d);
                        valres.set(2, 3).set_domain(dom).set_coef(posylm) += inv_odd(mm, j, inds) * (*Ryz.val_d);

                        if (doder) {
                            derres.set(1, 3).set_domain(dom).set_coef(posylm) += inv_odd(mm, j, inds) * (*Rxz.der_d);
                            derres.set(2, 3).set_domain(dom).set_coef(posylm) += inv_odd(mm, j, inds) * (*Ryz.der_d);
                        }

                        // Parts in sines
                        if ((k != kkmax - 1) && (k != 0)) {
                            posylm.set(2)++;
                            valres.set(1, 3).set_domain(dom).set_coef(posylm) += inv_odd(mm, j, inds) * (*Ixz.val_d);
                            valres.set(2, 3).set_domain(dom).set_coef(posylm) += inv_odd(mm, j, inds) * (*Iyz.val_d);

                            if (doder) {
                                derres.set(1, 3).set_domain(dom).set_coef(posylm) +=
                                    inv_odd(mm, j, inds) * (*Ixz.der_d);
                                derres.set(2, 3).set_domain(dom).set_coef(posylm) +=
                                    inv_odd(mm, j, inds) * (*Iyz.der_d);
                            }
                            posylm.set(2)--;
                        }
                    }

                    delete RdV4;
                    delete RdV5;
                    delete IdV4;
                    delete IdV5;
                }
            }

        if (doder)
            return Term_eq(dom, valres, derres);
        else
            return Term_eq(dom, valres);
    }
} // namespace Kadath
