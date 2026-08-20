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

#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
namespace Kadath
{
    void coef_1d(int, Array<double>&);
    double integral_1d(int, const Array<double>&);

    // Computation norm Legendre associated
    Array<double> legendre_norme(int mm, int nt)
    {

        int nt2 = 2 * nt - 1; // To avoid aliasing
        Array<double> res(nt2 - mm, nt2);

        Array<double> theta(nt2);
        Array<double> sintheta(nt2);
        Array<double> costheta(nt2);
        Array<double> coloc(nt2);
        for (int i = 0; i < nt2; i++) {
            theta.set(i) = M_PI * i / 2. / (nt2 - 1);
            sintheta.set(i) = sin(theta(i));
            costheta.set(i) = cos(theta(i));
        }

        for (int i = 0; i < nt2; i++)
            res.set(0, i) = pow(-sintheta(i), mm);
        for (int i = 0; i < nt2; i++)
            res.set(1, i) = res(0, i) * costheta(i) * (2 * mm + 1);

        int ll = mm;
        for (int j = 0; j < nt2 - mm - 2; j++) {
            for (int i = 0; i < nt2; i++)
                res.set(j + 2, i) =
                    ((2 * ll + 3) * costheta(i) * res(j + 1, i) - (ll + 1 + mm) * res(j, i)) / (ll - mm + 2);
            ll++;
        }

        for (int j = 0; j < nt2 - mm; j++) {
            for (int i = 0; i < nt2; i++)
                coloc.set(nt2 - i - 1) = res(j, i) * res(j, i);
            coef_1d(CHEB_EVEN, coloc);
            double norme = 2 * integral_1d(CHEB_EVEN, coloc);
            for (int i = 0; i < nt2; i++)
                res.set(j, i) /= sqrt(norme);
        }

        return res;
    }

    Array<double> mat_leg_even(int nt, int np)
    {

        int nm = int(np / 2) + 1;
        Dim_array dims(3);
        dims.set(0) = nm;
        dims.set(1) = nt;
        dims.set(2) = nt;
        Array<double> mat(dims);

        int nt2 = 2 * nt - 1;
        Array<double> coloc(nt2);

        Index pos(dims);

        // Loop on m :
        for (int m = 0; m < nm; m++) {
            pos.set(0) = m;
            Array<double> leg(legendre_norme(m, nt));
            if (m % 2 == 0) {
                // Case even :
                // Loop on l
                for (int l = int(m / 2); l < nt; l++) {
                    pos.set(1) = l;
                    int ll = 2 * l;
                    // Loop on j
                    for (int j = 0; j < nt; j++) {
                        pos.set(2) = j;
                        for (int nn = 0; nn < nt2; nn++)
                            coloc.set(nt2 - nn - 1) = cos(2 * j * M_PI * nn / (nt2 - 1) / 2.) * leg(ll - m, nn);
                        coef_1d(CHEB_EVEN, coloc);
                        mat.set(pos) = 2 * integral_1d(CHEB_EVEN, coloc);
                    }
                }
            } else {
                // Case odd
                // Loop on l
                for (int l = int((m - 1) / 2); l < nt - 1; l++) {
                    pos.set(1) = l;
                    int ll = 2 * l + 1;
                    // Loop on j
                    for (int j = 0; j < nt - 1; j++) {
                        pos.set(2) = j;
                        for (int nn = 0; nn < nt2; nn++)
                            coloc.set(nt2 - nn - 1) = sin((2 * j + 1) * M_PI * nn / (nt2 - 1) / 2.) * leg(ll - m, nn);
                        coef_1d(CHEB_EVEN, coloc);
                        mat.set(pos) = 2 * integral_1d(CHEB_EVEN, coloc);
                    }
                }
            }
        }
        return mat;
    }

    Array<double> mat_leg_odd(int nt, int np)
    {

        int nm = int(np / 2) + 1;
        Dim_array dims(3);
        dims.set(0) = nm;
        dims.set(1) = nt;
        dims.set(2) = nt;
        Array<double> mat(dims);

        int nt2 = 2 * nt - 1;
        Array<double> coloc(nt2);

        Index pos(dims);

        // Loop on m :
        for (int m = 0; m < nm; m++) {
            pos.set(0) = m;
            Array<double> leg(legendre_norme(m, nt));
            if (m % 2 == 0) {
                // Case even :
                // Loop on l
                for (int l = int(m / 2); l < nt - 1; l++) {
                    pos.set(1) = l;
                    int ll = 2 * l + 1;
                    // Loop on j
                    for (int j = 0; j < nt - 1; j++) {
                        pos.set(2) = j;
                        for (int nn = 0; nn < nt2; nn++)
                            coloc.set(nt2 - nn - 1) = cos((2 * j + 1) * M_PI * nn / (nt2 - 1) / 2.) * leg(ll - m, nn);

                        coef_1d(CHEB_EVEN, coloc);
                        mat.set(pos) = 2 * integral_1d(CHEB_EVEN, coloc);
                    }
                }
            } else {
                // Case odd
                // Loop on l
                for (int l = int((m + 1) / 2); l < nt - 1; l++) {
                    pos.set(1) = l;
                    int ll = 2 * l;
                    // Loop on j
                    for (int j = 0; j < nt - 1; j++) {
                        pos.set(2) = j;
                        for (int nn = 0; nn < nt2; nn++)
                            coloc.set(nt2 - nn - 1) = sin(2 * j * M_PI * nn / (nt2 - 1) / 2.) * leg(ll - m, nn);
                        coef_1d(CHEB_EVEN, coloc);
                        mat.set(pos) = 2 * integral_1d(CHEB_EVEN, coloc);
                    }
                }
            }
        }
        return mat;
    }

    Array<double> mat_inv_leg_even(int nt, int np)
    {

        int nm = int(np / 2) + 1;
        Dim_array dims(3);
        dims.set(0) = nm;
        dims.set(1) = nt;
        dims.set(2) = nt;
        Array<double> mat(dims);

        int nt2 = 2 * nt - 1;
        Array<double> coloc(nt2);

        Index pos(dims);

        // Loop on m :
        for (int m = 0; m < nm; m++) {
            pos.set(0) = m;
            Array<double> leg(legendre_norme(m, nt));
            if (m % 2 == 0) {
                // Case even :
                // Loop on l
                for (int l = int(m / 2); l < nt; l++) {
                    pos.set(1) = l;
                    int ll = 2 * l;
                    // Loop on j
                    for (int j = 0; j < nt; j++) {
                        pos.set(2) = j;
                        for (int nn = 0; nn < nt2; nn++)
                            coloc.set(nn) = leg(ll - m, nn);
                        coef_1d(COS_EVEN, coloc);
                        mat.set(pos) = coloc(j);
                    }
                }
            } else {
                // Case odd
                // Loop on l
                for (int l = int((m - 1) / 2); l < nt - 1; l++) {
                    pos.set(1) = l;
                    int ll = 2 * l + 1;
                    // Loop on j
                    for (int j = 0; j < nt; j++) {
                        pos.set(2) = j;
                        for (int nn = 0; nn < nt2; nn++)
                            coloc.set(nn) = leg(ll - m, nn);
                        coef_1d(SIN_ODD, coloc);
                        mat.set(pos) = coloc(j);
                    }
                }
            }
        }
        return mat;
    }

    Array<double> mat_inv_leg_odd(int nt, int np)
    {

        int nm = int(np / 2) + 1;
        Dim_array dims(3);
        dims.set(0) = nm;
        dims.set(1) = nt;
        dims.set(2) = nt;
        Array<double> mat(dims);

        int nt2 = 2 * nt - 1;
        Array<double> coloc(nt2);

        Index pos(dims);

        // Loop on m :
        for (int m = 0; m < nm; m++) {
            pos.set(0) = m;
            Array<double> leg(legendre_norme(m, nt));
            if (m % 2 == 0) {
                // Case even :
                // Loop on l
                for (int l = int(m / 2); l < nt - 1; l++) {
                    pos.set(1) = l;
                    int ll = 2 * l + 1;
                    // Loop on j
                    for (int j = 0; j < nt; j++) {
                        pos.set(2) = j;
                        for (int nn = 0; nn < nt2; nn++)
                            coloc.set(nn) = leg(ll - m, nn);
                        coef_1d(COS_ODD, coloc);
                        mat.set(pos) = coloc(j);
                    }
                }
            } else {
                // Case odd
                // Loop on l
                for (int l = int((m + 1) / 2); l < nt - 1; l++) {
                    pos.set(1) = l;
                    int ll = 2 * l;
                    // Loop on j
                    for (int j = 0; j < nt; j++) {
                        pos.set(2) = j;
                        for (int nn = 0; nn < nt2; nn++)
                            coloc.set(nn) = leg(ll - m, nn);
                        coef_1d(SIN_EVEN, coloc);
                        mat.set(pos) = coloc(j);
                    }
                }
            }
        }
        return mat;
    }

    Term_eq Domain_shell::multipoles_sym(int k, int j, int bound, const Term_eq& so, const Array<double>& passage) const
    {

        // Check if so is a tensor
        if (so.type_data != TERM_T) {
            KADATH_THROW("Domain_shell::multipoles_sym only defined for a tensor");
        }
        int valence = so.val_t->get_valence();
        if (valence != 0) {
            KADATH_THROW("Domain_shell::multipoles_sym only defined with respect to a component (i.e. valence must be 0)");
        }
        assert((bound == INNER_BC) || (bound == OUTER_BC));

        bool doder = (so.der_t == nullptr) ? false : true;
        int dom = so.get_dom();

        Index pos(passage.dimensions);
        Index pos_bound(nbr_coefs);

        double alm = 0;
        double alm_der = 0;

        int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
        pos.set(0) = mm;
        pos_bound.set(2) = k;
        if (mm % 2 == 0) {
            // Case m even
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1); i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, (*so.val_t)()(dom), pos_bound);
                if (doder)
                    alm_der += passage(pos) * val_boundary(bound, (*so.der_t)()(dom), pos_bound);
            }
        } else {
            // Case m odd
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1) - 1; i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, (*so.val_t)()(dom), pos_bound);
                if (doder)
                    alm_der += passage(pos) * val_boundary(bound, (*so.der_t)()(dom), pos_bound);
            }
        }

        Val_domain res(this);
        res.allocate_conf();
        *res.c = 0.;
        Val_domain der(this);
        der.allocate_conf();
        *der.c = 0.;

        // Get theta
        Array<double> phi(get_coloc(3));
        Array<double> leg(legendre_norme(mm, nbr_coefs(1)));
        if (mm % 2 == 0) {
            // Loop on l :
            Index pp(nbr_points);
            do {
                res.set(pp) = (k % 2 == 0) ? alm * cos(mm * phi(pp(2))) * leg(2 * (j - int(mm / 2)), 2 * pp(1))
                                           : alm * sin(mm * phi(pp(2))) * leg(2 * (j - int(mm / 2)), 2 * pp(1));
                if (doder)
                    der.set(pp) = (k % 2 == 0) ? alm_der * cos(mm * phi(pp(2))) * leg(2 * (j - int(mm / 2)), 2 * pp(1))
                                               : alm_der * sin(mm * phi(pp(2))) * leg(2 * (j - int(mm / 2)), 2 * pp(1));
            } while (pp.inc());
        } else {
            // Loop on l :
            Index pp(nbr_points);
            do {
                res.set(pp) = (k % 2 == 0) ? alm * cos(mm * phi(pp(2))) * leg(2 * (j - int((mm - 1) / 2)), 2 * pp(1))
                                           : alm * sin(mm * phi(pp(2))) * leg(2 * (j - int((mm - 1) / 2)), 2 * pp(1));
                if (doder)
                    der.set(pp) = (k % 2 == 0)
                                      ? alm_der * cos(mm * phi(pp(2))) * leg(2 * (j - int((mm - 1) / 2)), 2 * pp(1))
                                      : alm_der * sin(mm * phi(pp(2))) * leg(2 * (j - int((mm - 1) / 2)), 2 * pp(1));
            } while (pp.inc());
        }

        res.set_base() = (*so.val_t)()(dom).get_base();
        if (doder)
            der.set_base() = (*so.der_t)()(dom).get_base();

        // For speed
        if (fabs(alm) < PRECISION)
            res.set_zero();
        if (fabs(alm_der) < PRECISION)
            der.set_zero();

        Scalar res_scal(so.val_t->get_space());
        res_scal.set_domain(dom) = res;

        if (doder) {
            Scalar der_scal(so.val_t->get_space());
            der_scal.set_domain(dom) = der;
            return Term_eq(dom, res_scal, der_scal);
        } else {
            return Term_eq(dom, res_scal);
        }
    }

    Term_eq Domain_shell::multipoles_asym(int k, int j, int bound, const Term_eq& so,
                                          const Array<double>& passage) const
    {

        // Check if so is a tensor
        if (so.type_data != TERM_T) {
            KADATH_THROW("Domain_shell::multipoles_asym only defined for a tensor");
        }
        int valence = so.val_t->get_valence();
        if (valence != 0) {
            KADATH_THROW("Domain_shell::multipoles_asym only defined with respect to a component (i.e. valence must be 0)");
        }
        assert((bound == INNER_BC) || (bound == OUTER_BC));

        bool doder = (so.der_t == nullptr) ? false : true;
        int dom = so.get_dom();

        Index pos(passage.dimensions);
        Index pos_bound(nbr_coefs);

        double alm = 0;
        double alm_der = 0;

        int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
        pos.set(0) = mm;
        pos_bound.set(2) = k;
        if (mm % 2 == 0) {
            // Case m even
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1) - 1; i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, (*so.val_t)()(dom), pos_bound);
                if (doder)
                    alm_der += passage(pos) * val_boundary(bound, (*so.der_t)()(dom), pos_bound);
            }
        } else {
            // Case m odd
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1) - 1; i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, (*so.val_t)()(dom), pos_bound);
                if (doder)
                    alm_der += passage(pos) * val_boundary(bound, (*so.der_t)()(dom), pos_bound);
            }
        }

        Val_domain res(this);
        res.allocate_conf();
        *res.c = 0.;
        Val_domain der(this);
        der.allocate_conf();
        *der.c = 0.;

        // Get theta
        Array<double> phi(get_coloc(3));
        Array<double> leg(legendre_norme(mm, nbr_coefs(1)));
        if (mm % 2 == 0) {
            // Loop on l :
            Index pp(nbr_points);
            do {
                res.set(pp) = (k % 2 == 0) ? alm * cos(mm * phi(pp(2))) * leg(2 * (j - int(mm / 2)) + 1, 2 * pp(1))
                                           : alm * sin(mm * phi(pp(2))) * leg(2 * (j - int(mm / 2)) + 1, 2 * pp(1));
                if (doder)
                    der.set(pp) = (k % 2 == 0)
                                      ? alm_der * cos(mm * phi(pp(2))) * leg(2 * (j - int(mm / 2)) + 1, 2 * pp(1))
                                      : alm_der * sin(mm * phi(pp(2))) * leg(2 * (j - int(mm / 2)) + 1, 2 * pp(1));
            } while (pp.inc());
        } else {
            // Loop on l :
            Index pp(nbr_points);
            do {
                res.set(pp) = (k % 2 == 0)
                                  ? alm * cos(mm * phi(pp(2))) * leg(2 * (j - int((mm + 1) / 2)) + 1, 2 * pp(1))
                                  : alm * sin(mm * phi(pp(2))) * leg(2 * (j - int((mm + 1) / 2)) + 1, 2 * pp(1));
                if (doder)
                    der.set(pp) =
                        (k % 2 == 0) ? alm_der * cos(mm * phi(pp(2))) * leg(2 * (j - int((mm + 1) / 2)) + 1, 2 * pp(1))
                                     : alm_der * sin(mm * phi(pp(2))) * leg(2 * (j - int((mm + 1) / 2)) + 1, 2 * pp(1));
            } while (pp.inc());
        }

        res.set_base() = (*so.val_t)()(dom).get_base();
        if (doder)
            der.set_base() = (*so.der_t)()(dom).get_base();

        // For speed
        if (fabs(alm) < PRECISION)
            res.set_zero();
        if (fabs(alm_der) < PRECISION)
            der.set_zero();

        Scalar res_scal(so.val_t->get_space());
        res_scal.set_domain(dom) = res;

        if (doder) {
            Scalar der_scal(so.val_t->get_space());
            der_scal.set_domain(dom) = der;
            return Term_eq(dom, res_scal, der_scal);
        } else {
            return Term_eq(dom, res_scal);
        }
    }

    Term_eq Domain_shell::radial_part_sym(const Space& space, int k, int j, const Term_eq& ome,
                                          Term_eq (*fit)(const Space&, int, int, const Term_eq&, const Param&),
                                          const Param& parfit) const
    {

        // Check if omega is a double
        if (ome.type_data != TERM_D) {
            KADATH_THROW("Omega must be a double in Domain_harmonics::sym");
        }

        // Loop on dimensions of alm :
        int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
        int ll = (mm % 2 == 0) ? 2 * j : 2 * j + 1;
        return fit(space, mm, ll, ome, parfit);
    }

    Term_eq Domain_shell::radial_part_asym(const Space& space, int k, int j, const Term_eq& ome,
                                           Term_eq (*fit)(const Space&, int, int, const Term_eq&, const Param&),
                                           const Param& parfit) const
    {

        // Check if omega is a double
        if (ome.type_data != TERM_D) {
            KADATH_THROW("Omega must be a double in Domain_harmonics::sym");
        }

        // Loop on dimensions of alm :
        int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
        int ll = (mm % 2 == 0) ? 2 * j + 1 : 2 * j;
        return fit(space, mm, ll, ome, parfit);
    }

    Term_eq Domain_shell::harmonics_sym(const Term_eq& so, const Term_eq& ome, int bound,
                                        Term_eq (*fit)(const Space&, int, int, const Term_eq&, const Param&),
                                        const Param& parfit, const Array<double>& passage) const
    {

        Term_eq res(so);
        bool first = true;

        // Loop on k :
        for (int k = 0; k < nbr_coefs(2) - 1; k++)
            if (k != 1) {
                int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
                if (mm % 2 == 0) {
                    for (int j = int(mm / 2); j < nbr_coefs(1); j++) {
                        if (first) {
                            res = multipoles_sym(k, j, bound, so, passage) *
                                  radial_part_sym(so.val_t->get_space(), k, j, ome, fit, parfit);
                            first = false;
                        } else
                            res = res + multipoles_sym(k, j, bound, so, passage) *
                                            radial_part_sym(so.val_t->get_space(), k, j, ome, fit, parfit);
                    }
                } else {
                    for (int j = int((mm - 1) / 2); j < nbr_coefs(1) - 1; j++) {
                        if (first) {
                            res = multipoles_sym(k, j, bound, so, passage) *
                                  radial_part_sym(so.val_t->get_space(), k, j, ome, fit, parfit);
                            first = false;
                        } else
                            res = res + multipoles_sym(k, j, bound, so, passage) *
                                            radial_part_sym(so.val_t->get_space(), k, j, ome, fit, parfit);
                    }
                }
            }

        return res;
    }

    Term_eq Domain_shell::harmonics_asym(const Term_eq& so, const Term_eq& ome, int bound,
                                         Term_eq (*fit)(const Space&, int, int, const Term_eq&, const Param&),
                                         const Param& parfit, const Array<double>& passage) const
    {

        Term_eq res(so);
        bool first = true;

        // Loop on k :
        for (int k = 0; k < nbr_coefs(2) - 1; k++)
            if (k != 1) {
                int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
                if (mm % 2 == 0) {
                    for (int j = int(mm / 2); j < nbr_coefs(1) - 1; j++) {
                        if (first) {
                            res = multipoles_asym(k, j, bound, so, passage) *
                                  radial_part_asym(so.val_t->get_space(), k, j, ome, fit, parfit);
                            first = false;
                        } else
                            res = res + multipoles_asym(k, j, bound, so, passage) *
                                            radial_part_asym(so.val_t->get_space(), k, j, ome, fit, parfit);
                    }
                } else {
                    for (int j = int((mm + 1) / 2); j < nbr_coefs(1) - 1; j++) {
                        if (first) {
                            res = multipoles_asym(k, j, bound, so, passage) *
                                  radial_part_asym(so.val_t->get_space(), k, j, ome, fit, parfit);
                            first = false;
                        } else
                            res = res + multipoles_asym(k, j, bound, so, passage) *
                                            radial_part_asym(so.val_t->get_space(), k, j, ome, fit, parfit);
                    }
                }
            }

        return res;
    }
} // namespace Kadath
