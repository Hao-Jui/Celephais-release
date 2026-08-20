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
    Array<double> legendre_norme(int, int);

    Term_eq Domain_shell::der_multipoles_sym(int k, int j, int bound, const Term_eq& so,
                                             const Array<double>& passage) const
    {

        // Check if so is a tensor
        if (so.type_data != TERM_T) {
            KADATH_THROW("Domain_shell::der_multipoles_sym only defined for a tensor");
        }
        int valence = so.val_t->get_valence();
        if (valence != 0) {
            KADATH_THROW("Domain_shell::der_multipoles_sym only defined with respect to a component (i.e. valence must be 0)");
        }
        assert((bound == INNER_BC) || (bound == OUTER_BC));

        bool doder = (so.der_t == nullptr) ? false : true;
        int dom = so.get_dom();

        Index pos(passage.dimensions);
        Index pos_bound(nbr_coefs);

        double alm = 0;
        double alm_der = 0;

        Val_domain dval((*so.val_t)()(dom).der_r());
        Val_domain* pdder = (doder) ? new Val_domain((*so.der_t)()(dom).der_r()) : nullptr;

        int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
        pos.set(0) = mm;
        pos_bound.set(2) = k;
        if (mm % 2 == 0) {
            // Case m even
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1); i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, dval, pos_bound);
                if (doder)
                    alm_der += passage(pos) * val_boundary(bound, *pdder, pos_bound);
            }
        } else {
            // Case m odd
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1) - 1; i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, dval, pos_bound);
                if (doder)
                    alm_der += passage(pos) * val_boundary(bound, *pdder, pos_bound);
            }
        }

        if (pdder != nullptr)
            delete pdder;

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
                    der.set(pp) =
                        (k % 2 == 0) ? alm_der * cos(mm * phi(pp(2))) * leg(2 * (j - int((mm - 1) / 2)), 2 * pp(1)) :

                                     alm_der * sin(mm * phi(pp(2))) * leg(2 * (j - int((mm - 1) / 2)), 2 * pp(1));
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

    Term_eq Domain_shell::der_multipoles_asym(int k, int j, int bound, const Term_eq& so,
                                              const Array<double>& passage) const
    {

        // Check if so is a tensor
        if (so.type_data != TERM_T) {
            KADATH_THROW("Domain_shell::der_multipoles_sym only defined for a tensor");
        }
        int valence = so.val_t->get_valence();
        if (valence != 0) {
            KADATH_THROW("Domain_shell::der_multipoles_sym only defined with respect to a component (i.e. valence must be 0)");
        }
        assert((bound == INNER_BC) || (bound == OUTER_BC));

        bool doder = (so.der_t == nullptr) ? false : true;
        int dom = so.get_dom();

        Index pos(passage.dimensions);
        Index pos_bound(nbr_coefs);

        double alm = 0;
        double alm_der = 0;

        Val_domain dval((*so.val_t)()(dom).der_r());
        Val_domain* pdder = (doder) ? new Val_domain((*so.der_t)()(dom).der_r()) : nullptr;

        int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
        pos.set(0) = mm;
        pos_bound.set(2) = k;
        if (mm % 2 == 0) {
            // Case m even
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1) - 1; i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, dval, pos_bound);
                if (doder)
                    alm_der += passage(pos) * val_boundary(bound, *pdder, pos_bound);
            }
        } else {
            // Case m odd
            pos.set(1) = j;
            for (int i = 0; i < nbr_coefs(1) - 1; i++) {
                pos.set(2) = i;
                pos_bound.set(1) = i;
                alm += passage(pos) * val_boundary(bound, dval, pos_bound);
                if (doder)
                    alm_der += passage(pos) * val_boundary(bound, *pdder, pos_bound);
            }
        }

        if (pdder != nullptr)
            delete pdder;

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

    Term_eq Domain_shell::der_radial_part_sym(const Space& space, int k, int j, const Term_eq& ome,
                                              Term_eq (*fit)(const Space&, int, int, const Term_eq&, const Param&),
                                              const Param& parfit) const
    {

        // Check if omega is a double
        if (ome.type_data != TERM_D) {
            KADATH_THROW("Omega must be a double in Domain_shell::der_radial_part_sym");
        }

        // Loop on dimensions of alm :
        int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
        int ll = (mm % 2 == 0) ? 2 * j : 2 * j + 1;

        Term_eq fonction(fit(space, mm, ll, ome, parfit));

        int dom = fonction.get_dom();
        Val_domain val(fonction.get_val_t()()(dom).der_r());
        Index pos(nbr_coefs);
        double resval = val_boundary(OUTER_BC, val, pos);

        bool doder = (fonction.der_t == nullptr) ? false : true;
        if (doder) {
            Val_domain der(fonction.get_der_t()()(dom).der_r());
            double resder = val_boundary(OUTER_BC, val, pos);
            return Term_eq(dom, resval, resder);
        } else {
            return Term_eq(dom, resval);
        }
    }

    Term_eq Domain_shell::der_radial_part_asym(const Space& space, int k, int j, const Term_eq& ome,
                                               Term_eq (*fit)(const Space&, int, int, const Term_eq&, const Param&),
                                               const Param& parfit) const
    {

        // Check if omega is a double
        if (ome.type_data != TERM_D) {
            KADATH_THROW("Omega must be a double in Domain_shell::der_radial_part_sym");
        }

        // Loop on dimensions of alm :
        int mm = (k % 2 == 0) ? int(k / 2) : int((k - 1) / 2);
        int ll = (mm % 2 == 0) ? 2 * j + 1 : 2 * j;

        Term_eq fonction(fit(space, mm, ll, ome, parfit));

        int dom = fonction.get_dom();
        Val_domain val(fonction.get_val_t()()(dom).der_r());
        Index pos(nbr_coefs);
        double resval = val_boundary(OUTER_BC, val, pos);

        bool doder = (fonction.der_t == nullptr) ? false : true;
        if (doder) {
            Val_domain der(fonction.get_der_t()()(dom).der_r());
            double resder = val_boundary(OUTER_BC, val, pos);
            return Term_eq(dom, resval, resder);
        } else {
            return Term_eq(dom, resval);
        }
    }

    Term_eq Domain_shell::der_harmonics_sym(const Term_eq& so, const Term_eq& ome, int bound,
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
                        int ll = 2 * j;
                        //     if ((mm<=2) && (ll<=2)) {
                        if (first) {
                            res = der_multipoles_sym(k, j, bound, so, passage) /
                                  der_radial_part_sym(so.val_t->get_space(), k, j, ome, fit, parfit) *
                                  fit(so.val_t->get_space(), mm, ll, ome, parfit);
                            first = false;
                        } else
                            res = res + der_multipoles_sym(k, j, bound, so, passage) /
                                            der_radial_part_sym(so.val_t->get_space(), k, j, ome, fit, parfit) *
                                            fit(so.val_t->get_space(), mm, ll, ome, parfit);
                    } //}
                } else {
                    for (int j = int((mm - 1) / 2); j < nbr_coefs(1) - 1; j++) {
                        int ll = 2 * j + 1;
                        //    if ((mm<=2) && (ll<=2)) {
                        if (first) {
                            res = der_multipoles_sym(k, j, bound, so, passage) /
                                  der_radial_part_sym(so.val_t->get_space(), k, j, ome, fit, parfit) *
                                  fit(so.val_t->get_space(), mm, ll, ome, parfit);
                            first = false;
                        } else
                            res = res + der_multipoles_sym(k, j, bound, so, passage) /
                                            der_radial_part_sym(so.val_t->get_space(), k, j, ome, fit, parfit) *
                                            fit(so.val_t->get_space(), mm, ll, ome, parfit);
                    } //}
                }
            }
        return res;
    }

    Term_eq Domain_shell::der_harmonics_asym(const Term_eq& so, const Term_eq& ome, int bound,
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
                        int ll = 2 * j + 1;
                        //   if ((mm<=2) && (ll<=2)) {
                        if (first) {
                            res = der_multipoles_asym(k, j, bound, so, passage) /
                                  der_radial_part_asym(so.val_t->get_space(), k, j, ome, fit, parfit) *
                                  fit(so.val_t->get_space(), mm, ll, ome, parfit);
                            first = false;
                        } else
                            res = res + der_multipoles_asym(k, j, bound, so, passage) /
                                            der_radial_part_asym(so.val_t->get_space(), k, j, ome, fit, parfit) *
                                            fit(so.val_t->get_space(), mm, ll, ome, parfit);
                    } //}
                } else {
                    for (int j = int((mm + 1) / 2); j < nbr_coefs(1) - 1; j++) {
                        int ll = 2 * j;
                        //    if ((mm<=2) && (ll<=2)) {
                        if (first) {
                            res = der_multipoles_asym(k, j, bound, so, passage) /
                                  der_radial_part_asym(so.val_t->get_space(), k, j, ome, fit, parfit) *
                                  fit(so.val_t->get_space(), mm, ll, ome, parfit);
                            first = false;
                        } else
                            res = res + der_multipoles_asym(k, j, bound, so, passage) /
                                            der_radial_part_asym(so.val_t->get_space(), k, j, ome, fit, parfit) *
                                            fit(so.val_t->get_space(), mm, ll, ome, parfit);
                    } //}
                }
            }
        return res;
    }
} // namespace Kadath
