/*
    Copyright 2024 Philippe Grandclement

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

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Array/array.hpp"
namespace Kadath {
int mult_cos_1d(int, const double*, double*, int, int) ;
int mult_sin_1d(int, const double*, double*, int, int) ;
int div_sin_1d (int, Array<double>&) ;
int div_cos_1d (int, Array<double>&) ;
int div_x_1d (int, Array<double>&) ;
int mult_x_1d (int, Array<double>&) ;
double integral_1d (int, const Array<double>&) ;
int div_1mx2_1d (int, Array<double>&) ;
double integral_1d(int, const Array<double>&) ;

Val_domain Domain_nucleus_nosym::mult_cos_phi (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;

	res.base = so.base ;
	*res.base.bases_1d[2] = *so.base.bases_1d[2] ;

	Index index_t (res.base.bases_1d[1]->get_dimensions()) ;
	// Inversion in theta :
	do {
		switch ((*so.base.bases_1d[1])(index_t)) {
			case COS :
				res.base.bases_1d[1]->set(index_t) = SIN ;
				break ;
			case SIN :
				res.base.bases_1d[1]->set(index_t) = COS ;
				break ;
				default :
					KADATH_THROW("Unknown case in Domain_nucleus_nosym::mult_cos_phi");
				}
		}
	while (index_t.inc()) ;

	res.base.def = true ;

	res.cf = new Array<double> (so.base.ope_1d(mult_cos_1d, 2, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_nucleus_nosym::mult_sin_phi (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;

	res.base = so.base ;
	*res.base.bases_1d[2] = *so.base.bases_1d[2] ;

	Index index_t (res.base.bases_1d[1]->get_dimensions()) ;
	// Inversion in theta :
	do {
		switch ((*so.base.bases_1d[1])(index_t)) {
			case COS :
				res.base.bases_1d[1]->set(index_t) = SIN ;
				break ;
			case SIN :
				res.base.bases_1d[1]->set(index_t) = COS ;
				break ;
				default :
					KADATH_THROW("Unknown case in Domain_nucleus_nosym::mult_sin_phi");
				}
		}
	while (index_t.inc()) ;

	res.base.def = true ;

	res.cf = new Array<double> (so.base.ope_1d(mult_sin_1d, 2, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_nucleus_nosym::mult_cos_theta (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;
	res.base = so.base ;
	Index index_r (res.base.bases_1d[0]->get_dimensions()) ;
	// Inversion in r :
	do {
		switch ((*so.base.bases_1d[0])(index_r)) {
			case CHEB_EVEN :
				res.base.bases_1d[0]->set(index_r) = CHEB_ODD ;
				break ;
			case CHEB_ODD :
				res.base.bases_1d[0]->set(index_r) = CHEB_EVEN ;
				break ;
			case LEG_EVEN :
				res.base.bases_1d[0]->set(index_r) = LEG_ODD ;
				break ;
			case LEG_ODD :
				res.base.bases_1d[0]->set(index_r) = LEG_EVEN ;
				break ;
				default :
					KADATH_THROW("Unknown case in Domain_nucleus::mult_cos_phi");
				}
		}
	while (index_r.inc()) ;

	res.cf = new Array<double> (so.base.ope_1d(mult_cos_1d, 1, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_nucleus_nosym::mult_sin_theta (const Val_domain& so) const {

	so.coef() ;
	Val_domain res(this) ;
	res.base = so.base ;

	Index index_r (res.base.bases_1d[0]->get_dimensions()) ;
	// Inversion in r :
	do {
		switch ((*so.base.bases_1d[0])(index_r)) {
			case CHEB_EVEN :
				res.base.bases_1d[0]->set(index_r) = CHEB_ODD ;
				break ;
			case CHEB_ODD :
				res.base.bases_1d[0]->set(index_r) = CHEB_EVEN ;
				break ;
			case LEG_EVEN :
				res.base.bases_1d[0]->set(index_r) = LEG_ODD ;
				break ;
			case LEG_ODD :
				res.base.bases_1d[0]->set(index_r) = LEG_EVEN ;
				break ;
				default :
					KADATH_THROW("Unknown case in Domain_nucleus::mult_cos_phi");
				}
		}
	while (index_r.inc()) ;


	res.cf = new Array<double> (so.base.ope_1d(mult_sin_1d, 1, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_nucleus_nosym::div_sin_theta (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;

	res.base = so.base ;
	Index index_r (res.base.bases_1d[0]->get_dimensions()) ;
	// Inversion in r :
	do {
		switch ((*so.base.bases_1d[0])(index_r)) {
			case CHEB_EVEN :
				res.base.bases_1d[0]->set(index_r) = CHEB_ODD ;
				break ;
			case CHEB_ODD :
				res.base.bases_1d[0]->set(index_r) = CHEB_EVEN ;
				break ;
			case LEG_EVEN :
				res.base.bases_1d[0]->set(index_r) = LEG_ODD ;
				break ;
			case LEG_ODD :
				res.base.bases_1d[0]->set(index_r) = LEG_EVEN ;
				break ;
				default :
					KADATH_THROW("Unknown case in Domain_nucleus::mult_cos_phi");
				}
		}
	while (index_r.inc()) ;

	res.cf = new Array<double> (so.base.ope_1d(div_sin_1d, 1, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}


Val_domain Domain_nucleus_nosym::div_x (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;

	res.base= so.base ;

	res.cf = new Array<double> (so.base.ope_1d(div_x_1d, 0, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_nucleus_nosym::div_1mx2 (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;

	res.base= so.base ;

	res.cf = new Array<double> (so.base.ope_1d(div_1mx2_1d, 0, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_nucleus_nosym::mult_r (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;

	res.base= so.base ;

	res.cf = new Array<double> (so.base.ope_1d(mult_x_1d, 0, *so.cf, res.base)) ;
	*res.cf *= alpha ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_nucleus_nosym::div_r (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;

	res.base= so.base ;

	res.cf = new Array<double> (so.base.ope_1d(div_x_1d, 0, *so.cf, res.base)) ;
	*res.cf /= alpha ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_nucleus_nosym::der_r (const Val_domain& so) const {
  return (so.der_var(1)/alpha) ;
}

Val_domain Domain_nucleus_nosym::ddp (const Val_domain& so) const {
  return (so.der_var(3).der_var(3)) ;
}

Val_domain Domain_nucleus_nosym::srdr (const Val_domain& so) const {
  return (div_x(so.der_var(1)) / alpha / alpha) ;
}

Val_domain Domain_nucleus_nosym::laplacian2 (const Val_domain& so, int m) const {
  Val_domain derr (so.der_var(1)/alpha) ;
  Val_domain dert (so.der_var(2)) ;
  Val_domain res (derr.der_var(1)/alpha + div_r(derr + div_r(dert.der_var(2)))) ;
  if (m!=0)
    res -= m * m * div_r(div_r(so.div_sin_theta().div_sin_theta())) ;
  return res ;
}

double Domain_nucleus_nosym::integ_volume(const Val_domain& so) const {
    if (so.check_if_zero())
        return 0.;

    // Volume element: r² sin(θ) dr dθ dφ = mult_r²(mult_sin_θ(f)) * alpha * dr_numerical
    Val_domain integrant(mult_r(mult_r(mult_sin_theta(so))) * alpha);
    integrant.coef();

    double val = 0.;
    // Only k=0 (m=0 phi mode) survives the phi integral (factor 2pi).
    // Theta basis here is the full nosym SIN: mode j is sin(j*theta), with
    //   integral_0^pi sin(j*theta) dtheta = 2/j  (odd j; even j vanish).
    // The nucleus radial basis alternates with theta parity: odd-j carry
    // CHEB_EVEN coefficients (the contributors), even-j carry CHEB_ODD which
    // also have zero theta weight, so skip them (integral_1d has no CHEB_ODD).
    // NB: this differs from the equatorial-symmetric Domain_nucleus, which uses
    // SIN_ODD with weight 2/(2j+1).
    Index pos(nbr_coefs);
    for (int j = 1; j < nbr_coefs(1); j++) {
        pos.set(1) = j;
        if (j % 2 == 0)
            continue;  // even-j SIN modes are z-odd: ∫₀^π sin(jθ)dθ = 0
        int baser = (*integrant.get_base().bases_1d[0])(j, 0);
        if (baser != CHEB_EVEN)
            continue;
        Array<double> cf(nbr_coefs(0));
        for (int i = 0; i < nbr_coefs(0); i++) {
            pos.set(0) = i;
            cf.set(i) = integrant.get_coef(pos);
        }
        val += 2. / double(j) * integral_1d(CHEB_EVEN, cf);
    }

    return val * 2. * M_PI;
}

}
