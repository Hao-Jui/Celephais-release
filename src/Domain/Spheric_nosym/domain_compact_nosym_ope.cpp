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
int div_xm1_1d (int, Array<double>&) ;
int mult_xm1_1d (int, Array<double>&) ;
double integral_1d(int, const Array<double>&) ;

Val_domain Domain_compact_nosym::mult_cos_phi (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;

	res.base = so.base ;
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
					KADATH_THROW("Unknown case in Domain_compact_nosym::mult_cos_phi");
				}
		}
	while (index_t.inc()) ;

	res.base.def = true ;

	res.cf = new Array<double> (so.base.ope_1d(mult_cos_1d, 2, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_compact_nosym::mult_sin_phi (const Val_domain& so) const {

	so.coef() ;
	Val_domain res(this) ;

	res.base = so.base ;
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
					KADATH_THROW("Unknown case in Domain_compact_nosym::mult_cos_phi");
				}
		}
	while (index_t.inc()) ;
	res.base.def = true ;

	res.cf = new Array<double> (so.base.ope_1d(mult_sin_1d, 2, *so.cf, res.base)) ;
	res.in_coef = true ;

	return res ;
}

Val_domain Domain_compact_nosym::mult_cos_theta (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;
	res.base = so.base ;

	res.cf = new Array<double> (so.base.ope_1d(mult_cos_1d, 1, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_compact_nosym::mult_sin_theta (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;
	res.base= so.base ;

	res.cf = new Array<double> (so.base.ope_1d(mult_sin_1d, 1, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_compact_nosym::div_sin_theta (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;
	res.base = so.base ;

	res.cf = new Array<double> (so.base.ope_1d(div_sin_1d, 1, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_compact_nosym::div_cos_theta (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;
	res.base = so.base ;

	res.cf = new Array<double> (so.base.ope_1d(div_cos_1d, 1, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_compact_nosym::mult_xm1 (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;
	res.base= so.base ;

	res.cf = new Array<double> (so.base.ope_1d(mult_xm1_1d, 0, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_compact_nosym::div_xm1 (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(this) ;
	res.base = so.base ;

	res.cf = new Array<double> (so.base.ope_1d(div_xm1_1d, 0, *so.cf, res.base)) ;
	res.in_coef = true ;
	return res ;
}

Val_domain Domain_compact_nosym::mult_r (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(div_xm1(so)) ;
	res /= alpha ;
	return res ;
}

Val_domain Domain_compact_nosym::div_r (const Val_domain& so) const {
	so.coef() ;
	Val_domain res(mult_xm1(so)) ;
	res *= alpha ;
	return res ;
}

Val_domain Domain_compact_nosym::der_r (const Val_domain& so) const {
  return (-alpha*so.der_var(1).mult_xm1().mult_xm1()) ;
}

Val_domain Domain_compact_nosym::der_r_rtwo (const Val_domain& so) const {
  return (-so.der_var(1)/alpha) ;
}

Val_domain Domain_compact_nosym::dt (const Val_domain& so) const {
  return (so.der_var(2)) ;
}

Val_domain Domain_compact_nosym::ddp (const Val_domain& so) const {
  return (so.der_var(3).der_var(3)) ;
}


Val_domain Domain_compact_nosym::der_partial_var (const Val_domain& so, int which_var) const {

	switch (which_var) {
		case 0 :
			return so.der_r() ;
			break ;
		case 1 :
			return so.der_var(2) ;
			break ;
		case 2 :
			return so.der_var(3) ;
		default:
			KADATH_THROW("Unknown variable in Domain_compact_nosym::der_partial_var");
		}
}

Val_domain Domain_compact_nosym::laplacian2 (const Val_domain& so, int m) const {
  Val_domain derr (-alpha*so.der_var(1).mult_xm1().mult_xm1()) ;
  Val_domain dderr (-alpha*derr.der_var(1).mult_xm1().mult_xm1()) ;
  Val_domain dert (so.der_var(2)) ;
  Val_domain res (dderr + div_r(derr + div_r(dert.der_var(2)))) ;
  if (m!=0)
    res -= m * m * div_r(div_r(so.div_sin_theta().div_sin_theta())) ;
  return res ;
}

double Domain_compact_nosym::integ_volume(const Val_domain& so) const {

	if (so.check_if_zero())
		return 0 ;
	else {

		Val_domain integrant(-mult_r(mult_r(mult_r(mult_r(mult_sin_theta(so))))) * alpha) ;
		integrant.coef() ;

		double val = 0 ;
		// Only k = 0. NONSYM SIN has l_quant = j; even-j modes integrate to zero.
		[[maybe_unused]] int baset = (*integrant.get_base().bases_1d[1])(0) ;
		assert(baset == SIN) ;
		Index pos(nbr_coefs) ;
		for (int j=1 ; j<nbr_coefs(1) ; j+=2) {
			pos.set(1) = j ;
			[[maybe_unused]] int baser = (*integrant.get_base().bases_1d[0])(j, 0) ;
			assert(baser == CHEB) ;

			Array<double> cf(nbr_coefs(0)) ;
			for (int i=0 ; i<nbr_coefs(0) ; i++) {
				pos.set(0) = i ;
				cf.set(i) = integrant.get_coef(pos) ;
			}
			val += 2. / double(j) * integral_1d(CHEB, cf) ;
		}

		return val * 2 * M_PI ;
	}
}
}
