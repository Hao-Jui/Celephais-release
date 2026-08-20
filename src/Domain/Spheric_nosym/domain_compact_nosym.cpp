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

#include "For_Kadath/Array/headcpp.hpp"

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Val_domain/der_abs_lane_batch.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <algorithm>
#include <array>
#include <optional>
namespace Kadath {
// Standard constructor
Domain_compact_nosym::Domain_compact_nosym (int num, int ttype, double r, const Point& cr, const Dim_array& nbr) : Domain(num, ttype, nbr), alpha(-0.5/r),
								center(cr) {

     assert (nbr.get_ndim()==3) ;
     assert (cr.get_ndim()==3) ;
     do_coloc() ;

}

// Copy constructor
Domain_compact_nosym::Domain_compact_nosym (const Domain_compact_nosym& so) : Domain(so), alpha(so.alpha), center(so.center){}


Domain_compact_nosym::Domain_compact_nosym (int num, BinarySource& source) : Domain(num, source), center(source) {
	alpha = source.read<double>();
	do_coloc() ;
}

// Destructor
Domain_compact_nosym::~Domain_compact_nosym() {}

ostream& Domain_compact_nosym::print (ostream& o) const {
  o << "Compactified domain with no symmetry" << endl ;
  o << "Rmin    = " << -0.5/alpha << endl ;
  o << "Center  = " << center << endl ;
  o << "Nbr pts = " << nbr_points << endl ;
  o << endl ;
  return o ;
}

void Domain_compact_nosym::save(BinarySink& sink) const {
	nbr_points.save(sink) ;
	nbr_coefs.save(sink) ;
	sink.write<int>(ndim);
	sink.write<int>(type_base);
	center.save(sink) ;
	sink.write<double>(alpha);
}


Val_domain Domain_compact_nosym::der_normal (const Val_domain& so, int bound) const {

	Val_domain res (so.der_var(1)) ;
	switch (bound) {
		case INNER_BC :
			res = res.mult_xm1() ;
			res = res.mult_xm1() ;
			res *= -alpha ;
			break ;
		case OUTER_BC :
			res = res.mult_xm1() ;
			res = res.mult_xm1() ;
			res *= -alpha ;
			break ;
		default:
			KADATH_THROW("Unknown boundary case in Domain_compact_nosym::der_normal");
		}
return res ;
}

// Computes the Cartesian coordinates
void Domain_compact_nosym::do_absol () const  {
	for (int i=0 ; i<3 ; i++)
	   assert (coloc[i] != nullptr) ;
	for (int i=0 ; i<3 ; i++)
	   assert (absol[i] == nullptr) ;
	for (int i=0 ; i<3 ; i++) {
	   absol[i] = new Val_domain(this) ;
	   absol[i]->allocate_conf() ;
	   }
	Index index (nbr_points) ;
	do {
		absol[0]->set(index) = (1./alpha/ ((*coloc[0])(index(0))-1)) *
		    sin((*coloc[1])(index(1)))*cos((*coloc[2])(index(2))) + center(1) ;
		absol[1]->set(index) = (1./alpha/ ((*coloc[0])(index(0))-1)) *
		    sin((*coloc[1])(index(1)))*sin((*coloc[2])(index(2))) + center(2) ;
		absol[2]->set(index) = (1./alpha/ ((*coloc[0])(index(0))-1)) * cos((*coloc[1])(index(1))) +
					 center(3) ;
			}
	while (index.inc())  ;
}

// Computes the radius
void Domain_compact_nosym::do_radius () const  {
	for (int i=0 ; i<3 ; i++)
	   assert (coloc[i] != nullptr) ;
	assert (radius == nullptr) ;
	radius = new Val_domain(this) ;
	radius->allocate_conf() ;
	Index index (nbr_points) ;
	do
		radius->set(index) = 1./alpha/ ((*coloc[0])(index(0))-1) ;
	while (index.inc())  ;
}

// Computes the Cartesian coordinates
void Domain_compact_nosym::do_cart () const  {
	for (int i=0 ; i<3 ; i++)
	   assert (coloc[i] != nullptr) ;
	for (int i=0 ; i<3 ; i++)
	   assert (cart[i] == nullptr) ;
	for (int i=0 ; i<3 ; i++) {
	   cart[i] = new Val_domain(this) ;
	   cart[i]->allocate_conf() ;
	   }
	Index index (nbr_points) ;
	do {
		cart[0]->set(index) = (1./alpha/ ((*coloc[0])(index(0))-1)) *
		    sin((*coloc[1])(index(1)))*cos((*coloc[2])(index(2))) + center(1) ;
		cart[1]->set(index) = (1./alpha/ ((*coloc[0])(index(0))-1)) *
		    sin((*coloc[1])(index(1)))*sin((*coloc[2])(index(2))) + center(2) ;
		cart[2]->set(index) = (1./alpha/ ((*coloc[0])(index(0))-1)) * cos((*coloc[1])(index(1))) +
					 center(3) ;
			}
	while (index.inc())  ;
}

// Computes the Cartesian coordinates ovzr radius
void Domain_compact_nosym::do_cart_surr () const  {
	for (int i=0 ; i<3 ; i++)
	   assert (coloc[i] != nullptr) ;
	for (int i=0 ; i<3 ; i++)
	   assert (cart_surr[i] == nullptr) ;
	for (int i=0 ; i<3 ; i++) {
	   cart_surr[i] = new Val_domain(this) ;
	   cart_surr[i]->allocate_conf() ;
	   }
	Index index (nbr_points) ;
	do {
		cart_surr[0]->set(index) = sin((*coloc[1])(index(1)))*cos((*coloc[2])(index(2))) ;
		cart_surr[1]->set(index) =  sin((*coloc[1])(index(1)))*sin((*coloc[2])(index(2))) ;
		cart_surr[2]->set(index) = cos((*coloc[1])(index(1))) ;
			}
	while (index.inc())  ;
}

// Is inside ?
bool Domain_compact_nosym::is_in (const Point& xx, double prec) const {

	assert (xx.get_ndim()==3) ;

	double x_loc = xx(1) - center(1) ;
	double y_loc = xx(2) - center(2) ;
	double z_loc = xx(3) - center(3) ;
	double air_loc = sqrt (x_loc*x_loc + y_loc*y_loc + z_loc*z_loc) ;

	bool res = (1. + 1./(2*alpha*air_loc) >= -prec) ? true : false ;
	return res ;
}

// Converts the absolute coordinates to the numerical ones.
const Point Domain_compact_nosym::absol_to_num(const Point& abs) const {

	assert (is_in(abs)) ;
	Point num(3) ;

	double x_loc = abs(1) - center(1) ;
	double y_loc = abs(2) - center(2) ;
	double z_loc = abs(3) - center(3) ;
	double air = sqrt(x_loc*x_loc+y_loc*y_loc+z_loc*z_loc) ;
	num.set(1) = 1. + 1./air/alpha ;
	double rho = sqrt(x_loc*x_loc+y_loc*y_loc) ;

	num.set(2) = atan2(rho, z_loc) ;
	num.set(3) = atan2 (y_loc, x_loc) ;

	return num ;
}


// Convert absolute coordinates to numerical ones
const Point Domain_compact_nosym::absol_to_num_bound(const Point& abs, int bound) const {

	assert (bound==INNER_BC) ;
	assert (is_in(abs, 1e-3)) ;
	Point num(3) ;

	double x_loc = abs(1) - center(1) ;
	double y_loc = abs(2) - center(2) ;
	double z_loc = abs(3) - center(3) ;
	num.set(1) = -1 ;
	double rho = sqrt(x_loc*x_loc+y_loc*y_loc) ;

	num.set(2) = atan2(rho, z_loc) ;
	num.set(3) = atan2 (y_loc, x_loc) ;

	return num ;
}

double coloc_leg(int,int) ;
void Domain_compact_nosym::do_coloc () {

	switch (type_base) {
		case CHEB_TYPE:
			nbr_coefs = nbr_points ;
			nbr_coefs.set(2) += 2 ;
			del_deriv() ;
			for (int i=0 ; i<ndim ; i++)
				coloc[i] = new Array<double> (nbr_points(i)) ;
			for (int i=0 ; i<nbr_points(0) ; i++)
				coloc[0]->set(i) = -cos(M_PI*i/(nbr_points(0)-1)) ;
			for (int j=0 ; j<nbr_points(1) ; j++)
				coloc[1]->set(j) = M_PI*j/(nbr_points(1)-1) ;
			for (int k=0 ; k<nbr_points(2) ; k++)
				coloc[2]->set(k) = M_PI*2.*k/nbr_points(2) ;
			break ;
		case LEG_TYPE:
			nbr_coefs = nbr_points ;
			nbr_coefs.set(2) += 2 ;
			del_deriv() ;
			for (int i=0 ; i<ndim ; i++)
				coloc[i] = new Array<double> (nbr_points(i)) ;
			for (int i=0 ; i<nbr_points(0) ; i++)
				coloc[0]->set(i) = coloc_leg(i, nbr_points(0)) ;
			for (int j=0 ; j<nbr_points(1) ; j++)
				coloc[1]->set(j) = M_PI*j/(nbr_points(1)-1) ;
			for (int k=0 ; k<nbr_points(2) ; k++)
				coloc[2]->set(k) = M_PI*2.*k/nbr_points(2) ;
			break ;
		default :
			KADATH_THROW("Unknown type of basis in Domain_compact_nosym::do_coloc");
	}
}

// standard base for a symetric function using Chebyshev
void Domain_compact_nosym::set_cheb_base(Base_spectral& base) const {
	int m ;

	assert (type_base == CHEB_TYPE) ;

	base.allocate(nbr_coefs) ;

	Index index (base.bases_1d[0]->get_dimensions()) ;

	base.def=true ;
	base.bases_1d[2]->set(0) = COSSIN ;
	for (int k=0 ; k<nbr_coefs(2) ; k++) {
	        m = (k%2==0) ? k/2 : (k-1)/2 ;
		base.bases_1d[1]->set(k) = (m%2==0) ? COS : SIN ;
		for (int j=0 ; j<nbr_coefs(1) ; j++) {
		    index.set(0) = j ; index.set(1) = k ;
		    base.bases_1d[0]->set(index) = CHEB ;
		 }
	}
}

// standard base for a symetric function using Legendre
void Domain_compact_nosym::set_legendre_base(Base_spectral& base) const {

	int m ;

	assert (type_base == LEG_TYPE) ;

	base.allocate(nbr_coefs) ;

	Index index (base.bases_1d[0]->get_dimensions()) ;

	base.def=true ;
	base.bases_1d[2]->set(0) = COSSIN ;
	for (int k=0 ; k<nbr_coefs(2) ; k++) {
	        m = (k%2==0) ? k/2 : (k-1)/2 ;
		base.bases_1d[1]->set(k) = (m%2==0) ? COS : SIN ;
		for (int j=0 ; j<nbr_coefs(1) ; j++) {
		    index.set(0) = j ; index.set(1) = k ;
		    base.bases_1d[0]->set(index) = LEG ;
		 }
	}
}

// Base for a function anti-symmetric in z, using Chebyshev (compact excludes
// the origin → r-axis has no parity selection; NONSYM unified COSSIN basis
// spans both z-parities, so the anti dispatch is identical to the standard base).
void Domain_compact_nosym::set_anti_cheb_base(Base_spectral& base) const {
	set_cheb_base(base) ;
}

// Base for a function anti-symmetric in z, using Legendre — same delegation.
void Domain_compact_nosym::set_anti_legendre_base(Base_spectral& base) const {
	set_legendre_base(base) ;
}

void Domain_compact_nosym::set_cheb_base_x_cart(Base_spectral& base) const {
	set_cheb_base(base) ;
}

void Domain_compact_nosym::set_cheb_base_y_cart(Base_spectral& base) const {
	set_cheb_base(base) ;
}

void Domain_compact_nosym::set_cheb_base_z_cart(Base_spectral& base) const {
	set_anti_cheb_base(base) ;
}
void Domain_compact_nosym::set_legendre_base_x_cart(Base_spectral& base) const {
	set_legendre_base(base) ;
}

void Domain_compact_nosym::set_legendre_base_y_cart(Base_spectral& base) const {
	set_legendre_base(base) ;
}

void Domain_compact_nosym::set_legendre_base_z_cart(Base_spectral& base) const {
	set_anti_legendre_base(base) ;
}

// sets the value at infinity
void Domain_compact_nosym::set_val_inf (Val_domain& so, double x) const {

	assert (so.get_domain() == this) ;

	so.coef_i() ;
	so.set_in_conf() ;
	Index inf (nbr_points) ;
	inf.set(0) = nbr_points(0)-1 ;
	do
		so.set(inf) = x ;
	while (inf.inc(1, 1)) ;
}

// Computes the derivatives with respect to XYZ as function of the numerical ones.
void Domain_compact_nosym::do_der_abs_from_der_var_lanes(DerAbsLaneBatch& batch) const {
	constexpr std::size_t tile_width = 4;
	for (std::size_t tile_begin = 0; tile_begin < batch.lane_count(); tile_begin += tile_width) {
		const std::size_t tile_size = std::min(tile_width, batch.lane_count() - tile_begin);
		std::array<std::optional<Val_domain>, tile_width> dr;
		std::array<std::optional<Val_domain>, tile_width> sintdr;
		std::array<std::optional<Val_domain>, tile_width> costdt;
		std::array<std::optional<Val_domain>, tile_width> dpssint;
		std::array<std::optional<Val_domain>, tile_width> auxi_x;
		std::array<std::optional<Val_domain>, tile_width> out_x;
		std::array<std::optional<Val_domain>, tile_width> auxi_y;
		std::array<std::optional<Val_domain>, tile_width> out_y;
		std::array<std::optional<Val_domain>, tile_width> auxi_z;
		std::array<std::optional<Val_domain>, tile_width> out_z;

		for (std::size_t offset = 0; offset < tile_size; ++offset)
			dr[offset].emplace(-batch.der_var(tile_begin + offset, 0).mult_xm1());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			sintdr[offset].emplace(dr[offset]->mult_sin_theta());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			costdt[offset].emplace(batch.der_var(tile_begin + offset, 1).mult_cos_theta());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			dpssint[offset].emplace(batch.der_var(tile_begin + offset, 2).div_sin_theta());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			auxi_x[offset].emplace((*sintdr[offset]+*costdt[offset]).mult_cos_phi()
			                       - dpssint[offset]->mult_sin_phi());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			out_x[offset].emplace(alpha*auxi_x[offset]->mult_xm1());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			auxi_y[offset].emplace((*sintdr[offset]+*costdt[offset]).mult_sin_phi()
			                       + dpssint[offset]->mult_cos_phi());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			out_y[offset].emplace(alpha*auxi_y[offset]->mult_xm1());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			auxi_z[offset].emplace(dr[offset]->mult_cos_theta()
			                       - batch.der_var(tile_begin + offset, 1).mult_sin_theta());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			out_z[offset].emplace(alpha*auxi_z[offset]->mult_xm1());

		for (std::size_t offset = 0; offset < tile_size; ++offset) {
			batch.set_der_abs(tile_begin + offset, 0, std::move(*out_x[offset]));
			batch.set_der_abs(tile_begin + offset, 1, std::move(*out_y[offset]));
			batch.set_der_abs(tile_begin + offset, 2, std::move(*out_z[offset]));
		}
	}
}

void Domain_compact_nosym::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const {

	// d/dx :
	Val_domain dr (-der_var[0]->mult_xm1()) ;
	Val_domain sintdr (dr.mult_sin_theta()) ;
	Val_domain costdt (der_var[1]->mult_cos_theta()) ;
	Val_domain dpssint (der_var[2]->div_sin_theta()) ;

	Val_domain auxi_x ((sintdr+costdt).mult_cos_phi() - dpssint.mult_sin_phi()) ;
	der_abs[0] = new Val_domain (alpha*auxi_x.mult_xm1()) ;

	// d/dy :

	Val_domain auxi_y ((sintdr+costdt).mult_sin_phi() + dpssint.mult_cos_phi()) ;
	der_abs[1] = new Val_domain (alpha*auxi_y.mult_xm1()) ;

	// d/dz :
	Val_domain auxi_z (dr.mult_cos_theta()-der_var[1]->mult_sin_theta()) ;
	der_abs[2] = new Val_domain (alpha*auxi_z.mult_xm1()) ;
}

// Rules for basis multiplication
Base_spectral Domain_compact_nosym::mult (const Base_spectral& a, const Base_spectral& b) const {

	assert (a.ndim==3) ;
	assert (b.ndim==3) ;

	Base_spectral res(3) ;
	bool res_def = true ;

	if (!a.def)
		res_def=false ;
	if (!b.def)
		res_def=false ;

	if (res_def) {

	// Base in phi :
	res.bases_1d[2] = std::make_unique<Array<int>>(a.bases_1d[2]->get_dimensions()) ;
	switch ((*a.bases_1d[2])(0)) {
		case COSSIN:
			switch ((*b.bases_1d[2])(0)) {
				case COSSIN:
					res.bases_1d[2]->set(0) = COSSIN ;
					break ;
				default:
					res_def = false ;
					break ;
				}
			break ;
		default:
			res_def = false ;
			break ;
	}


	// Bases in theta :
	// On check l'alternance :
	Index index_1 (a.bases_1d[1]->get_dimensions()) ;
	res.bases_1d[1] = std::make_unique<Array<int>>(a.bases_1d[1]->get_dimensions()) ;
	do {
	switch ((*a.bases_1d[1])(index_1)) {
		case COS:
			switch ((*b.bases_1d[1])(index_1)) {
				case COS:
					res.bases_1d[1]->set(index_1) = (index_1(0)%4<2) ? COS : SIN ;
					break ;
				case SIN:
					res.bases_1d[1]->set(index_1) = (index_1(0)%4<2) ? SIN : COS ;
					break ;
				default:
					res_def = false ;
					break ;
				}
			break ;
		case SIN:
			switch ((*b.bases_1d[1])(index_1)) {
				case COS:
					res.bases_1d[1]->set(index_1) =  (index_1(0)%4<2) ? SIN : COS ;
					break ;
				case SIN:
					res.bases_1d[1]->set(index_1) =  (index_1(0)%4<2) ? COS : SIN ;
					break ;
				default:
					res_def = false ;
					break ;
				}
			break ;
		default:
			res_def = false ;
			break ;
		}
	}
	while (index_1.inc()) ;

	// Base in r :
	Index index_0 (a.bases_1d[0]->get_dimensions()) ;
	res.bases_1d[0] = std::make_unique<Array<int>>(a.bases_1d[0]->get_dimensions()) ;
	do {
	switch ((*a.bases_1d[0])(index_0)) {
		case CHEB:
			switch ((*b.bases_1d[0])(index_0)) {
				case CHEB:
					res.bases_1d[0]->set(index_0) = CHEB ;
					break ;
				default:
					res_def = false ;
					break ;
				}
			break ;
		case LEG:
			switch ((*b.bases_1d[0])(index_0)) {
				case LEG:
					res.bases_1d[0]->set(index_0) = LEG ;
					break ;
				default:
					res_def = false ;
					break ;
				}
			break ;
		default:
			res_def = false ;
			break ;
		}
	}
	while (index_0.inc()) ;
	}

	if (!res_def)
		for (int dim=0 ; dim<a.ndim ; dim++)
			if (res.bases_1d[dim]) {
				res.bases_1d[dim].reset();
				}
	res.def = res_def ;
	return res ;
}


double Domain_compact_nosym::integ (const Val_domain& so, int bound) const {

	Val_domain rrso (mult_r(mult_r(mult_sin_theta(so)))) ;

	double res = 0 ;
	if (!so.check_if_zero()) {
	  Index pcf (nbr_coefs) ;
	  int baset = (*rrso.get_base().bases_1d[1]) (0) ;
	switch (baset) {
	case SIN :
	  for (int j=1 ; j<nbr_coefs(1) ; j++) {
	    if (j%2==0) continue ;   // even-j SIN modes are z-odd: int_0^pi sin(j theta) dtheta = 0
	    pcf.set(1) = j ;
	    res += 2./double(j) * val_boundary(bound, rrso, pcf) ;
	  }
	  break ;

	default :
		KADATH_THROW("Case not yet implemented in Domain_shell::integ");
	}
	res *= 2*M_PI ;
	}
    return res ;
}

int Domain_compact_nosym::give_place_var (char* p) const {
    int res = -1 ;
    if (strcmp(p,"R ")==0)
	res = 0 ;
    if (strcmp(p,"T ")==0)
	res = 1 ;
    if (strcmp(p,"P ")==0)
	res = 2 ;
    return res ;
}}
