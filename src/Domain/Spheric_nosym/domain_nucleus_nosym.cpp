/*
    Copyright 2014 Philippe Grandclement

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
void coef_1d (int, Array<double>&) ;
void coef_i_1d (int, Array<double>&) ;
int der_1d (int, Array<double>&) ;

// Standard constructor
Domain_nucleus_nosym::Domain_nucleus_nosym (int num, int ttype, double r, const Point& cr, const Dim_array& nbr) :
		Domain(num, ttype, nbr), alpha(r),center(cr) {
     assert (nbr.get_ndim()==3) ;
     assert (cr.get_ndim()==3) ;
     do_coloc() ;
}

// Constructor by copy
Domain_nucleus_nosym::Domain_nucleus_nosym (const Domain_nucleus_nosym& so) : Domain(so), alpha(so.alpha), center(so.center) {
}

Domain_nucleus_nosym::Domain_nucleus_nosym (int num, BinarySource& source) : Domain(num, source), center(source) {
	alpha = source.read<double>();
	do_coloc() ;
}

// Destructor
Domain_nucleus_nosym::~Domain_nucleus_nosym() {}

void Domain_nucleus_nosym::save(BinarySink& sink) const {
	nbr_points.save(sink) ;
	nbr_coefs.save(sink) ;
	sink.write<int>(ndim);
	sink.write<int>(type_base);
	center.save(sink) ;
	sink.write<double>(alpha);
}

ostream& Domain_nucleus_nosym::print (ostream& o) const {
  o << "Nucleus nosym" << endl ;
  o << "Rmax    = " << alpha << endl ;
  o << "Center  = " << center << endl ;
  o << "Nbr pts = " << nbr_points << endl ;
  o << endl ;
  return o ;
}


Val_domain Domain_nucleus_nosym::der_normal (const Val_domain& so, int bound) const {

	Val_domain res (so.der_var(1)) ;
	switch (bound) {
		case OUTER_BC :
			res /= alpha ;
			break ;
		default:
			KADATH_THROW("Unknown boundary case in Domain_nucleus_nosym::der_normal");
		}
return res ;
}

// Computes the cartesian coordinates
void Domain_nucleus_nosym::do_absol () const {
	for (int i=0 ; i<3 ; i++)
	   assert (coloc[i] != nullptr) ;
	for (int i=0 ; i<3 ; i++)
	   assert (absol[i] == nullptr) ;
	for (int i=0 ; i<3 ; i++) {
	   absol[i] = new Val_domain(this) ;
	   absol[i]->allocate_conf() ;
	   }
	Index index (nbr_points) ;
	do  {
		absol[0]->set(index) = alpha* ((*coloc[0])(index(0))) *
			 sin((*coloc[1])(index(1)))*cos((*coloc[2])(index(2))) + center(1);
		absol[1]->set(index) = alpha* ((*coloc[0])(index(0))) *
			 sin((*coloc[1])(index(1)))*sin((*coloc[2])(index(2))) + center(2) ;
		absol[2]->set(index) = alpha* ((*coloc[0])(index(0))) * cos((*coloc[1])(index(1))) + center(3) ;
	}
	while (index.inc())  ;

}

// Computes the radius
void Domain_nucleus_nosym::do_radius ()  const {

	for (int i=0 ; i<3 ; i++)
	   assert (coloc[i] != nullptr) ;
	assert (radius == nullptr) ;
	radius = new Val_domain(this) ;
	radius->allocate_conf() ;
	Index index (nbr_points) ;
	do
		radius->set(index) = alpha* ((*coloc[0])(index(0))) ;
		while (index.inc())  ;
}

// Computes the cartesian coordinates
void Domain_nucleus_nosym::do_cart () const {
	for (int i=0 ; i<3 ; i++)
	   assert (coloc[i] != nullptr) ;
	for (int i=0 ; i<3 ; i++)
	   assert (cart[i] == nullptr) ;
	for (int i=0 ; i<3 ; i++) {
	   cart[i] = new Val_domain(this) ;
	   cart[i]->allocate_conf() ;
	   }
	Index index (nbr_points) ;
	do  {
		cart[0]->set(index) = alpha* ((*coloc[0])(index(0))) *
			 sin((*coloc[1])(index(1)))*cos((*coloc[2])(index(2))) + center(1);
		cart[1]->set(index) = alpha* ((*coloc[0])(index(0))) *
			 sin((*coloc[1])(index(1)))*sin((*coloc[2])(index(2))) + center(2) ;
		cart[2]->set(index) = alpha* ((*coloc[0])(index(0))) * cos((*coloc[1])(index(1))) + center(3) ;
	}
	while (index.inc())  ;

}
// Computes the cartesian coordinates over the radius
void Domain_nucleus_nosym::do_cart_surr () const {
	for (int i=0 ; i<3 ; i++)
	   assert (coloc[i] != nullptr) ;
	for (int i=0 ; i<3 ; i++)
	   assert (cart_surr[i] == nullptr) ;
	for (int i=0 ; i<3 ; i++) {
	   cart_surr[i] = new Val_domain(this) ;
	   cart_surr[i]->allocate_conf() ;
	   }
	Index index (nbr_points) ;
	do  {
		cart_surr[0]->set(index) = sin((*coloc[1])(index(1)))*cos((*coloc[2])(index(2))) ;
		cart_surr[1]->set(index) = sin((*coloc[1])(index(1)))*sin((*coloc[2])(index(2)))  ;
		cart_surr[2]->set(index) = cos((*coloc[1])(index(1)))  ;
	}
	while (index.inc())  ;

}
// Is a point inside this domain ?
bool Domain_nucleus_nosym::is_in (const Point& xx, double prec) const {

	assert (xx.get_ndim()==3) ;

	double x_loc = xx(1) - center(1) ;
	double y_loc = xx(2) - center(2) ;
	double z_loc = xx(3) - center(3) ;
	double air_loc = sqrt (x_loc*x_loc + y_loc*y_loc + z_loc*z_loc) ;

	bool res = (air_loc/alpha -1  <= prec) ? true : false ;
	return res ;
}


// Convert absolute coordinates to numerical ones
const Point Domain_nucleus_nosym::absol_to_num(const Point& abs) const {

	assert (is_in(abs)) ;
	Point num(3) ;

	double x_loc = abs(1) - center(1) ;
	double y_loc = abs(2) - center(2) ;
	double z_loc = abs(3) - center(3) ;
	double air = sqrt(x_loc*x_loc+y_loc*y_loc+z_loc*z_loc) ;
	num.set(1) = air/alpha ;
	double rho = sqrt(x_loc*x_loc+y_loc*y_loc) ;

	num.set(2) = atan2(rho, z_loc) ;
	num.set(3) = atan2 (y_loc, x_loc) ;

	return num ;
}


// Convert absolute coordinates to numerical ones
const Point Domain_nucleus_nosym::absol_to_num_bound(const Point& abs, int bound) const {

	assert (bound==OUTER_BC) ;
	assert (is_in(abs, 1e-3)) ;
	Point num(3) ;

	double x_loc = abs(1) - center(1) ;
	double y_loc = abs(2) - center(2) ;
	double z_loc = abs(3) - center(3) ;
	num.set(1) = 1 ;
	double rho = sqrt(x_loc*x_loc+y_loc*y_loc) ;

	num.set(2) = atan2(rho, z_loc) ;
	num.set(3) = atan2 (y_loc, x_loc) ;

	return num ;
}

double coloc_leg_parity(int, int) ;
void Domain_nucleus_nosym::do_coloc () {

	switch (type_base) {
		case CHEB_TYPE:
			nbr_coefs = nbr_points ;
			nbr_coefs.set(2) += 2 ;
			del_deriv() ;
			for (int i=0 ; i<ndim ; i++)
				coloc[i] = new Array<double> (nbr_points(i)) ;
			for (int i=0 ; i<nbr_points(0) ; i++)
				coloc[0]->set(i) = sin(M_PI/2.*i/(nbr_points(0)-1)) ;
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
				coloc[0]->set(i) = coloc_leg_parity(i, nbr_points(0)) ;
			for (int j=0 ; j<nbr_points(1) ; j++)
				coloc[1]->set(j) = M_PI*j/(nbr_points(1)-1) ;
			for (int k=0 ; k<nbr_points(2) ; k++)
				coloc[2]->set(k) = M_PI*2.*k/nbr_points(2) ;
			break ;
		default :
			KADATH_THROW("Unknown type of basis in Domain_nucleus_nosym::do_coloc");
	}
}

// Base for a function, using Chebyshev
void Domain_nucleus_nosym::set_cheb_base(Base_spectral& base) const {

	int m ;

	assert (type_base == CHEB_TYPE) ;
	base.allocate(nbr_coefs) ;

	Index index(base.bases_1d[0]->get_dimensions()) ;

	base.def=true ;
	base.bases_1d[2]->set(0) = COSSIN ;
	for (int k=0 ; k<nbr_coefs(2) ; k++) {
	        m = (k%2==0) ? k/2 : (k-1)/2 ;
		base.bases_1d[1]->set(k) = (m%2==0) ? COS : SIN ;
		for (int j=0 ; j<nbr_coefs(1) ; j++) {
		    index.set(0) = j ; index.set(1) = k ;
		    base.bases_1d[0]->set(index) = (j%2==0) ? CHEB_EVEN : CHEB_ODD ;
		 }
	}
}

// Base for a function, using Legendre
void Domain_nucleus_nosym::set_legendre_base(Base_spectral& base) const  {

	int m ;

	assert (type_base == LEG_TYPE) ;
	base.allocate(nbr_coefs) ;

	Index index(base.bases_1d[0]->get_dimensions()) ;

	base.def = true ;
	base.bases_1d[2]->set(0) = COSSIN ;
	for (int k=0 ; k<nbr_coefs(2) ; k++) {
	        m = (k%2==0) ? k/2 : (k-1)/2 ;
		base.bases_1d[1]->set(k) = (m%2==0) ? COS : SIN ;
		for (int j=0 ; j<nbr_coefs(1) ; j++) {
		    index.set(0) = j ; index.set(1) = k ;
		    base.bases_1d[0]->set(index) = (j%2==0) ? LEG_EVEN : LEG_ODD ;
		 }
	}
 }

void Domain_nucleus_nosym::set_anti_cheb_base(Base_spectral& base) const {
	set_cheb_base(base) ;
}

void Domain_nucleus_nosym::set_anti_legendre_base(Base_spectral& base) const {
	set_legendre_base(base) ;
}

void Domain_nucleus_nosym::set_cheb_base_x_cart(Base_spectral& base) const {
	set_cheb_base(base) ;
}

void Domain_nucleus_nosym::set_cheb_base_y_cart(Base_spectral& base) const {
	set_cheb_base(base) ;
}

void Domain_nucleus_nosym::set_cheb_base_z_cart(Base_spectral& base) const {
	set_anti_cheb_base(base) ;
}
void Domain_nucleus_nosym::set_legendre_base_x_cart(Base_spectral& base) const {
	set_legendre_base(base) ;
}

void Domain_nucleus_nosym::set_legendre_base_y_cart(Base_spectral& base) const {
	set_legendre_base(base) ;
}

void Domain_nucleus_nosym::set_legendre_base_z_cart(Base_spectral& base) const {
	set_anti_legendre_base(base) ;
}

// Computes the derivativeswith respect to XYZ as a function of the numerical ones.
void Domain_nucleus_nosym::do_der_abs_from_der_var_lanes(DerAbsLaneBatch& batch) const {
	constexpr std::size_t tile_width = 4;
	for (std::size_t tile_begin = 0; tile_begin < batch.lane_count(); tile_begin += tile_width) {
		const std::size_t tile_size = std::min(tile_width, batch.lane_count() - tile_begin);
		std::array<std::optional<Val_domain>, tile_width> sintdr;
		std::array<std::optional<Val_domain>, tile_width> dtsr;
		std::array<std::optional<Val_domain>, tile_width> dpsr;
		std::array<std::optional<Val_domain>, tile_width> costdtsr;
		std::array<std::optional<Val_domain>, tile_width> dpsrssint;
		std::array<std::optional<Val_domain>, tile_width> out_x;
		std::array<std::optional<Val_domain>, tile_width> out_y;
		std::array<std::optional<Val_domain>, tile_width> out_z;

		for (std::size_t offset = 0; offset < tile_size; ++offset)
			sintdr[offset].emplace(batch.der_var(tile_begin + offset, 0).mult_sin_theta()/alpha);
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			dtsr[offset].emplace(batch.der_var(tile_begin + offset, 1).div_x()/alpha);
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			dpsr[offset].emplace(batch.der_var(tile_begin + offset, 2).div_x()/alpha);
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			costdtsr[offset].emplace(dtsr[offset]->mult_cos_theta());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			dpsrssint[offset].emplace(dpsr[offset]->div_sin_theta());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			out_x[offset].emplace((*sintdr[offset]+*costdtsr[offset]).mult_cos_phi()
			                      - dpsrssint[offset]->mult_sin_phi());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			out_y[offset].emplace((*sintdr[offset]+*costdtsr[offset]).mult_sin_phi()
			                      + dpsrssint[offset]->mult_cos_phi());
		for (std::size_t offset = 0; offset < tile_size; ++offset)
			out_z[offset].emplace(batch.der_var(tile_begin + offset, 0).mult_cos_theta()/alpha
			                      - dtsr[offset]->mult_sin_theta());

		for (std::size_t offset = 0; offset < tile_size; ++offset) {
			batch.set_der_abs(tile_begin + offset, 0, std::move(*out_x[offset]));
			batch.set_der_abs(tile_begin + offset, 1, std::move(*out_y[offset]));
			batch.set_der_abs(tile_begin + offset, 2, std::move(*out_z[offset]));
		}
	}
}

void Domain_nucleus_nosym::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const {

	// d/dx :
	Val_domain sintdr (der_var[0]->mult_sin_theta()/alpha) ;
	Val_domain dtsr (der_var[1]->div_x()/alpha) ;
	Val_domain dpsr (der_var[2]->div_x()/alpha) ;
	Val_domain costdtsr (dtsr.mult_cos_theta()) ;
	Val_domain dpsrssint (dpsr.div_sin_theta()) ;

	der_abs[0] = new Val_domain ((sintdr+costdtsr).mult_cos_phi() - dpsrssint.mult_sin_phi()) ;

	// d/dy :
	der_abs[1] = new Val_domain ((sintdr+costdtsr).mult_sin_phi() + dpsrssint.mult_cos_phi()) ;
	// d/dz :
	der_abs[2] = new Val_domain (der_var[0]->mult_cos_theta()/alpha - dtsr.mult_sin_theta()) ;
}

// Rules for the multiplication of two basis.
Base_spectral Domain_nucleus_nosym::mult (const Base_spectral& a, const Base_spectral& b) const {

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
		case CHEB_EVEN:
			switch ((*b.bases_1d[0])(index_0)) {
				case CHEB_EVEN:
					res.bases_1d[0]->set(index_0) = (index_0(0)%2<1) ? CHEB_EVEN : CHEB_ODD ;
					break ;
				case CHEB_ODD:
					res.bases_1d[0]->set(index_0) = (index_0(0)%2<1) ? CHEB_ODD : CHEB_EVEN  ;
					break ;
				default:
					res_def = false ;
					break ;
				}
			break ;
		case CHEB_ODD:
			switch ((*b.bases_1d[0])(index_0)) {
				case CHEB_EVEN:
					res.bases_1d[0]->set(index_0) = (index_0(0)%2<1) ? CHEB_ODD : CHEB_EVEN  ;
					break ;
				case CHEB_ODD:
					res.bases_1d[0]->set(index_0) = (index_0(0)%2<1) ? CHEB_EVEN : CHEB_ODD ;
					break ;
				default:
					res_def = false ;
					break ;
				}
			break ;
		case LEG_EVEN:
			switch ((*b.bases_1d[0])(index_0)) {
				case LEG_EVEN:
					res.bases_1d[0]->set(index_0) = (index_0(0)%2<1) ? LEG_EVEN : LEG_ODD  ;
					break ;
				case LEG_ODD:
					res.bases_1d[0]->set(index_0) = (index_0(0)%2<1) ? LEG_ODD : LEG_EVEN ;
					break ;
				default:
					res_def = false ;
					break ;
				}
			break ;
		case LEG_ODD:
			switch ((*b.bases_1d[0])(index_0)) {
				case LEG_EVEN:
					res.bases_1d[0]->set(index_0) = (index_0(0)%2<1) ? LEG_ODD : LEG_EVEN  ;
					break ;
				case LEG_ODD:
					res.bases_1d[0]->set(index_0) = (index_0(0)%2<1) ? LEG_EVEN : LEG_ODD ;
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

int Domain_nucleus_nosym::give_place_var (char* p) const {
    int res = -1 ;
    if (strcmp(p,"R ")==0)
	res = 0 ;
    if (strcmp(p,"T ")==0)
	res = 1 ;
    if (strcmp(p,"P ")==0)
	res = 2 ;
    return res ;
}

}
