/*
    Copyright 2025 Philippe Grandclement

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
#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath {
int Domain_bispheric_rect_nosym::nbr_unknowns_val_domain (const Val_domain& so) const {

	int res = 0 ;
	assert (basep==COSSIN) ;

	Index pos (nbr_coefs) ;
	do {
		bool indic = true ;
		// Check true coefs in phi
		if ((pos(2)==1) || (pos(2)==nbr_coefs(2)-1))
			indic = false ;

		int basechi = (*so.get_base().bases_1d[1]) (pos(2)) ;
		// No last coef if odd
		if (((basechi==CHEB_ODD) || (basechi==LEG_ODD)) && (pos(1)==nbr_coefs(1)-1))
			indic = false ;

		//Galerkin for even ones
		if (((basechi==CHEB_EVEN) || (basechi==LEG_EVEN)) && (pos(2)!=0) && (pos(1)==0))
			indic = false ;

		if (indic)
			res ++ ;
	}
	while (pos.inc()) ;

	return res ;
}

int Domain_bispheric_rect_nosym::nbr_unknowns (const Tensor& tt, int dom) const {

	// Check right domain
	assert (tt.get_space().get_domain(dom)==this) ;

	int res = 0 ;
	int val = tt.get_valence() ;
	switch (val) {
		case 0 :
			res += nbr_unknowns_val_domain (tt()(dom)) ;
			break ;
		case 1 : {
			bool found = false ;
			// Cartesian basis
			if (tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) {
				res += nbr_unknowns_val_domain (tt(1)(dom)) ;
				res += nbr_unknowns_val_domain (tt(2)(dom)) ;
				res += nbr_unknowns_val_domain (tt(3)(dom)) ;
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of vector Domain_bispheric_rect_nosym::nbr_unknowns" << endl ;
				abort() ;
			}
		}
		    break ;
		case 2 : {
			bool found = false ;
			// Cartesian basis and symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==6)) {
				res += nbr_unknowns_val_domain (tt(1,1)(dom)) ;
				res += nbr_unknowns_val_domain (tt(1,2)(dom)) ;
				res += nbr_unknowns_val_domain (tt(1,3)(dom)) ;
				res += nbr_unknowns_val_domain (tt(2,2)(dom)) ;
				res += nbr_unknowns_val_domain (tt(2,3)(dom)) ;
				res += nbr_unknowns_val_domain (tt(3,3)(dom)) ;
				found = true ;
			}
			// Cartesian basis and not symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==9)) {
				for (int i=1 ; i<=3 ; i++)
				  for (int j=1 ; j<=3 ; j++)
				    res += nbr_unknowns_val_domain (tt(i,j)(dom)) ;
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of 2-tensor Domain_bispheric_rect_nosym::nbr_unknowns" << endl ;
				abort() ;
			}
		}
		break ;
		default :
			cerr << "Valence " << val << " not implemented in Domain_bispheric_rect_nosym::nbr_unknowns" << endl ;
			break ;
	}
	return res ;
}}
