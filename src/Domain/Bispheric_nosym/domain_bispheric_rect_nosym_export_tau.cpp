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

#include <cstddef>

namespace Kadath {
void Domain_bispheric_rect_nosym::export_tau_val_domain (const Val_domain& so, int order, Array<double>& sec, int& pos_sec, int ncond) const {

	if (so.check_if_zero())
		pos_sec += ncond ;
	else {
	int forgot_chi = 0;
	int forgot_eta = 0 ;
	switch (order) {
	  case 0 :
	      forgot_chi = 0 ;
	      forgot_eta = 0 ;
	      break ;
	  case 1 :
	      forgot_chi = 1 ;
	      forgot_eta = 1 ;
	      break ;
	  case 2 :
	      forgot_chi = 1 ;
	      forgot_eta = 2 ;
	      break ;
	  default:
	      cerr << "Unknown order in Domain_bispheric_rect_nosym::export_tau_val_domain" << endl ;
	      break ;
	}
	so.coef() ;
	const double* const coefficient_data = so.cf->get_data() ;
	double* const result_data = sec.get_data() ;
	const std::size_t phi_size = static_cast<std::size_t>(nbr_coefs(2)) ;
	const std::size_t radial_stride = static_cast<std::size_t>(nbr_coefs(1)) * phi_size ;

	// Loop on phi :
	for (int k=0 ; k<nbr_coefs(2)-1 ; k++)
	if (k!=1) {
		// Loop on chi :
		int basechi =  (*so.get_base().bases_1d[1]) (k) ;
		// bound for j
		int maxj = nbr_coefs(1)-forgot_chi ;
		// case for odd
		if ((basechi==CHEB_ODD) || (basechi==LEG_ODD))
			maxj -- ;
		for (int j=0 ; j<maxj ; j++) {
			const std::size_t row_offset = static_cast<std::size_t>(j) * phi_size +
				static_cast<std::size_t>(k) ;

			// Loop on eta ;
			for (int i=0 ; i<nbr_coefs(0)-forgot_eta ; i++) {
				const std::size_t coefficient_offset = static_cast<std::size_t>(i) * radial_stride +
					row_offset ;

				// even case
				if ((basechi==CHEB_EVEN) || (basechi==LEG_EVEN)) {
					if (k==0) {
						// The ones without regularity issues
							result_data[pos_sec] = coefficient_data[coefficient_offset] ;
							pos_sec ++ ;
					}
					else if (j!=0) {
						// Regularity on the axis thanks to Galerkin
						// Factor fo Galerkin different for Legendre or Chebyshev:
						double factor_galerkin ;
						switch (type_base) {
							case CHEB_TYPE :
								factor_galerkin = (j%2==1) ? -2. : 2. ;
								break ;
							case LEG_TYPE :
								factor_galerkin = -double(4*j+1) ;
								for (int jj=1 ; jj<=j ; jj++)
									factor_galerkin *= -double(2*jj-1)/double(2*jj) ;
								break ;
							default :
						cerr << "Unknown type of basis in Domain_bispheric_rect_nosym::export_tau_val_domain" << endl ;
										abort() ;
								}
							const std::size_t anchor_offset = static_cast<std::size_t>(i) * radial_stride +
								static_cast<std::size_t>(k) ;
							result_data[pos_sec] = coefficient_data[coefficient_offset] +
								factor_galerkin * coefficient_data[anchor_offset] ;
							pos_sec ++ ;
						}
				}

				// odd case
				if ((basechi==CHEB_ODD) || (basechi==LEG_ODD)) {
					result_data[pos_sec] = coefficient_data[coefficient_offset] ;
					pos_sec ++ ;
				}
				}
			}
		}
      }
}

void Domain_bispheric_rect_nosym::export_tau (const Tensor& tt, int dom, int order, Array<double>& res, int& pos_res, const Array<int>& ncond,
										int n_cmp, Array<int>** p_cmp) const {
	int val = tt.get_valence() ;
	switch (val) {
		case 0 :
			export_tau_val_domain (tt()(dom), order, res, pos_res, ncond(0)) ;

			break ;
		case 1 : {
			bool found = false ;
			// Cartesian basis
			if (tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) {
				if (n_cmp==-1) {
					export_tau_val_domain (tt(1)(dom), order, res, pos_res, ncond(0)) ;
					export_tau_val_domain (tt(2)(dom), order, res, pos_res, ncond(1)) ;
					export_tau_val_domain (tt(3)(dom), order, res, pos_res, ncond(2)) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if ((*p_cmp[i])(0)==1)
						export_tau_val_domain (tt(1)(dom), order, res, pos_res, ncond(i)) ;
					if ((*p_cmp[i])(0)==2)
						export_tau_val_domain (tt(2)(dom), order, res, pos_res, ncond(i)) ;
					if ((*p_cmp[i])(0)==3)
						export_tau_val_domain (tt(3)(dom), order, res, pos_res, ncond(i)) ;
				}
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of vector Domain_bispheric_rect_nosym::export_tau" << endl ;
				abort() ;
			}
		}
			break ;
		case 2 : {
			bool found = false ;
			// Cartesian basis and symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==6)) {
				if (n_cmp==-1) {
					export_tau_val_domain (tt(1,1)(dom), order, res, pos_res, ncond(0)) ;
					export_tau_val_domain (tt(1,2)(dom), order, res, pos_res, ncond(1)) ;
					export_tau_val_domain (tt(1,3)(dom), order, res, pos_res, ncond(2)) ;
					export_tau_val_domain (tt(2,2)(dom), order, res, pos_res, ncond(3)) ;
					export_tau_val_domain (tt(2,3)(dom), order, res, pos_res, ncond(4)) ;
					export_tau_val_domain (tt(3,3)(dom), order, res, pos_res, ncond(5)) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain (tt(1, 1)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain (tt(1, 2)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(1, 3)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain (tt(2, 2)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(2, 3)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(3, 3)(dom), order, res, pos_res, ncond(i)) ;
				}
				found = true ;
			}
			// Cartesian basis and not symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==9)) {
				if (n_cmp==-1) {
					export_tau_val_domain (tt(1,1)(dom), order, res, pos_res, ncond(0)) ;
					export_tau_val_domain (tt(1,2)(dom), order, res, pos_res, ncond(1)) ;
					export_tau_val_domain (tt(1,3)(dom), order, res, pos_res, ncond(2)) ;
					export_tau_val_domain (tt(2,1)(dom), order, res, pos_res, ncond(3)) ;
					export_tau_val_domain (tt(2,2)(dom), order, res, pos_res, ncond(4)) ;
					export_tau_val_domain (tt(2,3)(dom), order, res, pos_res, ncond(5)) ;
					export_tau_val_domain (tt(3,1)(dom), order, res, pos_res, ncond(6)) ;
					export_tau_val_domain (tt(3,2)(dom), order, res, pos_res, ncond(7)) ;
					export_tau_val_domain (tt(3,3)(dom), order, res, pos_res, ncond(8)) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain (tt(1, 1)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain (tt(1, 2)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(1, 3)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain (tt(2, 1)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain (tt(2, 2)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(2, 3)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain (tt(3, 1)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain (tt(3, 2)(dom), order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(3, 3)(dom), order, res, pos_res, ncond(i)) ;
				}
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of 2-tensor Domain_bispheric_rect_nosym::export_tau" << endl ;
				abort() ;
			}
		}
			break ;
		default :
			cerr << "Valence " << val << " not implemented in Domain_bispheric_rect_nosym::export_tau" << endl ;
			break ;
	}
}}
