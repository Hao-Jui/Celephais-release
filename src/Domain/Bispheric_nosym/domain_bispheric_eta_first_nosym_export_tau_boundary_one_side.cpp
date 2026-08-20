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
void Domain_bispheric_eta_first_nosym::export_tau_val_domain_boundary_one_side (const Val_domain& so, int bound, Array<double>& sec, int& pos_sec, int ncond) const {

	if (so.check_if_zero())
		pos_sec += ncond ;
	else {
	so.coef() ;

	Index pos (nbr_coefs) ;
	Index pos_galerkin (nbr_coefs) ;
	int basep = (*so.get_base().bases_1d[2]) (0) ;

	if ((bound==ETA_MINUS_BC) || (bound==ETA_PLUS_BC)) {
	// Loop on phi :
	for (int k=0 ; k<nbr_coefs(2) ; k++) {
		pos.set(2) = k ;
		double anchor_boundary = 0. ;
		bool anchor_ready = false ;
		// Loop on chi :
		for (int i=0 ; i<nbr_coefs(0)-1 ; i++) {
			pos.set(0) = i ;
			switch (basep) {
				case COS :
					// Avoid last odd ones
					if ((k%2!=1) || (i!=nbr_coefs(1)-2)) {
						if ((k==0) || (k%2==1)) {
							// The ones without regularity issues
							sec.set(pos_sec) = val_boundary (bound, so, pos) ;
							pos_sec ++ ;
						}
						else if (i!=0) {
							// Regularity on the axis thanks to Galerkin
							// Factor fo Galerkin different for Legendre or Chebyshev:
							double factor_galerkin ;
							switch (type_base) {
								case CHEB_TYPE :
									factor_galerkin = (i%2==1) ? -2. : 2. ;
									break ;
								case LEG_TYPE :
									factor_galerkin = -double(4*i+1) ;
									for (int ii=1 ; ii<=i ; ii++)
										factor_galerkin *= -double(2*ii-1)/double(2*ii) ;
									break ;
								default :
					cerr << "Unknown type of basis in Domain_bispheric_eta_first_nosym::export_tau_val_domain_boundary_one_side" << endl ;
										abort() ;
								}
						if (!anchor_ready) {
							pos_galerkin = pos ;
							pos_galerkin.set(0) = 0 ;
							anchor_boundary = val_boundary (bound, so, pos_galerkin) ;
							anchor_ready = true ;
						}
						sec.set(pos_sec) = val_boundary (bound, so, pos) +
								factor_galerkin * anchor_boundary ;
						pos_sec ++ ;
						}}
						break ;
					case SIN :
						// Avoid sin(0)
						if ((k!=0) && (k!=nbr_coefs(2)-1))
						//Avoid last odd ones
						if ((k%2!=1) || (i!=nbr_coefs(1)-2)) {
							if (k%2==1) {
								// The ones without regularity issues
								sec.set(pos_sec) = val_boundary (bound, so, pos) ;
								pos_sec ++ ;
							}
							else if (i!=0) {
								// Regularity on the axis thanks to Galerkin
								// Factor fo Galerkin different for Legendre or Chebyshev:
								double factor_galerkin ;
								switch (type_base) {
									case CHEB_TYPE :
										factor_galerkin = (i%2==1) ? -2. : 2. ;
										break ;
									case LEG_TYPE :
										factor_galerkin = -double(4*i+1) ;
										for (int ii=1 ; ii<=i ; ii++)
											factor_galerkin *= -double(2*ii-1)/double(2*ii) ;
										break ;
									default :
						cerr << "Unknown type of basis in Domain_bispheric_eta_first_nosym::export_tau_val_domain_boundary_one_side" << endl ;
										abort() ;
								}
							if (!anchor_ready) {
								pos_galerkin = pos ;
								pos_galerkin.set(0) = 0 ;
								anchor_boundary = val_boundary (bound, so, pos_galerkin) ;
								anchor_ready = true ;
							}
							sec.set(pos_sec) = val_boundary (bound, so, pos) +
								factor_galerkin * anchor_boundary ;
							pos_sec ++ ;
						}}
						break ;
					default :
						cerr << "Unknown base in Domain_bispheric_eta_first_nosym::export_tau_val_domain_boundary_one_side" << endl ;
					break ;
				}
			}
		}
	}

	if (bound==OUTER_BC) {
		for (int k=0 ; k<nbr_coefs(2) ; k++) {
			pos.set(2) = k ;
			if ((basep==COS) || ((k!=0) && (k!=nbr_coefs(2)-1)))
				for (int j=0 ; j<nbr_coefs(1) ; j++) {
					pos.set(1) = j ;
					sec.set(pos_sec) = val_boundary (bound, so, pos) ;
					pos_sec ++ ;
			}
		}
	}
  }
}

void Domain_bispheric_eta_first_nosym::export_tau_boundary_one_side (const Tensor& tt, int dom, int bound, Array<double>& res, int& pos_res, const Array<int>& ncond,
										int n_cmp, Array<int>** p_cmp) const {

	// Check boundary
	if ((bound!=ETA_MINUS_BC) && (bound!=ETA_PLUS_BC) && (bound!=OUTER_BC)) {
		cerr << "Unknown boundary in Domain_bispheric_eta_first_nosym::export_tau_boundary_one_side" << endl ;
		abort() ;
	}

	int val = tt.get_valence() ;
	switch (val) {
		case 0 :
			export_tau_val_domain_boundary_one_side (tt()(dom), bound, res, pos_res, ncond(0)) ;
			break ;
		case 1 : {
			bool found = false ;
			// Cartesian basis
			if (tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) {
				if (n_cmp==-1) {
					export_tau_val_domain_boundary_one_side (tt(1)(dom),  bound, res, pos_res, ncond(0)) ;
					export_tau_val_domain_boundary_one_side (tt(2)(dom), bound,  res, pos_res, ncond(1)) ;
					export_tau_val_domain_boundary_one_side (tt(3)(dom),  bound, res, pos_res, ncond(2)) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if ((*p_cmp[i])(0)==1)
						export_tau_val_domain_boundary_one_side (tt(1)(dom),  bound, res, pos_res, ncond(i)) ;
					if ((*p_cmp[i])(0)==2)
						export_tau_val_domain_boundary_one_side (tt(2)(dom),  bound, res, pos_res, ncond(i)) ;
					if ((*p_cmp[i])(0)==3)
						export_tau_val_domain_boundary_one_side (tt(3)(dom),  bound, res, pos_res, ncond(i)) ;
				}
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of vector Domain_bispheric_eta_first_nosym::export_tau_boundary_one_side" << endl ;
				abort() ;
			}
		}
			break ;
		case 2 : {
			bool found = false ;
			// Cartesian basis and symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==6)) {
				if (n_cmp==-1) {
					export_tau_val_domain_boundary_one_side (tt(1,1)(dom),  bound, res, pos_res, ncond(0)) ;
					export_tau_val_domain_boundary_one_side (tt(1,2)(dom),  bound, res, pos_res, ncond(1)) ;
					export_tau_val_domain_boundary_one_side (tt(1,3)(dom),  bound, res, pos_res, ncond(2)) ;
					export_tau_val_domain_boundary_one_side (tt(2,2)(dom),  bound, res, pos_res, ncond(3)) ;
					export_tau_val_domain_boundary_one_side (tt(2,3)(dom),  bound, res, pos_res, ncond(4)) ;
					export_tau_val_domain_boundary_one_side (tt(3,3)(dom), bound,  res, pos_res, ncond(5)) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain_boundary_one_side (tt(1, 1)(dom), bound,  res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain_boundary_one_side (tt(1, 2)(dom), bound,  res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain_boundary_one_side (tt(1, 3)(dom),  bound, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain_boundary_one_side (tt(2, 2)(dom),  bound, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain_boundary_one_side (tt(2, 3)(dom),  bound, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain_boundary_one_side (tt(3, 3)(dom),  bound, res, pos_res, ncond(i)) ;
				}
				found = true ;
			}
			// Cartesian basis and not symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==9)) {
				if (n_cmp==-1) {
					export_tau_val_domain_boundary_one_side (tt(1,1)(dom),  bound, res, pos_res, ncond(0)) ;
					export_tau_val_domain_boundary_one_side (tt(1,2)(dom),  bound, res, pos_res, ncond(1)) ;
					export_tau_val_domain_boundary_one_side (tt(1,3)(dom),  bound, res, pos_res, ncond(2)) ;
					export_tau_val_domain_boundary_one_side (tt(2,1)(dom),  bound, res, pos_res, ncond(3)) ;
					export_tau_val_domain_boundary_one_side (tt(2,2)(dom),  bound, res, pos_res, ncond(4)) ;
					export_tau_val_domain_boundary_one_side (tt(2,3)(dom), bound,  res, pos_res, ncond(5)) ;
					export_tau_val_domain_boundary_one_side (tt(3,1)(dom),  bound, res, pos_res, ncond(6)) ;
					export_tau_val_domain_boundary_one_side (tt(3,2)(dom),  bound, res, pos_res, ncond(7)) ;
					export_tau_val_domain_boundary_one_side (tt(3,3)(dom), bound,  res, pos_res, ncond(8)) ;

				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain_boundary_one_side (tt(1, 1)(dom), bound,  res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain_boundary_one_side (tt(1, 2)(dom), bound,  res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain_boundary_one_side (tt(1, 3)(dom),  bound, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain_boundary_one_side (tt(2, 1)(dom), bound,  res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain_boundary_one_side (tt(2, 2)(dom),  bound, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain_boundary_one_side (tt(2, 3)(dom),  bound, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain_boundary_one_side (tt(3, 1)(dom), bound,  res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain_boundary_one_side (tt(3, 2)(dom),  bound, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain_boundary_one_side (tt(3, 3)(dom),  bound, res, pos_res, ncond(i)) ;
				}
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of 2-tensor Domain_bispheric_eta_first_nosym::export_tau_boundary_one_side" << endl ;
				abort() ;
			}
		}
			break ;
		default :
			cerr << "Valence " << val << " not implemented in Domain_bispheric_eta_first_nosym::export_tau_boundary_one_side" << endl ;
			break ;
	}
}
}
