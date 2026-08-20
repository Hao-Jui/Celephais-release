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
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Domain/spheric_nosym_regularization.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

#include <cstddef>

namespace Kadath {

void Domain_nucleus_nosym::affecte_tau_val_domain (Val_domain& so, const Array<double>& values, int& conte) const {

	so.allocate_coef() ;
	*so.cf = 0. ;
	double* const coefficient_data = so.cf->get_data() ;
	const double* const values_data = values.get_data() ;
	const std::size_t phi_size = static_cast<std::size_t>(nbr_coefs(2)) ;
	const std::size_t radial_stride = static_cast<std::size_t>(nbr_coefs(1)) * phi_size ;
	double fact_t, fact_r, fact_rt ;

	// Loop on phi :
	for (int k=0 ; k<nbr_coefs(2)-1 ; k++)
		if (k!=1) {
			// Loop on theta
			int baset = (*so.get_base().bases_1d[1]) (k) ;
			for (int j=0 ; j<nbr_coefs(1) ; j++) {
				int baser = (*so.get_base().bases_1d[0]) (j, k) ;
				const std::size_t row_offset = static_cast<std::size_t>(j) * phi_size +
					static_cast<std::size_t>(k) ;
				int maxi = 0 ;
				switch (baser) {
					case CHEB_EVEN :
						maxi = nbr_coefs(0) ;
						break ;
					case LEG_EVEN :
						maxi = nbr_coefs(0) ;
						break ;
					case CHEB_ODD :
						maxi = nbr_coefs(0)-1 ;
						break ;
					case LEG_ODD :
						maxi = nbr_coefs(0)-1 ;

						break ;

					default :
						KADATH_THROW("Strange base in Domain_nucleus_nosym:export_tau_val_domain");
				}
				// Loop on r :
				for (int i=0 ; i<maxi ; i++) {
					const std::size_t coefficient_offset = static_cast<std::size_t>(i) * radial_stride +
						row_offset ;
					switch (baset) {
						case COS :
						if (j!=nbr_coefs(1)-1) {
							if (k<4) {
							if (j<=1) {
								coefficient_data[coefficient_offset] += values_data[conte] ;
								conte ++ ;
							}
							else  {
							if (i!=0) {
								// Galerkin base in r only
								switch (baser) {
									case CHEB_EVEN :
									  fact_r = (i % 2 == 0) ? -1. : 1. ;
									  break ;
									case LEG_EVEN : {
									  fact_r = -1. ;
									  for (int t=0 ; t<i ; t++)
									    fact_r *= -double(2*t+1)/double(2*t+2) ;
									  }
									  break ;
									 case CHEB_ODD :
										fact_r = (i % 2 == 0) ? -double(2*i+1) : double(2*i+1) ;
									break ;
									case LEG_ODD : {
										fact_r = -1. ;
										for (int t=0 ; t<i ; t++)
										fact_r *= -double(2*t+3)/double(2*t+2) ;
									}
									break ;
									default :
									  KADATH_THROW("Strange base in Domain_nucleus_nosym::affecte_tau_val_domain");
								}

								coefficient_data[coefficient_offset] += values_data[conte] ;
								coefficient_data[row_offset] += fact_r*values_data[conte] ;
								conte ++ ;
							  }
							}
							}
							else {
							if (detail::spheric_nosym_true_theta_coef(baset, j, k, 2, nbr_coefs(1)) && (i!=0)) {
							    // Need to use two_dimensional Galerkin basis (aouch !)
								    const std::size_t anchor_row_offset =
									static_cast<std::size_t>(detail::spheric_nosym_theta_anchor(baset, j)) * phi_size +
									static_cast<std::size_t>(k) ;
								    const std::size_t theta_anchor_offset = static_cast<std::size_t>(i) * radial_stride +
									anchor_row_offset ;
								    switch (baser) {
								      case CHEB_EVEN :
									fact_r = (i % 2 == 0) ? -1. : 1. ;
								break ;
							      case LEG_EVEN : {
								double l0 = 1 ;
								 for (int t=0 ; t<i ; t++)
								  l0 *= -double(2*t+1)/double(2*t+2) ;
								fact_r = - l0 ;
								}
									break ;
								      case CHEB_ODD :
									fact_r = (i % 2 == 0) ? -double(2*i+1) : double(2*i+1) ;
								break ;
							      case LEG_ODD : {
								double l0 = 1 ;
								 for (int t=0 ; t<i ; t++)
								  l0 *= -double(2*t+3)/double(2*t+2) ;
								fact_r = - l0 ;
								}
								break ;
							    default :
									  KADATH_THROW("Strange base in Domain_nucleus_nosym::affecte_tau_val_domain");
								}
								fact_t = -detail::spheric_nosym_basis_anchor_weight(baset, j) ;
								fact_rt = fact_r * fact_t ;
								      coefficient_data[coefficient_offset] += values_data[conte] ;
								      coefficient_data[row_offset] += fact_r*values_data[conte] ;
								      coefficient_data[theta_anchor_offset] += fact_t*values_data[conte] ;
								      coefficient_data[anchor_row_offset] += fact_rt*values_data[conte] ;
							      conte ++ ;
							    }
						}
						}
						break ;

					case SIN:
						if ((j!=0) && (j!=nbr_coefs(1)-1)) {
						if (k<4) {
							if (j<=1) {
								coefficient_data[coefficient_offset] += values_data[conte] ;
								conte ++ ;
							}
							else
							if (i!=0) {
							// Galerkin base in r only
							switch (baser) {
								case CHEB_EVEN :
								  fact_r = (i % 2 == 0) ? -1. : 1. ;
								  break ;
								case LEG_EVEN : {
								  fact_r = -1. ;
								  for (int t=0 ; t<i ; t++)
								  fact_r *= -double(2*t+1)/double(2*t+2) ;
								  }
								  break ;
								  case CHEB_ODD :
									fact_r = (i % 2 == 0) ? -double(2*i+1) : double(2*i+1) ;
								    break ;
								  case LEG_ODD : {
									fact_r = -1. ;
									for (int t=0 ; t<i ; t++)
									 fact_r *= -double(2*t+3)/double(2*t+2) ;
								    }
								    break ;
								default :
								  KADATH_THROW("Strange base in Domain_nucleus_nosym::affecte_tau_val_domain");
								}
							    coefficient_data[coefficient_offset] += values_data[conte] ;
							    coefficient_data[row_offset] += fact_r*values_data[conte] ;
							    conte ++ ;
							}
						}
						else
						//Double Galerkin
						if (detail::spheric_nosym_true_theta_coef(baset, j, k, 2, nbr_coefs(1)) && (i!=0)) {
								 // Need to use two_dimensional Galerkin basis (aouch !)
								    const std::size_t anchor_row_offset =
									static_cast<std::size_t>(detail::spheric_nosym_theta_anchor(baset, j)) * phi_size +
									static_cast<std::size_t>(k) ;
								    const std::size_t theta_anchor_offset = static_cast<std::size_t>(i) * radial_stride +
									anchor_row_offset ;
								     switch (baser) {
								      case CHEB_EVEN :
									fact_r = (i % 2 == 0) ? -1. : 1. ;
								break ;
							      case LEG_EVEN : {
								double l0 = 1 ;
								 for (int t=0 ; t<i ; t++)
								  l0 *= -double(2*t+1)/double(2*t+2) ;
								fact_r = - l0 ;
								}
									break ;
								case CHEB_ODD :
									fact_r = (i % 2 == 0) ? -double(2*i+1) : double(2*i+1) ;
									break ;
								case LEG_ODD : {
									double l0 = 1 ;
									for (int t=0 ; t<i ; t++)
										l0 *= -double(2*t+3)/double(2*t+2) ;
									fact_r = - l0 ;
									}
									break ;

							    default :
									  KADATH_THROW("Strange base in Domain_nucleus_nosym::affecte_tau_val_domain");
								}
								fact_t = -detail::spheric_nosym_basis_anchor_weight(baset, j) ;
								fact_rt = fact_r * fact_t ;

								      coefficient_data[coefficient_offset] += values_data[conte] ;
								      coefficient_data[row_offset] += fact_r*values_data[conte] ;
								      coefficient_data[theta_anchor_offset] += fact_t*values_data[conte] ;
								      coefficient_data[anchor_row_offset] += fact_rt*values_data[conte] ;

							      conte ++ ;
							}
						      }
						break ;
				}
				}
			}
		}
}

void Domain_nucleus_nosym::affecte_tau (Tensor& tt, int dom, const Array<double>& cf, int& pos_cf) const {

	// Check right domain
	assert (tt.get_space().get_domain(dom)==this) ;

	int val = tt.get_valence() ;
	switch (val) {
		case 0 :
			affecte_tau_val_domain (tt.set().set_domain(dom), cf, pos_cf) ;
			break ;
		case 1 : {
			bool found = false ;
			// Cartesian basis
			if (tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) {
				affecte_tau_val_domain (tt.set(1).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(2).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(3).set_domain(dom), cf, pos_cf) ;
				found = true ;
			}

			if (!found) {
				KADATH_THROW("Unknown type of vector Domain_nucleus_nosym::affecte_tau");
			}
		}
			break ;
		case 2 : {
			bool found = false ;
			// Cartesian basis and symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==6)) {
				affecte_tau_val_domain (tt.set(1,1).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(1,2).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(1,3).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(2,2).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(2,3).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(3,3).set_domain(dom), cf, pos_cf) ;
				found = true ;
			}
			// Cartesian basis and not symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==9)) {
				affecte_tau_val_domain (tt.set(1,1).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(1,2).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(1,3).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(2,1).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(2,2).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(2,3).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(3,1).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(3,2).set_domain(dom), cf, pos_cf) ;
				affecte_tau_val_domain (tt.set(3,3).set_domain(dom), cf, pos_cf) ;
				found = true ;
			}
			if (!found) {
				KADATH_THROW("Unknown type of 2-tensor Domain_nucleus_nosym::affecte_tau");
			}
		}
			break ;
		default :
			cerr << "Valence " << val << " not implemented in Domain_nucleus_nosym::affecte_tau" << endl ;
			break ;
	}
}}
