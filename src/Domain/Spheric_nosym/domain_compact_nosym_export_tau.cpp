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

void Domain_compact_nosym::export_tau_val_domain (const Val_domain& so, int mlim, int order, Array<double>& sec, int& pos_sec, int ncond) const {

	if (so.check_if_zero())
		pos_sec += ncond ;
	else {

	so.coef() ;
	int kmin = 2*mlim + 2 ;
	const double* const coefficient_data = so.cf->get_data() ;
	double* const result_data = sec.get_data() ;
	const std::size_t phi_size = static_cast<std::size_t>(nbr_coefs(2)) ;
	const std::size_t radial_stride = static_cast<std::size_t>(nbr_coefs(1)) * phi_size ;

	// Loop on phi :
	for (int k=0 ; k<nbr_coefs(2)-1 ; k++)
		if (k!=1) {
			// Loop on theta
			int baset = (*so.get_base().bases_1d[1]) (k) ;
			for (int j=0 ; j<nbr_coefs(1) ; j++) {
				const std::size_t row_offset = static_cast<std::size_t>(j) * phi_size +
					static_cast<std::size_t>(k) ;
				// Loop on r :
				for (int i=0 ; i<nbr_coefs(0)-order ; i++) {
					const std::size_t coefficient_offset = static_cast<std::size_t>(i) * radial_stride +
						row_offset ;
					switch (baset) {
						case COS:
						if (detail::spheric_nosym_true_theta_coef(baset, j, k, kmin, nbr_coefs(1))) {
							result_data[pos_sec] = coefficient_data[coefficient_offset] ;
							if (detail::spheric_nosym_uses_theta_galerkin(baset, k, kmin)) {
								const std::size_t anchor_offset = static_cast<std::size_t>(i) * radial_stride +
									static_cast<std::size_t>(detail::spheric_nosym_theta_anchor(baset, j)) *
									phi_size + static_cast<std::size_t>(k) ;
								result_data[pos_sec] -= detail::spheric_nosym_export_anchor_weight(baset, j) *
									coefficient_data[anchor_offset] ;
							}
							pos_sec ++ ;
						}
						break ;
					case SIN:
						if (detail::spheric_nosym_true_theta_coef(baset, j, k, kmin, nbr_coefs(1))) {
							result_data[pos_sec] = coefficient_data[coefficient_offset] ;
							if (detail::spheric_nosym_uses_theta_galerkin(baset, k, kmin)) {
								const std::size_t anchor_offset = static_cast<std::size_t>(i) * radial_stride +
									static_cast<std::size_t>(detail::spheric_nosym_theta_anchor(baset, j)) *
									phi_size + static_cast<std::size_t>(k) ;
								result_data[pos_sec] -= detail::spheric_nosym_export_anchor_weight(baset, j) *
									coefficient_data[anchor_offset] ;
							}
							pos_sec ++ ;
						}
						break ;
					default:
						KADATH_THROW("Unknow theta basis in Domain_compact_nosym::export_tau_val_domain");
					}
				}
			}
		}
	}
}

void Domain_compact_nosym::export_tau (const Tensor& tt, int dom, int order, Array<double>& res, int& pos_res, const Array<int>& ncond,
										int n_cmp, Array<int>** p_cmp) const {
	int val = tt.get_valence() ;
	switch (val) {
		case 0 :
			if (!tt.is_m_order_affected())
			  export_tau_val_domain (tt()(dom), 0, order, res, pos_res, ncond(0)) ;
			else
			    export_tau_val_domain (tt()(dom), tt.get_parameters()->get_m_order(), order, res, pos_res, ncond(0)) ;
			break ;
		case 1 : {
			bool found = false ;
			// Cartesian basis
			if (tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) {
				if (n_cmp==-1) {
					export_tau_val_domain (tt(1)(dom), 0, order, res, pos_res, ncond(0)) ;
					export_tau_val_domain (tt(2)(dom), 0, order, res, pos_res, ncond(1)) ;
					export_tau_val_domain (tt(3)(dom), 0, order, res, pos_res, ncond(2)) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if ((*p_cmp[i])(0)==1)
						export_tau_val_domain (tt(1)(dom), 0, order, res, pos_res, ncond(i)) ;
					if ((*p_cmp[i])(0)==2)
						export_tau_val_domain (tt(2)(dom), 0, order, res, pos_res, ncond(i)) ;
					if ((*p_cmp[i])(0)==3)
						export_tau_val_domain (tt(3)(dom), 0, order, res, pos_res, ncond(i)) ;
				}
				found = true ;
			}
			if (!found) {
				KADATH_THROW("Unknown type of vector Domain_compact_nosym::export_tau");
			}
		}
			break ;
		case 2 : {
			bool found = false ;
			// Cartesian basis and symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==6)) {
				if (n_cmp==-1) {
					export_tau_val_domain (tt(1,1)(dom), 0, order, res, pos_res, ncond(0)) ;
					export_tau_val_domain (tt(1,2)(dom), 0, order, res, pos_res, ncond(1)) ;
					export_tau_val_domain (tt(1,3)(dom), 0, order, res, pos_res, ncond(2)) ;
					export_tau_val_domain (tt(2,2)(dom), 0, order, res, pos_res, ncond(3)) ;
					export_tau_val_domain (tt(2,3)(dom), 0, order, res, pos_res, ncond(4)) ;
					export_tau_val_domain (tt(3,3)(dom), 0, order, res, pos_res, ncond(5)) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain (tt(1, 1)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain (tt(1, 2)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(1, 3)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain (tt(2, 2)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(2, 3)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(3, 3)(dom), 0, order, res, pos_res, ncond(i)) ;
				}
				found = true ;
			}
			// Cartesian basis and not symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==9)) {
				if (n_cmp==-1) {
					export_tau_val_domain (tt(1,1)(dom), 0, order, res, pos_res, ncond(0)) ;
					export_tau_val_domain (tt(1,2)(dom), 0, order, res, pos_res, ncond(1)) ;
					export_tau_val_domain (tt(1,3)(dom), 0, order, res, pos_res, ncond(2)) ;
					export_tau_val_domain (tt(2,1)(dom), 0, order, res, pos_res, ncond(3)) ;
					export_tau_val_domain (tt(2,2)(dom), 0, order, res, pos_res, ncond(4)) ;
					export_tau_val_domain (tt(2,3)(dom), 0, order, res, pos_res, ncond(5)) ;
					export_tau_val_domain (tt(3,1)(dom), 0, order, res, pos_res, ncond(6)) ;
					export_tau_val_domain (tt(3,2)(dom), 0, order, res, pos_res, ncond(7)) ;
					export_tau_val_domain (tt(3,3)(dom), 0, order, res, pos_res, ncond(8)) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain (tt(1, 1)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain (tt(1, 2)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(1, 3)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain (tt(2, 1)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain (tt(2, 2)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(2, 3)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==1))
						export_tau_val_domain (tt(3, 1)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==2))
						export_tau_val_domain (tt(3, 2)(dom), 0, order, res, pos_res, ncond(i)) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						export_tau_val_domain (tt(3, 3)(dom), 0, order, res, pos_res, ncond(i)) ;
				}
				found = true ;
			}
			if (!found) {
				KADATH_THROW("Unknown type of 2-tensor Domain_compact_nosym::export_tau");
			}
		}
			break ;
		default :
			cerr << "Valence " << val << " not implemented in Domain_compact_nosym::export_tau" << endl ;
			break ;
	}
}}
