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
int Domain_bispheric_rect_nosym::nbr_conditions_val_domain_boundary (const Val_domain& so, int bound) const {

	int res = 0 ;
	assert (basep==COSSIN) ;

	if (bound==INNER_BC) {
	Index pos (nbr_coefs) ;
	// Loop on phi :
	for (int k=0 ; k<nbr_coefs(2) ; k++) {
		pos.set(2) = k ;
		// Loop on chi ;
		for (int j=0 ; j<nbr_coefs(1) ; j++) {
			pos.set(1) = j ;
			bool indic = true ;

			// Check true coefs in phi
			if ((k==1) || (k==nbr_coefs(2)-1))
				indic = false ;

			int basechi = (*so.get_base().bases_1d[1]) (k) ;
			// No last coef if odd
			if (((basechi==CHEB_ODD) || (basechi==LEG_ODD)) && (j==nbr_coefs(1)-1))
				indic = false ;

			//Galerkin for even ones
			if (((basechi==CHEB_EVEN) || (basechi==LEG_EVEN)) && (k!=0) && (j==0))
				indic = false ;

		      if (indic)
			  res ++ ;
			}
		}
	}

	if (bound==ETA_PLUS_BC) {
		for (int k=0 ; k<nbr_coefs(2) ; k++)
			for (int j=0 ; j<nbr_coefs(1) ; j++) {
			bool true_other = true ;

			// Check true coefs in phi
			if ((k==1) || (k==nbr_coefs(2)-1))
				true_other = false ;

			int basechi = (*so.get_base().bases_1d[1]) (k) ;
			// No last coef if odd
			if (((basechi==CHEB_ODD) || (basechi==LEG_ODD)) && (j>=nbr_coefs(1)-2))
				true_other = false ;

			//Galerkin for even ones
			if (((basechi==CHEB_EVEN) || (basechi==LEG_EVEN)) && (k!=0) && (j==0))
				true_other = false ;

			if (((basechi==CHEB_EVEN) || (basechi==LEG_EVEN)) && (j>=nbr_coefs(1)-1))
				true_other = false ;

		if (true_other)
			res ++ ;
		}
	}

	if (bound==CHI_ONE_BC)
		res = (nbr_coefs(2)-2)*(nbr_coefs(0)-2) ;

	if (bound==OUTER_BC)
		res = nbr_coefs(2)-2 ;
	return res ;
}

Array<int> Domain_bispheric_rect_nosym::nbr_conditions_boundary (const Tensor& tt, int dom, int bound, int n_cmp, Array<int>** p_cmp) const {

	// Check boundary
	if ((bound!=INNER_BC) && (bound!=CHI_ONE_BC) && (bound!=ETA_PLUS_BC) && (bound!=OUTER_BC)) {
		cerr << "Unknown boundary in Domain_bispheric_rect_nosym::nbr_conditions_boundary" << endl ;
		abort() ;
	}

	int size = (n_cmp==-1) ? tt.get_n_comp() : n_cmp ;
	Array<int> res (size) ;
	int val = tt.get_valence() ;
	switch (val) {
		case 0 :
			res.set(0) = nbr_conditions_val_domain_boundary (tt()(dom), bound) ;
			break ;
		case 1 : {
			bool found = false ;
			// Cartesian basis
			if (tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1)(dom), bound) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(2)(dom), bound) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(3)(dom), bound) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if ((*p_cmp[i])(0)==1)
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1)(dom), bound) ;
					if ((*p_cmp[i])(0)==2)
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2)(dom), bound) ;
					if ((*p_cmp[i])(0)==3)
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3)(dom), bound) ;
				}
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of vector Domain_bispheric_rect_nosym::nbr_conditions_boundary" << endl ;
				abort() ;
			}
		}
			break ;
		case 2 : {
			bool found = false ;
			// Cartesian basis and symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==6)) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1,1)(dom), bound) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(1,2)(dom), bound) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(1,3)(dom), bound) ;
					res.set(3) = nbr_conditions_val_domain_boundary (tt(2,2)(dom), bound) ;
					res.set(4) = nbr_conditions_val_domain_boundary (tt(2,3)(dom), bound) ;
					res.set(5) = nbr_conditions_val_domain_boundary (tt(3,3)(dom), bound) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 1)(dom), bound) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 2)(dom), bound) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 3)(dom), bound) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 2)(dom), bound) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 3)(dom), bound) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 3)(dom), bound) ;
				}
				found = true ;
			}
			// Cartesian basis and not symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==9)) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1,1)(dom), bound) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(1,2)(dom), bound) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(1,3)(dom), bound) ;
					res.set(3) = nbr_conditions_val_domain_boundary (tt(2,1)(dom), bound) ;
					res.set(4) = nbr_conditions_val_domain_boundary (tt(2,2)(dom), bound) ;
					res.set(5) = nbr_conditions_val_domain_boundary (tt(2,3)(dom), bound) ;
					res.set(6) = nbr_conditions_val_domain_boundary (tt(3,1)(dom), bound) ;
					res.set(7) = nbr_conditions_val_domain_boundary (tt(3,2)(dom), bound) ;
					res.set(8) = nbr_conditions_val_domain_boundary (tt(3,3)(dom), bound) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 1)(dom), bound) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 2)(dom), bound) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 3)(dom), bound) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 1)(dom), bound) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 2)(dom), bound) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 3)(dom), bound) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 1)(dom), bound) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 2)(dom), bound) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 3)(dom), bound) ;
				}
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of 2-tensor Domain_bispheric_rect_nosym::nbr_conditions_boundary" << endl ;
				abort() ;
			}
		}
			break ;
		default :
			cerr << "Valence " << val << " not implemented in Domain_bispheric_rect_nosym::nbr_conditions_boundary" << endl ;
			break ;
	}
	return res ;
}

bool Domain_bispheric_rect_nosym::describe_boundary_residual_rows(
    const Tensor& tt, int dom, int bound, const Array<int>& ncond, int n_cmp,
    Array<int>** p_cmp, std::vector<ResidualRowDescriptor>& descriptors) const
{
    descriptors.clear();
    std::vector<int> components;
    if ((bound != INNER_BC && bound != ETA_PLUS_BC &&
         bound != CHI_ONE_BC && bound != OUTER_BC) ||
        !residual_tensor_components_in_tau_order(
            tt, dom, n_cmp, p_cmp, components) ||
        ncond.get_nbr() != components.size() ||
        (tt.get_valence() > 0 &&
         tt.get_basis().get_basis(dom) != CARTESIAN_BASIS)) {
        return false;
    }

    for (std::size_t slot = 0; slot < components.size(); ++slot) {
        const int component = components[slot];
        const Array<int> indices = tt.indices(component);
        const Val_domain& field = tt.get_valence() == 0
                                      ? tt()(dom)
                                      : tt(indices)(dom);
        const std::size_t before = descriptors.size();
        for (int k = 0; k < nbr_coefs(2) - 1; ++k) {
            if (k == 1)
                continue;
            if (bound == INNER_BC || bound == ETA_PLUS_BC) {
                const int chi_basis = (*field.get_base().bases_1d[1])(k);
                const bool even = chi_basis == CHEB_EVEN || chi_basis == LEG_EVEN;
                const bool odd = chi_basis == CHEB_ODD || chi_basis == LEG_ODD;
                if (!even && !odd) {
                    descriptors.clear();
                    return false;
                }
                int max_chi = nbr_coefs(1) -
                              (bound == ETA_PLUS_BC ? 1 : 0);
                if (odd)
                    --max_chi;
                for (int j = 0; j < max_chi; ++j) {
                    if (even && k != 0 && j == 0)
                        continue;
                    ResidualRowDescriptor descriptor;
                    if (!append_volume_residual_row(
                            field, dom, component, k, descriptor)) {
                        descriptors.clear();
                        return false;
                    }
                    descriptors.push_back(std::move(descriptor));
                }
            } else {
                const int repetitions = bound == CHI_ONE_BC
                                            ? nbr_coefs(0) - 2
                                            : 1;
                for (int i = 0; i < repetitions; ++i) {
                    ResidualRowDescriptor descriptor;
                    if (!append_volume_residual_row(
                            field, dom, component, k, descriptor)) {
                        descriptors.clear();
                        return false;
                    }
                    descriptors.push_back(std::move(descriptor));
                }
            }
        }
        if (descriptors.size() - before !=
            static_cast<std::size_t>(ncond(static_cast<int>(slot)))) {
            descriptors.clear();
            return false;
        }
    }
    return true;
}
}
