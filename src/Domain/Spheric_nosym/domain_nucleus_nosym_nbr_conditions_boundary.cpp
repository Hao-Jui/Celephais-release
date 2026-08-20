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

namespace Kadath {

// A.5-validated inline boundary count (74e7f645). The HEAD `mlim` parameter is
// retained for API/signature compatibility with the binary nosym space but is
// unused: the reference logic does not thread an m-order shift.
int Domain_nucleus_nosym::nbr_conditions_val_domain_boundary (const Val_domain& so, int mlim) const {
	(void)mlim;
	int res = 0 ;
	Index pos (nbr_coefs) ;
	do {
		bool indic = true ;
		// True coef in phi ?
		if ((pos(2)==1) || (pos(2)==nbr_coefs(2)-1))
			indic = false ;
		// Get base in theta :
		int baset = (*so.get_base().bases_1d[1]) (pos(2)) ;
		indic = indic && detail::spheric_nosym_true_theta_coef(baset, pos(1), pos(2), 2, nbr_coefs(1));
		if ((indic) && (pos(0)==0))
			res ++ ;
	}
	while (pos.inc()) ;

	return res ;
}

Array<int> Domain_nucleus_nosym::nbr_conditions_boundary (const Tensor& tt, int dom, int bound, int n_cmp, Array<int>** p_cmp) const {

	int size = (n_cmp==-1) ? tt.get_n_comp() : n_cmp ;
	Array<int> res (size) ;
	int val = tt.get_valence() ;
	switch (val) {
		case 0 :
			if (!tt.is_m_order_affected())
			  res.set(0) = nbr_conditions_val_domain_boundary (tt()(dom), 0) ;
			else
			  res.set(0) = nbr_conditions_val_domain_boundary (tt()(dom), tt.get_parameters()->get_m_order()) ;
			break ;
		case 1 : {
			bool found = false ;
			// Cartesian basis
			if (tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1)(dom), 0) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(2)(dom), 0) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(3)(dom), 0) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if ((*p_cmp[i])(0)==1)
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1)(dom), 0) ;
					if ((*p_cmp[i])(0)==2)
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2)(dom), 0) ;
					if ((*p_cmp[i])(0)==3)
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3)(dom), 0) ;
				}
				found = true ;
			}
			if (!found) {
				KADATH_THROW("Unknown type of vector Domain_nucleus_nosym::nbr_conditions_boundary");
			}
		}
			break ;
		case 2 : {
			bool found = false ;
			// Cartesian basis and symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==6)) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1,1)(dom), 0) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(1,2)(dom), 0) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(1,3)(dom), 0) ;
					res.set(3) = nbr_conditions_val_domain_boundary (tt(2,2)(dom), 0) ;
					res.set(4) = nbr_conditions_val_domain_boundary (tt(2,3)(dom), 0) ;
					res.set(5) = nbr_conditions_val_domain_boundary (tt(3,3)(dom), 0) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 1)(dom), 0) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 2)(dom), 0) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 3)(dom), 0) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 2)(dom), 0) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 3)(dom), 0) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 3)(dom), 0) ;
				}
				found = true ;
			}
			// Cartesian basis and not symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==9)) {
				if (n_cmp==-1) {
					res.set(0) = nbr_conditions_val_domain_boundary (tt(1,1)(dom), 0) ;
					res.set(1) = nbr_conditions_val_domain_boundary (tt(1,2)(dom), 0) ;
					res.set(2) = nbr_conditions_val_domain_boundary (tt(1,3)(dom), 0) ;
					res.set(3) = nbr_conditions_val_domain_boundary (tt(2,1)(dom), 0) ;
					res.set(4) = nbr_conditions_val_domain_boundary (tt(2,2)(dom), 0) ;
					res.set(5) = nbr_conditions_val_domain_boundary (tt(2,3)(dom), 0) ;
					res.set(6) = nbr_conditions_val_domain_boundary (tt(3,1)(dom), 0) ;
					res.set(7) = nbr_conditions_val_domain_boundary (tt(3,2)(dom), 0) ;
					res.set(8) = nbr_conditions_val_domain_boundary (tt(3,3)(dom), 0) ;
				}
				else for (int i=0 ; i<n_cmp ; i++) {
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 1)(dom), 0) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 2)(dom), 0) ;
					if (((*p_cmp[i])(0)==1) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(1, 3)(dom), 0) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 1)(dom), 0) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 2)(dom), 0) ;
					if (((*p_cmp[i])(0)==2) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(2, 3)(dom), 0) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==1))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 1)(dom), 0) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==2))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 2)(dom), 0) ;
					if (((*p_cmp[i])(0)==3) && ((*p_cmp[i])(1)==3))
						res.set(i) = nbr_conditions_val_domain_boundary (tt(3, 3)(dom), 0) ;
				}
				found = true ;
			}
			if (!found) {
				KADATH_THROW("Unknown type of 2-tensor Domain_nucleus_nosym::nbr_conditions_boundary");
			}
		}
			break ;
		default :
			cerr << "Valence " << val << " not implemented in Domain_nucleus_nosym::nbr_conditions_boundary" << endl ;
			break ;
	}
	return res ;
}

bool Domain_nucleus_nosym::describe_boundary_residual_rows(
    const Tensor& tt, int dom, int, const Array<int>& ncond, int n_cmp,
    Array<int>** p_cmp, std::vector<ResidualRowDescriptor>& descriptors) const
{
    descriptors.clear();
    std::vector<int> components;
    if (!residual_tensor_components_in_tau_order(
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
            const int theta_basis = (*field.get_base().bases_1d[1])(k);
            for (int j = 0; j < nbr_coefs(1); ++j) {
                if (!detail::spheric_nosym_true_theta_coef(
                        theta_basis, j, k, 2, nbr_coefs(1))) {
                    continue;
                }
                ResidualRowDescriptor descriptor;
                if (!append_volume_residual_row(
                        field, dom, component, k, descriptor)) {
                    descriptors.clear();
                    return false;
                }
                descriptors.push_back(std::move(descriptor));
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
