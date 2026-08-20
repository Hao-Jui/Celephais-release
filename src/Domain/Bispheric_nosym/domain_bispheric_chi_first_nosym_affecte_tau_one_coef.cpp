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
void Domain_bispheric_chi_first_nosym::affecte_tau_one_coef_val_domain (Val_domain& so,  int cc, int& conte) const {


	so.is_zero = false ;
	so.allocate_coef() ;
	*so.cf=0. ;
	Index pos_cf(nbr_coefs) ;

	bool found = false ;
	do {
		// Check if true :
		bool indic = true ;
		// Check true coefs in phi
		if ((pos_cf(2)==1) || (pos_cf(2)==nbr_coefs(2)-1))
			indic = false ;

		int basechi = (*so.get_base().bases_1d[1]) (pos_cf(2)) ;
		// No last coef if odd
		if (((basechi==CHEB_ODD) || (basechi==LEG_ODD)) && (pos_cf(1)==nbr_coefs(1)-1))
			indic = false ;

		//Galerkin for even ones
		if (((basechi==CHEB_EVEN) || (basechi==LEG_EVEN)) && (pos_cf(2)!=0) && (pos_cf(1)==0))
			indic = false ;

		if (indic) {
			if (conte==cc) {
				found = true ;
				so.cf->set(pos_cf) = 1;
				// Regularity on the axis :
				if (((basechi==CHEB_EVEN) || (basechi==LEG_EVEN)) && (pos_cf(2)!=0)) {
					Index pos_galerkin (pos_cf) ;
					pos_galerkin.set(1) = 0 ;
					double valreg ;
					int basechi = (*so.get_base().bases_1d[1])(pos_cf(2)) ;
					switch (basechi) {
						case CHEB_EVEN :
							valreg = (pos_cf(1)%2==0) ? -1 : 1  ;
							break ;
						case LEG_EVEN :
							valreg = 0.5 ;
							for (int i=1 ; i<pos_cf(1) ; i++)
								valreg *= - double(2*i+1)/double(2*i+2) ;
							break ;
						default :
							cerr << "Unknown base in Domain_bispheric_chi_first_nosym::affecte_one_coef" << endl ;
							abort() ;
						}
					so.cf->set(pos_galerkin) = valreg ;
					}
				}
			else
				so.cf->set(pos_cf) = 0. ;
			conte ++ ;
		}
	}
	while (pos_cf.inc()) ;

	// If not found put to zero :
	if (!found)
		so.set_zero() ;
}

void Domain_bispheric_chi_first_nosym::affecte_tau_one_coef (Tensor& tt, int dom, int cc, int& pos_cf) const {

	// Check right domain
	assert (tt.get_space().get_domain(dom)==this) ;

	int val = tt.get_valence() ;
	switch (val) {
		case 0 :
			affecte_tau_one_coef_val_domain (tt.set().set_domain(dom), cc, pos_cf) ;
			break ;
		case 1 : {
			bool found = false ;
			// Cartesian basis
			if (tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) {
				affecte_tau_one_coef_val_domain (tt.set(1).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(2).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(3).set_domain(dom), cc, pos_cf) ;
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of vector Domain_bispheric_chi_first_nosym::affecte_tau_one_coef" << endl ;
				abort() ;
			}
		}
			break ;
		case 2 : {
			bool found = false ;
			// Cartesian basis and symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==6)) {
				affecte_tau_one_coef_val_domain (tt.set(1,1).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(1,2).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(1,3).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(2,2).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(2,3).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(3,3).set_domain(dom), cc, pos_cf) ;
				found = true ;
			}
			// Cartesian basis and not symetric
			if ((tt.get_basis().get_basis(dom)==CARTESIAN_BASIS) && (tt.get_n_comp()==9)) {
				affecte_tau_one_coef_val_domain (tt.set(1,1).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(1,2).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(1,3).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(2,1).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(2,2).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(2,3).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(3,1).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(3,2).set_domain(dom), cc, pos_cf) ;
				affecte_tau_one_coef_val_domain (tt.set(3,3).set_domain(dom), cc, pos_cf) ;
				found = true ;
			}
			if (!found) {
				cerr << "Unknown type of 2-tensor Domain_bispheric_chi_first_nosym::affecte_tau_one_coef" << endl ;
				abort() ;
			}
		}
			break ;
		default :
			cerr << "Valence " << val << " not implemented in Domain_bispheric_chi_first_nosym::affecte_tau" << endl ;
			break ;
	}
}

bool Domain_bispheric_chi_first_nosym::describe_tau_seed_block(
	const Tensor& tt, int dom, std::vector<TauSeedDescriptor>& descriptors) const
{
	descriptors.clear();
	if (tt.get_space().get_domain(dom) != this || tt.get_valence() < 0 ||
		tt.get_valence() > 2 ||
		(tt.get_valence() == 1 && tt.get_n_comp() != 3) ||
		(tt.get_valence() == 2 && tt.get_n_comp() != 6 && tt.get_n_comp() != 9) ||
		(tt.get_valence() > 0 && tt.get_basis().get_basis(dom) != CARTESIAN_BASIS)) {
		return false;
	}
	const std::size_t phi_size = static_cast<std::size_t>(nbr_coefs(2));
	const std::size_t first_stride =
		static_cast<std::size_t>(nbr_coefs(1)) * phi_size;
	for (int component = 0; component < tt.get_n_comp(); ++component) {
		const Array<int> index(tt.indices(component));
		const Val_domain& field = tt(index)(dom);
		Index coefficient(nbr_coefs);
		do {
			const int phi = coefficient(2);
			const int chi = coefficient(1);
			const int chi_basis = (*field.get_base().get_base_1d(1))(phi);
			const bool odd = chi_basis == CHEB_ODD || chi_basis == LEG_ODD;
			const bool even = chi_basis == CHEB_EVEN || chi_basis == LEG_EVEN;
			if (phi == 1 || phi == nbr_coefs(2) - 1 ||
				(odd && chi == nbr_coefs(1) - 1) ||
				(even && phi != 0 && chi == 0)) {
				continue;
			}
			TauSeedDescriptor descriptor;
			descriptor.component = component;
			descriptor.writes[0] = {
				static_cast<std::size_t>(coefficient(0)) * first_stride +
					static_cast<std::size_t>(chi) * phi_size +
					static_cast<std::size_t>(phi),
				1.};
			descriptor.write_count = 1;
			if (even && phi != 0) {
				double regularity;
				if (chi_basis == CHEB_EVEN) {
					regularity = (chi % 2 == 0) ? -1. : 1.;
				} else {
					regularity = .5;
					for (int i = 1; i < chi; ++i)
						regularity *= -double(2 * i + 1) / double(2 * i + 2);
				}
				descriptor.writes[1] = {
					static_cast<std::size_t>(coefficient(0)) * first_stride +
						static_cast<std::size_t>(phi),
					regularity};
				descriptor.write_count = 2;
			}
			descriptors.push_back(descriptor);
		} while (coefficient.inc());
	}
	return descriptors.size() ==
		static_cast<std::size_t>(nbr_unknowns(tt, dom));
}
}
