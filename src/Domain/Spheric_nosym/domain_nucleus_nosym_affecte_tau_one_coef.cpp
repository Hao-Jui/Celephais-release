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


void Domain_nucleus_nosym::affecte_tau_one_coef_val_domain (Val_domain& so, int cc, int& conte) const {

	so.is_zero = false ;
	so.allocate_coef() ;
	*so.cf=0. ;
	double* const coefficient_data = so.cf->get_data() ;
	const std::size_t phi_size = static_cast<std::size_t>(nbr_coefs(2)) ;
	const std::size_t radial_stride = static_cast<std::size_t>(nbr_coefs(1)) * phi_size ;

	bool found = false ;

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
						maxi = nbr_coefs(0);
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
							//k==0
							if (k<4) {
								if (j<=1) {
									// No galerkin :
									if (conte==cc)  {
								  found = true ;
								  coefficient_data[coefficient_offset] = 1. ;
								  }
								  conte ++ ;
							}
							else { // Galerlin en r
								 if (i!=0) {
								if (conte==cc) {
								found = true ;
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
									  KADATH_THROW("Strange base in Domain_nucleus_nosym::affecte_one_coef_val_domain");
								}
								coefficient_data[coefficient_offset] = 1 ;
								coefficient_data[row_offset] += fact_r ;
								}
								conte ++ ;
								}
								}
							}
							// Other k for COS in j
							else {
								// Double Garlerkin
							if (detail::spheric_nosym_true_theta_coef(baset, j, k, 2, nbr_coefs(1)) && (i!=0)) {
							    if (conte==cc) {
							    found = true ;
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
									  KADATH_THROW("Strange base in Domain_nucleus_nosym::affecte_one_coef_val_domain");
								}
								fact_t = -detail::spheric_nosym_basis_anchor_weight(baset, j) ;
								fact_rt = fact_r * fact_t ;

								      coefficient_data[coefficient_offset] = 1. ;
								      coefficient_data[row_offset] = fact_r ;
								      coefficient_data[theta_anchor_offset] = fact_t ;
								      coefficient_data[anchor_row_offset] = fact_rt ;

							      }
							      conte ++ ;
							    }
						}
						}
						break ;

					case SIN:
						if ((j!=0) && (j!=nbr_coefs(1)-1)) {
						if (k<4) {
								if (j<=1) {
								if (conte==cc)  {
								  found = true ;
								  coefficient_data[coefficient_offset] = 1. ;
								  }
								  conte ++ ;
								}
						else
							// Galerkin in r
						 if (i!=0) {
							// Galerkin base in r only
							 if (conte==cc) {
								found = true ;
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
								  KADATH_THROW("Strange base in Domain_nucleus_nosym_::affecte_one_coef_val_domain");
								}
							    coefficient_data[coefficient_offset] = 1. ;
								coefficient_data[row_offset] = fact_r ;
								}
							    conte ++ ;
							}
						}
						// Other k double Galerkin
						else
						if (detail::spheric_nosym_true_theta_coef(baset, j, k, 2, nbr_coefs(1)) && (i!=0)) {

								 if (conte==cc) {
								    found = true ;
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
									  KADATH_THROW("Strange base in Domain_nucleus_nosym::affecte_one_domain_val_domain");
								}
								fact_t = -detail::spheric_nosym_basis_anchor_weight(baset, j) ;
								fact_rt = fact_r * fact_t ;
								      coefficient_data[coefficient_offset] = 1 ;
								      coefficient_data[row_offset] = fact_r ;
								      coefficient_data[theta_anchor_offset] = fact_t ;
								      coefficient_data[anchor_row_offset] = fact_rt ;

								}
							      conte ++ ;
							}
						      }
						break ;
					default:
						KADATH_THROW("Unknow theta basis in Domain_nucleus_nosym::affecte_coef_val_domain");
					}
				}
			}
	}
	// If not found put to zero :
	if (!found)
		so.set_zero() ;
}

void Domain_nucleus_nosym::affecte_tau_one_coef (Tensor& tt, int dom, int cc, int& pos_cf) const {

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
				KADATH_THROW("Unknown type of vector Domain_nucleus_nosym::affecte_tau_one_coef");
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
				KADATH_THROW("Unknown type of 2-tensor Domain_nucleus_nosym::affecte_tau_one_coef");
			}
		}
			break ;
		default :
			cerr << "Valence " << val << " not implemented in Domain_nucleus_nosym::affecte_tau" << endl ;
			break ;
	}
}

namespace
{
double nucleus_radial_anchor_weight(int basis, int radial_index)
{
	switch (basis) {
		case CHEB_EVEN:
			return (radial_index % 2 == 0) ? -1. : 1.;
		case LEG_EVEN: {
			double factor = -1.;
			for (int t = 0; t < radial_index; ++t)
				factor *= -double(2 * t + 1) / double(2 * t + 2);
			return factor;
		}
		case CHEB_ODD:
			return (radial_index % 2 == 0)
				? -double(2 * radial_index + 1)
				: double(2 * radial_index + 1);
		case LEG_ODD: {
			double factor = -1.;
			for (int t = 0; t < radial_index; ++t)
				factor *= -double(2 * t + 3) / double(2 * t + 2);
			return factor;
		}
		default:
			KADATH_THROW("Strange radial basis in Domain_nucleus_nosym::describe_tau_seed_block");
	}
}

void append_nucleus_tau_seed_component(
	const Dim_array& nbr_coefs, const Val_domain& field, int component,
	std::vector<TauSeedDescriptor>& descriptors)
{
	const std::size_t phi_size = static_cast<std::size_t>(nbr_coefs(2));
	const std::size_t radial_stride =
		static_cast<std::size_t>(nbr_coefs(1)) * phi_size;

	for (int k = 0; k < nbr_coefs(2) - 1; ++k) {
		if (k == 1)
			continue;
		const int theta_basis = (*field.get_base().get_base_1d(1))(k);
		for (int j = 0; j < nbr_coefs(1); ++j) {
			const int radial_basis = (*field.get_base().get_base_1d(0))(j, k);
			const int radial_count =
				(radial_basis == CHEB_ODD || radial_basis == LEG_ODD)
				? nbr_coefs(0) - 1
				: nbr_coefs(0);
			if (radial_basis != CHEB_EVEN && radial_basis != LEG_EVEN &&
				radial_basis != CHEB_ODD && radial_basis != LEG_ODD) {
				KADATH_THROW("Strange radial basis in Domain_nucleus_nosym::describe_tau_seed_block");
			}
			const std::size_t row_offset =
				static_cast<std::size_t>(j) * phi_size +
				static_cast<std::size_t>(k);
			for (int i = 0; i < radial_count; ++i) {
				bool include = false;
				bool radial_galerkin = false;
				bool double_galerkin = false;
				switch (theta_basis) {
					case COS:
						if (j != nbr_coefs(1) - 1) {
							if (k < 4) {
								include = (j <= 1) || (i != 0);
								radial_galerkin = (j > 1 && i != 0);
							} else {
								include = i != 0 && detail::spheric_nosym_true_theta_coef(
									theta_basis, j, k, 2, nbr_coefs(1));
								double_galerkin = include;
							}
						}
						break;
					case SIN:
						if (j != 0 && j != nbr_coefs(1) - 1) {
							if (k < 4) {
								include = (j <= 1) || (i != 0);
								radial_galerkin = (j > 1 && i != 0);
							} else {
								include = i != 0 && detail::spheric_nosym_true_theta_coef(
									theta_basis, j, k, 2, nbr_coefs(1));
								double_galerkin = include;
							}
						}
						break;
					default:
						KADATH_THROW("Unknown theta basis in Domain_nucleus_nosym::describe_tau_seed_block");
				}
				if (!include)
					continue;

				TauSeedDescriptor descriptor;
				descriptor.component = component;
				const std::size_t coefficient_offset =
					static_cast<std::size_t>(i) * radial_stride + row_offset;
				descriptor.writes[0] = {coefficient_offset, 1.};
				descriptor.write_count = 1;
				if (radial_galerkin || double_galerkin) {
					const double radial_factor =
						nucleus_radial_anchor_weight(radial_basis, i);
					descriptor.writes[1] = {row_offset, radial_factor};
					descriptor.write_count = 2;
					if (double_galerkin) {
						const std::size_t anchor_row_offset =
							static_cast<std::size_t>(detail::spheric_nosym_theta_anchor(
								theta_basis, j)) * phi_size +
							static_cast<std::size_t>(k);
						const double theta_factor =
							-detail::spheric_nosym_basis_anchor_weight(theta_basis, j);
						descriptor.writes[2] = {
							static_cast<std::size_t>(i) * radial_stride +
								anchor_row_offset,
							theta_factor};
						descriptor.writes[3] = {
							anchor_row_offset, radial_factor * theta_factor};
						descriptor.write_count = 4;
					}
				}
				descriptors.push_back(descriptor);
			}
		}
	}
}
} // namespace

bool Domain_nucleus_nosym::describe_tau_seed_block(
	const Tensor& tt, int dom, std::vector<TauSeedDescriptor>& descriptors) const
{
	descriptors.clear();
	if (tt.get_space().get_domain(dom) != this)
		return false;
	if (tt.get_valence() > 0 &&
		tt.get_basis().get_basis(dom) != CARTESIAN_BASIS) {
		return false;
	}
	if (tt.get_valence() < 0 || tt.get_valence() > 2 ||
		(tt.get_valence() == 1 && tt.get_n_comp() != 3) ||
		(tt.get_valence() == 2 && tt.get_n_comp() != 6 &&
		 tt.get_n_comp() != 9)) {
		return false;
	}

	for (int component = 0; component < tt.get_n_comp(); ++component) {
		const Array<int> index(tt.indices(component));
		append_nucleus_tau_seed_component(
			nbr_coefs, tt(index)(dom), component, descriptors);
	}
	return descriptors.size() ==
		static_cast<std::size_t>(nbr_unknowns(tt, dom));
}
}
