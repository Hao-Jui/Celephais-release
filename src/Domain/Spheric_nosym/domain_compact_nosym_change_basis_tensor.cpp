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
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath {
Tensor Domain_compact_nosym::change_basis_spher_to_cart (int dd, const Tensor& so) const {

  // Lust start from spherical tensorial basis
  if (so.get_basis().get_basis(dd) != SPHERICAL_BASIS) {
      KADATH_THROW("The input tensorial basis must be spherical in Domain_compact_nosyml::change_basis_spher_to_cart");
  }

  // Need to remove the symetry :
  int val = so.get_valence() ;
  Array<int> type_ind (so.get_index_type()) ;
  Tensor res (so.get_space(), val, type_ind, so.get_basis()) ;

  if (so.is_name_affected()) {
      res.set_name_affected() ;
      for (int i=0 ; i<val ; i++)
	  res.set_name_ind (i, so.get_name_ind()[i]) ;
  }
  for (int i=0 ; i<res.get_n_comp() ; i++)
    res.set(res.indices(i)).set_domain(dd) = so(res.indices(i))(dd) ;

  // Loop on the number of indices :
  Dim_array dimother (so.get_valence()-1) ;
  for (int i=0 ; i<so.get_valence()-1 ; i++)
    dimother.set(i) = 3 ;

  for (int ind=0 ; ind<so.get_valence() ; ind++) {


   Index posother (dimother) ;
    do {


      Index posr (so) ;
      Index post (so) ;
      Index posp (so) ;
      int pos_inother= 0 ;
      for (int conte=0 ; conte<so.get_valence() ; conte++) {
	if (conte==ind) {
	  posr.set(conte) = 0 ;
	  post.set(conte) = 1 ;
	  posp.set(conte) = 2 ;
	}
	else {
	  posr.set(conte) = posother(pos_inother) ;
	  post.set(conte) = posother(pos_inother) ;
	  posp.set(conte) = posother(pos_inother) ;
	  pos_inother ++ ;
	}
      }

     Val_domain tmp (res(posr)(dd).mult_sin_theta() + res(post)(dd).mult_cos_theta()) ;
     Val_domain vx (tmp.mult_cos_phi() - res(posp)(dd).mult_sin_phi()) ;
     Val_domain vy (tmp.mult_sin_phi() + res(posp)(dd).mult_cos_phi()) ;
     Val_domain vz (res(posr)(dd).mult_cos_theta() - res(post)(dd).mult_sin_theta()) ;
     res.set(posr).set_domain(dd) = vx ;
     res.set(post).set_domain(dd) = vy ;
     res.set(posp).set_domain(dd) = vz ;
    }
    while (posother.inc()) ;
  }

  res.set_basis(dd) = CARTESIAN_BASIS ;
  return res ;
}

Tensor Domain_compact_nosym::change_basis_cart_to_spher (int dd, const Tensor& so) const {
  // Must start from spherical tensorial basis
  if (so.get_basis().get_basis(dd) != CARTESIAN_BASIS) {
      KADATH_THROW("The input tensorial basis must be cartesian in Domain_compact_nosym::change_basis_cart_to_spher");
  }
   // Need to remove the symetry :
  int val = so.get_valence() ;
  Array<int> type_ind (so.get_index_type()) ;
  Tensor res (so.get_space(), val, type_ind, so.get_basis()) ;

  if (so.is_name_affected()) {
      res.set_name_affected() ;
      for (int i=0 ; i<val ; i++)
	  res.set_name_ind (i, so.get_name_ind()[i]) ;
  }
  for (int i=0 ; i<res.get_n_comp() ; i++)
    res.set(res.indices(i)).set_domain(dd) = so(res.indices(i))(dd) ;


  // Loop on the number of indices :
  Dim_array dimother (so.get_valence()-1) ;
  for (int i=0 ; i<so.get_valence()-1 ; i++)
    dimother.set(i) = 3 ;

  for (int ind=0 ; ind<so.get_valence() ; ind++) {


   Index posother (dimother) ;
    do {
      Index posx (so) ;
      Index posy (so) ;
      Index posz (so) ;
      int pos_inother= 0 ;
      for (int conte=0 ; conte<so.get_valence() ; conte++) {
	if (conte==ind) {
	  posx.set(conte) = 0 ;
	  posy.set(conte) = 1 ;
	  posz.set(conte) = 2 ;
	}
	else {
	  posx.set(conte) = posother(pos_inother) ;
	  posy.set(conte) = posother(pos_inother) ;
	  posz.set(conte) = posother(pos_inother) ;
	  pos_inother ++ ;
	}
      }

     Val_domain tmp (res(posx)(dd).mult_cos_phi() + res(posy)(dd).mult_sin_phi()) ;
     Val_domain vr (tmp.mult_sin_theta() + res(posz)(dd).mult_cos_theta()) ;
     Val_domain vt (tmp.mult_cos_theta() - res(posz)(dd).mult_sin_theta()) ;
     Val_domain vp (-res(posx)(dd).mult_sin_phi() + res(posy)(dd).mult_cos_phi()) ;

     res.set(posx).set_domain(dd) = vr ;
     res.set(posy).set_domain(dd) = vt ;
     res.set(posz).set_domain(dd) = vp ;
    }
    while (posother.inc()) ;
  }


  res.set_basis(dd) = SPHERICAL_BASIS ;
  return res ;
}}
