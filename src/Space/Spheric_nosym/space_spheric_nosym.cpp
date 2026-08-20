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
#include <assert.h>
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath {
Space_spheric_nosym::Space_spheric_nosym(int ttype, const Point& center, const Dim_array& res, const Array<double>& bounds, bool withzec) {

    // Verif :
    assert (bounds.get_ndim()== 1) ;

    ndim = 3 ;

    nbr_domains = (withzec) ? bounds.get_size(0)+1 : bounds.get_size(0) ;
    type_base = ttype ;
    domains = new Domain* [nbr_domains] ;
    // Nucleus
    domains[0] = new Domain_nucleus_nosym(0, ttype, bounds(0), center, res) ;
    if (withzec) {
   for (int i=1 ; i<nbr_domains-1 ; i++)
       domains[i] = new Domain_shell_nosym(i, ttype, bounds(i-1), bounds(i), center, res) ;

   domains[nbr_domains-1] = new Domain_compact_nosym (nbr_domains-1, ttype, bounds(nbr_domains-2), center, res) ;
   }
	else {
	for (int i=1 ; i<nbr_domains ; i++)
	       domains[i] = new Domain_shell_nosym(i, ttype, bounds(i-1), bounds(i), center, res) ;
   }

}


Space_spheric_nosym::Space_spheric_nosym(BinarySource& source, bool withzec) {
	nbr_domains = source.read<int>();
	ndim = source.read<int>();
	type_base = source.read<int>();
	domains = new Domain* [nbr_domains] ;
	//nucleus :
	domains[0] = new Domain_nucleus_nosym(0, source) ;
		if (withzec) {
		//Shells :
		for (int i=1 ; i<nbr_domains-1 ; i++)
			domains[i] = new Domain_shell_nosym(i, source) ;
		// Compactified
		domains[nbr_domains-1] = new Domain_compact_nosym(nbr_domains-1, source) ;
	}
	else {
		//Shells :
		for (int i=1 ; i<nbr_domains ; i++)
			domains[i] = new Domain_shell_nosym(i, source) ;
	}
}

Space_spheric_nosym::~Space_spheric_nosym() {
}

void Space_spheric_nosym::save(BinarySink& sink) const  {
	sink.write<int>(nbr_domains);
	sink.write<int>(ndim);
	sink.write<int>(type_base);
	for (int i=0 ; i<nbr_domains ; i++)
		domains[i]->save(sink) ;
}

Array<int> Space_spheric_nosym::get_indices_matching_non_std(int dom, int bound) const {

	assert ((dom>=0) && (dom<nbr_domains)) ;
	Array<int> res (2, 1) ;
	switch (bound) {
		case OUTER_BC :
			res.set(0,0) = dom+1 ;
			res.set(1,0) = INNER_BC ;
			break ;
		case INNER_BC :
			res.set(0,0) = dom-1 ;
			res.set(1,0) = OUTER_BC ;
			break ;
		default :
			KADATH_THROW("Unknown boundary in ");
		}
	return res ;
}
}
