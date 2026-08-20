/*
    Copyright 2017 Philippe Grandclement

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

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Space/bhns_nosym.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "../Shared/binary_co_add_eq_common.hpp"

namespace Kadath
{

    void Space_bhns_nosym::add_bc_sphere_one(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(ADAPTEDNS + 1, INNER_BC, name, nused, pused);
    }

    void Space_bhns_nosym::add_bc_sphere_two(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(ADAPTEDBH + 1, INNER_BC, name, nused, pused);
    }

    void Space_bhns_nosym::add_bc_outer(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(OUTER + n_shells_outer + 5, OUTER_BC, name, nused, pused);
    }

    void Space_bhns_nosym::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der, int nused,
                                  Array<int>** pused)
    {
        BinaryCoSpaceEquations::bhns_add_eq(*this, sys, eq, rac, rac_der, nused, pused);
    }

    void Space_bhns_nosym::add_eq_int_inf(System_of_eqs& sys, const char* nom)
    {
        BinaryCoSpaceEquations::add_eq_int_inf<Domain_compact_nosym>(*this, sys, nom);
    }

    void Space_bhns_nosym::add_eq_int_volume(System_of_eqs& sys, int dmin, int dmax, const char* nom)
    {
        BinaryCoSpaceEquations::add_eq_int_volume(sys, dmin, dmax, nom);
    }

    void Space_bhns_nosym::add_eq_ori_NS(System_of_eqs& sys, const char* name)
    {
        BinaryCoSpaceEquations::add_eq_val_origin(*this, sys, NS, name);
    }

    void Space_bhns_nosym::add_eq_int_outer_NS(System_of_eqs& sys, const char* nom)
    {
        BinaryCoSpaceEquations::add_eq_int_bc(sys, ADAPTEDNS + 1, OUTER_BC, nom);
    }

    void Space_bhns_nosym::add_eq_int_BH(System_of_eqs& sys, const char* nom)
    {
        BinaryCoSpaceEquations::add_eq_int_bc_parse_first(sys, ADAPTEDBH + 1, INNER_BC, nom);
    }

} // namespace Kadath
