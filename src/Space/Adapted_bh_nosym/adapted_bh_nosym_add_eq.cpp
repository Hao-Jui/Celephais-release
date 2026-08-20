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
#include "For_Kadath/Space/adapted_bh_nosym.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "../Shared/binary_co_add_eq_common.hpp"

namespace Kadath
{
    void Space_adapted_bh_nosym::add_bc_bh(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(2, INNER_BC, name, nused, pused);
    }
    void Space_adapted_bh_nosym::add_bc_inf(System_of_eqs& sys, const char* name, int nused, Array<int>** pused)
    {
        sys.add_eq_bc(nbr_domains - 1, OUTER_BC, name);
    }

    void Space_adapted_bh_nosym::add_eq(System_of_eqs& sys, const char* eq, const char* rac, const char* rac_der,
                                        int nused, Array<int>** pused)
    {
        BinaryCoSpaceEquations::adapted_bh_add_eq(sys, eq, rac, rac_der, nused, pused);
    }

    void Space_adapted_bh_nosym::add_eq_int_inf(System_of_eqs& sys, const char* nom)
    {
        BinaryCoSpaceEquations::add_eq_int_inf_check_first<Domain_compact_nosym>(*this, sys, nom);
    }

    void Space_adapted_bh_nosym::add_eq_int_bh(System_of_eqs& sys, const char* nom)
    {
        BinaryCoSpaceEquations::add_eq_int_bc_parse_first(sys, 2, INNER_BC, nom);
    }

    void Space_adapted_bh_nosym::add_eq_zero_mode_inf(System_of_eqs& sys, const char* name, int j, int k)
    {
        BinaryCoSpaceEquations::add_eq_zero_mode_inf(*this, sys, name, j, k);
    }

} // namespace Kadath
