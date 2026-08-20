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

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/critic.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    int Domain_critic_inner::nbr_conditions_val_domain_boundary(const Val_domain& so) const
    {
        int base_t = (*so.get_base().bases_1d[1])(0);
        if ((base_t != COSSIN_EVEN) && (base_t != COSSIN_ODD)) {
            KADATH_THROW("Unknown base in Domain_critic_inner::nbr_conditions_val_domain_boundary");
        }
        return nbr_coefs(1) - 2;
    }

    Array<int> Domain_critic_inner::nbr_conditions_boundary(const Tensor& tt, int dom, int bound, int n_cmp,
                                                            Array<int>**) const
    {

        // Check boundary
        if ((bound != INNER_BC) && (bound != OUTER_BC)) {
            KADATH_THROW("Unknown boundary in Domain_critic_inner::nbr_conditions_boundary");
        }

        int size = (n_cmp == -1) ? tt.get_n_comp() : n_cmp;
        Array<int> res(size);
        int val = tt.get_valence();
        switch (val) {
            case 0:
                res.set(0) = nbr_conditions_val_domain_boundary(tt()(dom));
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_critic_inner::nbr_conditions_boundary" << endl;
                break;
        }
        return res;
    }
} // namespace Kadath
