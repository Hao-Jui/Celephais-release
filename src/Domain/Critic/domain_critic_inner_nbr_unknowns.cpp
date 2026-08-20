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
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
namespace Kadath
{
    int Domain_critic_inner::nbr_unknowns_val_domain(const Val_domain& so) const
    {

        int res = 0;
        Index pos(nbr_coefs);
        do {
            bool indic = true;
            int base_t = (*so.get_base().bases_1d[1])(0);

            assert((base_t == COSSIN_EVEN) || (base_t == COSSIN_ODD));
            if (pos(1) == nbr_coefs(1) - 1)
                indic = false;
            if ((pos(1) == 1) && (base_t == COSSIN_EVEN))
                indic = false;
            if ((pos(1) == nbr_coefs(1) - 2) && (base_t == COSSIN_ODD))
                indic = false;

            int base_r = (*so.get_base().bases_1d[0])(0);
            bool even;
            switch (base_r) {
                case CHEB_EVEN:
                    even = true;
                    break;
                case LEG_EVEN:
                    even = true;
                    break;
                case CHEB_ODD:
                    even = false;
                    break;
                case LEG_ODD:
                    even = false;
                    break;
                default:
                    KADATH_THROW("Uknown base in Domain_critic_inner::nbr_unknowns");
            }

            if ((!even) && (pos(0) == nbr_coefs(0) - 1))
                indic = false;
            if (indic)
                res++;
        } while (pos.inc());
        return res;
    }

    int Domain_critic_inner::nbr_unknowns(const Tensor& tt, int dom) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int res = 0;
        int val = tt.get_valence();
        switch (val) {
            case 0:
                res += nbr_unknowns_val_domain(tt()(dom));
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_critic_inner::nbr_unknowns" << endl;
                break;
        }
        return res;
    }
} // namespace Kadath
