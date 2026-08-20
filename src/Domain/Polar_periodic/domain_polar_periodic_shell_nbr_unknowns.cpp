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
#include "For_Kadath/Domain/polar_periodic.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/tensor.hpp"

namespace Kadath
{
    int Domain_polar_periodic_shell::nbr_unknowns_val_domain(const Val_domain& so) const
    {

        int res = 0;
        Index pos(nbr_coefs);
        do {
            bool indic = true;

            // Base in time
            int basetime = (*so.get_base().bases_1d[2])(0);
            switch (basetime) {
                case COS:
                    break;
                case SIN:
                    if ((pos(2) == 0) || (pos(2) == nbr_coefs(2) - 1))
                        indic = false;
                    break;
                default:
                    KADATH_THROW("Unknow time basis in Domain_polar_polar_shell::nbr_unknowns_val_domain");
            }

            // Get base in theta :
            int baset = (*so.get_base().bases_1d[1])(pos(2));
            switch (baset) {
                case COS_EVEN:
                    break;
                case COS_ODD:
                    if (pos(1) == nbr_coefs(1) - 1)
                        indic = false;
                    break;
                case SIN_EVEN:
                    if ((pos(1) == 0) || (pos(1) == nbr_coefs(1) - 1))
                        indic = false;
                    break;
                case SIN_ODD:
                    if (pos(1) == nbr_coefs(1) - 1)
                        indic = false;
                    break;
                default:
                    KADATH_THROW("Unknow theta basis in Domain_polar_periodic_shell::nbr_unknowns_val_domain");
            }

            if (indic)
                res++;
        } while (pos.inc());

        return res;
    }

    int Domain_polar_periodic_shell::nbr_unknowns(const Tensor& tt, int dom) const
    {

        // Check right domain
        assert(tt.get_space().get_domain(dom) == this);

        int res = 0;
        int val = tt.get_valence();
        switch (val) {
            case 0:
                res += nbr_unknowns_val_domain(tt()(dom));
                break;
            case 1:
                res += nbr_unknowns_val_domain(tt(1)(dom));
                res += nbr_unknowns_val_domain(tt(2)(dom));
                res += nbr_unknowns_val_domain(tt(3)(dom));
                break;
            case 2:
                if (tt.get_n_comp() == 6) {
                    res += nbr_unknowns_val_domain(tt(1, 1)(dom));
                    res += nbr_unknowns_val_domain(tt(1, 2)(dom));
                    res += nbr_unknowns_val_domain(tt(1, 3)(dom));
                    res += nbr_unknowns_val_domain(tt(2, 2)(dom));
                    res += nbr_unknowns_val_domain(tt(2, 3)(dom));
                    res += nbr_unknowns_val_domain(tt(3, 3)(dom));
                } else {
                    res += nbr_unknowns_val_domain(tt(1, 1)(dom));
                    res += nbr_unknowns_val_domain(tt(1, 2)(dom));
                    res += nbr_unknowns_val_domain(tt(1, 3)(dom));
                    res += nbr_unknowns_val_domain(tt(2, 1)(dom));
                    res += nbr_unknowns_val_domain(tt(2, 2)(dom));
                    res += nbr_unknowns_val_domain(tt(2, 3)(dom));
                    res += nbr_unknowns_val_domain(tt(3, 1)(dom));
                    res += nbr_unknowns_val_domain(tt(3, 2)(dom));
                    res += nbr_unknowns_val_domain(tt(3, 3)(dom));
                }
                break;
            default:
                cerr << "Valence " << val << " not implemented in Domain_polar_periodic_shell::nbr_unknowns" << endl;
                break;
        }
        return res;
    }
} // namespace Kadath
