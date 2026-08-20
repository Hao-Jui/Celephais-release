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
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Space_critic::Space_critic(int ttype, double xlim, const Dim_array& res_inner, const Dim_array& res_outer)
    {

        ndim = 2;
        nbr_domains = 2;
        type_base = ttype;
        // Two domains
        domains = new Domain*[2];
        // Inner one
        domains[0] = new Domain_critic_inner(0, ttype, res_inner, xlim);
        // Outer one
        domains[1] = new Domain_critic_outer(1, ttype, res_outer, xlim);
    }

    Space_critic::Space_critic(BinarySource& source)
    {
        nbr_domains = 2;
        ndim = source.read<int>();
        type_base = source.read<int>();
        domains = new Domain*[2];
        domains[0] = new Domain_critic_inner(0, source);
        domains[1] = new Domain_critic_outer(1, source);
    }

    Space_critic::~Space_critic() {}

    void Space_critic::save(BinarySink& sink) const
    {
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        domains[0]->save(sink);
        domains[1]->save(sink);
    }

    Array<int> Space_critic::get_indices_matching_non_std(int dom, int bound) const
    {

        switch (dom) {
            case 0: {
                // Inner
                Array<int> res(2, 1);
                switch (bound) {
                    case OUTER_BC:
                        res.set(0, 0) = 1; // Outer domain
                        res.set(1, 0) = INNER_BC;
                        break;
                    default:
                        KADATH_THROW("Bad bound in Space_critic::get_indices_matching_non_std");
                }
                return res;
            }
            case 1: {
                // Outer domain
                Array<int> res(2, 1);
                switch (bound) {
                    case INNER_BC:
                        res.set(0, 0) = 0; // Inner domain
                        res.set(1, 0) = OUTER_BC;
                        break;
                    default:
                        KADATH_THROW("Bad bound in Space_critic::get_indices_matching_non_std");
                }
                return res;
            }
            default:
                KADATH_THROW("Bad domain in Space_critic::get_indices_matching_non_std");
        }
    }
} // namespace Kadath
