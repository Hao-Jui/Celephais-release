/*
    Copyright 2020 Samuel Tootle

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

#include "For_Kadath/Space/binary_co_domains.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Domain/adapted.hpp"

namespace Kadath
{
    void build_adapted_star_domains(const Space& space, Domain** domains, int nucleus_index, int adapted_index,
                                    int ttype, const std::vector<double>& bounds, int n_outer_shells,
                                    const Point& center, const std::vector<Dim_array>& res_per_domain)
    {
        // Bounds layout: [rin, rmid, rout, outer shells..., r_bisph]. The
        // adapted pair starts right after the nucleus; rmid/rout are the inner
        // adapted shell boundaries and any ordinary shells sit outside rout.
        const int rin = 0;
        const int rmid = 1;

        domains[nucleus_index] =
            new Domain_nucleus(nucleus_index, ttype, bounds[rin], center, res_per_domain[nucleus_index]);

        domains[adapted_index] =
            new Domain_shell_outer_adapted(space, adapted_index, ttype, bounds[rin], bounds[rmid], center,
                                           res_per_domain[adapted_index]);
        domains[adapted_index + 1] =
            new Domain_shell_inner_adapted(space, adapted_index + 1, ttype, bounds[rmid], bounds[rmid + 1], center,
                                           res_per_domain[adapted_index + 1]);

        for (int i = 0; i < n_outer_shells; ++i)
            domains[adapted_index + 2 + i] = new Domain_shell(adapted_index + 2 + i, ttype, bounds[rmid + 1 + i],
                                                              bounds[rmid + 1 + i + 1], center,
                                                              res_per_domain[adapted_index + 2 + i]);
    }

    void build_adapted_star_domains(const Space& space, Domain** domains, int nucleus_index, int adapted_index,
                                    int ttype, const std::vector<double>& bounds, int n_outer_shells,
                                    const Point& center, const Dim_array& res)
    {
        const std::vector<Dim_array> uniform(static_cast<std::size_t>(adapted_index + 2 + n_outer_shells), res);
        build_adapted_star_domains(space, domains, nucleus_index, adapted_index, ttype, bounds, n_outer_shells,
                                   center, uniform);
    }
} // namespace Kadath
