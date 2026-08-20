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

#pragma once

#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/IO/binary_source.hpp"
#include "space.hpp"
#include <vector>

namespace Kadath
{
    /**
     * Builds the domain sequence for one neutron-star compact object, shared by
     * \c Space_bhns (NS side) and \c Space_bin_ns (both stars). The sequence is,
     * in increasing domain index:
     * \li one \c Domain_nucleus of radius \c bounds[0],
     * \li one \c Domain_shell_outer_adapted (the variable outer surface, inside
     *     the star),
     * \li one \c Domain_shell_inner_adapted (the variable inner surface, outside
     *     the star),
     * \li \c n_outer_shells ordinary \c Domain_shell (outside the star, before
     *     the bispheric region).
     *
     * The bounds vector is laid out as
     * \f$[r_{in}, r_{mid}, r_{out}, \text{outer shells}\ldots, r_{bisph}]\f$.
     * \f$r_{out}\f$ is the outer edge of the inner adapted shell; when no
     * ordinary shells are present it is also the bispheric matching radius.
     * \c adapted_index must equal \c nucleus_index+1, so the outer/inner
     * adapted pair occupies \c adapted_index and \c adapted_index+1.
     *
     * @param space [input] : the owning space (passed to the adapted-domain
     *        constructors).
     * @param domains [input/output] : the domain table being populated.
     * @param nucleus_index [input] : domain index of the star nucleus.
     * @param adapted_index [input] : domain index of the outer adapted shell.
     * @param ttype [input] : the spectral basis type.
     * @param bounds [input] : the radial boundaries (see layout above).
     * @param n_outer_shells [input] : number of ordinary shells after the
     *        adapted pair.
     * @param center [input] : the star center.
     * @param res [input] : the per-domain resolution.
     */
    void build_adapted_star_domains(const Space& space, Domain** domains, int nucleus_index, int adapted_index,
                                    int ttype, const std::vector<double>& bounds, int n_outer_shells,
                                    const Point& center, const Dim_array& res);

    /**
     * Per-domain-resolution overload of \c build_adapted_star_domains.
     * \c res_per_domain is indexed by ABSOLUTE domain index (same indexing as
     * \c domains) and must cover at least
     * \c [nucleus_index, adapted_index+1+n_outer_shells]; each entry is the full
     * \c Dim_array for that domain. Used by per-domain p-refinement, where the
     * star's domains may carry different radial resolutions.
     */
    void build_adapted_star_domains(const Space& space, Domain** domains, int nucleus_index, int adapted_index,
                                    int ttype, const std::vector<double>& bounds, int n_outer_shells,
                                    const Point& center, const std::vector<Dim_array>& res_per_domain);

    /**
     * Deserialise mirror of \c build_adapted_star_domains: reads one adapted
     * compact object's domains from \c source in increasing index order —
     * nucleus, outer-adapted, inner-adapted, then \c n_outer_shells ordinary
     * shells (the per-star outer shells). \c adapted_index must equal
     * \c nucleus_index+1 (no inner shells). Templated on the nucleus, adapted
     * outer/inner, and ordinary-shell domain types so the sym
     * (\c Domain_nucleus, \c Domain_shell_{outer,inner}_adapted, \c Domain_shell)
     * and nosym (\c *_nosym) binary-NS / BHNS spaces share one read path and
     * stay in sync. The owning space's deserialising constructor supplies the
     * indices and the n_outer_shells count it read from the file header.
     */
    template <typename Nucleus, typename OuterAdapted, typename InnerAdapted, typename Shell>
    inline void read_adapted_star_domains(const Space& space, Domain** domains, int nucleus_index, int adapted_index,
                                          int n_outer_shells, BinarySource& source)
    {
        domains[nucleus_index] = new Nucleus(nucleus_index, source);
        domains[adapted_index] = new OuterAdapted(space, adapted_index, source);
        domains[adapted_index + 1] = new InnerAdapted(space, adapted_index + 1, source);
        for (int i = 0; i < n_outer_shells; ++i)
            domains[adapted_index + 2 + i] = new Shell(adapted_index + 2 + i, source);
    }
} // namespace Kadath
