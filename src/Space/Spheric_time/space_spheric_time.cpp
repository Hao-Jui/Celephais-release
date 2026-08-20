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
#include "For_Kadath/Domain/spheric_time.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Space_spheric_time::Space_spheric_time(int ttype, const Dim_array& res, const Array<double>& bounds, double ttmin,
                                           double ttmax, bool wc)
        : tmin(ttmin), tmax(ttmax), withcompact(wc)
    {

        // Verif :
        assert(bounds.get_ndim() == 1);

        ndim = 3;

        nbr_domains = (withcompact) ? bounds.get_size(0) + 1 : bounds.get_size(0);
        type_base = ttype;
        domains = new Domain*[nbr_domains];
        // Nucleus
        domains[0] = new Domain_spheric_time_nucleus(0, ttype, tmin, tmax, bounds(0), res);
        if (!withcompact) {
            for (int i = 1; i < nbr_domains; i++)
                domains[i] = new Domain_spheric_time_shell(i, ttype, tmin, tmax, bounds(i - 1), bounds(i), res);
        } else {
            for (int i = 1; i < nbr_domains - 1; i++)
                domains[i] = new Domain_spheric_time_shell(i, ttype, tmin, tmax, bounds(i - 1), bounds(i), res);
            domains[nbr_domains - 1] =
                new Domain_spheric_time_compact(nbr_domains - 1, ttype, tmin, tmax, bounds(nbr_domains - 2), res);
        }
    }

    Space_spheric_time::Space_spheric_time(BinarySource& source)
    {
        nbr_domains = source.read<int>();
        ndim = source.read<int>();
        type_base = source.read<int>();
        tmin = source.read<double>();
        tmax = source.read<double>();
        int indcomp = source.read<int>();
        withcompact = (indcomp == 1) ? true : false;
        domains = new Domain*[nbr_domains];
        domains[0] = new Domain_spheric_time_nucleus(0, source);
        if (!withcompact) {
            for (int i = 1; i < nbr_domains; i++)
                domains[i] = new Domain_spheric_time_shell(i, source);
        } else {
            for (int i = 1; i < nbr_domains - 1; i++)
                domains[i] = new Domain_spheric_time_shell(i, source);
            domains[nbr_domains - 1] = new Domain_spheric_time_compact(nbr_domains - 1, source);
        }
    }

    Space_spheric_time::~Space_spheric_time() {}

    void Space_spheric_time::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        sink.write<double>(tmin);
        sink.write<double>(tmax);
        int indcomp = (withcompact) ? 1 : 0;
        sink.write<int>(indcomp);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }
} // namespace Kadath
