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
#include "For_Kadath/Domain/oned.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Space_oned::Space_oned(int ttype, const Dim_array& res, const Array<double>& bounds)
    {

        // Verif :
        assert(bounds.get_ndim() == 1);

        ndim = 1;

        nbr_domains = bounds.get_size(0) + 1;
        type_base = ttype;
        domains = new Domain*[nbr_domains];

        domains[0] = new Domain_oned_ori(0, ttype, bounds(0), res);
        for (int i = 1; i < nbr_domains - 1; i++)
            domains[i] = new Domain_oned_qcq(i, ttype, bounds(i - 1), bounds(i), res);
        domains[nbr_domains - 1] = new Domain_oned_inf(nbr_domains - 1, ttype, bounds(nbr_domains - 2), res);
    }

    Space_oned::Space_oned(BinarySource& source)
    {
        nbr_domains = source.read<int>();
        ndim = source.read<int>();
        type_base = source.read<int>();
        domains = new Domain*[nbr_domains];
        domains[0] = new Domain_oned_ori(0, source);
        for (int i = 1; i < nbr_domains - 1; i++)
            domains[i] = new Domain_oned_qcq(i, source);
        domains[nbr_domains - 1] = new Domain_oned_inf(nbr_domains - 1, source);
    }

    Space_oned::~Space_oned() {}

    void Space_oned::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }
} // namespace Kadath
