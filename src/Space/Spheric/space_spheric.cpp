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

#include <sstream>
#include "For_Kadath/Domain/spheric.hpp"
#include <assert.h>
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Space_spheric::Space_spheric(int ttype, const Point& center, const Dim_array& res, const Array<double>& bounds,
                                 bool withzec)
    {

        // Verif :
        assert(bounds.get_ndim() == 1);

        ndim = 3;

        nbr_domains = (withzec) ? bounds.get_size(0) + 1 : bounds.get_size(0);
        type_base = ttype;
        domains = new Domain*[nbr_domains];
        // Nucleus
        domains[0] = new Domain_nucleus(0, ttype, bounds(0), center, res);
        if (withzec) {
            for (int i = 1; i < nbr_domains - 1; i++)
                domains[i] = new Domain_shell(i, ttype, bounds(i - 1), bounds(i), center, res);

            domains[nbr_domains - 1] = new Domain_compact(nbr_domains - 1, ttype, bounds(nbr_domains - 2), center, res);
        } else {
            for (int i = 1; i < nbr_domains; i++)
                domains[i] = new Domain_shell(i, ttype, bounds(i - 1), bounds(i), center, res);
        }
    }

    Space_spheric::Space_spheric(int ttype, const Point& center, Dim_array** res, const Array<double>& bounds)
    {

        // Verif :
        assert(bounds.get_ndim() == 1);

        ndim = 3;

        nbr_domains = bounds.get_size(0) + 1;
        type_base = ttype;
        domains = new Domain*[nbr_domains];
        // Nucleus
        domains[0] = new Domain_nucleus(0, ttype, bounds(0), center, *res[0]);
        for (int i = 1; i < nbr_domains - 1; i++)
            domains[i] = new Domain_shell(i, ttype, bounds(i - 1), bounds(i), center, *res[i]);

        domains[nbr_domains - 1] =
            new Domain_compact(nbr_domains - 1, ttype, bounds(nbr_domains - 2), center, *res[nbr_domains - 1]);
    }

    Space_spheric::Space_spheric(int ttype, const Point& center, const Dim_array& res, const Array<double>& bounds,
                                 const Array<int>& type_shells)
    {

        // Verif :
        assert(bounds.get_ndim() == 1);

        ndim = 3;

        nbr_domains = bounds.get_size(0) + 1;
        type_base = ttype;
        domains = new Domain*[nbr_domains];
        // Nucleus
        domains[0] = new Domain_nucleus(0, ttype, bounds(0), center, res);
        for (int i = 1; i < nbr_domains - 1; i++) {
            switch (type_shells(i - 1)) {
                case STD_TYPE:
                    domains[i] = new Domain_shell(i, ttype, bounds(i - 1), bounds(i), center, res);
                    break;
                case LOG_TYPE:
                    domains[i] = new Domain_shell_log(i, ttype, bounds(i - 1), bounds(i), center, res);
                    break;
                case SURR_TYPE:
                    domains[i] = new Domain_shell_surr(i, ttype, bounds(i - 1), bounds(i), center, res);
                    break;
                default:
                    KADATH_THROW("Unknown type of shell");
            }
        }

        domains[nbr_domains - 1] = new Domain_compact(nbr_domains - 1, ttype, bounds(nbr_domains - 2), center, res);
    }

    Space_spheric::Space_spheric(BinarySource& source, bool withzec)
    {
        nbr_domains = source.read<int>();
        ndim = source.read<int>();
        type_base = source.read<int>();
        domains = new Domain*[nbr_domains];
        domains[0] = new Domain_nucleus(0, source);
        if (withzec) {
            for (int i = 1; i < nbr_domains - 1; i++)
                domains[i] = new Domain_shell(i, source);
            domains[nbr_domains - 1] = new Domain_compact(nbr_domains - 1, source);
        } else {
            for (int i = 1; i < nbr_domains; i++)
                domains[i] = new Domain_shell(i, source);
        }
    }

    Space_spheric::~Space_spheric() {}

    void Space_spheric::save(BinarySink& sink) const
    {
        sink.write<int>(nbr_domains);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        for (int i = 0; i < nbr_domains; i++)
            domains[i]->save(sink);
    }

    Array<int> Space_spheric::get_indices_matching_non_std(int dom, int bound) const
    {

        assert((dom >= 0) && (dom < nbr_domains));
        Array<int> res(2, 1);
        switch (bound) {
            case OUTER_BC:
                res.set(0, 0) = dom + 1;
                res.set(1, 0) = INNER_BC;
                break;
            case INNER_BC:
                res.set(0, 0) = dom - 1;
                res.set(1, 0) = OUTER_BC;
                break;
            default:
                cerr << "Unknown boundary in " << endl;
                std::ostringstream oss;
                oss << *this << endl;
                KADATH_THROW(oss.str());
        }
        return res;
    }

    double Space_spheric::int_inf(const Scalar& so) const
    {

        const Domain_compact* p_cmp = dynamic_cast<const Domain_compact*>(domains[nbr_domains - 1]);
        if (p_cmp == nullptr) {
            KADATH_THROW("No compactified domain in Space_spheric::int_inf");
        }
        return p_cmp->integ(so(nbr_domains - 1), OUTER_BC);
    }
} // namespace Kadath
