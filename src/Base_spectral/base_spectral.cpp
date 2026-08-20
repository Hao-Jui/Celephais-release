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
 *   2026-08-11  Shared the outer basis-slot snapshot and deferred its
 *               three-dimensional allocation until first mutation.
 */

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/array.hpp"

namespace Kadath
{
    Base_spectral::Base_spectral(int dimensions) : def(false), ndim(dimensions), bases_1d(dimensions) {}

    Base_spectral::Base_spectral(const Base_spectral& so)
        : def(so.def), ndim(so.ndim), bases_1d(so.bases_1d)
    {}

    Base_spectral::Base_spectral(Base_spectral&& so) noexcept
        : def{so.def}, ndim{so.ndim}, bases_1d{std::move(so.bases_1d)}
    {
        so.def = false;
    }

    void Base_spectral::swap(Base_spectral& so) noexcept
    {
        std::swap(def, so.def);
        std::swap(ndim, so.ndim);
        bases_1d.swap(so.bases_1d);
    }

    Base_spectral::~Base_spectral() = default;

    void Base_spectral::set_non_def()
    {
        bases_1d.clear();
        def = false;
    }

    Base_spectral::Base_spectral(BinarySource& source)
    {
        const int indic = source.read<int>();
        def = (indic == 0);
        ndim = source.read<int>();
        bases_1d.reset(ndim);
        if (def) {
            for (int i = 0; i < ndim; i++)
                bases_1d[i] = std::make_unique<Array<int>>(source);
        }
    }

    void Base_spectral::save(BinarySink& sink) const
    {
        sink.write<int>(def ? 0 : 1);
        sink.write<int>(ndim);
        if (def) {
            for (int i = 0; i < ndim; i++)
                bases_1d[i]->save(sink);
        }
    }

    void Base_spectral::operator=(const Base_spectral& so)
    {
        assert(ndim == so.ndim);
        def = so.def;
        if (so.def)
            bases_1d = so.bases_1d;
        else
            bases_1d.clear();
    }

    Base_spectral& Base_spectral::operator=(Base_spectral&& so) noexcept
    {
        swap(so);
        return *this;
    }

    void Base_spectral::allocate(const Dim_array& nbr_coef)
    {
        bases_1d[ndim - 1] = std::make_unique<Array<int>>(1);
        for (int i = ndim - 2; i >= 0; i--) {
            Dim_array taille(ndim - 1 - i);
            for (int k = 0; k < ndim - 1 - i; k++)
                taille.set(k) = nbr_coef(i + k + 1);
            bases_1d[i] = std::make_unique<Array<int>>(taille);
        }
    }

    bool operator==(const Base_spectral& a, const Base_spectral& b)
    {
        bool res = true;
        if ((!a.def) || (!b.def) || (a.ndim != b.ndim))
            res = false;
        for (int i = 0; i < a.ndim; i++) {
            Index index(a.bases_1d[i]->get_dimensions());
            do
                if ((*a.bases_1d[i])(index) != (*b.bases_1d[i])(index))
                    res = false;
            while (index.inc());
        }
        return res;
    }

    ostream& operator<<(ostream& o, const Base_spectral& so)
    {

        o << so.ndim << "-dimensional spectral base" << endl;
        if (so.def) {
            for (int l = 0; l < so.ndim; l++) {
                o << "Variable " << l << endl;
                o << *so.bases_1d[l] << endl;
            }
        } else
            o << "Base not defined" << endl;
        return o;
    }

    void Base_spectral::set(Dim_array const& nbr_coefs, int BASEPHI, int BASETHETA, int BASER)
    {

        assert(nbr_coefs.get_ndim() == 3);

        allocate(nbr_coefs);
        def = true;
        bases_1d[2]->set(0) = BASEPHI;
        Index index(bases_1d[0]->get_dimensions());
        for (int k(0); k < nbr_coefs(2); ++k) {
            bases_1d[1]->set(k) = BASETHETA;
            for (int j(0); j < nbr_coefs(1); ++j) {
                index.set(0) = j;
                index.set(1) = k;
                bases_1d[0]->set(index) = BASER;
            }
        }
    }
} // namespace Kadath
