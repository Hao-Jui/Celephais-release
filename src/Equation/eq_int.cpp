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
 *   2026-08-06  RAII/span modernization.
 */

#include <algorithm>

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    Eq_int::Eq_int(int np) : n_ope(np), parts(n_ope) {}

    Eq_int::~Eq_int() = default;

    double Eq_int::get_val() const
    {
        double res = 0.;
        for (int i = 0; i < n_ope; i++) {
            res += parts[i]->action().get_val_d();
        }
        return res;
    }

    double Eq_int::get_der() const
    {
        double res = 0.;
        for (int i = 0; i < n_ope; i++)
            res += parts[i]->action().get_der_d();
        return res;
    }

    double Eq_int::get_der(int lane) const
    {
        double res = 0.;
        for (int i = 0; i < n_ope; i++) {
            const Term_eq action(parts[i]->action());
            if (action.has_der_d(lane))
                res += action.get_der_d(lane);
        }
        return res;
    }

    void Eq_int::get_der_lanes(int lane_count, double* derivatives) const
    {
        std::fill(derivatives, derivatives + lane_count, 0.);
        for (int i = 0; i < n_ope; i++) {
            const Term_eq action(parts[i]->action());
            for (int lane = 0; lane < lane_count; ++lane) {
                if (action.has_der_d(lane))
                    derivatives[lane] += action.get_der_d(lane);
            }
        }
    }

    void Eq_int::set_part(int pos, Ope_eq* so)
    {
        assert((pos >= 0) && (pos < n_ope));
        parts[pos].reset(so);
    }

    void Eq_int::set_reflection_sector(int sector)
    {
        if (sector != -1 && sector != 1)
            KADATH_THROW("Eq_int reflection sector must be -1 or +1");
        if (reflection_sector != 0 && reflection_sector != sector)
            KADATH_THROW("Eq_int reflection sector conflicts with its existing tag");
        reflection_sector = static_cast<signed char>(sector);
    }
} // namespace Kadath
