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

#include "For_Kadath/Val_domain/val_domain.hpp"
namespace Kadath
{
    Val_domain Val_domain::mult_cos_phi() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->mult_cos_phi(*this));
    }

    Val_domain Val_domain::mult_sin_phi() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->mult_sin_phi(*this));
    }

    Val_domain Val_domain::mult_cos_theta() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->mult_cos_theta(*this));
    }

    Val_domain Val_domain::mult_sin_theta() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->mult_sin_theta(*this));
    }

    Val_domain Val_domain::div_sin_theta() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->div_sin_theta(*this));
    }

    Val_domain Val_domain::div_cos_theta() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->div_cos_theta(*this));
    }

    Val_domain Val_domain::div_x() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->div_x(*this));
    }

    Val_domain Val_domain::div_chi() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->div_chi(*this));
    }

    Val_domain Val_domain::div_xm1() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->div_xm1(*this));
    }

    Val_domain Val_domain::div_1mx2() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->div_1mx2(*this));
    }

    Val_domain Val_domain::div_1mrsL() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->div_1mrsL(*this));
    }

    Val_domain Val_domain::div_xp1() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->div_xp1(*this));
    }

    Val_domain Val_domain::mult_xm1() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->mult_xm1(*this));
    }

    Val_domain Val_domain::div_sin_chi() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->div_sin_chi(*this));
    }

    Val_domain Val_domain::mult_cos_time() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->mult_cos_time(*this));
    }

    Val_domain Val_domain::mult_sin_time() const
    {
        if (is_zero)
            return *this;
        else
            return (zone->mult_sin_time(*this));
    }
} // namespace Kadath
