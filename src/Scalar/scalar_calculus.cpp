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

#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"
namespace Kadath
{
    Vector Scalar::grad() const
    {

        Base_tensor base(espace, CARTESIAN_BASIS);
        Vector res(espace, COV, base);
        for (int i = 0; i < ndim; i++)
            res.set(i + 1) = der_abs(i + 1);
        return res;
    }

    Scalar Scalar::div_r() const
    {
        Scalar res(*this, false);
        for (int dom = 0; dom < ndom; dom++)
            res.set_domain(dom) = get_domain(dom)->div_r(operator()(dom));

        return res;
    }

    Scalar Scalar::mult_r() const
    {
        Scalar res(*this, false);
        for (int dom = 0; dom < ndom; dom++)
            res.set_domain(dom) = get_domain(dom)->mult_r(operator()(dom));

        return res;
    }

    Scalar Scalar::div_rsint() const
    {
        Scalar res(*this, false);
        for (int dom = 0; dom < ndom; dom++) {
            Val_domain auxi(get_domain(dom)->div_r(operator()(dom)));
            res.set_domain(dom) = auxi.div_sin_theta();
        }
        return res;
    }

    Scalar Scalar::div_1mx2() const
    {
        Scalar res(*this, false);
        for (int dom(0); dom < ndom; ++dom) {
            res.set_domain(dom) = operator()(dom).div_1mx2();
        }
        return res;
    }

    Scalar Scalar::mult_cos_theta() const
    {
        Scalar res(*this, false);
        for (int dom(0); dom < ndom; ++dom)
            res.set_domain(dom) = operator()(dom).mult_cos_theta();
        return res;
    }

    Scalar Scalar::mult_sin_theta() const
    {
        Scalar res(*this, false);
        for (int dom(0); dom < ndom; ++dom)
            res.set_domain(dom) = operator()(dom).mult_sin_theta();
        return res;
    }

    Scalar Scalar::mult_cos_phi() const
    {
        Scalar res(*this, false);
        for (int dom(0); dom < ndom; ++dom)
            res.set_domain(dom) = operator()(dom).mult_cos_phi();
        return res;
    }

    Scalar Scalar::mult_sin_phi() const
    {
        Scalar res(*this, false);
        for (int dom(0); dom < ndom; ++dom)
            res.set_domain(dom) = operator()(dom).mult_sin_phi();
        return res;
    }
} // namespace Kadath
