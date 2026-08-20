/*
 * This file is part of the KADATH library.
 * Author: Samuel Tootle
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include <cmath>
#include <sstream>

using namespace Kadath;

namespace bco_utils
{
    template <typename dom_t>
    void update_adapted_field(Scalar& res, const int from_dom, const int to_dom, const dom_t* dom, const int bound)
    {
        Tensor* ptr = &res;
        Array<int> doms(2);
        doms.set(0) = from_dom;
        doms.set(1) = to_dom;

        Scalar import = dom->import(to_dom, bound, 1, doms, &ptr);
        res.set().set_domain(to_dom) = import(to_dom);
    }

    template <typename adapted_t>
    void interp_adapted_mapping(const adapted_t* new_shell, const int old_outer_adapted_dom,
                                const Scalar& old_radius_field)
    {
        Val_domain new_mapping = new_shell->get_radius();
        int const ndim = new_shell->get_ndim();

        auto old_shell = old_radius_field.get_space().get_domain(old_outer_adapted_dom);
        int const old_ndim = old_shell->get_ndim();
        double rinner = get_radius(old_shell, INNER_BC);

        Index new_pos(new_shell->get_nbr_points());
        double xc_new = new_shell->get_center()(1);
        double xc_old = old_shell->get_center()(1);
        const Val_domain cart_1 = new_shell->get_cart(1);
        const Val_domain cart_2 = new_shell->get_cart(2);
        const Val_domain cart_3 = (ndim == 3) ? new_shell->get_cart(3) : cart_2;

        do {
            double x, y, z;
            if (ndim == 3) {
                x = cart_1(new_pos) - xc_new;
                y = cart_2(new_pos);
                z = cart_3(new_pos);
            } else {
                x = cart_1(new_pos) - xc_new;
                y = 0;
                z = cart_2(new_pos);
            }

            double r = std::sqrt(x * x + y * y + z * z);
            if (!std::isfinite(r) || !std::isfinite(rinner) || r <= 0. || rinner <= 0.) {
                std::ostringstream oss;
                oss << "interp_adapted_mapping: invalid radial normalization at new index "
                    << new_pos << " (r=" << r << ", rinner=" << rinner << ")";
                KADATH_THROW(oss.str());
            }

            x /= r / rinner;
            y /= r / rinner;
            z /= r / rinner;

            Point absol(old_ndim);
            if (old_ndim == 2) {
                auto rsq_xy = x * x + y * y;
                auto r_xy = std::sqrt(rsq_xy);
                absol.set(1) = r_xy + xc_old;
                absol.set(2) = z;
            } else {
                absol.set(1) = x + xc_old;
                absol.set(2) = y;
                absol.set(3) = z;
            }

            const double mapped_radius = old_radius_field.val_point(absol);
            if (!std::isfinite(mapped_radius)) {
                std::ostringstream oss;
                oss << "interp_adapted_mapping: non-finite imported radius at new index "
                    << new_pos << " from old-space point " << absol;
                KADATH_THROW(oss.str());
            }
            new_mapping.set(new_pos) = mapped_radius;

        } while (new_pos.inc());

        new_mapping.std_base();
        new_shell->set_mapping(new_mapping);
    }
} // namespace bco_utils
