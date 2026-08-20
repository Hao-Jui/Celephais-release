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

#include "For_Kadath/Kadath_point_h/kadath.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace Kadath;

#ifndef EQUI
#define EQUI -11
#endif

namespace bco_utils
{
    template <typename dom_t> double get_radius(const dom_t* dom, const int bound)
    {
        auto const npts = dom->get_nbr_points();
        const int ndim = npts.get_ndim();
        Index pos(npts);

        switch (bound) {
            case INNER_BC:
                break;
            case OUTER_BC:
                pos.set(0) = npts(0) - 1;
                if (ndim > 1)
                    pos.set(1) = npts(1) - 1;
                if (ndim > 2)
                    pos.set(2) = npts(2) - 1;
                break;
            case EQUI:
                pos.set(0) = npts(0) - 1;
                if (ndim > 1)
                    pos.set(1) = npts(1) - 1;
                break;
            default:
                std::cout << "Unknown bound sent to get_radius: " << bound << std::endl;
                break;
        }
        double val = dom->get_radius()(pos);
        return (std::isinf(val)) ? std::numeric_limits<double>::infinity() : val;
    }

    inline std::array<double, 2> get_field_min_max(const Scalar& field, const int dom, const int bound = OUTER_BC)
    {
        const int npts_r = field.get_domain(dom)->get_nbr_points()(0);

        Index pos(field.get_domain(dom)->get_nbr_points());
        Index bpos(field.get_domain(dom)->get_nbr_points());

        int r_bound = 0;
        switch (bound) {
            case INNER_BC:
                break;
            case OUTER_BC:
                r_bound = npts_r - 1;
                break;
            default:
                std::cout << "Unknown bound sent to get_field_min_max: " << bound << std::endl;
                break;
        }

        double fmax = field(dom)(pos);
        double fmin = field(dom)(pos);

        do {
            bpos.set(0) = r_bound;
            bpos.set(1) = pos(1);
            bpos.set(2) = pos(2);
            double f = field(dom)(bpos);

            if (f > fmax)
                fmax = f;
            if (f < fmin)
                fmin = f;
        } while (pos.inc());

        return std::array<double, 2>{fmin, fmax};
    }

    template <typename space_t> std::array<double, 2> get_rmin_rmax(const space_t& space, const int dom)
    {
        const int npts_r = space.get_domain(dom)->get_nbr_points()(0);
        Index pos(space.get_domain(dom)->get_nbr_points());
        pos.set_start();
        pos.set(0) = npts_r - 1;

        double rmax = space.get_domain(dom)->get_radius()(pos);
        double rmin = space.get_domain(dom)->get_radius()(pos);

        do {
            pos.set(0) = npts_r - 1;
            double r = space.get_domain(dom)->get_radius()(pos);
            if (r > rmax)
                rmax = r;
            if (r < rmin)
                rmin = r;
        } while (pos.inc());

        return std::array<double, 2>{rmin, rmax};
    }

    template <typename space_t> double get_center(const space_t& space, const int dom)
    {
        Index pos(space.get_domain(dom)->get_nbr_points());
        return space.get_domain(dom)->get_cart(1)(pos);
    }

    inline double get_boundary_val(const int dom, const Scalar& field, const int bound = INNER_BC)
    {
        auto this_domain = field(dom).get_domain();
        Index pos(this_domain->get_nbr_points());
        switch (bound) {
            case INNER_BC:
                break;
            case OUTER_BC:
                pos.set(0) = this_domain->get_nbr_points()(0) - 1;
                pos.set(1) = this_domain->get_nbr_points()(1) - 1;
                pos.set(2) = this_domain->get_nbr_points()(2) - 1;
                break;
            case EQUI:
                pos.set(0) = this_domain->get_nbr_points()(0) - 1;
                break;
            default:
                std::cout << "Unknown bound sent to get_boundary_val: " << bound << std::endl;
                break;
        }
        return field(dom)(pos);
    }

    template <typename config_t, typename space_t, typename... idx_t>
    void set_radius(const int& dom, const space_t& space, config_t& bconfig, const idx_t... idxs)
    {
        bconfig.set(idxs...) = get_radius(space.get_domain(dom), OUTER_BC);
    }

    template <typename ary_t> void print_bounds(std::string name, const ary_t& bary)
    {
        std::cout << name << ": ";
        for (auto& e : bary) {
            std::cout << e << " ";
        }
        std::cout << std::endl;
    }

    inline Scalar get_bound_filled_field(Scalar const& field, int const bound)
    {
        int const ndom = field.get_nbr_domains();
        Scalar out(field, false);
        for (auto d = 0; d < ndom; ++d) {
            auto npts = field(d).get_conf().get_dimensions();
            Index pos(npts);
            Index pos_b(pos);
            switch (bound) {
                case INNER_BC:
                    break;
                case OUTER_BC:
                    pos_b.set(1) = npts(1) - 1;
                    break;
            }
            do {
                pos_b.set(2) = pos(2);
                pos_b.set(3) = pos(3);
                out.set_domain(d).set(pos) = field(d)(pos_b);
            } while (pos.inc());
            out.set_domain(d).set_base() = field(d).get_base();
        }
        return out;
    }

    inline Scalar get_bound_filled_field_from_one_dom(Scalar const& field, int const bound, int const dom)
    {
        auto bound_field(get_bound_filled_field(field, bound));
        int const ndom = field.get_nbr_domains();
        Scalar out(bound_field);
        for (auto d = 0; d < ndom; ++d) {
            out.set_domain(d) = bound_field(dom);
            out.set_domain(d).set_base() = bound_field(dom).get_base();
        }
        return out;
    }

    template <class space_t> void print_bounds_from_space(space_t const& space, int bound = OUTER_BC)
    {
        for (int i = 0; i < space.get_nbr_domains(); ++i)
            std::cout << bco_utils::get_radius(space.get_domain(i), bound) << " ";
        std::cout << std::endl;
    };

    template <class space_t> void print_constant_space_resolution(space_t const& space)
    {
        auto dom = space.get_domain(0);
        auto ndim = dom->get_ndim();
        std::array<std::string, 3> directions{"r", "theta", "phi"};
        for (auto i = 0; i < ndim; ++i)
            std::cout << dom->get_nbr_points()(i) << " (" << directions[i] << ")     ";
        std::cout << "\n";
    };
} // namespace bco_utils
