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

/*
 * Modifications (Celephais):
 *   2026-06-19  Updated BH/NS radial-bounds convention so ROUT is the inner
 *               adapted-domain outer radius and r_bisph remains fixed across
 *               shell-count changes; see PATCHES-KADATH-UPSTREAM.md and
 *               LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include <cstddef>
#include <vector>

using namespace Kadath;

namespace bco_utils
{
    // Fill a star's radial-bounds vector in the layout
    // [rin, rmid, rout, outer shells..., r_bisph]. The caller sizes the
    // vector 3 + NSHELLS (see make_NS_bounds). The configured ROUT is the
    // no-shell outer target; adding shells subdivides (RMID, r_bisph] so the
    // bispheric/outer matching radius remains fixed.
    template <typename ary_t, typename config_t, typename... idx_t>
    void set_NS_bounds(ary_t& bounds, config_t& bconfig, idx_t... bco)
    {
        const int size = bounds.size();
        const int nshells = bconfig.has_value(NSHELLS, bco...)
                                ? static_cast<int>(bconfig(NSHELLS, bco...))
                                : 0;
        if (size != 3 + nshells)
            KADATH_THROW("set_NS_bounds: bounds size must be 3 + NSHELLS");

        const int rin = 0;
        const int r = 1; // RMID slot (the adapted pair starts right after the nucleus)
        const int rout = 2;

        bounds[rin] = bconfig(RIN, bco...);
        bounds[r] = bconfig(RMID, bco...);
        const double r_bisph = bconfig(ROUT, bco...);

        if (!(r_bisph > bounds[r]))
            KADATH_THROW("set_NS_bounds: r_bisph must exceed RMID");

        const double delta_r = (r_bisph - bounds[r]) / (nshells + 1.);
        for (int i = 0; i <= nshells; ++i)
            bounds[rout + i] = bounds[r] + delta_r * (i + 1);
        bconfig.set(ROUT, bco...) = bounds[rout];
    }

    // Allocate AND fill a star's radial-bounds vector in one call: sizes it
    // 3 + NSHELLS (nucleus + adapted pair + the per-star outer shells) and fills
    // it via set_NS_bounds. Deriving the size from the same NSHELLS the fill
    // reads makes a size/config mismatch (and the resulting overflow) impossible,
    // and single-sources what every BNS/BHNS/NS caller used to hand-size.
    template <typename config_t, typename... idx_t>
    std::vector<double> make_NS_bounds(config_t& bconfig, idx_t... bco)
    {
        const int nshells = bconfig.has_value(NSHELLS, bco...)
                                ? static_cast<int>(bconfig(NSHELLS, bco...))
                                : 0;
        std::vector<double> bounds(3 + nshells);
        set_NS_bounds(bounds, bconfig, bco...);
        return bounds;
    }

    // Blended bispheric matching radius: r_bisph = rmid + fill * (dist/2 - rmid).
    // Any fill in (0,1) keeps rmid < r_bisph < dist/2 structurally — at every
    // separation and mass ratio. With NSHELLS == 0 this r_bisph is also ROUT;
    // with shells, the fixed band (RMID, r_bisph] is subdivided.
    inline double blended_rbisph(double rmid, double dist, double fill)
    {
        return rmid + fill * (dist / 2. - rmid);
    }

    // Binary variant with an explicit r_bisph. The adapted bulk is
    // [RIN, RMID, ROUT]; ordinary shells sit between ROUT and r_bisph.
    // Therefore r_bisph == ROUT for NSHELLS == 0 and
    // r_bisph = ROUT + shell_width * NSHELLS otherwise, with shell_width chosen
    // so r_bisph stays fixed when NSHELLS changes.
    template <typename ary_t, typename config_t, typename... idx_t>
    void set_NS_bounds_fixed_rbisph(ary_t& bounds, config_t& bconfig, double r_bisph, idx_t... bco)
    {
        const int size = bounds.size();
        const int nshells = size - 3;
        const int rin = 0;
        const int r = 1; // RMID slot
        const int rout = 2;

        bounds[rin] = bconfig(RIN, bco...);
        bounds[r] = bconfig(RMID, bco...);
        if (!(r_bisph > bounds[r]))
            KADATH_THROW("set_NS_bounds_fixed_rbisph: r_bisph must exceed RMID");

        const double delta_r = (r_bisph - bounds[r]) / (nshells + 1.);
        for (int i = 0; i <= nshells; ++i)
            bounds[rout + i] = bounds[r] + delta_r * (i + 1);
        bconfig.set(ROUT, bco...) = bounds[rout];
    }

    template <typename config_t, typename... idx_t>
    std::vector<double> make_NS_bounds_fixed_rbisph(config_t& bconfig, double r_bisph, idx_t... bco)
    {
        const int nshells = bconfig.has_value(NSHELLS, bco...)
                                ? static_cast<int>(bconfig(NSHELLS, bco...))
                                : 0;
        std::vector<double> bounds(3 + nshells);
        set_NS_bounds_fixed_rbisph(bounds, bconfig, r_bisph, bco...);
        return bounds;
    }

    // Binary-config entry point. An ABSENT rbisph_fill key means the default
    // blend fill = 1/3 (the historical no-shell setup formula). An EXPLICIT
    // rbisph_fill = 0 uses the config ROUT as the fixed outer target. Only
    // binary configs carry DIST/RBISPH_FILL — single-NS callers keep using
    // make_NS_bounds directly.
    inline constexpr double default_rbisph_fill = 1. / 3.;

    // Single source of the "absent key means the default blend" semantics.
    // Both the bounds builder below and the AMR hp-mode guard resolve the fill
    // through this helper; fill <= 0 (only reachable via an explicit
    // rbisph_fill = 0) selects the config ROUT as the fixed outer target.
    template <typename config_t>
    double resolved_rbisph_fill(config_t& bconfig)
    {
        return bconfig.has_value(RBISPH_FILL) ? static_cast<double>(bconfig(RBISPH_FILL))
                                              : default_rbisph_fill;
    }

    template <typename config_t, typename node_t>
    std::vector<double> make_binary_NS_bounds(config_t& bconfig, node_t bco)
    {
        const double fill = resolved_rbisph_fill(bconfig);
        if (fill <= 0.)
            return make_NS_bounds(bconfig, bco);
        if (fill >= 1.)
            KADATH_THROW("[binary] rbisph_fill must lie in (0,1)");
        return make_NS_bounds_fixed_rbisph(
            bconfig, blended_rbisph(bconfig(RMID, bco), bconfig(DIST), fill), bco);
    }

    template <typename space_t, typename config_t, typename... Idx>
    void update_config_NS_radii(space_t& space, config_t& bconfig, const size_t dom, Idx... idx)
    {
        auto [r_min, r_max] = bco_utils::get_rmin_rmax(space, dom);
        bconfig.set(RIN, idx...) = 0.5 * r_min;
        bconfig.set(RMID, idx...) = r_max;
        bconfig.set(ROUT, idx...) = 1.5 * r_max;
    }
} // namespace bco_utils
