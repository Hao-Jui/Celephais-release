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
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include <cmath>
#include <string>
#include <utility>

using namespace Kadath;

namespace bco_utils
{
    constexpr double psi = 1.7;
    constexpr double psisq = psi * psi;
    constexpr double invpsisq = 1. / psisq;
    const double gold_ratio = (1 + std::sqrt(5)) / 2.;

    template <typename ary_t, typename config_t> void set_isolated_BH_bounds(ary_t& bounds, config_t& bconfig)
    {
        const int size = bounds.size();

        const int rin = 0;
        const int r = rin + 1;
        const int rout = 2;
        const int shells = bconfig(NSHELLS);
        if (size != 3 + shells)
            KADATH_THROW("set_isolated_BH_bounds: bounds size must be 3 + NSHELLS");

        bounds[rin] = bconfig(RIN);
        bounds[r] = bconfig(RMID);
        const double r_bisph = bconfig(ROUT);

        if (!(r_bisph > bounds[r]))
            KADATH_THROW("set_isolated_BH_bounds: r_bisph must exceed RMID");

        const double delta_r = (r_bisph - bounds[r]) / (shells + 1.);
        for (int i = 0; i <= shells; ++i)
            bounds[rout + i] = bounds[r] + delta_r * (i + 1);
        bconfig.set(ROUT) = bounds[rout];
    }

    template <typename ary_t, typename config_t>
    void set_BH_bounds(ary_t& bounds, config_t& bconfig, NODES bco, [[maybe_unused]] const bool adapt_shells = false)
    {
        const int size = bounds.size();

        const int rin = 0;
        const int r = rin + 1;
        const int rout = 2;
        const int shells = bconfig(NSHELLS, bco);
        if (size != 3 + shells)
            KADATH_THROW("set_BH_bounds: bounds size must be 3 + NSHELLS");

        bounds[rin] = bconfig(RIN, bco);
        bounds[r] = bconfig(RMID, bco);
        const double r_bisph = bconfig(ROUT, bco);

        if (!(r_bisph > bounds[r]))
            KADATH_THROW("set_BH_bounds: r_bisph must exceed RMID");

        const double delta_r = (r_bisph - bounds[r]) / (shells + 1.);
        for (int i = 0; i <= shells; ++i)
            bounds[rout + i] = bounds[r] + delta_r * (i + 1);
        bconfig.set(ROUT, bco) = bounds[rout];
    }

    inline double mirr_from_mch(const double chi, const double mch)
    {
        return std::sqrt((1 + std::sqrt(1 - chi * chi)) / 2.) * mch;
    }

    template <typename config_t, typename... Idx>
    void sync_mirr_from_mch(config_t& bconfig, Idx... idx)
    {
        bconfig.set(MIRR, idx...) = mirr_from_mch(bconfig(CHI, idx...), bconfig(MCH, idx...));
    }

    template <typename space_t>
    double syst_mch(System_of_eqs& syst, const space_t& space, const std::string eq, const int dom)
    {
        Val_domain integS(syst.give_val_def(eq.c_str())()(dom));
        double S = space.get_domain(dom)->integ(integS, INNER_BC);

        Val_domain integMsq(syst.give_val_def("intMsq")()(dom));
        double Mirrsq = space.get_domain(dom)->integ(integMsq, INNER_BC);
        return std::sqrt(Mirrsq + S * S / 4. / Mirrsq);
    }

    template <typename space_t, typename config_t, typename... Idx>
    void update_config_BH_radii(space_t& space, config_t& bconfig, const size_t dom, const Scalar& conf, Idx... idx)
    {
        [[maybe_unused]] auto [rmin, trmax] = bco_utils::get_rmin_rmax(space, dom);

        double conf_inner = bco_utils::get_boundary_val(dom + 1, conf, INNER_BC);
        double conf_i_sq = conf_inner * conf_inner;
        double est_r_div2 = bconfig(MCH, idx...) / conf_i_sq;
        bconfig.set(RIN, idx...) = est_r_div2;

        bco_utils::set_radius(1, space, bconfig, RMID);
    }

    template <typename config_t, typename... coIdx> double compute_kerr_mirr(config_t& bconfig, coIdx... bco)
    {
        const double MCHsq = bconfig(MCH, bco...) * bconfig(MCH, bco...);
        const double S = bconfig(CHI, bco...) * MCHsq;
        const double Mirrsq = (MCHsq + std::sqrt(MCHsq * MCHsq - S * S)) / 2;
        return std::sqrt(Mirrsq);
    }
} // namespace bco_utils
