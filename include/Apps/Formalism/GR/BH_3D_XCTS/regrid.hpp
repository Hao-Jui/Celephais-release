/*
 * Copyright 2022
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 *
 * Author:
 * Samuel D. Tootle <tootle@itp.uni-frankfurt.de>
 * L. Jens Papenfort <papenfort@th.physik.uni-frankfurt.de>
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
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once
#include "For_Kadath/Kadath_point_h/kadath_adapted_bh.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "Apps/Policy/app_resolution.hpp"
#include "Apps/Seed/GR/bh_seed.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include <math.h>
#include <sstream>
#include <cmath>
#include <memory>

namespace Kadath {

template <typename config_t> int bh_3d_xcts_regrid(config_t& bconfig, const int new_res, std::string outputfile)
{
    int exit_status = EXIT_SUCCESS;

    std::string in_spacefile = bconfig.space_filename();

    if (!fs::exists(in_spacefile)) {
        std::ostringstream oss;
        oss << "File: " << in_spacefile << " not found.\n\n";
        KADATH_THROW(oss.str());
    }

    using space_t = Space_adapted_bh;
    BeFileSource ff1(in_spacefile);
    space_t old_space(ff1);
    Scalar old_conf(old_space, ff1);
    Scalar old_lapse(old_space, ff1);
    Vector old_shift(old_space, ff1);

    // FIXME not sure if it's only about oddness...
    validate_resolution(new_res);

    // Make sure the domains have adequate space between them based on
    // the previous solution
    bconfig.set(RMID) = bco_utils::get_radius(old_space.get_domain(1), OUTER_BC);

    // estimate how small the inner radius should be based on relation
    // between conformal factor and numerical radius.
    // see https://arxiv.org/pdf/0805.4192, eq(64)
    double conf_inner = bco_utils::get_boundary_val(2, old_conf, INNER_BC);
    double conf_i_sq = conf_inner * conf_inner;
    double est_r_div2 = bconfig(MCH) / conf_i_sq;
    bconfig.set(RIN) = est_r_div2;
    bconfig.set(ROUT) = 4. * est_r_div2 * 2.;

    bconfig.set(BCO_RES) = new_res;
    bconfig.set_filename(outputfile);

    // Generate new space and fields using the radii computed above
    Seed::setup_co<BH, Space_adapted_bh>(bconfig, true);

    // Read in new fields
    auto new_spacefile = bconfig.space_filename();
    BeFileSource ff2(new_spacefile);
    space_t space(ff2);
    Scalar conf(space, ff2);
    Scalar lapse(space, ff2);
    Vector shift(space, ff2);

    std::cout << "Resolution of old space: ";
    bco_utils::print_constant_space_resolution(old_space);

    std::cout << "Resolution of new space: ";
    bco_utils::print_constant_space_resolution(space);

    std::cout << "\nold bounds:" << std::endl;
    bco_utils::print_bounds_from_space(old_space);

    std::cout << "New bounds:" << std::endl;
    bco_utils::print_bounds_from_space(space);
    std::cout << endl;

    // needed in some cases, since the interpolation can go crazy
    const Domain_shell_outer_homothetic* old_outer_homothetic =
        dynamic_cast<const Domain_shell_outer_homothetic*>(old_space.get_domain(1));

    // import fields
    bco_utils::update_adapted_field(old_conf, 2, 1, old_outer_homothetic, OUTER_BC);
    bco_utils::update_adapted_field(old_lapse, 2, 1, old_outer_homothetic, OUTER_BC);
    for (int i = 1; i <= 3; ++i)
        bco_utils::update_adapted_field(old_shift.set(i), 2, 1, old_outer_homothetic, OUTER_BC);

    conf.import(old_conf);
    lapse.import(old_lapse);

    shift.set(1).import(old_shift.set(1));
    shift.set(2).import(old_shift.set(2));
    shift.set(3).import(old_shift.set(3));
    // end import fields

    // reset fields to zero inside excised region
    for (auto& d : {0, 1}) {
        lapse.set_domain(d).annule_hard();
        conf.set_domain(d).annule_hard();
        for (int i = 1; i <= 3; ++i) {
            shift.set(i).set_domain(d).annule_hard();
        }
    }
    lapse.std_base();
    conf.std_base();
    shift.std_base();

    bco_utils::save_to_file(space, bconfig, conf, lapse, shift);
    return exit_status;
}

} // namespace Kadath
