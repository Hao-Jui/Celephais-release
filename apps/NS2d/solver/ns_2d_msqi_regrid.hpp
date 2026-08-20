// ns_2d_msqi_regrid.hpp
#pragma once

#include "For_Kadath/Kadath_point_h/kadath.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include <math.h>
#include <sstream>
#include "For_Kadath/Config/config_bco.hpp"
#include "ns2d_solver_utils.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"

namespace Kadath {

using space_t = Space_polar_adapted;

template <typename config_t> int ns_2d_msqi_regrid(config_t& bconfig, const int new_res, std::string outputfile)
{
    std::string in_spacefile = bconfig.space_filename();

    if (!fs::exists(in_spacefile)) {
        std::ostringstream oss;
        oss << "File: " << in_spacefile << " not found.\n\n";
        KADATH_THROW(oss.str());
    }

    BeFileSource ff1(in_spacefile);
    space_t old_space(ff1);
    Scalar old_conf(old_space, ff1);
    Scalar old_lapse(old_space, ff1);
    Scalar old_shift(old_space, ff1);
    Scalar old_metQ(old_space, ff1);
    Scalar old_logh(old_space, ff1);
    Scalar old_Omg(old_space, ff1);

    if ((new_res % 2) == 0) {
        KADATH_THROW("New Resolution is invalid.  Must be odd (9,11,13,etc)");
    }

    std::cout << "Resolution of old space: " << old_space.get_domain(0)->get_nbr_points()(0) << " (r), "
              << old_space.get_domain(0)->get_nbr_points()(1) << " (theta)" << std::endl;

    auto [r_min, r_max] = bco_utils::get_rmin_rmax(old_space, 1);

    bconfig.set(BCO_RES) = new_res;

    bconfig.set_filename(outputfile);
    setup_co<NS>(bconfig, true);

    const Domain_polar_shell_outer_adapted* old_outer_adapted =
        dynamic_cast<const Domain_polar_shell_outer_adapted*>(old_space.get_domain(1));

    // get the radius from each domain
    Scalar old_space_radius(old_space);
    old_space_radius = 0.;
    for (int i = 0; i < old_space.get_nbr_domains(); ++i) {
        old_space_radius.set_domain(i) = old_space.get_domain(i)->get_radius();
    }

    // get the adapted radius of the adapted domain
    old_space_radius.set_domain(1) = old_outer_adapted->get_outer_radius();

    // define a standard decomposition, compatible with the parity of this field
    old_space_radius.std_base();
    // end setup old radius field

    std::cout << "Rmin/max: " << r_min << " " << r_max << "\n";

    auto new_spacefile = bconfig.space_filename();
    BeFileSource ff2(new_spacefile);
    space_t space(ff2);
    Scalar conf(space, ff2);
    Scalar lapse(space, ff2);
    Scalar shift(space, ff2);
    Scalar metQ(space, ff2);
    Scalar logh(space, ff2);
    Scalar Omg(space, ff2);

    const Dim_array res = space.get_domain(0)->get_nbr_points();

    std::cout << "Resolution of new space: " << res(0) << " (r), " << res(1) << " (theta)" << std::endl;

    bco_utils::print_bounds_from_space(old_space);
    bco_utils::print_bounds_from_space(space);

    // get adapted domains to update the radius
    const Domain_polar_shell_outer_adapted* new_outer_adapted =
        dynamic_cast<const Domain_polar_shell_outer_adapted*>(space.get_domain(1));
    const Domain_polar_shell_inner_adapted* new_inner_adapted =
        dynamic_cast<const Domain_polar_shell_inner_adapted*>(space.get_domain(2));

    // update adapted domain mapping: physical coordinates to numerical ones
    bco_utils::interp_adapted_mapping(new_outer_adapted, 1, old_space_radius);
    bco_utils::interp_adapted_mapping(new_inner_adapted, 1, old_space_radius);

    conf.import(old_conf);
    lapse.import(old_lapse);
    logh.import(old_logh);
    shift.import(old_shift);
    metQ.import(old_metQ);
    Omg.import(old_Omg);

    lapse.std_base();
    conf.std_base();
    logh.std_base();
    shift.std_base();
    metQ.std_base();
    Omg.std_base();

    bco_utils::save_to_file(space, bconfig, conf, lapse, shift, metQ, logh, Omg);

    return EXIT_SUCCESS;
}

} // namespace Kadath
