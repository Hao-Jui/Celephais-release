// NS_2D_XCTS regrid support.
#pragma once

#include "For_Kadath/Kadath_point_h/kadath.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include <math.h>
#include <sstream>
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "Apps/Formalism/Shared/NS_2D_XCTS/scalar_import_batch.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include <array>
#include <utility>

namespace Kadath {

using space_t = Space_polar_adapted;

template <typename config_t>
int ns_2d_xcts_regrid_from_fields(config_t& bconfig, const space_t& old_space,
                                  const Scalar& old_conf, const Scalar& old_lapse,
                                  const Scalar& old_shift, const Scalar& old_logh,
                                  const Scalar& old_Omg, const int new_res,
                                  std::string outputfile)
{
    if ((new_res % 2) == 0) {
        KADATH_THROW("New Resolution is invalid.  Must be odd (9,11,13,etc)");
    }
    if (&old_conf.get_space() != &old_space || &old_lapse.get_space() != &old_space ||
        &old_shift.get_space() != &old_space || &old_logh.get_space() != &old_space ||
        &old_Omg.get_space() != &old_space)
        KADATH_THROW("NS2d_xcts live regrid fields must belong to the supplied source space");

    std::cout << "Resolution of old space: " << old_space.get_domain(0)->get_nbr_points()(0) << " (r), "
              << old_space.get_domain(0)->get_nbr_points()(1) << " (theta)" << std::endl;

    auto [r_min, r_max] = bco_utils::get_rmin_rmax(old_space, 1);

    bconfig.set(BCO_RES) = new_res;

    bconfig.set_filename(outputfile);

    const Domain_polar_shell_outer_adapted* old_outer_adapted =
        dynamic_cast<const Domain_polar_shell_outer_adapted*>(old_space.get_domain(1));
    if (old_outer_adapted == nullptr)
        KADATH_THROW("NS2d_xcts regrid could not identify the old adapted surface");

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

    Dim_array new_resolution(bconfig(DIM));
    new_resolution.set(0) = new_res;
    new_resolution.set(1) = new_res;

    Point center(bconfig(DIM));
    for (int i = 1; i <= bconfig(DIM); ++i) {
        center.set(i) = 0.;
    }

    auto bounds = bco_utils::make_NS_bounds(bconfig);
    space_t space(old_space.get_type_base(), center, new_resolution, bounds);
    Scalar conf(space);
    conf = 1.;
    Scalar lapse(space);
    lapse = 1.;
    Scalar shift(space);
    shift.annule_hard();
    Scalar logh(space);
    logh.annule_hard();
    Scalar Omg(space);
    Omg.annule_hard();

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

    const std::array import_fields{
        ns_2d_xcts_import::import_field(conf, old_conf),
        ns_2d_xcts_import::import_field(lapse, old_lapse),
        ns_2d_xcts_import::import_field(logh, old_logh),
        ns_2d_xcts_import::import_field(shift, old_shift),
        ns_2d_xcts_import::import_field(Omg, old_Omg),
    };
    ns_2d_xcts_import::import_scalar_batch(import_fields);

    lapse.std_base();
    conf.std_base();
    logh.std_base();
    shift.std_base();
    Omg.std_base();

    bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh, Omg);

    return EXIT_SUCCESS;
}

template <typename config_t>
int ns_2d_xcts_regrid(config_t& bconfig, const int new_res, std::string outputfile)
{
    const std::string in_spacefile = bconfig.space_filename();

    if (!fs::exists(in_spacefile)) {
        std::ostringstream oss;
        oss << "File: " << in_spacefile << " not found.\n\n";
        KADATH_THROW(oss.str());
    }

    BeFileSource source(in_spacefile);
    space_t old_space(source);
    Scalar old_conf(old_space, source);
    Scalar old_lapse(old_space, source);
    Scalar old_shift(old_space, source);
    Scalar old_logh(old_space, source);
    Scalar old_Omg(old_space, source);

    return ns_2d_xcts_regrid_from_fields(bconfig, old_space, old_conf, old_lapse,
                                         old_shift, old_logh, old_Omg, new_res,
                                         std::move(outputfile));
}

} // namespace Kadath
