#pragma once

#include "For_Kadath/Kadath_point_h/kadath.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include <math.h>
#include <sstream>
#include <utility>
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "Apps/Policy/app_resolution.hpp"
#include "Apps/Seed/GR/ns_seed.hpp"
#include "Apps/Seed/GR/ns_nosym_seed.hpp"
#include "For_Kadath/Space/spheric_homothetic.hpp"

namespace Kadath {

// Space-type traits — adapted-domain types + theta resolution per space.
// (Formerly Orchestration/ns_space_traits.hpp; folded in as sole consumer.)
template <typename space_t>
struct ns_space_traits;

template <>
struct ns_space_traits<Space_spheric_adapted> {
    using outer_adapted = Domain_shell_outer_adapted;
    using inner_adapted = Domain_shell_inner_adapted;
    static Dim_array resolution(int r_res) { return spheric_res(r_res); }
};

template <>
struct ns_space_traits<Space_spheric_adapted_nosym> {
    using outer_adapted = Domain_shell_outer_adapted_nosym;
    using inner_adapted = Domain_shell_inner_adapted_nosym;
    static Dim_array resolution(int r_res) { return spheric_res_nosym(r_res); }
};

template <>
struct ns_space_traits<Space_spheric_homothetic> {
    using outer_adapted = Domain_shell_outer_homothetic;
    using inner_adapted = Domain_shell_inner_homothetic;
    static Dim_array resolution(int r_res) { return spheric_res(r_res); }
};

template <typename config_t, typename space_t, typename TransferFields>
int ns_3d_xcts_interpolate_on_new_grid_fields(config_t& bconfig, const int new_res, std::string outputfile,
                                              bool use_config_vars, TransferFields&& transfer_fields)
{
    using traits = ns_space_traits<space_t>;
    using outer_adapted_t = typename traits::outer_adapted;
    using inner_adapted_t = typename traits::inner_adapted;

    std::string kadath_filename = bconfig.space_filename();

    if (!fs::exists(kadath_filename)) {
        std::ostringstream oss;
        oss << "File: " << kadath_filename << " not found.\n\n";
        KADATH_THROW(oss.str());
    }

    BeFileSource ff1(kadath_filename);
    space_t old_space(ff1);

    std::cout << "Resolution of old space: " << old_space.get_domain(0)->get_nbr_points()(0) << " (r), "
              << old_space.get_domain(0)->get_nbr_points()(1) << " (theta), "
              << old_space.get_domain(0)->get_nbr_points()(2) << " (phi)" << std::endl;

    const outer_adapted_t* old_outer_adapted =
        dynamic_cast<const outer_adapted_t*>(old_space.get_domain(1));

    Scalar old_space_radius(old_space);
    old_space_radius = 0.;

    for (int i = 0; i < old_space.get_nbr_domains(); ++i)
        old_space_radius.set_domain(i) = old_space.get_domain(i)->get_radius();
    old_space_radius.set_domain(1) = old_outer_adapted->get_outer_radius();

    old_space_radius.std_base();

    auto [r_min, r_max] = bco_utils::get_rmin_rmax(old_space, 1);

    std::cout << "Old bounds: ";
    for (int i = 0; i < old_space.get_nbr_domains() - 1; ++i)
        std::cout << bco_utils::get_radius(old_space.get_domain(i), OUTER_BC) << " ";
    std::cout << "\n";

    bconfig.set(BCO_RES) = new_res;
    Dim_array res = traits::resolution(static_cast<int>(bconfig(BCO_RES)));

    validate_resolution(res(0));

    std::cout << "Resolution of new space: " << res(0) << " (r), " << res(1) << " (theta), " << res(2) << " (phi)"
              << std::endl;

    int type_coloc = old_space.get_type_base();

    if (!use_config_vars) {
        bconfig.set(RIN) = 0.5 * r_min;
        bconfig.set(ROUT) = 1.5 * r_max;
        bconfig.set(RMID) = r_min;
    }

    int ndom = 4 + bconfig(NSHELLS);
    std::vector<double> bounds(ndom - 1);
    bco_utils::set_NS_bounds(bounds, bconfig);

    bco_utils::print_bounds("New bounds", bounds);

    Point center = old_space.get_domain(0)->get_center();

    space_t space(type_coloc, center, res, bounds);
    Base_tensor basis(space, CARTESIAN_BASIS);

    const outer_adapted_t* new_outer_adapted =
        dynamic_cast<const outer_adapted_t*>(space.get_domain(1));
    const inner_adapted_t* new_inner_adapted =
        dynamic_cast<const inner_adapted_t*>(space.get_domain(2));

    bco_utils::interp_adapted_mapping(new_outer_adapted, 1, old_space_radius);
    bco_utils::interp_adapted_mapping(new_inner_adapted, 1, old_space_radius);

    bconfig.set_filename(outputfile);
    std::forward<TransferFields>(transfer_fields)(ff1, old_space, space, basis, bconfig);

    return EXIT_SUCCESS;
}

template <typename config_t, typename space_t>
int ns_3d_xcts_interpolate_on_new_grid(config_t& bconfig, const int new_res, std::string outputfile,
                                        bool use_config_vars = false)
{
    auto transfer_fields = [](BeFileSource& ff1, space_t& old_space, space_t& space, Base_tensor& basis,
                              config_t& cfg) {
        Scalar old_conf(old_space, ff1);
        Scalar old_lapse(old_space, ff1);
        Vector old_shift(old_space, ff1);
        Scalar old_logh(old_space, ff1);

        Scalar conf(space);
        conf = 1.;
        conf.std_base();

        Scalar lapse(space);
        lapse = 1.;
        lapse.std_base();

        Vector shift(space, CON, basis);
        for (int i = 1; i <= 3; i++)
            shift.set(i).annule_hard();
        shift.std_base();

        Scalar logh(space);
        logh.annule_hard();
        logh.std_base();

        conf.import(old_conf);
        lapse.import(old_lapse);
        logh.import(old_logh);

        shift.set(1).import(old_shift.set(1));
        shift.set(2).import(old_shift.set(2));
        shift.set(3).import(old_shift.set(3));

        lapse.std_base();
        conf.std_base();
        logh.std_base();
        shift.std_base();

        bco_utils::save_to_file(space, cfg, conf, lapse, shift, logh);
    };

    return ns_3d_xcts_interpolate_on_new_grid_fields<config_t, space_t>(
        bconfig, new_res, outputfile, use_config_vars, transfer_fields);
}

} // namespace Kadath
