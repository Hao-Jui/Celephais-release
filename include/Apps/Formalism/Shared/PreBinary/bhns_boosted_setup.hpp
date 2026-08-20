#pragma once
// Shared binary BH-NS "boosted" superposition setup. Builds the binary initial
// guess by superimposing a converged isolated-NS seed and a converged
// isolated-BH seed with exponential decay weights, then saves the binary
// dataset. Hoisted from the two identical per-symmetry copies (the symmetric
// BHNS_XCTS solver_imp and the BHNS_XCTS_nosym solver_imp).
//
// Space / adapted-domain types are template parameters (fixed per symmetry in
// the thin wrappers that each solver_imp provides), so no formalism / symmetry
// solver bucket is pulled in here. The body is line-for-line the historical
// per-symmetry function; the symmetric and nosym variants differed only in the
// concrete Space / Domain type names.

#include "mpi.h"
#include "Hydro/EOS.hh"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"
#include "For_Kadath/IO/be_file_source.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace Kadath {

// Build the boosted BHNS binary seed. Type parameters:
//   NsSpaceT          - isolated-NS space type on disk
//   BhSpaceT          - isolated-BH space type on disk
//   BinSpaceT         - target binary BHNS space type
//   NsOuterAdaptedT   - NS outer adapted-shell domain type
//   NsInnerAdaptedT   - NS inner adapted-shell domain type
//   BhOuterHomotheticT- BH outer homothetic-shell domain type
template <typename eos_t, typename NsSpaceT, typename BhSpaceT, typename BinSpaceT,
          typename NsOuterAdaptedT, typename NsInnerAdaptedT, typename BhOuterHomotheticT>
inline void bhns_setup_boosted_3d_impl(kadath_config<BCO_NS_INFO>& NSconfig,
                                       kadath_config<BCO_BH_INFO>& BHconfig,
                                       kadath_config<BIN_INFO>& bconfig)
{
    using namespace Kadath::Margherita;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // open previous ns solution
    std::string nsspaceinf = NSconfig.space_filename();
    BeFileSource ff1(nsspaceinf);
    NsSpaceT nsspacein(ff1);
    Scalar nsconf(nsspacein, ff1);
    Scalar nslapse(nsspacein, ff1);
    Vector nsshift(nsspacein, ff1);
    Scalar nslogh(nsspacein, ff1);
    Scalar nsphi(nsspacein, ff1);
    // end opening bns solution

    // update NSconfig quantities before updating binary configuration file
    NSconfig.set(HC) = std::exp(bco_utils::get_boundary_val(0, nslogh));
    NSconfig.set(NC) = EOS<eos_t, eos_var_t::DENSITY>::get(NSconfig(HC));
    bco_utils::update_config_NS_radii(nsspacein, NSconfig, 1);

    // obtain adapted NS shells for radius information and copying adapted mapping later
    const NsOuterAdaptedT* old_outer_adapted1 =
        dynamic_cast<const NsOuterAdaptedT*>(nsspacein.get_domain(1));
    const NsInnerAdaptedT* old_inner_adapted1 =
        dynamic_cast<const NsInnerAdaptedT*>(nsspacein.get_domain(2));

    // setup radius field - needed for copying the adapted domain mappings.
    Scalar old_space_radius(nsspacein);
    old_space_radius.annule_hard();

    int ndominns = nsspacein.get_nbr_domains();

    for (int d = 0; d < ndominns; ++d)
        old_space_radius.set_domain(d) = nsspacein.get_domain(d)->get_radius();

    old_space_radius.set_domain(1) = old_outer_adapted1->get_outer_radius();
    old_space_radius.std_base();
    // end setup radius field

    // open old BH solution
    std::string bhspaceinf = BHconfig.space_filename();
    BeFileSource ff2(bhspaceinf);
    BhSpaceT bhspacein(ff2);
    Scalar bhconf(bhspacein, ff2);
    Scalar bhlapse(bhspacein, ff2);
    Vector bhshift(bhspacein, ff2);
    // end open BH solution
    bco_utils::update_config_BH_radii(bhspacein, BHconfig, 1, bhconf);

    auto interp_field = [&](auto& space, int outer_dom, auto& old_phi) {
        const int d = outer_dom;
        const NsInnerAdaptedT* old_inner =
            dynamic_cast<const NsInnerAdaptedT*>(space.get_domain(d + 1));
        // interpolate old_phi field outside of the star for import
        bco_utils::update_adapted_field(old_phi, d, d + 1, old_inner, INNER_BC);
    };

    // in case we used boosted TOVs, we need to import PHI
    if (NSconfig.set_field(PHI) == true)
        interp_field(nsspacein, 1, nsphi);

    // start Update config vars
    const double rbisph_fill = bco_utils::resolved_rbisph_fill(bconfig);
    if (rbisph_fill > 0.)
        bconfig.set(ROUT, BCO2) =
            bco_utils::blended_rbisph(bconfig(RMID, BCO2), bconfig(DIST), rbisph_fill);
    // end updating config vars

    // setup domain boundaries
    std::vector<double> out_bounds(1 + bconfig(OUTER_SHELLS));
    std::vector<double> NS_bounds = bco_utils::make_binary_NS_bounds(bconfig, BCO1);
    std::vector<double> BH_bounds(3 + bconfig(NSHELLS, BCO2));

    bco_utils::set_BH_bounds(BH_bounds, bconfig, BCO2, true);

    // for out_bounds.size > 1 - add equi-distance shells
    for (int e = 0; e < static_cast<int>(out_bounds.size()); ++e)
        out_bounds[e] = bconfig(REXT) + e * 0.25 * bconfig(REXT);

    bco_utils::print_bounds("NS-bounds", NS_bounds);
    bco_utils::print_bounds("BH-bounds", BH_bounds);
    bco_utils::print_bounds("outer-bounds", out_bounds);

    // Setup actual space
    int typer = CHEB_TYPE;
    BinSpaceT space(typer, bconfig(DIST), NS_bounds, BH_bounds, out_bounds, bconfig(BIN_RES));
    Base_tensor basis(space, CARTESIAN_BASIS);

    const NsInnerAdaptedT* new_ns_inner =
        dynamic_cast<const NsInnerAdaptedT*>(space.get_domain(space.ADAPTEDNS + 1));
    const NsOuterAdaptedT* new_ns_outer =
        dynamic_cast<const NsOuterAdaptedT*>(space.get_domain(space.ADAPTEDNS));

    const BhOuterHomotheticT* old_bh_outer =
        dynamic_cast<const BhOuterHomotheticT*>(bhspacein.get_domain(1));

    // Update BH fields based to help with interpolation later
    bco_utils::update_adapted_field(bhconf, 2, 1, old_bh_outer, OUTER_BC);
    bco_utils::update_adapted_field(bhlapse, 2, 1, old_bh_outer, OUTER_BC);

    // Updated mapping for NS
    bco_utils::interp_adapted_mapping(new_ns_inner, 1, old_space_radius);
    bco_utils::interp_adapted_mapping(new_ns_outer, 1, old_space_radius);

    double xc1 = bco_utils::get_center(space, space.NS);
    double xc2 = bco_utils::get_center(space, space.BH);

    std::cout << "xc1: " << xc1 << std::endl;
    std::cout << "xc2: " << xc2 << std::endl;

    if (NSconfig.set_field(PHI) == true)
        bco_utils::update_adapted_field(nsphi, 1, 2, old_inner_adapted1, INNER_BC);

    Scalar logh(space);
    logh.annule_hard();
    logh.std_base();

    Scalar conf(space);
    conf.annule_hard();

    Scalar lapse(space);
    lapse.annule_hard();

    Vector shift(space, CON, basis);
    for (int i = 1; i <= 3; i++)
        shift.set(i).annule_hard();

    Scalar phi(space);
    phi.annule_hard();

    const double ns_invw4 = bco_utils::set_decay(bconfig, BCO1);
    const double bh_invw4 = bco_utils::set_decay(bconfig, BCO2);

    if (rank == 0)
        std::cout << "WeightNS: " << bconfig(DECAY, BCO1) << ", "
                  << "WeightBH: " << bconfig(DECAY, BCO2) << std::endl;

    int ndom = space.get_nbr_domains();
    for (int dom = 0; dom < ndom; dom++) {
        // get an index in each domain to iterate over all colocation points
        Index new_pos(space.get_domain(dom)->get_nbr_points());

        do {
            // get cartesian coordinates of the current colocation point
            double x = space.get_domain(dom)->get_cart(1)(new_pos);
            double y = space.get_domain(dom)->get_cart(2)(new_pos);
            double z = space.get_domain(dom)->get_cart(3)(new_pos);

            // define a point shifted suitably to the stellar centers in the binary
            Point absol1(3);
            absol1.set(1) = (x - xc1);
            absol1.set(2) = y;
            absol1.set(3) = z;
            double r2 = y * y + z * z;
            double r2_1 = (x - xc1) * (x - xc1) + r2;
            double r4_1 = r2_1 * r2_1;
            double r4_invw4_1 = r4_1 * ns_invw4;
            double decay_1 = std::exp(-r4_invw4_1);

            Point absol2(3);
            absol2.set(1) = (x - xc2);
            absol2.set(2) = y;
            absol2.set(3) = z;
            double r2_2 = (x - xc2) * (x - xc2) + r2;
            double r4_2 = r2_2 * r2_2;
            double r4_invw4_2 = r4_2 * bh_invw4;
            double decay_2 = std::exp(-r4_invw4_2);

            if (dom < ndom - 1) {
                conf.set_domain(dom).set(new_pos) =
                    1. + decay_1 * (nsconf.val_point(absol1) - 1.) + decay_2 * (bhconf.val_point(absol2) - 1.);

                lapse.set_domain(dom).set(new_pos) =
                    1. + decay_1 * (nslapse.val_point(absol1) - 1.) + decay_2 * (bhlapse.val_point(absol2) - 1.);

                logh.set_domain(dom).set(new_pos) = 0. + decay_1 * nslogh.val_point(absol1);

                phi.set_domain(dom).set(new_pos) = 0;
                if (NSconfig.set_field(PHI) == true)
                    phi.set_domain(dom).set(new_pos) += decay_1 * nsphi.val_point(absol1);

                for (int i = 1; i <= 3; i++)
                    shift.set(i).set_domain(dom).set(new_pos) =
                        0. + decay_1 * nsshift(i).val_point(absol1) + decay_2 * bhshift(i).val_point(absol2);

            } else {
                // We have to set the compactified domain manually
                // since the outer collocation point is always
                // at inf which is undefined numerically
                conf.set_domain(dom).set(new_pos) = 1.;
                lapse.set_domain(dom).set(new_pos) = 1.;
            }
            // loop over all colocation points
        } while (new_pos.inc());
    } // end importing fields

    // Want to make sure there is no matter around the BH
    for (int d = space.BH; d < space.OUTER; ++d) {
        logh.set_domain(d).annule_hard();
        phi.set_domain(d).annule_hard();
    }
    for (int d = space.BH; d < space.BH + 2; ++d) {
        conf.set_domain(d).annule_hard();
        lapse.set_domain(d).annule_hard();
    }

    // employ standard spectral expansion, compatible with the given paraties
    conf.std_base();
    lapse.std_base();
    logh.std_base();
    shift.std_base();
    phi.std_base();

    // save everything to a binary file
    bco_utils::sync_mirr_from_mch(bconfig, BCO2);
    bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh, phi);
}

} // namespace Kadath
