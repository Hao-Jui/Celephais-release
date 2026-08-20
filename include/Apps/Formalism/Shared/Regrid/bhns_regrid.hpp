#pragma once

// Shared black-hole–neutron-star regrid core. Templated on space_t so it serves
// both Space_bhns and Space_bhns_nosym. BHNS geometry is asymmetric: the NS side
// carries nucleus + (inner/outer) adapted shells, the BH side carries
// nucleus + (inner/outer) homothetic shells. The shared core owns all geometry
// (config-var update, bounds, adapted-mapping interpolation, old_space_radius,
// excision setup) and file IO; the per-formalism field set is supplied by the
// transfer_fields callback (mirrors Shared/Regrid/bns_regrid.hpp and Shared/Regrid/ns_regrid.hpp).
//
// The conformal factor (old_conf) is read by the core because the BH inner-radius
// estimate (see https://arxiv.org/pdf/0805.4192, eq(64)) needs it; the file read
// order old_space -> old_conf -> [callback: old_lapse, old_shift, old_logh, old_phi]
// reproduces the historical sequential BeFileSource read byte-for-byte.

#include "For_Kadath/Kadath_point_h/kadath.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "Apps/Policy/app_resolution.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>

namespace Kadath {

// ---------------------------------------------------------------------------
// Space-type traits — adapted (NS-side) and homothetic (BH-side) domain types
// per BHNS space. The GR (Space_bhns) specialisation lives here; the nosym
// specialisation lives in GR/BHNS_XCTS_nosym/regrid.hpp next to the nosym
// domain types it references.
// ---------------------------------------------------------------------------
template <typename space_t>
struct bhns_space_traits;

template <>
struct bhns_space_traits<Space_bhns> {
    using outer_adapted = Domain_shell_outer_adapted;
    using inner_adapted = Domain_shell_inner_adapted;
    using outer_homothetic = Domain_shell_outer_homothetic;
    using inner_homothetic = Domain_shell_inner_homothetic;
};

// ---------------------------------------------------------------------------
// Geometry/field context handed to the per-formalism field-transfer callback.
// The core owns all shared geometry (old_space, new space, basis, old/new NS
// adapted-domain pointers, BH outer-homothetic pointer, domain indices); the
// callback owns the formalism-specific field set (GR: conf/lapse/shift/logh/phi).
// old_conf is read by the core (the BH inner-radius estimate needs it) and is
// handed to the callback for the import step.
// ---------------------------------------------------------------------------
template <typename space_t>
struct bhns_regrid_transfer_context {
    using traits = bhns_space_traits<space_t>;
    using outer_homothetic_t = typename traits::outer_homothetic;
    using inner_adapted_t = typename traits::inner_adapted;

    BeFileSource& fin;
    space_t& old_space;
    space_t& space;
    Base_tensor& basis;
    Scalar& old_conf;                          // read by the core, imported by the callback
    const outer_homothetic_t* old_bh_outer;    // BH-side outer homothetic (old space)
    const inner_adapted_t* old_inner_adaptedNS; // NS-side inner adapted (old space)
};

// ---------------------------------------------------------------------------
// Shared BHNS regrid core. Templated on space_t (Space_bhns / Space_bhns_nosym).
// The per-formalism field set is supplied by the transfer_fields callback.
//
// use_config_vars=false (default): recompute RIN/RMID/ROUT from the old space
// (the historical path); true: keep the config values as supplied.
// ---------------------------------------------------------------------------
// res_per_domain: optional per-domain resolution for the NEW space (absolute
// domain indexing, one Dim_array{nr,nt,np} per domain). Empty means uniform
// BIN_RES everywhere (the historical path); non-empty drives per-domain AMR
// p-refinement (including the bispheric phi block) via the vector<Dim_array>
// space constructor. Mirrors Shared/Regrid/bns_regrid.hpp.
template <typename config_t, typename space_t, typename TransferFields>
int bhns_xcts_regrid_impl_fields(config_t& bconfig, std::string output_fname,
                                 bool use_config_vars,
                                 TransferFields&& transfer_fields,
                                 const std::vector<Dim_array>& res_per_domain = {})
{
    using traits = bhns_space_traits<space_t>;
    using outer_adapted_t = typename traits::outer_adapted;
    using inner_adapted_t = typename traits::inner_adapted;
    using outer_homothetic_t = typename traits::outer_homothetic;

    bconfig.set(Q) = bconfig(MADM, BCO1) / bconfig(MCH, BCO2);

    if (std::isnan(bconfig.set(OUTER_SHELLS)))
        bconfig.set(OUTER_SHELLS) = 0;

    std::string kadath_filename = bconfig.space_filename();

    BeFileSource fin(kadath_filename);
    space_t old_space(fin);
    Scalar old_conf(old_space, fin);

    const std::array<int, 2> old_adapted_doms{old_space.ADAPTEDNS, old_space.ADAPTEDBH};
    const std::array<int, 2> old_nuc_doms{old_space.NS, old_space.BH};

    const outer_adapted_t* old_outer_adaptedNS =
        dynamic_cast<const outer_adapted_t*>(old_space.get_domain(old_space.ADAPTEDNS));

    const inner_adapted_t* old_inner_adaptedNS =
        dynamic_cast<const inner_adapted_t*>(old_space.get_domain(old_space.ADAPTEDNS + 1));

    const outer_homothetic_t* old_bh_outer =
        dynamic_cast<const outer_homothetic_t*>(old_space.get_domain(old_space.ADAPTEDBH));

    std::cout << "Resolution of old space: " << old_space.get_domain(0)->get_nbr_points()(0) << " (r), "
              << old_space.get_domain(0)->get_nbr_points()(1) << " (theta), "
              << old_space.get_domain(0)->get_nbr_points()(2) << " (phi)" << std::endl;

    int ndom = old_space.get_nbr_domains();

    int res = bconfig(BIN_RES);

    validate_resolution(res);

    // The new-space resolution banner only accompanies uniform regrids; AMR
    // p-refinement keeps the geometry fixed and prints its own per-domain table.
    if (res_per_domain.empty())
        std::cout << "Resolution of new space: " << res << " (r), " << res << " (theta), " << res - 1 << " (phi)"
                  << std::endl;

    int type_coloc = old_space.get_type_base();

    // Update config vars
    // This control was mainly for testing
    if (!use_config_vars) {
        // Loop for ease of initialization
        for (int i = 0; i < 2; ++i) {
            bco_utils::set_radius(old_nuc_doms[i], old_space, bconfig, RIN, i);
            bco_utils::set_radius(old_adapted_doms[i], old_space, bconfig, FIXED_R, i);
        }

        // update NS radii
        auto [rmin, rmax] = bco_utils::get_rmin_rmax(old_space, old_space.ADAPTEDNS);

        std::cout << "Rmin/max: " << std::endl << rmin << " " << rmax << std::endl;
        bconfig.set(RMID, BCO1) = rmin;
        bconfig.set(RIN, BCO1) = 0.5 * rmin;
        // end update NS radii

        // estimate how small the inner radius should be based on relation
        // between conformal factor and numerical radius.
        // see https://arxiv.org/pdf/0805.4192, eq(64)
        auto [cmin, cmax] = bco_utils::get_field_min_max(old_conf, old_space.ADAPTEDBH + 1, INNER_BC);
        double conf_inner = cmin;
        double conf_i_sq = conf_inner * conf_inner;
        double est_r_div2 = bconfig(MCH, BCO2) / conf_i_sq;
        // [REGRID-FIX] BH inner (excision) radius: reuse the converged seam rather
        // than the conformal MCH/Psi^2 estimate, which drifts it (10.38 -> 12.87)
        // and lets import() smear fields across the moved excision boundary. The
        // estimate is only needed for large M_BH continuation, not a pure res bump.
        bconfig.set(RIN, BCO2) = bco_utils::get_radius(old_space.get_domain(old_space.BH), OUTER_BC);

        // this estimate is critical for calculating domain bounds
        // especially when attempting large changes in M_BH
        bconfig.set(RMID, BCO2) = 2 * est_r_div2;

        // [REGRID-FIX] BH outer-shell radius: reuse the converged seam instead of
        // the geometric (DIST/2-rmax)/3+rmax estimate. A pure resolution bump must
        // preserve the old domain boundaries; the estimate drifts the BH outer
        // (51.6 -> 63.8) and import() then smears conf/shift across the moved seam
        // in the neck, producing a spurious BH spin.
        //
        // The seam is the OUTER_BC of the LAST BH-side domain before the bispheric
        // block, i.e. OUTER-1 == BH + 2 + n_shells2. The earlier `BH + 2` was the
        // outermost only for n_shells2 == 0; once an h-move added a BH shell it
        // pointed at an inner homothetic shell, collapsing the seam (51.57 -> 38.53)
        // and globally misregistering the imported fields (NS Mb lost ~12 %, BH Mirr
        // corrupted). Index from OUTER so it is robust to any BH shell count.
        bconfig.set(ROUT, BCO2) =
            bco_utils::get_radius(old_space.get_domain(old_space.OUTER - 1), OUTER_BC);
    } // end updating config vars

    // create old radius scalar field
    Scalar old_space_radius(old_space);
    old_space_radius.annule_hard();
    for (int d = 0; d < ndom; ++d) {
        old_space_radius.set_domain(d) = old_space.get_domain(d)->get_radius();
    }

    // Get outer domain radii
    old_space_radius.set_domain(old_space.ADAPTEDNS) = old_outer_adaptedNS->get_outer_radius();
    old_space_radius.set_domain(old_space.ADAPTEDBH) = old_bh_outer->get_outer_radius();

    old_space_radius.std_base();
    // end creating old radius scalar fields

    // setup bounds for creating new space
    std::vector<double> out_bounds(1 + bconfig(OUTER_SHELLS));
    std::vector<double> NS_bounds = bco_utils::make_binary_NS_bounds(bconfig, BCO1);
    std::vector<double> BH_bounds(3 + bconfig(NSHELLS, BCO2));

    for (int e = 0; e < static_cast<int>(out_bounds.size()); ++e)
        out_bounds[e] = bconfig(REXT) * (1. + e * 0.25);

    // Keep the BH homothetic outer radius at the current value so the solver
    // starts from the original solution.
    bconfig.set(RMID, BCO2) = bco_utils::get_radius(old_space.get_domain(old_space.BH + 1), OUTER_BC);

    bco_utils::set_BH_bounds(BH_bounds, bconfig, BCO2);
    // end setup bounds

    // Bounds going into the new space (mirrors the BNS regrid print). The NS
    // side carries [rin, rmid, rout, shells..., r_bisph]; r_bisph (last entry)
    // is the bispheric matching radius. Serves Space_bhns and Space_bhns_nosym
    // alike (shared templated core).
    std::cout << "Bounds: [rin, rmid, rout, shells] + r_bisph" << std::endl;
    // Each star/BH line: "[full bound vector] + matching radius" (the last bound:
    // r_bisph on the NS side, the outer-homothetic radius on the BH side).
    auto line = [](const char* label, const std::vector<double>& b) {
        std::cout << label << ": [";
        for (std::size_t i = 0; i < b.size(); ++i)
            std::cout << (i ? " " : "") << b[i];
        std::cout << "] + " << b.back() << std::endl;
    };
    line("NS", NS_bounds);
    line("BH", BH_bounds);
    std::cout << "Outer bounds:";
    for (const double v : out_bounds)
        std::cout << " " << v;
    std::cout << std::endl;

    space_t space = [&]() -> space_t {
        if (res_per_domain.empty())
            return space_t(type_coloc, bconfig(DIST), NS_bounds, BH_bounds, out_bounds, bconfig(BIN_RES));
        return space_t(type_coloc, bconfig(DIST), NS_bounds, BH_bounds, out_bounds, res_per_domain);  // vector<Dim_array> ctor
    }();
    Base_tensor basis(space, CARTESIAN_BASIS);

    const inner_adapted_t* new_ns_inner =
        dynamic_cast<const inner_adapted_t*>(space.get_domain(space.ADAPTEDNS + 1));
    const outer_adapted_t* new_ns_outer =
        dynamic_cast<const outer_adapted_t*>(space.get_domain(space.ADAPTEDNS));

    // Updated mapping for NS adapted fields
    bco_utils::interp_adapted_mapping(new_ns_inner, old_space.ADAPTEDNS, old_space_radius);
    bco_utils::interp_adapted_mapping(new_ns_outer, old_space.ADAPTEDNS, old_space_radius);

    bconfig.set_filename(output_fname);

    bhns_regrid_transfer_context<space_t> ctx{
        fin, old_space, space, basis,
        old_conf, old_bh_outer, old_inner_adaptedNS};
    std::forward<TransferFields>(transfer_fields)(ctx, bconfig);

    return EXIT_SUCCESS;
}

} // namespace Kadath
