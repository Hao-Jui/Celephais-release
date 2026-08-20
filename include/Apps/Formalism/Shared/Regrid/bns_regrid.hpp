#pragma once

#include "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include <sstream>
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "Apps/Policy/app_resolution.hpp"
#include "Apps/Formalism/Shared/Stages/bns_stage_diagnostics.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

namespace Kadath {

// ---------------------------------------------------------------------------
// Space-type traits — adapted-domain types per binary-NS space. The GR
// (Space_bin_ns) specialisation lives here; the nosym specialisation lives in
// GR/BNS_XCTS_nosym/regrid.hpp next to the nosym domain types it references.
// ---------------------------------------------------------------------------
template <typename space_t>
struct bns_space_traits;

template <>
struct bns_space_traits<Space_bin_ns> {
    using outer_adapted = Domain_shell_outer_adapted;
    using inner_adapted = Domain_shell_inner_adapted;
};

template <typename space_t>
struct bns_regrid_transfer_context {
    using traits = bns_space_traits<space_t>;
    using inner_adapted_t = typename traits::inner_adapted;

    BeFileSource& fin;
    space_t& old_space;
    space_t& space;
    Base_tensor& basis;
    const std::array<int, 2>& old_adapted_doms;
    const std::array<int, 2>& new_adapted_doms;
    const std::array<const inner_adapted_t*, 2>& old_inner_adapted;
    int ndom;       // new-space domain count
    int outer_dom;  // index of the first exterior domain (space.OUTER)

    // Zero matter (logh) and velocity potential (phi) outside the stars: in the
    // outer adapted shell of each star and in every exterior domain. Shared by
    // all formalisms (varscal is a global field and is NOT zeroed here).
    void zero_matter_outside_stars(Scalar& logh, Scalar& phi) const
    {
        for (auto& dom : new_adapted_doms) {
            phi.set_domain(dom + 1).annule_hard();
            logh.set_domain(dom + 1).annule_hard();
        }
        for (int i = outer_dom; i < ndom; ++i) {
            phi.set_domain(i).annule_hard();
            logh.set_domain(i).annule_hard();
        }
    }
};

// ---------------------------------------------------------------------------
// Diagnostic helpers for the regrid bounds dump.
//
// bns_star_domain_bounds reconstructs a star's radial boundary vector from a
// built space: the OUTER radius of each of the star's domains in order ->
// [rin, rmid, rout, outer shells..., r_bisph].
// star 0 = NS1 (domains NS1..NS2-1), star 1 = NS2 (domains NS2..OUTER-1).
// rin = nucleus, rmid = adapted surface, rout = adapted-pair outer edge,
// r_bisph (last) = NS_bounds.back() = the bispheric matching radius; with no
// outer shells routstar and r_bisph coincide.
// ---------------------------------------------------------------------------
template <typename space_t>
std::vector<double> bns_star_domain_bounds(const space_t& space, int star)
{
    const int begin = (star == 0) ? space.NS1 : space.NS2;
    const int end = (star == 0) ? space.NS2 : space.OUTER; // one past the star's last domain
    std::vector<double> bounds;
    for (int d = begin; d < end; ++d)
        bounds.push_back(bco_utils::get_radius(space.get_domain(d), OUTER_BC));
    return bounds;
}

// Exterior radial boundaries of a built space: the inner radius of every
// domain past the bispheric block (exterior shells, then the compactified
// domain) — i.e. [rext, shell boundaries...], the out_bounds vector the space
// was built with.
template <typename space_t>
std::vector<double> bns_exterior_bounds(const space_t& space)
{
    std::vector<double> bounds;
    for (int d = space.OUTER + 5; d < space.get_nbr_domains(); ++d)
        bounds.push_back(bco_utils::get_radius(space.get_domain(d), INNER_BC));
    return bounds;
}

inline void print_bns_bounds(const std::vector<double>& ns1, const std::vector<double>& ns2,
                             const std::vector<double>& outer)
{
    std::cout << "Bounds: [rin, rmid, rout, shells] + r_bisph" << std::endl;
    // Each star line: "[full bound vector] + r_bisph" (the last bound = the
    // bispheric matching radius).
    auto line = [](const char* label, const std::vector<double>& b) {
        std::cout << label << ": [";
        for (std::size_t i = 0; i < b.size(); ++i)
            std::cout << (i ? " " : "") << b[i];
        std::cout << "] + " << b.back() << std::endl;
    };
    line("NS1", ns1);
    line("NS2", ns2);
    std::cout << "Outer bounds:";
    for (const double v : outer)
        std::cout << " " << v;
    std::cout << std::endl;
}

template <typename config_t, typename space_t, typename TransferFields>
int bns_xcts_regrid_impl_fields(config_t& bconfig, std::string output_fname,
                                bool use_config_vars,
                                TransferFields&& transfer_fields,
                                const std::vector<Dim_array>& res_per_domain = {})
{
    using traits = bns_space_traits<space_t>;
    using outer_adapted_t = typename traits::outer_adapted;
    using inner_adapted_t = typename traits::inner_adapted;

    int exit_status = EXIT_SUCCESS;

    std::string kadath_filename = bconfig.space_filename();
    if (!fs::exists(kadath_filename)) {
        std::ostringstream oss;
        oss << "File: " << kadath_filename << " not found.\n\n";
        KADATH_THROW(oss.str());
    }

    BeFileSource fin(kadath_filename);
    space_t old_space(fin);

    const std::array<int, 2> old_adapted_doms{old_space.ADAPTED1, old_space.ADAPTED2};

    std::array<const outer_adapted_t*, 2> old_outer_adapted;
    std::array<const inner_adapted_t*, 2> old_inner_adapted;

    for (int i = 0; i < 2; ++i) {
        int const d = old_adapted_doms[i];
        old_outer_adapted[i] = dynamic_cast<const outer_adapted_t*>(old_space.get_domain(d));
        old_inner_adapted[i] = dynamic_cast<const inner_adapted_t*>(old_space.get_domain(d + 1));
    }

    int res = bconfig.set(BIN_RES);
    validate_resolution(res);
    int ndom = old_space.get_nbr_domains();

    // The old/new-space banner only accompanies uniform regrids (legacy ladder
    // and AMR h-moves, where the domain layout changes). AMR p-refinement
    // prints its own per-domain decision table and keeps the bounds fixed.
    if (res_per_domain.empty()) {
        std::cout << "Resolution of old space: " << old_space.get_domain(0)->get_nbr_points()(0) << " (r), "
                  << old_space.get_domain(0)->get_nbr_points()(1) << " (theta), "
                  << old_space.get_domain(0)->get_nbr_points()(2) << " (phi)" << std::endl;
        print_bns_bounds(bns_star_domain_bounds(old_space, 0), bns_star_domain_bounds(old_space, 1),
                         bns_exterior_bounds(old_space));
        std::cout << std::endl;
    }

    int type_coloc = old_space.get_type_base();

    double r_max_tot = 0.;

    for (int i = 0; i < 2; ++i) {
        int const dom = old_adapted_doms[i];
        auto [rmin, rmax] = bco_utils::get_rmin_rmax(old_space, dom);
        bconfig.set(RIN, i) = 0.5 * rmin;
        bconfig.set(FIXED_R, i) = rmin;
        r_max_tot = (rmax > r_max_tot) ? rmax : r_max_tot;
    }
    bconfig.set(ROUT, BCO1) = (bconfig(DIST) / 2. - r_max_tot) / 3. + r_max_tot;
    bconfig.set(ROUT, BCO2) = (bconfig(DIST) / 2. - r_max_tot) / 3. + r_max_tot;

    Scalar old_space_radius(old_space);
    old_space_radius.annule_hard();

    for (int d = 0; d < ndom; ++d) {
        old_space_radius.set_domain(d) = old_space.get_domain(d)->get_radius();
    }

    for (int i = 0; i < 2; ++i) {
        int const dom = old_adapted_doms[i];
        old_space_radius.set_domain(dom) = old_outer_adapted[i]->get_outer_radius();
    }
    old_space_radius.std_base();

    std::vector<double> out_bounds(1 + bconfig(OUTER_SHELLS));

    if (!use_config_vars) {
        for (int e = 0; e < static_cast<int>(out_bounds.size()); ++e)
            out_bounds[e] = bconfig(REXT) * (1. + e * 0.25);
    } else {
        out_bounds[0] = bconfig(REXT);
        for (int e = 1; e < static_cast<int>(out_bounds.size()); ++e)
            out_bounds[e] = 2 * out_bounds[e - 1];
    }

    // Allocate + fill each star's radial bounds in one call: size 3 + NSHELLS,
    // uniform across formalisms and both stars (the size is derived from the same
    // NSHELLS the fill reads, so it can never mismatch).
    std::vector<double> NS1_bounds = bco_utils::make_binary_NS_bounds(bconfig, BCO1);
    std::vector<double> NS2_bounds = bco_utils::make_binary_NS_bounds(bconfig, BCO2);

    space_t space = [&]() -> space_t {
        if (res_per_domain.empty())
            return space_t(type_coloc, bconfig(DIST), NS1_bounds, NS2_bounds, out_bounds, res);
        return space_t(type_coloc, bconfig(DIST), NS1_bounds, NS2_bounds, out_bounds, res_per_domain);   // vector<Dim_array> ctor
    }();
    ndom = space.get_nbr_domains();
    if (res_per_domain.empty()) {
        std::cout << "Resolution of new space: " << res << " (r), " << res << " (theta), " << res - 1 << " (phi)"
                  << std::endl;
        print_bns_bounds(NS1_bounds, NS2_bounds, out_bounds);
        std::cout << std::endl;
    }

    Base_tensor basis(space, CARTESIAN_BASIS);

    const std::array<int, 2> new_adapted_doms{space.ADAPTED1, space.ADAPTED2};
    const std::array<double, 2> xc{space.get_domain(space.NS1)->get_center()(1),
                                   space.get_domain(space.NS2)->get_center()(1)};

    if (res_per_domain.empty()) {
        std::cout << "xc1: " << xc[0] << std::endl;
        std::cout << "xc2: " << xc[1] << std::endl;
    }

    std::array<const outer_adapted_t*, 2> new_outer_adapted;
    std::array<const inner_adapted_t*, 2> new_inner_adapted;

    for (int i = 0; i < 2; ++i) {
        auto& d = new_adapted_doms[i];
        new_outer_adapted[i] = dynamic_cast<const outer_adapted_t*>(space.get_domain(d));
        new_inner_adapted[i] = dynamic_cast<const inner_adapted_t*>(space.get_domain(d + 1));
    }

    for (int i = 0; i < 2; ++i) {
        int const dom = old_adapted_doms[i];
        bco_utils::interp_adapted_mapping(new_inner_adapted[i], dom, old_space_radius);
        bco_utils::interp_adapted_mapping(new_outer_adapted[i], dom, old_space_radius);
    }

    bconfig.set_filename(output_fname);

    bns_regrid_transfer_context<space_t> ctx{
        fin, old_space, space, basis,
        old_adapted_doms, new_adapted_doms, old_inner_adapted,
        ndom, space.OUTER};
    std::forward<TransferFields>(transfer_fields)(ctx, bconfig);

    return exit_status;
}

} // namespace Kadath
