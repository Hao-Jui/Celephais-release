#pragma once

#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_three_body.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace Kadath
{

inline std::string converged_stage_prefix(const std::string& base, const std::string& stage)
{
    std::stringstream ss;
    ss << base;
    if (!stage.empty())
        ss << "_" << stage << ".";
    else
        ss << ".";
    return ss.str();
}

template <typename config_t>
std::string converged_eos_stem(config_t& bconfig)
{
    const std::string eos_file = bconfig.template eos<std::string>(EOSFILE);
    return eos_file.substr(0, eos_file.find("."));
}

template <typename config_t, typename node_t>
std::string converged_eos_stem(config_t& bconfig, node_t node)
{
    const std::string eos_file = bconfig.template eos<std::string>(EOSFILE, node);
    return eos_file.substr(0, eos_file.find("."));
}

// Resolution token for converged filenames. The legacy radial token is preserved
// for ordinary layouts: "NN" on a radial-uniform space, and "NN_MM" when radial
// p-refinement makes the per-domain nr range mixed. Angular AMR appends only the
// extra mixed axes that can otherwise collide with the same radial token, e.g.
// "07_t07_09" for theta-only p-refinement and "07_p06_08" for nonstandard phi
// spread. The usual spherical/bispheric phi offset (np = nr-1 versus nr) is not
// enough to change the token, so existing converged files still resume.
template <typename space_t>
std::string converged_resolution(const space_t& space)
{
    std::array<int, 3> lowest{{std::numeric_limits<int>::max(),
                               std::numeric_limits<int>::max(),
                               std::numeric_limits<int>::max()}};
    std::array<int, 3> highest{{0, 0, 0}};
    int observed_axes = 0;
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const auto points = space.get_domain(d)->get_nbr_points();
        const int axes = std::min(points.get_ndim(), 3);
        observed_axes = std::max(observed_axes, axes);
        for (int a = 0; a < axes; ++a) {
            const int n = points(a);
            lowest[static_cast<std::size_t>(a)] =
                std::min(lowest[static_cast<std::size_t>(a)], n);
            highest[static_cast<std::size_t>(a)] =
                std::max(highest[static_cast<std::size_t>(a)], n);
        }
    }

    const auto append_two_digits = [](std::stringstream& ss, int value) {
        ss << std::setfill('0') << std::setw(2) << value;
    };

    std::stringstream ss;
    append_two_digits(ss, lowest[0]);
    if (highest[0] != lowest[0]) {
        ss << "_";
        append_two_digits(ss, highest[0]);
    }
    if (observed_axes > 1 && highest[1] != lowest[1]) {
        ss << "_t";
        append_two_digits(ss, lowest[1]);
        ss << "_";
        append_two_digits(ss, highest[1]);
    }
    if (observed_axes > 2 && highest[2] - lowest[2] > 1) {
        ss << "_p";
        append_two_digits(ss, lowest[2]);
        ss << "_";
        append_two_digits(ss, highest[2]);
    }
    return ss.str();
}

template <typename config_t, typename space_t>
std::string converged_ns_filename(config_t& bconfig, const space_t& space,
                                  const std::string& stage, const std::string& base,
                                  bool include_spin_axis_degree,
                                  const std::string& suffix = "")
{
    const auto res = converged_resolution(space);
    std::stringstream ss;
    ss << converged_stage_prefix(base, stage);
    ss << converged_eos_stem(bconfig) << "." << bconfig(MADM) << ".";
    if (stage != "NOROT") {
        ss << bconfig(CHI) << ".";
        if (include_spin_axis_degree)
            ss << "z" << bconfig(DEG) << ".";
    } else {
        ss << "0.";
    }
    ss << bconfig(NSHELLS) << "." << res;
    ss << suffix;
    return ss.str();
}

template <typename config_t, typename space_t>
std::string converged_gr_ns_filename(config_t& bconfig, const space_t& space,
                                     const std::string& stage)
{
    return converged_ns_filename(bconfig, space, stage, "converged_NS", false);
}

template <typename config_t, typename space_t>
std::string converged_gr_ns_nosym_filename(config_t& bconfig, const space_t& space,
                                           const std::string& stage)
{
    return converged_ns_filename(bconfig, space, stage, "converged_NS_NOSYM", true);
}


template <typename config_t, typename space_t>
std::string converged_ns2d_diffrot_filename(config_t& bconfig, const space_t& space,
                                            const std::string& /*stage*/)
{
    const auto res = converged_resolution(space);
    const std::string law = bconfig.template diffrot<std::string>(DIFF_LAW);
    std::stringstream ss;
    ss << "converged_NS_2D.";
    ss << converged_eos_stem(bconfig) << "." << bconfig(MADM) << "." << bconfig(CHI) << ".";
    ss << bconfig(NSHELLS) << "." << res;
    ss << "." << law;
    if (law == "keh") {
        ss << "_cstA_" << bconfig.template diffrot<double>(CSTA)
           << "_cstp_" << bconfig.template diffrot<double>(CSTP);
    }
    return ss.str();
}

template <typename config_t, typename space_t>
std::string converged_ns_diffrot_filename(config_t& bconfig, const space_t& space,
                                          const std::string& stage)
{
    return converged_ns2d_diffrot_filename(bconfig, space, stage);
}

template <typename config_t, typename space_t>
std::string converged_bh_filename(config_t& bconfig, const space_t& space,
                                  const std::string& stage, const std::string& base,
                                  const std::string& scalar_tag = "")
{
    const auto res = converged_resolution(space);
    std::stringstream ss;
    ss << converged_stage_prefix(base, stage);
    ss << bconfig(MCH) << "." << bconfig(CHI) << "." << scalar_tag
       << std::setfill('0') << std::setw(1) << bconfig(NSHELLS) << "."
       << res;
    return ss.str();
}

template <typename config_t, typename space_t>
std::string converged_bh_scalar_filename(config_t& bconfig, const space_t& space,
                                         const std::string& stage, const std::string& base,
                                         const std::string& active_scalar_tag)
{
    const std::string scalar_tag = bconfig.template gravity<double>(GRAV_MASS_SCAL) == 0.0
                                       ? ""
                                       : active_scalar_tag;
    return converged_bh_filename(bconfig, space, stage, base, scalar_tag);
}

template <typename config_t, typename space_t>
std::string converged_gr_bh_filename(config_t& bconfig, const space_t& space,
                                     const std::string& stage)
{
    return converged_bh_filename(bconfig, space, stage, "converged_BH");
}

template <typename config_t, typename space_t>
std::string converged_trumpet_filename(config_t& bconfig, const space_t& space,
                                       const std::string& stage)
{
    return converged_bh_filename(bconfig, space, stage, "converged_Trumpet");
}

template <typename config_t, typename space_t>
std::string converged_bh2d_scalar_filename(config_t& bconfig, const space_t& space,
                                           const std::string& stage)
{
    return converged_bh_scalar_filename(bconfig, space, stage, "converged_BH", "scal.");
}

template <typename config_t, typename space_t>
std::string converged_bh2d_xi_filename(config_t& bconfig, const space_t& space,
                                       const std::string& stage)
{
    return converged_bh_scalar_filename(bconfig, space, stage, "converged_BH", "xi.");
}

template <typename config_t, typename space_t>
std::string converged_bns_filename(config_t& bconfig, const space_t& space,
                                   const std::string& stage, const std::string& base,
                                   const std::string& suffix = "")
{
    const auto res = converged_resolution(space);
    auto M1 = bconfig(MADM, BCO1);
    auto M2 = bconfig(MADM, BCO2);
    if (M2 > M1)
        std::swap(M1, M2);
    bconfig.set(Q) = M2 / M1;

    std::stringstream ss;
    ss << converged_stage_prefix(base, stage);
    ss << converged_eos_stem(bconfig, BCO1) << "." << bconfig(DIST) << ".";
    // Per-body spin magnitude (CHI) + spin-axis tilt as z<deg> (matching the NS
    // seed naming). Sym apps carry DEG=0 (the symmetric space can't represent a
    // tilt) so this reads z0, unchanged from before; nosym apps get the actual
    // angle (z30, z45, ...).
    ss << bconfig(CHI, BCO1) << ".z" << bconfig(DEG, BCO1) << ".";
    ss << bconfig(CHI, BCO2) << ".z" << bconfig(DEG, BCO2) << ".";
    ss << bconfig(MADM, BCO1) + bconfig(MADM, BCO2) << ".q" << bconfig(Q) << ".";
    ss << std::setfill('0') << std::setw(1) << bconfig(OUTER_SHELLS) << ".";
    ss << std::setfill('0') << std::setw(1) << bconfig(NSHELLS, BCO1) << "."
       << std::setfill('0') << std::setw(1) << bconfig(NSHELLS, BCO2) << "."
       << res;
    ss << suffix;
    return ss.str();
}

template <typename config_t, typename space_t>
std::string converged_gr_bns_filename(config_t& bconfig, const space_t& space,
                                      const std::string& stage)
{
    return converged_bns_filename(bconfig, space, stage, "converged_BNS");
}

template <typename config_t, typename space_t>
std::string converged_gr_bns_nosym_filename(config_t& bconfig, const space_t& space,
                                            const std::string& stage)
{
    return converged_bns_filename(bconfig, space, stage, "converged_BNS_NOSYM");
}


template <typename config_t, typename space_t>
std::string converged_bhns_filename(config_t& bconfig, const space_t& space,
                                    const std::string& stage,
                                    bool include_spin_axis_degree = false)
{
    const auto res = converged_resolution(space);
    auto M1 = bconfig(MADM, BCO1);
    auto M2 = bconfig(MCH, BCO2);
    if (M2 > M1)
        std::swap(M1, M2);
    bconfig.set(Q) = M2 / M1;
    const auto Mtot = M1 + M2;

    std::stringstream ss;
    ss << converged_stage_prefix("converged_BHNS", stage);
    ss << converged_eos_stem(bconfig, BCO1) << "." << bconfig(DIST) << "."
       << bconfig(CHI, BCO1) << ".";
    // nosym apps encode the per-body spin-axis tilt as z<deg>; sym BHNS omits it
    // (DEG=0) to keep existing filenames unchanged. Unlike the BNS name, sym here
    // emits no z-token at all, so this stays conditional.
    if (include_spin_axis_degree)
        ss << "z" << bconfig(DEG, BCO1) << ".";
    ss << bconfig(CHI, BCO2) << ".";
    if (include_spin_axis_degree)
        ss << "z" << bconfig(DEG, BCO2) << ".";
    ss << Mtot << ".q" << bconfig(Q) << "." << std::setfill('0') << std::setw(1)
       << bconfig(NSHELLS, BCO1) << "." << std::setfill('0') << std::setw(1)
       << bconfig(NSHELLS, BCO2) << "." << res;
    return ss.str();
}

template <typename config_t, typename space_t>
std::string converged_gr_bhns_filename(config_t& bconfig, const space_t& space,
                                       const std::string& stage)
{
    return converged_bhns_filename(bconfig, space, stage);
}

template <typename config_t, typename space_t>
std::string converged_gr_bhns_nosym_filename(config_t& bconfig, const space_t& space,
                                             const std::string& stage)
{
    return converged_bhns_filename(bconfig, space, stage, /*include_spin_axis_degree=*/true);
}

template <typename config_t, typename space_t>
std::string converged_bbh_filename(config_t& bconfig, const space_t& space,
                                   const std::string& stage)
{
    const auto res = converged_resolution(space);
    auto M1 = bconfig(MCH, BCO1);
    auto M2 = bconfig(MCH, BCO2);
    if (M2 > M1)
        std::swap(M1, M2);
    bconfig.set(Q) = M2 / M1;
    const auto Mtot = M1 + M2;

    std::stringstream ss;
    ss << converged_stage_prefix("converged_BBH", stage);
    ss << bconfig(DIST) << "." << bconfig(CHI, BCO1) << "."
       << bconfig(CHI, BCO2) << "." << Mtot << ".q" << bconfig(Q)
       << "." << std::setfill('0') << std::setw(1) << bconfig(NSHELLS, BCO1)
       << "." << std::setfill('0') << std::setw(1) << bconfig(NSHELLS, BCO2)
       << "." << res;
    return ss.str();
}

template <typename config_t, typename space_t>
std::string converged_gr_bbh_filename(config_t& bconfig, const space_t& space,
                                      const std::string& stage)
{
    return converged_bbh_filename(bconfig, space, stage);
}

template <typename config_t, typename space_t>
std::string converged_three_body_filename(config_t& bconfig, const space_t& space,
                                          const std::string& stage)
{
    const auto res = converged_resolution(space);
    std::stringstream ss;
    ss << converged_stage_prefix("converged_3NS", stage);
    ss << converged_eos_stem(bconfig, BCO1) << "." << bconfig(PARENT_DIST)
       << "." << bconfig(CHILD_DIST) << "."
       << bconfig(MADM, BCO1) + bconfig(MADM, BCO2) + bconfig(MADM, BCO3)
       << "." << res;
    return ss.str();
}

} // namespace Kadath
