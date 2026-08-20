#pragma once

// Space-/physics-agnostic core of the binary AMR policy, shared by the BNS and
// BHNS AMR seeders (Apps/AMR/{bns,bhns}_amr_policy.hpp). Holds the per-domain
// per-axis spectral-tail decision type, the field/axis observation helpers, and
// the resolution-dispatch (amr_refine_if_needed). Each binary variant supplies
// only its own TailDecision seed (domain anchors + roles) and field-tail
// evaluator, then calls the shared dispatch here.

#include "Apps/AMR/bns_hp_indicators.hpp"
#include "Apps/AMR/refinement_policy_state.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "Apps/Policy/app_resolution.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace KadathApps::bns_amr
{
    // Per-domain per-direction refine decision. current_res / refine_axis /
    // domain_eligible are indexed by absolute domain index of the evaluated
    // space; only eligible domains can be flagged (the five bispheric domains are
    // masked — they pin the shared angular base). `p_refine_allowed`, when set,
    // separates measurement from actionability: a structurally unsupported axis
    // still contributes to the tail demands and h gate, but is recorded in
    // blocked_refine_axis instead of becoming a p candidate. `maximum` tracks the
    // largest tail ratio among eligible domains across ALL axes for the banner.
    //   current_res[d] = {nr, nt, np} as the space currently carries them.
    //   refine_axis[d] = {refine_r, refine_theta, refine_phi}: 1 when that axis'
    //     spectral tail on that domain exceeds the thresholds. Axis index follows
    //     TailAxisSelection: Radial=0, Theta=1, Phi=2.
    //   current_nr[d] = current_res[d][0] — the radial count, kept as a flat
    //     vector because the h-move region logic and the radial banners read it.
    // The adapted-pair indices feed the angular lock that keeps each star's two
    // adapted domains at identical nt/np (they share the deformable surface).
    struct TailDecision
    {
        bool refine = false;
        FieldTailMaximum maximum{};
        double radial_demand = 0.0;
        double angular_demand = 0.0;
        std::array<double, 3> radial_region_demand{{0.0, 0.0, 0.0}};
        double l2_threshold = 0.0;
        double linf_threshold = 0.0;
        std::vector<int> current_nr{};
        std::vector<std::array<int, 3>> current_res{};
        std::vector<std::array<char, 3>> refine_axis{};
        std::vector<std::array<char, 3>> blocked_refine_axis{};
        std::vector<std::array<double, 3>> refine_demand{};
        std::vector<std::array<double, 3>> blocked_refine_demand{};
        std::vector<std::array<char, 3>> p_refine_allowed{};
        std::vector<char> domain_eligible{};
        std::vector<std::string> domain_label{};
        int second_star_first_domain = -1;
        int first_exterior_domain = -1;
        int adapted1 = -1;  ///< First domain of star1's adapted pair (pair = adapted1, adapted1+1).
        int adapted2 = -1;  ///< First domain of star2's adapted pair (pair = adapted2, adapted2+1).
        // The five bispheric domains [bispheric_first, bispheric_first+5) share ONE
        // Dim_array and are masked from the per-domain machinery above. When
        // bispheric_enabled they refine as a LOCKED BLOCK in PHI (axis 2) ONLY: any
        // of the five exceeding a threshold arms that axis (bispheric_refine_axis,
        // all three tracked for the banner), and the dispatch bumps all five
        // together by +2 on phi. Phi is azimuthal — the same physical coordinate in
        // every bipolar variant — so a uniform bump keeps the conforming internal
        // seams square and the star-side / outer (import) seams absorb the jump to
        // the spherical neighbours. Axes 0/1 (the bipolar chi/eta, whose Dim_array
        // slot is a different physical direction per variant) are NOT refinable:
        // bumping them unbalances the conforming internal seams. No h-fallback exists
        // (no shell fits inside the chimera). bispheric_maximum tracks the block's
        // largest tail for the banner.
        int bispheric_first = -1;
        bool bispheric_enabled = false;
        std::array<char, 3> bispheric_refine_axis{{0, 0, 0}};
        std::array<double, 3> bispheric_refine_demand{{0.0, 0.0, 0.0}};
        FieldTailMaximum bispheric_maximum{};
    };

    inline bool mpi_is_initialized()
    {
        int initialized = 0;
        MPI_Initialized(&initialized);
        return initialized != 0;
    }

    inline void broadcast_string_from_root(std::string& value)
    {
        int length = static_cast<int>(value.size());
        MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
        value.resize(static_cast<std::size_t>(length));
        if (length > 0)
            MPI_Bcast(value.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    template <typename config_t>
    void apply_outputdir(config_t& bconfig, const std::string& outputdir, int rank, bool mpi_initialized)
    {
        if (outputdir.empty())
            return;

        if (rank == 0 || !mpi_initialized)
            std::filesystem::create_directories(outputdir);
        bconfig.set_outputdir(outputdir);
    }

    inline bool exceeds_threshold(const Kadath::bns_hp::SpectralTailRatio& ratio,
                                  double l2_threshold,
                                  double linf_threshold)
    {
        return ratio.l2_ratio > l2_threshold || ratio.linf_ratio > linf_threshold;
    }

    inline double tail_peak(const FieldTailMaximum& maximum)
    {
        return std::max(maximum.l2_ratio, maximum.linf_ratio);
    }

    inline double normalized_tail_demand(const Kadath::bns_hp::SpectralTailRatio& ratio,
                                           double l2_threshold,
                                           double linf_threshold)
    {
        if (!(l2_threshold > 0.0) || !(linf_threshold > 0.0))
            KADATH_THROW(
                "adaptive_mesh_refinement tail thresholds must be positive before normalizing tail demand");
        const double l2_demand = ratio.l2_ratio / l2_threshold;
        const double linf_demand = ratio.linf_ratio / linf_threshold;
        return std::max(l2_demand, linf_demand);
    }

    inline void update_tail_maximum(FieldTailMaximum& maximum,
                                    std::string_view field,
                                    int domain,
                                    int axis,
                                    double l2_ratio,
                                    double linf_ratio)
    {
        const double current_peak = std::max(l2_ratio, linf_ratio);
        if (current_peak <= tail_peak(maximum))
            return;
        maximum.field = std::string(field);
        maximum.domain = domain;
        maximum.l2_ratio = l2_ratio;
        maximum.linf_ratio = linf_ratio;
        maximum.axis = axis;
    }

    inline void ensure_refine_demand_shape(TailDecision& decision)
    {
        if (decision.refine_demand.size() != decision.refine_axis.size())
            decision.refine_demand.assign(decision.refine_axis.size(), {0.0, 0.0, 0.0});
        if (decision.blocked_refine_axis.size() != decision.refine_axis.size())
            decision.blocked_refine_axis.assign(decision.refine_axis.size(), {0, 0, 0});
        if (decision.blocked_refine_demand.size() != decision.refine_axis.size())
            decision.blocked_refine_demand.assign(decision.refine_axis.size(), {0.0, 0.0, 0.0});
    }

    inline bool p_refinement_allowed(const TailDecision& decision, int domain, int axis)
    {
        if (decision.p_refine_allowed.empty())
            return true;
        if (domain < 0 || domain >= static_cast<int>(decision.p_refine_allowed.size()))
            return false;
        return decision.p_refine_allowed[static_cast<std::size_t>(domain)]
                                        [static_cast<std::size_t>(axis)] != 0;
    }

    inline int h_region_index(const TailDecision& decision, int domain)
    {
        if (decision.second_star_first_domain < 0 || decision.first_exterior_domain < 0)
            return -1;
        if (domain < decision.second_star_first_domain)
            return 0;
        if (domain < decision.first_exterior_domain)
            return 1;
        return 2;
    }

    // Fold one field's per-domain tail ratios for ONE coordinate axis into the
    // decision. `axis` is the TailAxisSelection index (Radial=0, Theta=1, Phi=2)
    // and selects which slot of refine_axis[domain] this measurement arms.
    // Ineligible / out-of-range domains are skipped. `decision.maximum` tracks
    // the single largest tail ratio across every field and axis for the banner.
    inline void observe_axis(TailDecision& decision,
                             std::string_view field,
                             const std::vector<Kadath::bns_hp::SpectralTailRatio>& ratios,
                             int axis,
                             double l2_threshold,
                             double linf_threshold)
    {
        ensure_refine_demand_shape(decision);
        for (const auto& ratio : ratios) {
            if (ratio.domain < 0 || ratio.domain >= static_cast<int>(decision.domain_eligible.size()))
                continue;
            const double demand = normalized_tail_demand(ratio, l2_threshold, linf_threshold);

            // Bispheric block: the five domains are masked from the per-domain
            // machinery but, when enabled, fold into the locked-block decision —
            // any of the five exceeding a threshold on an axis arms that axis for
            // the whole block (they refine together). Tracked separately so the
            // eligible-domain `maximum` and the existing tests stay unchanged.
            if (decision.bispheric_enabled && decision.bispheric_first >= 0 &&
                ratio.domain >= decision.bispheric_first &&
                ratio.domain < decision.bispheric_first + 5) {
                update_tail_maximum(decision.bispheric_maximum, field, ratio.domain, axis,
                                    ratio.l2_ratio, ratio.linf_ratio);
                if (axis != 0 && demand > decision.angular_demand)
                    decision.angular_demand = demand;
                if (exceeds_threshold(ratio, l2_threshold, linf_threshold)) {
                    decision.refine = true;
                    decision.bispheric_refine_axis[static_cast<std::size_t>(axis)] = 1;
                    decision.bispheric_refine_demand[static_cast<std::size_t>(axis)] =
                        std::max(decision.bispheric_refine_demand[static_cast<std::size_t>(axis)],
                                 demand);
                }
                continue;
            }

            if (!decision.domain_eligible[static_cast<std::size_t>(ratio.domain)])
                continue;

            update_tail_maximum(decision.maximum, field, ratio.domain, axis,
                                ratio.l2_ratio, ratio.linf_ratio);
            if (axis == 0) {
                if (demand > decision.radial_demand)
                    decision.radial_demand = demand;
                const int region = h_region_index(decision, ratio.domain);
                if (region >= 0)
                    decision.radial_region_demand[static_cast<std::size_t>(region)] =
                        std::max(decision.radial_region_demand[static_cast<std::size_t>(region)],
                                 demand);
            } else if (demand > decision.angular_demand) {
                decision.angular_demand = demand;
            }

            if (exceeds_threshold(ratio, l2_threshold, linf_threshold)) {
                decision.refine = true;
                if (p_refinement_allowed(decision, ratio.domain, axis)) {
                    decision.refine_axis[static_cast<std::size_t>(ratio.domain)]
                                        [static_cast<std::size_t>(axis)] = 1;
                    decision.refine_demand[static_cast<std::size_t>(ratio.domain)]
                                            [static_cast<std::size_t>(axis)] =
                        std::max(decision.refine_demand[static_cast<std::size_t>(ratio.domain)]
                                                       [static_cast<std::size_t>(axis)],
                                 demand);
                } else {
                    decision.blocked_refine_axis[static_cast<std::size_t>(ratio.domain)]
                                                [static_cast<std::size_t>(axis)] = 1;
                    decision.blocked_refine_demand[static_cast<std::size_t>(ratio.domain)]
                                                    [static_cast<std::size_t>(axis)] =
                        std::max(decision.blocked_refine_demand
                                      [static_cast<std::size_t>(ratio.domain)]
                                      [static_cast<std::size_t>(axis)],
                                 demand);
                }
            }
        }
    }

    template <typename config_t>
    Kadath::bns_hp::SpectralTailOptions tail_options_from_config(const config_t& bconfig)
    {
        Kadath::bns_hp::SpectralTailOptions options;
        options.tail_width = bconfig.template amr_setting_as<int>(AMR_TAIL_WIDTH);
        if (options.tail_width <= 0)
            KADATH_THROW("adaptive_mesh_refinement.tail_width must be positive");
        options.norm_floor = bconfig.template amr_setting_as<double>(AMR_NORM_FLOOR);
        if (!(options.norm_floor >= 0.0))
            KADATH_THROW("adaptive_mesh_refinement.norm_floor must be non-negative");
        // The base tail width / norm floor only; the axis is set per call now that
        // each coordinate direction (radial, theta, phi) is judged separately.
        return options;
    }

    inline void observe_binary_common_fields(TailDecision& decision,
                                             const Kadath::bns_hp::SpectralTailOptions& options,
                                             double l2_threshold, double linf_threshold,
                                             const Kadath::Scalar& conf, const Kadath::Scalar& lapse,
                                             const Kadath::Vector& shift, const Kadath::Scalar& logh,
                                             const Kadath::Scalar& phi)
    {
        // Each coordinate direction is judged independently against the same
        // thresholds: the per-axis options share the base tail width / norm floor
        // and differ only in which axis' highest modes form the tail.
        const auto observe_scalar = [&](std::string_view name, const Kadath::Scalar& field) {
            const auto ratios = Kadath::bns_hp::scalar_spectral_axis_tail_ratios(field, options);
            for (std::size_t axis = 0; axis < ratios.size(); ++axis)
                observe_axis(decision, name, ratios[axis], static_cast<int>(axis),
                             l2_threshold, linf_threshold);
        };
        const auto observe_shift = [&]() {
            // Probe 12: judge each shift component's tail on its OWN norm and OR
            // the per-component flags (max the demand). A single aggregate norm
            // dilutes a small under-resolved component (e.g. shift_z) below
            // threshold; observe_axis already ORs flags / maxes demand when fed
            // each component's ratios, so looping the components is the fix.
            std::vector<Kadath::bns_hp::SpectralAxisRatioVectors> component_ratios;
            component_ratios.reserve(static_cast<std::size_t>(shift.get_n_comp()));
            for (int component = 0; component < shift.get_n_comp(); ++component)
                component_ratios.push_back(
                    Kadath::bns_hp::tensor_component_spectral_axis_tail_ratios(
                        shift, component, options));
            for (std::size_t axis = 0; axis < 3; ++axis)
                for (const auto& ratios : component_ratios)
                    observe_axis(decision, "shift", ratios[axis], static_cast<int>(axis),
                                 l2_threshold, linf_threshold);
        };
        observe_scalar("conf", conf);
        observe_scalar("lapse", lapse);
        observe_shift();
        observe_scalar("logh", logh);
        observe_scalar("phi", phi);
    }


    inline std::string amr_regrid_filename(int cycle, int resolution)
    {
        std::ostringstream name;
        name << "bns_amr_cycle" << cycle << "_res" << resolution;
        return name.str();
    }

    inline double dof_proxy(const std::vector<std::array<int, 3>>& resolution)
    {
        double total = 0.0;
        for (const auto& points : resolution)
            total += static_cast<double>(points[0]) * points[1] * points[2];
        return total;
    }

    inline void lock_star_core(std::vector<std::array<int, 3>>& target, int adapted)
    {
        const int ndom = static_cast<int>(target.size());
        if (adapted < 1 || adapted + 1 >= ndom)
            return;
        const std::size_t nuc = static_cast<std::size_t>(adapted - 1);
        const std::size_t out = static_cast<std::size_t>(adapted);
        const std::size_t inn = static_cast<std::size_t>(adapted + 1);
        const int nt = std::max({target[nuc][1], target[out][1], target[inn][1]});
        const int np = std::max({target[nuc][2], target[out][2], target[inn][2]});
        target[nuc][1] = target[out][1] = target[inn][1] = nt;
        target[nuc][2] = target[out][2] = target[inn][2] = np;
    }

    inline void lock_star_cores(std::vector<std::array<int, 3>>& target, int adapted1, int adapted2)
    {
        lock_star_core(target, adapted1);
        lock_star_core(target, adapted2);
    }

    // Legal action primitive: refine the bispheric block. The five domains
    // [first, first+5) share one Dim_array and bump together on the azimuthal
    // (phi, slot 2) axis only — the one refinable bispheric direction.
    inline void apply_bispheric_phi_block(std::vector<std::array<int, 3>>& grid, int first, int points)
    {
        for (int d = first; d < first + 5; ++d)
            grid[static_cast<std::size_t>(d)][2] = points;
    }

    // Per-direction flag tally over eligible domains: how many domains each axis
    // (radial / theta / phi) flagged this cycle, and how many are measured but
    // structurally blocked. Feeds the per-cycle diagnostic summary.
    struct FlaggedDirectionCounts
    {
        std::array<int, 3> flagged{0, 0, 0};
        std::array<int, 3> blocked{0, 0, 0};
    };

    inline FlaggedDirectionCounts count_flagged_directions(const TailDecision& decision, int ndom)
    {
        FlaggedDirectionCounts counts;
        for (int d = 0; d < ndom; ++d) {
            if (!decision.domain_eligible[static_cast<std::size_t>(d)])
                continue;
            for (int a = 0; a < 3; ++a)
                if (decision.refine_axis[static_cast<std::size_t>(d)][static_cast<std::size_t>(a)])
                    ++counts.flagged[static_cast<std::size_t>(a)];
            if (d < static_cast<int>(decision.blocked_refine_axis.size()))
                for (int a = 0; a < 3; ++a)
                    if (decision.blocked_refine_axis[static_cast<std::size_t>(d)]
                                                    [static_cast<std::size_t>(a)])
                        ++counts.blocked[static_cast<std::size_t>(a)];
        }
        return counts;
    }

    // Materialize an accepted {nr,nt,np} target as the space's regrid contract:
    // one Dim_array per domain, absolute domain indexing.
    inline std::vector<Kadath::Dim_array>
    build_res_per_domain(const std::vector<std::array<int, 3>>& target)
    {
        std::vector<Kadath::Dim_array> res_per_domain;
        res_per_domain.reserve(target.size());
        for (const auto& points3 : target) {
            Kadath::Dim_array points(3);
            points.set(0) = points3[0];
            points.set(1) = points3[1];
            points.set(2) = points3[2];
            res_per_domain.push_back(points);
        }
        return res_per_domain;
    }

    // AMR refinement dispatch. A domain+axis is flagged when that direction's
    // spectral tail ratio on that domain exceeds the thresholds (refine_axis,
    // axis index Radial=0/Theta=1/Phi=2). In hp mode a radial h-move is a gated
    // layout move, not the default response forever: after an h solve the next
    // cycle measures the radial-demand drop in the regions actually shelled.
    // Another shell is allowed only if that measured layout gain is large enough
    // and angular demand is not already the dominant remaining error. Theta and
    // phi get independent per-domain p-refinement.
    //   mode="hp" (default): radial h-first only while the h gate stays open.
    //     Any flagged region (NS1 / NS2 / exterior, by refine_axis[d][0]) whose
    //     shell count is below max_shells may take an h-move — one more shell in
    //     each such region, regrid uniformly at the base resolution. Once the h
    //     gate closes or no radially flagged region can add a shell, the cycle
    //     falls to bounded per-domain per-direction p-refinement (+2 points on
    //     each accepted domain+axis, max_nr ceiling, optional DOF-growth trust
    //     limit); with everything capped or outside the trust limit it stops.
    //   mode="p": per-domain per-direction p-refinement only, +2 points, all
    //     three axes at once. max_nr is honoured as a hard ceiling here too.
    // The two adapted domains of each star share the deformable stellar surface,
    // so an angular lock forces both to the pair-max nt and np after refinement
    // (their radial nr may still differ). BIN_RES tracks the maximum radial count
    // so the resolution ladder, filenames and banners stay meaningful on a
    // mixed-resolution space. The regrid callback receives the full per-domain
    // {nr,nt,np} target as a std::vector<Dim_array> (absolute domain indexing);
    // an empty vector requests a uniform regrid at BIN_RES.
    template <typename config_t, typename space_t, typename EvaluateTails, typename Regrid>
    bool amr_refine_if_needed(config_t& bconfig,
                              const std::string& outputdir,
                              int rank,
                              int cycle,
                              EvaluateTails&& evaluate_tails,
                              Regrid&& regrid,
                              RefinementPolicyState* policy_state = nullptr)
    {
        const std::string mode = bconfig.template amr_setting_as<std::string>(AMR_MODE);
        if (mode != "p" && mode != "hp") {
            KADATH_THROW(
                std::string{"adaptive_mesh_refinement.mode='"} + mode +
                "' is not supported; BNS AMR dispatch supports mode='p' or mode='hp'");
        }
        const bool allow_h = (mode == "hp");
        if (allow_h) {
            // resolved_rbisph_fill owns the absent-key-means-default semantics;
            // only the explicit rbisph_fill = 0 legacy opt-out conflicts with hp.
            if (!(bco_utils::resolved_rbisph_fill(bconfig) > 0.))
                KADATH_THROW(
                    "adaptive_mesh_refinement.mode='hp' conflicts with the explicit "
                    "[binary] rbisph_fill = 0 opt-out; hp mode requires a blended "
                    "fixed r_bisph target");
        }

        const bool initialized = mpi_is_initialized();
        apply_outputdir(bconfig, outputdir, rank, initialized);

        int refine_flag = 0;
        int target_resolution = static_cast<int>(bconfig(BIN_RES));
        std::string output_filename;

        if (rank == 0 || !initialized) {
            const TailDecision decision = std::forward<EvaluateTails>(evaluate_tails)(bconfig);

            const int ndom = static_cast<int>(decision.current_res.size());
            if (decision.first_exterior_domain < 0 || decision.first_exterior_domain >= ndom)
                KADATH_THROW("AMR refine needs first_exterior_domain set in the TailDecision");
            if (decision.refine_axis.size() != decision.current_res.size())
                KADATH_THROW("AMR refine needs refine_axis sized like current_res in the TailDecision");
            if (!decision.p_refine_allowed.empty() &&
                decision.p_refine_allowed.size() != decision.current_res.size())
                KADATH_THROW(
                    "AMR refine needs p_refine_allowed sized like current_res in the TailDecision");
            if (decision.domain_eligible.size() != decision.current_res.size())
                KADATH_THROW("AMR refine needs domain_eligible sized like current_res in the TailDecision");

            const auto label = [&](int d) -> std::string {
                return d >= 0 && d < static_cast<int>(decision.domain_label.size())
                           ? decision.domain_label[static_cast<std::size_t>(d)]
                           : std::string{};
            };

            // Compact formatters for the per-cycle AMR summary lines: e1 = one
            // significant-digit scientific (aligned ratios), fx = fixed precision.
            const auto e1 = [](double x) {
                std::ostringstream o;
                o << std::scientific << std::setprecision(1) << x;
                return o.str();
            };
            const auto fx = [](double x, int prec) {
                std::ostringstream o;
                o << std::fixed << std::setprecision(prec) << x;
                return o.str();
            };
            const auto axis_name = [](int a) -> const char* {
                return a == 0 ? "radial" : a == 1 ? "theta" : a == 2 ? "phi" : "?";
            };

            // Per-direction summary — printed every cycle. Counts how many
            // eligible domains each axis (radial / theta / phi) flagged this cycle
            // so a glance shows which direction is driving the refinement. A
            // structurally blocked flag is a measured tail that is not an
            // actionable per-domain p move in this space.
            const FlaggedDirectionCounts flag_counts = count_flagged_directions(decision, ndom);
            const std::array<int, 3>& flagged = flag_counts.flagged;
            const std::array<int, 3>& blocked_flagged = flag_counts.blocked;
            const double angular_to_radial =
                decision.radial_demand > 0.0
                    ? decision.angular_demand / decision.radial_demand
                    : 0.0;
            std::cout << "AMR c" << cycle << " worst : "
                      << (decision.maximum.field.empty() ? "-" : decision.maximum.field)
                      << "  dom" << decision.maximum.domain
                      << " (" << label(decision.maximum.domain) << ")"
                      << "  axis" << decision.maximum.axis << "/"
                      << axis_name(decision.maximum.axis)
                      << "   l2=" << e1(decision.maximum.l2_ratio)
                      << "  linf=" << e1(decision.maximum.linf_ratio) << "\n";
            std::cout << "AMR c" << cycle << " demand: radial=" << flagged[0]
                      << " theta=" << flagged[1] << " phi=" << flagged[2];
            if (blocked_flagged[0] + blocked_flagged[1] + blocked_flagged[2] > 0)
                std::cout << " (blocked r=" << blocked_flagged[0]
                          << " t=" << blocked_flagged[1]
                          << " p=" << blocked_flagged[2] << ")";
            std::cout << "   R=" << e1(decision.radial_demand)
                      << " A=" << e1(decision.angular_demand)
                      << "  A/R=" << fx(angular_to_radial, 1) << std::endl;

            // Bispheric block status (locked unit, separate from the per-domain
            // count above): the five domains refine together. Their axes are bipolar
            // (chi/eta/phi, slot order differs per variant), not spherical, so they
            // are reported by Dim_array index. Only axis2 (phi, azimuthal) is
            // refinable — axes 0/1 (chi/eta) map to different physical directions
            // across the variants and would unbalance the conforming internal seams,
            // so a flag there is informational (under-resolution we cannot act on).
            if (decision.bispheric_enabled) {
                std::cout << "AMR c" << cycle << " bisph : axis0="
                          << static_cast<int>(decision.bispheric_refine_axis[0])
                          << " axis1=" << static_cast<int>(decision.bispheric_refine_axis[1])
                          << " phi=" << static_cast<int>(decision.bispheric_refine_axis[2])
                          << "   worst "
                          << (decision.bispheric_maximum.field.empty()
                                  ? "-"
                                  : decision.bispheric_maximum.field)
                          << " dom" << decision.bispheric_maximum.domain
                          << "  l2=" << e1(decision.bispheric_maximum.l2_ratio)
                          << " linf=" << e1(decision.bispheric_maximum.linf_ratio) << std::endl;
            }

            // Radial h-first (hp mode only): a flagged region (by the radial axis
            // flag) below max_shells takes one more shell, regrid uniformly at the
            // base resolution. A failed post-h layout-gain check closes h for the
            // rest of the chain; angular dominance only skips h for this cycle.
            bool h_done = false;
            bool h_closed_this_cycle = false;
            bool post_h_layout_check_this_cycle = false;
            if (allow_h) {
                if (decision.second_star_first_domain < 0)
                    KADATH_THROW(
                        "adaptive_mesh_refinement.mode='hp' needs "
                        "second_star_first_domain set in the TailDecision");
                const int max_shells = bconfig.template amr_setting_as<int>(AMR_MAX_SHELLS);
                if (max_shells < 0)
                    KADATH_THROW("adaptive_mesh_refinement.max_shells must be non-negative");

                bool flagged_star1 = false, flagged_star2 = false, flagged_exterior = false;
                for (int d = 0; d < ndom; ++d) {
                    if (!decision.refine_axis[static_cast<std::size_t>(d)][0])
                        continue;
                    if (d < decision.second_star_first_domain)
                        flagged_star1 = true;
                    else if (d < decision.first_exterior_domain)
                        flagged_star2 = true;
                    else
                        flagged_exterior = true;
                }
                const auto shell_count = [&](auto key, auto... node) {
                    return bconfig.has_value(key, node...)
                               ? static_cast<int>(bconfig(key, node...))
                               : 0;
                };
                const bool h_star1 = flagged_star1 && shell_count(NSHELLS, BCO1) < max_shells;
                const bool h_star2 = flagged_star2 && shell_count(NSHELLS, BCO2) < max_shells;
                const bool h_exterior = flagged_exterior && shell_count(OUTER_SHELLS) < max_shells;
                const std::array<char, 3> h_regions{{
                    static_cast<char>(h_star1 ? 1 : 0),
                    static_cast<char>(h_star2 ? 1 : 0),
                    static_cast<char>(h_exterior ? 1 : 0),
                }};
                const bool h_candidate = h_star1 || h_star2 || h_exterior;
                bool h_policy_allows = h_candidate;
                std::string h_policy_reason;
                const auto angular_ratio = [&]() {
                    return decision.radial_demand > 0.0
                               ? decision.angular_demand / decision.radial_demand
                               : std::numeric_limits<double>::infinity();
                };
                const auto angular_dominates_for = [&](double threshold) {
                    if (threshold <= 0.0)
                        return false;
                    return decision.radial_demand > 0.0
                               ? decision.angular_demand >= threshold * decision.radial_demand
                               : decision.angular_demand > 0.0;
                };
                if (policy_state != nullptr && policy_state->has_h_baseline) {
                    post_h_layout_check_this_cycle = true;
                    const double h_gain_min =
                        bconfig.template amr_setting_as<double>(AMR_H_GAIN_MIN);
                    if (!(h_gain_min > 1.0))
                        KADATH_THROW("adaptive_mesh_refinement.h_gain_min must be greater than 1");

                    struct HGainCheck
                    {
                        double weakest_gain = std::numeric_limits<double>::infinity();
                        bool improved = true;
                        std::string details{};
                    };
                    const auto scalar_gain_check = [&]() {
                        const double previous_R = policy_state->previous_radial_demand;
                        const double current_R = decision.radial_demand;
                        const double gain = current_R > 0.0
                                                ? previous_R / current_R
                                                : std::numeric_limits<double>::infinity();
                        const bool improved =
                            previous_R <= 0.0 || current_R <= previous_R / h_gain_min;
                        std::ostringstream details;
                        details << "global=" << gain;
                        return HGainCheck{gain, improved, details.str()};
                    };
                    const auto region_gain_check = [&]() {
                        const bool has_region_baseline =
                            std::any_of(policy_state->previous_h_regions.begin(),
                                        policy_state->previous_h_regions.end(),
                                        [](char active) { return active != 0; });
                        if (!has_region_baseline)
                            return scalar_gain_check();

                        bool compared_region = false;
                        bool wrote_detail = false;
                        bool improved = true;
                        double weakest_gain = std::numeric_limits<double>::infinity();
                        std::ostringstream details;
                        const std::array<const char*, 3> region_label{{
                            "compact1",
                            "compact2",
                            "exterior",
                        }};
                        for (std::size_t r = 0; r < policy_state->previous_h_regions.size(); ++r) {
                            if (!policy_state->previous_h_regions[r])
                                continue;
                            const double previous_R =
                                policy_state->previous_radial_region_demand[r];
                            if (!(previous_R > 0.0))
                                continue;
                            compared_region = true;
                            const double current_R = decision.radial_region_demand[r];
                            const double gain =
                                current_R > 0.0 ? previous_R / current_R
                                                : std::numeric_limits<double>::infinity();
                            weakest_gain = std::min(weakest_gain, gain);
                            if (wrote_detail)
                                details << " ";
                            details << region_label[r] << "=" << gain;
                            wrote_detail = true;
                            if (current_R > previous_R / h_gain_min)
                                improved = false;
                        }
                        if (!compared_region)
                            return scalar_gain_check();
                        return HGainCheck{weakest_gain, improved, details.str()};
                    };
                    const HGainCheck h_gain = region_gain_check();
                    std::cout << "AMR h-layout probe cycle " << cycle
                              << ": moved-region radial demand gain=" << h_gain.weakest_gain
                              << " [" << h_gain.details << "]"
                              << " required>=" << h_gain_min << " | R="
                              << decision.radial_demand << " A=" << decision.angular_demand
                              << " A/R=" << angular_ratio() << " | radial flags "
                              << policy_state->previous_radial_flagged_domains << "->"
                              << flagged[0] << std::endl;

                    if (!h_gain.improved) {
                        h_policy_allows = false;
                        h_closed_this_cycle = true;
                        std::ostringstream reason;
                        reason << "moved-region radial demand gain=" << h_gain.weakest_gain
                               << " [" << h_gain.details << "]"
                               << " (<" << h_gain_min << ")";
                        h_policy_reason = reason.str();
                        policy_state->close_h_gate();
                    } else {
                        policy_state->clear_h_baseline();
                    }
                }
                if (h_candidate && policy_state != nullptr && policy_state->h_closed &&
                    !h_closed_this_cycle) {
                    h_policy_allows = false;
                    h_policy_reason = "h gate already closed after a failed layout trial";
                }
                if (h_candidate && h_policy_allows) {
                    const double h_angular_dominance =
                        bconfig.template amr_setting_as<double>(AMR_H_ANGULAR_DOMINANCE);
                    if (!(h_angular_dominance >= 0.0))
                        KADATH_THROW(
                            "adaptive_mesh_refinement.h_angular_dominance must be non-negative");
                    if (angular_dominates_for(h_angular_dominance)) {
                        h_policy_allows = false;
                        std::ostringstream reason;
                        reason << "angular demand dominates (A/R=" << angular_ratio() << ")";
                        h_policy_reason = reason.str();
                    }
                }

                if (h_candidate && h_policy_allows) {
                    auto bump = [&](auto key, auto... node) {
                        const int current = bconfig.has_value(key, node...)
                                                ? static_cast<int>(bconfig(key, node...))
                                                : 0;
                        bconfig.set(key, node...) = current + 1;
                        return current + 1;
                    };
                    const int base =
                        decision.current_res[static_cast<std::size_t>(decision.first_exterior_domain)][0];
                    Kadath::validate_resolution(base);
                    target_resolution = base;
                    bconfig.set(BIN_RES) = base;
                    output_filename = amr_regrid_filename(cycle, base) + "_h";

                    std::cout << "AMR c" << cycle << " action: h ADD shells";
                    if (h_star1)
                        std::cout << "  NS1 nshells->" << bump(NSHELLS, BCO1);
                    if (h_star2)
                        std::cout << "  NS2 nshells->" << bump(NSHELLS, BCO2);
                    if (h_exterior)
                        std::cout << "  OUTER_SHELLS->" << bump(OUTER_SHELLS);
                    std::cout << "   (regrid uniform nr=" << base
                              << ", angular re-eval next cycle)" << std::endl;

                    std::forward<Regrid>(regrid)(bconfig, output_filename, std::vector<Kadath::Dim_array>{});
                    if (policy_state != nullptr)
                        policy_state->record_h_baseline(
                            decision.radial_demand, flagged[0], decision.maximum,
                            decision.radial_region_demand, h_regions);
                    refine_flag = 1;
                    h_done = true;
                } else if (h_candidate) {
                    std::cout << "AMR c" << cycle << " action: "
                              << (h_closed_this_cycle ? "h gate closed" : "h SKIP");
                    if (!h_policy_reason.empty())
                        std::cout << " (" << h_policy_reason << ")";
                    std::cout << " -> p-refine" << std::endl;
                }
            }

            // Per-domain per-direction p-refinement. mode='p' takes this path
            // unconditionally; hp falls here once no radially flagged region can
            // add a shell. +2 points on each flagged domain+axis (the +2 step
            // keeps the count odd, as the spectral bases require); in hp the +2
            // also honours max_nr — a domain+axis whose target would exceed it
            // drops out, no partial clamp (which could land on an even count).
            if (!h_done) {
                std::vector<std::array<int, 3>> target = decision.current_res;
                const int nr_cap = bconfig.template amr_setting_as<int>(AMR_MAX_NR);
                if (nr_cap <= 0)
                    KADATH_THROW("adaptive_mesh_refinement.max_nr must be positive");
                const double p_dof_growth_limit =
                    bconfig.template amr_setting_as<double>(AMR_P_DOF_GROWTH_LIMIT);
                if (!(p_dof_growth_limit >= 0.0))
                    KADATH_THROW(
                        "adaptive_mesh_refinement.p_dof_growth_limit must be non-negative");
                const bool staged_post_h_fallback =
                    allow_h && policy_state != nullptr &&
                    (post_h_layout_check_this_cycle || h_closed_this_cycle);
                const double p_fallback_dof_growth_limit =
                    bconfig.template amr_setting_as<double>(AMR_P_FALLBACK_DOF_GROWTH_LIMIT);
                const int p_fallback_max_candidates =
                    bconfig.template amr_setting_as<int>(AMR_P_FALLBACK_MAX_CANDIDATES);
                if (!(p_fallback_dof_growth_limit >= 0.0))
                    KADATH_THROW(
                        "adaptive_mesh_refinement.p_fallback_dof_growth_limit must be "
                        "non-negative");
                if (p_fallback_max_candidates < 0)
                    KADATH_THROW(
                        "adaptive_mesh_refinement.p_fallback_max_candidates must be "
                        "non-negative");
                double active_p_dof_growth_limit = p_dof_growth_limit;
                if (staged_post_h_fallback && p_fallback_dof_growth_limit > 0.0)
                    active_p_dof_growth_limit =
                        active_p_dof_growth_limit > 0.0
                            ? std::min(active_p_dof_growth_limit, p_fallback_dof_growth_limit)
                            : p_fallback_dof_growth_limit;
                const int active_p_candidate_limit =
                    staged_post_h_fallback ? p_fallback_max_candidates : 0;
                struct PRefinementCandidate
                {
                    int domain = -1;
                    int axis = -1;
                    int target_points = 0;
                    double demand = 0.0;
                    bool bispheric_block = false;
                };
                std::vector<PRefinementCandidate> p_candidates;
                bool saw_capped_p_candidate = false;
                int structurally_blocked_p_candidates = 0;
                const bool dominant_tail_blocked =
                    decision.maximum.domain >= 0 && decision.maximum.axis >= 0 &&
                    decision.maximum.domain < static_cast<int>(decision.blocked_refine_axis.size()) &&
                    decision.blocked_refine_axis[static_cast<std::size_t>(decision.maximum.domain)]
                                                [static_cast<std::size_t>(decision.maximum.axis)] != 0;
                const bool staged_dominant_tail_blocked =
                    staged_post_h_fallback && dominant_tail_blocked;
                for (int d = 0; d < ndom; ++d) {
                    if (!decision.domain_eligible[static_cast<std::size_t>(d)])
                        continue;
                    for (int a = 0; a < 3; ++a) {
                        if (d < static_cast<int>(decision.blocked_refine_axis.size()) &&
                            decision.blocked_refine_axis[static_cast<std::size_t>(d)]
                                                        [static_cast<std::size_t>(a)]) {
                            ++structurally_blocked_p_candidates;
                        }
                        if (!decision.refine_axis[static_cast<std::size_t>(d)]
                                                 [static_cast<std::size_t>(a)])
                            continue;
                        const int candidate = decision.current_res[static_cast<std::size_t>(d)]
                                                                   [static_cast<std::size_t>(a)] + 2;
                        if (candidate > nr_cap) {
                            saw_capped_p_candidate = true;
                            continue;  // at the ceiling: drop this domain+axis, no clamp
                        }
                        double demand = 0.0;
                        if (d < static_cast<int>(decision.refine_demand.size()))
                            demand = decision.refine_demand[static_cast<std::size_t>(d)]
                                                           [static_cast<std::size_t>(a)];
                        p_candidates.push_back({d, a, candidate, demand, false});
                    }
                }

                // Star-core angular lock: the nucleus and its adapted pair share
                // ONE nt/np. Two reasons the angular jump cannot live inside the
                // star core: (1) the two adapted domains share the deformable
                // stellar surface, so its angular representation must be single-
                // valued; (2) the nucleus uses a parity-restricted angular basis
                // whose import (non-conforming) matching is NOT square against a
                // different-nt neighbour -- a refined-adapted / unrefined-nucleus
                // seam leaves the system under-determined by (Δnt·np). Locking the
                // core (nucleus = adapted-outer = adapted-inner, the domain triple
                // {adapted-1, adapted, adapted+1}) keeps every intra-core seam
                // conforming (proven square). nr stays per-domain; post-adapted
                // shells and the exterior keep per-domain angular (regular spheric
                // domains whose import matching is square across an nt jump).
                // Bispheric block (locked unit) — PHI AXIS ONLY. The five share one
                // Dim_array and bump together by +2 so their mutual (conforming)
                // seams stay matched and their star-side / outer (import) seams
                // absorb the order jump to the spherical neighbours. Only axis 2
                // (phi, azimuthal) is refinable: it is the SAME physical coordinate
                // in all five bipolar variants (chi_first / rect / eta_first), so a
                // uniform +2 keeps every conforming internal seam square (verified:
                // unknowns==conditions). Axes 0/1 are the bipolar (chi, eta)
                // coordinates whose Dim_array slot maps to DIFFERENT physical
                // directions across the variants; bumping them uniformly unbalances
                // the conforming internal seams (measured: +100 / -37 conditions on
                // a 12-domain probe) and is therefore not refinable here. The block
                // has no h-fallback (no shell fits inside the chimera), so this is
                // its only refinement path; the +2 honours the same hp max_nr ceiling
                // (a capped phi drops, keeping the five identical and the count odd).
                if (decision.bispheric_enabled && decision.bispheric_first >= 0 &&
                    decision.bispheric_first + 5 <= ndom &&
                    decision.bispheric_refine_axis[2]) {
                    const std::size_t lead = static_cast<std::size_t>(decision.bispheric_first);
                    const int candidate = decision.current_res[lead][2] + 2;
                    const bool at_ceiling = candidate > nr_cap;
                    if (at_ceiling)  // else drop the bump, keep the five locked at base
                        saw_capped_p_candidate = true;
                    else
                        p_candidates.push_back(
                            {decision.bispheric_first, 2, candidate,
                             decision.bispheric_refine_demand[2], true});
                }

                std::sort(p_candidates.begin(), p_candidates.end(),
                          [](const auto& lhs, const auto& rhs) {
                              if (lhs.demand != rhs.demand)
                                  return lhs.demand > rhs.demand;
                              if (lhs.bispheric_block != rhs.bispheric_block)
                                  return lhs.bispheric_block > rhs.bispheric_block;
                              if (lhs.domain != rhs.domain)
                                  return lhs.domain < rhs.domain;
                              return lhs.axis < rhs.axis;
                          });

                const double base_dof = std::max(1.0, dof_proxy(decision.current_res));
                int accepted_p_candidates = 0;
                int trust_skipped_p_candidates = 0;
                int batch_skipped_p_candidates = 0;
                for (const auto& candidate : p_candidates) {
                    if (staged_dominant_tail_blocked)
                        break;
                    if (active_p_candidate_limit > 0 &&
                        accepted_p_candidates >= active_p_candidate_limit) {
                        ++batch_skipped_p_candidates;
                        continue;
                    }
                    std::vector<std::array<int, 3>> trial = target;
                    if (candidate.bispheric_block) {
                        apply_bispheric_phi_block(trial, decision.bispheric_first,
                                                  candidate.target_points);
                    } else {
                        trial[static_cast<std::size_t>(candidate.domain)]
                             [static_cast<std::size_t>(candidate.axis)] = candidate.target_points;
                    }
                    lock_star_cores(trial, decision.adapted1, decision.adapted2);
                    const double growth = dof_proxy(trial) / base_dof;
                    if (active_p_dof_growth_limit > 0.0 && growth > active_p_dof_growth_limit) {
                        ++trust_skipped_p_candidates;
                        continue;
                    }
                    target = std::move(trial);
                    ++accepted_p_candidates;
                }

                if (target == decision.current_res) {
                    if (policy_state != nullptr)
                        policy_state->reset_h_state();
                    if (decision.refine && staged_dominant_tail_blocked)
                        std::cout << "AMR cycle " << cycle
                                  << ": dominant tail is structurally unsupported by "
                                     "per-domain p-refinement in this space (field="
                                  << (decision.maximum.field.empty() ? "-" : decision.maximum.field)
                                  << " domain=" << decision.maximum.domain << " "
                                  << label(decision.maximum.domain)
                                  << " axis=" << decision.maximum.axis
                                  << "); stopping" << std::endl;
                    else if (decision.refine && trust_skipped_p_candidates > 0)
                        std::cout << "AMR cycle " << cycle
                                  << ": p-refinement projected DOF growth exceeds "
                                     "active p DOF growth limit="
                                  << active_p_dof_growth_limit << "; stopping" << std::endl;
                    else if (decision.refine && saw_capped_p_candidate)
                        std::cout << "AMR cycle " << cycle
                                  << ": tails above thresholds but every flagged domain/axis is at "
                                     "max_nr; stopping"
                                  << std::endl;
                    else if (decision.refine && structurally_blocked_p_candidates > 0)
                        std::cout << "AMR cycle " << cycle
                                  << ": tails above thresholds but structurally supported p "
                                     "candidates are exhausted; blocked candidates="
                                  << structurally_blocked_p_candidates << "; stopping" << std::endl;
                    else if (decision.refine)
                        std::cout << "AMR cycle " << cycle
                                  << ": tails above thresholds but no refinable p candidate remains; "
                                     "stopping"
                                  << std::endl;
                    else
                        std::cout << "AMR cycle " << cycle
                                  << ": spectral tails below thresholds; stopping" << std::endl;

                    // Canonical machine-detectable marker: refinement was wanted
                    // this cycle but the terminal grid was reached (every flagged
                    // axis at max_nr, structurally blocked, or vetoed by the DOF-
                    // trust limit), so AMR stops while a tail is still above
                    // threshold. Downstream tooling greps this one stable token
                    // instead of the verbose, drift-prone human reasons above.
                    if (decision.refine)
                        std::cout << "AMR_RESIDUAL_RESOLUTION cycle " << cycle
                                  << " field="
                                  << (decision.maximum.field.empty() ? "-" : decision.maximum.field)
                                  << " domain=" << decision.maximum.domain
                                  << " axis=" << decision.maximum.axis
                                  << " l2=" << decision.maximum.l2_ratio
                                  << " linf=" << decision.maximum.linf_ratio << std::endl;
                } else {
                    if (policy_state != nullptr)
                        // Probe 11: a committed p-refinement closes the h gate, not
                        // merely clears the baseline. A later h-cycle regrids
                        // uniformly at base and would erase the per-domain p gains
                        // just installed; close_h_gate() blocks that reopening. An
                        // explicit layout reset re-arms h via reset_h_state().
                        policy_state->close_h_gate();
                    // Radial ladder: BIN_RES tracks the largest radial count so
                    // filenames and the resolution sequence stay meaningful on a
                    // mixed-resolution space.
                    int max_nr = 0;
                    for (int d = 0; d < ndom; ++d)
                        max_nr = std::max(max_nr, target[static_cast<std::size_t>(d)][0]);
                    Kadath::validate_resolution(max_nr);
                    target_resolution = max_nr;
                    output_filename = amr_regrid_filename(cycle, target_resolution);

                    const int total_p_candidates =
                        accepted_p_candidates + trust_skipped_p_candidates +
                        batch_skipped_p_candidates + structurally_blocked_p_candidates;
                    std::cout << "AMR c" << cycle << " p-ref : " << accepted_p_candidates
                              << "/" << total_p_candidates << " candidates applied"
                              << " (rejected trust " << trust_skipped_p_candidates
                              << ", cap " << batch_skipped_p_candidates
                              << ", struct " << structurally_blocked_p_candidates << ")"
                              << "   dof x" << fx(dof_proxy(target) / base_dof, 2)
                              << " (limit x" << fx(active_p_dof_growth_limit, 1) << ")";
                    if (active_p_candidate_limit > 0)
                        std::cout << "   cap=" << active_p_candidate_limit;
                    if (staged_post_h_fallback)
                        std::cout << "   (post-h stage)";
                    std::cout << "\n"
                              << "AMR c" << cycle
                              << " changed (nr,nt,np):\n";
                    for (int d = 0; d < ndom; ++d) {
                        const auto& before = decision.current_res[static_cast<std::size_t>(d)];
                        const auto& after = target[static_cast<std::size_t>(d)];
                        if (before == after)
                            continue;
                        std::cout << "    domain " << std::setw(2) << d << "  "
                                  << std::setw(22) << std::left << label(d) << std::right << "  ("
                                  << before[0] << "," << before[1] << "," << before[2] << ") -> ("
                                  << after[0] << "," << after[1] << "," << after[2] << ")\n";
                    }
                    std::cout << "  -> max radial=" << target_resolution << std::endl;

                    // Convert the {nr,nt,np} target to the space's
                    // std::vector<Dim_array> regrid contract.
                    const std::vector<Kadath::Dim_array> res_per_domain =
                        build_res_per_domain(target);

                    bconfig.set(BIN_RES) = target_resolution;
                    std::forward<Regrid>(regrid)(bconfig, output_filename, res_per_domain);
                    refine_flag = 1;
                }
            }
        }

        if (initialized) {
            MPI_Bcast(&refine_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&target_resolution, 1, MPI_INT, 0, MPI_COMM_WORLD);
            broadcast_string_from_root(output_filename);
        }

        if (refine_flag == 0)
            return false;

        if (rank != 0 && initialized) {
            bconfig.set(BIN_RES) = target_resolution;
            bconfig.set_filename(output_filename);
            bconfig.open_config();
        }

        return true;
    }


} // namespace KadathApps::bns_amr
