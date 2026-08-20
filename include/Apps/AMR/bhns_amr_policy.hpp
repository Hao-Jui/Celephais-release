#pragma once

// Black-hole–neutron-star AMR seeding. BHNS is asymmetric (one adapted NS, one
// homothetic BH), so it cannot reuse the BNS make_tail_decision (which assumes
// NS1/NS2/ADAPTED1/ADAPTED2). This header supplies the BHNS-specific decision
// seed and field-tail evaluator, then drives the SHARED, space-agnostic
// dispatch (amr_refine_if_needed) and the SHARED field observation
// (observe_binary_common_fields) from binary_amr_core.hpp. The five bispheric
// domains refine as a locked PHI block exactly as for BNS.

#include "Apps/AMR/binary_amr_core.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"

#include <algorithm>
#include <array>
#include <string>

namespace KadathApps::bns_amr
{
    // Human-readable role of each BHNS domain, used by the refine banner. Layout:
    // NS nucleus + adapted pair + NS shells, BH nucleus + homothetic pair + BH
    // shells, five bispheric domains, exterior shells, compactified.
    template <typename space_t>
    std::string bhns_domain_role(const space_t& space, int d, int ndom)
    {
        if (d == space.NS)
            return "NS nucleus";
        if (d == space.ADAPTEDNS)
            return "NS adapted-outer";
        if (d == space.ADAPTEDNS + 1)
            return "NS adapted-inner";
        if (d < space.BH)
            return "NS shell " + std::to_string(d - space.ADAPTEDNS - 2);
        if (d == space.BH)
            return "BH nucleus";
        if (d == space.ADAPTEDBH)
            return "BH homothetic-outer";
        if (d == space.ADAPTEDBH + 1)
            return "BH homothetic-inner";
        if (d < space.OUTER)
            return "BH shell " + std::to_string(d - space.ADAPTEDBH - 2);
        if (d < space.OUTER + 5) {
            static const char* bispheric[5] = {"bispheric chi-first(NS)", "bispheric rect(NS)",
                                               "bispheric eta-first", "bispheric rect(BH)",
                                               "bispheric chi-first(BH)"};
            return bispheric[d - space.OUTER];
        }
        if (d == ndom - 1)
            return "compactified";
        return "exterior shell " + std::to_string(d - space.OUTER - 5);
    }

    // Seed a TailDecision from a BHNS space. Same shape as make_tail_decision but
    // with the asymmetric anchors: the h-first region split is NS region [0, BH),
    // BH region [BH, OUTER), exterior [OUTER, ...); each compact object's core is
    // angular-locked (the NS adapted pair shares the deformable surface, the BH
    // homothetic pair shares the excision surface); the five bispheric domains are
    // masked from the per-domain machinery and refine as a locked PHI block.
    // Ordinary BHNS domains now allow per-domain p-refinement on all axes; the
    // shared dispatch still applies the compact-object angular locks below.
    template <typename config_t, typename space_t>
    TailDecision make_bhns_tail_decision(const config_t& bconfig, const space_t& space)
    {
        TailDecision decision;
        const int ndom = space.get_nbr_domains();
        decision.current_nr.resize(static_cast<std::size_t>(ndom));
        decision.current_res.resize(static_cast<std::size_t>(ndom));
        decision.domain_label.resize(static_cast<std::size_t>(ndom));
        for (int d = 0; d < ndom; ++d) {
            const Kadath::Dim_array points = space.get_domain(d)->get_nbr_points();
            decision.current_res[static_cast<std::size_t>(d)] = {points(0), points(1), points(2)};
            decision.current_nr[static_cast<std::size_t>(d)] = points(0);
            decision.domain_label[static_cast<std::size_t>(d)] = bhns_domain_role(space, d, ndom);
        }
        decision.refine_axis.assign(static_cast<std::size_t>(ndom), {0, 0, 0});
        decision.refine_demand.assign(static_cast<std::size_t>(ndom), {0.0, 0.0, 0.0});
        decision.domain_eligible.assign(static_cast<std::size_t>(ndom), 1);
        decision.p_refine_allowed.assign(static_cast<std::size_t>(ndom), {1, 1, 1});
        // h-first region split: the BH region begins at space.BH (the NS region is
        // everything before it), the exterior begins at space.OUTER. The h-move
        // bumps NSHELLS(BCO1) for the NS region, NSHELLS(BCO2) for the BH region,
        // OUTER_SHELLS for the exterior.
        decision.second_star_first_domain = space.BH;
        decision.first_exterior_domain = space.OUTER;
        // Angular-lock each compact object's core (nucleus + its inner/outer pair):
        // the NS adapted pair shares the one deformable stellar surface; the BH
        // homothetic pair shares the one excision surface. Both keep the
        // nucleus<->pair import seams square across an nt jump.
        decision.adapted1 = space.ADAPTEDNS;
        decision.adapted2 = space.ADAPTEDBH;
        // The five bispheric domains are masked from the per-domain machinery; when
        // refine_bispheric is on they refine as a locked PHI block (handled in the
        // shared dispatch). See binary_amr_core.hpp for the mechanism.
        decision.bispheric_first = space.OUTER;
        decision.bispheric_enabled =
            bconfig.template amr_setting_as<bool>(AMR_REFINE_BISPHERIC);
        for (int d = space.OUTER; d < std::min(ndom, space.OUTER + 5); ++d)
            decision.domain_eligible[static_cast<std::size_t>(d)] = 0;
        return decision;
    }

    // BHNS field-tail evaluator. The converged BHNS file stores the same five GR
    // fields as BNS in the same stream order (conf, lapse, shift, logh, phi), so
    // the shared observation helper applies verbatim.
    template <typename config_t, typename space_t>
    TailDecision evaluate_bhns_field_tails(config_t& bconfig)
    {
        const auto options = tail_options_from_config(bconfig);
        const double l2_threshold =
            bconfig.template amr_setting_as<double>(AMR_L2_TAIL_THRESHOLD);
        const double linf_threshold =
            bconfig.template amr_setting_as<double>(AMR_LINF_TAIL_THRESHOLD);

        Kadath::BeFileSource source(bconfig.space_filename());
        space_t space(source);
        Kadath::Scalar conf(space, source);
        Kadath::Scalar lapse(space, source);
        Kadath::Vector shift(space, source);
        Kadath::Scalar logh(space, source);
        Kadath::Scalar phi(space, source);

        TailDecision decision = make_bhns_tail_decision(bconfig, space);
        decision.l2_threshold = l2_threshold;
        decision.linf_threshold = linf_threshold;
        observe_binary_common_fields(decision, options, l2_threshold, linf_threshold, conf, lapse, shift, logh,
                                     phi);
        return decision;
    }

    // BHNS AMR entry point — mirrors amr_refine_common_fields_if_needed but seeds
    // the decision from the BHNS evaluator. The dispatch itself is space-agnostic.
    template <typename config_t, typename space_t, typename Regrid>
    bool amr_refine_bhns_fields_if_needed(config_t& bconfig,
                                          const std::string& outputdir,
                                          int rank,
                                          int cycle,
                                          Regrid&& regrid,
                                          RefinementPolicyState* policy_state = nullptr)
    {
        return amr_refine_if_needed<config_t, space_t>(
            bconfig, outputdir, rank, cycle,
            [](config_t& config) {
                return evaluate_bhns_field_tails<config_t, space_t>(config);
            },
            std::forward<Regrid>(regrid), policy_state);
    }
} // namespace KadathApps::bns_amr
