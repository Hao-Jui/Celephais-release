#pragma once


#include "Apps/AMR/binary_amr_core.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "For_Kadath/Config/config_binary.hpp"

#include <array>
#include <string>

namespace KadathApps::bns_amr
{
    // Human-readable role of each domain in the binary-NS layout, used by the
    // refine banner ("which domain went from what to what").
    template <typename space_t>
    std::string bns_domain_role(const space_t& space, int d, int ndom)
    {
        auto star_role = [&](int nucleus, int adapted, std::string star) -> std::string {
            if (d == nucleus)
                return star + " nucleus";
            if (d == adapted)
                return star + " adapted-outer";
            if (d == adapted + 1)
                return star + " adapted-inner";
            return star + " shell " + std::to_string(d - adapted - 2);
        };
        if (d < space.NS2)
            return star_role(space.NS1, space.ADAPTED1, "NS1");
        if (d < space.OUTER)
            return star_role(space.NS2, space.ADAPTED2, "NS2");
        if (d < space.OUTER + 5) {
            static const char* bispheric[5] = {"bispheric chi-first(1)", "bispheric rect(1)",
                                               "bispheric eta-first", "bispheric rect(2)",
                                               "bispheric chi-first(2)"};
            return bispheric[d - space.OUTER];
        }
        if (d == ndom - 1)
            return "compactified";
        return "exterior shell " + std::to_string(d - space.OUTER - 5);
    }

    // Seed a TailDecision from the evaluated space: record each domain's
    // current per-direction point counts {nr, nt, np} and mark which domains may
    // refine.
    template <typename config_t, typename space_t>
    TailDecision make_tail_decision(const config_t& bconfig, const space_t& space)
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
            decision.domain_label[static_cast<std::size_t>(d)] = bns_domain_role(space, d, ndom);
        }
        decision.refine_axis.assign(static_cast<std::size_t>(ndom), {0, 0, 0});
        decision.refine_demand.assign(static_cast<std::size_t>(ndom), {0.0, 0.0, 0.0});
        decision.domain_eligible.assign(static_cast<std::size_t>(ndom), 1);
        decision.second_star_first_domain = space.NS2;
        decision.first_exterior_domain = space.OUTER;
        // The two adapted domains of each star share the deformable stellar
        // surface, so their nt/np must stay identical after refinement (the
        // angular lock in the dispatch enforces it). Each pair is the two domains
        // right after the star's nucleus.
        decision.adapted1 = space.ADAPTED1;
        decision.adapted2 = space.ADAPTED2;
        // The five bispheric domains are masked from the PER-DOMAIN machinery
        // (their mutual seams keep them at one shared Dim_array, and their radial
        // axis must never feed the h-first region scan — there is no shell to add
        // inside the chimera). When refine_bispheric is on they still refine, but
        // as a LOCKED BLOCK handled separately (observe_axis aggregates them into
        // decision.bispheric_refine_axis; the dispatch bumps all five together).
        decision.bispheric_first = space.OUTER;
        decision.bispheric_enabled =
            bconfig.template amr_setting_as<bool>(AMR_REFINE_BISPHERIC);
        for (int d = space.OUTER; d < std::min(ndom, space.OUTER + 5); ++d)
            decision.domain_eligible[static_cast<std::size_t>(d)] = 0;
        return decision;
    }

    template <typename config_t, typename space_t>
    TailDecision evaluate_bns_field_tails(config_t& bconfig, bool read_scalar_slot)
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

        TailDecision decision = make_tail_decision(bconfig, space);
        decision.l2_threshold = l2_threshold;
        decision.linf_threshold = linf_threshold;

        observe_binary_common_fields(decision, options, l2_threshold, linf_threshold, conf, lapse, shift, logh,
                                     phi);
        if (read_scalar_slot) {
            Kadath::Scalar scalar_slot(space, source);
            for (int axis = 0; axis < 3; ++axis) {
                Kadath::bns_hp::SpectralTailOptions slot_options = options;
                slot_options.axes = axis == 0   ? Kadath::bns_hp::TailAxisSelection::RadialOnly
                                    : axis == 1 ? Kadath::bns_hp::TailAxisSelection::ThetaOnly
                                                : Kadath::bns_hp::TailAxisSelection::PhiOnly;
                observe_axis(decision, "scalar",
                             Kadath::bns_hp::scalar_spectral_tail_ratios(scalar_slot, slot_options), axis,
                             l2_threshold, linf_threshold);
            }
        }
        return decision;
    }

    template <typename config_t, typename space_t>
    TailDecision evaluate_common_bns_field_tails(config_t& bconfig)
    {
        return evaluate_bns_field_tails<config_t, space_t>(bconfig, /*read_scalar_slot=*/false);
    }

    template <typename config_t, typename space_t>
    TailDecision evaluate_scalar_tensor_bns_field_tails(config_t& bconfig)
    {
        return evaluate_bns_field_tails<config_t, space_t>(bconfig, /*read_scalar_slot=*/true);
    }

    template <typename config_t, typename space_t, typename Regrid>
    bool amr_refine_common_fields_if_needed(config_t& bconfig,
                                            const std::string& outputdir,
                                            int rank,
                                            int cycle,
                                            Regrid&& regrid,
                                            RefinementPolicyState* policy_state = nullptr)
    {
        return amr_refine_if_needed<config_t, space_t>(
            bconfig, outputdir, rank, cycle,
            [](config_t& config) {
                return evaluate_common_bns_field_tails<config_t, space_t>(config);
            },
            std::forward<Regrid>(regrid), policy_state);
    }

    template <typename config_t, typename space_t, typename Regrid>
    bool amr_refine_scalar_tensor_fields_if_needed(config_t& bconfig,
                                                   const std::string& outputdir,
                                                   int rank,
                                                   int cycle,
                                                   Regrid&& regrid,
                                                   RefinementPolicyState* policy_state = nullptr)
    {
        return amr_refine_if_needed<config_t, space_t>(
            bconfig, outputdir, rank, cycle,
            [](config_t& config) {
                return evaluate_scalar_tensor_bns_field_tails<config_t, space_t>(config);
            },
            std::forward<Regrid>(regrid), policy_state);
    }

} // namespace KadathApps::bns_amr
