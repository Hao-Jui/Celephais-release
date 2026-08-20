#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "Apps/AMR/bns_hp_indicators.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Tensor/vector.hpp"

using namespace Kadath;

namespace {

Space_spheric make_hp_indicator_space()
{
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;

    Dim_array resolution(3);
    resolution.set(0) = 9;
    resolution.set(1) = 5;
    resolution.set(2) = 4;

    Dim_array bounds_dimension(1);
    bounds_dimension.set(0) = 2;
    Array<double> bounds(bounds_dimension);
    bounds.set(0) = 1.0;
    bounds.set(1) = 10.0;

    return Space_spheric(CHEB_TYPE, center, resolution, bounds);
}

Index first_coefficient(const Val_domain& value)
{
    return Index(value.get_domain()->get_nbr_coefs());
}

Index last_radial_coefficient(const Val_domain& value)
{
    const Dim_array dimensions = value.get_domain()->get_nbr_coefs();
    Index position(dimensions);
    position.set(0) = dimensions(0) - 1;
    return position;
}

bns_hp::SpectralTailOptions one_mode_tail()
{
    bns_hp::SpectralTailOptions options;
    options.tail_width = 1;
    return options;
}

} // namespace

TEST_CASE("BNS hp indicators report scalar spectral tail ratios per domain",
          "[bns-hp-indicators]")
{
    auto space = make_hp_indicator_space();

    Scalar field(space);
    field.std_base();
    field.annule_hard_coef();

    field.set_domain(0).set_coef(first_coefficient(field(0))) = 3.0;
    field.set_domain(0).set_coef(last_radial_coefficient(field(0))) = 4.0;

    const auto ratios = bns_hp::scalar_spectral_tail_ratios(field, one_mode_tail());

    REQUIRE(ratios.size() == static_cast<std::size_t>(space.get_nbr_domains()));
    CHECK(ratios[0].domain == 0);
    CHECK(ratios[0].l2_ratio == Catch::Approx(0.8));
    CHECK(ratios[0].linf_ratio == Catch::Approx(1.0));
    CHECK(ratios[0].tail_l2 == Catch::Approx(4.0));
    CHECK(ratios[0].total_l2 == Catch::Approx(5.0));
    CHECK(ratios[0].tail_modes > 0);
    CHECK(ratios[0].total_modes > ratios[0].tail_modes);

    CHECK(ratios[1].l2_ratio == Catch::Approx(0.0));
    CHECK(ratios[1].linf_ratio == Catch::Approx(0.0));
}

TEST_CASE("BNS hp indicators aggregate tensor component tails per domain",
          "[bns-hp-indicators]")
{
    auto space = make_hp_indicator_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Vector field(space, CON, basis);
    field.std_base();
    for (int component = 1; component <= 3; ++component)
        field.set(component).annule_hard_coef();

    field.set(1).set_domain(0).set_coef(last_radial_coefficient(field(1)(0))) = 3.0;
    field.set(2).set_domain(0).set_coef(first_coefficient(field(2)(0))) = 4.0;

    const auto aggregate = bns_hp::tensor_aggregate_spectral_tail_ratios(
        field, one_mode_tail());
    const auto first_component = bns_hp::tensor_component_spectral_tail_ratios(
        field, 0, one_mode_tail());

    CHECK(aggregate[0].l2_ratio == Catch::Approx(0.6));
    CHECK(aggregate[0].linf_ratio == Catch::Approx(0.75));
    CHECK(aggregate[0].tail_l2 == Catch::Approx(3.0));
    CHECK(aggregate[0].total_l2 == Catch::Approx(5.0));

    CHECK(first_component[0].l2_ratio == Catch::Approx(1.0));
    CHECK(first_component[0].linf_ratio == Catch::Approx(1.0));
}

TEST_CASE("BNS hp indicators flag a small tensor component the aggregate dilutes",
          "[bns-hp-indicators]")
{
    // Probe 12: a single under-resolved shift component must be judged on its own
    // norm, not drowned by a large well-resolved component in the aggregate norm.
    auto space = make_hp_indicator_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Vector field(space, CON, basis);
    field.std_base();
    for (int component = 1; component <= 3; ++component)
        field.set(component).annule_hard_coef();

    // Component index 0 (field(1)): large, fully resolved (bulk only, no tail).
    field.set(1).set_domain(0).set_coef(first_coefficient(field(1)(0))) = 10.0;
    // Component index 2 (field(3)): small, but its top mode carries half its
    // energy (under-resolved). The aggregate total norm is dominated by
    // component 0, so this tail is diluted ~70x.
    field.set(3).set_domain(0).set_coef(first_coefficient(field(3)(0))) = 0.1;
    field.set(3).set_domain(0).set_coef(last_radial_coefficient(field(3)(0))) = 0.1;

    const auto aggregate = bns_hp::tensor_aggregate_spectral_tail_ratios(field, one_mode_tail());
    const auto comp2 = bns_hp::tensor_component_spectral_tail_ratios(field, 2, one_mode_tail());

    // Aggregate: tail 0.1 / sqrt(10^2 + 0.1^2 + 0.1^2) ~ 0.01 (diluted).
    CHECK(aggregate[0].l2_ratio == Catch::Approx(0.1 / std::sqrt(100.02)));
    // Per-component: tail 0.1 / sqrt(0.1^2 + 0.1^2) ~ 0.707 (NOT diluted).
    CHECK(comp2[0].l2_ratio == Catch::Approx(0.1 / std::sqrt(0.02)));
    CHECK(comp2[0].l2_ratio > 50.0 * aggregate[0].l2_ratio);
}

TEST_CASE("BNS hp indicators isolate angular tails from radial tails",
          "[bns-hp-indicators]")
{
    // Pure axis-selection check on the tail predicate — independent of any
    // spectral basis. The refine decision watches RadialOnly; the polar
    // diagnostic watches AngularOnly, and the two must not see each other's
    // highest modes.
    Dim_array dimensions(3);
    dimensions.set(0) = 9;  // radial coefficients
    dimensions.set(1) = 5;  // theta coefficients
    dimensions.set(2) = 4;  // phi coefficients

    Index top_radial(dimensions);
    top_radial.set(0) = dimensions(0) - 1;
    Index top_theta(dimensions);
    top_theta.set(1) = dimensions(1) - 1;
    Index top_phi(dimensions);
    top_phi.set(2) = dimensions(2) - 1;
    Index bulk(dimensions);  // (0, 0, 0)

    using bns_hp::TailAxisSelection;
    const int width = 1;

    // RadialOnly: only the highest radial mode is a tail.
    CHECK(bns_hp::detail::is_tail_mode(top_radial, dimensions, width, TailAxisSelection::RadialOnly));
    CHECK_FALSE(bns_hp::detail::is_tail_mode(top_theta, dimensions, width, TailAxisSelection::RadialOnly));
    CHECK_FALSE(bns_hp::detail::is_tail_mode(top_phi, dimensions, width, TailAxisSelection::RadialOnly));

    // AngularOnly: the highest theta or phi mode is a tail; the radial one is not.
    CHECK(bns_hp::detail::is_tail_mode(top_theta, dimensions, width, TailAxisSelection::AngularOnly));
    CHECK(bns_hp::detail::is_tail_mode(top_phi, dimensions, width, TailAxisSelection::AngularOnly));
    CHECK_FALSE(bns_hp::detail::is_tail_mode(top_radial, dimensions, width, TailAxisSelection::AngularOnly));

    // ThetaOnly (axis 1): only the highest theta mode is a tail.
    CHECK(bns_hp::detail::is_tail_mode(top_theta, dimensions, width, TailAxisSelection::ThetaOnly));
    CHECK_FALSE(bns_hp::detail::is_tail_mode(top_radial, dimensions, width, TailAxisSelection::ThetaOnly));
    CHECK_FALSE(bns_hp::detail::is_tail_mode(top_phi, dimensions, width, TailAxisSelection::ThetaOnly));

    // PhiOnly (axis 2): only the highest phi mode is a tail. Per-direction AMR
    // refines theta and phi independently, so the two must not alias.
    CHECK(bns_hp::detail::is_tail_mode(top_phi, dimensions, width, TailAxisSelection::PhiOnly));
    CHECK_FALSE(bns_hp::detail::is_tail_mode(top_radial, dimensions, width, TailAxisSelection::PhiOnly));
    CHECK_FALSE(bns_hp::detail::is_tail_mode(top_theta, dimensions, width, TailAxisSelection::PhiOnly));

    // All: any highest mode counts; the bulk coefficient never does.
    CHECK(bns_hp::detail::is_tail_mode(top_radial, dimensions, width, TailAxisSelection::All));
    CHECK(bns_hp::detail::is_tail_mode(top_theta, dimensions, width, TailAxisSelection::All));
    CHECK_FALSE(bns_hp::detail::is_tail_mode(bulk, dimensions, width, TailAxisSelection::All));
}

TEST_CASE("BNS hp indicators fuse radial theta and phi coefficient scans exactly",
          "[bns-hp-indicators]")
{
    auto space = make_hp_indicator_space();
    Scalar field(space);
    field.std_base();
    field.annule_hard_coef();

    for (int domain = 0; domain < space.get_nbr_domains(); ++domain) {
        const Dim_array dimensions = field(domain).get_domain()->get_nbr_coefs();
        Index position(dimensions);
        do {
            const double value = 0.125 * (1 + domain) + 0.25 * position(0) -
                                 0.5 * position(1) + 0.75 * position(2);
            field.set_domain(domain).set_coef(position) = value;
        } while (position.inc());
    }

    auto options = one_mode_tail();
    const auto fused = bns_hp::scalar_spectral_axis_tail_ratios(field, options);
    const std::array<bns_hp::TailAxisSelection, 3> selections = {
        bns_hp::TailAxisSelection::RadialOnly,
        bns_hp::TailAxisSelection::ThetaOnly,
        bns_hp::TailAxisSelection::PhiOnly};

    for (std::size_t axis = 0; axis < selections.size(); ++axis) {
        options.axes = selections[axis];
        const auto separate = bns_hp::scalar_spectral_tail_ratios(field, options);
        REQUIRE(fused[axis].size() == separate.size());
        for (std::size_t domain = 0; domain < separate.size(); ++domain) {
            CHECK(fused[axis][domain].domain == separate[domain].domain);
            CHECK(fused[axis][domain].l2_ratio == separate[domain].l2_ratio);
            CHECK(fused[axis][domain].linf_ratio == separate[domain].linf_ratio);
            CHECK(fused[axis][domain].tail_l2 == separate[domain].tail_l2);
            CHECK(fused[axis][domain].total_l2 == separate[domain].total_l2);
            CHECK(fused[axis][domain].tail_linf == separate[domain].tail_linf);
            CHECK(fused[axis][domain].total_linf == separate[domain].total_linf);
            CHECK(fused[axis][domain].tail_modes == separate[domain].tail_modes);
            CHECK(fused[axis][domain].total_modes == separate[domain].total_modes);
        }
    }
}

TEST_CASE("BNS hp indicators fold the azimuthal tail and floor the aliased spike",
          "[bns-hp-indicators]")
{
    // Direct kernel check on the PhiOnly fold + envelope-floor estimator
    // (detail::PhiMarginal::finish) — the riskiest, most recently added
    // indicator and the one the PhiOnly ratio path routes through, bypassing
    // is_tail_mode entirely. energy_by_mode holds the per-phi-coefficient
    // summed squared amplitude (what PhiMarginal::add accumulates); cos/sin
    // pairs (2k, 2k+1) fold into one azimuthal-number amplitude
    // a_m = sqrt(e[2k] + e[2k+1]). tail_width=2 is the production default, so
    // the tail is the MIN over the top 3 folded bins. Odd phi-coef indices
    // (the structural sin zeros) stay 0.
    const int tail_width = 2;

    SECTION("an inflated top azimuthal mode is skipped by the envelope floor")
    {
        // Folded amplitudes per m: [10, 5, 1, 0.2, 4]. The TOP folded bin (4) is
        // an aliased spike; the bin just below it (0.2) is the true decayed
        // floor. The MIN over the top 3 folded bins must pick 0.2, NOT the spike
        // — this is the entire reason the fold+floor exists. The legacy literal
        // top-bin tail would have reported 4.0.
        bns_hp::detail::PhiMarginal marginal;
        marginal.energy_by_mode = {100.0, 0.0, 25.0, 0.0, 1.0, 0.0, 0.04, 0.0, 16.0, 0.0};
        for (double e : marginal.energy_by_mode)
            marginal.total_l2_sq += e;

        const auto ratio = marginal.finish(/*domain=*/0, tail_width, /*norm_floor=*/0.0);

        CHECK(ratio.tail_l2 == Catch::Approx(0.2));   // floored, not the 4.0 spike
        CHECK(ratio.tail_l2 < 4.0);                   // spike demonstrably skipped
        CHECK(ratio.total_linf == Catch::Approx(10.0));
        CHECK(ratio.linf_ratio == Catch::Approx(0.2 / 10.0));
        CHECK(ratio.l2_ratio == Catch::Approx(0.2 / std::sqrt(142.04)));
    }

    SECTION("a cleanly decaying azimuthal spectrum recovers the literal top-bin tail")
    {
        // Folded amplitudes per m: [10, 5, 2, 1, 0.5], monotone. With no spike
        // the top folded bin IS the min, so the floor must recover the usual
        // truncation-tail magnitude (the smallest, highest-m amplitude).
        bns_hp::detail::PhiMarginal marginal;
        marginal.energy_by_mode = {100.0, 0.0, 25.0, 0.0, 4.0, 0.0, 1.0, 0.0, 0.25, 0.0};
        for (double e : marginal.energy_by_mode)
            marginal.total_l2_sq += e;

        const auto ratio = marginal.finish(/*domain=*/0, tail_width, /*norm_floor=*/0.0);

        CHECK(ratio.tail_l2 == Catch::Approx(0.5));   // == amp.back(), clean decay
        CHECK(ratio.total_linf == Catch::Approx(10.0));
    }
}
