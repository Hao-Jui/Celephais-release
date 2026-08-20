#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "Apps/Bco_utils/ns_bounds.hpp"
#include "For_Kadath/Config/config_binary.hpp"

#include <vector>

using Catch::Approx;

namespace {

using BoundsConfig = kadath_config<BIN_INFO>;

BoundsConfig make_bounds_config(double fill)
{
    BoundsConfig config;
    config.initialize_binary({"ns", "ns"});
    config.set_defaults();
    config.set(DIST) = 35.;
    config.set(RBISPH_FILL) = fill;
    for (auto bco : {BCO1, BCO2}) {
        config.set(RIN, bco) = 3.0;
        config.set(RMID, bco) = 6.64;
        config.set(ROUT, bco) = 8.0;
        config.set(NSHELLS, bco) = 0;
    }
    return config;
}

} // namespace

TEST_CASE("blended_rbisph reproduces the historical setup formula at fill=1/3",
          "[ns-bounds][rbisph]")
{
    const double rmid = 6.64;
    const double dist = 35.;
    const double legacy_setup = (dist / 2. - rmid) / 3. + rmid;
    CHECK(bco_utils::blended_rbisph(rmid, dist, 1. / 3.) == Approx(legacy_setup));
    // fill in (0,1) keeps r_bisph inside dist/2 structurally.
    CHECK(bco_utils::blended_rbisph(rmid, dist, 0.999) < dist / 2.);
    CHECK(bco_utils::blended_rbisph(rmid, dist, 0.001) > rmid);
}

TEST_CASE("binary NS bounds subdivide a fixed r_bisph band",
          "[ns-bounds][rbisph]")
{
    BoundsConfig config = make_bounds_config(1. / 3.);
    const double r_bisph = bco_utils::blended_rbisph(6.64, 35., 1. / 3.);

    config.set(NSHELLS, BCO1) = 2;
    const auto bounds = bco_utils::make_NS_bounds_fixed_rbisph(config, r_bisph, BCO1);
    REQUIRE(bounds.size() == 5);
    const double shell_width = (r_bisph - 6.64) / 3.;
    CHECK(bounds[0] == Approx(3.0));
    CHECK(bounds[1] == Approx(6.64));
    CHECK(bounds[2] == Approx(6.64 + shell_width));
    CHECK(bounds[3] == Approx(6.64 + 2. * shell_width));
    CHECK(bounds[4] == Approx(r_bisph));
    CHECK(static_cast<double>(config(ROUT, BCO1)) == Approx(bounds[2]));

    // Adding a shell subdivides the same band: r_bisph does not move.
    config.set(NSHELLS, BCO1) = 3;
    const auto more_shells = bco_utils::make_NS_bounds_fixed_rbisph(config, r_bisph, BCO1);
    REQUIRE(more_shells.size() == 6);
    CHECK(more_shells[2] < bounds[2]);
    CHECK(more_shells.back() == Approx(bounds.back()));

    CHECK_THROWS(bco_utils::make_NS_bounds_fixed_rbisph(config, 6.0, BCO1)); // r_bisph <= RMID
}

TEST_CASE("make_binary_NS_bounds dispatches on rbisph_fill",
          "[ns-bounds][rbisph]")
{
    SECTION("fill = 1/3 selects the blended fixed r_bisph")
    {
        BoundsConfig config = make_bounds_config(1. / 3.);
        const auto bounds = bco_utils::make_binary_NS_bounds(config, BCO1);
        CHECK(bounds.back() == Approx(bco_utils::blended_rbisph(6.64, 35., 1. / 3.)));
        CHECK(bounds.back() < 35. / 2.);
    }
    SECTION("nonzero shells subdivide the blended r_bisph band")
    {
        BoundsConfig config = make_bounds_config(1. / 3.);
        config.set(NSHELLS, BCO1) = 2;
        const auto bounds = bco_utils::make_binary_NS_bounds(config, BCO1);
        const double r_bisph = bco_utils::blended_rbisph(6.64, 35., 1. / 3.);
        const double shell_width = (r_bisph - 6.64) / 3.;
        REQUIRE(bounds.size() == 5);
        CHECK(bounds[2] == Approx(6.64 + shell_width));
        CHECK(bounds.back() == Approx(r_bisph));
        CHECK(static_cast<double>(config(ROUT, BCO1)) == Approx(bounds[2]));
    }
    SECTION("absent key defaults to the blend (old configs migrate)")
    {
        BoundsConfig config;
        config.initialize_binary({"ns", "ns"}); // no set_defaults: RBISPH_FILL stays unset
        config.set(DIST) = 35.;
        config.set(RIN, BCO1) = 3.0;
        config.set(RMID, BCO1) = 6.64;
        config.set(NSHELLS, BCO1) = 0;
        const auto bounds = bco_utils::make_binary_NS_bounds(config, BCO1);
        CHECK(bounds.back() == Approx(bco_utils::blended_rbisph(6.64, 35., 1. / 3.)));
    }
    SECTION("explicit fill = 0 uses config ROUT as the fixed outer target")
    {
        BoundsConfig config = make_bounds_config(0.);
        config.set(NSHELLS, BCO1) = 1;
        const auto bounds = bco_utils::make_binary_NS_bounds(config, BCO1);
        REQUIRE(bounds.size() == 4);
        const double shell_width = (8.0 - 6.64) / 2.;
        CHECK(bounds[2] == Approx(6.64 + shell_width));
        CHECK(bounds.back() == Approx(8.0));
        CHECK(static_cast<double>(config(ROUT, BCO1)) == Approx(bounds[2]));
    }
    SECTION("fill >= 1 is rejected")
    {
        BoundsConfig config = make_bounds_config(1.5);
        CHECK_THROWS(bco_utils::make_binary_NS_bounds(config, BCO1));
    }
    SECTION("fresh binary configs default to fill = 1/3")
    {
        BoundsConfig config;
        config.initialize_binary({"ns", "ns"});
        config.set_defaults();
        CHECK(static_cast<double>(config(RBISPH_FILL)) == Approx(1. / 3.));
    }
}

TEST_CASE("binary NS shell count keeps the bispheric radius fixed beyond ROUT",
          "[ns-bounds][rbisph]")
{
    BoundsConfig config = make_bounds_config(1. / 3.);
    const auto no_shells = bco_utils::make_binary_NS_bounds(config, BCO1);
    const double fixed_r_bisph = no_shells.back();

    config.set(NSHELLS, BCO1) = 3;
    const auto bounds = bco_utils::make_binary_NS_bounds(config, BCO1);
    REQUIRE(bounds.size() == 6);
    CHECK(bounds.back() == Approx(fixed_r_bisph));
    CHECK(bounds[2] < bounds.back());
    for (std::size_t i = 1; i < bounds.size(); ++i)
        CHECK(bounds[i] > bounds[i - 1]);
}
