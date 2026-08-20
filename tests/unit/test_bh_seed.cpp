#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "Apps/Seed/GR/bh_seed_utils.hpp"
#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;

TEST_CASE("Non-rotating trumpet BH profile samples paper trumpet slice", "[seed][bh]")
{
    Kadath::Seed::NonRotatingTrumpetProfile profile(1.0);

    const double horizon_radius = profile.horizon_isotropic_radius();
    REQUIRE_THAT(horizon_radius, WithinAbs(0.7793271080557972, 1e-14));
    REQUIRE_THAT(profile.limiting_areal_radius(), WithinAbs(1.5, 1e-14));
    REQUIRE_THAT(profile.areal_radius(0.0), WithinAbs(1.5, 1e-14));
    REQUIRE_THAT(profile.areal_radius(horizon_radius), WithinAbs(2.0, 1e-13));

    REQUIRE_THAT(profile.conformal_factor(horizon_radius), WithinAbs(std::sqrt(2.0 / horizon_radius), 1e-14));
    REQUIRE_THAT(profile.lapse(horizon_radius), WithinAbs(3.0 * std::sqrt(3.0) / 16.0, 1e-14));
    REQUIRE_THAT(profile.radial_shift(horizon_radius),
                 WithinAbs(3.0 * std::sqrt(3.0) * horizon_radius / 32.0, 1e-14));

    const double infinity = std::numeric_limits<double>::infinity();
    REQUIRE_THAT(profile.conformal_factor(infinity), WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(profile.lapse(infinity), WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(profile.lapse_times_conformal_factor(infinity), WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(profile.radial_shift(infinity), WithinAbs(0.0, 1e-14));
}

TEST_CASE("Non-rotating trumpet BH profile validates mass", "[seed][bh]")
{
    REQUIRE_THROWS(Kadath::Seed::NonRotatingTrumpetProfile(0.0));
    REQUIRE_NOTHROW(Kadath::Seed::NonRotatingTrumpetProfile(1.0));
}

TEST_CASE("Binary BH irreducible mass is synchronized from Christodoulou mass and spin", "[bh][config]")
{
    kadath_config<BIN_INFO> config;
    config.initialize_binary({"ns", "bh"});
    config.set_defaults();
    config.set(MCH, BCO2) = 0.8;
    config.set(CHI, BCO2) = 0.6;
    config.set(MIRR, BCO2) = 0.1;

    bco_utils::sync_mirr_from_mch(config, BCO2);

    const double expected_mirr = std::sqrt((1.0 + std::sqrt(1.0 - 0.6 * 0.6)) / 2.0) * 0.8;
    REQUIRE_THAT(config(MIRR, BCO2), WithinAbs(expected_mirr, 1e-15));

    const auto output_path = std::filesystem::temp_directory_path() / "kadath_binary_bh_mirr_sync.toml";
    config.write_config(output_path.string());
    kadath_config<BIN_INFO> reread{output_path.string()};
    REQUIRE_THAT(reread(MIRR, BCO2), WithinAbs(expected_mirr, 1e-15));
}

TEST_CASE("Binary BH bounds subdivide a fixed r_bisph band", "[bh][bounds]")
{
    kadath_config<BIN_INFO> config;
    config.initialize_binary({"bh", "bh"});
    config.set_defaults();
    config.set(RIN, BCO1) = 1.0;
    config.set(RMID, BCO1) = 2.0;
    config.set(ROUT, BCO1) = 5.0;
    config.set(NSHELLS, BCO1) = 2;

    std::vector<double> bounds(5);
    bco_utils::set_BH_bounds(bounds, config, BCO1);

    REQUIRE_THAT(bounds[0], WithinAbs(1.0, 1e-15));
    REQUIRE_THAT(bounds[1], WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(bounds[2], WithinAbs(3.0, 1e-15));
    REQUIRE_THAT(bounds[3], WithinAbs(4.0, 1e-15));
    REQUIRE_THAT(bounds[4], WithinAbs(5.0, 1e-15));
    REQUIRE_THAT(config(ROUT, BCO1), WithinAbs(bounds[2], 1e-15));
    REQUIRE(static_cast<int>(config(NSHELLS, BCO1)) == 2);
}
