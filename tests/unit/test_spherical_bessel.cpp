#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Utilities/spherical_bessel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace
{
    struct GoldenValue
    {
        int order;
        double argument;
        double expected_j;
        double expected_y;
    };

    // Generated with GSL 2.8 gsl_sf_bessel_jl/yl.  The table spans the
    // small-argument series, the l ~= x transition where j_l switches to
    // downward recurrence, and the oscillatory x > l region.
    constexpr std::array golden_values{
        GoldenValue{0, 1.e-12, 1., -1.e12},
        GoldenValue{1, 1.e-2, 3.3333000001190475e-3, -1.0000499987500069e4},
        GoldenValue{2, 1.e-1, 6.6619060844556877e-4, -3.0050124791753442e3},
        GoldenValue{5, 1., 9.2561158611258144e-5, -9.9944034339223640e2},
        GoldenValue{10, 3.25, 7.5839040020531456e-6, -2.0330183273853768e3},
        GoldenValue{20, 10., 2.3083719613194699e-6, -1.2112106053526034e3},
        GoldenValue{20, 50., -1.5785029898269291e-2, 1.3759531302541211e-2},
        GoldenValue{50, 50., 1.8829107369282640e-2, -4.1900001504607758e-2},
    };

    void require_relative_match(double actual, double expected, double relative_tolerance = 5.e-13)
    {
        INFO("actual=" << actual << ", expected=" << expected);
        REQUIRE(std::isfinite(actual));
        REQUIRE(std::abs(actual - expected) <= relative_tolerance * std::abs(expected));
    }
} // namespace

TEST_CASE("Internal spherical Bessel functions match GSL golden values", "[spherical-bessel]")
{
    for (const auto& value : golden_values) {
        CAPTURE(value.order, value.argument);
        require_relative_match(
            Kadath::special_functions::spherical_bessel_j(value.order, value.argument), value.expected_j);
        require_relative_match(
            Kadath::special_functions::spherical_bessel_y(value.order, value.argument), value.expected_y);
    }
}

TEST_CASE("Internal spherical Bessel functions preserve small nonzero values", "[spherical-bessel][edge]")
{
    // These values exercise underflow-sensitive high-order behaviour.  An
    // absolute tolerance would incorrectly allow an implementation returning
    // zero for j_20 here.
    require_relative_match(Kadath::special_functions::spherical_bessel_j(20, 1.e-12),
                           7.6259790048921402e-266,
                           2.e-12);
    require_relative_match(Kadath::special_functions::spherical_bessel_y(20, 1.e-12),
                           -3.1983098677287807e275,
                           2.e-12);
}

TEST_CASE("Internal spherical Bessel functions define GSL-compatible domains", "[spherical-bessel][edge]")
{
    using Kadath::special_functions::spherical_bessel_j;
    using Kadath::special_functions::spherical_bessel_y;

    REQUIRE(spherical_bessel_j(0, 0.) == 1.);
    for (const int order : {1, 2, 20})
        REQUIRE(spherical_bessel_j(order, 0.) == 0.);

    REQUIRE(std::isnan(spherical_bessel_j(-1, 1.)));
    REQUIRE(std::isnan(spherical_bessel_j(0, -1.)));
    REQUIRE(std::isnan(spherical_bessel_y(-1, 1.)));
    REQUIRE(std::isnan(spherical_bessel_y(0, 0.)));
    REQUIRE(std::isnan(spherical_bessel_y(0, -1.)));

    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    REQUIRE(std::isnan(spherical_bessel_j(0, infinity)));
    REQUIRE(std::isnan(spherical_bessel_y(0, infinity)));
    REQUIRE(std::isnan(spherical_bessel_j(0, nan)));
    REQUIRE(std::isnan(spherical_bessel_y(0, nan)));
}

TEST_CASE("Internal spherical Bessel recurrences hold across solver-scale orders", "[spherical-bessel]")
{
    using Kadath::special_functions::spherical_bessel_j;
    using Kadath::special_functions::spherical_bessel_y;

    for (const int order : {1, 2, 5, 10, 20}) {
        for (const double x : {0.1, 1., 3.25, 10., 50.}) {
            CAPTURE(order, x);
            const double j_residual = spherical_bessel_j(order + 1, x) -
                                      (2 * order + 1) * spherical_bessel_j(order, x) / x +
                                      spherical_bessel_j(order - 1, x);
            const double y_residual = spherical_bessel_y(order + 1, x) -
                                      (2 * order + 1) * spherical_bessel_y(order, x) / x +
                                      spherical_bessel_y(order - 1, x);
            const double j_scale = std::max({std::abs(spherical_bessel_j(order - 1, x)),
                                             std::abs(spherical_bessel_j(order, x)),
                                             std::abs(spherical_bessel_j(order + 1, x))});
            const double y_scale = std::max({std::abs(spherical_bessel_y(order - 1, x)),
                                             std::abs(spherical_bessel_y(order, x)),
                                             std::abs(spherical_bessel_y(order + 1, x))});
            REQUIRE(std::abs(j_residual) <= 2.e-13 * j_scale);
            REQUIRE(std::abs(y_residual) <= 2.e-13 * y_scale);
        }
    }
}
