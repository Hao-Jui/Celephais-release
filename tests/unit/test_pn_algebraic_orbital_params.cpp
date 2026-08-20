#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

using Catch::Matchers::WithinAbs;

namespace
{
    struct NonspinningFixture
    {
        double mass1;
        double mass2;
        double separation;
        double omega;
        double adot;
        double radial_velocity;
    };

    constexpr NonspinningFixture representative_nonspinning{
        1.35,
        1.05,
        32.,
        7.79949155198471356e-03,
        -2.39556938409805369e-05,
        -7.66582202911377181e-04,
    };

    constexpr std::array<NonspinningFixture, 6> nonspinning_grid{{
        {1., 1., 30., 7.92106539261248042e-03, -1.97029198510680015e-05, -5.91087595532040037e-04},
        {2., 1., 50., 4.53809716499945122e-03, -8.15908571428571353e-06, -4.07954285714285704e-04},
        {4., 1., 80., 2.87752877264854830e-03, -4.10495721726190492e-06, -3.28396577380952407e-04},
        {8., 1., 120., 2.06152509883278118e-03, -2.64972810111699049e-06, -3.17967372134038862e-04},
        {16., 1., 170., 1.62141143792751664e-03, -1.95579400568071987e-06, -3.32484980965722413e-04},
        {32., 1., 230., 1.35101102929406506e-03, -1.18296811891917063e-06, -2.72082667351409256e-04},
    }};

    double relative_tolerance(double expected, double scale)
    {
        return std::abs(expected) * scale;
    }

    Kadath::PN::AlgebraicPNParameters representative_binary()
    {
        Kadath::PN::AlgebraicPNParameters params;
        params.mass1 = representative_nonspinning.mass1;
        params.mass2 = representative_nonspinning.mass2;
        return params;
    }

    double vector_dot(Kadath::PN::CartesianVector a, Kadath::PN::CartesianVector b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    double vector_norm(Kadath::PN::CartesianVector v)
    {
        return std::sqrt(vector_dot(v, v));
    }
} // namespace

TEST_CASE("PN zero-spin aligned seed matches nonspinning algebraic parameters",
          "[pn][algebraic-orbital]")
{
    const auto params = representative_binary();
    const double separation = representative_nonspinning.separation;
    const double center_of_mass = bco_utils::com_estimate(separation, params.mass1, params.mass2);
    const auto reference = representative_nonspinning;
    const auto zero_spin =
        Kadath::PN::aligned_spin_algebraic_pn_parameters(params,
                                                         separation,
                                                         {0., 0., 0.},
                                                         {0., 0., 0.},
                                                         center_of_mass);

    REQUIRE_THAT(zero_spin.omega,
                 WithinAbs(reference.omega, relative_tolerance(reference.omega, 1.e-13)));
    REQUIRE_THAT(zero_spin.adot,
                 WithinAbs(reference.adot, relative_tolerance(reference.adot, 1.e-13)));
    REQUIRE_THAT(zero_spin.radial_velocity,
                 WithinAbs(reference.radial_velocity,
                           relative_tolerance(reference.radial_velocity, 1.e-13)));
}

TEST_CASE("PN aligned-spin algebraic seed changes omega and adot from the nonspinning seed",
          "[pn][algebraic-orbital]")
{
    const auto params = representative_binary();
    const double separation = representative_nonspinning.separation;
    const auto chi1 = Kadath::PN::SpinVector{0., 0., 0.6};
    const auto chi2 = Kadath::PN::SpinVector{0., 0., 0.};
    const double center_of_mass = bco_utils::com_estimate(separation, params.mass1, params.mass2);

    const auto nonspinning = representative_nonspinning;
    const auto aligned =
        Kadath::PN::aligned_spin_algebraic_pn_parameters(params, separation, chi1, chi2, center_of_mass);

    REQUIRE(aligned.omega < nonspinning.omega);
    REQUIRE(std::abs(aligned.omega - nonspinning.omega) >
            relative_tolerance(nonspinning.omega, 1.e-3));
    REQUIRE(std::abs(aligned.adot - nonspinning.adot) >
            relative_tolerance(nonspinning.adot, 1.e-2));
}

TEST_CASE("PN zero-spin aligned seed matches nonspinning regression grid",
          "[pn][algebraic-orbital]")
{
    for (const auto point : nonspinning_grid)
    {
        Kadath::PN::AlgebraicPNParameters params;
        params.mass1 = point.mass1;
        params.mass2 = point.mass2;

        const double center_of_mass =
            bco_utils::com_estimate(point.separation, params.mass1, params.mass2);
        const auto zero_spin =
            Kadath::PN::aligned_spin_algebraic_pn_parameters(params,
                                                             point.separation,
                                                             {0., 0., 0.},
                                                             {0., 0., 0.},
                                                             center_of_mass);

        INFO("mass1=" << point.mass1
                      << " mass2=" << params.mass2
                      << " separation=" << point.separation
                      << " center_of_mass=" << center_of_mass);
        REQUIRE_THAT(zero_spin.omega,
                     WithinAbs(point.omega, relative_tolerance(point.omega, 1.e-13)));
        REQUIRE_THAT(zero_spin.adot,
                     WithinAbs(point.adot, relative_tolerance(point.adot, 1.e-13)));
        REQUIRE_THAT(zero_spin.radial_velocity,
                     WithinAbs(point.radial_velocity,
                               relative_tolerance(point.radial_velocity, 1.e-13)));
    }
}

TEST_CASE("PN aligned-spin adot uses Kidder's circular inspiral coefficient",
          "[pn][algebraic-orbital]")
{
    const auto params = representative_binary();
    const double separation = representative_nonspinning.separation;
    const auto chi1 = Kadath::PN::SpinVector{0., 0., 0.45};
    const auto chi2 = Kadath::PN::SpinVector{0., 0., -0.25};
    const double center_of_mass = bco_utils::com_estimate(separation, params.mass1, params.mass2);

    const auto nonspinning = representative_nonspinning;
    const auto aligned =
        Kadath::PN::aligned_spin_algebraic_pn_parameters(params, separation, chi1, chi2, center_of_mass);

    const double mass = params.mass1 + params.mass2;
    const double eta = params.mass1 * params.mass2 / (mass * mass);
    const double x1 = params.mass1 / mass;
    const double x2 = params.mass2 / mass;
    const double gamma = mass / separation;
    const double spin_orbit_coefficient =
        (7. / 12.) * ((19. * x1 * x1 + 15. * eta) * chi1.z +
                      (19. * x2 * x2 + 15. * eta) * chi2.z);
    const double expected_radial_velocity =
        nonspinning.radial_velocity * (1. - spin_orbit_coefficient * std::pow(gamma, 1.5));

    REQUIRE_THAT(aligned.radial_velocity,
                 WithinAbs(expected_radial_velocity,
                           relative_tolerance(expected_radial_velocity, 1.e-13)));
    REQUIRE_THAT(aligned.adot,
                 WithinAbs(expected_radial_velocity / separation,
                           relative_tolerance(expected_radial_velocity / separation, 1.e-13)));
}

TEST_CASE("PN spin projection keeps only the orbital-axis component",
          "[pn][algebraic-orbital]")
{
    REQUIRE_THAT(Kadath::PN::dimensionless_spin_along_orbital_axis(0.6, 0.),
                 WithinAbs(0.6, 1.e-15));
    REQUIRE_THAT(Kadath::PN::dimensionless_spin_along_orbital_axis(0.6, 60.),
                 WithinAbs(0.3, 1.e-15));
    REQUIRE_THAT(Kadath::PN::dimensionless_spin_along_orbital_axis(0.6, 90.),
                 WithinAbs(0., 1.e-15));
}

TEST_CASE("PN precessing seed reduces to the aligned-spin seed",
          "[pn][algebraic-orbital]")
{
    const auto params = representative_binary();
    Kadath::PN::PrecessingPNState state;
    state.separation = representative_nonspinning.separation;
    state.dimensionless_spin1 = {0., 0., 0.45};
    state.dimensionless_spin2 = {0., 0., -0.25};
    state.center_of_mass = bco_utils::com_estimate(state.separation, params.mass1, params.mass2);

    const auto aligned =
        Kadath::PN::aligned_spin_algebraic_pn_parameters(params,
                                                         state.separation,
                                                         state.dimensionless_spin1,
                                                         state.dimensionless_spin2,
                                                         state.center_of_mass);
    const auto precessing = Kadath::PN::precessing_algebraic_pn_parameters(params, state);

    REQUIRE_THAT(precessing.omega,
                 WithinAbs(aligned.omega, relative_tolerance(aligned.omega, 1.e-13)));
    REQUIRE_THAT(precessing.adot,
                 WithinAbs(aligned.adot, relative_tolerance(aligned.adot, 1.e-13)));
    REQUIRE_THAT(precessing.radial_velocity,
                 WithinAbs(aligned.radial_velocity,
                           relative_tolerance(aligned.radial_velocity, 1.e-13)));
    REQUIRE_THAT(vector_norm(precessing.l_hat_dot), WithinAbs(0., 1.e-14));
    REQUIRE_THAT(vector_norm(precessing.dimensionless_spin1_dot), WithinAbs(0., 1.e-14));
    REQUIRE_THAT(vector_norm(precessing.dimensionless_spin2_dot), WithinAbs(0., 1.e-14));
    REQUIRE_THAT(vector_dot(precessing.acceleration, state.frame.n_hat) +
                     state.separation * precessing.omega * precessing.omega,
                 WithinAbs(0., 1.e-14));
}

TEST_CASE("PN precessing seed reports orbital and spin precession for tilted spins",
          "[pn][algebraic-orbital]")
{
    const auto params = representative_binary();
    Kadath::PN::PrecessingPNState state;
    state.separation = representative_nonspinning.separation;
    state.dimensionless_spin1 = {0.6, 0., 0.};
    state.dimensionless_spin2 = {0., 0., 0.};
    state.center_of_mass = bco_utils::com_estimate(state.separation, params.mass1, params.mass2);

    const auto precessing = Kadath::PN::precessing_algebraic_pn_parameters(params, state);

    REQUIRE(std::isfinite(precessing.omega));
    REQUIRE(std::abs(precessing.acceleration.z) > 1.e-8);
    REQUIRE(std::abs(precessing.l_hat_dot.y) > 1.e-8);
    REQUIRE_THAT(vector_dot(precessing.l_hat_dot, state.frame.l_hat), WithinAbs(0., 1.e-14));
    REQUIRE(std::abs(precessing.dimensionless_spin1_dot.y) > 1.e-8);
    REQUIRE_THAT(vector_norm(precessing.dimensionless_spin2_dot), WithinAbs(0., 1.e-15));
    REQUIRE_THAT(precessing.adot,
                 WithinAbs(representative_nonspinning.adot,
                           relative_tolerance(representative_nonspinning.adot, 1.e-13)));
}

TEST_CASE("PN tilt seed reduces to the aligned-spin seed at zero tilt",
          "[pn][algebraic-orbital]")
{
    const auto params = representative_binary();
    const double separation = representative_nonspinning.separation;
    const auto chi1 = Kadath::PN::SpinVector{0., 0., 0.45};
    const auto chi2 = Kadath::PN::SpinVector{0., 0., -0.25};
    const double center_of_mass = bco_utils::com_estimate(separation, params.mass1, params.mass2);

    const auto aligned =
        Kadath::PN::aligned_spin_algebraic_pn_parameters(params, separation, chi1, chi2, center_of_mass);
    const auto tilt = Kadath::PN::precessing_seed_from_tilt(params.mass1,
                                                           params.mass2,
                                                           separation,
                                                           chi1.z,
                                                           0.,
                                                           chi2.z,
                                                           0.,
                                                           center_of_mass);

    REQUIRE_THAT(tilt.omega, WithinAbs(aligned.omega, relative_tolerance(aligned.omega, 1.e-13)));
    REQUIRE_THAT(tilt.adot, WithinAbs(aligned.adot, relative_tolerance(aligned.adot, 1.e-13)));
}

TEST_CASE("PN tilt seed: in-plane spins keep nonspinning adot but shift omega via spin-spin",
          "[pn][algebraic-orbital]")
{
    const auto params = representative_binary();
    const double separation = representative_nonspinning.separation;
    const double center_of_mass = bco_utils::com_estimate(separation, params.mass1, params.mass2);

    // Both spins tilted 90 degrees from the orbital axis -> aligned component is
    // zero, so the radiation-reaction adot must collapse to the nonspinning value,
    // while the leading in-plane spin-spin term (on by default) shifts omega.
    const auto tilt = Kadath::PN::precessing_seed_from_tilt(params.mass1,
                                                           params.mass2,
                                                           separation,
                                                           0.6,
                                                           90.,
                                                           0.5,
                                                           90.,
                                                           center_of_mass);

    REQUIRE_THAT(tilt.adot,
                 WithinAbs(representative_nonspinning.adot,
                           relative_tolerance(representative_nonspinning.adot, 1.e-13)));
    REQUIRE(std::abs(tilt.omega - representative_nonspinning.omega) >
            relative_tolerance(representative_nonspinning.omega, 1.e-6));
}

TEST_CASE("PN tilt seed rejects non-finite spin and axis inputs", "[pn][algebraic-orbital]")
{
    const auto params = representative_binary();
    REQUIRE_THROWS_AS(Kadath::PN::precessing_seed_from_tilt(params.mass1,
                                                            params.mass2,
                                                            32.,
                                                            std::numeric_limits<double>::quiet_NaN(),
                                                            0.,
                                                            0.,
                                                            0.,
                                                            0.),
                      std::invalid_argument);
}

TEST_CASE("PN precessing seed validates the orbital frame",
          "[pn][algebraic-orbital]")
{
    const auto params = representative_binary();
    Kadath::PN::PrecessingPNState state;
    state.separation = representative_nonspinning.separation;
    state.center_of_mass = bco_utils::com_estimate(state.separation, params.mass1, params.mass2);

    state.frame.lambda_hat = {0., 2., 0.};
    REQUIRE_THROWS_AS(Kadath::PN::precessing_algebraic_pn_parameters(params, state),
                      std::invalid_argument);

    state.frame.lambda_hat = {0., 1., 0.};
    state.frame.l_hat = {0., 0., -1.};
    REQUIRE_THROWS_AS(Kadath::PN::precessing_algebraic_pn_parameters(params, state),
                      std::invalid_argument);
}

TEST_CASE("PN aligned-spin algebraic seed rejects generic spin directions",
          "[pn][algebraic-orbital]")
{
    const auto params = representative_binary();
    const double separation = 32.;
    const auto generic_spin = Kadath::PN::SpinVector{0.1, 0., 0.6};
    const auto aligned_spin = Kadath::PN::SpinVector{0., 0., 0.};

    REQUIRE_THROWS_AS(Kadath::PN::aligned_spin_algebraic_pn_parameters(params,
                                                                       separation,
                                                                       generic_spin,
                                                                       aligned_spin),
                      std::invalid_argument);
}

TEST_CASE("PN aligned-spin algebraic seed validates scalar inputs", "[pn][algebraic-orbital]")
{
    REQUIRE_THROWS_AS(Kadath::PN::aligned_spin_algebraic_pn_parameters(-1.,
                                                                       1.,
                                                                       32.,
                                                                       {0., 0., 0.},
                                                                       {0., 0., 0.},
                                                                       0.),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Kadath::PN::aligned_spin_algebraic_pn_parameters(1.,
                                                                       1.,
                                                                       -32.,
                                                                       {0., 0., 0.},
                                                                       {0., 0., 0.},
                                                                       0.),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Kadath::PN::aligned_spin_algebraic_pn_parameters(1.,
                                                                       1.,
                                                                       32.,
                                                                       {0., 0., 0.},
                                                                       {0., 0., 0.},
                                                                       std::numeric_limits<double>::quiet_NaN()),
                      std::invalid_argument);
}
