#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Apps/Formalism/Shared/ns_binary_boost_kinematics.hpp"

#include <stdexcept>

using Catch::Matchers::WithinAbs;
using Kadath::ns_binary_boost::CartesianVector;

namespace
{
    CartesianVector add(CartesianVector left, CartesianVector right)
    {
        return {left.x + right.x, left.y + right.y, left.z + right.z};
    }

    CartesianVector scale(double factor, CartesianVector vector)
    {
        return {factor * vector.x, factor * vector.y, factor * vector.z};
    }

    void require_vector(CartesianVector actual, CartesianVector expected, double tolerance = 1.e-14)
    {
        REQUIRE_THAT(actual.x, WithinAbs(expected.x, tolerance));
        REQUIRE_THAT(actual.y, WithinAbs(expected.y, tolerance));
        REQUIRE_THAT(actual.z, WithinAbs(expected.z, tolerance));
    }
} // namespace

TEST_CASE("local NS boost equals the global helical generator", "[ns-binary-boost][generator-identity]")
{
    const CartesianVector beta{2.1e-3, -7.5e-4, 1.7e-4};
    const CartesianVector center{-13.5, 2.25, -0.4};
    const CartesianVector local_position{1.75, -0.6, 0.9};
    constexpr double omega = 7.3e-3;
    constexpr double xaxis = 1.2;
    constexpr double zvel = -4.1e-4;

    const auto global_position = add(center, local_position);
    const auto global_rotation = scale(omega, Kadath::ns_binary_boost::rotation_about_z(global_position));
    const auto global_generator = add(beta, {
                                                global_rotation.x,
                                                global_rotation.y + omega * xaxis,
                                                global_rotation.z + zvel,
                                            });

    const auto local_rotation = scale(omega, Kadath::ns_binary_boost::rotation_about_z(local_position));
    const auto local_generator =
        add(beta, add(local_rotation, Kadath::ns_binary_boost::local_translation(center, omega, xaxis, zvel)));

    require_vector(local_generator, global_generator);
}

TEST_CASE("opposite binary centres receive opposite boost translations", "[ns-binary-boost][inversion-symmetry]")
{
    constexpr double separation = 35.;
    constexpr double omega = 6.2e-3;
    const auto center1 = Kadath::ns_binary_boost::component_center(separation, BCO1);
    const auto center2 = Kadath::ns_binary_boost::component_center(separation, BCO2);

    require_vector(center1, scale(-1., center2));

    const auto boost1 = Kadath::ns_binary_boost::local_translation(center1, omega, 0., 0.);
    const auto boost2 = Kadath::ns_binary_boost::local_translation(center2, omega, 0., 0.);
    require_vector(boost1, scale(-1., boost2));
}

TEST_CASE("zero orbital frequency leaves only the vertical boost", "[ns-binary-boost][edge]")
{
    const CartesianVector center{-17.5, 0., 0.};
    constexpr double zvel = -3.4e-4;

    const auto boost = Kadath::ns_binary_boost::local_translation(center, 0., 2.5, zvel);
    require_vector(boost, {0., 0., zvel});
}

TEST_CASE("NS binary boost rejects non-component nodes", "[ns-binary-boost][failure]")
{
    REQUIRE_THROWS_AS(Kadath::ns_binary_boost::component_center(35., BINARY), std::invalid_argument);
}
