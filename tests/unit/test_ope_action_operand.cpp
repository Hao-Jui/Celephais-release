#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "../../src/Domain/Adapted_polar/adapted_polar_term_eq_lanes.hpp"

#include <cmath>
#include <optional>

using namespace Kadath;

namespace
{
    class FixedActionOpe : public Ope_eq
    {
      public:
        FixedActionOpe(const Term_eq& result, int& calls)
            : Ope_eq(nullptr, result.get_dom()), result_(result), calls_(calls)
        {
        }

        Term_eq action() const override
        {
            ++calls_;
            return result_;
        }

      private:
        Term_eq result_;
        int& calls_;
    };
}

TEST_CASE("Exact Ope_id leaves lend their persistent result to parent operators", "[ope][action-operand]")
{
    Term_eq target(4, 3.0, 5.0);
    target.set_derivative_lane_count(3);
    target.set_der_d(1, 7.0);
    target.set_der_d(2, 11.0);
    Ope_id identity(nullptr, &target);

    std::optional<Term_eq> storage;
    const Term_eq& operand = identity.action_operand(storage);

    REQUIRE(&operand == &target);
    REQUIRE_FALSE(storage.has_value());
    REQUIRE(operand.get_val_d() == 3.0);
    REQUIRE(operand.get_der_d(0) == 5.0);
    REQUIRE(operand.get_der_d(1) == 7.0);
    REQUIRE(operand.get_der_d(2) == 11.0);
}

TEST_CASE("Exact Ope_id leaves never overwrite an occupied scratch slot", "[ope][action-operand][alias]")
{
    Term_eq target(4, 3.0, 5.0);
    target.set_der_d(31, 7.0);
    Ope_id identity(nullptr, &target);

    std::optional<Term_eq> scratch(std::in_place, 4, -11.0, -13.0);
    scratch->set_der_d(31, -17.0);
    Term_eq* const scratch_address = &*scratch;
    const Term_eq& operand = identity.action_operand(scratch);

    REQUIRE(&operand == &target);
    REQUIRE(&*scratch == scratch_address);
    REQUIRE(scratch->get_val_d() == -11.0);
    REQUIRE(scratch->get_der_d(0) == -13.0);
    REQUIRE(scratch->get_der_d(31) == -17.0);
    REQUIRE(target.get_val_d() == 3.0);
    REQUIRE(target.get_der_d(0) == 5.0);
    REQUIRE(target.get_der_d(31) == 7.0);
}

TEST_CASE("Non-identity operators retain action results in caller storage", "[ope][action-operand]")
{
    Term_eq source(2, 13.0, 17.0);
    source.set_derivative_lane_count(2);
    source.set_der_d(1, 19.0);
    int calls = 0;
    FixedActionOpe operation(source, calls);

    std::optional<Term_eq> storage;
    const Term_eq& operand = operation.action_operand(storage);

    REQUIRE(calls == 1);
    REQUIRE(storage.has_value());
    REQUIRE(&operand == &*storage);
    REQUIRE(&operand != &source);
    REQUIRE(operand.get_val_d() == 13.0);
    REQUIRE(operand.get_der_d(0) == 17.0);
    REQUIRE(operand.get_der_d(1) == 19.0);
}

TEST_CASE("Axisymmetric adapted-polar Laplacians omit the zero azimuthal branch",
          "[adapted-polar][laplacian]")
{
    Term_eq source(0, 3.0, 5.0);
    source.set_derivative_lane_count(2);
    source.set_der_d(1, 7.0);
    const Term_eq radius(0, 11.0, 0.0);

    auto radial_derivative = [](const Term_eq& value) { return 2.0 * value; };
    auto theta_derivative = [](const Term_eq& value) { return 3.0 * value; };
    auto multiply_cos_theta = [](const Term_eq& value) { return 7.0 * value; };

    int optimized_divisions = 0;
    auto optimized_divide_sin_theta = [&](const Term_eq& value) {
        ++optimized_divisions;
        return value / 5.0;
    };
    const Term_eq optimized = adapted_polar_detail::scalar_laplacian_term_eq(
        source,
        0,
        radius,
        radial_derivative,
        theta_derivative,
        optimized_divide_sin_theta,
        multiply_cos_theta);

    int legacy_divisions = 0;
    auto legacy_divide_sin_theta = [&](const Term_eq& value) {
        ++legacy_divisions;
        return value / 5.0;
    };
    const Term_eq source_over_sin_theta = legacy_divide_sin_theta(source);
    const Term_eq derivative_theta = theta_derivative(source);
    const Term_eq second_derivative_theta = theta_derivative(derivative_theta);
    const Term_eq cotangent_and_azimuthal = legacy_divide_sin_theta(
        multiply_cos_theta(derivative_theta) - 0 * source_over_sin_theta);
    const Term_eq derivative_radius = radial_derivative(source);
    const Term_eq legacy = radial_derivative(derivative_radius) + 2 * derivative_radius / radius +
                           (second_derivative_theta + cotangent_and_azimuthal) / radius / radius;

    REQUIRE(optimized_divisions == 1);
    REQUIRE(legacy_divisions == 2);
    REQUIRE(optimized.get_val_d() == legacy.get_val_d());
    REQUIRE(optimized.get_der_d(0) == legacy.get_der_d(0));
    REQUIRE(optimized.get_der_d(1) == legacy.get_der_d(1));
}

TEST_CASE("Non-axisymmetric adapted-polar Laplacians retain the azimuthal branch",
          "[adapted-polar][laplacian]")
{
    const Term_eq source(0, 3.0, 5.0);
    const Term_eq radius(0, 11.0, 0.0);
    int divisions = 0;

    const Term_eq result = adapted_polar_detail::scalar_laplacian_term_eq(
        source,
        2,
        radius,
        [](const Term_eq& value) { return 2.0 * value; },
        [](const Term_eq& value) { return 3.0 * value; },
        [&](const Term_eq& value) {
            ++divisions;
            return value / 5.0;
        },
        [](const Term_eq& value) { return 7.0 * value; });

    REQUIRE(divisions == 2);
    REQUIRE(std::isfinite(result.get_val_d()));
    REQUIRE(std::isfinite(result.get_der_d()));
}
