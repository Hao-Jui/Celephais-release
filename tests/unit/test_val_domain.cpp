#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Domain/fourD_periodic.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <utility>
#include <cmath>
#include <limits>

using namespace Kadath;
using Catch::Matchers::WithinAbs;

namespace {
Space_spheric make_space() {
    Point center(3);
    center.set(1) = 0; center.set(2) = 0; center.set(3) = 0;
    Dim_array res(3);
    res.set(0) = 9; res.set(1) = 5; res.set(2) = 4;
    Dim_array bd(1); bd.set(0) = 2;
    Array<double> bounds(bd);
    bounds.set(0) = 1.0; bounds.set(1) = 10.0;
    return Space_spheric(CHEB_TYPE, center, res, bounds);
}
}

TEST_CASE("Val_domain construction from Domain", "[val_domain]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    Val_domain vd(dom);
    // Default-constructed Val_domain is uninitialized (not zero).
    // Verify it can be set to zero explicitly.
    vd.set_zero();
    REQUIRE(vd.check_if_zero() == true);
}

TEST_CASE("Val_domain set constant", "[val_domain]") {
    auto space = make_space();
    Val_domain vd(space.get_domain(0));
    vd = 3.14;
    vd.coef_i();
    Index idx(vd.get_conf().get_dimensions());
    REQUIRE_THAT(vd(idx), WithinAbs(3.14, 1e-12));
}

TEST_CASE("Val_domain arithmetic", "[val_domain]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    Val_domain a(dom), b(dom);
    a = 2.0; b = 3.0;
    a += b;
    a.coef_i();
    Index idx(a.get_conf().get_dimensions());
    REQUIRE_THAT(a(idx), WithinAbs(5.0, 1e-12));
}

TEST_CASE("Val_domain compound arithmetic matches out-of-place values", "[val_domain][compound]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    Val_domain lhs(dom), rhs(dom);
    lhs = dom->get_cart(1) + 2.0;
    rhs = dom->get_cart(2) + 3.0;

    auto require_same = [](const Val_domain& expected, const Val_domain& actual) {
        REQUIRE_THAT(diffmax(expected, actual), WithinAbs(0.0, 0.0));
    };

    Val_domain actual(lhs);
    const Val_domain expected_add = lhs + rhs;
    actual += rhs;
    require_same(expected_add, actual);

    actual = lhs;
    const Val_domain expected_sub = lhs - rhs;
    actual -= rhs;
    require_same(expected_sub, actual);

    actual = lhs;
    const Val_domain expected_mul = lhs * rhs;
    actual *= rhs;
    require_same(expected_mul, actual);

    actual = lhs;
    const Val_domain expected_div = lhs / rhs;
    actual /= rhs;
    require_same(expected_div, actual);

    actual = lhs;
    actual += actual;
    require_same(lhs + lhs, actual);
    actual = lhs;
    actual -= actual;
    require_same(lhs - lhs, actual);
    actual = lhs;
    actual *= actual;
    require_same(lhs * lhs, actual);
    actual = lhs;
    actual /= actual;
    require_same(lhs / lhs, actual);
}

TEST_CASE("Val_domain product chains reuse a dead left temporary bit exactly",
          "[val_domain][move][product]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    Val_domain lhs(dom), middle(dom), rhs(dom);
    lhs = dom->get_cart(1) + 2.0;
    middle = dom->get_cart(2) + 3.0;
    rhs = dom->get_cart(3) + 4.0;

    const Val_domain first = lhs * middle;
    const Val_domain expected = first * rhs;
    const Val_domain actual = (lhs * middle) * rhs;

    REQUIRE_THAT(diffmax(expected, actual), WithinAbs(0.0, 0.0));
}

TEST_CASE("Val_domain transform caches survive reads and invalidate after two-sided writes",
          "[val_domain][transform-cache]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    Val_domain field(dom);
    field = dom->get_cart(1) + 2.0;
    field.std_base();

    Transform1dTrafficScope first_forward_scope(true);
    field.coef();
    const Transform1dTrafficSnapshot first_forward = first_forward_scope.finish();
    REQUIRE(first_forward.forward > 0);
    REQUIRE(first_forward.backward == 0);

    Transform1dTrafficScope cached_forward_scope(true);
    field.coef();
    const Transform1dTrafficSnapshot cached_forward = cached_forward_scope.finish();
    REQUIRE(cached_forward.forward == 0);
    REQUIRE(cached_forward.backward == 0);

    Index point(field.get_conf().get_dimensions());
    field.set(point) += 1.0;
    Transform1dTrafficScope invalidated_forward_scope(true);
    field.coef();
    const Transform1dTrafficSnapshot invalidated_forward = invalidated_forward_scope.finish();
    REQUIRE(invalidated_forward.forward == first_forward.forward);
    REQUIRE(invalidated_forward.backward == 0);

    field.set_in_coef();
    Transform1dTrafficScope first_backward_scope(true);
    field.coef_i();
    const Transform1dTrafficSnapshot first_backward = first_backward_scope.finish();
    REQUIRE(first_backward.forward == 0);
    REQUIRE(first_backward.backward > 0);

    Transform1dTrafficScope cached_backward_scope(true);
    field.coef_i();
    const Transform1dTrafficSnapshot cached_backward = cached_backward_scope.finish();
    REQUIRE(cached_backward.forward == 0);
    REQUIRE(cached_backward.backward == 0);

    Index coefficient(field.get_coef().get_dimensions());
    field.set_coef(coefficient) += 1.0;
    Transform1dTrafficScope invalidated_backward_scope(true);
    field.coef_i();
    const Transform1dTrafficSnapshot invalidated_backward = invalidated_backward_scope.finish();
    REQUIRE(invalidated_backward.forward == 0);
    REQUIRE(invalidated_backward.backward == first_backward.backward);
}

TEST_CASE("Val_domain linear rvalue chains stay in coefficient space",
          "[val_domain][transform-cache][linear-chain]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    Val_domain lhs(dom), middle(dom), rhs(dom);
    lhs = dom->get_cart(1) + 2.0;
    middle = dom->get_cart(2) + 3.0;
    rhs = dom->get_cart(3) + 4.0;
    lhs.std_base();
    middle.std_base();
    rhs.std_base();
    lhs.coef();
    middle.coef();
    rhs.coef();
    lhs.set_in_coef();
    middle.set_in_coef();
    rhs.set_in_coef();

    const Val_domain first = lhs + middle;
    const Val_domain expected = first - rhs;
    Transform1dTrafficScope chain_scope(true);
    const Val_domain actual = (lhs + middle) - rhs;
    const Transform1dTrafficSnapshot chain = chain_scope.finish();

    REQUIRE(chain.forward == 0);
    REQUIRE(chain.backward == 0);
    REQUIRE_THAT(diffmax(expected, actual), WithinAbs(0.0, 0.0));
}

TEST_CASE("Val_domain scalar compound arithmetic covers zero, negative, NaN, and infinity",
          "[val_domain][compound]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    Index point(dom->get_nbr_points());

    Val_domain zero(dom);
    zero.set_zero();
    zero += 0.0;
    REQUIRE_FALSE(zero.check_if_zero());
    REQUIRE(zero(point) == 0.0);
    zero -= -2.0;
    REQUIRE(zero(point) == 2.0);
    zero *= -3.0;
    REQUIRE(zero(point) == -6.0);
    zero /= 2.0;
    REQUIRE(zero(point) == -3.0);

    Val_domain non_finite(dom);
    non_finite = std::numeric_limits<double>::infinity();
    non_finite += 1.0;
    REQUIRE(std::isinf(non_finite(point)));
    non_finite = std::numeric_limits<double>::quiet_NaN();
    non_finite *= -1.0;
    REQUIRE(std::isnan(non_finite(point)));
}

TEST_CASE("Val_domain compound division refuses a zero divisor before modifying the dividend",
          "[val_domain][compound]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    Index point(dom->get_nbr_points());
    Val_domain numerator(dom), zero_divisor(dom);
    numerator = -4.0;
    zero_divisor.set_zero();

    REQUIRE_THROWS(numerator /= zero_divisor);
    REQUIRE(numerator(point) == -4.0);

    Val_domain zero_numerator(dom);
    zero_numerator.set_zero();
    zero_numerator /= zero_divisor;
    REQUIRE(zero_numerator.check_if_zero());
}

TEST_CASE("Val_domain assignment preserves values across reused and changed representations",
          "[val_domain][assignment]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    Val_domain source(dom), target(dom);

    source = 2.0;
    target = -1.0;
    target = source;
    Index point(source.get_conf().get_dimensions());
    REQUIRE_THAT(target(point), WithinAbs(2.0, 1e-12));

    // Reassign an equal-shape configuration buffer, then cross from
    // coefficient storage back to configuration storage.
    source = 3.0;
    source.std_base();
    target = source;
    REQUIRE_THAT(target(point), WithinAbs(3.0, 1e-12));

    source.coef();
    target.coef();
    target = source;
    target.coef_i();
    REQUIRE_THAT(target(point), WithinAbs(3.0, 1e-12));
    REQUIRE(target.get_base() == source.get_base());

    source = 4.0;
    target = source;
    REQUIRE_THAT(target(point), WithinAbs(4.0, 1e-12));

    source.set_zero();
    target = source;
    REQUIRE(target.check_if_zero());

    // Self-assignment must preserve a populated field.
    target = 5.0;
    target = target;
    REQUIRE_THAT(target(point), WithinAbs(5.0, 1e-12));
}

TEST_CASE("Val_domain derivative caches survive copy and move ownership transfers",
          "[val_domain][derivative][move]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    Val_domain original(dom);
    original = dom->get_cart(1);
    original.std_base_x_cart();
    const Val_domain expected = original.der_var(1);

    Val_domain copied(original);
    REQUIRE_THAT(diffmax(expected, copied.der_var(1)), WithinAbs(0.0, 0.0));

    Val_domain moved(std::move(copied));
    REQUIRE_THAT(diffmax(expected, moved.der_var(1)), WithinAbs(0.0, 0.0));

    Val_domain assigned(dom);
    assigned = std::move(moved);
    REQUIRE_THAT(diffmax(expected, assigned.der_var(1)), WithinAbs(0.0, 0.0));
}

TEST_CASE("Val_domain scalar compound scaling invalidates cached derivatives",
          "[val_domain][derivative-cache][compound]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    auto coordinate = [dom]() {
        Val_domain value(dom);
        value = dom->get_cart(1);
        value.std_base_x_cart();
        return value;
    };

    SECTION("multiplication") {
        Val_domain actual = coordinate();
        actual.der_var(1);
        actual.der_abs(1);
        actual *= 3.0;

        Val_domain fresh = coordinate();
        fresh *= 3.0;
        CHECK_THAT(diffmax(fresh, actual), WithinAbs(0.0, 1e-12));
        CHECK_THAT(diffmax(fresh.der_var(1), actual.der_var(1)), WithinAbs(0.0, 1e-12));
        CHECK_THAT(diffmax(fresh.der_abs(1), actual.der_abs(1)), WithinAbs(0.0, 1e-12));
    }

    SECTION("division") {
        Val_domain actual = coordinate();
        actual.der_var(1);
        actual.der_abs(1);
        actual /= 4.0;

        Val_domain fresh = coordinate();
        fresh /= 4.0;
        CHECK_THAT(diffmax(fresh, actual), WithinAbs(0.0, 1e-12));
        CHECK_THAT(diffmax(fresh.der_var(1), actual.der_var(1)), WithinAbs(0.0, 1e-12));
        CHECK_THAT(diffmax(fresh.der_abs(1), actual.der_abs(1)), WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("Val_domain zero-lhs subtraction does not reuse unnegated rhs derivative caches",
          "[val_domain][derivative-cache][compound]") {
    auto space = make_space();
    const Domain* dom = space.get_domain(0);
    auto coordinate = [dom]() {
        Val_domain value(dom);
        value = dom->get_cart(1);
        value.std_base_x_cart();
        return value;
    };

    Val_domain rhs = coordinate();
    Val_domain expected_var = rhs.der_var(1);
    Val_domain expected_abs = rhs.der_abs(1);
    expected_var *= -1.0;
    expected_abs *= -1.0;

    Val_domain lhs(dom);
    lhs.set_zero();
    lhs -= rhs;

    CHECK_THAT(diffmax(-rhs, lhs), WithinAbs(0.0, 1e-12));
    CHECK_THAT(diffmax(expected_var, lhs.der_var(1)), WithinAbs(0.0, 1e-12));
    CHECK_THAT(diffmax(expected_abs, lhs.der_abs(1)), WithinAbs(0.0, 1e-12));

    Val_domain uncached_rhs = coordinate();
    Val_domain uncached_lhs(dom);
    uncached_lhs.set_zero();
    uncached_lhs -= uncached_rhs;
    CHECK_THAT(diffmax(expected_var, uncached_lhs.der_var(1)), WithinAbs(0.0, 1e-12));
    CHECK_THAT(diffmax(expected_abs, uncached_lhs.der_abs(1)), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Val_domain inline derivative storage supports four-dimensional ownership transfers",
          "[val_domain][derivative][four-dimensional]") {
    Dim_array resolution(4);
    resolution.set(0) = 5;
    resolution.set(1) = 5;
    resolution.set(2) = 4;
    resolution.set(3) = 5;
    Domain_fourD_periodic_nucleus domain(0, CHEB_TYPE, 1.0, 0.5, resolution);

    Val_domain original(&domain);
    original = 2.0;

    Val_domain copied(original);
    Val_domain moved(std::move(copied));
    Val_domain assigned(&domain);
    assigned = std::move(moved);
    Index point(assigned.get_conf().get_dimensions());
    REQUIRE_THAT(assigned(point), WithinAbs(2.0, 0.0));
}

TEST_CASE("one-domain Tensor results copy, move, and lazily materialize inactive domains",
          "[tensor][one-domain][storage]") {
    auto space = make_space();
    Scalar lhs(space);
    Scalar rhs(space);
    lhs = 2.0;
    rhs = 3.0;
    lhs.std_base();
    rhs.std_base();

    Tensor result = add_one_dom(1, lhs, rhs);
    result.std_base();

    Tensor saved_result = add_one_dom(1, lhs, rhs);
    MemorySink sink;
    saved_result.save(sink);
    REQUIRE_FALSE(sink.buffer().empty());

    Tensor copied(result);
    Tensor moved(std::move(copied));
    REQUIRE(moved().get_domain(0) == space.get_domain(0));

    moved.set().set_domain(0) = 7.0;
    Index active_point(moved()(1).get_conf().get_dimensions());
    Index materialized_point(moved()(0).get_conf().get_dimensions());
    REQUIRE_THAT(moved()(1)(active_point), WithinAbs(5.0, 1e-12));
    REQUIRE_THAT(moved()(0)(materialized_point), WithinAbs(7.0, 1e-12));

    REQUIRE_THROWS(add_one_dom(space.get_nbr_domains(), lhs, rhs));
}
