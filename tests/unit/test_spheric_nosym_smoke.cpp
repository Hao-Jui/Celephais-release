// Smoke test for Spheric_nosym thin port (no plane symmetry).
//
// Verifies that the ported Spheric_nosym Domain + Space family compiles, runs,
// and exercises the spectral round-trip on a z-symmetric scalar field.
// z-symmetric content embeds trivially in the unrestricted COSSIN basis, so
// the spectral primitives can be exercised at zero physics risk.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

using namespace Kadath;
using Catch::Matchers::WithinAbs;

namespace {

Space_spheric_nosym build_space()
{
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;

    Dim_array res(3);
    res.set(0) = 9;
    res.set(1) = 9;
    res.set(2) = 8;

    Dim_array bd(1);
    bd.set(0) = 2;
    Array<double> bounds(bd);
    bounds.set(0) = 1.0;
    bounds.set(1) = 3.0;

    // 3 domains: nucleus + 1 shell + compactified outer.
    return Space_spheric_nosym(CHEB_TYPE, center, res, bounds, true);
}

template <typename T>
void check_binary_round_trip(T& original)
{
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    T restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

void require_coefficient_bytes_equal(const Val_domain& actual, const Val_domain& expected)
{
    REQUIRE(actual.check_if_zero() == expected.check_if_zero());
    REQUIRE(actual.get_base() == expected.get_base());
    actual.coef();
    expected.coef();
    const Array<double> actual_coefficients(actual.get_coef());
    const Array<double> expected_coefficients(expected.get_coef());
    REQUIRE(actual_coefficients.get_nbr() == expected_coefficients.get_nbr());
    REQUIRE(std::memcmp(actual_coefficients.get_data(),
                        expected_coefficients.get_data(),
                        actual_coefficients.get_nbr() * sizeof(double)) == 0);
}

Val_domain make_lane_fixture(const Domain& domain, int seed)
{
    Val_domain result(&domain);
    result.annule_hard();
    const Array<double> configuration(result.get_conf());
    Index point(configuration.get_dimensions());
    int position = 0;
    do {
        const int centered = (13 * position + 7 * seed) % 31 - 15;
        result.set(point) = static_cast<double>(centered) / 11.;
        ++position;
    } while (point.inc());
    result.std_base();
    result.coef();
    return result;
}

void require_lane_batch_matches_scalar(const Domain& batch_domain, const Domain& scalar_domain)
{
    Val_domain first_batch(make_lane_fixture(batch_domain, 1));
    Val_domain second_batch(make_lane_fixture(batch_domain, 2));
    Val_domain first_scalar(make_lane_fixture(scalar_domain, 1));
    Val_domain second_scalar(make_lane_fixture(scalar_domain, 2));

    std::vector<Val_domain> first_expected;
    std::vector<Val_domain> second_expected;
    first_expected.reserve(3);
    second_expected.reserve(3);
    for (int axis = 1; axis <= 3; ++axis) {
        first_expected.push_back(first_scalar.der_abs(axis));
        second_expected.push_back(second_scalar.der_abs(axis));
    }

    const std::array<const Val_domain*, 2> fields = {&first_batch, &second_batch};
    Val_domain::prepare_der_abs_batch(fields);
    for (int axis = 1; axis <= 3; ++axis) {
        require_coefficient_bytes_equal(
            first_batch.der_abs(axis), first_expected[static_cast<std::size_t>(axis - 1)]);
        require_coefficient_bytes_equal(
            second_batch.der_abs(axis), second_expected[static_cast<std::size_t>(axis - 1)]);
    }
}

} // namespace

TEST_CASE("Domain_nucleus_nosym binary round-trip", "[spheric_nosym_smoke]")
{
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;
    Domain_nucleus_nosym dom(0, CHEB_TYPE, 1.0, center, nbr);
    check_binary_round_trip(dom);
}

TEST_CASE("Domain_shell_nosym binary round-trip", "[spheric_nosym_smoke]")
{
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;
    Domain_shell_nosym dom(1, CHEB_TYPE, 1.0, 3.0, center, nbr);
    check_binary_round_trip(dom);
}

TEST_CASE("Domain_compact_nosym binary round-trip", "[spheric_nosym_smoke]")
{
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;
    Domain_compact_nosym dom(2, CHEB_TYPE, 3.0, center, nbr);
    check_binary_round_trip(dom);
}

TEST_CASE("Space_spheric_nosym construction", "[spheric_nosym_smoke]")
{
    auto space = build_space();
    // 3 domains: nucleus (idx 0), 1 shell (idx 1), compactified outer (idx 2).
    REQUIRE(space.get_nbr_domains() == 3);
    REQUIRE(space.get_ndim() == 3);
    REQUIRE(space.get_domain(0) != nullptr);
    REQUIRE(space.get_domain(1) != nullptr);
    REQUIRE(space.get_domain(2) != nullptr);
}

TEST_CASE("Space_spheric_nosym binary round-trip", "[spheric_nosym_smoke]")
{
    auto space = build_space();
    MemorySink sink1;
    space.save(sink1);
    MemorySource source(sink1.buffer());
    Space_spheric_nosym restored(source, true);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
    REQUIRE(restored.get_nbr_domains() == 3);
}

TEST_CASE("Spheric_nosym scalar z-symmetric round-trip",
          "[spheric_nosym_smoke]")
{
    auto space = build_space();
    Scalar f(space);

    // Assign f = 1 everywhere and verify the forward + inverse spectral
    // transform round-trips through the COSSIN basis.
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        f.set_domain(d) = 1.0;
    }
    f.std_base(); // Configures the standard basis via set_cheb_base on each domain.

    // Trigger forward + inverse spectral transform in every domain.
    f.coef();
    f.coef_i();

    // Read back the value at a configuration point; for f ≡ 1 it must round-trip.
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const Val_domain& vd = f(d);
        Index idx(vd.get_conf().get_dimensions());
        REQUIRE_THAT(vd(idx), WithinAbs(1.0, 1e-12));
    }
}

TEST_CASE("Spheric_nosym set_cheb_base produces COSSIN dispatch",
          "[spheric_nosym_smoke]")
{
    auto space = build_space();
    // The headcpp.hpp constants COSSIN, COS, SIN, CHEB_EVEN, CHEB_ODD are
    // available; verify that constructing a Scalar exercises the spectral
    // base-setting path without aborting. Successful coef() means
    // set_cheb_base + the COSSIN dispatch in mult() worked.
    Scalar f(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        f.set_domain(d) = 2.5;
    }
    f.std_base();
    f.coef(); // forces forward FFT on every domain.

    // Check the coefficient base is populated for each domain.
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const Val_domain& vd = f(d);
        REQUIRE(vd.get_base().is_def());
    }
}

TEST_CASE("nucleus_nosym derivative lane batch is bitwise scalar-identical",
          "[spheric_nosym_smoke][derivative-lanes]")
{
    if (!Val_domain::derivative_lane_tiling_enabled()) {
        SUCCEED("derivative lane tiling is disabled by the process environment");
        return;
    }

    auto space = build_space();
    const Domain& domain = *space.get_domain(0);
    const Val_domain x = domain.get_cart(1);
    const Val_domain y = domain.get_cart(2);
    const Val_domain z = domain.get_cart(3);

    Val_domain first = x * x + 0.25 * y + z * z;
    Val_domain second = x * y - 0.5 * z + y * y;
    first.std_base();
    second.std_base();
    first.coef();
    second.coef();

    Val_domain first_scalar(first);
    Val_domain second_scalar(second);
    std::vector<Val_domain> first_expected;
    std::vector<Val_domain> second_expected;
    first_expected.reserve(3);
    second_expected.reserve(3);
    for (int axis = 1; axis <= 3; ++axis) {
        first_expected.push_back(first_scalar.der_abs(axis));
        second_expected.push_back(second_scalar.der_abs(axis));
    }

    Val_domain zero(&domain);
    zero.set_zero();
    const std::array<const Val_domain*, 3> fields = {&first, &second, &zero};
    Val_domain::prepare_der_abs_batch(fields);
    for (int axis = 1; axis <= 3; ++axis) {
        require_coefficient_bytes_equal(first.der_abs(axis), first_expected[static_cast<std::size_t>(axis - 1)]);
        require_coefficient_bytes_equal(second.der_abs(axis), second_expected[static_cast<std::size_t>(axis - 1)]);
        REQUIRE(zero.der_abs(axis).check_if_zero());
    }
}

TEST_CASE("derivative lane batch rejects repeated field ownership",
          "[spheric_nosym_smoke][derivative-lanes]")
{
    if (!Val_domain::derivative_lane_tiling_enabled()) {
        SUCCEED("derivative lane tiling is disabled by the process environment");
        return;
    }

    auto space = build_space();
    const Domain& domain = *space.get_domain(0);
    Val_domain field = domain.get_cart(1) + domain.get_cart(2);
    field.std_base();
    field.coef();

    const std::array<const Val_domain*, 2> repeated = {&field, &field};
    REQUIRE_THROWS(Val_domain::prepare_der_abs_batch(repeated));

    std::array<const Val_domain*, Term_eq::max_derivative_lanes + 1> oversized{};
    oversized.fill(&field);
    REQUIRE_THROWS(Val_domain::prepare_der_var_batch(oversized));
    REQUIRE_THROWS(Val_domain::prepare_der_abs_batch(oversized));
}

TEST_CASE("compact_nosym derivative lane batch is bitwise scalar-identical",
          "[spheric_nosym_smoke][derivative-lanes]")
{
    if (!Val_domain::derivative_lane_tiling_enabled()) {
        SUCCEED("derivative lane tiling is disabled by the process environment");
        return;
    }

    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;
    Domain_compact_nosym batch_domain(0, CHEB_TYPE, 3.0, center, nbr);
    Domain_compact_nosym scalar_domain(0, CHEB_TYPE, 3.0, center, nbr);
    require_lane_batch_matches_scalar(batch_domain, scalar_domain);
}
