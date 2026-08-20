// Verify integ_volume for Spheric_nosym domains against analytic values.
//
// For a constant integrand f=1 the volume integral is the geometric volume:
//   nucleus [0, R1]      -> (4/3) pi R1^3
//   shell   [R1, R2]     -> (4/3) pi (R2^3 - R1^3)
// This is the canonical correctness check for integ_volume: it verifies the
// full r^2 sin(theta) dr dtheta dphi measure is assembled correctly.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"

#include <cmath>

using namespace Kadath;
using Catch::Matchers::WithinRel;

namespace {

constexpr double R1 = 1.0;
constexpr double R2 = 3.0;

// nucleus + 1 shell + compact outer
Space_spheric_nosym build_space()
{
    Point center(3);
    center.set(1) = 0.; center.set(2) = 0.; center.set(3) = 0.;

    Dim_array res(3);
    res.set(0) = 9; res.set(1) = 9; res.set(2) = 8;

    Dim_array bd(1);
    bd.set(0) = 2;
    Array<double> bounds(bd);
    bounds.set(0) = R1;
    bounds.set(1) = R2;

    return Space_spheric_nosym(CHEB_TYPE, center, res, bounds, /*withzec=*/true);
}

// Integrate f over a single domain using the domain's integ_volume.
double domain_integ(const Space_spheric_nosym& space, const Scalar& f, int dom)
{
    return space.get_domain(dom)->integ_volume(f(dom));
}

} // namespace

TEST_CASE("nucleus_nosym integ_volume: f=1 gives 4pi/3 * R^3", "[spheric_nosym_integ_volume]")
{
    auto space = build_space();
    Scalar f(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d)
        f.set_domain(d) = 1.;
    f.std_base();

    const double expected = (4. * M_PI / 3.) * std::pow(R1, 3);
    REQUIRE_THAT(domain_integ(space, f, 0), WithinRel(expected, 1e-10));
}

TEST_CASE("shell_nosym integ_volume: f=1 gives 4pi/3 * (R2^3 - R1^3)", "[spheric_nosym_integ_volume]")
{
    auto space = build_space();
    Scalar f(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d)
        f.set_domain(d) = 1.;
    f.std_base();

    const double expected = (4. * M_PI / 3.) * (std::pow(R2, 3) - std::pow(R1, 3));
    REQUIRE_THAT(domain_integ(space, f, 1), WithinRel(expected, 1e-10));
}

// Regression test for the anti-symmetric (even-j SIN) integ weight bug.
// f = cos(theta) is equatorially anti-symmetric (z-odd), so its volume
// integral over any sphere/shell is exactly 0:
//   INT cos(theta) r^2 sin(theta) dr dtheta dphi
//     = 2pi INT r^2 dr * INT_0^pi cos(theta) sin(theta) dtheta = 0.
// Inside integ_volume, mult_sin_theta(cos theta) = (1/2) sin(2 theta): a
// NONZERO even-j (j=2) SIN coefficient. The old code applied weight 2/j to
// every j, so this even-j term contributed a spurious nonzero value. The
// f=1 fixtures above cannot catch this (they only excite j<=1). With the
// even-j skip in place, the result must be ~0.
TEST_CASE("nucleus_nosym integ_volume: z-odd cos(theta) integrand integrates to 0",
          "[spheric_nosym_integ_volume]")
{
    auto space = build_space();
    Scalar one(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d)
        one.set_domain(d) = 1.;
    one.std_base();
    one.coef();

    const Val_domain z_odd(one(0).mult_cos_theta());
    REQUIRE(std::abs(space.get_domain(0)->integ_volume(z_odd)) < 1e-10);
}

TEST_CASE("shell_nosym integ_volume: z-odd cos(theta) integrand integrates to 0",
          "[spheric_nosym_integ_volume]")
{
    auto space = build_space();
    Scalar one(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d)
        one.set_domain(d) = 1.;
    one.std_base();
    one.coef();

    const Val_domain z_odd(one(1).mult_cos_theta());
    REQUIRE(std::abs(space.get_domain(1)->integ_volume(z_odd)) < 1e-10);
}
