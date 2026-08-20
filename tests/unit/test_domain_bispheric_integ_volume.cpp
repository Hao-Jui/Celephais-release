// Verify Domain::integ_volume for the bispherical domains.
//
// A Space_bispheric (two nuclei + five bispherical fillers + a compactified
// exterior) tiles the ball of radius rext with domains 0..6; domain 7 is the
// r > rext compactified zone. For a constant integrand the sum of the volume
// integrals over domains 0..6 must therefore equal the volume of that ball:
//
//     sum_{d=0}^{6} integ_volume(1) = (4/3) pi rext^3.
//
// Domains 0,1 are spheres (the already-trusted spherical integ_volume); the
// five bispherical domains 2..6 exercise the implementation under test, so the
// global identity is an independent end-to-end check of all three bispherical
// kinds at once. A second check uses an x-odd integrand, whose integral over the
// (origin-centred) ball vanishes by symmetry.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Domain/bispheric.hpp"

#include <cmath>

using namespace Kadath;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

namespace {

constexpr double kDistance = 10.0;
constexpr double kR1 = 1.5;
constexpr double kR2 = 1.5;
constexpr double kRext = 20.0;
constexpr int kRes = 13;

// nuclei (0,1) + bispherical fillers (2..6) + compactified exterior (7)
Space_bispheric build_space()
{
    return Space_bispheric(CHEB_TYPE, kDistance, kR1, kR2, kRext, kRes);
}

// Sum of integ_volume over the bounded domains 0..6 (everything except the
// compactified zone, whose volume is infinite).
double bounded_volume_integral(const Space_bispheric& space, const Scalar& f)
{
    double total = 0.;
    for (int d = 0; d < space.get_nbr_domains() - 1; ++d)
        total += f(d).integ_volume();
    return total;
}

} // namespace

TEST_CASE("bispheric Scalar val_point reproduces a constant field", "[bispheric_integ_volume]")
{
    // Sanity: spectral point evaluation (used by integ_volume) must reconstruct
    // a constant to machine precision in the bispherical zones.
    auto space = build_space();
    Scalar one(space);
    one = 1.;
    one.std_base();

    Point p(3);
    p.set(1) = 0.5 * kDistance; // between the two stars, inside a bispherical domain
    p.set(2) = 3.0;
    p.set(3) = 1.0;
    REQUIRE_THAT(one.val_point(p), WithinAbs(1.0, 1e-10));
}

TEST_CASE("bispheric integ_volume: bounded domains tile the ball", "[bispheric_integ_volume]")
{
    auto space = build_space();
    Scalar one(space);
    one = 1.;
    one.std_base();

    const double expected = (4. / 3.) * M_PI * std::pow(kRext, 3);
    // The Gauss-Legendre quadrature is decoupled from the field collocation grid
    // and resolves the peaked a^3 sin(chi)/D^3 measure independently, so a unit
    // field integrates to the geometric volume to ~1e-6.
    REQUIRE_THAT(bounded_volume_integral(space, one), WithinRel(expected, 1e-5));
}

TEST_CASE("bispheric integ_volume: x-odd integrand cancels over the ball", "[bispheric_integ_volume]")
{
    auto space = build_space();
    Scalar fx(space);
    // f = x (the first Cartesian coordinate), odd about the origin-centred ball.
    for (int d = 0; d < space.get_nbr_domains(); ++d)
        fx.set_domain(d) = space.get_domain(d)->get_cart(1);
    fx.std_base();

    const double expected = 0.;
    REQUIRE_THAT(bounded_volume_integral(space, fx), WithinAbs(expected, 1e-5 * std::pow(kRext, 4)));
}

TEST_CASE("bispheric integ_volume: phi-varying integrand integrates over the full azimuth",
          "[bispheric_integ_volume]")
{
    // The constant and x-odd fields above are both phi-independent, so they
    // never exercise the [0, 2*pi) azimuthal quadrature. f = y^2 carries genuine
    // azimuthal content (y = a sin(chi) cos(phi)/D, so y^2 ~ cos^2(phi) =
    // (1 + cos 2phi)/2, a k=0 plus k=2 mode). Over the origin-centred ball,
    //     INT y^2 dV = (1/3) INT r^2 dV = (1/3)(4 pi rext^5 / 5) = (4/15) pi rext^5.
    // A phi range mistakenly collapsed to [0, pi] would halve the cos^2 average
    // and miss the answer by a factor of two, which the constant/x-odd checks
    // cannot see.
    auto space = build_space();
    Scalar fy2(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const Val_domain& yy = space.get_domain(d)->get_cart(2);
        fy2.set_domain(d) = yy * yy;
    }
    fy2.std_base();

    const double expected = (4. / 15.) * M_PI * std::pow(kRext, 5);
    // Observed ~1e-7 at kRes=13 (set by the field's spectral truncation of y^2,
    // not the quadrature: the unit-field geometric measure integrates to ~1e-12).
    REQUIRE_THAT(bounded_volume_integral(space, fy2), WithinRel(expected, 1e-5));
}
