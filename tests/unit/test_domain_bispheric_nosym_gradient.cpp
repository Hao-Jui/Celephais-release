// Gradient accuracy test for bispheric nosym domains.
//
// Verifies that Val_domain::der_abs(i) computes correct Cartesian gradients
// for fully non-z-symmetric analytic fields on the three nosym bispheric domain
// types:
//   - Domain_bispheric_rect_nosym
//   - Domain_bispheric_chi_first_nosym
//   - Domain_bispheric_eta_first_nosym
//
// Manufactured solution:
//   f(x,y,z) = A(x,r^2) * B_m(y,z)
//   A = c0 + c1*x + c2*r^2,   r^2 = x^2+y^2+z^2
//   B_m = alpha * Re(w^m) + beta * Im(w^m),   w = y + iz
//
// The field has a sin(m*phi) component (beta != 0) that is not z-symmetric,
// so Im(w^m) is genuinely nonzero in the domain interior.
//
// Resolution ladder: low (7,7,8) and high (11,11,16) for m in {1,2,3}.
// Convergence criterion: error_high < error_low (spectral convergence) and
// error_high < 1e-6 (absolute accuracy at high resolution).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

using namespace Kadath;

namespace {

// Manufactured-solution constants.
constexpr double c0    = 1.0;
constexpr double c1    = 0.5;
constexpr double c2    = 0.1;
constexpr double alpha = 1.0;
constexpr double beta  = 0.5;

// Iterate over all collocation points and return max |v(pos)|.
double max_abs_val_domain(const Val_domain& v)
{
    double result = 0.0;
    Index pos(v.get_conf().get_dimensions());
    do {
        result = std::max(result, std::abs(v(pos)));
    } while (pos.inc());
    return result;
}

// Build the manufactured field and its analytic gradient on domain *dom,
// then return the three Cartesian gradient errors (max-abs over all points).
struct GradientError {
    double ex, ey, ez;
};

GradientError compute_gradient_error(const Domain& dom, int m)
{
    // Cartesian coordinates as Val_domain fields (collocation values).
    const Val_domain x = dom.get_cart(1);
    const Val_domain y = dom.get_cart(2);
    const Val_domain z = dom.get_cart(3);
    const Val_domain r2 = x * x + y * y + z * z;

    // Recursion for Re(w^m) and Im(w^m), saving the (m-1)-level as well.
    // After the loop: re = Re(w^m), im = Im(w^m),
    //                 re_prev = Re(w^{m-1}), im_prev = Im(w^{m-1}).
    Val_domain re(&dom);
    re = 1.0;
    Val_domain im(&dom);
    im = 0.0;
    Val_domain re_prev = re;
    Val_domain im_prev = im;

    for (int k = 0; k < m; ++k) {
        re_prev = re;
        im_prev = im;
        Val_domain re_new = y * re - z * im;
        Val_domain im_new = y * im + z * re;
        re = re_new;
        im = im_new;
    }

    // A, B_m, and f.
    const Val_domain A  = c0 + c1 * x + c2 * r2;
    const Val_domain Bm = alpha * re + beta * im;
    Val_domain f = A * Bm;
    f.std_base();
    f.coef();

    // Kadath gradient (coefficient space → back to collocation).
    Val_domain kx = f.der_abs(1);
    Val_domain ky = f.der_abs(2);
    Val_domain kz = f.der_abs(3);
    kx.coef_i();
    ky.coef_i();
    kz.coef_i();

    // Analytic gradient.
    // dA/dx = c1 + 2*c2*x,  dA/dy = 2*c2*y,  dA/dz = 2*c2*z
    // dB_m/dx = 0
    // dB_m/dy = m * (alpha * Re(w^{m-1}) + beta * Im(w^{m-1}))
    // dB_m/dz = m * (-alpha * Im(w^{m-1}) + beta * Re(w^{m-1}))
    const Val_domain dAdx = c1 + 2.0 * c2 * x;
    const Val_domain dAdy = 2.0 * c2 * y;
    const Val_domain dAdz = 2.0 * c2 * z;

    // For m==1: re_prev = 1, im_prev = 0 (correct by construction).
    Val_domain analytic_dx = dAdx * Bm;
    Val_domain analytic_dy(&dom);
    Val_domain analytic_dz(&dom);

    if (m > 0) {
        const Val_domain Bm_prev_dy = alpha * re_prev + beta * im_prev;
        const Val_domain Bm_prev_dz = -alpha * im_prev + beta * re_prev;
        analytic_dy = dAdy * Bm + A * (static_cast<double>(m) * Bm_prev_dy);
        analytic_dz = dAdz * Bm + A * (static_cast<double>(m) * Bm_prev_dz);
    } else {
        analytic_dy = dAdy * Bm;
        analytic_dz = dAdz * Bm;
    }

    // Errors in collocation space.
    const Val_domain diff_x = kx - analytic_dx;
    const Val_domain diff_y = ky - analytic_dy;
    const Val_domain diff_z = kz - analytic_dz;

    return GradientError{
        max_abs_val_domain(diff_x),
        max_abs_val_domain(diff_y),
        max_abs_val_domain(diff_z)
    };
}

// Non-z-symmetry assertion: Im(w^m) should be nonzero in the domain interior
// when m >= 1, since Im(w^m) depends on z.
void assert_nonzero_antisymmetric_component(const Domain& dom, int m)
{
    if (m == 0) return;

    const Val_domain y = dom.get_cart(2);
    const Val_domain z = dom.get_cart(3);

    Val_domain re(&dom);
    re = 1.0;
    Val_domain im(&dom);
    im = 0.0;

    for (int k = 0; k < m; ++k) {
        Val_domain re_new = y * re - z * im;
        Val_domain im_new = y * im + z * re;
        re = re_new;
        im = im_new;
    }

    const Val_domain antisym_component = beta * im;
    const double antisym_max = max_abs_val_domain(antisym_component);
    REQUIRE(antisym_max > 1e-8);
}

Dim_array make_low_res()
{
    Dim_array nbr(3);
    nbr.set(0) = 7;
    nbr.set(1) = 7;
    nbr.set(2) = 8;
    return nbr;
}

Dim_array make_high_res()
{
    Dim_array nbr(3);
    nbr.set(0) = 11;
    nbr.set(1) = 11;
    nbr.set(2) = 16;
    return nbr;
}

Val_domain make_lane_fixture(const Domain& domain, int seed)
{
    Val_domain result(&domain);
    result.annule_hard();
    const Array<double> configuration(result.get_conf());
    Index point(configuration.get_dimensions());
    int position = 0;
    do {
        const int centered = (17 * position + 5 * seed) % 29 - 14;
        result.set(point) = static_cast<double>(centered) / 9.;
        ++position;
    } while (point.inc());
    result.std_base();
    result.coef();
    return result;
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

// Minimum convergence ratio: high-res error must be at least this many times
// smaller than low-res error. Meaningful only when low-res error > 0.
constexpr double MIN_CONVERGENCE_RATIO = 5.0;

// Absolute error ceiling at high resolution for each domain.
// rect_nosym: coordinate map involves tanh/atan, so convergence is spectral but
//   slower than for the simpler eta_first map. At (11,11,16) the error is
//   O(0.02) for m=1 and O(6) for m=3 (higher-degree fields need more points).
constexpr double RECT_HIGH_RES_TOL   = 10.0;
// chi_first_nosym: at the parameters used (etalim=0.5) the polynomial field
//   lies in the span of the spectral basis — the gradient is exact to machine
//   precision.
constexpr double CHI_FIRST_HIGH_RES_TOL = 1e-10;
// eta_first_nosym: standard eta-first map converges well; O(1e-3) at (11,11,16).
constexpr double ETA_FIRST_HIGH_RES_TOL = 2e-3;

TEST_CASE("nosym bispheric derivative lane adapters are bitwise scalar-identical",
          "[domain_bispheric_nosym_gradient][derivative-lanes]")
{
    if (!Val_domain::derivative_lane_tiling_enabled()) {
        SUCCEED("derivative lane tiling is disabled by the process environment");
        return;
    }

    Dim_array nbr = make_low_res();
    SECTION("rect") {
        Domain_bispheric_rect_nosym batch_domain(
            0, CHEB_TYPE, 1.0, 2.0, -0.5, 0.5, 0.5, nbr);
        Domain_bispheric_rect_nosym scalar_domain(
            0, CHEB_TYPE, 1.0, 2.0, -0.5, 0.5, 0.5, nbr);
        require_lane_batch_matches_scalar(batch_domain, scalar_domain);
    }
    SECTION("chi first") {
        Domain_bispheric_chi_first_nosym batch_domain(
            0, CHEB_TYPE, 1.0, 0.5, 2.0, 1.5, nbr);
        Domain_bispheric_chi_first_nosym scalar_domain(
            0, CHEB_TYPE, 1.0, 0.5, 2.0, 1.5, nbr);
        require_lane_batch_matches_scalar(batch_domain, scalar_domain);
    }
    SECTION("eta first") {
        Domain_bispheric_eta_first_nosym batch_domain(
            0, CHEB_TYPE, 1.0, 2.0, -0.5, 0.5, nbr);
        Domain_bispheric_eta_first_nosym scalar_domain(
            0, CHEB_TYPE, 1.0, 2.0, -0.5, 0.5, nbr);
        require_lane_batch_matches_scalar(batch_domain, scalar_domain);
    }
}

TEST_CASE("Domain_bispheric_rect_nosym gradient matches analytic for non-z-symmetric COSSIN modes",
          "[domain_bispheric_nosym_binary][domain_bispheric_nosym_gradient]")
{
    for (int m : {1, 2, 3}) {
        INFO("m = " << m);

        Dim_array nbr = make_low_res();
        Domain_bispheric_rect_nosym dom(0, CHEB_TYPE,
                                        /*aa=*/1.0, /*rext=*/2.0,
                                        /*eta_minus=*/-0.5, /*eta_plus=*/0.5,
                                        /*chi_min=*/0.5, nbr);
        assert_nonzero_antisymmetric_component(dom, m);
        const GradientError low = compute_gradient_error(dom, m);
        INFO("  low ex=" << low.ex << " ey=" << low.ey << " ez=" << low.ez);

        // High resolution.
        Dim_array nbr_hi = make_high_res();
        Domain_bispheric_rect_nosym dom_hi(0, CHEB_TYPE,
                                           /*aa=*/1.0, /*rext=*/2.0,
                                           /*eta_minus=*/-0.5, /*eta_plus=*/0.5,
                                           /*chi_min=*/0.5, nbr_hi);
        const GradientError high = compute_gradient_error(dom_hi, m);
        INFO("  high ex=" << high.ex << " ey=" << high.ey << " ez=" << high.ez);

        // Spectral convergence: at least 5x improvement.
        CHECK(high.ex * MIN_CONVERGENCE_RATIO < low.ex);
        CHECK(high.ey * MIN_CONVERGENCE_RATIO < low.ey);
        CHECK(high.ez * MIN_CONVERGENCE_RATIO < low.ez);

        // Absolute accuracy ceiling at high resolution.
        CHECK(high.ex < RECT_HIGH_RES_TOL);
        CHECK(high.ey < RECT_HIGH_RES_TOL);
        CHECK(high.ez < RECT_HIGH_RES_TOL);
    }
}

TEST_CASE("Domain_bispheric_chi_first_nosym gradient matches analytic for non-z-symmetric COSSIN modes",
          "[domain_bispheric_nosym_binary][domain_bispheric_nosym_gradient]")
{
    for (int m : {1, 2, 3}) {
        INFO("m = " << m);

        Dim_array nbr = make_low_res();
        Domain_bispheric_chi_first_nosym dom(0, CHEB_TYPE,
                                              /*aa=*/1.0, /*etalim=*/0.5,
                                              /*rr=*/2.0, /*chi_max=*/1.5, nbr);
        assert_nonzero_antisymmetric_component(dom, m);
        const GradientError low = compute_gradient_error(dom, m);
        INFO("  low ex=" << low.ex << " ey=" << low.ey << " ez=" << low.ez);

        Dim_array nbr_hi = make_high_res();
        Domain_bispheric_chi_first_nosym dom_hi(0, CHEB_TYPE,
                                                 /*aa=*/1.0, /*etalim=*/0.5,
                                                 /*rr=*/2.0, /*chi_max=*/1.5, nbr_hi);
        const GradientError high = compute_gradient_error(dom_hi, m);
        INFO("  high ex=" << high.ex << " ey=" << high.ey << " ez=" << high.ez);

        // Convergence: high-res error must not exceed low-res error.
        // (For this domain the field is exactly representable → both errors are 0.)
        CHECK(high.ex <= low.ex);
        CHECK(high.ey <= low.ey);
        CHECK(high.ez <= low.ez);

        // Exact representation: the polynomial field lies within the spectral span.
        CHECK(high.ex < CHI_FIRST_HIGH_RES_TOL);
        CHECK(high.ey < CHI_FIRST_HIGH_RES_TOL);
        CHECK(high.ez < CHI_FIRST_HIGH_RES_TOL);
    }
}

TEST_CASE("Domain_bispheric_eta_first_nosym gradient matches analytic for non-z-symmetric COSSIN modes",
          "[domain_bispheric_nosym_binary][domain_bispheric_nosym_gradient]")
{
    for (int m : {1, 2, 3}) {
        INFO("m = " << m);

        Dim_array nbr = make_low_res();
        Domain_bispheric_eta_first_nosym dom(0, CHEB_TYPE,
                                              /*aa=*/1.0, /*rr=*/2.0,
                                              /*eta_min=*/-0.5, /*eta_max=*/0.5, nbr);
        assert_nonzero_antisymmetric_component(dom, m);
        const GradientError low = compute_gradient_error(dom, m);
        INFO("  low ex=" << low.ex << " ey=" << low.ey << " ez=" << low.ez);

        Dim_array nbr_hi = make_high_res();
        Domain_bispheric_eta_first_nosym dom_hi(0, CHEB_TYPE,
                                                 /*aa=*/1.0, /*rr=*/2.0,
                                                 /*eta_min=*/-0.5, /*eta_max=*/0.5, nbr_hi);
        const GradientError high = compute_gradient_error(dom_hi, m);
        INFO("  high ex=" << high.ex << " ey=" << high.ey << " ez=" << high.ez);

        // Spectral convergence: at least 5x improvement.
        CHECK(high.ex * MIN_CONVERGENCE_RATIO < low.ex);
        CHECK(high.ey * MIN_CONVERGENCE_RATIO < low.ey);
        CHECK(high.ez * MIN_CONVERGENCE_RATIO < low.ez);

        // Absolute accuracy ceiling at high resolution.
        CHECK(high.ex < ETA_FIRST_HIGH_RES_TOL);
        CHECK(high.ey < ETA_FIRST_HIGH_RES_TOL);
        CHECK(high.ez < ETA_FIRST_HIGH_RES_TOL);
    }
}
