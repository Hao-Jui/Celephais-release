// der_normal contract for translated bispheric nosym charts.
//
// Space_three_body builds its child aggregate from the five bispheric nosym
// domains translated by set_origin_x(child_origin_x). The spherical boundaries
// of those charts (OUTER_BC sphere, INNER_BC pole spheres) are centered on the
// *chart* origin, not the global origin, so der_normal must build the unit
// normal from chart-local coordinates:
//
//   n_outer = ((x - origin_x)/r_ext, y/r_ext, z/r_ext)
//   n_inner = ((x - origin_x - xc)/rr, y/rr, z/rr)
//
// Two checks per domain type / boundary:
//   1. Translation invariance: der_normal of the same local field on a
//      translated chart matches the untranslated chart to FP-roundoff.
//   2. Analytic volume identity on the translated chart:
//      f = |X - C|^2 about the sphere center C gives dn(f) = 2*|X - C|^2 / R.
//
// Boundaries built from der_var (CHI_ONE_BC, ETA_PLUS_BC, ETA_MINUS_BC) are
// trivially translation-invariant and are not exercised here.
//
// The chart parameters come from a geometrically consistent bispheric map
// (symmetric two-sphere configuration, so the coordinate scale aa is closed
// form). Inconsistent standalone parameters leave part of the collocation
// grid outside the chart where bound_eta/bound_chi go NaN; every value is
// therefore REQUIREd finite before being compared.

#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <algorithm>
#include <cmath>

namespace Kadath {
double chi_lim_eta(double eta, double rext, double a, double chi_c);
}

using namespace Kadath;

namespace {

constexpr double c0 = 1.0;
constexpr double c1 = 0.5;
constexpr double c2 = 0.1;
constexpr double alpha = 1.0;
constexpr double beta = 0.5;
constexpr int m_mode = 2;

constexpr double TRANSLATION = 35.0;
constexpr double INVARIANCE_TOL = 1e-10;

// Symmetric bispheric collection: two exposed spheres of radius r_body whose
// centers are dist apart, wrapped by an outer sphere of radius r_out.
struct SymmetricBisphericMap {
    double aa;
    double eta_pole;   // |eta| of the two pole spheres
    double eta_lim;    // transition eta of the rect/eta_first split
    double chi_lim;
};

SymmetricBisphericMap make_symmetric_map(double r_body, double dist, double r_out)
{
    SymmetricBisphericMap map;
    map.aa = std::sqrt(0.25 * dist * dist - r_body * r_body);
    map.eta_pole = std::asinh(map.aa / r_body);
    const double chi_c = 2.0 * std::atan(map.aa / r_out);
    const double eta_c = std::log((1.0 + r_out / map.aa) / (r_out / map.aa - 1.0));
    map.eta_lim = eta_c / 2.0;
    map.chi_lim = chi_lim_eta(map.eta_lim, r_out, map.aa, chi_c);
    return map;
}

constexpr double R_BODY = 1.0;
constexpr double DIST = 4.0;
constexpr double R_OUT = 4.0;

double finite_max_abs(const Val_domain& v)
{
    double result = 0.0;
    Index pos(v.get_conf().get_dimensions());
    do {
        const double val = v(pos);
        REQUIRE(std::isfinite(val));
        result = std::max(result, std::abs(val));
    } while (pos.inc());
    return result;
}

double finite_max_abs_diff(const Val_domain& a, const Val_domain& b)
{
    double result = 0.0;
    Index pos(a.get_conf().get_dimensions());
    do {
        const double va = a(pos);
        const double vb = b(pos);
        REQUIRE(std::isfinite(va));
        REQUIRE(std::isfinite(vb));
        result = std::max(result, std::abs(va - vb));
    } while (pos.inc());
    return result;
}

// Manufactured field expressed in chart-local coordinates so the translated
// and untranslated charts carry the same physical field.
template <class DomType>
Val_domain make_local_field(const DomType& dom)
{
    const Val_domain x_loc = dom.get_cart(1) - dom.get_origin_x();
    const Val_domain y = dom.get_cart(2);
    const Val_domain z = dom.get_cart(3);
    const Val_domain r2 = x_loc * x_loc + y * y + z * z;

    Val_domain re(&dom);
    re = 1.0;
    Val_domain im(&dom);
    im = 0.0;
    for (int k = 0; k < m_mode; ++k) {
        Val_domain re_new = y * re - z * im;
        Val_domain im_new = y * im + z * re;
        re = re_new;
        im = im_new;
    }

    Val_domain f = (c0 + c1 * x_loc + c2 * r2) * (alpha * re + beta * im);
    f.std_base();
    f.coef();
    return f;
}

template <class DomType>
void check_translation_invariance(const DomType& dom_base, const DomType& dom_shift, int bound)
{
    Val_domain f_base = make_local_field(dom_base);
    Val_domain f_shift = make_local_field(dom_shift);

    Val_domain dn_base = dom_base.der_normal(f_base, bound);
    Val_domain dn_shift = dom_shift.der_normal(f_shift, bound);
    dn_base.coef_i();
    dn_shift.coef_i();

    const double scale = finite_max_abs(dn_base);
    REQUIRE(scale > 1e-3); // degenerate dn would make the comparison vacuous

    const double diff = finite_max_abs_diff(dn_base, dn_shift);
    INFO("bound = " << bound << "  diff = " << diff << "  scale = " << scale);
    CHECK(diff < INVARIANCE_TOL * scale);
}

// dn(|X - C|^2) = 2 |X - C|^2 / R as a volume identity, with C the sphere
// center of the given boundary and R its radius. The residual is pure
// spectral-truncation error of der_abs, so the translated chart must match
// the untranslated chart's residual; a center error would add O(D/R) on top.
template <class DomType>
double analytic_sphere_error(const DomType& dom, int bound, double center_x_loc, double sphere_radius)
{
    const Val_domain dx = dom.get_cart(1) - (dom.get_origin_x() + center_x_loc);
    const Val_domain y = dom.get_cart(2);
    const Val_domain z = dom.get_cart(3);

    Val_domain f = dx * dx + y * y + z * z;
    f.std_base();
    f.coef();

    Val_domain dn = dom.der_normal(f, bound);
    dn.coef_i();

    const Val_domain expected = 2.0 * (dx * dx + y * y + z * z) / sphere_radius;
    return finite_max_abs_diff(dn, expected);
}

Dim_array make_res()
{
    Dim_array nbr(3);
    nbr.set(0) = 11;
    nbr.set(1) = 11;
    nbr.set(2) = 16;
    return nbr;
}

} // namespace

TEST_CASE("Domain_bispheric_chi_first_nosym der_normal is translation invariant and matches sphere normals",
          "[domain_bispheric_nosym_binary][domain_bispheric_nosym_der_normal]")
{
    const Dim_array nbr = make_res();
    const SymmetricBisphericMap map = make_symmetric_map(R_BODY, DIST, R_OUT);
    const double eta_pole = -map.eta_pole; // minus-side pole, as in Space_three_body

    Domain_bispheric_chi_first_nosym dom_base(0, CHEB_TYPE, map.aa, eta_pole, R_OUT, map.chi_lim, nbr);
    Domain_bispheric_chi_first_nosym dom_shift(0, CHEB_TYPE, map.aa, eta_pole, R_OUT, map.chi_lim, nbr);
    dom_shift.set_origin_x(TRANSLATION);

    check_translation_invariance(dom_base, dom_shift, OUTER_BC);
    check_translation_invariance(dom_base, dom_shift, INNER_BC);

    const double xc = map.aa * cosh(eta_pole) / sinh(eta_pole);
    const double rr = map.aa / sinh(std::fabs(eta_pole));
    const double err_outer_base = analytic_sphere_error(dom_base, OUTER_BC, 0.0, R_OUT);
    const double err_outer_shift = analytic_sphere_error(dom_shift, OUTER_BC, 0.0, R_OUT);
    const double err_inner_base = analytic_sphere_error(dom_base, INNER_BC, xc, rr);
    const double err_inner_shift = analytic_sphere_error(dom_shift, INNER_BC, xc, rr);
    CHECK(err_outer_shift < 1.01 * err_outer_base + 1e-9);
    CHECK(err_inner_shift < 1.01 * err_inner_base + 1e-9);
    CHECK(err_outer_base < 0.05);
    CHECK(err_inner_base < 0.05);
}

TEST_CASE("Domain_bispheric_rect_nosym der_normal is translation invariant and matches sphere normals",
          "[domain_bispheric_nosym_binary][domain_bispheric_nosym_der_normal]")
{
    const Dim_array nbr = make_res();
    const SymmetricBisphericMap map = make_symmetric_map(R_BODY, DIST, R_OUT);
    const double eta_pole = -map.eta_pole; // minus-side rect block

    Domain_bispheric_rect_nosym dom_base(0, CHEB_TYPE, map.aa, R_OUT, eta_pole, -map.eta_lim, map.chi_lim, nbr);
    Domain_bispheric_rect_nosym dom_shift(0, CHEB_TYPE, map.aa, R_OUT, eta_pole, -map.eta_lim, map.chi_lim, nbr);
    dom_shift.set_origin_x(TRANSLATION);

    check_translation_invariance(dom_base, dom_shift, OUTER_BC);
    check_translation_invariance(dom_base, dom_shift, INNER_BC);

    const double xc = map.aa * cosh(eta_pole) / sinh(eta_pole);
    const double rr = map.aa / sinh(std::fabs(eta_pole));
    const double err_outer_base = analytic_sphere_error(dom_base, OUTER_BC, 0.0, R_OUT);
    const double err_outer_shift = analytic_sphere_error(dom_shift, OUTER_BC, 0.0, R_OUT);
    const double err_inner_base = analytic_sphere_error(dom_base, INNER_BC, xc, rr);
    const double err_inner_shift = analytic_sphere_error(dom_shift, INNER_BC, xc, rr);
    CHECK(err_outer_shift < 1.01 * err_outer_base + 1e-9);
    CHECK(err_inner_shift < 1.01 * err_inner_base + 1e-9);
    CHECK(err_outer_base < 0.05);
    CHECK(err_inner_base < 0.05);
}

TEST_CASE("Domain_bispheric_eta_first_nosym der_normal is translation invariant and matches sphere normals",
          "[domain_bispheric_nosym_binary][domain_bispheric_nosym_der_normal]")
{
    const Dim_array nbr = make_res();
    const SymmetricBisphericMap map = make_symmetric_map(R_BODY, DIST, R_OUT);

    Domain_bispheric_eta_first_nosym dom_base(0, CHEB_TYPE, map.aa, R_OUT, -map.eta_lim, map.eta_lim, nbr);
    Domain_bispheric_eta_first_nosym dom_shift(0, CHEB_TYPE, map.aa, R_OUT, -map.eta_lim, map.eta_lim, nbr);
    dom_shift.set_origin_x(TRANSLATION);

    check_translation_invariance(dom_base, dom_shift, OUTER_BC);

    const double err_base = analytic_sphere_error(dom_base, OUTER_BC, 0.0, R_OUT);
    const double err_shift = analytic_sphere_error(dom_shift, OUTER_BC, 0.0, R_OUT);
    CHECK(err_shift < 1.01 * err_base + 1e-9);
    CHECK(err_base < 0.05);
}

// The byte-symmetric round-trip test cannot catch a field that save() and the
// BinarySource constructor both omit. der_normal(OUTER_BC) divides by r_ext,
// so a deserialized rect chart must reproduce the original's normal exactly.
TEST_CASE("Domain_bispheric_rect_nosym der_normal survives binary round-trip",
          "[domain_bispheric_nosym_binary][domain_bispheric_nosym_der_normal]")
{
    const Dim_array nbr = make_res();
    const SymmetricBisphericMap map = make_symmetric_map(R_BODY, DIST, R_OUT);
    const double eta_pole = -map.eta_pole;

    Domain_bispheric_rect_nosym original(0, CHEB_TYPE, map.aa, R_OUT, eta_pole, -map.eta_lim, map.chi_lim, nbr);

    MemorySink sink;
    original.save(sink);
    MemorySource source(sink.buffer());
    Domain_bispheric_rect_nosym restored(0, source);

    Val_domain f_orig = make_local_field(original);
    Val_domain f_rest = make_local_field(restored);
    Val_domain dn_orig = original.der_normal(f_orig, OUTER_BC);
    Val_domain dn_rest = restored.der_normal(f_rest, OUTER_BC);
    dn_orig.coef_i();
    dn_rest.coef_i();

    CHECK(finite_max_abs_diff(dn_orig, dn_rest) == 0.0);
}
