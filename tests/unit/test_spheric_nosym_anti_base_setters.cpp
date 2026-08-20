// Phase 7: Spheric_nosym anti-symmetric base setters — NONSYM collapse.
//
// Locks contracts on the parity-relaxed Spheric_nosym anti-base dispatch:
//   1. set_anti_cheb_base / set_anti_legendre_base populate a defined base
//      with COSSIN phi-axis and unified COS/SIN theta-axis (never the
//      parity-restricted COS_EVEN / COS_ODD / SIN_EVEN / SIN_ODD enums).
//   2. Nucleus_nosym r-axis: anti-base is structurally identical to
//      sym-base. In the full NONSYM basis l_quant = j already spans both
//      equatorial parities, and regularity at the origin still requires the
//      radial parity to follow j.
//   3. Shell_nosym + Compact_nosym: the same delegation holds because
//      non-origin domains carry no r-parity selection.
//
// References:
//   - src/Domain/Spheric_nosym/domain_nucleus_nosym.cpp:251-292
//     (NONSYM Cheb/Leg dispatch; l_quant = j full-basis collapse)
//   - src/Domain/Spheric_adapted_nosym/domain_shell_outer_adapted_nosym.cpp:725
//     (Phase 4 NONSYM precedent for unified-basis anti-base)
//   - projects/plane_sym_withdraw/lorene_nonsym_extraction_verdict.md
//     (Phase 7 plan + recipe derivation)

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <cmath>

using namespace Kadath;
using Catch::Matchers::WithinAbs;

namespace {

Point make_origin()
{
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;
    return center;
}

Dim_array make_resolution()
{
    Dim_array res(3);
    res.set(0) = 9;
    res.set(1) = 9;
    res.set(2) = 8;
    return res;
}

Array<double> make_bounds()
{
    Dim_array bd(1);
    bd.set(0) = 2;
    Array<double> bounds(bd);
    bounds.set(0) = 1.0;
    bounds.set(1) = 3.0;
    return bounds;
}

bool is_parity_restricted_theta_basis(int baset)
{
    return baset == COS_EVEN || baset == COS_ODD ||
           baset == SIN_EVEN || baset == SIN_ODD;
}

bool is_unified_theta_basis(int baset)
{
    return baset == COS || baset == SIN || baset == COSSIN;
}

} // namespace

TEST_CASE("Spheric_nosym anti-base theta axis is unified COS/SIN, never parity-restricted",
          "[spheric_nosym_anti_base_setters]")
{
    Space_spheric_nosym space(CHEB_TYPE, make_origin(), make_resolution(), make_bounds(), true);

    // Build a Scalar, dispatch to anti-base on every domain, and verify
    // theta-axis enums on bases_1d[1] are all unified COS/SIN/COSSIN.
    Scalar u(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        u.set_domain(d) = 1.0;
        u.set_domain(d).std_anti_base();
    }

    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const Val_domain& vd = u(d);
        const Base_spectral& base = vd.get_base();
        REQUIRE(base.is_def());

        const Array<int>* theta_bases = base.get_base_1d(1);
        REQUIRE(theta_bases != nullptr);

        const int n_phi = theta_bases->get_nbr();
        for (int k = 0; k < n_phi; ++k) {
            const int baset = theta_bases->get_data()[k];
            INFO("domain=" << d << " k=" << k << " baset=" << baset);
            CHECK_FALSE(is_parity_restricted_theta_basis(baset));
            CHECK(is_unified_theta_basis(baset));
        }

        const Array<int>* phi_bases = base.get_base_1d(2);
        REQUIRE(phi_bases != nullptr);
        REQUIRE(phi_bases->get_data()[0] == COSSIN);
    }
}

TEST_CASE("Nucleus_nosym set_anti_cheb_base delegates to set_cheb_base",
          "[spheric_nosym_anti_base_setters]")
{
    // Direct domain construction to compare sym vs anti dispatch byte-by-byte.
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Domain_nucleus_nosym dom(0, CHEB_TYPE, 1.0, make_origin(), nbr);

    // Construct a Val_domain on this domain, then exercise std_base() and
    // std_anti_base() (in turn) to populate the base both ways.
    Val_domain sym_field(&dom);
    sym_field = 1.0;
    sym_field.std_base();

    Val_domain anti_field(&dom);
    anti_field = 1.0;
    anti_field.std_anti_base();

    const Base_spectral& sym = sym_field.get_base();
    const Base_spectral& anti = anti_field.get_base();
    REQUIRE(sym.is_def());
    REQUIRE(anti.is_def());

    // Both should expose unified COS/SIN theta. Sample a few (j, k) pairs and
    // verify the origin regularity rule is unchanged: radial parity follows
    // l_quant = j.
    const Array<int>* sym_r = sym.get_base_1d(0);
    const Array<int>* anti_r = anti.get_base_1d(0);
    REQUIRE(sym_r != nullptr);
    REQUIRE(anti_r != nullptr);

    // Index space for bases_1d[0] is (j, k); enumerate first several.
    Index idx(sym_r->get_dimensions());
    Index aidx(anti_r->get_dimensions());

    // Sample (j=0,k=0): both j-even → CHEB_EVEN.
    idx.set(0) = 0;
    idx.set(1) = 0;
    aidx.set(0) = 0;
    aidx.set(1) = 0;
    CHECK((*sym_r)(idx) == CHEB_EVEN);
    CHECK((*anti_r)(aidx) == CHEB_EVEN);

    // Sample (j=1,k=0): both j-odd → CHEB_ODD.
    idx.set(0) = 1;
    idx.set(1) = 0;
    aidx.set(0) = 1;
    aidx.set(1) = 0;
    CHECK((*sym_r)(idx) == CHEB_ODD);
    CHECK((*anti_r)(aidx) == CHEB_ODD);

    // Sample (j=2,k=2): both j-even → CHEB_EVEN.
    idx.set(0) = 2;
    idx.set(1) = 2;
    aidx.set(0) = 2;
    aidx.set(1) = 2;
    CHECK((*sym_r)(idx) == CHEB_EVEN);
    CHECK((*anti_r)(aidx) == CHEB_EVEN);

    // Theta axis (bases_1d[1]) at k=0 (m=0, m%2==0): both should be COS.
    const Array<int>* sym_t = sym.get_base_1d(1);
    const Array<int>* anti_t = anti.get_base_1d(1);
    REQUIRE(sym_t != nullptr);
    REQUIRE(anti_t != nullptr);
    CHECK(sym_t->get_data()[0] == COS);
    CHECK(anti_t->get_data()[0] == COS);
}

TEST_CASE("Nucleus_nosym set_anti_legendre_base delegates to set_legendre_base",
          "[spheric_nosym_anti_base_setters]")
{
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Domain_nucleus_nosym dom(0, LEG_TYPE, 1.0, make_origin(), nbr);

    Val_domain sym_field(&dom);
    sym_field = 1.0;
    sym_field.std_base();

    Val_domain anti_field(&dom);
    anti_field = 1.0;
    anti_field.std_anti_base();

    const Base_spectral& sym = sym_field.get_base();
    const Base_spectral& anti = anti_field.get_base();
    REQUIRE(sym.is_def());
    REQUIRE(anti.is_def());

    const Array<int>* sym_r = sym.get_base_1d(0);
    const Array<int>* anti_r = anti.get_base_1d(0);
    REQUIRE(sym_r != nullptr);
    REQUIRE(anti_r != nullptr);

    Index idx(sym_r->get_dimensions());
    Index aidx(anti_r->get_dimensions());

    // (j=0, k=0): both LEG_EVEN.
    idx.set(0) = 0;
    idx.set(1) = 0;
    aidx.set(0) = 0;
    aidx.set(1) = 0;
    CHECK((*sym_r)(idx) == LEG_EVEN);
    CHECK((*anti_r)(aidx) == LEG_EVEN);

    // (j=1, k=0): both LEG_ODD.
    idx.set(0) = 1;
    idx.set(1) = 0;
    aidx.set(0) = 1;
    aidx.set(1) = 0;
    CHECK((*sym_r)(idx) == LEG_ODD);
    CHECK((*anti_r)(aidx) == LEG_ODD);
}

TEST_CASE("Nucleus_nosym z Cartesian component dispatches through anti-base",
          "[spheric_nosym_anti_base_setters]")
{
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Domain_nucleus_nosym dom(0, CHEB_TYPE, 1.0, make_origin(), nbr);

    Val_domain x_component(&dom);
    x_component = 1.0;
    x_component.std_base_x_cart();

    Val_domain z_component(&dom);
    z_component = 1.0;
    z_component.std_base_z_cart();

    const Base_spectral& x_base = x_component.get_base();
    const Base_spectral& z_base = z_component.get_base();
    REQUIRE(x_base.is_def());
    REQUIRE(z_base.is_def());

    const Array<int>* x_r = x_base.get_base_1d(0);
    const Array<int>* z_r = z_base.get_base_1d(0);
    REQUIRE(x_r != nullptr);
    REQUIRE(z_r != nullptr);

    Index idx(x_r->get_dimensions());
    idx.set(0) = 0;
    idx.set(1) = 0;
    CHECK((*x_r)(idx) == CHEB_EVEN);
    CHECK((*z_r)(idx) == CHEB_EVEN);

    idx.set(0) = 1;
    idx.set(1) = 0;
    CHECK((*x_r)(idx) == CHEB_ODD);
    CHECK((*z_r)(idx) == CHEB_ODD);
}

TEST_CASE("Shell_nosym set_anti_cheb_base delegates to set_cheb_base (no origin)",
          "[spheric_nosym_anti_base_setters]")
{
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Domain_shell_nosym dom(1, CHEB_TYPE, 1.0, 3.0, make_origin(), nbr);

    Val_domain sym_field(&dom);
    sym_field = 1.0;
    sym_field.std_base();

    Val_domain anti_field(&dom);
    anti_field = 1.0;
    anti_field.std_anti_base();

    const Base_spectral& sym = sym_field.get_base();
    const Base_spectral& anti = anti_field.get_base();
    REQUIRE(sym.is_def());
    REQUIRE(anti.is_def());

    // r-axis: both should be plain CHEB (no parity selection in shell).
    const Array<int>* sym_r = sym.get_base_1d(0);
    const Array<int>* anti_r = anti.get_base_1d(0);
    REQUIRE(sym_r != nullptr);
    REQUIRE(anti_r != nullptr);

    Index idx(sym_r->get_dimensions());
    Index aidx(anti_r->get_dimensions());
    idx.set(0) = 0;
    idx.set(1) = 0;
    aidx.set(0) = 0;
    aidx.set(1) = 0;
    CHECK((*sym_r)(idx) == CHEB);
    CHECK((*anti_r)(aidx) == CHEB);

    idx.set(0) = 3;
    idx.set(1) = 2;
    aidx.set(0) = 3;
    aidx.set(1) = 2;
    CHECK((*sym_r)(idx) == CHEB);
    CHECK((*anti_r)(aidx) == CHEB);
}

TEST_CASE("Shell_nosym set_anti_legendre_base delegates to set_legendre_base",
          "[spheric_nosym_anti_base_setters]")
{
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Domain_shell_nosym dom(1, LEG_TYPE, 1.0, 3.0, make_origin(), nbr);

    Val_domain anti_field(&dom);
    anti_field = 1.0;
    anti_field.std_anti_base();

    const Base_spectral& anti = anti_field.get_base();
    REQUIRE(anti.is_def());

    const Array<int>* anti_r = anti.get_base_1d(0);
    REQUIRE(anti_r != nullptr);
    Index aidx(anti_r->get_dimensions());
    aidx.set(0) = 0;
    aidx.set(1) = 0;
    CHECK((*anti_r)(aidx) == LEG);
}

TEST_CASE("Compact_nosym set_anti_cheb_base delegates to set_cheb_base (no origin)",
          "[spheric_nosym_anti_base_setters]")
{
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Domain_compact_nosym dom(2, CHEB_TYPE, 3.0, make_origin(), nbr);

    Val_domain anti_field(&dom);
    anti_field = 1.0;
    anti_field.std_anti_base();

    const Base_spectral& anti = anti_field.get_base();
    REQUIRE(anti.is_def());

    const Array<int>* anti_r = anti.get_base_1d(0);
    REQUIRE(anti_r != nullptr);
    Index aidx(anti_r->get_dimensions());
    aidx.set(0) = 0;
    aidx.set(1) = 0;
    CHECK((*anti_r)(aidx) == CHEB);
}

TEST_CASE("Compact_nosym set_anti_legendre_base delegates to set_legendre_base",
          "[spheric_nosym_anti_base_setters]")
{
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 9;
    nbr.set(2) = 8;
    Domain_compact_nosym dom(2, LEG_TYPE, 3.0, make_origin(), nbr);

    Val_domain anti_field(&dom);
    anti_field = 1.0;
    anti_field.std_anti_base();

    const Base_spectral& anti = anti_field.get_base();
    REQUIRE(anti.is_def());

    const Array<int>* anti_r = anti.get_base_1d(0);
    REQUIRE(anti_r != nullptr);
    Index aidx(anti_r->get_dimensions());
    aidx.set(0) = 0;
    aidx.set(1) = 0;
    CHECK((*anti_r)(aidx) == LEG);
}
