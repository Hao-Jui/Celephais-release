// Phase 4 bucket 1: Spheric_adapted_nosym base setters — unified COS/SIN/COSSIN.
//
// Locks two contracts on the parity-relaxed Adapted basis dispatch:
//   1. Theta-axis basis (bases_1d[1]) is unified COS / SIN / COSSIN, never
//      parity-restricted COS_EVEN / COS_ODD / SIN_EVEN / SIN_ODD. This is
//      the structural signature that the per-Domain set_cheb_base /
//      set_legendre_base / set_*_base_*_spher / set_anti_*_base / set_*_r_base
//      families have been collapsed to the NONSYM convention.
//   2. Forward + inverse spectral round-trip on a z-symmetric scalar
//      preserves collocation values to FP precision. Confirms the unified
//      basis dispatch is consistent end-to-end (set_*_base → coef → coef_i).
//
// References:
//   - src/Domain/Spheric_nosym/domain_shell_nosym.cpp:266-321
//     (production NONSYM precedent in non-Adapted Spheric_nosym)
//   - projects/plane_sym_withdraw/lorene_nonsym_extraction_verdict.md
//     (Phase 4 plan + bucket definitions)

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
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
    res.set(0) = 5;
    res.set(1) = 5;
    res.set(2) = 4;
    return res;
}

Array<double> make_bounds()
{
    Dim_array dim(1);
    dim.set(0) = 3;
    Array<double> bounds(dim);
    bounds.set(0) = 1.0;
    bounds.set(1) = 2.0;
    bounds.set(2) = 10.0;
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

TEST_CASE("Spheric_adapted_nosym theta basis is unified COS/SIN, never parity-restricted",
          "[spheric_adapted_nosym_base_setters]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    Space_spheric_adapted_nosym space(CHEB_TYPE, center, res, bounds);

    Scalar u(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        u.set_domain(d) = 1.0;
    }
    u.std_base();

    // For each domain, scan bases_1d[1] over all phi-modes. NONSYM contract:
    // every entry is COS / SIN / COSSIN, never *_EVEN / *_ODD.
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
    }
}

TEST_CASE("Spheric_adapted_nosym scalar coef forward+inverse round-trip preserves constant",
          "[spheric_adapted_nosym_base_setters]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    Space_spheric_adapted_nosym space(CHEB_TYPE, center, res, bounds);

    Scalar f(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        f.set_domain(d) = 2.5;
    }
    f.std_base();
    f.coef();
    f.coef_i();

    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const Val_domain& vd = f(d);
        Index idx(vd.get_conf().get_dimensions());
        REQUIRE_THAT(vd(idx), WithinAbs(2.5, 1e-12));
    }
}

TEST_CASE("Spheric_adapted_nosym theta collocation spans the full NONSYM interval",
          "[spheric_adapted_nosym_base_setters]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    Space_spheric_adapted_nosym space(CHEB_TYPE, center, res, bounds);

    for (int d = 1; d <= 2; ++d) {
        const Array<double> theta = space.get_domain(d)->get_coloc(2);
        REQUIRE(theta.get_nbr() == res(1));
        CHECK_THAT(theta(0), WithinAbs(0.0, 1e-14));
        CHECK_THAT(theta(theta.get_nbr() - 1), WithinAbs(M_PI, 1e-14));
    }
}
