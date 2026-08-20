// Per-domain POLAR (angular) p-refinement across a non-conforming seam.
//
// Sibling to test_bns_per_domain_p_refinement.cpp, which covers the RADIAL
// case.  Here neighbouring spherical domains carry different ntheta, and the
// shared interface is coupled with Eq_matching_non_std (add_eq_matching_non_std)
// instead of the conforming point-to-point Eq_matching.  Standard matching
// pairs boundary angular modes 1:1 and therefore needs equal ntheta on both
// sides; the non-standard matching interpolates (absol_to_num_bound +
// Base_spectral::summation), so the ntheta jump is allowed.
//
// Two MPI-free, Newton-free probes via System_of_eqs::sec_member():
//   1. square system + zero residual on u == 1 (well-posedness of the mixed-
//      ntheta matching), mirroring the radial gate's require_square pattern;
//   2. an axisymmetric (purely polar) manufactured field crosses the ntheta
//      jump with no O(1) seam penalty -- its mixed-grid residual stays close to
//      the conforming uniform-fine residual.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace Kadath;

namespace {

// nucleus(0->1) + inner shell(1->3) + outer shell(3->8) + compactified(8->inf).
Array<double> ball_bounds()
{
    Array<double> bounds(3);
    bounds.set(0) = 1.0;
    bounds.set(1) = 3.0;
    bounds.set(2) = 8.0;
    return bounds;
}

// Build a Space_spheric with the requested per-domain (nr, ntheta, nphi) layout
// through the Dim_array** constructor.  Returned as a prvalue (guaranteed copy
// elision), so the local Dim_array storage need only outlive construction.
Space_spheric make_spheric(const std::vector<std::array<int, 3>>& res_per_domain)
{
    Point center(3);
    center.set(1) = 0.;
    center.set(2) = 0.;
    center.set(3) = 0.;

    std::vector<Dim_array> dims;
    dims.reserve(res_per_domain.size());
    for (const auto& r : res_per_domain) {
        Dim_array d(3);
        d.set(0) = r[0];
        d.set(1) = r[1];
        d.set(2) = r[2];
        dims.push_back(d);
    }
    std::vector<Dim_array*> ptrs;
    ptrs.reserve(dims.size());
    for (auto& d : dims)
        ptrs.push_back(&d);

    return Space_spheric(CHEB_TYPE, center, ptrs.data(), ball_bounds());
}

double max_abs_residual(System_of_eqs& syst)
{
    const Array<double> residual = syst.sec_member();
    double m = 0.0;
    for (int i = 0; i < syst.get_nbr_conditions(); ++i)
        m = std::max(m, std::abs(residual.get_data()[i]));
    return m;
}

// Which non-conforming matching primitive couples the seams.
enum class Matching : std::uint8_t { NonStd, Import };

struct Probe {
    int unknowns = 0;
    int conditions = 0;
    double residual = 0.0;
    bool square() const { return unknowns == conditions; }
};

// Manufactured Laplace with the continuity SPLIT across each internal seam
// (value from the inner domain's outer face, derivative from the outer domain's
// inner face). u == 1 solves lap(u) = 0 at any ntheta (l = 0), so a square
// system must drive every residual row -- matching rows included -- to roundoff.
//
// The two primitives differ in HOW they count seam conditions, and that is the
// whole point of this gate:
//   * Eq_matching_non_std  -> nbr_points_boundary  (collocation-POINT space)
//   * Eq_matching_import   -> nbr_conditions_boundary (tau-COEFFICIENT space)
// The bulk Eq_inside and the conforming Eq_matching both count in coefficient
// space, so only `import` is consistent with them and stays square under an
// ntheta jump. `non_std` is off by (#points - #coefs) per face and only appears
// square at the resolution where that per-face defect cancels globally.
Probe probe(const std::vector<std::array<int, 3>>& res, Matching matching)
{
    Space_spheric space = make_spheric(res);
    const int ndom = space.get_nbr_domains();

    Scalar u(space);
    u = 1.;
    u.std_base();

    System_of_eqs syst(space);
    syst.add_var("u", u);
    const int last = ndom - 1;
    for (int d = 0; d < last; ++d) {
        syst.add_eq_inside(d, "lap(u) = 0", -1, nullptr, "u");
        if (matching == Matching::Import) {
            syst.add_eq_matching_import(d, OUTER_BC, "u", -1, nullptr, "u");
            syst.add_eq_matching_import(d + 1, INNER_BC, "dn(u)", -1, nullptr, "u");
        } else {
            syst.add_eq_matching_non_std(d, OUTER_BC, "u", -1, nullptr, "u");
            syst.add_eq_matching_non_std(d + 1, INNER_BC, "dn(u)", -1, nullptr, "u");
        }
    }
    syst.add_eq_inside(last, "lap(u) = 0", -1, nullptr, "u");
    syst.add_eq_bc(last, OUTER_BC, "u = 1");

    Probe p;
    p.residual = max_abs_residual(syst);  // triggers the condition count
    p.unknowns = syst.get_nbr_unknowns();
    p.conditions = syst.get_nbr_conditions();
    return p;
}

} // namespace

TEST_CASE("Per-domain polar refinement stays square only with coefficient-space (import) matching",
          "[spheric][angular-p]")
{
    // np deliberately NOT 4: at np=4 the non_std point/coef defect cancels and
    // hides the bug. With per-domain ntheta and np=6 the two primitives diverge.
    const std::vector<std::array<int, 3>> mixed = {{13, 11, 6}, {13, 9, 6}, {13, 7, 6}, {13, 5, 6}};

    const Probe im = probe(mixed, Matching::Import);
    const Probe ns = probe(mixed, Matching::NonStd);
    INFO("import unk/cond = " << im.unknowns << "/" << im.conditions
                              << ", non_std unk/cond = " << ns.unknowns << "/" << ns.conditions);

    // import counts seam conditions in coefficient space, consistent with the
    // bulk -> square + zero residual for an arbitrary per-domain ntheta jump.
    CHECK(im.square());
    CHECK(im.residual < 1.e-10);

    // non_std counts in point space -> NOT square once np != 4 exposes the
    // per-face defect. This is why the BNS conversion must use import, and why
    // the earlier np=4 "proof" was a resolution coincidence.
    CHECK_FALSE(ns.square());
}

TEST_CASE("Import matching is square across several per-domain polar layouts",
          "[spheric][angular-p]")
{
    const auto require_square = [](const std::vector<std::array<int, 3>>& res) {
        const Probe p = probe(res, Matching::Import);
        INFO("unk/cond = " << p.unknowns << "/" << p.conditions << ", residual = " << p.residual);
        CHECK(p.square());
        CHECK(p.residual < 1.e-10);
    };

    SECTION("uniform control") { require_square({{13, 9, 6}, {13, 9, 6}, {13, 9, 6}, {13, 9, 6}}); }
    SECTION("fine inner, coarse outer") { require_square({{13, 9, 6}, {13, 9, 6}, {13, 5, 6}, {13, 5, 6}}); }
    SECTION("coarse inner, fine outer") { require_square({{13, 5, 6}, {13, 5, 6}, {13, 9, 6}, {13, 9, 6}}); }
    SECTION("different ntheta per domain") { require_square({{13, 11, 6}, {13, 9, 6}, {13, 7, 6}, {13, 5, 6}}); }
}
