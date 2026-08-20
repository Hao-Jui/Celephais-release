// Plane-symmetry-withdraw parity gate for Spheric_nosym vs plain Space_spheric.
//
// Builds an identical scalar problem on:
//   - Space_spheric          : z-plane symmetry (theta in [0, pi/2]),
//                              full phi range [0, 2pi). Production analog
//                              along the SAME symmetry axis being withdrawn.
//   - Space_spheric_nosym    : no plane symmetry (theta in [0, pi]),
//                              full phi range. Recently ported in 644c40c7.
//
// (Note: Space_spheric_symphi is NOT the right counterpart — it carries an
// additional phi-quadrant symmetry orthogonal to the plane-sym axis we are
// trying to relax. Compare against the wrong-axis variant and the numbers
// diverge ~31x as expected.)
//
// Evaluates first-iteration residual via System_of_eqs::sec_member() and
// asserts:
//   1. nosym dispatch runs end-to-end without crashing (positive gate);
//   2. ||r||_inf of the residual agrees with the plain Space_spheric
//      counterpart to within FP — this confirms that the symmetric subspace
//      embeds correctly in the unrestricted COSSIN basis on z-symmetric input
//      (per the thin-port verdict prediction).
//
// nbr_conditions and nbr_unknowns differ between the two bases by basis
// organization (plain Spheric stores even/odd parity sectors separately;
// nosym stores a single unified COSSIN packing). The counts are captured
// as INFO diagnostics — not asserted.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <cmath>
#include <vector>

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
    // bounds=[1.0, 10.0] → 2 domains: nucleus + 1 shell.
    // Avoids the compactified outer domain so the parity check is restricted
    // to dispatch paths with direct plain-spheric ↔ nosym analogs.
    Dim_array dim(1);
    dim.set(0) = 2;
    Array<double> bounds(dim);
    bounds.set(0) = 1.0;
    bounds.set(1) = 10.0;
    return bounds;
}

double residual_infinity_norm(const Array<double>& residual)
{
    double max_abs = 0.0;
    const int n = residual.get_nbr();
    for (int i = 0; i < n; ++i) {
        const double v = std::abs(residual.get_data()[i]);
        if (v > max_abs) {
            max_abs = v;
        }
    }
    return max_abs;
}

// Build a minimal scalar problem on a 2-domain Spheric_* space (nucleus + 1
// shell). Set `u` to a constant per domain (intrinsically z-symmetric — all
// energy in the (0,0,0) spectral mode of either basis) and register the
// same algebraic + matching + boundary conditions used in
// test_system_of_eqs_row_classification.cpp.
template <typename SpaceT>
double assemble_residual_infinity_norm(SpaceT& space,
                                       double u_value_nucleus,
                                       double u_value_shell)
{
    System_of_eqs sys(space, 0, 1);

    Scalar u(space);
    u.set_domain(0) = u_value_nucleus;
    u.set_domain(1) = u_value_shell;
    u.std_base();
    u.coef();
    sys.add_var("u", u);

    sys.add_eq_inside(0, "u=0");
    sys.add_eq_order(1, 2, "u=0");
    sys.add_eq_bc(1, OUTER_BC, "u=0");
    sys.add_eq_matching(0, OUTER_BC, "u");

    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());

    return residual_infinity_norm(residual);
}

} // namespace

TEST_CASE("Spheric_nosym sec_member residual against plain Space_spheric on z-symmetric data",
          "[spheric_nosym_parity]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    // Space_spheric : z-plane sym, full phi → the natural production analog
    //                 differing from Spheric_nosym by exactly one symmetry
    //                 (the plane symmetry being withdrawn). Pass withzec=false
    //                 to suppress the compact outer (plain ctor default is true).
    Space_spheric plane_space(CHEB_TYPE, center, res, bounds, false);
    Space_spheric_nosym nosym_space(CHEB_TYPE, center, res, bounds, false);

    REQUIRE(plane_space.get_nbr_domains() == 2);
    REQUIRE(nosym_space.get_nbr_domains() == 2);

    const double u_nuc = 0.5;
    const double u_shell = 0.3;

    const double plane_norm =
        assemble_residual_infinity_norm(plane_space, u_nuc, u_shell);
    const double nosym_norm =
        assemble_residual_infinity_norm(nosym_space, u_nuc, u_shell);

    INFO("Space_spheric (plane sym)    ||r||_inf = " << plane_norm);
    INFO("Space_spheric_nosym          ||r||_inf = " << nosym_norm);

    // Dispatch reachability: nosym ran end-to-end + returned a real residual.
    REQUIRE(std::isfinite(plane_norm));
    REQUIRE(std::isfinite(nosym_norm));
    REQUIRE(plane_norm > 0.0);
    REQUIRE(nosym_norm > 0.0);

    // Strict parity: symmetric subspace embeds in COSSIN, so on z-symmetric
    // input (per-domain constants live entirely in the (0,0,0) mode shared
    // by both bases) the infinity norm of the residual must agree to FP.
    REQUIRE_THAT(nosym_norm, WithinAbs(plane_norm, 1e-12));
}

TEST_CASE("Spheric_nosym System_of_eqs reports finite nbr_conditions / nbr_unknowns",
          "[spheric_nosym_parity]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    Space_spheric plane_space(CHEB_TYPE, center, res, bounds, false);
    Space_spheric_nosym nosym_space(CHEB_TYPE, center, res, bounds, false);

    auto build_sys = [](auto& space) {
        System_of_eqs sys(space, 0, 1);
        Scalar u(space);
        u = 0.0;
        u.std_base();
        u.coef();
        sys.add_var("u", u);
        sys.add_eq_inside(0, "u=0");
        sys.add_eq_order(1, 2, "u=0");
        sys.add_eq_bc(1, OUTER_BC, "u=0");
        sys.add_eq_matching(0, OUTER_BC, "u");
        (void)sys.sec_member();  // realizes nbr_conditions from -1 sentinel.
        return std::pair{sys.get_nbr_conditions(), sys.get_nbr_unknowns()};
    };

    const auto [plane_nc, plane_nu] = build_sys(plane_space);
    const auto [nosym_nc, nosym_nu] = build_sys(nosym_space);

    INFO("Space_spheric       : nbr_conditions=" << plane_nc << " nbr_unknowns=" << plane_nu);
    INFO("Space_spheric_nosym : nbr_conditions=" << nosym_nc << " nbr_unknowns=" << nosym_nu);

    REQUIRE(plane_nc > 0);
    REQUIRE(plane_nu > 0);
    REQUIRE(nosym_nc > 0);
    REQUIRE(nosym_nu > 0);
}

TEST_CASE("spheric-nosym volume row descriptors preserve component and tau order",
          "[residual_row_descriptor][spheric_nosym]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();
    Space_spheric_nosym space(CHEB_TYPE, center, res, bounds, true);
    REQUIRE(space.get_nbr_domains() == 3);

    Base_tensor cartesian(space, CARTESIAN_BASIS);
    Vector bet(space, CON, cartesian);
    bet = 0.0;
    bet.std_base();
    bet.coef();

    System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
    system.add_var("bet", bet);
    system.add_eq_inside(0, "bet^i=0");
    for (int domain = 1; domain < space.get_nbr_domains(); ++domain)
        system.add_eq_order(domain, 2, "bet^i=0");
    (void)system.sec_member();

    std::vector<ResidualRowDescriptor> descriptors;
    REQUIRE(system.describe_residual_rows(descriptors));
    REQUIRE(descriptors.size() ==
            static_cast<std::size_t>(system.get_nbr_conditions()));

    std::size_t row = 0;
    for (int domain = 0; domain < space.get_nbr_domains(); ++domain) {
        const Array<int> counts =
            space.get_domain(domain)->nbr_conditions(bet, domain, 2);
        REQUIRE(counts.get_nbr() == 3);
        for (int component = 0; component < 3; ++component) {
            for (int local = 0; local < counts(component); ++local, ++row) {
                REQUIRE(row < descriptors.size());
                const ResidualRowDescriptor& descriptor = descriptors[row];
                REQUIRE(descriptor.available);
                CHECK(descriptor.equation_index == domain);
                REQUIRE(descriptor.sides.size() == 1);
                const ResidualRowCoordinate& coordinate =
                    descriptor.sides.front();
                CHECK(coordinate.domain == domain);
                CHECK(coordinate.component == component);
                CHECK(coordinate.phi_basis == COSSIN);
                CHECK(coordinate.phi_index >= 0);
                CHECK(coordinate.phi_index != 1);
            }
        }
    }
    CHECK(row == descriptors.size());
}

TEST_CASE("nucleus-nosym order-one volume descriptors match the order-two exporter span",
          "[residual_row_descriptor][spheric_nosym][order-one]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();
    Space_spheric_nosym space(CHEB_TYPE, center, res, bounds, false);

    Scalar u(space);
    u = 0.0;
    u.std_base();
    u.coef();

    const Domain* nucleus = space.get_domain(0);
    const Array<int> order_one_counts = nucleus->nbr_conditions(u, 0, 1);
    const Array<int> order_two_counts = nucleus->nbr_conditions(u, 0, 2);
    REQUIRE(order_one_counts.get_nbr() == 1);
    REQUIRE(order_two_counts.get_nbr() == 1);
    REQUIRE(order_one_counts(0) == order_two_counts(0));

    std::vector<ResidualRowDescriptor> order_one;
    std::vector<ResidualRowDescriptor> order_two;
    const bool order_one_available = nucleus->describe_volume_residual_rows(
        u, 0, 1, order_one_counts, -1, nullptr, order_one);
    const bool order_two_available = nucleus->describe_volume_residual_rows(
        u, 0, 2, order_two_counts, -1, nullptr, order_two);
    INFO("order-one available=" << order_one_available
         << " rows=" << order_one.size()
         << "; order-two available=" << order_two_available
         << " rows=" << order_two.size());
    REQUIRE(order_two_available);
    REQUIRE(order_one_available);
    REQUIRE(order_one.size() == order_two.size());
    for (std::size_t row = 0; row < order_one.size(); ++row) {
        REQUIRE(order_one[row].sides.size() == 1);
        REQUIRE(order_two[row].sides.size() == 1);
        CHECK(order_one[row].sides.front().domain ==
              order_two[row].sides.front().domain);
        CHECK(order_one[row].sides.front().component ==
              order_two[row].sides.front().component);
        CHECK(order_one[row].sides.front().phi_basis ==
              order_two[row].sides.front().phi_basis);
        CHECK(order_one[row].sides.front().phi_index ==
              order_two[row].sides.front().phi_index);
    }
}
