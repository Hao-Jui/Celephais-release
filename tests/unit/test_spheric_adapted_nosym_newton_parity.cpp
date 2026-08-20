// Phase 4 bucket 5: Adapted-scale parity gate.
//
// Mirrors tests/unit/test_spheric_nosym_newton_parity.cpp at the surface-
// tracking scope. Builds an identical scalar problem on:
//   - Space_spheric_adapted        : z-plane sym, full phi, with adapted
//                                    inner+outer shells. Production analog.
//   - Space_spheric_adapted_nosym  : no plane sym, full phi, same adapted
//                                    layout. Phase-4-collapsed in
//                                    commits 6f559e95 + 980ffe1f + 23291b40.
//
// Evaluates first-iteration residual via System_of_eqs::sec_member() and
// asserts:
//   1. Both spaces produce finite, positive residuals (dispatch reachable).
//   2. nbr_conditions / nbr_unknowns differ predictably (basis-storage
//      organization).
//   3. (Acceptance for Phase 4) ||r||_inf agrees within FP precision —
//      the symmetric subspace embeds in the unrestricted basis for
//      z-symmetric input.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "Linear_algebra/jacobian_parity_mask.hpp"

#include <algorithm>
#include <array>
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
    Dim_array dim(1);
    dim.set(0) = 3;
    Array<double> bounds(dim);
    bounds.set(0) = 1.0;
    bounds.set(1) = 2.0;
    bounds.set(2) = 10.0;
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

template <typename SpaceT>
double assemble_residual_infinity_norm(SpaceT& space, double u_const)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);

    Scalar u(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        u.set_domain(d) = u_const;
    }
    u.std_base();
    u.coef();
    sys.add_var("u", u);

    sys.add_eq_inside(0, "u=0");
    for (int d = 1; d < space.get_nbr_domains(); ++d) {
        sys.add_eq_order(d, 2, "u=0");
    }
    sys.add_eq_bc(space.get_nbr_domains() - 1, OUTER_BC, "u=0");
    for (int d = 0; d < space.get_nbr_domains() - 1; ++d) {
        sys.add_eq_matching(d, OUTER_BC, "u");
    }

    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());

    return residual_infinity_norm(residual);
}

} // namespace

TEST_CASE("Spheric_adapted_nosym sec_member residual against plain Space_spheric_adapted on z-symmetric data",
          "[spheric_adapted_nosym_parity]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    Space_spheric_adapted plane_space(CHEB_TYPE, center, res, bounds);
    Space_spheric_adapted_nosym nosym_space(CHEB_TYPE, center, res, bounds);

    REQUIRE(plane_space.get_nbr_domains() == nosym_space.get_nbr_domains());

    const double u_const = 0.5;

    const double plane_norm = assemble_residual_infinity_norm(plane_space, u_const);
    const double nosym_norm = assemble_residual_infinity_norm(nosym_space, u_const);

    INFO("Space_spheric_adapted (plane sym) ||r||_inf = " << plane_norm);
    INFO("Space_spheric_adapted_nosym       ||r||_inf = " << nosym_norm);

    // Dispatch reachability: nosym ran end-to-end and returned a real
    // residual through System_of_eqs::sec_member().
    REQUIRE(std::isfinite(plane_norm));
    REQUIRE(std::isfinite(nosym_norm));
    REQUIRE(plane_norm > 0.0);
    REQUIRE(nosym_norm > 0.0);

    // Phase 4 acceptance: symmetric subspace embeds in NONSYM basis, so
    // ||r||_inf must agree to FP precision on z-symmetric input (per-domain
    // constant in the (0,0,0) mode of both bases).
    REQUIRE_THAT(nosym_norm, WithinAbs(plane_norm, 1e-12));
}

TEST_CASE("Spheric_adapted_nosym System_of_eqs reports finite nbr_conditions / nbr_unknowns",
          "[spheric_adapted_nosym_parity]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    Space_spheric_adapted plane_space(CHEB_TYPE, center, res, bounds);
    Space_spheric_adapted_nosym nosym_space(CHEB_TYPE, center, res, bounds);

    auto build_sys = [](auto& space) {
        System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
        Scalar u(space);
        u = 0.0;
        u.std_base();
        u.coef();
        sys.add_var("u", u);
        sys.add_eq_inside(0, "u=0");
        for (int d = 1; d < space.get_nbr_domains(); ++d) {
            sys.add_eq_order(d, 2, "u=0");
        }
        sys.add_eq_bc(space.get_nbr_domains() - 1, OUTER_BC, "u=0");
        for (int d = 0; d < space.get_nbr_domains() - 1; ++d) {
            sys.add_eq_matching(d, OUTER_BC, "u");
        }
        (void)sys.sec_member();
        return std::pair{sys.get_nbr_conditions(), sys.get_nbr_unknowns()};
    };

    const auto [plane_nc, plane_nu] = build_sys(plane_space);
    const auto [nosym_nc, nosym_nu] = build_sys(nosym_space);

    INFO("Space_spheric_adapted       : nbr_conditions=" << plane_nc << " nbr_unknowns=" << plane_nu);
    INFO("Space_spheric_adapted_nosym : nbr_conditions=" << nosym_nc << " nbr_unknowns=" << nosym_nu);

    REQUIRE(plane_nc > 0);
    REQUIRE(plane_nu > 0);
    REQUIRE(nosym_nc > 0);
    REQUIRE(nosym_nu > 0);
}

TEST_CASE("boost-like adapted-nosym mixed-field descriptors match full-J matrix grading",
          "[residual_row_descriptor][parity_mask][boost_fixture]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();
    Space_spheric_adapted_nosym space(CHEB_TYPE, center, res, bounds);

    System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
    Scalar P(space);
    P = 0.0;
    P.std_base();
    P.coef();

    Base_tensor cartesian(space, CARTESIAN_BASIS);
    Vector bet(space, CON, cartesian);
    bet = 0.0;
    bet.std_base();
    bet.coef();

    Scalar phi(space);
    phi = 0.0;
    phi.std_base();
    phi.coef();

    system.add_var("P", P);
    system.add_var("bet", bet);
    system.add_var("phi", phi);

    const int domain_count = space.get_nbr_domains();
    system.add_eq_inside(0, "P=0");
    for (int domain = 1; domain < domain_count; ++domain)
        system.add_eq_order(domain, 2, "P=0");
    system.add_eq_bc(domain_count - 1, OUTER_BC, "P=0");
    for (int domain = 0; domain < domain_count - 1; ++domain)
        system.add_eq_matching(domain, OUTER_BC, "P");

    system.add_eq_inside(0, "bet^i=0");
    for (int domain = 1; domain < domain_count; ++domain)
        system.add_eq_order(domain, 2, "bet^i=0");

    system.add_eq_inside(0, "phi=0");
    for (int domain = 1; domain < domain_count; ++domain)
        system.add_eq_order(domain, 2, "phi=0");

    const Array<double> residual = system.sec_member();
    REQUIRE(residual.get_nbr() == system.get_nbr_conditions());

    std::vector<ResidualRowDescriptor> descriptors;
    REQUIRE(system.describe_residual_rows(descriptors));
    REQUIRE(descriptors.size() ==
            static_cast<std::size_t>(system.get_nbr_conditions()));

    long long expected_p_rows = 0;
    long long expected_phi_rows = 0;
    std::array<long long, 3> expected_bet_rows{};
    for (int domain = 0; domain < domain_count; ++domain) {
        const Array<int> p_counts =
            space.get_domain(domain)->nbr_conditions(P, domain, 2);
        const Array<int> bet_counts =
            space.get_domain(domain)->nbr_conditions(bet, domain, 2);
        const Array<int> phi_counts =
            space.get_domain(domain)->nbr_conditions(phi, domain, 2);
        REQUIRE(p_counts.get_nbr() == 1);
        REQUIRE(bet_counts.get_nbr() == 3);
        REQUIRE(phi_counts.get_nbr() == 1);
        expected_p_rows += p_counts(0);
        expected_phi_rows += phi_counts(0);
        for (int component = 0; component < 3; ++component)
            expected_bet_rows[static_cast<std::size_t>(component)] +=
                bet_counts(component);
    }
    const int outer_domain = domain_count - 1;
    const Array<int> boundary_counts =
        space.get_domain(outer_domain)->nbr_conditions_boundary(
            P, outer_domain, OUTER_BC);
    REQUIRE(boundary_counts.get_nbr() == 1);
    expected_p_rows += boundary_counts(0);
    for (int domain = 0; domain < domain_count - 1; ++domain) {
        const Array<int> matching_counts =
            space.get_domain(domain)->nbr_conditions_boundary(
                P, domain, OUTER_BC);
        REQUIRE(matching_counts.get_nbr() == 1);
        expected_p_rows += matching_counts(0);
    }

    const int bet_first_equation = 2 * domain_count;
    const int phi_first_equation = 3 * domain_count;
    long long p_rows = 0;
    long long phi_rows = 0;
    std::array<long long, 3> bet_rows{};
    for (std::size_t row = 0; row < descriptors.size(); ++row) {
        const ResidualRowDescriptor& descriptor = descriptors[row];
        REQUIRE(descriptor.available);
        REQUIRE(descriptor.family == ResidualRowEquationFamily::Field);
        REQUIRE(descriptor.explicit_sector == 0);
        REQUIRE_FALSE(descriptor.sides.empty());
        const ResidualRowCoordinate& coordinate = descriptor.sides.front();
        CHECK(coordinate.domain >= 0);
        CHECK(coordinate.domain < domain_count);
        CHECK(coordinate.phi_basis == COSSIN);
        CHECK(coordinate.phi_index >= 0);
        CHECK(coordinate.phi_index <
              space.get_domain(coordinate.domain)->get_nbr_coefs()(2));
        CHECK(coordinate.phi_index != 1);

        if (descriptor.equation_index < 2 * domain_count) {
            CHECK(coordinate.component == 0);
            const std::size_t expected_sides =
                descriptor.equation_index < domain_count + 1 ? 1 : 2;
            CHECK(descriptor.sides.size() == expected_sides);
            ++p_rows;
        } else if (descriptor.equation_index >= bet_first_equation &&
                   descriptor.equation_index < phi_first_equation) {
            REQUIRE(coordinate.component >= 0);
            REQUIRE(coordinate.component < 3);
            ++bet_rows[static_cast<std::size_t>(coordinate.component)];
        } else if (descriptor.equation_index >= phi_first_equation &&
                   descriptor.equation_index < 4 * domain_count) {
            CHECK(coordinate.component == 0);
            ++phi_rows;
        } else {
            FAIL("descriptor equation index is outside the fixture");
        }
    }
    REQUIRE_FALSE(descriptors.empty());
    CHECK(p_rows == expected_p_rows);
    CHECK(phi_rows == expected_phi_rows);
    CHECK(bet_rows == expected_bet_rows);
    const long long covered_rows = static_cast<long long>(descriptors.size());
    CHECK(covered_rows == expected_p_rows + expected_phi_rows +
                              expected_bet_rows[0] + expected_bet_rows[1] +
                              expected_bet_rows[2]);

    const JacobianParityRowPrediction prediction =
        predict_jacobian_parity_rows(system);
    REQUIRE(prediction.sector.size() == descriptors.size());
    REQUIRE(prediction.all_rows_available);
    CHECK(prediction.unavailable_rows == 0);
    CHECK(prediction.ungraded_rows == 0);
    for (const signed char sector : prediction.sector)
        CHECK((sector == 1 || sector == -1));

    const JacobianParityColumnGrading column_grading =
        grade_jacobian_parity_columns(system);
    REQUIRE(jacobian_parity_column_grading_disable_reason(column_grading).empty());
    REQUIRE(column_grading.sector.size() ==
            static_cast<std::size_t>(system.get_nbr_unknowns()));

    std::vector<double> symmetric_mass(descriptors.size(), 0.0);
    std::vector<double> antisymmetric_mass(descriptors.size(), 0.0);
    for (int column = 0; column < system.get_nbr_unknowns(); ++column) {
        const Array<double> values = system.do_col_J(column);
        REQUIRE(values.get_nbr() == system.get_nbr_conditions());
        std::vector<double>& mass = column_grading.sector[column] > 0
                                        ? symmetric_mass
                                        : antisymmetric_mass;
        for (std::size_t matrix_row = 0; matrix_row < descriptors.size();
             ++matrix_row) {
            mass[matrix_row] += std::abs(values(static_cast<int>(matrix_row)));
        }
    }

    JacobianParityMaskState matrix_grading;
    derive_jacobian_parity_row_sectors(
        matrix_grading, symmetric_mass, antisymmetric_mass);
    const JacobianParityRowOracleComparison oracle =
        compare_jacobian_parity_row_prediction(
            prediction, matrix_grading.row_sector);
    INFO("covered P/bet/phi rows=" << covered_rows
                                 << " unavailable rows=" << oracle.unavailable_rows
                                 << " first mismatch=" << oracle.first_mismatch);
    REQUIRE(oracle.failure_reason.empty());
    REQUIRE(oracle.exact_on_covered_rows);
    REQUIRE(oracle.whole_fixture_covered);
    CHECK(oracle.compared_rows == covered_rows);
    CHECK(oracle.mismatched_rows == 0);
    CHECK(oracle.first_mismatch == -1);
}
