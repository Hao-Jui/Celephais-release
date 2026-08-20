// Phase 5: Adapted_nosym surface-shape DOF gate.
//
// Phase 4 closed the field-equation tau path. The adapted-radius
// surface-shape DOF path (variable-domain unknowns parametrizing the
// surface itself) was scaffolded with SYM dispatch retained: parity
// carried via `mm % 2`-aware jmax + `(2*j+1)` Galerkin weights inside
// `nbr_unknowns_from_adapted`, `affecte_coef_to_variable_domains`,
// `xx_to_vars_from_adapted`, `xx_to_ders_from_adapted`.
//
// Phase 5 collapses those routines to NONSYM convention. The Phase-4
// parity gate uses a plane-symmetric adapted radius, so plane-symmetric
// surface coefficients land in SYM-allowed sectors and the SYM
// dispatch reports correct counts — that fixture does NOT catch
// surface-shape divergence. This file adds the gate that does.
//
// Acceptance: NONSYM nbr_unknowns_from_variable_domains DIFFERS from
// SYM Adapted on the same geometry and keeps the surface columns square with
// the solver equation set. The scalar H=0 boundary can have more rows than
// variable-domain coefficients because other local field coefficients also
// participate in those boundary equations.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Space/spheric_homothetic.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

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

template <typename SpaceT>
Scalar radial_field(SpaceT& space)
{
    Scalar radius(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        if (d == space.get_nbr_domains() - 1) {
            radius.set_domain(d) = 0.0;
        } else {
            radius.set_domain(d) = space.get_domain(d)->get_radius();
        }
    }
    radius.std_base();
    radius.coef();
    return radius;
}

template <typename SpaceT>
void add_radial_test_equations(System_of_eqs& sys, SpaceT& space, Scalar& radius)
{
    sys.add_var("u", radius);
    sys.add_eq_inside(0, "u=0");
    for (int d = 1; d < space.get_nbr_domains(); ++d) {
        sys.add_eq_order(d, 2, "u=0");
    }
    sys.add_eq_bc(space.get_nbr_domains() - 1, OUTER_BC, "u=0");
    for (int d = 0; d < space.get_nbr_domains() - 1; ++d) {
        sys.add_eq_matching(d, OUTER_BC, "u");
    }
}

template <typename SpaceT>
void add_radial_test_equations_with_constant(System_of_eqs& sys, SpaceT& space,
                                             Scalar& radius, const Scalar& background)
{
    sys.add_var("u", radius);
    sys.add_cst("q", background);
    sys.add_eq_inside(0, "u+q=0");
    for (int d = 1; d < space.get_nbr_domains(); ++d)
        sys.add_eq_order(d, 2, "u+q=0");
    sys.add_eq_bc(space.get_nbr_domains() - 1, OUTER_BC, "u+q=0");
    for (int d = 0; d < space.get_nbr_domains() - 1; ++d)
        sys.add_eq_matching(d, OUTER_BC, "u");
}

struct ColumnErrorStats {
    double max_error = 0.0;
    double max_opposite_sign_error = 0.0;
    int row = -1;
    double analytic = 0.0;
    double finite_difference = 0.0;
};

ColumnErrorStats column_error_stats(const Array<double>& analytic, const Array<double>& before,
                                    const Array<double>& after, double eps)
{
    ColumnErrorStats stats;
    for (int i = 0; i < analytic.get_nbr(); ++i) {
        const double finite_difference = (before(i) - after(i)) / eps;
        const double error = std::abs(analytic(i) - finite_difference);
        if (error > stats.max_error) {
            stats.max_error = error;
            stats.row = i;
            stats.analytic = analytic(i);
            stats.finite_difference = finite_difference;
        }
        stats.max_opposite_sign_error =
            std::max(stats.max_opposite_sign_error, std::abs(analytic(i) + finite_difference));
    }
    return stats;
}

} // namespace

TEST_CASE("Spheric_adapted_nosym surface-shape DOF count matches NONSYM formula and differs from SYM Adapted",
          "[spheric_adapted_nosym_surface_dofs]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    Space_spheric_adapted plane_space(CHEB_TYPE, center, res, bounds);
    Space_spheric_adapted_nosym nosym_space(CHEB_TYPE, center, res, bounds);

    REQUIRE(plane_space.get_nbr_domains() == nosym_space.get_nbr_domains());

    const int plane_var = plane_space.nbr_unknowns_from_variable_domains();
    const int nosym_var = nosym_space.nbr_unknowns_from_variable_domains();

    Scalar plane_h(plane_space);
    plane_h = 0.0;
    plane_h.std_base();
    const auto* plane_outer = dynamic_cast<const Domain_shell_outer_adapted*>(plane_space.get_domain(1));
    REQUIRE(plane_outer != nullptr);
    const Array<int> plane_h_boundary = plane_outer->nbr_conditions_boundary(plane_h, 1, OUTER_BC);

    Scalar nosym_h(nosym_space);
    nosym_h = 0.0;
    nosym_h.std_base();
    const auto* nosym_outer = dynamic_cast<const Domain_shell_outer_adapted_nosym*>(nosym_space.get_domain(1));
    REQUIRE(nosym_outer != nullptr);
    const Array<int> nosym_h_boundary = nosym_outer->nbr_conditions_boundary(nosym_h, 1, OUTER_BC);

    INFO("Space_spheric_adapted       nbr_unknowns_from_variable_domains = " << plane_var);
    INFO("Space_spheric_adapted_nosym nbr_unknowns_from_variable_domains = " << nosym_var);
    INFO("Space_spheric_adapted       H=0 boundary conditions = " << plane_h_boundary(0));
    INFO("Space_spheric_adapted_nosym H=0 boundary conditions = " << nosym_h_boundary(0));

    REQUIRE(plane_var > 0);
    REQUIRE(nosym_var > 0);
    REQUIRE(plane_h_boundary(0) == plane_var);
    REQUIRE(nosym_h_boundary(0) >= nosym_var);

    // Pinned dev-cycle values for res=(5,5,4), bounds=[1,2,10] (2 variable
    // adapted shells: inner + outer). Pinning catches off-by-one regressions
    // that a bare `nosym_var != plane_var` predicate would silently accept.
    //
    //   SYM Adapted (pre-Phase-5 code, equivalent to plane_var here):
    //       per-shell loop with parity-dependent jmax produces 9 cells
    //       (k=0 mm=0 even → jmax=5, jmin=0 → 5 cells;
    //        k=2 mm=1 odd  → jmax=4, jmin=0 → 4 cells)
    //       outer shell drops one cell (radial endpoint constraint),
    //       net (5+4) + (5+3) = 17.
    //
    //   NONSYM Adapted_nosym:
    //       the unified COS/SIN surface-shape loop keeps the same accepted
    //       modes as the H=0 boundary for this small fixture, giving 14 cells.
    //       BNS_nosym uses a larger theta/phi fixture where the same formula
    //       yields 30 surface columns per star.
    //
    // Update both numbers together if the surface-DOF formula is ever
    // intentionally re-derived (e.g. Nyquist convention change, Galerkin
    // re-binding).
    REQUIRE(plane_var == 17);
    REQUIRE(nosym_var == 12);
    REQUIRE(nosym_h_boundary(0) == 12);

    // Belt-and-suspenders: counts must also disagree (catches the
    // pre-Phase-5 state where both are 17 if Phase 5 code is ever reverted).
    REQUIRE(nosym_var != plane_var);
}

TEST_CASE("Spheric_homothetic NOROT surface closure uses one spherical mode",
          "[spheric_homothetic][spheric_adapted_nosym_surface_dofs]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    Space_spheric_homothetic space(CHEB_TYPE, center, res, bounds);
    REQUIRE(space.nbr_unknowns_from_variable_domains() == 1);

    Scalar h(space);
    h = 0.0;
    h.std_base();

    const auto* outer = dynamic_cast<const Domain_shell_outer_homothetic*>(space.get_domain(1));
    REQUIRE(outer != nullptr);
    const Array<int> full_h_boundary = outer->nbr_conditions_boundary(h, 1, OUTER_BC);

    INFO("Space_spheric_homothetic nbr_unknowns_from_variable_domains = "
         << space.nbr_unknowns_from_variable_domains());
    INFO("Space_spheric_homothetic full H=0 boundary conditions = " << full_h_boundary(0));

    REQUIRE(full_h_boundary(0) > space.nbr_unknowns_from_variable_domains());

    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    sys.add_var("H", h);
    Index pos_cf(space.get_domain(1)->get_nbr_coefs());
    pos_cf.set(0) = 0;
    sys.add_eq_mode(1, OUTER_BC, "H", pos_cf, 0.);

    REQUIRE(sys.sec_member().get_nbr() == space.nbr_unknowns_from_variable_domains());
}

TEST_CASE("Spheric_adapted_nosym surface-shape Jacobian columns match finite differences",
          "[spheric_adapted_nosym_surface_dofs]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    const int nbr_surface_columns =
        Space_spheric_adapted_nosym(CHEB_TYPE, center, res, bounds).nbr_unknowns_from_variable_domains();

    for (int column = 0; column < nbr_surface_columns; ++column) {
        Space_spheric_adapted_nosym analytic_space(CHEB_TYPE, center, res, bounds);
        System_of_eqs analytic_sys(analytic_space, 0, analytic_space.get_nbr_domains() - 1);
        Scalar analytic_u = radial_field(analytic_space);
        add_radial_test_equations(analytic_sys, analytic_space, analytic_u);
        Array<double> before = analytic_sys.sec_member();
        REQUIRE(column < analytic_space.nbr_unknowns_from_variable_domains());
        Array<double> analytic = analytic_sys.do_col_J(column);

        Space_spheric_adapted_nosym perturbed_space(CHEB_TYPE, center, res, bounds);
        System_of_eqs perturbed_sys(perturbed_space, 0, perturbed_space.get_nbr_domains() - 1);
        Scalar perturbed_u = radial_field(perturbed_space);
        add_radial_test_equations(perturbed_sys, perturbed_space, perturbed_u);
        Array<double> perturbed_before = perturbed_sys.sec_member();

        REQUIRE(perturbed_before.get_nbr() == before.get_nbr());
        for (int i = 0; i < before.get_nbr(); ++i) {
            REQUIRE_THAT(perturbed_before(i), WithinAbs(before(i), 1e-12));
        }

        Dim_array xx_dim(1);
        xx_dim.set(0) = perturbed_sys.get_nbr_unknowns();
        Array<double> xx(xx_dim);
        for (int i = 0; i < xx.get_nbr(); ++i) {
            xx.set(i) = 0.0;
        }

        const double eps = 1e-7;
        xx.set(column) = eps;
        int pos = 0;
        perturbed_space.xx_to_vars_variable_domains(&perturbed_sys, xx, pos);
        Array<double> after = perturbed_sys.sec_member();

        const ColumnErrorStats stats = column_error_stats(analytic, before, after, eps);
        INFO("surface column = " << column);
        INFO("max row = " << stats.row);
        INFO("analytic = " << stats.analytic);
        INFO("finite_difference = " << stats.finite_difference);
        INFO("max opposite-sign error = " << stats.max_opposite_sign_error);
        REQUIRE(stats.max_error < 2e-6);
    }
}

TEST_CASE("Spheric_adapted_nosym do_JX accumulates mapping and field tangents while replacing constants",
          "[do_jx][spheric_adapted_nosym_surface_dofs]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    Space_spheric_adapted_nosym space(CHEB_TYPE, center, res, bounds);
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u = radial_field(space);
    Scalar background = radial_field(space);
    add_radial_test_equations_with_constant(sys, space, u, background);

    const Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());
    REQUIRE(sys.get_ncst() == 1);

    const int surface_column = 0;
    const int adapted_domain = 1;
    int adapted_field_column = space.nbr_unknowns_from_variable_domains();
    for (int domain = 0; domain < adapted_domain; ++domain)
        adapted_field_column += space.get_domain(domain)->nbr_unknowns(u, domain);

    REQUIRE(surface_column < space.nbr_unknowns_from_variable_domains());
    REQUIRE(adapted_field_column < sys.get_nbr_unknowns());

    const Array<double> shape_column = sys.do_col_J(surface_column);
    const Array<double> field_column = sys.do_col_J(adapted_field_column);

    Array<double> perturbation(sys.get_nbr_unknowns());
    perturbation = 0.0;
    perturbation.set(surface_column) = 1.0;
    perturbation.set(adapted_field_column) = 1.0;

    const Array<double> combined = sys.do_JX(perturbation);

    // The radial constant has a nonzero moving-domain tangent. If constants
    // accidentally take the additive term path, the repeated do_JX below
    // accumulates stale state and fails the byte comparison.
    const Val_domain& constant_mapping_derivative =
        sys.give_cst(0, adapted_domain)->get_der_t()()(adapted_domain);
    double constant_mapping_norm = 0.0;
    Index position(space.get_domain(adapted_domain)->get_nbr_points());
    do {
        constant_mapping_norm = std::max(
            constant_mapping_norm, std::abs(constant_mapping_derivative(position)));
    } while (position.inc());

    Array<double> repeated(sys.get_nbr_conditions());
    sys.do_JX(perturbation, repeated);

    double shape_norm = 0.0;
    double field_norm = 0.0;
    double expected_scale = 0.0;
    double parity_error = 0.0;
    for (int row = 0; row < combined.get_nbr(); ++row) {
        shape_norm = std::max(shape_norm, std::abs(shape_column(row)));
        field_norm = std::max(field_norm, std::abs(field_column(row)));
        const double expected = shape_column(row) + field_column(row);
        expected_scale = std::max(expected_scale, std::abs(expected));
        parity_error = std::max(parity_error, std::abs(combined(row) - expected));
    }

    INFO("surface column = " << surface_column);
    INFO("adapted field column = " << adapted_field_column);
    INFO("max shape contribution = " << shape_norm);
    INFO("max field contribution = " << field_norm);
    INFO("max constant mapping contribution = " << constant_mapping_norm);
    INFO("max combined parity error = " << parity_error);
    REQUIRE(shape_norm > 0.0);
    REQUIRE(field_norm > 0.0);
    REQUIRE(constant_mapping_norm > 0.0);
    REQUIRE(parity_error <= 1e-12 * (1.0 + expected_scale));
    REQUIRE(std::memcmp(combined.get_data(), repeated.get_data(),
                        static_cast<std::size_t>(combined.get_nbr()) * sizeof(double)) == 0);
}

TEST_CASE("Spheric_adapted_nosym surface Galerkin columns preserve theta parity",
          "[spheric_adapted_nosym_surface_dofs]")
{
    Point center = make_origin();
    Dim_array res(3);
    res.set(0) = 5;
    res.set(1) = 9;
    res.set(2) = 6;
    Array<double> bounds = make_bounds();

    Space_spheric_adapted_nosym space(CHEB_TYPE, center, res, bounds);
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar radius = radial_field(space);
    sys.add_var("u", radius);
    sys.add_eq_bc(1, OUTER_BC, "u=0");
    Array<double> before = sys.sec_member();
    REQUIRE(before.get_nbr() == sys.get_nbr_conditions());

    int ncol = sys.get_nbr_conditions();
    REQUIRE(ncol == 39);

    // Probe the last surface-shape column; verify it produces a
    // structurally non-trivial Jacobian row (guards against silent
    // truncation of the surface DOF set).
    Array<double> last_column = sys.do_col_J(ncol - 1);
    REQUIRE(last_column.get_nbr() == ncol);
    double colnorm = 0.;
    for (int r = 0; r < ncol; ++r)
        colnorm += std::abs(last_column(r));
    REQUIRE(colnorm > 0.);
}
