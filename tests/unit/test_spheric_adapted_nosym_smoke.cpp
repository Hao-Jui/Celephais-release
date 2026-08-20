// Smoke test for Spheric_adapted_nosym.
//
// Adapted_nosym is the surface-tracking sibling of Spheric_nosym, ported by
// cloning the production Spheric_adapted code with class-name renames + Domain
// type substitution (Domain_nucleus → Domain_nucleus_nosym, etc.) so the new
// Space wires into the parity-relaxed Spheric_nosym domain hierarchy.
//
// Status (post-Phase-4 + Phase-5): the NONSYM basis collapse has been applied
// to BOTH the field-equation tau path (base setters, nbr_unknowns /
// nbr_conditions, export_tau / affecte_tau Galerkin family, integration
// kernels — commits 6f559e95, 980ffe1f, 23291b40, eb96369f) AND the surface-
// shape DOF path (variable-domain Galerkin in domain_shell_{inner,outer}_
// adapted_nosym.cpp — commit 74e7f645). Spheric_adapted_nosym now uses the
// unified COS / SIN / COSSIN theta basis end-to-end and matches plain
// Spheric_adapted on z-symmetric input to FP precision
// ([spheric_adapted_nosym_parity] gate). Stronger gates:
//   - [spheric_adapted_nosym_base_setters]   theta basis is unified,
//                                            forward+inverse round-trip.
//   - [spheric_adapted_nosym_parity]         ||sec_member||_inf FP-match
//                                            against SYM Adapted.
//   - [spheric_adapted_nosym_surface_dofs]   surface-shape DOF count pinned
//                                            against NONSYM formula
//                                            (13 vs SYM 17 on this fixture).
//
// This file is the smallest gate: Space ctor + dtor + Domain count.

#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <cmath>
#include <cstring>

using namespace Kadath;

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

void require_bitwise_equal(const Val_domain& actual, const Array<double>& expected)
{
    actual.coef_i();
    const Array<double> values = actual.get_conf();
    REQUIRE(values.get_dimensions() == expected.get_dimensions());
    std::size_t mismatch = values.get_nbr();
    for (std::size_t offset = 0; offset < values.get_nbr(); ++offset) {
        if (std::memcmp(values.get_data() + offset, expected.get_data() + offset,
                        sizeof(double)) != 0) {
            mismatch = offset;
            break;
        }
    }
    const double mismatch_actual = mismatch == values.get_nbr() ? 0.0 : values.get_data()[mismatch];
    const double mismatch_expected = mismatch == values.get_nbr() ? 0.0 : expected.get_data()[mismatch];
    CAPTURE(mismatch, mismatch_actual, mismatch_expected);
    REQUIRE(std::memcmp(values.get_data(), expected.get_data(), values.get_nbr() * sizeof(double)) == 0);
}

} // namespace

TEST_CASE("Space_spheric_adapted_nosym constructs and reports domain count",
          "[spheric_adapted_nosym_smoke]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();

    Space_spheric_adapted_nosym space(CHEB_TYPE, center, res, bounds);

    // Adapted layout: nucleus + inner_adapted_shell + outer_adapted_shell +
    // (bounds.size - 3) outer fixed shells + compactified outer.
    // For bounds=[1.0, 2.0, 10.0] (3 entries) → 5 domains:
    //   0: nucleus, 1: inner_adapted, 2: outer_adapted, 3: compact.
    REQUIRE(space.get_nbr_domains() == 4);
    REQUIRE(space.get_ndim() == 3);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        REQUIRE(space.get_domain(d) != nullptr);
    }
}

TEST_CASE("Outer adapted no-sym basis multiplication cache preserves value semantics",
          "[spheric_adapted_nosym_smoke][basis_mult_cache]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();
    Space_spheric_adapted_nosym space(CHEB_TYPE, center, res, bounds);

    const Domain* const domain = space.get_domain(1);
    REQUIRE(dynamic_cast<const Domain_shell_outer_adapted_nosym*>(domain) != nullptr);

    Base_spectral a(3);
    Base_spectral b(3);
    a.set(res, COSSIN, COS, CHEB);
    b.set(res, COSSIN, COS, CHEB);

    const Base_spectral expected = domain->mult(a, b);
    REQUIRE(expected.is_def());
    REQUIRE(domain->mult(a, b) == expected);

    Base_spectral independently_owned = domain->mult(a, b);
    independently_owned.set_non_def();
    REQUIRE(domain->mult(a, b) == expected);

    b.set(res, COSSIN, SIN, CHEB);
    const Base_spectral mutated_input_result = domain->mult(a, b);
    REQUIRE(mutated_input_result.is_def());
    REQUIRE_FALSE(mutated_input_result == expected);

    b.set(res, COSSIN, COS, CHEB);
    REQUIRE(domain->mult(a, b) == expected);

    b.set(res, COSSIN, COS, LEG);
    REQUIRE_FALSE(domain->mult(a, b).is_def());
    REQUIRE_FALSE(domain->mult(a, b).is_def());

    b.set_non_def();
    REQUIRE_FALSE(domain->mult(a, b).is_def());
}

TEST_CASE("Spheric_adapted_nosym physical-order point sweeps are bitwise equivalent to canonical Index evaluation",
          "[spheric_adapted_nosym_smoke][physical_traversal]")
{
    Point center = make_origin();
    Dim_array res = make_resolution();
    Array<double> bounds = make_bounds();
    Space_spheric_adapted_nosym space(CHEB_TYPE, center, res, bounds);

    for (int domain_number = 1; domain_number <= 2; ++domain_number) {
        const Domain* const domain = space.get_domain(domain_number);
        const Dim_array dimensions = domain->get_nbr_points();
        const Array<double> xi = domain->get_coloc(1);
        const Array<double> theta = domain->get_coloc(2);
        const Array<double> phi = domain->get_coloc(3);
        const double inner_boundary = domain_number == 1 ? bounds(0) : bounds(1);
        const double outer_boundary = domain_number == 1 ? bounds(1) : bounds(2);

        Array<double> expected_radius(dimensions);
        Array<double> expected_x(dimensions);
        Array<double> expected_y(dimensions);
        Array<double> expected_z(dimensions);
        Array<double> expected_x_over_r(dimensions);
        Array<double> expected_y_over_r(dimensions);
        Array<double> expected_z_over_r(dimensions);
        Index index(dimensions);
        do {
            const double rr = (outer_boundary - inner_boundary) / 2. * xi(index(0)) +
                              (outer_boundary + inner_boundary) / 2.;
            expected_radius.set(index) = rr;
            expected_x.set(index) =
                rr * std::sin(theta(index(1))) * std::cos(phi(index(2))) + center(1);
            expected_y.set(index) =
                rr * std::sin(theta(index(1))) * std::sin(phi(index(2))) + center(2);
            expected_z.set(index) = rr * std::cos(theta(index(1))) + center(3);
            expected_x_over_r.set(index) = std::sin(theta(index(1))) * std::cos(phi(index(2)));
            expected_y_over_r.set(index) = std::sin(theta(index(1))) * std::sin(phi(index(2)));
            expected_z_over_r.set(index) = std::cos(theta(index(1)));
        } while (index.inc());

        require_bitwise_equal(domain->get_radius(), expected_radius);
        require_bitwise_equal(domain->get_absol(1), expected_x);
        require_bitwise_equal(domain->get_absol(2), expected_y);
        require_bitwise_equal(domain->get_absol(3), expected_z);
        require_bitwise_equal(domain->get_cart(1), expected_x);
        require_bitwise_equal(domain->get_cart(2), expected_y);
        require_bitwise_equal(domain->get_cart(3), expected_z);
        require_bitwise_equal(domain->get_cart_surr(1), expected_x_over_r);
        require_bitwise_equal(domain->get_cart_surr(2), expected_y_over_r);
        require_bitwise_equal(domain->get_cart_surr(3), expected_z_over_r);

        Scalar old(space);
        old = 0.;
        // Include angular structure so an incorrect flat-offset mapping in
        // update_variable cannot be masked by a purely radial field.
        old.set_domain(domain_number) =
            domain->get_radius() + 0.125 * domain->get_cart(1);
        Val_domain correction(domain);
        correction = 0.125;
        correction.std_base();
        Scalar actual_update(space);
        actual_update = 0.;

        Val_domain radial_derivative(old(domain_number).der_r());
        Val_domain expected_update(domain);
        expected_update.allocate_conf();
        index.set_start();
        do {
            const double radial_weight = domain_number == 1 ? (1. + xi(index(0))) / 2.
                                                             : (1. - xi(index(0))) / 2.;
            expected_update.set(index) = radial_derivative(index) * radial_weight;
        } while (index.inc());
        expected_update = correction * expected_update + old(domain_number);

        if (domain_number == 1) {
            const auto* adapted = dynamic_cast<const Domain_shell_outer_adapted_nosym*>(domain);
            REQUIRE(adapted != nullptr);
        } else {
            const auto* adapted = dynamic_cast<const Domain_shell_inner_adapted_nosym*>(domain);
            REQUIRE(adapted != nullptr);
        }
        // The override is protected in the concrete adapted classes but is a
        // public virtual operation on Domain, matching production dispatch.
        domain->update_variable(correction, old, actual_update);
        require_bitwise_equal(actual_update(domain_number), expected_update.get_conf());
    }
}
