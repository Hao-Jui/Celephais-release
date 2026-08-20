#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/Space/bin_ns.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "Linear_algebra/jacobian_parity_mask.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace Kadath;
using Catch::Matchers::WithinAbs;

namespace {

std::vector<double> ns_bounds()
{
    return {1.0, 2.0, 4.0};
}

std::vector<double> outer_bounds()
{
    return {10.0};
}

int unrelated_definition_calls = 0;

Term_eq counted_identity(const Term_eq& value, Param*)
{
    ++unrelated_definition_calls;
    return value;
}

bool same_double_bits(double lhs, double rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(double)) == 0;
}

template <typename SpaceT>
Scalar make_z_symmetric_scalar(SpaceT& space)
{
    Scalar u(space);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const Domain* domain = space.get_domain(d);
        const Val_domain x = domain->get_cart(1);
        const Val_domain y = domain->get_cart(2);
        const Val_domain z = domain->get_cart(3);
        u.set_domain(d) = 1.0 + 0.01 * x + 0.02 * y + 0.03 * z * z;
    }
    u.std_base();
    u.coef();
    return u;
}

template <typename SpaceT>
Vector make_z_symmetric_vector(SpaceT& space)
{
    Base_tensor basis(space, CARTESIAN_BASIS);
    Vector v(space, CON, basis);
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        const Domain* domain = space.get_domain(d);
        const Val_domain x = domain->get_cart(1);
        const Val_domain y = domain->get_cart(2);
        const Val_domain z = domain->get_cart(3);
        v.set(1).set_domain(d) = 0.1 + 0.01 * x + 0.02 * y + 0.03 * z * z;
        v.set(2).set_domain(d) = -0.2 + 0.03 * x - 0.01 * y + 0.02 * z * z;
        v.set(3).set_domain(d) = z * (0.04 + 0.01 * x + 0.02 * y);
    }
    v.std_base();
    v.coef();
    return v;
}

double residual_infinity_norm(const Array<double>& residual)
{
    double max_abs = 0.0;
    for (int i = 0; i < residual.get_nbr(); ++i) {
        max_abs = std::max(max_abs, std::abs(residual.get_data()[i]));
    }
    return max_abs;
}

template <typename SpaceT>
double scalar_matching_norm(SpaceT& space, int domain, const char* expression)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u = make_z_symmetric_scalar(space);
    sys.add_var("u", u);
    sys.add_eq_matching(domain, OUTER_BC, expression);
    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());
    return residual_infinity_norm(residual);
}

template <typename SpaceT>
double velocity_potential_norm(SpaceT& space, int domain)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u = make_z_symmetric_scalar(space);
    sys.add_var("u", u);
    sys.add_eq_vel_pot(domain, 2, "u = 0", "u = 0");
    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());
    return residual_infinity_norm(residual);
}

template <typename SpaceT>
double full_space_scalar_norm(SpaceT& space)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u = make_z_symmetric_scalar(space);
    sys.add_var("u", u);
    space.add_eq(sys, "u = 0", "u", "dn(u)");
    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());
    return residual_infinity_norm(residual);
}

template <typename SpaceT>
double full_space_vector_norm(SpaceT& space)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Vector v = make_z_symmetric_vector(space);
    sys.add_var("v", v);
    space.add_eq(sys, "v^i = 0", "v^i", "dn(v^i)");
    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());
    return residual_infinity_norm(residual);
}

template <typename SpaceT>
double first_integral_norm(SpaceT& space, int dmin, int dmax, bool restrict_constant_m_order = false)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u = make_z_symmetric_scalar(space);
    Scalar cst = make_z_symmetric_scalar(space);
    if (restrict_constant_m_order) {
        cst.affect_parameters();
        cst.set_parameters()->set_m_order() = 1;
    }
    sys.add_var("u", u);
    sys.add_var("cst", cst);
    sys.add_eq_first_integral(dmin, dmax, "u", "cst");
    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());
    return residual_infinity_norm(residual);
}

template <typename SpaceT>
int zero_scalar_condition_count(SpaceT& space, int domain)
{
    Scalar zero(space);
    zero = 0.0;
    zero.std_base();
    zero.coef();
    Array<int> conditions = space.get_domain(domain)->nbr_conditions(zero, domain, 0);
    REQUIRE(conditions.get_size(0) == 1);
    return conditions(0);
}

template <typename SpaceT>
double volume_integral_residual(SpaceT& space, int dmin, int dmax)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u = make_z_symmetric_scalar(space);
    sys.add_var("u", u);
    space.add_eq_int_volume(sys, dmin, dmax, "integvolume(u) = 0");
    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());
    REQUIRE(residual.get_nbr() >= 1);
    return residual(0);
}

template <typename SpaceT>
double infinity_integral_residual(SpaceT& space)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u(space);
    u = 1.0;
    u.std_base();
    u.coef();
    sys.add_var("u", u);
    space.add_eq_int_inf(sys, "integ(u) = 0");
    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());
    REQUIRE(residual.get_nbr() >= 1);
    return residual(0);
}

template <typename SpaceT>
double outer_sphere_one_integral_residual(SpaceT& space)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u = make_z_symmetric_scalar(space);
    sys.add_var("u", u);
    space.add_eq_int_outer_sphere_one(sys, "integ(u) = 0");
    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());
    REQUIRE(residual.get_nbr() >= 1);
    return residual(0);
}

template <typename SpaceT>
double outer_sphere_two_integral_residual(SpaceT& space)
{
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u = make_z_symmetric_scalar(space);
    sys.add_var("u", u);
    space.add_eq_int_outer_sphere_two(sys, "integ(u) = 0");
    Array<double> residual = sys.sec_member();
    REQUIRE(residual.get_nbr() == sys.get_nbr_conditions());
    REQUIRE(residual.get_nbr() >= 1);
    return residual(0);
}

void require_field_descriptor(const ResidualRowDescriptor& descriptor,
                              int equation, std::size_t sides)
{
    REQUIRE(descriptor.family == ResidualRowEquationFamily::Field);
    REQUIRE(descriptor.equation_index == equation);
    REQUIRE(descriptor.available);
    REQUIRE(descriptor.explicit_sector == 0);
    REQUIRE(descriptor.sides.size() == sides);
    for (const ResidualRowCoordinate& coordinate : descriptor.sides) {
        CHECK(coordinate.domain >= 0);
        CHECK(coordinate.component >= 0);
        CHECK(coordinate.phi_basis != 0);
        CHECK(coordinate.phi_index >= 0);
    }
}

template <typename SpaceT>
Scalar make_zero_scalar(SpaceT& space, bool antisymmetric = false)
{
    Scalar field(space);
    field = 0.0;
    if (antisymmetric)
        field.std_anti_base();
    else
        field.std_base();
    field.coef();
    return field;
}

} // namespace

TEST_CASE("symmetric BNS emitted QE topology has complete structural rows",
          "[residual_row_descriptor][sym][bin_ns_matching]")
{
    Space_bin_ns space(
        CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    SECTION("full scalar wiring covers volume and both matching families")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        system.add_var("P", P);
        space.add_eq(system, "P=0", "P", "dn(P)");

        std::vector<ResidualRowDescriptor> descriptors;
        REQUIRE(system.describe_residual_rows(descriptors));
        REQUIRE(descriptors.size() ==
                static_cast<std::size_t>(system.get_nbr_conditions()));
        REQUIRE_FALSE(descriptors.empty());
        for (const ResidualRowDescriptor& descriptor : descriptors) {
            REQUIRE(descriptor.family == ResidualRowEquationFamily::Field);
            REQUIRE(descriptor.available);
            REQUIRE(descriptor.explicit_sector == 0);
            REQUIRE((descriptor.sides.size() == 1 ||
                     descriptor.sides.size() == 2));
            for (const ResidualRowCoordinate& coordinate : descriptor.sides) {
                CHECK((coordinate.phi_basis == COSSIN ||
                       coordinate.phi_basis == COS ||
                       coordinate.phi_basis == SIN));
                CHECK(space.get_domain(coordinate.domain)
                          ->phi_coefficient_parity(
                              coordinate.phi_index,
                              coordinate.phi_basis) != 0);
            }
        }

        const JacobianParityRowPrediction prediction =
            predict_jacobian_parity_rows(system);
        REQUIRE(prediction.all_rows_available);
        CHECK(prediction.unavailable_rows == 0);
        CHECK(prediction.ungraded_rows == 0);
        CHECK(prediction.unsupported_phi_basis_rows == 0);
        REQUIRE(std::all_of(
            prediction.sector.begin(), prediction.sector.end(),
            [](signed char sector) { return sector == -1 || sector == 1; }));
    }

    SECTION("uncontracted import remains unavailable")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        system.add_var("P", P);
        const int source = space.ADAPTED1 + 1;
        system.add_eq_matching_import(
            source, OUTER_BC, "P=import(P)", -1, nullptr, "P");

        std::vector<ResidualRowDescriptor> descriptors;
        REQUIRE_FALSE(system.describe_residual_rows(descriptors));
        REQUIRE_FALSE(descriptors.empty());
        for (const ResidualRowDescriptor& descriptor : descriptors) {
            CHECK_FALSE(descriptor.available);
            CHECK(descriptor.sides.empty());
        }
    }

    SECTION("malformed boundary counts clear partial descriptors")
    {
        Scalar P = make_zero_scalar(space);
        const int domain_index = space.OUTER + 1;
        const Domain* domain = space.get_domain(domain_index);
        Array<int> counts = domain->nbr_conditions_boundary(
            P, domain_index, ETA_PLUS_BC);
        REQUIRE(counts.get_nbr() == 1);
        counts.set(0) = counts(0) + 1;
        std::vector<ResidualRowDescriptor> descriptors(1);
        REQUIRE_FALSE(domain->describe_boundary_residual_rows(
            P, domain_index, ETA_PLUS_BC, counts, -1, nullptr,
            descriptors));
        CHECK(descriptors.empty());
    }
}

TEST_CASE("nosym Eq_bc descriptors preserve boundary exporter order",
          "[residual_row_descriptor][eq_bc][bin_ns_nosym_matching]")
{
    Space_bin_ns_nosym space(
        CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
    Scalar P = make_zero_scalar(space);
    system.add_var("P", P);

    const int domain = space.get_nbr_domains() - 1;
    const Array<int> expected = space.get_domain(domain)->nbr_conditions_boundary(
        P, domain, OUTER_BC);
    REQUIRE(expected.get_nbr() == 1);
    system.add_eq_bc(domain, OUTER_BC, "P=0", -1, nullptr, "P");

    std::vector<ResidualRowDescriptor> descriptors;
    REQUIRE(system.describe_residual_rows(descriptors));
    REQUIRE(descriptors.size() == static_cast<std::size_t>(expected(0)));
    for (const ResidualRowDescriptor& descriptor : descriptors) {
        require_field_descriptor(descriptor, 0, 1);
        CHECK(descriptor.sides.front().domain == domain);
        CHECK(descriptor.sides.front().component == 0);
    }
}

TEST_CASE("nosym standard matching descriptors zip both boundary sides",
          "[residual_row_descriptor][eq_matching][bin_ns_nosym_matching]")
{
    Space_bin_ns_nosym space(
        CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
    Scalar P = make_zero_scalar(space);
    system.add_var("P", P);
    system.add_eq_matching(space.NS1, OUTER_BC, "P", -1, nullptr, "P");

    std::vector<ResidualRowDescriptor> descriptors;
    REQUIRE(system.describe_residual_rows(descriptors));
    REQUIRE_FALSE(descriptors.empty());
    for (const ResidualRowDescriptor& descriptor : descriptors) {
        require_field_descriptor(descriptor, 0, 2);
        const ResidualRowCoordinate& local = descriptor.sides[0];
        const ResidualRowCoordinate& remote = descriptor.sides[1];
        CHECK(local.domain == space.NS1);
        CHECK(remote.domain == space.ADAPTED1);
        CHECK(local.component == remote.component);
        CHECK(local.phi_index == remote.phi_index);
    }

    const JacobianParityRowPrediction prediction =
        predict_jacobian_parity_rows(system);
    REQUIRE(prediction.all_rows_available);
    REQUIRE(prediction.unavailable_rows == 0);
    REQUIRE(prediction.ungraded_rows == 0);
    REQUIRE(std::all_of(prediction.sector.begin(), prediction.sector.end(),
                        [](signed char sector) {
                            return sector == -1 || sector == 1;
                        }));
}

TEST_CASE("nosym matching-import descriptors require an explicit parity contract",
          "[residual_row_descriptor][eq_matching_import][parity_mask]")
{
    Space_bin_ns_nosym space(
        CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    const int source = space.ADAPTED1 + 1;

    SECTION("same-field audited import exposes one canonical local side")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        system.add_var("P", P);
        system.add_eq_matching_import(
            source, OUTER_BC, "P=import(P)", -1, nullptr, "P", true);

        std::vector<ResidualRowDescriptor> descriptors;
        REQUIRE(system.describe_residual_rows(descriptors));
        REQUIRE_FALSE(descriptors.empty());
        for (const ResidualRowDescriptor& descriptor : descriptors) {
            require_field_descriptor(descriptor, 0, 1);
            CHECK(descriptor.sides.front().domain == source);
        }
        REQUIRE(predict_jacobian_parity_rows(system).all_rows_available);
    }

    SECTION("mixed-field untagged import fails closed")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        Scalar phi = make_zero_scalar(space, true);
        system.add_var("P", P);
        system.add_var("phi", phi);
        system.add_eq_matching_import(
            source, OUTER_BC, "P=import(phi)", -1, nullptr, "P");

        std::vector<ResidualRowDescriptor> descriptors;
        REQUIRE_FALSE(system.describe_residual_rows(descriptors));
        REQUIRE_FALSE(descriptors.empty());
        for (const ResidualRowDescriptor& descriptor : descriptors) {
            CHECK(descriptor.family == ResidualRowEquationFamily::Unavailable);
            CHECK_FALSE(descriptor.available);
            CHECK(descriptor.explicit_sector == 0);
            CHECK(descriptor.sides.empty());
        }
        const JacobianParityRowPrediction prediction =
            predict_jacobian_parity_rows(system);
        CHECK_FALSE(prediction.all_rows_available);
        CHECK(prediction.unavailable_rows ==
              static_cast<long long>(descriptors.size()));
    }
}

TEST_CASE("nosym first-integral descriptors require an audited constant sector",
          "[residual_row_descriptor][eq_first_integral][parity_mask]")
{
    Space_bin_ns_nosym space(
        CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    const int first_domain = space.NS1;
    const int last_domain = space.ADAPTED1 + 1;

    SECTION("audited same-field contract emits complete domain blocks")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        system.add_var("P", P);
        system.add_eq_first_integral(
            first_domain, last_domain, "P", "P", true);

        std::vector<ResidualRowDescriptor> descriptors;
        REQUIRE(system.describe_residual_rows(descriptors));
        std::size_t row = 0;
        for (int domain = first_domain; domain <= last_domain; ++domain) {
            const int count = zero_scalar_condition_count(space, domain);
            for (int local_row = 0; local_row < count; ++local_row, ++row) {
                REQUIRE(row < descriptors.size());
                require_field_descriptor(descriptors[row], 0, 1);
                CHECK(descriptors[row].sides.front().domain == domain);
                CHECK(descriptors[row].sides.front().component == 0);
            }
        }
        CHECK(row == descriptors.size());
        REQUIRE(predict_jacobian_parity_rows(system).all_rows_available);
    }

    SECTION("opposite-parity constant without a contract fails closed")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        Scalar phi = make_zero_scalar(space, true);
        system.add_var("P", P);
        system.add_var("phi", phi);
        system.add_eq_first_integral(
            first_domain, last_domain, "P", "phi");

        std::vector<ResidualRowDescriptor> descriptors;
        REQUIRE_FALSE(system.describe_residual_rows(descriptors));
        REQUIRE_FALSE(descriptors.empty());
        for (const ResidualRowDescriptor& descriptor : descriptors) {
            CHECK(descriptor.family == ResidualRowEquationFamily::Unavailable);
            CHECK_FALSE(descriptor.available);
            CHECK(descriptor.sides.empty());
        }
        CHECK_FALSE(predict_jacobian_parity_rows(system).all_rows_available);
    }
}

TEST_CASE("nosym velocity-potential descriptors require an audited constant sector",
          "[residual_row_descriptor][eq_vel_pot][parity_mask]")
{
    Space_bin_ns_nosym space(
        CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    SECTION("same-field audited constant exposes volume rows")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        system.add_var("P", P);
        system.add_eq_vel_pot(
            space.NS1, 2, "P=0", "P=0", "P", true);

        std::vector<ResidualRowDescriptor> descriptors;
        REQUIRE(system.describe_residual_rows(descriptors));
        REQUIRE_FALSE(descriptors.empty());
        for (const ResidualRowDescriptor& descriptor : descriptors) {
            require_field_descriptor(descriptor, 0, 1);
            CHECK(descriptor.sides.front().domain == space.NS1);
            CHECK(descriptor.sides.front().component == 0);
        }
        REQUIRE(predict_jacobian_parity_rows(system).all_rows_available);
    }

    SECTION("opposite-parity constant without a contract fails closed")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        Scalar phi = make_zero_scalar(space, true);
        system.add_var("P", P);
        system.add_var("phi", phi);
        system.add_eq_vel_pot(
            space.NS1, 2, "P=0", "phi=0", "P");

        std::vector<ResidualRowDescriptor> descriptors;
        REQUIRE_FALSE(system.describe_residual_rows(descriptors));
        REQUIRE_FALSE(descriptors.empty());
        for (const ResidualRowDescriptor& descriptor : descriptors) {
            CHECK(descriptor.family == ResidualRowEquationFamily::Unavailable);
            CHECK_FALSE(descriptor.available);
            CHECK(descriptor.sides.empty());
        }
        CHECK_FALSE(predict_jacobian_parity_rows(system).all_rows_available);
    }
}

TEST_CASE("Eq_int reflection metadata is explicit, ordered, and fail-closed",
          "[residual_row_descriptor][eq_int][parity_mask]")
{
    Space_bin_ns_nosym space(
        CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    SECTION("two tagged integral rows prefix field rows in insertion order")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        system.add_var("P", P);
        space.add_eq_int_volume(
            system, space.NS1, space.ADAPTED1, "integvolume(P)=0");
        system.set_last_eq_int_reflection_sector(+1);
        space.add_eq_int_volume(
            system, space.NS2, space.ADAPTED2, "integvolume(P)=0");
        system.set_last_eq_int_reflection_sector(-1);
        system.add_eq_bc(space.get_nbr_domains() - 1, OUTER_BC,
                         "P=0", -1, nullptr, "P");

        std::vector<ResidualRowDescriptor> descriptors;
        REQUIRE(system.describe_residual_rows(descriptors));
        REQUIRE(descriptors.size() > 2);
        for (int row = 0; row < 2; ++row) {
            const ResidualRowDescriptor& descriptor = descriptors[row];
            CHECK(descriptor.family == ResidualRowEquationFamily::Integral);
            CHECK(descriptor.equation_index == row);
            CHECK(descriptor.available);
            CHECK(descriptor.explicit_sector == (row == 0 ? +1 : -1));
            CHECK(descriptor.sides.empty());
        }
        for (std::size_t row = 2; row < descriptors.size(); ++row)
            require_field_descriptor(descriptors[row], 0, 1);

        const JacobianParityRowPrediction prediction =
            predict_jacobian_parity_rows(system);
        REQUIRE(prediction.all_rows_available);
        CHECK(prediction.sector[0] == +1);
        CHECK(prediction.sector[1] == -1);
    }

    SECTION("invalid, absent, and conflicting tags are refused")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        system.add_var("P", P);
        REQUIRE_THROWS_AS(
            system.set_last_eq_int_reflection_sector(+1), KadathError);
        space.add_eq_int_volume(
            system, space.NS1, space.ADAPTED1, "integvolume(P)=0");
        REQUIRE_THROWS_AS(
            system.set_last_eq_int_reflection_sector(0), KadathError);
        REQUIRE_THROWS_AS(
            system.set_last_eq_int_reflection_sector(2), KadathError);
        system.set_last_eq_int_reflection_sector(+1);
        REQUIRE_NOTHROW(system.set_last_eq_int_reflection_sector(+1));
        REQUIRE_THROWS_AS(
            system.set_last_eq_int_reflection_sector(-1), KadathError);
    }

    SECTION("untagged integral metadata remains unavailable")
    {
        System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
        Scalar P = make_zero_scalar(space);
        system.add_var("P", P);
        space.add_eq_int_volume(
            system, space.NS1, space.ADAPTED1, "integvolume(P)=0");

        std::vector<ResidualRowDescriptor> descriptors;
        REQUIRE_FALSE(system.describe_residual_rows(descriptors));
        REQUIRE(descriptors.size() == 1);
        CHECK(descriptors.front().family ==
              ResidualRowEquationFamily::Unavailable);
        CHECK_FALSE(descriptors.front().available);
        CHECK(descriptors.front().explicit_sector == 0);
        CHECK(descriptors.front().sides.empty());

        const JacobianParityRowPrediction prediction =
            predict_jacobian_parity_rows(system);
        CHECK_FALSE(prediction.all_rows_available);
        CHECK(prediction.unavailable_rows == 1);
        REQUIRE(prediction.sector.size() == 1);
        CHECK(prediction.sector.front() == 0);
    }
}

TEST_CASE("Space_bin_ns_nosym scalar matching embeds z-symmetric Space_bin_ns data",
          "[bin_ns_nosym_matching]")
{
    Space_bin_ns plane(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    REQUIRE(plane.NS2 == nosym.NS2);

    const double plane_norm = scalar_matching_norm(plane, plane.NS2, "u");
    const double nosym_norm = scalar_matching_norm(nosym, nosym.NS2, "u");

    INFO("Space_bin_ns       NS2 matching ||r||_inf = " << plane_norm);
    INFO("Space_bin_ns_nosym NS2 matching ||r||_inf = " << nosym_norm);

    REQUIRE(std::isfinite(plane_norm));
    REQUIRE(std::isfinite(nosym_norm));
    REQUIRE_THAT(nosym_norm, WithinAbs(plane_norm, 1e-11));
}

TEST_CASE("Space_bin_ns_nosym scalar normal-derivative matching embeds z-symmetric Space_bin_ns data",
          "[bin_ns_nosym_matching]")
{
    Space_bin_ns plane(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    REQUIRE(plane.NS2 == nosym.NS2);

    const double plane_norm = scalar_matching_norm(plane, plane.NS2, "dn(u)");
    const double nosym_norm = scalar_matching_norm(nosym, nosym.NS2, "dn(u)");

    INFO("Space_bin_ns       NS2 dn matching ||r||_inf = " << plane_norm);
    INFO("Space_bin_ns_nosym NS2 dn matching ||r||_inf = " << nosym_norm);

    REQUIRE(std::isfinite(plane_norm));
    REQUIRE(std::isfinite(nosym_norm));
    REQUIRE_THAT(nosym_norm, WithinAbs(plane_norm, 1e-10));
}

TEST_CASE("Space_bin_ns_nosym velocity-potential tau embeds z-symmetric Space_bin_ns data",
          "[bin_ns_nosym_matching]")
{
    Space_bin_ns plane(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    REQUIRE(plane.NS1 == nosym.NS1);
    REQUIRE(plane.NS2 == nosym.NS2);

    const double plane_ns1_norm = velocity_potential_norm(plane, plane.NS1);
    const double nosym_ns1_norm = velocity_potential_norm(nosym, nosym.NS1);
    const double plane_ns2_norm = velocity_potential_norm(plane, plane.NS2);
    const double nosym_ns2_norm = velocity_potential_norm(nosym, nosym.NS2);

    INFO("Space_bin_ns       NS1 Eq_vel_pot ||r||_inf = " << plane_ns1_norm);
    INFO("Space_bin_ns_nosym NS1 Eq_vel_pot ||r||_inf = " << nosym_ns1_norm);
    INFO("Space_bin_ns       NS2 Eq_vel_pot ||r||_inf = " << plane_ns2_norm);
    INFO("Space_bin_ns_nosym NS2 Eq_vel_pot ||r||_inf = " << nosym_ns2_norm);

    REQUIRE(std::isfinite(plane_ns1_norm));
    REQUIRE(std::isfinite(nosym_ns1_norm));
    REQUIRE(std::isfinite(plane_ns2_norm));
    REQUIRE(std::isfinite(nosym_ns2_norm));
    REQUIRE_THAT(nosym_ns1_norm, WithinAbs(plane_ns1_norm, 1e-10));
    REQUIRE_THAT(nosym_ns2_norm, WithinAbs(plane_ns2_norm, 1e-10));
}

TEST_CASE("Space_bin_ns_nosym full scalar equations embed z-symmetric Space_bin_ns data",
          "[bin_ns_nosym_matching]")
{
    Space_bin_ns plane(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    const double plane_norm = full_space_scalar_norm(plane);
    const double nosym_norm = full_space_scalar_norm(nosym);

    INFO("Space_bin_ns       full scalar ||r||_inf = " << plane_norm);
    INFO("Space_bin_ns_nosym full scalar ||r||_inf = " << nosym_norm);

    REQUIRE(std::isfinite(plane_norm));
    REQUIRE(std::isfinite(nosym_norm));
    REQUIRE_THAT(nosym_norm, WithinAbs(plane_norm, 1e-10));
}

TEST_CASE("Space_bin_ns_nosym first-integral rows embed z-symmetric Space_bin_ns data",
          "[bin_ns_nosym_matching]")
{
    Space_bin_ns plane(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    const double plane_ns1_norm = first_integral_norm(plane, plane.NS1, plane.ADAPTED1);
    const double nosym_ns1_norm = first_integral_norm(nosym, nosym.NS1, nosym.ADAPTED1);
    const double plane_ns2_norm = first_integral_norm(plane, plane.NS2, plane.ADAPTED2);
    const double nosym_ns2_norm = first_integral_norm(nosym, nosym.NS2, nosym.ADAPTED2);
    const double nosym_ns1_m_order_norm = first_integral_norm(nosym, nosym.NS1, nosym.ADAPTED1, true);
    const double nosym_ns2_m_order_norm = first_integral_norm(nosym, nosym.NS2, nosym.ADAPTED2, true);

    INFO("Space_bin_ns       NS1 Eq_first_integral ||r||_inf = " << plane_ns1_norm);
    INFO("Space_bin_ns_nosym NS1 Eq_first_integral ||r||_inf = " << nosym_ns1_norm);
    INFO("Space_bin_ns       NS2 Eq_first_integral ||r||_inf = " << plane_ns2_norm);
    INFO("Space_bin_ns_nosym NS2 Eq_first_integral ||r||_inf = " << nosym_ns2_norm);
    INFO("Space_bin_ns_nosym NS1 m-order Eq_first_integral ||r||_inf = " << nosym_ns1_m_order_norm);
    INFO("Space_bin_ns_nosym NS2 m-order Eq_first_integral ||r||_inf = " << nosym_ns2_m_order_norm);

    REQUIRE(std::isfinite(plane_ns1_norm));
    REQUIRE(std::isfinite(nosym_ns1_norm));
    REQUIRE(std::isfinite(plane_ns2_norm));
    REQUIRE(std::isfinite(nosym_ns2_norm));
    REQUIRE(std::isfinite(nosym_ns1_m_order_norm));
    REQUIRE(std::isfinite(nosym_ns2_m_order_norm));
    REQUIRE_THAT(nosym_ns1_norm, WithinAbs(plane_ns1_norm, 1e-10));
    REQUIRE_THAT(nosym_ns2_norm, WithinAbs(plane_ns2_norm, 1e-10));
    REQUIRE_THAT(nosym_ns1_m_order_norm, WithinAbs(nosym_ns1_norm, 1e-10));
    REQUIRE_THAT(nosym_ns2_m_order_norm, WithinAbs(nosym_ns2_norm, 1e-10));
}

TEST_CASE("Space_bin_ns_nosym first-integral rows count zero adapted-domain residuals",
          "[bin_ns_nosym_matching]")
{
    Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    const int adapted1_conditions = zero_scalar_condition_count(nosym, nosym.ADAPTED1);
    const int adapted2_conditions = zero_scalar_condition_count(nosym, nosym.ADAPTED2);

    INFO("Space_bin_ns_nosym zero ADAPTED1 scalar rows = " << adapted1_conditions);
    INFO("Space_bin_ns_nosym zero ADAPTED2 scalar rows = " << adapted2_conditions);

    REQUIRE(adapted1_conditions > 0);
    REQUIRE(adapted2_conditions > 0);
}

TEST_CASE("Space_bin_ns_nosym integral equations embed z-symmetric Space_bin_ns data",
          "[bin_ns_nosym_matching]")
{
    Space_bin_ns plane(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    const double plane_mb1 = volume_integral_residual(plane, plane.NS1, plane.ADAPTED1);
    const double nosym_mb1 = volume_integral_residual(nosym, nosym.NS1, nosym.ADAPTED1);
    const double plane_mb2 = volume_integral_residual(plane, plane.NS2, plane.ADAPTED2);
    const double nosym_mb2 = volume_integral_residual(nosym, nosym.NS2, nosym.ADAPTED2);
    const double plane_inf = infinity_integral_residual(plane);
    const double nosym_inf = infinity_integral_residual(nosym);
    const double plane_spin1 = outer_sphere_one_integral_residual(plane);
    const double nosym_spin1 = outer_sphere_one_integral_residual(nosym);
    const double plane_spin2 = outer_sphere_two_integral_residual(plane);
    const double nosym_spin2 = outer_sphere_two_integral_residual(nosym);

    INFO("Space_bin_ns       NS1 volume integral = " << plane_mb1);
    INFO("Space_bin_ns_nosym NS1 volume integral = " << nosym_mb1);
    INFO("Space_bin_ns       NS2 volume integral = " << plane_mb2);
    INFO("Space_bin_ns_nosym NS2 volume integral = " << nosym_mb2);
    INFO("Space_bin_ns       infinity integral = " << plane_inf);
    INFO("Space_bin_ns_nosym infinity integral = " << nosym_inf);
    INFO("Space_bin_ns       NS1 outer-sphere integral = " << plane_spin1);
    INFO("Space_bin_ns_nosym NS1 outer-sphere integral = " << nosym_spin1);
    INFO("Space_bin_ns       NS2 outer-sphere integral = " << plane_spin2);
    INFO("Space_bin_ns_nosym NS2 outer-sphere integral = " << nosym_spin2);

    REQUIRE_THAT(nosym_mb1, WithinAbs(plane_mb1, 1e-10));
    REQUIRE_THAT(nosym_mb2, WithinAbs(plane_mb2, 1e-10));
    REQUIRE_THAT(nosym_inf, WithinAbs(plane_inf, 1e-10));
    REQUIRE_THAT(nosym_spin1, WithinAbs(plane_spin1, 1e-10));
    REQUIRE_THAT(nosym_spin2, WithinAbs(plane_spin2, 1e-10));
}

TEST_CASE("System_of_eqs selectively evaluates one integral residual and its definition closure",
          "[bin_ns_nosym_matching][system_of_eqs][selective_eq_int]")
{
    Space_bin_ns_nosym space(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    System_of_eqs sys(space, 0, space.get_nbr_domains() - 1);
    Scalar u = make_z_symmetric_scalar(space);
    Param counted_parameters;

    sys.add_var("u", u);
    sys.add_ope("counted", counted_identity, &counted_parameters);
    sys.add_def("base = u * u");
    sys.add_def("target = base + u");
    sys.add_def("unrelated = counted(u)");
    space.add_eq_int_volume(sys, space.NS1, space.ADAPTED1, "integvolume(target) = 0");
    space.add_eq_int_volume(sys, space.NS2, space.ADAPTED2, "integvolume(unrelated) = 0");

    unrelated_definition_calls = 0;
    REQUIRE(sys.get_nbr_conditions() == -1);
    const double selected_initial = sys.sec_member_eq_int(0);
    REQUIRE(unrelated_definition_calls == 0);
    REQUIRE(sys.get_nbr_conditions() == -1);

    const Array<double> full_initial = sys.sec_member();
    REQUIRE(unrelated_definition_calls > 0);
    REQUIRE(full_initial.get_nbr() == 2);
    REQUIRE(same_double_bits(selected_initial, full_initial(0)));

    for (int d = space.NS1; d <= space.ADAPTED1; ++d)
        u.set_domain(d) = u(d) + 0.125;
    u.std_base();
    u.coef();

    unrelated_definition_calls = 0;
    const double selected_mutated = sys.sec_member_eq_int(0);
    REQUIRE(unrelated_definition_calls == 0);
    REQUIRE_FALSE(same_double_bits(selected_initial, selected_mutated));

    const Array<double> full_mutated = sys.sec_member();
    REQUIRE(unrelated_definition_calls > 0);
    REQUIRE(same_double_bits(selected_mutated, full_mutated(0)));

    const double selected_repeated = sys.sec_member_eq_int(0);
    const Array<double> full_repeated = sys.sec_member();
    REQUIRE(same_double_bits(selected_mutated, selected_repeated));
    REQUIRE(full_mutated.get_nbr() == full_repeated.get_nbr());
    REQUIRE(std::memcmp(full_mutated.get_data(), full_repeated.get_data(),
                        sizeof(double) * static_cast<std::size_t>(full_mutated.get_nbr())) == 0);

    REQUIRE_THROWS_AS(sys.sec_member_eq_int(-1), KadathError);
    REQUIRE_THROWS_AS(sys.sec_member_eq_int(2), KadathError);
}

TEST_CASE("Space_bin_ns_nosym full vector equations embed z-symmetric Space_bin_ns data",
          "[bin_ns_nosym_matching]")
{
    Space_bin_ns plane(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);
    Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), 5);

    const double plane_norm = full_space_vector_norm(plane);
    const double nosym_norm = full_space_vector_norm(nosym);

    INFO("Space_bin_ns       full vector ||r||_inf = " << plane_norm);
    INFO("Space_bin_ns_nosym full vector ||r||_inf = " << nosym_norm);

    REQUIRE(std::isfinite(plane_norm));
    REQUIRE(std::isfinite(nosym_norm));
    REQUIRE_THAT(nosym_norm, WithinAbs(plane_norm, 1e-10));
}

TEST_CASE("Space_bin_ns_nosym bispheric phi resolution is N-1 (Phillipe; same as sym spheric phi)",
          "[bin_ns_nosym_matching]")
{
    // Phillipe convention: the bispheric outer phi count is N-1 for the no-symmetry
    // space (matching the reference Bispheric_nosym), one less than the z-symmetric
    // bispheric phi (= N). The previous 2*(N-1) "mode-embedding" doubling is dropped.
    constexpr int nr = 7;
    Space_bin_ns plane(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), nr);
    Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), nr);

    const int plane_half_phi_points = plane.get_domain(plane.OUTER)->get_nbr_points()(2);
    const int nosym_full_phi_points = nosym.get_domain(nosym.OUTER)->get_nbr_points()(2);
    const int nosym_phi_coefs = nosym.get_domain(nosym.OUTER)->get_nbr_coefs()(2);

    REQUIRE(plane_half_phi_points == nr);
    REQUIRE(nosym_full_phi_points == plane_half_phi_points - 1);
    REQUIRE(nosym_phi_coefs == nosym_full_phi_points + 2);
}

TEST_CASE("Space_bin_ns_nosym spherical theta resolution is N (Phillipe; same as sym)",
          "[bin_ns_nosym_matching]")
{
    // Phillipe convention: theta = N for the no-symmetry spheric/adapted domains,
    // identical to the z-symmetric space. The previous 2N-1 / 2*(N-2)-1 full-theta
    // counts are dropped. Bispheric theta stays N for both symmetries.
    constexpr int nr = 7;
    Space_bin_ns plane(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), nr);
    Space_bin_ns_nosym nosym(CHEB_TYPE, 12.0, ns_bounds(), ns_bounds(), outer_bounds(), nr);

    const int plane_half_theta_points = plane.get_domain(plane.NS1)->get_nbr_points()(1);
    const int nosym_full_theta_points = nosym.get_domain(nosym.NS1)->get_nbr_points()(1);
    const int nosym_compact_theta_points = nosym.get_domain(nosym.get_nbr_domains() - 1)->get_nbr_points()(1);
    const int nosym_bispheric_theta_points = nosym.get_domain(nosym.OUTER)->get_nbr_points()(1);

    REQUIRE(plane_half_theta_points == nr);
    REQUIRE(nosym_full_theta_points == plane_half_theta_points);
    REQUIRE(nosym_compact_theta_points == nosym_full_theta_points);
    REQUIRE(nosym_bispheric_theta_points == nr);
}

TEST_CASE("Space_bin_ns_nosym shelled (n_shells>0) Laplace system is square per-direction",
          "[bin_ns_nosym_matching][angular-p]")
{
    // Regression for the shells-after-adapted matching layout. With NSHELLS=1 the
    // star block is nucleus(0/4), adapted-outer(1/5), adapted-inner(2/6),
    // shell(3/7); bispheric 8-12; compact 13. The retired shells-before layout
    // double-wired the adapted pair and dropped the real shell, leaving the tau
    // system non-square for NSHELLS>0. u==1 solves lap(u)=0 exactly, so a square
    // system drives every residual row (matching included) to roundoff.
    const std::vector<double> shelled{1.0, 2.0, 3.0, 4.0};  // rin, rmid, shell, rout -> n_shells=1
    const std::vector<double> obounds{10.0};

    const auto require_square = [](Space_bin_ns_nosym& space) {
        Scalar u(space);
        u = 1.;
        u.std_base();
        System_of_eqs sys(space);
        sys.add_var("u", u);
        space.add_eq(sys, "lap(u) = 0", "u", "dn(u)");
        sys.add_eq_bc(space.get_nbr_domains() - 1, OUTER_BC, "u = 1");
        // Close the adapted-surface shape unknowns with a BC on the variable.
        sys.add_eq_bc(space.ADAPTED1, OUTER_BC, "u = 1");
        sys.add_eq_bc(space.ADAPTED2, OUTER_BC, "u = 1");
        Array<double> residual = sys.sec_member();
        INFO("unknowns=" << sys.get_nbr_unknowns() << " conditions=" << sys.get_nbr_conditions());
        REQUIRE(sys.get_nbr_unknowns() == sys.get_nbr_conditions());
        REQUIRE(residual_infinity_norm(residual) < 1.e-10);
    };

    SECTION("uniform resolution (conforming seams)")
    {
        Space_bin_ns_nosym space(CHEB_TYPE, 12.0, shelled, shelled, obounds, 7);
        REQUIRE(space.get_nbr_domains() == 14);
        require_square(space);
    }
    SECTION("per-domain angular refinement (import seams across the ntheta jump)")
    {
        // Lift each star's adapted pair (1,2 and 5,6) to ntheta=9/nphi=8 while the
        // nuclei and shells stay at the base 7/6 -> non-conforming seams at
        // nucleus<->adapted and adapted-inner<->shell, coupled with import.
        std::vector<Dim_array> res;
        for (int d = 0; d < 14; ++d) {
            const bool bispheric = (d >= 8 && d <= 12);
            Dim_array r(3);
            r.set(0) = 7;
            r.set(1) = 7;
            r.set(2) = bispheric ? 8 : 6;
            res.push_back(r);
        }
        for (int d : {1, 2, 5, 6}) {
            res[static_cast<std::size_t>(d)].set(1) = 9;
            res[static_cast<std::size_t>(d)].set(2) = 8;
        }
        Space_bin_ns_nosym space(CHEB_TYPE, 12.0, shelled, shelled, obounds, res);
        REQUIRE(space.get_nbr_domains() == 14);
        require_square(space);
    }
}
