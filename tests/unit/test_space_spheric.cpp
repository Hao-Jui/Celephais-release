#include <catch2/catch_test_macros.hpp>
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"

using namespace Kadath;

namespace {

Space_spheric make_spheric_space()
{
    Point center(3);
    center.set(1) = 0;
    center.set(2) = 0;
    center.set(3) = 0;

    Dim_array res(3);
    res.set(0) = 9;
    res.set(1) = 5;
    res.set(2) = 4;

    Dim_array bounds_dim(1);
    bounds_dim.set(0) = 2;
    Array<double> bounds(bounds_dim);
    bounds.set(0) = 1.0;
    bounds.set(1) = 10.0;
    return Space_spheric(CHEB_TYPE, center, res, bounds);
}

void require_same_scalar(const Scalar& actual, const Scalar& expected)
{
    REQUIRE(actual.get_nbr_domains() == expected.get_nbr_domains());
    for (int dom = 0; dom < actual.get_nbr_domains(); ++dom) {
        if (actual(dom).check_if_zero() || expected(dom).check_if_zero()) {
            REQUIRE(actual(dom).check_if_zero());
            REQUIRE(expected(dom).check_if_zero());
            continue;
        }
        Index pos(actual.get_space().get_domain(dom)->get_nbr_points());
        do {
            REQUIRE(actual(dom)(pos) == expected(dom)(pos));
        } while (pos.inc());
    }
}

void require_same_vector(const Vector& actual, const Vector& expected)
{
    REQUIRE(actual.get_space().get_nbr_domains() == expected.get_space().get_nbr_domains());
    for (int dom = 0; dom < actual.get_space().get_nbr_domains(); ++dom)
        REQUIRE(actual.get_basis().get_basis(dom) == expected.get_basis().get_basis(dom));
    for (int component = 1; component <= 3; ++component)
        require_same_scalar(actual(component), expected(component));
}

} // namespace

TEST_CASE("Space_spheric construction", "[space]") {
    Point center(3);
    center.set(1) = 0; center.set(2) = 0; center.set(3) = 0;

    Dim_array res(3);
    res.set(0) = 9; res.set(1) = 5; res.set(2) = 4;

    Dim_array bounds_dim(1);
    bounds_dim.set(0) = 2;
    Array<double> bounds(bounds_dim);
    bounds.set(0) = 1.0;
    bounds.set(1) = 10.0;

    Space_spheric space(CHEB_TYPE, center, res, bounds);
    REQUIRE(space.get_nbr_domains() == 3);
}

TEST_CASE("coord_fields enum array indexing", "[coord_fields]") {
    REQUIRE(cv_names[to_int(coord_vector::GLOBAL_ROT)] == "mg^i");
    REQUIRE(cv_names[to_int(coord_vector::EX)] == "ex^i");
    REQUIRE(cs_names[to_int(coord_scalar::R_BCO1)] == "rm");
    REQUIRE(cs_names[to_int(coord_scalar::R_BCO2)] == "rp");
}

TEST_CASE("coordinate vectors can be activated stage by stage", "[coord_fields]")
{
    auto space = make_spheric_space();
    vec_ary_t fields{};

    activate_coordinate_vector(fields, space, coord_vector::GLOBAL_ROT);
    activate_coordinate_vector(fields, space, coord_vector::BCO1_ROTx);
    activate_coordinate_vector(fields, space, coord_vector::BCO1_ROTz);
    activate_coordinate_vector(fields, space, coord_vector::S_BCO1);
    activate_coordinate_vector(fields, space, coord_vector::S_INF);

    REQUIRE(fields[to_int(coord_vector::GLOBAL_ROT)]);
    REQUIRE(fields[to_int(coord_vector::BCO1_ROTx)]);
    REQUIRE(fields[to_int(coord_vector::BCO1_ROTz)]);
    REQUIRE(fields[to_int(coord_vector::S_BCO1)]);
    REQUIRE(fields[to_int(coord_vector::S_INF)]);
    REQUIRE_FALSE(fields[to_int(coord_vector::EX)]);
    REQUIRE_FALSE(fields[to_int(coord_vector::EY)]);
    REQUIRE_FALSE(fields[to_int(coord_vector::EZ)]);

    activate_coordinate_vector(fields, space, coord_vector::EX);
    activate_coordinate_vector(fields, space, coord_vector::EY);
    REQUIRE(fields[to_int(coord_vector::EX)]);
    REQUIRE(fields[to_int(coord_vector::EY)]);
    REQUIRE_FALSE(fields[to_int(coord_vector::EZ)]);

    const auto legacy_defaults = default_co_vector_ary(space);
    REQUIRE(legacy_defaults[to_int(coord_vector::EX)]);
    REQUIRE(legacy_defaults[to_int(coord_vector::EY)]);
    REQUIRE(legacy_defaults[to_int(coord_vector::EZ)]);
}

TEST_CASE("compact-object coordinate refresh matches the general refresh", "[coord_fields]")
{
    auto space = make_spheric_space();
    CoordFields<Space_spheric> generator(space);
    auto expected_vectors = default_binary_vector_ary(space);
    auto actual_vectors = default_binary_vector_ary(space);
    scalar_ary_t expected_scalars{};
    scalar_ary_t actual_scalars{};
    expected_scalars[to_int(coord_scalar::R_BCO1)] = Scalar(space);
    expected_scalars[to_int(coord_scalar::R_BCO2)] = Scalar(space);
    actual_scalars[to_int(coord_scalar::R_BCO1)] = Scalar(space);
    actual_scalars[to_int(coord_scalar::R_BCO2)] = Scalar(space);

    constexpr double origin = 0.375;
    update_fields(generator, expected_vectors, expected_scalars, origin, origin, 0.);
    update_fields_co(generator, actual_vectors, actual_scalars, origin);
    // Exercise the persistent in-place refresh path a second time.
    update_fields_co(generator, actual_vectors, actual_scalars, origin);

    for (int i = 0; i < NUM_VECTORS_V; ++i)
        require_same_vector(*actual_vectors[i], *expected_vectors[i]);
    for (int i = 0; i < NUM_SCALARS_V; ++i)
        require_same_scalar(*actual_scalars[i], *expected_scalars[i]);
}

TEST_CASE("binary coordinate refresh preserves independently shifted fields", "[coord_fields]")
{
    auto space = make_spheric_space();
    CoordFields<Space_spheric> generator(space);
    auto expected_vectors = default_binary_vector_ary(space);
    auto actual_vectors = default_binary_vector_ary(space);
    scalar_ary_t expected_scalars{};
    scalar_ary_t actual_scalars{};
    expected_scalars[to_int(coord_scalar::R_BCO1)] = Scalar(space);
    expected_scalars[to_int(coord_scalar::R_BCO2)] = Scalar(space);
    actual_scalars[to_int(coord_scalar::R_BCO1)] = Scalar(space);
    actual_scalars[to_int(coord_scalar::R_BCO2)] = Scalar(space);

    constexpr double global_origin = 0.125;
    constexpr double first_origin = -0.75;
    constexpr double second_origin = 0.625;
    *expected_vectors[to_int(coord_vector::GLOBAL_ROT)] = generator.rot_z(global_origin);
    *expected_vectors[to_int(coord_vector::BCO1_ROTx)] = generator.rot_x(first_origin);
    *expected_vectors[to_int(coord_vector::BCO1_ROTz)] = generator.rot_z(first_origin);
    *expected_vectors[to_int(coord_vector::BCO2_ROTx)] = generator.rot_x(second_origin);
    *expected_vectors[to_int(coord_vector::BCO2_ROTz)] = generator.rot_z(second_origin);
    *expected_vectors[to_int(coord_vector::EX)] = generator.e_cart<COV>(1);
    *expected_vectors[to_int(coord_vector::EY)] = generator.e_cart<COV>(2);
    *expected_vectors[to_int(coord_vector::EZ)] = generator.e_cart<COV>(3);
    *expected_vectors[to_int(coord_vector::S_BCO1)] = generator.e_rad<COV>(first_origin);
    *expected_vectors[to_int(coord_vector::S_BCO2)] = generator.e_rad<COV>(second_origin);
    *expected_vectors[to_int(coord_vector::S_INF)] = generator.e_rad<COV>(global_origin);
    *expected_scalars[to_int(coord_scalar::R_BCO1)] = generator.radius(first_origin);
    *expected_scalars[to_int(coord_scalar::R_BCO2)] = generator.radius(second_origin);

    update_fields(generator, actual_vectors, actual_scalars, global_origin, first_origin, second_origin);
    update_fields(generator, actual_vectors, actual_scalars, global_origin, first_origin, second_origin);

    for (int i = 0; i < NUM_VECTORS_V; ++i)
        require_same_vector(*actual_vectors[i], *expected_vectors[i]);
    for (int i = 0; i < NUM_SCALARS_V; ++i)
        require_same_scalar(*actual_scalars[i], *expected_scalars[i]);
}

TEST_CASE("coordinate-system bindings resolve only active constants", "[coord_fields]")
{
    auto space = make_spheric_space();
    vec_ary_t fields{};
    activate_coordinate_vector(fields, space, coord_vector::GLOBAL_ROT);
    activate_coordinate_vector(fields, space, coord_vector::BCO1_ROTx);
    activate_coordinate_vector(fields, space, coord_vector::BCO1_ROTz);
    activate_coordinate_vector(fields, space, coord_vector::S_BCO1);
    activate_coordinate_vector(fields, space, coord_vector::S_INF);
    CoordFields<Space_spheric> generator(space);
    update_fields_co(generator, fields, {}, 0.);

    System_of_eqs syst(space, 0, space.get_nbr_domains() - 1);
    syst.add_cst("mg", *fields[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_cst("mmx", *fields[to_int(coord_vector::BCO1_ROTx)]);
    syst.add_cst("mmz", *fields[to_int(coord_vector::BCO1_ROTz)]);
    syst.add_cst("sm", *fields[to_int(coord_vector::S_BCO1)]);
    syst.add_cst("einf", *fields[to_int(coord_vector::S_INF)]);

    const auto binding = bind_coordinate_fields(syst, fields);
    REQUIRE(binding.vector_constants[to_int(coord_vector::GLOBAL_ROT)] >= 0);
    REQUIRE(binding.vector_constants[to_int(coord_vector::BCO1_ROTx)] >= 0);
    REQUIRE(binding.vector_constants[to_int(coord_vector::BCO1_ROTz)] >= 0);
    REQUIRE(binding.vector_constants[to_int(coord_vector::S_BCO1)] >= 0);
    REQUIRE(binding.vector_constants[to_int(coord_vector::S_INF)] >= 0);
    REQUIRE(binding.vector_constants[to_int(coord_vector::EX)] == -1);
    REQUIRE(binding.vector_constants[to_int(coord_vector::EY)] == -1);
    REQUIRE(binding.vector_constants[to_int(coord_vector::EZ)] == -1);
}
