#include <catch2/catch_test_macros.hpp>

#include <filesystem>
namespace fs = std::filesystem;

#include "Apps/Formalism/Shared/PreBinary/bns_separation_seed.hpp"
#include "Apps/Formalism/Shared/scalar_point_batch.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>

using namespace Kadath;

namespace {

Space_bin_ns_nosym make_bns_space(double separation)
{
    const std::vector<double> ns1_bounds{0.50, 1.00, 1.25, 1.50};
    const std::vector<double> ns2_bounds{0.55, 1.05, 1.35, 1.65};
    const std::vector<double> outer_bounds{8.0, 12.0};
    return Space_bin_ns_nosym(
        CHEB_TYPE, separation, ns1_bounds, ns2_bounds, outer_bounds, 5);
}

void fill_domain_polynomial(Scalar& field, double offset)
{
    field.set_in_conf();
    field.allocate_conf();
    const Space& space = field.get_space();
    for (int dom = 0; dom < space.get_nbr_domains(); ++dom) {
        const Dim_array points = space.get_domain(dom)->get_nbr_points();
        Index pos(points);
        do {
            field.set_domain(dom).set(pos) =
                offset + 0.125 * dom + 0.01 * pos(0) - 0.002 * pos(1) + 0.0003 * pos(2);
        } while (pos.inc());
    }
    field.std_base();
}

void fill_domain_constants(Scalar& field)
{
    field.set_in_conf();
    field.allocate_conf();
    const Space& space = field.get_space();
    for (int dom = 0; dom < space.get_nbr_domains(); ++dom) {
        Index pos(space.get_domain(dom)->get_nbr_points());
        do {
            field.set_domain(dom).set(pos) = static_cast<double>(dom + 1);
        } while (pos.inc());
    }
    field.std_base();
}

void require_bitwise_equal(const Scalar& actual, const Scalar& expected)
{
    REQUIRE(actual.get_nbr_domains() == expected.get_nbr_domains());
    for (int dom = 0; dom < actual.get_nbr_domains(); ++dom) {
        REQUIRE(actual.at(dom).check_if_zero() == expected.at(dom).check_if_zero());
        if (actual.at(dom).check_if_zero())
            continue;
        Index pos(actual.get_space().get_domain(dom)->get_nbr_points());
        do {
            REQUIRE(std::bit_cast<std::uint64_t>(actual(dom)(pos)) ==
                    std::bit_cast<std::uint64_t>(expected(dom)(pos)));
        } while (pos.inc());
    }
}

void cold_scalar_oracle(
    Scalar& background, Scalar& weighted, Scalar& accumulated,
    const Scalar& background1, const Scalar& background2,
    const Scalar& weighted1, const Scalar& weighted2,
    const Scalar& accumulated1,
    const std::array<double, 2>& centers,
    const std::array<double, 2>& invw4)
{
    const Space& space = background.get_space();
    const int ndom = space.get_nbr_domains();
    for (int dom = 0; dom < ndom; ++dom) {
        const Domain& domain = *space.get_domain(dom);
        Index pos(domain.get_nbr_points());
        do {
            const double x = domain.get_cart(1)(pos);
            const double y = domain.get_cart(2)(pos);
            const double z = domain.get_cart(3)(pos);

            Point point1(3);
            point1.set(1) = x - centers[0];
            point1.set(2) = y;
            point1.set(3) = z;
            const double r2 = y * y + z * z;
            const double r2_1 = (x - centers[0]) * (x - centers[0]) + r2;
            const double r4_1 = r2_1 * r2_1;
            const double decay1 = std::exp(-r4_1 * invw4[0]);

            Point point2(3);
            point2.set(1) = x - centers[1];
            point2.set(2) = y;
            point2.set(3) = z;
            const double r2_2 = (x - centers[1]) * (x - centers[1]) + r2;
            const double r4_2 = r2_2 * r2_2;
            const double decay2 = std::exp(-r4_2 * invw4[1]);

            if (dom < ndom - 1) {
                background.set_domain(dom).set(pos) =
                    1. + decay1 * (background1.val_point(point1) - 1.)
                       + decay2 * (background2.val_point(point2) - 1.);
                weighted.set_domain(dom).set(pos) =
                    decay1 * weighted1.val_point(point1)
                  + decay2 * weighted2.val_point(point2);
                accumulated.set_domain(dom).set(pos) = 0.;
                accumulated.set_domain(dom).set(pos) +=
                    decay1 * accumulated1.val_point(point1);
            } else {
                background.set_domain(dom).set(pos) = 1.;
            }
        } while (pos.inc());
    }
}

} // namespace

TEST_CASE("Prepared BNS scalar point batch matches Scalar::val_point",
          "[bns][field-transfer][batch]")
{
    Space_bin_ns_nosym source_space = make_bns_space(8.0);
    Scalar source_first(source_space);
    Scalar source_second(source_space);
    Scalar source_zero(source_space);
    fill_domain_polynomial(source_first, 1.25);
    fill_domain_polynomial(source_second, -0.75);
    source_zero = 0.;

    const std::array<const Scalar*, 3> fields{
        &source_first, &source_second, &source_zero};
    bns_field_transfer::scalar_source_batch batch(fields);
    batch.prepare_coefficients();

    for (int dom = 0; dom < source_space.get_nbr_domains(); ++dom) {
        Index pos(source_space.get_domain(dom)->get_nbr_points());
        Point physical(3);
        for (int component = 1; component <= 3; ++component)
            physical.set(component) = source_space.get_domain(dom)->get_cart(component)(pos);

        const auto located = batch.locate(physical);
        std::array<double, 3> actual{};
        batch.values(located, actual);
        REQUIRE(std::bit_cast<std::uint64_t>(actual[0]) ==
                std::bit_cast<std::uint64_t>(source_first.val_point(physical)));
        REQUIRE(std::bit_cast<std::uint64_t>(actual[1]) ==
                std::bit_cast<std::uint64_t>(source_second.val_point(physical)));
        REQUIRE(actual[2] == 0.);
    }
}

TEST_CASE("Prepared BNS point lanes match scalar values within and across domains",
          "[bns][field-transfer][batch]")
{
    Space_bin_ns_nosym source_space = make_bns_space(8.0);
    Scalar source(source_space);
    Scalar source_zero(source_space);
    fill_domain_polynomial(source, 0.625);
    source_zero = 0.;

    const std::array<const Scalar*, 2> fields{&source, &source_zero};
    bns_field_transfer::scalar_source_batch batch(fields);
    batch.prepare_coefficients();

    const double center1 = source_space.get_domain(source_space.NS1)->get_center()(1);
    const double center2 = source_space.get_domain(source_space.NS2)->get_center()(1);
    std::array<Point, 4> physical{
        Point(3), Point(3), Point(3), Point(3)};
    for (std::size_t point = 0; point < physical.size(); ++point) {
        physical[point].set(1) = center1 + 0.1 + 0.025 * static_cast<double>(point);
        physical[point].set(2) = 0.02 * static_cast<double>(point);
        physical[point].set(3) = 0.01;
    }
    std::array<bns_field_transfer::located_source_point, 4> located{
        batch.locate(physical[0]), batch.locate(physical[1]),
        batch.locate(physical[2]), batch.locate(physical[3])};
    std::array<const bns_field_transfer::located_source_point*, 4> located_ptrs{
        &located[0], &located[1], &located[2], &located[3]};
    REQUIRE(located[0].domain == located[1].domain);
    REQUIRE(located[0].domain == located[2].domain);
    REQUIRE(located[0].domain == located[3].domain);

    std::array<double, 4> observed{};
    batch.value_points4(located_ptrs, 0, observed);
    for (std::size_t point = 0; point < observed.size(); ++point)
        REQUIRE(std::bit_cast<std::uint64_t>(observed[point]) ==
                std::bit_cast<std::uint64_t>(source.val_point(physical[point])));
    batch.value_points4(located_ptrs, 1, observed);
    REQUIRE((observed == std::array<double, 4>{0., 0., 0., 0.}));

    physical[1].set(1) = center2 + 0.1;
    physical[1].set(2) = 0.;
    physical[2].set(1) = 6.;
    physical[2].set(2) = 0.25;
    located = {
        batch.locate(physical[0]), batch.locate(physical[1]),
        batch.locate(physical[2]), batch.locate(physical[3])};
    REQUIRE_FALSE((located[0].domain == located[1].domain &&
                   located[0].domain == located[2].domain &&
                   located[0].domain == located[3].domain));
    batch.value_points4(located_ptrs, 0, observed);
    for (std::size_t point = 0; point < observed.size(); ++point)
        REQUIRE(std::bit_cast<std::uint64_t>(observed[point]) ==
                std::bit_cast<std::uint64_t>(source.val_point(physical[point])));
}

TEST_CASE("BNS scalar import batch is bitwise Scalar::import identical",
          "[bns][field-transfer][batch]")
{
    Space_bin_ns_nosym source_space = make_bns_space(8.0);
    Space_bin_ns_nosym target_space = make_bns_space(8.0);

    Scalar source_first(source_space);
    Scalar source_second(source_space);
    Scalar source_domains(source_space);
    Scalar source_zero(source_space);
    fill_domain_polynomial(source_first, 1.25);
    fill_domain_polynomial(source_second, -0.75);
    fill_domain_constants(source_domains);
    source_zero = 0.;

    Scalar oracle_first(target_space); oracle_first = 3.;
    Scalar oracle_second(target_space); oracle_second = -4.;
    Scalar oracle_domains(target_space); oracle_domains = 9.;
    oracle_first.import(source_first);
    oracle_second.import(source_second);
    oracle_domains.import(source_domains);

    Scalar batch_first(target_space); batch_first = 3.;
    Scalar batch_second(target_space); batch_second = -4.;
    Scalar batch_domains(target_space); batch_domains = 9.;
    Scalar batch_zero(target_space); batch_zero = 7.;
    const std::array fields{
        bns_field_transfer::import_field(batch_first, source_first),
        bns_field_transfer::import_field(batch_second, source_second),
        bns_field_transfer::import_field(batch_domains, source_domains),
        bns_field_transfer::import_field(batch_zero, source_zero),
    };
    bns_field_transfer::import_scalar_batch(fields);

    require_bitwise_equal(batch_first, oracle_first);
    require_bitwise_equal(batch_second, oracle_second);
    require_bitwise_equal(batch_domains, oracle_domains);

    const int last = target_space.get_nbr_domains() - 1;
    const Dim_array last_points = target_space.get_domain(last)->get_nbr_points();
    Index infinity(last_points);
    infinity.set(0) = last_points(0) - 1;
    REQUIRE(batch_zero(last)(infinity) == 7.);

    Index finite(target_space.get_domain(0)->get_nbr_points());
    REQUIRE(batch_zero(0)(finite) == 0.);

    const Dim_array seam_points = target_space.get_domain(0)->get_nbr_points();
    Index seam(seam_points);
    seam.set(0) = seam_points(0) - 1;
    Point seam_point(3);
    for (int component = 1; component <= 3; ++component)
        seam_point.set(component) = target_space.get_domain(0)->get_cart(component)(seam);
    int selected = -1;
    for (int dom = source_space.get_nbr_domains() - 1; dom >= 0; --dom) {
        if (source_space.get_domain(dom)->is_in(seam_point)) {
            selected = dom;
            break;
        }
    }
    REQUIRE(selected >= 0);
    REQUIRE(batch_domains(0)(seam) == static_cast<double>(selected + 1));
}

TEST_CASE("BNS cold two-source batch is bitwise scalar-oracle identical",
          "[bns][field-transfer][batch]")
{
    Space_bin_ns_nosym source_space1 = make_bns_space(8.0);
    Space_bin_ns_nosym source_space2 = make_bns_space(8.0);
    Space_bin_ns_nosym target_space = make_bns_space(8.0);

    Scalar background1(source_space1), weighted1(source_space1), accumulated1(source_space1);
    Scalar background2(source_space2), weighted2(source_space2);
    fill_domain_polynomial(background1, 1.2);
    fill_domain_polynomial(weighted1, -0.2);
    fill_domain_polynomial(accumulated1, 0.4);
    fill_domain_polynomial(background2, 0.8);
    fill_domain_polynomial(weighted2, 0.3);

    Scalar oracle_background(target_space); oracle_background.annule_hard();
    Scalar oracle_weighted(target_space); oracle_weighted.annule_hard();
    Scalar oracle_accumulated(target_space); oracle_accumulated.annule_hard();
    Scalar batch_background(target_space); batch_background.annule_hard();
    Scalar batch_weighted(target_space); batch_weighted.annule_hard();
    Scalar batch_accumulated(target_space); batch_accumulated.annule_hard();

    const std::array<double, 2> centers{
        target_space.get_domain(target_space.NS1)->get_center()(1),
        target_space.get_domain(target_space.NS2)->get_center()(1)};
    const std::array<double, 2> invw4{0.002, 0.003};
    cold_scalar_oracle(
        oracle_background, oracle_weighted, oracle_accumulated,
        background1, background2, weighted1, weighted2, accumulated1,
        centers, invw4);

    const std::array fields{
        bns_field_transfer::two_source_field{
            &batch_background, &background1, &background2, 1.,
            bns_field_transfer::two_source_combination::background, true},
        bns_field_transfer::two_source_field{
            &batch_weighted, &weighted1, &weighted2, 0.,
            bns_field_transfer::two_source_combination::weighted_sum, false},
        bns_field_transfer::two_source_field{
            &batch_accumulated, &accumulated1, nullptr, 0.,
            bns_field_transfer::two_source_combination::accumulate_from_zero, false},
    };
    bns_field_transfer::superpose_two_source_batch(fields, centers, invw4);

    require_bitwise_equal(batch_background, oracle_background);
    require_bitwise_equal(batch_weighted, oracle_weighted);
    require_bitwise_equal(batch_accumulated, oracle_accumulated);
}

TEST_CASE("BNS separation blend batch is bitwise blend_scalar identical",
          "[bns][field-transfer][batch]")
{
    Space_bin_ns_nosym old_space = make_bns_space(8.0);
    Space_bin_ns_nosym new_space = make_bns_space(8.5);

    Scalar old_background(old_space), old_matter(old_space), old_shift(old_space);
    fill_domain_polynomial(old_background, 1.1);
    fill_domain_polynomial(old_matter, 0.2);
    fill_domain_polynomial(old_shift, -0.1);

    Scalar oracle_background(new_space); oracle_background.annule_hard();
    Scalar oracle_matter(new_space); oracle_matter.annule_hard();
    Scalar oracle_shift(new_space); oracle_shift.annule_hard();
    Scalar batch_background(new_space); batch_background.annule_hard();
    Scalar batch_matter(new_space); batch_matter.annule_hard();
    Scalar batch_shift(new_space); batch_shift.annule_hard();

    bns_separation_seed::two_center_drift drift{};
    drift.xc_old[0] = old_space.get_domain(old_space.NS1)->get_center()(1);
    drift.xc_old[1] = old_space.get_domain(old_space.NS2)->get_center()(1);
    drift.xc_new[0] = new_space.get_domain(new_space.NS1)->get_center()(1);
    drift.xc_new[1] = new_space.get_domain(new_space.NS2)->get_center()(1);
    drift.invw4[0] = 0.002;
    drift.invw4[1] = 0.003;

    bns_separation_seed::blend_scalar(
        oracle_background, old_background, drift, true, 1.0);
    bns_separation_seed::blend_scalar(
        oracle_matter, old_matter, drift, false, 0.0);
    bns_separation_seed::blend_scalar(
        oracle_shift, old_shift, drift, true, 0.0);

    const std::array fields{
        bns_separation_seed::scalar_blend_field{
            &batch_background, &old_background, true, 1.0},
        bns_separation_seed::scalar_blend_field{
            &batch_matter, &old_matter, false, 0.0},
        bns_separation_seed::scalar_blend_field{
            &batch_shift, &old_shift, true, 0.0},
    };
    bns_separation_seed::blend_scalar_batch(fields, drift);

    require_bitwise_equal(batch_background, oracle_background);
    require_bitwise_equal(batch_matter, oracle_matter);
    require_bitwise_equal(batch_shift, oracle_shift);
}
