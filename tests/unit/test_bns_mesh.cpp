#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Array/index.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Domain/adapted.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Domain/spheric_nosym_regularization.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/bin_ns.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <tuple>
#include <utility>
#include <vector>

using namespace Kadath;
using Catch::Matchers::WithinAbs;

namespace {

constexpr double seam_tolerance = 1.e-10;
constexpr int direction_samples = 128;
constexpr int volume_samples = 4096;
constexpr int deformed_direction_samples = 96;

struct SurfaceShape {
    double mean_radius;
    double z_amplitude;
    double x_amplitude;
    double quadrupole_amplitude;

    double operator()(double theta, double phi) const
    {
        const double sin_theta = std::sin(theta);
        return mean_radius + z_amplitude * std::cos(theta) + x_amplitude * sin_theta * std::cos(phi)
               + quadrupole_amplitude * sin_theta * sin_theta * std::sin(2.0 * phi);
    }
};

struct DeformedStar {
    int adapted_domain;
    double fixed_inner_radius;
    double fixed_outer_radius;
    SurfaceShape shape;
    const char* label;
};

struct FixedCoordinate {
    int axis;
    bool upper;
};

double radical_inverse(std::size_t index, unsigned int base)
{
    double result = 0.0;
    double weight = 1.0 / static_cast<double>(base);
    while (index != 0) {
        result += static_cast<double>(index % base) * weight;
        index /= base;
        weight /= static_cast<double>(base);
    }
    return result;
}

int containment_count(const Space& space, const Point& point, double tolerance, int excluded_domain = -1)
{
    int count = 0;
    for (int domain = 0; domain < space.get_nbr_domains(); ++domain) {
        if (domain != excluded_domain && space.get_domain(domain)->is_in(point, tolerance))
            ++count;
    }
    return count;
}

bool contained_in_any(const Space& space, const Point& point, std::initializer_list<int> domains)
{
    return std::any_of(domains.begin(), domains.end(), [&](int domain) {
        return space.get_domain(domain)->is_in(point, seam_tolerance);
    });
}

Point point_on_ray(const Point& center, double radius, double theta, double phi)
{
    Point point(3);
    const double sin_theta = std::sin(theta);
    if (std::abs(sin_theta) < 1.e-15) {
        point.set(1) = center(1);
        point.set(2) = center(2);
    } else {
        point.set(1) = center(1) + radius * sin_theta * std::cos(phi);
        point.set(2) = center(2) + radius * sin_theta * std::sin(phi);
    }
    point.set(3) = center(3) + radius * std::cos(theta);
    return point;
}

Val_domain make_surface(const Domain* domain, const SurfaceShape& shape, double offset = 0.0)
{
    Val_domain surface(domain);
    surface.allocate_conf();
    const Array<double> theta = domain->get_coloc(2);
    const Array<double> phi = domain->get_coloc(3);
    Index index(domain->get_nbr_points());
    do {
        surface.set(index) = shape(theta(index(1)), phi(index(2))) + offset;
    } while (index.inc());
    surface.std_base();
    return surface;
}

double evaluate_surface(const Val_domain& surface, const Array<double>& coefficients, double theta, double phi)
{
    Point numerical(3);
    numerical.set(1) = 1.0;
    numerical.set(2) = theta;
    numerical.set(3) = phi;
    return surface.get_base().summation(numerical, coefficients);
}

void set_shared_surface(const Space_bin_ns_nosym& space, const DeformedStar& star, double outer_offset = 0.0,
                        double inner_offset = 0.0)
{
    const auto* outer =
        dynamic_cast<const Domain_shell_outer_adapted_nosym*>(space.get_domain(star.adapted_domain));
    const auto* inner =
        dynamic_cast<const Domain_shell_inner_adapted_nosym*>(space.get_domain(star.adapted_domain + 1));
    REQUIRE(outer != nullptr);
    REQUIRE(inner != nullptr);

    // set_mapping requires a Val_domain attached to the receiving domain, so
    // construct the same analytic surface independently on the two conforming
    // angular grids instead of assigning one domain's Val_domain to the other.
    Val_domain outer_surface = make_surface(outer, star.shape, outer_offset);
    Val_domain inner_surface = make_surface(inner, star.shape, inner_offset);
    outer->set_mapping(outer_surface);
    inner->set_mapping(inner_surface);
}

void require_positive_radial_order(const Domain_shell_outer_adapted_nosym& outer,
                                   const Domain_shell_inner_adapted_nosym& inner, const DeformedStar& star)
{
    const Val_domain outer_surface = outer.get_outer_radius();
    const Val_domain inner_surface = inner.get_inner_radius();
    const Val_domain outer_radius = outer.get_radius();
    const Val_domain inner_radius = inner.get_radius();
    const Dim_array points = outer.get_nbr_points();

    for (int j = 0; j < points(1); ++j) {
        for (int k = 0; k < points(2); ++k) {
            Index outer_inner_face(points);
            Index outer_surface_face(points);
            Index inner_surface_face(points);
            Index inner_outer_face(points);
            outer_inner_face.set(1) = j;
            outer_inner_face.set(2) = k;
            outer_surface_face.set(0) = points(0) - 1;
            outer_surface_face.set(1) = j;
            outer_surface_face.set(2) = k;
            inner_surface_face.set(1) = j;
            inner_surface_face.set(2) = k;
            inner_outer_face.set(0) = points(0) - 1;
            inner_outer_face.set(1) = j;
            inner_outer_face.set(2) = k;

            INFO(star.label << ", angular collocation (" << j << ", " << k << ")");
            REQUIRE_THAT(outer_radius(outer_inner_face), WithinAbs(star.fixed_inner_radius, 1.e-12));
            REQUIRE_THAT(outer_radius(outer_surface_face), WithinAbs(outer_surface(outer_surface_face), 1.e-12));
            REQUIRE(outer_radius(outer_surface_face) > outer_radius(outer_inner_face));
            REQUIRE_THAT(inner_radius(inner_surface_face), WithinAbs(inner_surface(inner_surface_face), 1.e-12));
            REQUIRE_THAT(inner_radius(inner_outer_face), WithinAbs(star.fixed_outer_radius, 1.e-12));
            REQUIRE(inner_radius(inner_outer_face) > inner_radius(inner_surface_face));
        }
    }
}

void require_deformed_star_covered(const Space_bin_ns_nosym& space, const DeformedStar& star)
{
    const auto* outer =
        dynamic_cast<const Domain_shell_outer_adapted_nosym*>(space.get_domain(star.adapted_domain));
    const auto* inner =
        dynamic_cast<const Domain_shell_inner_adapted_nosym*>(space.get_domain(star.adapted_domain + 1));
    REQUIRE(outer != nullptr);
    REQUIRE(inner != nullptr);

    Val_domain outer_surface = outer->get_outer_radius();
    Val_domain inner_surface = inner->get_inner_radius();
    REQUIRE(diffmax(outer_surface, inner_surface) < 1.e-12);
    outer_surface.coef();
    inner_surface.coef();
    const Array<double> outer_coefficients = outer_surface.get_coef();
    const Array<double> inner_coefficients = inner_surface.get_coef();
    const Point center = outer->get_center();

    // A nonzero north/south difference makes this explicitly stronger than a
    // plane-symmetric adapted-surface smoke test.
    REQUIRE(std::abs(star.shape(0.0, 0.0) - star.shape(M_PI, 0.0)) > 0.05);

    Scalar one(space);
    one = 1.0;
    one.std_base();

    auto require_direction = [&](double theta, double phi, int sample) {
        const double outer_value = evaluate_surface(outer_surface, outer_coefficients, theta, phi);
        const double inner_value = evaluate_surface(inner_surface, inner_coefficients, theta, phi);
        INFO(star.label << ", direction sample " << sample << ", theta = " << theta << ", phi = " << phi
                        << ", outer surface = " << outer_value << ", inner surface = " << inner_value);
        REQUIRE_THAT(outer_value, WithinAbs(star.shape(theta, phi), 2.e-11));
        REQUIRE_THAT(outer_value, WithinAbs(inner_value, 2.e-11));
        REQUIRE(outer_value > star.fixed_inner_radius + 0.1);
        REQUIRE(outer_value < star.fixed_outer_radius - 0.1);

        const Point seam = point_on_ray(center, outer_value, theta, phi);
        const Point outer_numerical = outer->absol_to_num(seam);
        const Point inner_numerical = inner->absol_to_num(seam);
        REQUIRE_THAT(outer_numerical(1), WithinAbs(1.0, 2.e-10));
        REQUIRE_THAT(inner_numerical(1), WithinAbs(-1.0, 2.e-10));
        REQUIRE(outer->is_in(seam, seam_tolerance));
        REQUIRE(inner->is_in(seam, seam_tolerance));
        REQUIRE(containment_count(space, seam, seam_tolerance) == 2);

        constexpr double radial_offset = 1.e-5;
        const Point just_inside = point_on_ray(center, outer_value - radial_offset, theta, phi);
        const Point just_outside = point_on_ray(center, outer_value + radial_offset, theta, phi);
        REQUIRE(outer->is_in(just_inside));
        REQUIRE_FALSE(inner->is_in(just_inside));
        REQUIRE_FALSE(outer->is_in(just_outside));
        REQUIRE(inner->is_in(just_outside));
        REQUIRE(containment_count(space, just_inside, 0.0) == 1);
        REQUIRE(containment_count(space, just_outside, 0.0) == 1);

        const std::array<double, 4> radial_samples{
            0.5 * (star.fixed_inner_radius + outer_value), outer_value - radial_offset,
            outer_value + radial_offset, 0.5 * (outer_value + star.fixed_outer_radius)};
        for (double radius : radial_samples) {
            const Point point = point_on_ray(center, radius, theta, phi);
            REQUIRE(containment_count(space, point, 0.0) == 1);
            double value = 0.0;
            REQUIRE_NOTHROW(value = one.val_point(point));
            REQUIRE_THAT(value, WithinAbs(1.0, 1.e-10));
        }

        double seam_value = 0.0;
        REQUIRE_NOTHROW(seam_value = one.val_point(seam));
        REQUIRE_THAT(seam_value, WithinAbs(1.0, 1.e-10));
    };

    require_direction(0.0, 0.0, -2);
    require_direction(M_PI, 0.0, -1);
    const double golden_angle = M_PI * (3.0 - std::sqrt(5.0));
    for (int sample = 0; sample < deformed_direction_samples; ++sample) {
        const double z = 1.0 - 2.0 * (static_cast<double>(sample) + 0.5) / deformed_direction_samples;
        require_direction(std::acos(z), golden_angle * sample, sample);
    }

    require_positive_radial_order(*outer, *inner, star);
}

template <typename DomainT>
void require_domain_type(const Space& space, int domain)
{
    INFO("domain " << domain);
    REQUIRE(dynamic_cast<const DomainT*>(space.get_domain(domain)) != nullptr);
}

using MatchingTarget = std::pair<int, int>;

void require_matching_targets(const Space_bin_ns_nosym& space, int domain, int bound,
                              std::initializer_list<MatchingTarget> expected)
{
    const Array<int> actual = space.get_indices_matching_non_std(domain, bound);
    REQUIRE(actual.get_ndim() == 2);
    REQUIRE(actual.get_size(0) == 2);
    REQUIRE(actual.get_size(1) == static_cast<int>(expected.size()));
    std::size_t target = 0;
    for (const auto [expected_domain, expected_bound] : expected) {
        INFO("source domain " << domain << ", source bound " << bound << ", target " << target);
        CHECK(actual(0, static_cast<int>(target)) == expected_domain);
        CHECK(actual(1, static_cast<int>(target)) == expected_bound);
        ++target;
    }
}

void require_complete_nonstandard_matching_topology(
    const Space_bin_ns_nosym& space, int star1_shells, int star2_shells)
{
    std::vector<std::pair<int, int>> accepted;
    const auto require_edge = [&](int domain, int bound,
                                  std::initializer_list<MatchingTarget> targets) {
        require_matching_targets(space, domain, bound, targets);
        accepted.emplace_back(domain, bound);
    };

    require_edge(space.NS1, OUTER_BC, {{space.ADAPTED1, INNER_BC}});
    require_edge(space.ADAPTED1, INNER_BC, {{space.NS1, OUTER_BC}});
    for (int shell = 0; shell < star1_shells; ++shell) {
        const int inner = space.ADAPTED1 + 1 + shell;
        require_edge(inner, OUTER_BC, {{inner + 1, INNER_BC}});
        require_edge(inner + 1, INNER_BC, {{inner, OUTER_BC}});
    }
    const int star1_outer = space.ADAPTED1 + 1 + star1_shells;
    require_edge(star1_outer, OUTER_BC,
                 {{space.OUTER, INNER_BC}, {space.OUTER + 1, INNER_BC}});

    require_edge(space.NS2, OUTER_BC, {{space.ADAPTED2, INNER_BC}});
    require_edge(space.ADAPTED2, INNER_BC, {{space.NS2, OUTER_BC}});
    for (int shell = 0; shell < star2_shells; ++shell) {
        const int inner = space.ADAPTED2 + 1 + shell;
        require_edge(inner, OUTER_BC, {{inner + 1, INNER_BC}});
        require_edge(inner + 1, INNER_BC, {{inner, OUTER_BC}});
    }
    const int star2_outer = space.ADAPTED2 + 1 + star2_shells;
    require_edge(star2_outer, OUTER_BC,
                 {{space.OUTER + 3, INNER_BC}, {space.OUTER + 4, INNER_BC}});

    require_edge(space.OUTER, INNER_BC, {{star1_outer, OUTER_BC}});
    require_edge(space.OUTER + 1, INNER_BC, {{star1_outer, OUTER_BC}});
    require_edge(space.OUTER + 3, INNER_BC, {{star2_outer, OUTER_BC}});
    require_edge(space.OUTER + 4, INNER_BC, {{star2_outer, OUTER_BC}});
    for (int bispheric = 0; bispheric < 5; ++bispheric)
        require_edge(space.OUTER + bispheric, OUTER_BC,
                     {{space.OUTER + 5, INNER_BC}});
    require_edge(space.OUTER + 5, INNER_BC,
                 {{space.OUTER, OUTER_BC}, {space.OUTER + 1, OUTER_BC},
                  {space.OUTER + 2, OUTER_BC}, {space.OUTER + 3, OUTER_BC},
                  {space.OUTER + 4, OUTER_BC}});

    // Every other radial domain/boundary pair is outside the non-standard
    // matching graph. This fail-closed sweep catches accidental topology
    // aliases as shell counts move the absolute indices.
    for (int domain = 0; domain < space.get_nbr_domains(); ++domain) {
        for (int bound : {INNER_BC, OUTER_BC}) {
            const bool is_accepted =
                std::find(accepted.begin(), accepted.end(),
                          std::pair<int, int>{domain, bound}) != accepted.end();
            if (!is_accepted) {
                INFO("unexpected non-standard topology entry at domain " << domain
                     << ", bound " << bound);
                REQUIRE_THROWS_AS(space.get_indices_matching_non_std(domain, bound),
                                  KadathError);
            }
        }
    }
}

void require_domain_types(const Space_bin_ns& space)
{
    require_domain_type<Domain_nucleus>(space, 0);
    require_domain_type<Domain_shell_outer_adapted>(space, 1);
    require_domain_type<Domain_shell_inner_adapted>(space, 2);
    require_domain_type<Domain_shell>(space, 3);
    require_domain_type<Domain_nucleus>(space, 4);
    require_domain_type<Domain_shell_outer_adapted>(space, 5);
    require_domain_type<Domain_shell_inner_adapted>(space, 6);
    require_domain_type<Domain_shell>(space, 7);
    require_domain_type<Domain_bispheric_chi_first>(space, 8);
    require_domain_type<Domain_bispheric_rect>(space, 9);
    require_domain_type<Domain_bispheric_eta_first>(space, 10);
    require_domain_type<Domain_bispheric_rect>(space, 11);
    require_domain_type<Domain_bispheric_chi_first>(space, 12);
    require_domain_type<Domain_shell>(space, 13);
    require_domain_type<Domain_compact>(space, 14);
}

void require_domain_types(const Space_bin_ns_nosym& space)
{
    require_domain_type<Domain_nucleus_nosym>(space, 0);
    require_domain_type<Domain_shell_outer_adapted_nosym>(space, 1);
    require_domain_type<Domain_shell_inner_adapted_nosym>(space, 2);
    require_domain_type<Domain_shell_nosym>(space, 3);
    require_domain_type<Domain_nucleus_nosym>(space, 4);
    require_domain_type<Domain_shell_outer_adapted_nosym>(space, 5);
    require_domain_type<Domain_shell_inner_adapted_nosym>(space, 6);
    require_domain_type<Domain_shell_nosym>(space, 7);
    require_domain_type<Domain_bispheric_chi_first_nosym>(space, 8);
    require_domain_type<Domain_bispheric_rect_nosym>(space, 9);
    require_domain_type<Domain_bispheric_eta_first_nosym>(space, 10);
    require_domain_type<Domain_bispheric_rect_nosym>(space, 11);
    require_domain_type<Domain_bispheric_chi_first_nosym>(space, 12);
    require_domain_type<Domain_shell_nosym>(space, 13);
    require_domain_type<Domain_compact_nosym>(space, 14);
}

bool lies_on_face(const Index& index, const Dim_array& points, std::initializer_list<FixedCoordinate> fixed)
{
    return std::all_of(fixed.begin(), fixed.end(), [&](const FixedCoordinate& coordinate) {
        const int expected = coordinate.upper ? points(coordinate.axis) - 1 : 0;
        return index(coordinate.axis) == expected;
    });
}

void require_chart_face_covered(const Space& space, int source_domain, int target_domain,
                                std::initializer_list<FixedCoordinate> fixed, const char* label)
{
    const Domain* source = space.get_domain(source_domain);
    const Domain* target = space.get_domain(target_domain);
    const Dim_array points = source->get_nbr_points();
    Index index(points);
    do {
        if (!lies_on_face(index, points, fixed))
            continue;

        Point point(3);
        point.set(1) = source->get_cart(1)(index);
        point.set(2) = source->get_cart(2)(index);
        point.set(3) = source->get_cart(3)(index);

        INFO(label << ", source domain " << source_domain << ", target domain " << target_domain
                   << ", point = (" << point(1) << ", " << point(2) << ", " << point(3) << ")");
        REQUIRE(source->is_in(point, seam_tolerance));
        REQUIRE(target->is_in(point, seam_tolerance));
    } while (index.inc());
}

void require_spherical_handoff(const Space& space, const Point& center, double radius,
                               std::initializer_list<int> inner_domains,
                               std::initializer_list<int> outer_domains, const char* label)
{
    const double golden_angle = std::acos(-1.0) * (3.0 - std::sqrt(5.0));
    for (int sample = 0; sample < direction_samples; ++sample) {
        const double z = 1.0 - 2.0 * (static_cast<double>(sample) + 0.5) / direction_samples;
        const double cylindrical_radius = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double phi = golden_angle * sample;

        Point point(3);
        point.set(1) = center(1) + radius * cylindrical_radius * std::cos(phi);
        point.set(2) = center(2) + radius * cylindrical_radius * std::sin(phi);
        point.set(3) = center(3) + radius * z;

        INFO(label << ", direction sample " << sample << ", point = (" << point(1) << ", " << point(2) << ", "
                   << point(3) << ")");
        REQUIRE(contained_in_any(space, point, inner_domains));
        REQUIRE(contained_in_any(space, point, outer_domains));
    }
}

template <typename SpaceT>
void require_layout_smoke()
{
    // Four entries give one ordinary post-adapted shell: the last entry is both
    // that shell's outer radius and the star-to-bispheric handoff radius.
    const std::vector<double> ns1_bounds{0.50, 1.00, 1.25, 1.50};
    const std::vector<double> ns2_bounds{0.55, 1.05, 1.35, 1.65};
    // The first entry is the bispheric outer radius; the second is the outer
    // shell/compactified handoff, hence exactly one exterior shell.
    const std::vector<double> outer_bounds{8.0, 12.0};
    SpaceT space(CHEB_TYPE, 8.0, ns1_bounds, ns2_bounds, outer_bounds, 7);

    REQUIRE(space.get_nbr_domains() == 15);
    REQUIRE(space.NS1 == 0);
    REQUIRE(space.ADAPTED1 == 1);
    REQUIRE(space.NS2 == 4);
    REQUIRE(space.ADAPTED2 == 5);
    REQUIRE(space.OUTER == 8);
    REQUIRE(space.get_n_shells_outer() == 1);
    for (int domain = 0; domain < space.get_nbr_domains(); ++domain)
        REQUIRE(space.get_domain(domain) != nullptr);
    require_domain_types(space);

    const Point ns1_center = space.get_domain(space.NS1)->get_center();
    const Point ns2_center = space.get_domain(space.NS2)->get_center();
    Point origin(3);

    // Fixed-radius and initially spherical adapted handoffs within NS1.
    require_spherical_handoff(space, ns1_center, ns1_bounds[0], {0}, {1}, "NS1 nucleus/adapted-outer");
    require_spherical_handoff(space, ns1_center, ns1_bounds[1], {1}, {2}, "NS1 adapted surface");
    require_spherical_handoff(space, ns1_center, ns1_bounds[2], {2}, {3}, "NS1 adapted-inner/ordinary-shell");
    require_spherical_handoff(space, ns1_center, ns1_bounds[3], {3}, {8, 9},
                              "NS1 ordinary-shell/bispheric");

    // Corresponding handoffs within NS2.
    require_spherical_handoff(space, ns2_center, ns2_bounds[0], {4}, {5}, "NS2 nucleus/adapted-outer");
    require_spherical_handoff(space, ns2_center, ns2_bounds[1], {5}, {6}, "NS2 adapted surface");
    require_spherical_handoff(space, ns2_center, ns2_bounds[2], {6}, {7}, "NS2 adapted-inner/ordinary-shell");
    require_spherical_handoff(space, ns2_center, ns2_bounds[3], {7}, {11, 12},
                              "NS2 ordinary-shell/bispheric");

    // The five bispheric charts collectively meet the exterior shell, which in
    // turn meets the compactified domain.
    require_spherical_handoff(space, origin, outer_bounds[0], {8, 9, 10, 11, 12}, {13},
                              "bispheric/exterior-shell");
    require_spherical_handoff(space, origin, outer_bounds[1], {13}, {14}, "exterior-shell/compactified");

    // Internal bispheric handoffs use different numerical axes on the two
    // charts. Sample each face in both directions so a narrow chart-seam gap
    // cannot hide between the volume probes below.
    require_chart_face_covered(space, 8, 9, {{1, true}}, "left chi-first/rect");
    require_chart_face_covered(space, 9, 8, {{1, true}}, "left rect/chi-first");
    require_chart_face_covered(space, 9, 10, {{0, true}}, "left rect/eta-first");
    require_chart_face_covered(space, 10, 9, {{1, false}}, "eta-first/left rect");
    require_chart_face_covered(space, 10, 11, {{1, true}}, "eta-first/right rect");
    require_chart_face_covered(space, 11, 10, {{0, true}}, "right rect/eta-first");
    require_chart_face_covered(space, 11, 12, {{1, true}}, "right rect/chi-first");
    require_chart_face_covered(space, 12, 11, {{1, true}}, "right chi-first/rect");

    // Off-interface quasi-random probes must have one and only one owner. This
    // catches holes and finite-volume overlaps, including internal bispheric
    // chart seams which are not spherical coordinate surfaces.
    constexpr double cube_half_width = 16.0;
    for (int sample = 1; sample <= volume_samples; ++sample) {
        Point point(3);
        point.set(1) = cube_half_width * (2.0 * radical_inverse(sample, 2) - 1.0);
        point.set(2) = cube_half_width * (2.0 * radical_inverse(sample, 3) - 1.0);
        point.set(3) = cube_half_width * (2.0 * radical_inverse(sample, 5) - 1.0);

        const int owners = containment_count(space, point, 0.0);
        INFO("volume sample " << sample << ", point = (" << point(1) << ", " << point(2) << ", " << point(3)
                              << "), owners = " << owners);
        REQUIRE(owners == 1);
    }

    // Falsifying control: at radius 10 only domain 13 can own the point. Omitting
    // that shell must expose a gap between the radius-8 bispheric boundary and
    // the radius-12 compactified boundary.
    Point exterior_shell_point(3);
    exterior_shell_point.set(1) = 3.6;
    exterior_shell_point.set(2) = -4.8;
    exterior_shell_point.set(3) = 8.0;
    REQUIRE(containment_count(space, exterior_shell_point, 0.0) == 1);
    REQUIRE(space.get_domain(13)->is_in(exterior_shell_point, 0.0));
    REQUIRE(containment_count(space, exterior_shell_point, 0.0, 13) == 0);
}

double legacy_galerkin_factor(int type_base, int coefficient)
{
    if (type_base == CHEB_TYPE)
        return coefficient % 2 == 1 ? -2. : 2.;

    REQUIRE(type_base == LEG_TYPE);
    double factor = -double(4 * coefficient + 1);
    for (int index = 1; index <= coefficient; ++index) {
        factor *= -double(2 * index - 1) / double(2 * index);
    }
    return factor;
}

template <typename DomainT>
void legacy_spheric_interior_export(const DomainT& domain, const Val_domain& field, int mlim, int order,
                                    Array<double>& result, int& position)
{
    field.coef();
    const Dim_array coefficients = domain.get_nbr_coefs();
    const int kmin = 2 * mlim + 2;
    Index coefficient(coefficients);
    Index anchor(coefficients);
    for (int k = 0; k < coefficients(2) - 1; ++k) {
        if (k == 1)
            continue;
        coefficient.set(2) = k;
        const int theta_basis = (*field.get_base().get_base_1d(1))(k);
        for (int j = 0; j < coefficients(1); ++j) {
            coefficient.set(1) = j;
            for (int i = 0; i < coefficients(0) - order; ++i) {
                coefficient.set(0) = i;
                if (!detail::spheric_nosym_true_theta_coef(theta_basis, j, k, kmin, coefficients(1)))
                    continue;
                result.set(position) = field.get_coef(coefficient);
                if (detail::spheric_nosym_uses_theta_galerkin(theta_basis, k, kmin)) {
                    anchor = coefficient;
                    anchor.set(1) = detail::spheric_nosym_theta_anchor(theta_basis, j);
                    result.set(position) -=
                        detail::spheric_nosym_export_anchor_weight(theta_basis, j) * field.get_coef(anchor);
                }
                ++position;
            }
        }
    }
}

template <typename DomainT>
void legacy_bispheric_interior_export(const DomainT& domain, const Val_domain& field, int order,
                                      bool eta_first, Array<double>& result, int& position)
{
    field.coef();
    const Dim_array coefficients = domain.get_nbr_coefs();
    const int forgot_chi = order == 0 ? 0 : 1;
    const int forgot_eta = eta_first && order != 0 ? 2 : order;
    Index coefficient(coefficients);
    Index anchor(coefficients);
    for (int k = 0; k < coefficients(2) - 1; ++k) {
        if (k == 1)
            continue;
        coefficient.set(2) = k;
        const int fixed_chi_basis = eta_first ? 0 : (*field.get_base().get_base_1d(1))(k);
        int outer_limit = eta_first ? coefficients(1) - forgot_eta : coefficients(1) - forgot_chi;
        if (!eta_first && (fixed_chi_basis == CHEB_ODD || fixed_chi_basis == LEG_ODD))
            --outer_limit;
        for (int outer = 0; outer < outer_limit; ++outer) {
            coefficient.set(1) = outer;
            const int chi_basis = eta_first ? (*field.get_base().get_base_1d(0))(outer, k)
                                            : fixed_chi_basis;
            int inner_limit = eta_first ? coefficients(0) - forgot_chi : coefficients(0) - forgot_eta;
            if (eta_first && (chi_basis == CHEB_ODD || chi_basis == LEG_ODD))
                --inner_limit;
            for (int inner = 0; inner < inner_limit; ++inner) {
                coefficient.set(0) = inner;
                const int chi_coefficient = eta_first ? inner : outer;
                if (chi_basis == CHEB_EVEN || chi_basis == LEG_EVEN) {
                    if (k == 0) {
                        result.set(position++) = field.get_coef(coefficient);
                    } else if (chi_coefficient != 0) {
                        anchor = coefficient;
                        anchor.set(eta_first ? 0 : 1) = 0;
                        result.set(position++) = field.get_coef(coefficient) +
                            legacy_galerkin_factor(domain.get_type_base(), chi_coefficient) * field.get_coef(anchor);
                    }
                }
                if (chi_basis == CHEB_ODD || chi_basis == LEG_ODD)
                    result.set(position++) = field.get_coef(coefficient);
            }
        }
    }
}

template <typename DomainT>
void legacy_standard_boundary_export(const DomainT& domain, const Val_domain& field, int bound,
                                     int chi_axis, int omitted, Array<double>& result, int& position)
{
    field.coef();
    const Dim_array coefficients = domain.get_nbr_coefs();
    Index coefficient(coefficients);
    Index anchor(coefficients);
    for (int k = 0; k < coefficients(2) - 1; ++k) {
        if (k == 1)
            continue;
        coefficient.set(2) = k;
        const int chi_basis = chi_axis == 0 ? (*field.get_base().get_base_1d(0))(0, k)
                                            : (*field.get_base().get_base_1d(1))(k);
        int limit = coefficients(chi_axis) - omitted;
        if (chi_basis == CHEB_ODD || chi_basis == LEG_ODD)
            --limit;
        for (int chi = 0; chi < limit; ++chi) {
            coefficient.set(chi_axis) = chi;
            if (chi_basis == CHEB_EVEN || chi_basis == LEG_EVEN) {
                if (k == 0) {
                    result.set(position++) = domain.val_boundary(bound, field, coefficient);
                } else if (chi != 0) {
                    anchor = coefficient;
                    anchor.set(chi_axis) = 0;
                    result.set(position++) = domain.val_boundary(bound, field, coefficient) +
                        legacy_galerkin_factor(domain.get_type_base(), chi) *
                            domain.val_boundary(bound, field, anchor);
                }
            }
            if (chi_basis == CHEB_ODD || chi_basis == LEG_ODD)
                result.set(position++) = domain.val_boundary(bound, field, coefficient);
        }
    }
}

template <typename DomainT>
void legacy_one_side_boundary_export(const DomainT& domain, const Val_domain& field, int bound,
                                     int chi_axis, int omitted, Array<double>& result, int& position)
{
    field.coef();
    const Dim_array coefficients = domain.get_nbr_coefs();
    const int phi_basis = (*field.get_base().get_base_1d(2))(0);
    const int limit = coefficients(chi_axis) - omitted;
    Index coefficient(coefficients);
    Index anchor(coefficients);
    for (int k = 0; k < coefficients(2); ++k) {
        coefficient.set(2) = k;
        for (int chi = 0; chi < limit; ++chi) {
            coefficient.set(chi_axis) = chi;
            const bool retained = k % 2 != 1 || chi != limit - 1;
            if (phi_basis == COS && retained) {
                if (k == 0 || k % 2 == 1) {
                    result.set(position++) = domain.val_boundary(bound, field, coefficient);
                } else if (chi != 0) {
                    anchor = coefficient;
                    anchor.set(chi_axis) = 0;
                    result.set(position++) = domain.val_boundary(bound, field, coefficient) +
                        legacy_galerkin_factor(domain.get_type_base(), chi) *
                            domain.val_boundary(bound, field, anchor);
                }
            } else if (phi_basis == SIN && k != 0 && k != coefficients(2) - 1 && retained) {
                if (k % 2 == 1) {
                    result.set(position++) = domain.val_boundary(bound, field, coefficient);
                } else if (chi != 0) {
                    anchor = coefficient;
                    anchor.set(chi_axis) = 0;
                    result.set(position++) = domain.val_boundary(bound, field, coefficient) +
                        legacy_galerkin_factor(domain.get_type_base(), chi) *
                            domain.val_boundary(bound, field, anchor);
                }
            }
        }
    }
}

Val_domain make_tau_characterization_field(const Domain* domain, bool antisymmetric)
{
    Val_domain field(domain);
    if (antisymmetric)
        field.std_anti_base();
    else
        field.std_base();
    field.allocate_coef();

    Index coefficient(domain->get_nbr_coefs());
    do {
        const double value = 0.125 + 100.0 * coefficient(0) + 10.0 * coefficient(1) + coefficient(2);
        field.set_coef(coefficient) = value;
    } while (coefficient.inc());
    return field;
}

Val_domain make_tau_one_side_characterization_field(const Domain* domain, int phi_basis, bool eta_first)
{
    Val_domain field(domain);
    const bool chebyshev = domain->get_type_base() == CHEB_TYPE;
    const int unrestricted_basis = chebyshev ? CHEB : LEG;
    const int even_basis = chebyshev ? CHEB_EVEN : LEG_EVEN;
    if (eta_first)
        field.set_base().set(domain->get_nbr_coefs(), phi_basis, unrestricted_basis, even_basis);
    else
        field.set_base().set(domain->get_nbr_coefs(), phi_basis, even_basis, unrestricted_basis);
    field.allocate_coef();

    Index coefficient(domain->get_nbr_coefs());
    do {
        const double value = 0.125 + 100.0 * coefficient(0) + 10.0 * coefficient(1) + coefficient(2);
        field.set_coef(coefficient) = value;
    } while (coefficient.inc());
    return field;
}

template <typename Exporter, typename LegacyExporter>
void require_tau_export_bytes(int ncond, Exporter&& exporter, LegacyExporter&& legacy_exporter)
{
    REQUIRE(ncond > 0);
    Array<double> actual(ncond);
    Array<double> legacy(ncond);
    int actual_position = 0;
    int legacy_position = 0;
    exporter(actual, actual_position, ncond);
    legacy_exporter(legacy, legacy_position);
    REQUIRE(actual_position == ncond);
    REQUIRE(legacy_position == ncond);
    REQUIRE(std::memcmp(actual.get_data(), legacy.get_data(), std::size_t(ncond) * sizeof(double)) == 0);
}

template <typename DomainT>
const DomainT& require_tau_domain(const Space_bin_ns_nosym& space, int domain_index)
{
    const auto* domain = dynamic_cast<const DomainT*>(space.get_domain(domain_index));
    REQUIRE(domain != nullptr);
    return *domain;
}

template <typename DomainT>
void require_spheric_interior_exports(const DomainT& domain, bool antisymmetric)
{
    Val_domain field = make_tau_characterization_field(&domain, antisymmetric);
    for (int mlim : {0, 1}) {
        for (int order : {0, 1, 2}) {
            const int ncond = domain.nbr_conditions_val_domain(field, mlim, order);
            require_tau_export_bytes(
                ncond,
                [&](Array<double>& residual, int& position, int count) {
                    domain.export_tau_val_domain(field, mlim, order, residual, position, count);
                },
                [&](Array<double>& residual, int& position) {
                    legacy_spheric_interior_export(domain, field, mlim, order, residual, position);
                });
        }
    }
}

template <typename DomainT>
void require_bispheric_interior_exports(const DomainT& domain, bool antisymmetric, bool eta_first)
{
    Val_domain field = make_tau_characterization_field(&domain, antisymmetric);
    for (int order : {0, 1, 2}) {
        const int ncond = domain.nbr_conditions_val_domain(field, order);
        require_tau_export_bytes(
            ncond,
            [&](Array<double>& residual, int& position, int count) {
                domain.export_tau_val_domain(field, order, residual, position, count);
            },
            [&](Array<double>& residual, int& position) {
                legacy_bispheric_interior_export(domain, field, order, eta_first, residual, position);
            });
    }
}

template <typename DomainT>
void require_bispheric_boundary_exports(const DomainT& domain, bool antisymmetric, bool eta_first, int chi_axis,
                                        std::initializer_list<std::pair<int, int>> bounds_and_omitted)
{
    Val_domain field = make_tau_characterization_field(&domain, antisymmetric);
    for (const auto [bound, omitted] : bounds_and_omitted) {
        const int ncond = domain.nbr_conditions_val_domain_boundary(field, bound);
        require_tau_export_bytes(
            ncond,
            [&](Array<double>& residual, int& position, int count) {
                domain.export_tau_val_domain_boundary(field, bound, residual, position, count);
            },
            [&](Array<double>& residual, int& position) {
                legacy_standard_boundary_export(domain, field, bound, chi_axis, omitted, residual, position);
            });
    }

    for (int phi_basis : {COS, SIN}) {
        Val_domain one_side_field = make_tau_one_side_characterization_field(&domain, phi_basis, eta_first);
        for (const auto [bound, omitted] : bounds_and_omitted) {
            const int ncond = domain.nbr_conditions_val_domain_boundary_one_side(one_side_field, bound);
            require_tau_export_bytes(
                ncond,
                [&](Array<double>& residual, int& position, int count) {
                    domain.export_tau_val_domain_boundary_one_side(one_side_field, bound, residual, position, count);
                },
                [&](Array<double>& residual, int& position) {
                    legacy_one_side_boundary_export(
                        domain, one_side_field, bound, chi_axis, omitted, residual, position);
                });
        }
    }
}

int residual_row_count(const Array<int>& counts)
{
    int total = 0;
    for (int component = 0; component < counts.get_nbr(); ++component)
        total += counts(component);
    return total;
}

void require_domain_descriptor_rows(
    const Domain& domain, int domain_index, int expected_count,
    const std::vector<ResidualRowDescriptor>& descriptors)
{
    REQUIRE(descriptors.size() == static_cast<std::size_t>(expected_count));
    int previous_phi = -1;
    for (const ResidualRowDescriptor& descriptor : descriptors) {
        REQUIRE(descriptor.family == ResidualRowEquationFamily::Unavailable);
        REQUIRE_FALSE(descriptor.available);
        REQUIRE(descriptor.equation_index == -1);
        REQUIRE(descriptor.explicit_sector == 0);
        REQUIRE(descriptor.sides.size() == 1);
        const ResidualRowCoordinate& coordinate = descriptor.sides.front();
        CHECK(coordinate.domain == domain_index);
        CHECK(coordinate.component == 0);
        CHECK((coordinate.phi_basis == COSSIN ||
               coordinate.phi_basis == COS ||
               coordinate.phi_basis == SIN));
        CHECK(domain.phi_coefficient_parity(
                  coordinate.phi_index, coordinate.phi_basis) != 0);
        CHECK(coordinate.phi_index >= previous_phi);
        CHECK(coordinate.phi_index >= 0);
        if (coordinate.phi_basis == COSSIN) {
            CHECK(coordinate.phi_index != 1);
            CHECK(coordinate.phi_index < domain.get_nbr_coefs()(2) - 1);
        } else {
            CHECK(coordinate.phi_index < domain.get_nbr_coefs()(2));
        }
        previous_phi = coordinate.phi_index;
    }
}

std::vector<int> descriptor_boundary_set(const Domain& domain)
{
    if (dynamic_cast<const Domain_nucleus_nosym*>(&domain) != nullptr)
        return {OUTER_BC};
    if (dynamic_cast<const Domain_nucleus*>(&domain) != nullptr)
        return {OUTER_BC};
    if (dynamic_cast<const Domain_bispheric_chi_first_nosym*>(&domain) != nullptr)
        return {INNER_BC, CHI_ONE_BC, OUTER_BC};
    if (dynamic_cast<const Domain_bispheric_chi_first*>(&domain) != nullptr)
        return {INNER_BC, CHI_ONE_BC, OUTER_BC};
    if (dynamic_cast<const Domain_bispheric_rect_nosym*>(&domain) != nullptr)
        return {INNER_BC, ETA_PLUS_BC, CHI_ONE_BC, OUTER_BC};
    if (dynamic_cast<const Domain_bispheric_rect*>(&domain) != nullptr)
        return {INNER_BC, ETA_PLUS_BC, CHI_ONE_BC, OUTER_BC};
    if (dynamic_cast<const Domain_bispheric_eta_first_nosym*>(&domain) != nullptr)
        return {ETA_MINUS_BC, ETA_PLUS_BC, OUTER_BC};
    if (dynamic_cast<const Domain_bispheric_eta_first*>(&domain) != nullptr)
        return {ETA_MINUS_BC, ETA_PLUS_BC, OUTER_BC};
    return {INNER_BC, OUTER_BC};
}

} // namespace

TEMPLATE_TEST_CASE("BNS mesh has no gaps with one shell per star and one exterior shell",
                   "[bns-domain-layout][bns-mesh][smoke]", Space_bin_ns, Space_bin_ns_nosym)
{
    require_layout_smoke<TestType>();
}

TEST_CASE("Space_bin_ns_nosym non-standard matching topology is complete and fail-closed",
          "[bns-domain-layout][bin-ns-nosym-topology]")
{
    const auto star_bounds = [](double inner, int shells) {
        std::vector<double> bounds{inner, inner + 0.35, inner + 0.70};
        for (int shell = 0; shell < shells; ++shell)
            bounds.push_back(inner + 0.95 + 0.25 * shell);
        return bounds;
    };
    const auto exterior_bounds = [](int shells) {
        std::vector<double> bounds{8.0};
        for (int shell = 0; shell < shells; ++shell)
            bounds.push_back(10.0 + 2.0 * shell);
        return bounds;
    };

    const std::array<std::tuple<int, int, int>, 4> layouts{
        std::tuple{0, 0, 0}, std::tuple{1, 2, 0},
        std::tuple{2, 1, 1}, std::tuple{2, 2, 2}};
    for (const auto [star1_shells, star2_shells, exterior_shells] : layouts) {
        INFO("star1 shells=" << star1_shells << ", star2 shells=" << star2_shells
             << ", exterior shells=" << exterior_shells);
        Space_bin_ns_nosym space(
            CHEB_TYPE, 8.0, star_bounds(0.50, star1_shells),
            star_bounds(0.55, star2_shells), exterior_bounds(exterior_shells), 5);
        REQUIRE(space.NS1 == 0);
        REQUIRE(space.ADAPTED1 == 1);
        REQUIRE(space.NS2 == 3 + star1_shells);
        REQUIRE(space.ADAPTED2 == 4 + star1_shells);
        REQUIRE(space.OUTER == 6 + star1_shells + star2_shells);
        REQUIRE(space.get_n_shells_outer() == exterior_shells);
        REQUIRE(space.get_nbr_domains() ==
                12 + star1_shells + star2_shells + exterior_shells);
        require_complete_nonstandard_matching_topology(
            space, star1_shells, star2_shells);
    }
}

TEST_CASE("BNS nosym deformed adapted surfaces remain ordered and gap-free after reload",
          "[bns-domain-layout][bns-mesh][deformed]")
{
    // Three bounds give nucleus + adapted pair, with no ordinary star shell.
    const std::vector<double> ns1_bounds{0.50, 1.00, 1.50};
    const std::vector<double> ns2_bounds{0.55, 1.05, 1.60};
    const std::vector<double> outer_bounds{8.0};
    Space_bin_ns_nosym space(CHEB_TYPE, 8.0, ns1_bounds, ns2_bounds, outer_bounds, 9);

    REQUIRE(space.get_nbr_domains() == 12);
    REQUIRE(space.NS1 == 0);
    REQUIRE(space.ADAPTED1 == 1);
    REQUIRE(space.NS2 == 3);
    REQUIRE(space.ADAPTED2 == 4);
    REQUIRE(space.OUTER == 6);
    REQUIRE(space.get_n_shells_outer() == 0);

    const std::array<DeformedStar, 2> stars{
        DeformedStar{space.ADAPTED1, ns1_bounds.front(), ns1_bounds.back(),
                     SurfaceShape{1.00, 0.06, 0.04, 0.025}, "NS1"},
        DeformedStar{space.ADAPTED2, ns2_bounds.front(), ns2_bounds.back(),
                     SurfaceShape{1.05, -0.055, 0.035, -0.020}, "NS2"}};

    for (const DeformedStar& star : stars)
        set_shared_surface(space, star);
    for (const DeformedStar& star : stars)
        require_deformed_star_covered(space, star);

    MemorySink first_sink;
    space.save(first_sink);
    MemorySource source(first_sink.buffer());
    Space_bin_ns_nosym restored(source);
    for (const DeformedStar& star : stars)
        require_deformed_star_covered(restored, star);

    MemorySink second_sink;
    restored.save(second_sink);
    REQUIRE(first_sink.buffer() == second_sink.buffer());
}

TEST_CASE("BNS nosym adapted-surface audit detects a deliberate finite gap",
          "[bns-domain-layout][bns-mesh][deformed][control]")
{
    const std::vector<double> ns1_bounds{0.50, 1.00, 1.50};
    const std::vector<double> ns2_bounds{0.55, 1.05, 1.60};
    const std::vector<double> outer_bounds{8.0};
    Space_bin_ns_nosym space(CHEB_TYPE, 8.0, ns1_bounds, ns2_bounds, outer_bounds, 9);
    const DeformedStar star{space.ADAPTED1, ns1_bounds.front(), ns1_bounds.back(),
                            SurfaceShape{1.00, 0.06, 0.04, 0.025}, "NS1 gap control"};

    constexpr double gap_width = 0.02;
    set_shared_surface(space, star, -0.5 * gap_width, 0.5 * gap_width);
    const auto* outer =
        dynamic_cast<const Domain_shell_outer_adapted_nosym*>(space.get_domain(space.ADAPTED1));
    const auto* inner =
        dynamic_cast<const Domain_shell_inner_adapted_nosym*>(space.get_domain(space.ADAPTED1 + 1));
    REQUIRE(outer != nullptr);
    REQUIRE(inner != nullptr);

    Val_domain outer_surface = outer->get_outer_radius();
    Val_domain inner_surface = inner->get_inner_radius();
    outer_surface.coef();
    inner_surface.coef();
    const Array<double> outer_coefficients = outer_surface.get_coef();
    const Array<double> inner_coefficients = inner_surface.get_coef();
    constexpr double theta = 1.1;
    constexpr double phi = 0.7;
    const double outer_value = evaluate_surface(outer_surface, outer_coefficients, theta, phi);
    const double inner_value = evaluate_surface(inner_surface, inner_coefficients, theta, phi);
    REQUIRE_THAT(inner_value - outer_value, WithinAbs(gap_width, 2.e-11));

    const Point gap_point = point_on_ray(outer->get_center(), 0.5 * (outer_value + inner_value), theta, phi);
    REQUIRE(outer->absol_to_num(gap_point)(1) > 1.0);
    REQUIRE(inner->absol_to_num(gap_point)(1) < -1.0);
    REQUIRE_FALSE(outer->is_in(gap_point));
    REQUIRE_FALSE(inner->is_in(gap_point));
    REQUIRE(containment_count(space, gap_point, 0.0) == 0);
}

TEST_CASE("Space_bin_ns_nosym tau exporters match legacy Index traversal byte-for-byte",
          "[bns-domain-layout][tau-export-order]")
{
    const std::vector<double> shelled_bounds{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> outer_shell_bounds{10.0};

    for (int basis_type : {CHEB_TYPE, LEG_TYPE}) {
        Space_bin_ns_nosym space(basis_type, 12.0, shelled_bounds, shelled_bounds, outer_shell_bounds, 5);
        const auto& outer_adapted =
            require_tau_domain<Domain_shell_outer_adapted_nosym>(space, space.ADAPTED1);
        const auto& inner_adapted =
            require_tau_domain<Domain_shell_inner_adapted_nosym>(space, space.ADAPTED1 + 1);
        const auto& shell = require_tau_domain<Domain_shell_nosym>(space, space.ADAPTED1 + 2);
        const auto& compact =
            require_tau_domain<Domain_compact_nosym>(space, space.get_nbr_domains() - 1);
        const auto& chi = require_tau_domain<Domain_bispheric_chi_first_nosym>(space, space.OUTER);
        const auto& rect = require_tau_domain<Domain_bispheric_rect_nosym>(space, space.OUTER + 1);
        const auto& eta = require_tau_domain<Domain_bispheric_eta_first_nosym>(space, space.OUTER + 2);

        for (bool antisymmetric : {false, true}) {
            require_spheric_interior_exports(outer_adapted, antisymmetric);
            require_spheric_interior_exports(inner_adapted, antisymmetric);
            require_spheric_interior_exports(shell, antisymmetric);
            require_spheric_interior_exports(compact, antisymmetric);
            require_bispheric_interior_exports(chi, antisymmetric, false);
            require_bispheric_interior_exports(rect, antisymmetric, false);
            require_bispheric_interior_exports(eta, antisymmetric, true);
            require_bispheric_boundary_exports(
                chi, antisymmetric, false, 1, {{INNER_BC, 0}, {OUTER_BC, 0}});
            require_bispheric_boundary_exports(
                eta, antisymmetric, true, 0, {{ETA_MINUS_BC, 1}, {ETA_PLUS_BC, 1}});
            require_bispheric_boundary_exports(
                rect, antisymmetric, false, 1, {{INNER_BC, 0}, {ETA_PLUS_BC, 1}});
        }
    }
}

TEST_CASE("Symmetric eta-first boundary export matches its count for unequal axes",
          "[bns-domain-layout][tau-export-order][eta-first-boundary]")
{
    Dim_array points(3);
    points.set(0) = 7;
    points.set(1) = 9;
    points.set(2) = 7;
    Domain_bispheric_eta_first domain(
        0, CHEB_TYPE, 1.0, 2.0, -0.5, 0.5, points);

    constexpr double canary = -1.23456789e300;
    for (int phi_basis : {COS, SIN}) {
        CAPTURE(phi_basis);
        Val_domain field =
            make_tau_one_side_characterization_field(&domain, phi_basis, true);
        const int count =
            domain.nbr_conditions_val_domain_boundary(field, ETA_MINUS_BC);
        REQUIRE(count > 0);

        Array<double> residual(count + points(2));
        residual = canary;
        int position = 0;
        domain.export_tau_val_domain_boundary(
            field, ETA_MINUS_BC, residual, position, count);

        CHECK(position == count);
        for (int index = count; index < residual.get_nbr(); ++index)
            CHECK(residual(index) == canary);
    }
}

TEMPLATE_TEST_CASE("BNS descriptors cover every emitted domain exporter order and boundary",
                   "[bns-domain-layout][residual_row_descriptor][tau-export-order]",
                   Space_bin_ns, Space_bin_ns_nosym)
{
    const std::vector<double> shelled_bounds{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> outer_shell_bounds{10.0, 14.0};

    for (int basis_type : {CHEB_TYPE, LEG_TYPE}) {
        TestType space(
            basis_type, 12.0, shelled_bounds, shelled_bounds,
            outer_shell_bounds, 5);
        for (bool antisymmetric : {false, true}) {
            Scalar field(space);
            field = 0.0;
            if (antisymmetric)
                field.std_anti_base();
            else
                field.std_base();
            field.coef();

            for (int domain_index = 0;
                 domain_index < space.get_nbr_domains(); ++domain_index) {
                const Domain& domain = *space.get_domain(domain_index);
                for (int order : {0, 1, 2}) {
                    CAPTURE(basis_type, antisymmetric, domain_index, order);
                    const Array<int> counts =
                        domain.nbr_conditions(field, domain_index, order);
                    std::vector<ResidualRowDescriptor> descriptors;
                    REQUIRE(domain.describe_volume_residual_rows(
                        field, domain_index, order, counts, -1, nullptr,
                        descriptors));
                    require_domain_descriptor_rows(
                        domain, domain_index, residual_row_count(counts),
                        descriptors);
                }

                for (int bound : descriptor_boundary_set(domain)) {
                    CAPTURE(basis_type, antisymmetric, domain_index, bound);
                    const Array<int> counts = domain.nbr_conditions_boundary(
                        field, domain_index, bound);
                    std::vector<ResidualRowDescriptor> descriptors;
                    REQUIRE(domain.describe_boundary_residual_rows(
                        field, domain_index, bound, counts, -1, nullptr,
                        descriptors));
                    require_domain_descriptor_rows(
                        domain, domain_index, residual_row_count(counts),
                        descriptors);
                }
            }
        }
    }
}
