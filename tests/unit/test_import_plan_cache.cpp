#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Diagnostics/matching_lane_profile.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"

#include <array>
#include <cstring>
#include <memory>
#include <new>
#include <utility>
#include <vector>

using namespace Kadath;

namespace {

std::vector<double> star_bounds()
{
    return {1.0, 2.0, 4.0};
}

std::vector<double> outer_bounds()
{
    return {10.0};
}

std::unique_ptr<Space_bin_ns_nosym> make_space()
{
    return std::make_unique<Space_bin_ns_nosym>(
        CHEB_TYPE, 12.0, star_bounds(), star_bounds(), outer_bounds(), 5);
}

Scalar make_scalar(Space_bin_ns_nosym& space, double seed)
{
    Scalar result(space);
    for (int domain = 0; domain < space.get_nbr_domains(); ++domain) {
        const Domain* current = space.get_domain(domain);
        result.set_domain(domain) = seed + 0.01 * current->get_cart(1) -
                                    0.02 * current->get_cart(2) +
                                    0.03 * current->get_cart(3);
    }
    result.std_base();
    result.coef();
    return result;
}

struct ImportInputs
{
    ImportInputs(int target, int boundary, int part_count)
        : target_domain(target), bound(boundary), source_domains(part_count)
    {
    }

    int target_domain;
    int bound;
    Array<int> source_domains;
    std::vector<Tensor*> lane_parts;
    std::vector<Tensor*> value_parts;
};

ImportInputs make_import_inputs(Space_bin_ns_nosym& space, Scalar& value,
                                Scalar& derivative)
{
    const Array<int> matching(
        space.get_indices_matching_non_std(space.OUTER, INNER_BC));
    const int part_count = matching.get_size(1);
    ImportInputs result(space.OUTER, INNER_BC, part_count);
    result.value_parts.reserve(static_cast<std::size_t>(part_count));
    result.lane_parts.reserve(static_cast<std::size_t>(2 * part_count));
    for (int part = 0; part < part_count; ++part) {
        result.source_domains.set(part) = matching(0, part);
        result.value_parts.push_back(&value);
        result.lane_parts.push_back(&value);
    }
    for (int part = 0; part < part_count; ++part)
        result.lane_parts.push_back(&derivative);
    return result;
}

std::array<std::unique_ptr<Tensor>, 2> run_native(
    const Domain& target, const ImportInputs& inputs)
{
    std::array<Tensor*, 2> raw{};
    const int part_count = static_cast<int>(inputs.source_domains.get_nbr());
    REQUIRE(target.import_lanes_native(
        inputs.target_domain, inputs.bound, part_count,
        inputs.source_domains, 2, inputs.lane_parts.data(), raw.data()));
    return {
        std::unique_ptr<Tensor>(raw[0]),
        std::unique_ptr<Tensor>(raw[1]),
    };
}

Tensor run_scalar_reference(const Domain& target, ImportInputs& inputs)
{
    const int part_count = static_cast<int>(inputs.source_domains.get_nbr());
    return target.import(inputs.target_domain, inputs.bound,
                         part_count, inputs.source_domains,
                         inputs.value_parts.data());
}

void require_domain_bytes_equal(const Tensor& actual, const Tensor& expected,
                                int domain)
{
    REQUIRE(actual.get_n_comp() == expected.get_n_comp());
    for (int component = 0; component < actual.get_n_comp(); ++component) {
        const Array<int> index(actual.indices(component));
        const Val_domain& actual_field = actual(index)(domain);
        const Val_domain& expected_field = expected(index)(domain);
        actual_field.coef();
        expected_field.coef();
        const Array<double>& actual_coefficients = actual_field.get_coef_ref();
        const Array<double>& expected_coefficients = expected_field.get_coef_ref();
        REQUIRE(actual_coefficients.get_nbr() == expected_coefficients.get_nbr());
        CHECK(std::memcmp(actual_coefficients.get_data(), expected_coefficients.get_data(),
                          actual_coefficients.get_nbr() * sizeof(double)) == 0);
    }
}

std::vector<double> domain_coefficients(const Tensor& tensor, int domain)
{
    std::vector<double> result;
    for (int component = 0; component < tensor.get_n_comp(); ++component) {
        const Array<int> index(tensor.indices(component));
        const Val_domain& field = tensor(index)(domain);
        field.coef();
        const Array<double>& coefficients = field.get_coef_ref();
        result.insert(result.end(), coefficients.get_data(),
                      coefficients.get_data() + coefficients.get_nbr());
    }
    return result;
}

bool bytes_differ(const std::vector<double>& left, const std::vector<double>& right)
{
    REQUIRE(left.size() == right.size());
    return std::memcmp(left.data(), right.data(), left.size() * sizeof(double)) != 0;
}

} // namespace

TEST_CASE("Import plans hit unchanged shape content and rebuild after source changes",
          "[import-plan-cache]")
{
    std::unique_ptr<Space_bin_ns_nosym> space = make_space();
    Scalar value(make_scalar(*space, 0.75));
    Scalar derivative(make_scalar(*space, -1.25));
    ImportInputs inputs(make_import_inputs(*space, value, derivative));
    const Domain* target = space->get_domain(inputs.target_domain);

    reset_matching_lane_stats();
    const auto first = run_native(*target, inputs);
    const auto unchanged = run_native(*target, inputs);
    require_domain_bytes_equal(*unchanged[0], *first[0], inputs.target_domain);
    CHECK(matching_lane_stats().import_plan_cache_hits == 1);
    CHECK(matching_lane_stats().import_plan_cache_misses == 1);
    CHECK(matching_lane_stats().import_plan_cache_rebuilds == 1);

    Domain* adapted_source = nullptr;
    std::vector<Val_domain> original_mapping;
    for (int part = 0; part < inputs.source_domains.get_nbr(); ++part) {
        Domain* candidate =
            const_cast<Domain*>(space->get_domain(inputs.source_domains(part)));
        std::vector<Val_domain> candidate_mapping;
        candidate->snapshot_mapping(candidate_mapping);
        if (!candidate_mapping.empty()) {
            adapted_source = candidate;
            original_mapping = std::move(candidate_mapping);
            break;
        }
    }
    REQUIRE(adapted_source != nullptr);
    Val_domain correction(original_mapping.front());
    correction *= 1.0e-6;
    adapted_source->update_mapping(correction);

    const auto changed = run_native(*target, inputs);
    const Tensor expected(run_scalar_reference(*target, inputs));
    require_domain_bytes_equal(*changed[0], expected, inputs.target_domain);
    CHECK(matching_lane_stats().import_plan_cache_hits == 1);
    CHECK(matching_lane_stats().import_plan_cache_misses == 2);
    CHECK(matching_lane_stats().import_plan_cache_rebuilds == 2);
}

TEST_CASE("Import plan ownership survives same-address Space reconstruction",
          "[import-plan-cache][aba]")
{
    struct DestroyOnly
    {
        void operator()(Space_bin_ns_nosym* space) const { std::destroy_at(space); }
    };

    alignas(Space_bin_ns_nosym)
        unsigned char storage[sizeof(Space_bin_ns_nosym)];
    const auto construct_space = [&]() {
        return std::unique_ptr<Space_bin_ns_nosym, DestroyOnly>(
            std::construct_at(reinterpret_cast<Space_bin_ns_nosym*>(storage),
                              CHEB_TYPE, 12.0, star_bounds(), star_bounds(),
                              outer_bounds(), 5));
    };

    reset_matching_lane_stats();
    Space_bin_ns_nosym* first_space_address = nullptr;
    std::vector<double> first_values;
    {
        auto first_space = construct_space();
        first_space_address = first_space.get();
        Scalar value(make_scalar(*first_space, 0.5));
        Scalar derivative(make_scalar(*first_space, -0.5));
        ImportInputs inputs(make_import_inputs(*first_space, value, derivative));
        const Domain* target = first_space->get_domain(inputs.target_domain);
        const auto first = run_native(*target, inputs);
        const auto cached = run_native(*target, inputs);
        require_domain_bytes_equal(*cached[0], *first[0], inputs.target_domain);
        first_values = domain_coefficients(*cached[0], inputs.target_domain);
    }

    {
        auto second_space = construct_space();
        REQUIRE(second_space.get() == first_space_address);
        Scalar value(make_scalar(*second_space, 1.5));
        Scalar derivative(make_scalar(*second_space, -1.5));
        ImportInputs inputs(make_import_inputs(*second_space, value, derivative));
        const Domain* target = second_space->get_domain(inputs.target_domain);
        const auto fresh = run_native(*target, inputs);
        const Tensor expected(run_scalar_reference(*target, inputs));
        require_domain_bytes_equal(*fresh[0], expected, inputs.target_domain);
        CHECK(bytes_differ(domain_coefficients(*fresh[0], inputs.target_domain),
                           first_values));
    }

    const MatchingLaneStats stats = matching_lane_stats();
    CHECK(stats.import_plan_cache_hits == 1);
    CHECK(stats.import_plan_cache_misses == 2);
    CHECK(stats.import_plan_cache_rebuilds == 2);
}
