#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Diagnostics/matching_lane_profile.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Domain/spheric_nosym.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Tensor/vector.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace Kadath;

namespace {

class ScopedEnvironment
{
  public:
    ScopedEnvironment(const char* name, const char* value) : name_(name)
    {
        if (const char* current = std::getenv(name))
            old_value_ = current;
        if (value == nullptr)
            ::unsetenv(name);
        else
            ::setenv(name, value, 1);
    }

    ~ScopedEnvironment()
    {
        if (old_value_)
            ::setenv(name_.c_str(), old_value_->c_str(), 1);
        else
            ::unsetenv(name_.c_str());
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

  private:
    std::string name_;
    std::optional<std::string> old_value_;
};

Point origin()
{
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;
    return center;
}

Dim_array resolution()
{
    Dim_array res(3);
    res.set(0) = 5;
    res.set(1) = 5;
    res.set(2) = 4;
    return res;
}

Array<double> bounds()
{
    Array<double> result(2);
    result.set(0) = 1.0;
    result.set(1) = 3.0;
    return result;
}

std::vector<double> star_bounds()
{
    return {1.0, 2.0, 4.0};
}

std::vector<double> outer_bounds()
{
    return {10.0};
}

Vector make_cartesian_vector(Space_bin_ns_nosym& space, double seed)
{
    Base_tensor basis(space, CARTESIAN_BASIS);
    Vector result(space, CON, basis);
    for (int domain = 0; domain < space.get_nbr_domains(); ++domain) {
        const Domain* current = space.get_domain(domain);
        const Val_domain x = current->get_cart(1);
        const Val_domain y = current->get_cart(2);
        const Val_domain z = current->get_cart(3);
        result.set(1).set_domain(domain) = seed + 0.01 * x + 0.02 * y - 0.03 * z;
        result.set(2).set_domain(domain) = -0.5 * seed + 0.04 * x - 0.02 * y + 0.01 * z;
        result.set(3).set_domain(domain) = 0.25 * seed - 0.03 * x + 0.01 * y + 0.02 * z;
    }
    result.std_base();
    result.coef();
    return result;
}

void require_tensor_domain_bytes_equal(const Tensor& actual, const Tensor& expected, int domain)
{
    REQUIRE(actual.get_valence() == expected.get_valence());
    REQUIRE(actual.get_n_comp() == expected.get_n_comp());
    REQUIRE(actual.get_basis().get_basis(domain) == expected.get_basis().get_basis(domain));
    for (int component = 0; component < actual.get_n_comp(); ++component) {
        const Array<int> index(actual.indices(component));
        const Val_domain& actual_domain = actual(index)(domain);
        const Val_domain& expected_domain = expected(index)(domain);
        REQUIRE(actual_domain.check_if_zero() == expected_domain.check_if_zero());
        REQUIRE(actual_domain.get_base() == expected_domain.get_base());
        actual_domain.coef();
        expected_domain.coef();
        const Array<double> actual_coefficients(actual_domain.get_coef());
        const Array<double> expected_coefficients(expected_domain.get_coef());
        REQUIRE(actual_coefficients.get_nbr() == expected_coefficients.get_nbr());
        CHECK(std::memcmp(actual_coefficients.get_data(), expected_coefficients.get_data(),
                          actual_coefficients.get_nbr() * sizeof(double)) == 0);
    }
}

void require_native_vector_import_matches_scalar(Space_bin_ns_nosym& space, int target_domain, int bound)
{
    const Array<int> matching(space.get_indices_matching_non_std(target_domain, bound));
    const int part_count = matching.get_size(1);
    Array<int> source_domains(part_count);
    for (int part = 0; part < part_count; ++part)
        source_domains.set(part) = matching(0, part);

    Vector value(make_cartesian_vector(space, 1.0));
    Vector first_derivative(make_cartesian_vector(space, 2.0));
    Vector second_derivative(make_cartesian_vector(space, -3.0));
    std::vector<Tensor*> value_parts(static_cast<std::size_t>(part_count), &value);
    std::vector<Tensor*> first_parts(static_cast<std::size_t>(part_count), &first_derivative);
    std::vector<Tensor*> second_parts(static_cast<std::size_t>(part_count), &second_derivative);

    const Domain* target = space.get_domain(target_domain);
    Tensor expected_value(target->import(target_domain, bound, part_count, source_domains,
                                         value_parts.data()));
    Tensor expected_first(target->import(target_domain, bound, part_count, source_domains,
                                         first_parts.data()));
    Tensor expected_second(target->import(target_domain, bound, part_count, source_domains,
                                          second_parts.data()));

    std::vector<Tensor*> lane_parts;
    lane_parts.reserve(static_cast<std::size_t>(3 * part_count));
    lane_parts.insert(lane_parts.end(), value_parts.begin(), value_parts.end());
    lane_parts.insert(lane_parts.end(), first_parts.begin(), first_parts.end());
    lane_parts.insert(lane_parts.end(), second_parts.begin(), second_parts.end());
    std::array<Tensor*, 3> raw_outputs{};
    REQUIRE(target->import_lanes_native(target_domain, bound, part_count, source_domains, 3,
                                        lane_parts.data(), raw_outputs.data()));
    std::array<std::unique_ptr<Tensor>, 3> outputs{
        std::unique_ptr<Tensor>(raw_outputs[0]),
        std::unique_ptr<Tensor>(raw_outputs[1]),
        std::unique_ptr<Tensor>(raw_outputs[2]),
    };
    require_tensor_domain_bytes_equal(*outputs[0], expected_value, target_domain);
    require_tensor_domain_bytes_equal(*outputs[1], expected_first, target_domain);
    require_tensor_domain_bytes_equal(*outputs[2], expected_second, target_domain);
}

template <std::size_t W, typename SpaceT>
std::array<int, W> build_matching_system(SpaceT& space, Scalar& u, System_of_eqs& system)
{
    for (int domain = 0; domain < space.get_nbr_domains(); ++domain) {
        const Domain* current = space.get_domain(domain);
        u.set_domain(domain) = 1.0 + 0.01 * current->get_cart(1)
                                    - 0.02 * current->get_cart(2)
                                    + 0.03 * current->get_cart(3);
    }
    u.std_base();
    u.coef();
    system.add_var("u", u);
    for (int domain = 0; domain < space.get_nbr_domains(); ++domain)
        system.add_eq_full(domain, "u*u=0", -1, nullptr, "u");
    system.add_eq_matching(0, OUTER_BC, "u", -1, nullptr, "u");
    system.add_eq_matching_import(1, OUTER_BC, "u", -1, nullptr, "u");
    (void)system.sec_member();

    std::vector<ColumnMetadata> metadata;
    system.classify_columns(metadata);
    std::vector<std::vector<int>> by_domain(static_cast<std::size_t>(space.get_nbr_domains()));
    for (const ColumnMetadata& column : metadata) {
        if (column.column_class != ColumnClass::VarDomain && column.domain >= 0)
            by_domain[static_cast<std::size_t>(column.domain)].push_back(column.column);
    }

    // Interleave domains so a packed tile carries derivatives on both sides
    // of each seam and leaves genuinely missing lanes on each individual term.
    std::array<int, W> columns{};
    int count = 0;
    for (std::size_t offset = 0; count < static_cast<int>(columns.size()); ++offset) {
        bool found = false;
        for (const std::vector<int>& domain_columns : by_domain) {
            if (offset >= domain_columns.size())
                continue;
            columns[static_cast<std::size_t>(count++)] = domain_columns[offset];
            found = true;
            if (count == static_cast<int>(columns.size()))
                break;
        }
        REQUIRE(found);
    }
    return columns;
}

struct SparseCollector
{
    std::vector<std::pair<int, double>>* entries = nullptr;
    void operator()(int row, double value) { entries->emplace_back(row, value); }
};

struct FinalizeProbe
{
    inline static int live_objects = 0;

    FinalizeProbe() { ++live_objects; }
    ~FinalizeProbe() { --live_objects; }
};

template <std::size_t W, std::size_t... Lane>
std::array<SparseColumnEmitter, W> make_sparse_emitters(
    std::array<SparseCollector, W>& collectors, std::index_sequence<Lane...>)
{
    return {SparseColumnEmitter{collectors[Lane]}...};
}

template <std::size_t W>
void require_wide_sparse_parity(System_of_eqs& system, const std::array<int, W>& columns)
{
    static_assert(W == 16 || W == 32);
    std::array<std::vector<std::pair<int, double>>, W> scalar;
    std::array<std::vector<std::pair<int, double>>, W> packed;
    std::array<SparseCollector, W> scalar_collectors{};
    std::array<SparseCollector, W> packed_collectors{};
    for (std::size_t lane = 0; lane < columns.size(); ++lane) {
        scalar_collectors[lane].entries = &scalar[lane];
        packed_collectors[lane].entries = &packed[lane];
        system.do_col_J_sparse(columns[lane], 0.0, SparseColumnEmitter{scalar_collectors[lane]});
    }
    auto emitters = make_sparse_emitters<W>(packed_collectors, std::make_index_sequence<W>{});
    std::string failure_reason;
    bool packed_ok = false;
    if constexpr (W == 16)
        packed_ok = system.do_cols_J_wlane16_sparse(columns, 0.0, emitters, failure_reason);
    else
        packed_ok = system.do_cols_J_wlane32_sparse(columns, 0.0, emitters, failure_reason);
    REQUIRE(packed_ok);
    INFO(failure_reason);
    for (std::size_t lane = 0; lane < columns.size(); ++lane) {
        REQUIRE(packed[lane].size() == scalar[lane].size());
        for (std::size_t entry = 0; entry < scalar[lane].size(); ++entry) {
            CHECK(packed[lane][entry].first == scalar[lane][entry].first);
            CHECK(std::memcmp(&packed[lane][entry].second, &scalar[lane][entry].second,
                              sizeof(double)) == 0);
        }
    }
}

template <std::size_t W>
void require_w2_w4_w8_w16_parity(System_of_eqs& system, const std::array<int, W>& columns)
{
    static_assert(W >= 16);
    std::ostringstream report;
    REQUIRE(system.validate_packed_wlane2_columns(columns[0], columns[1], 0.0, 0.0, report));
    const std::array<int, 4> quartet = {columns[0], columns[1], columns[2], columns[3]};
    REQUIRE(system.validate_packed_wlane4_columns(quartet, 0.0, 0.0, report));
    const std::array<int, 8> octet = {
        columns[0], columns[1], columns[2], columns[3],
        columns[4], columns[5], columns[6], columns[7],
    };
    REQUIRE(system.validate_packed_wlane8_columns(octet, 0.0, 0.0, report));
    INFO(report.str());
    std::array<int, 16> lanes16{};
    std::copy_n(columns.begin(), lanes16.size(), lanes16.begin());
    require_wide_sparse_parity(system, lanes16);
}

} // namespace

TEST_CASE("Matching lane output ownership remains local when finalization throws",
          "[matching-lanes]")
{
    std::vector<std::unique_ptr<FinalizeProbe>> outputs;
    for (int lane = 0; lane < 3; ++lane)
        outputs.push_back(std::make_unique<FinalizeProbe>());
    std::array<FinalizeProbe*, 3> raw_outputs{};

    REQUIRE(FinalizeProbe::live_objects == 3);
    CHECK_THROWS_AS(
        matching_lane_detail::finalize_then_release(
            outputs, raw_outputs.data(),
            [](FinalizeProbe&, std::size_t lane) {
                if (lane == 1)
                    throw std::runtime_error("injected lane finalization failure");
            }),
        std::runtime_error);
    CHECK(raw_outputs[0] == nullptr);
    CHECK(raw_outputs[1] == nullptr);
    CHECK(raw_outputs[2] == nullptr);
    CHECK(FinalizeProbe::live_objects == 3);

    outputs.clear();
    CHECK(FinalizeProbe::live_objects == 0);
}

TEST_CASE("Matching lane gates can select the scalar fallbacks", "[matching-lanes]")
{
    ScopedEnvironment export_gate("MATCHING_LANE_EXPORT", "0");
    ScopedEnvironment import_gate("MATCHING_IMPORT_LANE_BATCH", "0");
    Space_spheric_nosym space(CHEB_TYPE, origin(), resolution(), bounds(), true);
    Scalar u(space);
    System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
    const std::array<int, 16> columns = build_matching_system<16>(space, u, system);

    reset_matching_lane_stats();
    std::ostringstream report;
    REQUIRE(system.validate_packed_wlane2_columns(columns[0], columns[1], 0.0, 0.0, report));
    INFO(report.str());
    const MatchingLaneStats stats = matching_lane_stats();
    CHECK(stats.export_native_calls == 0);
    CHECK(stats.export_scalar_fallback_calls > 0);
    CHECK(stats.import_native_calls == 0);
    CHECK(stats.import_scalar_fallback_calls > 0);
}

TEST_CASE("No-sym matching lane paths preserve W2 W4 W8 W16 W32 columns", "[matching-lanes]")
{
    ScopedEnvironment export_gate("MATCHING_LANE_EXPORT", nullptr);
    ScopedEnvironment import_gate("MATCHING_IMPORT_LANE_BATCH", nullptr);
    Space_spheric_nosym space(CHEB_TYPE, origin(), resolution(), bounds(), true);
    Scalar u(space);
    System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
    const std::array<int, 32> columns = build_matching_system<32>(space, u, system);

    reset_matching_lane_stats();
    require_w2_w4_w8_w16_parity(system, columns);
    require_wide_sparse_parity(system, columns);
    const MatchingLaneStats stats = matching_lane_stats();
    CHECK(stats.export_native_calls > 0);
    CHECK(stats.export_scalar_fallback_calls == 0);
    CHECK(stats.export_missing_lanes > 0);
    CHECK(stats.import_native_calls > 0);
    CHECK(stats.import_refusals == 0);
    CHECK(stats.import_plan_points > 0);
    CHECK(stats.import_missing_inputs > 0);
    CHECK(stats.import_scalar_fallback_calls > 0);
}

TEST_CASE("Adapted and bispheric no-sym lane imports preserve Cartesian vectors",
          "[matching-lanes]")
{
    Space_bin_ns_nosym space(CHEB_TYPE, 12.0, star_bounds(), star_bounds(), outer_bounds(), 5);

    // The star-side outer-adapted domain imports from two bispheric charts.
    require_native_vector_import_matches_scalar(space, space.ADAPTED1 + 1, OUTER_BC);
    // The first bispheric chi chart imports from the outer-adapted star domain.
    require_native_vector_import_matches_scalar(space, space.OUTER, INNER_BC);
}

TEST_CASE("Unsupported symmetric import batching refuses to the scalar virtual path", "[matching-lanes]")
{
    ScopedEnvironment export_gate("MATCHING_LANE_EXPORT", "1");
    ScopedEnvironment import_gate("MATCHING_IMPORT_LANE_BATCH", "1");
    Space_spheric space(CHEB_TYPE, origin(), resolution(), bounds());
    Scalar u(space);
    System_of_eqs system(space, 0, space.get_nbr_domains() - 1);
    const std::array<int, 16> columns = build_matching_system<16>(space, u, system);

    reset_matching_lane_stats();
    std::ostringstream report;
    REQUIRE(system.validate_packed_wlane2_columns(columns[0], columns[1], 0.0, 0.0, report));
    INFO(report.str());
    const MatchingLaneStats stats = matching_lane_stats();
    CHECK(stats.import_native_calls == 0);
    CHECK(stats.import_refusals > 0);
    CHECK(stats.import_scalar_fallback_calls > 0);
}
