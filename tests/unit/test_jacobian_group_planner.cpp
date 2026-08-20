#include <catch2/catch_test_macros.hpp>

#include "Linear_algebra/jacobian_group_planner.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace Kadath;

namespace
{

    struct PlannerFixture {
        std::vector<ColumnMetadata> metadata;
        std::vector<bool> direct;
    };

    void append_bucket(PlannerFixture& fixture, int count, ColumnClass column_class, int domain,
                       const char* variable_name)
    {
        for (int i = 0; i < count; ++i) {
            ColumnMetadata column;
            column.column = static_cast<int>(fixture.metadata.size());
            column.column_class = column_class;
            column.domain = domain;
            column.var_name = variable_name;
            fixture.metadata.push_back(column);
            fixture.direct.push_back(false);
        }
    }

    PlannerFixture mixed_fixture()
    {
        PlannerFixture fixture;
        append_bucket(fixture, 35, ColumnClass::FieldInterior, 0, "u");
        fixture.direct[2] = true;
        append_bucket(fixture, 18, ColumnClass::FieldBoundary, 1, "v");
        append_bucket(fixture, 1, ColumnClass::VarDomain, 2, "shape");
        return fixture;
    }

    JacobianGroupPlannerOptions options(int nproc)
    {
        JacobianGroupPlannerOptions result;
        result.nproc = nproc;
        return result;
    }

} // namespace

TEST_CASE("global Jacobian group planner is deterministic", "[jacobian-group-planner]")
{
    const PlannerFixture fixture = mixed_fixture();
    const JacobianGlobalGroupPlan first =
        build_global_jacobian_group_plan(fixture.metadata, fixture.direct, options(4));
    const JacobianGlobalGroupPlan second =
        build_global_jacobian_group_plan(fixture.metadata, fixture.direct, options(4));

    CHECK(first == second);
}

TEST_CASE("global Jacobian group planner gives every column one whole owner", "[jacobian-group-planner]")
{
    const PlannerFixture fixture = mixed_fixture();
    const JacobianGlobalGroupPlan plan = build_global_jacobian_group_plan(fixture.metadata, fixture.direct, options(4));
    std::vector<int> owner_count(fixture.metadata.size(), 0);

    for (const JacobianColumnGroup& group : plan.groups) {
        REQUIRE(group.owner_rank >= 0);
        REQUIRE(group.owner_rank < 4);
        for (int column : group.columns) {
            REQUIRE(column >= 0);
            REQUIRE(static_cast<std::size_t>(column) < owner_count.size());
            ++owner_count[static_cast<std::size_t>(column)];
        }
    }
    CHECK(std::all_of(owner_count.begin(), owner_count.end(), [](int count) { return count == 1; }));

    for (int rank = 0; rank < 4; ++rank) {
        const std::vector<std::size_t>& rank_groups = plan.group_indices_by_rank[static_cast<std::size_t>(rank)];
        CHECK(std::is_sorted(rank_groups.begin(), rank_groups.end()));
        for (std::size_t group_index : rank_groups) {
            CHECK(plan.groups[group_index].owner_rank == rank);
        }
    }
}

TEST_CASE("global Jacobian group planner preserves packed cascade counts", "[jacobian-group-planner]")
{
    const PlannerFixture fixture = mixed_fixture();
    const JacobianGlobalGroupPlan plan = build_global_jacobian_group_plan(fixture.metadata, fixture.direct, options(4));
    std::array<int, 33> groups_by_width{};
    int direct_groups = 0;
    for (const JacobianColumnGroup& group : plan.groups) {
        ++groups_by_width[group.columns.size()];
        if (group.direct)
            ++direct_groups;
    }

    // Bucket u: direct column removed, leaving W32 + W2. Bucket v: W16 + W2.
    // The direct u column and VarDomain column remain scalar singletons.
    CHECK(groups_by_width[32] == 1);
    CHECK(groups_by_width[16] == 1);
    CHECK(groups_by_width[8] == 0);
    CHECK(groups_by_width[4] == 0);
    CHECK(groups_by_width[2] == 2);
    CHECK(groups_by_width[1] == 2);
    CHECK(direct_groups == 1);

    JacobianGroupPlannerOptions narrower = options(4);
    narrower.use_wlane32 = false;
    const JacobianGlobalGroupPlan narrower_plan = build_global_jacobian_group_plan(
        fixture.metadata, fixture.direct, narrower);
    groups_by_width.fill(0);
    for (const JacobianColumnGroup& group : narrower_plan.groups)
        ++groups_by_width[group.columns.size()];
    CHECK(groups_by_width[32] == 0);
    CHECK(groups_by_width[16] == 3);
    CHECK(groups_by_width[2] == 2);
    CHECK(groups_by_width[1] == 2);
}

TEST_CASE("global Jacobian group planner cost assignment is balanced", "[jacobian-group-planner]")
{
    PlannerFixture fixture;
    append_bucket(fixture, 96, ColumnClass::FieldInterior, 0, "u");
    append_bucket(fixture, 64, ColumnClass::FieldBoundary, 1, "v");
    append_bucket(fixture, 37, ColumnClass::FieldGauge, 2, "w");
    const JacobianGlobalGroupPlan plan = build_global_jacobian_group_plan(fixture.metadata, fixture.direct, options(4));

    const auto minmax = std::minmax_element(plan.estimated_cost_by_rank.begin(), plan.estimated_cost_by_rank.end());
    const long long max_group_cost =
        std::max_element(plan.groups.begin(), plan.groups.end(),
                         [](const JacobianColumnGroup& lhs, const JacobianColumnGroup& rhs) {
                             return lhs.estimated_cost < rhs.estimated_cost;
                         })
            ->estimated_cost;
    long long rank_cost_sum = 0;
    for (long long cost : plan.estimated_cost_by_rank)
        rank_cost_sum += cost;
    long long group_cost_sum = 0;
    for (const JacobianColumnGroup& group : plan.groups)
        group_cost_sum += group.estimated_cost;

    // Deterministic list scheduling bounds the final load spread by the
    // largest indivisible group cost.
    CHECK(rank_cost_sum == group_cost_sum);
    CHECK(*minmax.second - *minmax.first <= max_group_cost);
}

TEST_CASE("global Jacobian group planner handles empty work and rejects invalid layouts",
          "[jacobian-group-planner]")
{
    const JacobianGlobalGroupPlan empty = build_global_jacobian_group_plan(
        {}, {}, options(6));
    CHECK(empty.groups.empty());
    CHECK(empty.group_indices_by_rank.size() == 6);
    CHECK(empty.estimated_cost_by_rank == std::vector<long long>(6, 0));

    JacobianGroupPlannerOptions invalid = options(0);
    CHECK_THROWS_AS(build_global_jacobian_group_plan({}, {}, invalid),
                    std::invalid_argument);
    CHECK_THROWS_AS(build_global_jacobian_group_plan(
                        std::vector<ColumnMetadata>(1), {}, options(1)),
                    std::invalid_argument);
}
