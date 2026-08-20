#include <catch2/catch_test_macros.hpp>

#include "Linear_algebra/jacobian_assembler.hpp"
#include "Linear_algebra/jacobian_parity_mask.hpp"

#include <cstddef>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

using namespace Kadath;

namespace
{

using SelectionPlanPtr = std::shared_ptr<const JacobianSelectionPlan>;

SelectionPlanPtr make_sector_plan(int selected_label)
{
    const int excluded_label = -selected_label;
    const JacobianSelectionPlanBuild built = make_jacobian_selection_plan(
        {1, -1, 1, -1}, {-1, 1, 1, -1}, selected_label,
        excluded_label);
    REQUIRE(built.plan);
    REQUIRE(built.fallback_reason.empty());
    return built.plan;
}

AssembledJacobianCooBlock make_plus_block()
{
    AssembledJacobianCooBlock block;
    block.parity_label = 1;
    block.n = 2;
    block.nnz = 4;
    block.selection_plan = make_sector_plan(1);
    block.irn = {1, 2, 1, 2};
    block.jcn = {1, 1, 2, 2};
    block.a = {4.0, 1.0, 1.0, 3.0};
    return block;
}

AssembledJacobianCooBlock make_minus_block()
{
    AssembledJacobianCooBlock block;
    block.parity_label = -1;
    block.n = 2;
    block.nnz = 4;
    block.selection_plan = make_sector_plan(-1);
    block.irn = {1, 2, 1, 2};
    block.jcn = {1, 1, 2, 2};
    block.a = {5.0, 2.0, 2.0, 6.0};
    return block;
}

void check_block_storage(const AssembledJacobianCooBlock& block)
{
    REQUIRE(block.nnz >= 0);
    const auto nnz = static_cast<std::size_t>(block.nnz);
    CHECK(block.irn.size() == nnz);
    CHECK(block.jcn.size() == nnz);
    CHECK(block.a.size() == nnz);
    for (const int row : block.irn) {
        CHECK(row >= 1);
        CHECK(row <= block.n);
    }
    for (const int column : block.jcn) {
        CHECK(column >= 1);
        CHECK(column <= block.n);
    }
}

void check_physical_storage(const AssembledJacobianCoo& matrix)
{
    CHECK(matrix.irn.empty());
    CHECK(matrix.jcn.empty());
    CHECK(matrix.a.empty());
    const long long aggregate_nnz = std::accumulate(
        matrix.parity_blocks.begin(), matrix.parity_blocks.end(), 0LL,
        [](long long sum, const AssembledJacobianCooBlock& block) {
            return sum + block.nnz;
        });
    CHECK(matrix.nnz == aggregate_nnz);
    for (const AssembledJacobianCooBlock& block : matrix.parity_blocks) {
        check_block_storage(block);
    }
}

} // namespace

TEST_CASE("legacy assembled Jacobian COO has no physical parity blocks",
          "[jacobian-assembler][physical-parity-blocks]")
{
    AssembledJacobianCoo matrix;
    matrix.n = 2;
    matrix.nnz = 2;
    matrix.irn = {1, 2};
    matrix.jcn = {1, 2};
    matrix.a = {3.0, 4.0};

    CHECK(matrix.parity_blocks.empty());
    CHECK(matrix.irn == AssembledJacobianCoo::IndexVector{1, 2});
    CHECK(matrix.jcn == AssembledJacobianCoo::IndexVector{1, 2});
    CHECK(matrix.a == AssembledJacobianCoo::ValueVector{3.0, 4.0});
}

TEST_CASE("selected pre-J1 physical COO contains one ordered plus block",
          "[jacobian-assembler][physical-parity-blocks][selection_plan]")
{
    static_assert(std::is_same_v<decltype(AssembledJacobianCooBlock::selection_plan),
                                 SelectionPlanPtr>);

    AssembledJacobianCoo matrix;
    matrix.n = 2;
    matrix.nnz = 4;
    matrix.parity_sector_block_diagonal = true;
    matrix.parity_blocks.push_back(make_plus_block());

    REQUIRE(matrix.parity_blocks.size() == 1);
    const AssembledJacobianCooBlock& plus = matrix.parity_blocks.front();
    CHECK(plus.parity_label == 1);
    CHECK(plus.selection_plan->selected_block() == 1);
    CHECK(plus.selection_plan->selected_rows() == std::vector<int>{0, 2});
    CHECK(plus.selection_plan->selected_columns() == std::vector<int>{1, 2});
    CHECK(plus.irn == AssembledJacobianCooBlock::IndexVector{1, 2, 1, 2});
    CHECK(plus.jcn == AssembledJacobianCooBlock::IndexVector{1, 1, 2, 2});
    CHECK(plus.a == AssembledJacobianCooBlock::ValueVector{4.0, 1.0, 1.0, 3.0});
    check_physical_storage(matrix);
}

TEST_CASE("full physical COO keeps plus then minus as independent ordered blocks",
          "[jacobian-assembler][physical-parity-blocks]")
{
    AssembledJacobianCoo matrix;
    matrix.n = 4;
    matrix.nnz = 8;
    matrix.parity_sector_block_diagonal = true;
    matrix.parity_blocks.push_back(make_plus_block());
    matrix.parity_blocks.push_back(make_minus_block());

    REQUIRE(matrix.parity_blocks.size() == 2);
    const AssembledJacobianCooBlock& plus = matrix.parity_blocks[0];
    const AssembledJacobianCooBlock& minus = matrix.parity_blocks[1];
    CHECK(plus.parity_label == 1);
    CHECK(minus.parity_label == -1);
    CHECK(plus.selection_plan->selected_rows() == std::vector<int>{0, 2});
    CHECK(plus.selection_plan->selected_columns() == std::vector<int>{1, 2});
    CHECK(minus.selection_plan->selected_rows() == std::vector<int>{1, 3});
    CHECK(minus.selection_plan->selected_columns() == std::vector<int>{0, 3});
    CHECK(plus.a == AssembledJacobianCooBlock::ValueVector{4.0, 1.0, 1.0, 3.0});
    CHECK(minus.a == AssembledJacobianCooBlock::ValueVector{5.0, 2.0, 2.0, 6.0});
    check_physical_storage(matrix);
}
