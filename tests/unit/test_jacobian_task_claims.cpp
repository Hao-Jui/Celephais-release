#include <catch2/catch_test_macros.hpp>

#include "Linear_algebra/jacobian_task_claims.hpp"

#include <array>
#include <barrier>
#include <cstdint>
#include <limits>
#include <span>
#include <thread>

using namespace Kadath;

namespace
{

    template <typename Group>
    concept ClaimableJacobianGroup = requires(JacobianTaskClaims& claims, Group group) { claims.try_claim(group); };

    using FixedGroup = std::span<const int, 2>;
    using DynamicGroup = std::span<const int>;

    static_assert(ClaimableJacobianGroup<FixedGroup>);
    static_assert(!ClaimableJacobianGroup<DynamicGroup>);
    static_assert(alignof(jacobian_task_claim_detail::Slot) >= JacobianTaskClaims::required_alignment);

} // namespace

TEST_CASE("Jacobian task claims support release, reclaim, and completion", "[jacobian-task-claims]")
{
    JacobianTaskClaims claims(6);
    const std::array group{1, 3, 5};
    const std::span<const int, 3> group_view{group};

    REQUIRE(claims.try_claim(group_view));
    CHECK(claims.state(1) == JacobianTaskState::Claimed);
    CHECK(claims.state(3) == JacobianTaskState::Claimed);
    CHECK(claims.state(5) == JacobianTaskState::Claimed);

    REQUIRE(claims.release(group_view));
    CHECK(claims.state(1) == JacobianTaskState::Pending);
    CHECK(claims.state(3) == JacobianTaskState::Pending);
    CHECK(claims.state(5) == JacobianTaskState::Pending);

    REQUIRE(claims.try_claim(group_view));
    REQUIRE(claims.complete(group_view));
    CHECK(claims.state(1) == JacobianTaskState::Complete);
    CHECK(claims.state(3) == JacobianTaskState::Complete);
    CHECK(claims.state(5) == JacobianTaskState::Complete);
    CHECK_FALSE(claims.try_claim(group_view));
}

TEST_CASE("only one contender claims the same Jacobian task group", "[jacobian-task-claims][threaded]")
{
    JacobianTaskClaims claims(4);
    const std::array group{1, 2};
    const std::span<const int, 2> group_view{group};
    std::array<bool, 2> claimed{};
    std::barrier start_line(3);

    std::thread first([&] {
        start_line.arrive_and_wait();
        claimed[0] = claims.try_claim(group_view);
    });
    std::thread second([&] {
        start_line.arrive_and_wait();
        claimed[1] = claims.try_claim(group_view);
    });
    start_line.arrive_and_wait();
    first.join();
    second.join();

    CHECK(claimed[0] != claimed[1]);
    CHECK(claims.state(1) == JacobianTaskState::Claimed);
    CHECK(claims.state(2) == JacobianTaskState::Claimed);
}

TEST_CASE("failed overlapping Jacobian task claim rolls back its prefix", "[jacobian-task-claims]")
{
    JacobianTaskClaims claims(5);
    const std::array occupied{2};
    const std::array overlapping{0, 1, 2};

    REQUIRE(claims.try_claim(std::span<const int, 1>{occupied}));
    CHECK_FALSE(claims.try_claim(std::span<const int, 3>{overlapping}));

    CHECK(claims.state(0) == JacobianTaskState::Pending);
    CHECK(claims.state(1) == JacobianTaskState::Pending);
    CHECK(claims.state(2) == JacobianTaskState::Claimed);
}

TEST_CASE("invalid Jacobian task groups are refused without partial state changes", "[jacobian-task-claims]")
{
    JacobianTaskClaims claims(4);
    const std::array duplicate{1, 1};
    const std::array past_end{0, 4};
    const std::array negative{-1, 2};
    const std::array<std::uint64_t, 1> unrepresentable_or_past_end{std::numeric_limits<std::uint64_t>::max()};

    CHECK_FALSE(claims.try_claim(std::span<const int, 2>{duplicate}));
    CHECK_FALSE(claims.try_claim(std::span<const int, 2>{past_end}));
    CHECK_FALSE(claims.try_claim(std::span<const int, 2>{negative}));
    CHECK_FALSE(claims.try_claim(std::span<const std::uint64_t, 1>{unrepresentable_or_past_end}));

    for (std::size_t index = 0; index < claims.size(); ++index)
        CHECK(claims.state(index) == JacobianTaskState::Pending);
    CHECK_THROWS_AS(claims.state(claims.size()), std::out_of_range);
}

TEST_CASE("empty fixed Jacobian task groups are valid no-ops", "[jacobian-task-claims]")
{
    JacobianTaskClaims claims(0);
    const std::array<int, 0> empty{};
    const std::span<const int, 0> empty_view{empty};

    CHECK(claims.try_claim(empty_view));
    CHECK(claims.release(empty_view));
    CHECK(claims.complete(empty_view));
}
