#include <catch2/catch_test_macros.hpp>
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/index.hpp"
#include "For_Kadath/Array/memory.hpp"
#include "For_Kadath/Array/point.hpp"

#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace Kadath;

namespace
{
    class LateReleaseProbe
    {
      public:
        void arm()
        {
            allocation_ = MemoryMapper::get_memory(allocation_size);
        }

        [[nodiscard]] bool armed() const { return allocation_ != nullptr; }

        ~LateReleaseProbe()
        {
            if (allocation_ == nullptr)
                return;
            // The process-lifetime initial-thread cache remains valid while
            // static objects run down.
            MemoryMapper::release_memory(allocation_, allocation_size);
            void* reused = MemoryMapper::get_memory(allocation_size);
            if (reused != allocation_)
                std::abort();
            MemoryMapper::release_memory(reused, allocation_size);
        }

      private:
        static constexpr std::size_t allocation_size = 63487U;
        void* allocation_ = nullptr;
    };

    LateReleaseProbe late_release_probe;

    class EarlyTlsReleaseProbe
    {
      public:
        void arm(void* allocation, std::size_t size)
        {
            allocation_ = allocation;
            size_ = size;
        }

        ~EarlyTlsReleaseProbe()
        {
            if (allocation_ == nullptr)
                return;
            // This worker object is initialized before MemoryMapper's retire
            // guard, so its destructor deliberately runs after that guard.
            // The call pair must use the central pool and retain active-scope
            // traffic.
            MemoryMapper::release_memory(allocation_, size_);
            void* reused = MemoryMapper::get_memory(size_);
            if (reused != allocation_)
                std::abort();
            MemoryMapper::release_memory(reused, size_);
        }

      private:
        void* allocation_ = nullptr;
        std::size_t size_ = 0U;
    };

    EarlyTlsReleaseProbe& early_tls_release_probe()
    {
        thread_local EarlyTlsReleaseProbe probe;
        return probe;
    }
}

TEST_CASE("MemoryMapper alloc and release", "[memory]") {
    size_t sz = 128;  // bytes, not elements
    void* ptr = MemoryMapper::get_memory(sz);
    REQUIRE(ptr != nullptr);
    MemoryMapper::release_memory(ptr, sz);
}

TEST_CASE("MemoryMapper reuse", "[memory]") {
    size_t sz = 256;
    void* ptr1 = MemoryMapper::get_memory(sz);
    MemoryMapper::release_memory(ptr1, sz);
    void* ptr2 = MemoryMapper::get_memory(sz);
    // Pool should return the same pointer
    REQUIRE(ptr2 == ptr1);
    MemoryMapper::release_memory(ptr2, sz);
}

TEST_CASE("MemoryMapper isolates the initial direct cache and recovers retired workers",
          "[memory][memory-threading]") {
    constexpr std::size_t sz = 64229U;
    void* initial = MemoryMapper::get_memory(sz);
    MemoryMapper::release_memory(initial, sz);

    void* worker = nullptr;
    std::thread producer([&] {
        worker = MemoryMapper::get_memory(sz);
        MemoryMapper::release_memory(worker, sz);
    });
    producer.join();

    REQUIRE(worker != initial);
    void* initial_reused = MemoryMapper::get_memory(sz);
    REQUIRE(initial_reused == initial);
    void* worker_recovered = MemoryMapper::get_memory(sz);
    REQUIRE(worker_recovered == worker);
    MemoryMapper::release_memory(worker_recovered, sz);
    MemoryMapper::release_memory(initial_reused, sz);
}

TEST_CASE("MemoryMapper preserves aligned live slabs", "[memory]") {
    constexpr size_t sz = 257;
    void* first = MemoryMapper::get_memory(sz);
    void* second = MemoryMapper::get_memory(sz);

    REQUIRE(first != nullptr);
    REQUIRE(second != nullptr);
    REQUIRE(first != second);
    REQUIRE(reinterpret_cast<std::uintptr_t>(first) % alignof(std::max_align_t) == 0);
    REQUIRE(reinterpret_cast<std::uintptr_t>(second) % alignof(std::max_align_t) == 0);

    MemoryMapper::release_memory(first, sz);
    MemoryMapper::release_memory(second, sz);
    REQUIRE(MemoryMapper::get_memory(sz) == second);
    MemoryMapper::release_memory(second, sz);
}

TEST_CASE("MemoryMapper Jacobian traffic scope reuses released group blocks without live aliasing",
          "[memory][jacobian-traffic]") {
    constexpr size_t sz = 4097;
    MemoryMapperJacobianTrafficScope traffic_scope(true);

    void* first_live = MemoryMapper::get_memory(sz);
    void* second_live = MemoryMapper::get_memory(sz);
    REQUIRE(first_live != second_live);

    MemoryMapper::release_memory(first_live, sz);
    MemoryMapper::release_memory(second_live, sz);
    void* next_group = MemoryMapper::get_memory(sz);
    REQUIRE(next_group == second_live);
    MemoryMapper::release_memory(next_group, sz);

    const MemoryMapperTrafficSnapshot traffic = traffic_scope.finish();
    REQUIRE(traffic.get_calls == 3);
    REQUIRE(traffic.release_calls == 3);
    REQUIRE(traffic.requested_get_bytes == 3 * sz);
    REQUIRE(traffic.requested_release_bytes == 3 * sz);
}

TEST_CASE("MemoryMapper Jacobian traffic epochs reset live worker shards",
          "[memory][jacobian-traffic][memory-threading]") {
    constexpr std::array<std::size_t, 2> calls{7U, 11U};
    constexpr std::array<std::size_t, 2> sizes{1031U, 2053U};
    std::array<MemoryMapperTrafficSnapshot, 2> snapshots{};
    std::barrier boundary(2);

    std::thread worker([&] {
        for (std::size_t round = 0; round < calls.size(); ++round) {
            boundary.arrive_and_wait();
            for (std::size_t call = 0; call < calls[round]; ++call) {
                void* allocation = MemoryMapper::get_memory(sizes[round]);
                MemoryMapper::release_memory(allocation, sizes[round]);
            }
            boundary.arrive_and_wait();
            boundary.arrive_and_wait();
        }
    });

    for (std::size_t round = 0; round < calls.size(); ++round) {
        MemoryMapperJacobianTrafficScope traffic_scope(true);
        boundary.arrive_and_wait();
        boundary.arrive_and_wait();
        snapshots[round] = traffic_scope.finish();
        boundary.arrive_and_wait();
    }
    worker.join();

    for (std::size_t round = 0; round < calls.size(); ++round) {
        REQUIRE(snapshots[round].get_calls == calls[round]);
        REQUIRE(snapshots[round].release_calls == calls[round]);
        REQUIRE(snapshots[round].requested_get_bytes ==
                calls[round] * sizes[round]);
        REQUIRE(snapshots[round].requested_release_bytes ==
                calls[round] * sizes[round]);
    }
}

TEST_CASE("MemoryMapper rejects overlapping Jacobian traffic scopes without cancelling the owner",
          "[memory][jacobian-traffic]") {
    constexpr std::size_t sz = 3083U;
    MemoryMapperJacobianTrafficScope traffic_scope(true);
    REQUIRE_THROWS_AS(MemoryMapper::begin_jacobian_traffic_profile(true),
                      std::logic_error);

    void* allocation = MemoryMapper::get_memory(sz);
    MemoryMapper::release_memory(allocation, sz);
    const MemoryMapperTrafficSnapshot traffic = traffic_scope.finish();
    REQUIRE(traffic.get_calls == 1U);
    REQUIRE(traffic.release_calls == 1U);
    REQUIRE(traffic.requested_get_bytes == sz);
    REQUIRE(traffic.requested_release_bytes == sz);
}

TEST_CASE("MemoryMapper accounts allocator calls after a worker cache retires",
          "[memory][jacobian-traffic][memory-threading]") {
    constexpr std::size_t sz = 62003U;
    MemoryMapperJacobianTrafficScope traffic_scope(true);
    std::thread worker([] {
        EarlyTlsReleaseProbe& probe = early_tls_release_probe();
        probe.arm(MemoryMapper::get_memory(sz), sz);
    });
    worker.join();

    const MemoryMapperTrafficSnapshot traffic = traffic_scope.finish();
    REQUIRE(traffic.get_calls == 2U);
    REQUIRE(traffic.release_calls == 2U);
    REQUIRE(traffic.requested_get_bytes == 2U * sz);
    REQUIRE(traffic.requested_release_bytes == 2U * sz);
}

TEST_CASE("MemoryMapper free-list metadata fits tiny odd requests", "[memory]") {
    constexpr size_t tiny_sz = 1;
    auto* tiny = static_cast<unsigned char*>(MemoryMapper::get_memory(tiny_sz));
    tiny[0] = 0xa5;
    MemoryMapper::release_memory(tiny, tiny_sz);

    auto* reused = static_cast<unsigned char*>(MemoryMapper::get_memory(tiny_sz));
    REQUIRE(reused == tiny);
    reused[0] = 0x5a;
    REQUIRE(reused[0] == 0x5a);
    MemoryMapper::release_memory(reused, tiny_sz);

    constexpr size_t odd_sz = sizeof(void*) + 3U;
    auto* odd = static_cast<unsigned char*>(MemoryMapper::get_memory(odd_sz));
    for (size_t i = 0; i < odd_sz; ++i)
        odd[i] = static_cast<unsigned char>(i);
    for (size_t i = 0; i < odd_sz; ++i)
        REQUIRE(odd[i] == static_cast<unsigned char>(i));
    MemoryMapper::release_memory(odd, odd_sz);
}

TEST_CASE("MemoryMapper reuses large slabs", "[memory]") {
    constexpr size_t sz = 128U * 1024U + 3U;
    void* first = MemoryMapper::get_memory(sz);
    MemoryMapper::release_memory(first, sz);
    void* second = MemoryMapper::get_memory(sz);
    REQUIRE(second == first);
    MemoryMapper::release_memory(second, sz);
}

TEST_CASE("MemoryMapper keeps concurrent direct allocations distinct",
          "[memory][memory-threading]") {
    constexpr std::size_t worker_count = 4U;
    constexpr std::size_t blocks_per_worker = 64U;
    constexpr std::size_t sz = 32003U;
    std::vector<void*> live_blocks(worker_count * blocks_per_worker);
    std::barrier all_allocated(static_cast<std::ptrdiff_t>(worker_count));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);

    MemoryMapperJacobianTrafficScope traffic_scope(true);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            const std::size_t offset = worker * blocks_per_worker;
            for (std::size_t block = 0; block < blocks_per_worker; ++block)
                live_blocks[offset + block] = MemoryMapper::get_memory(sz);
            all_allocated.arrive_and_wait();
            for (std::size_t block = 0; block < blocks_per_worker; ++block)
                MemoryMapper::release_memory(live_blocks[offset + block], sz);
        });
    }
    for (auto& worker : workers)
        worker.join();

    const std::unordered_set<void*> unique_blocks(live_blocks.begin(),
                                                   live_blocks.end());
    const MemoryMapperTrafficSnapshot traffic = traffic_scope.finish();
    REQUIRE(unique_blocks.size() == live_blocks.size());
    REQUIRE(traffic.get_calls == live_blocks.size());
    REQUIRE(traffic.release_calls == live_blocks.size());
    REQUIRE(traffic.requested_get_bytes == live_blocks.size() * sz);
    REQUIRE(traffic.requested_release_bytes == live_blocks.size() * sz);
}

TEST_CASE("MemoryMapper permits direct and large cross-thread release",
          "[memory][memory-threading]") {
    constexpr std::size_t direct_sz = 49171U;
    constexpr std::size_t large_sz = 128U * 1024U + 317U;
    std::atomic<void*> direct{nullptr};
    std::atomic<void*> large{nullptr};
    std::barrier handoff(2);

    std::thread producer([&] {
        direct.store(MemoryMapper::get_memory(direct_sz),
                     std::memory_order_release);
        large.store(MemoryMapper::get_memory(large_sz),
                    std::memory_order_release);
        handoff.arrive_and_wait();
        handoff.arrive_and_wait();
    });

    std::thread releaser([&] {
        handoff.arrive_and_wait();
        MemoryMapper::release_memory(direct.load(std::memory_order_acquire),
                                     direct_sz);
        MemoryMapper::release_memory(large.load(std::memory_order_acquire),
                                     large_sz);
        handoff.arrive_and_wait();
    });
    producer.join();
    releaser.join();

    void* reused_direct = MemoryMapper::get_memory(direct_sz);
    void* reused_large = MemoryMapper::get_memory(large_sz);
    REQUIRE(reused_direct == direct.load(std::memory_order_acquire));
    REQUIRE(reused_large == large.load(std::memory_order_acquire));
    MemoryMapper::release_memory(reused_direct, direct_sz);
    MemoryMapper::release_memory(reused_large, large_sz);
}

TEST_CASE("MemoryMapper recovers a direct cache when its thread exits",
          "[memory][memory-threading]") {
    constexpr std::size_t sz = 57347U;
    void* released = nullptr;
    std::thread producer([&] {
        released = MemoryMapper::get_memory(sz);
        MemoryMapper::release_memory(released, sz);
    });
    producer.join();

    void* recovered = nullptr;
    std::thread consumer([&] {
        recovered = MemoryMapper::get_memory(sz);
        MemoryMapper::release_memory(recovered, sz);
    });
    consumer.join();
    REQUIRE(recovered == released);
}

TEST_CASE("MemoryMapper permits release after initial thread cache teardown",
          "[memory][memory-late-release]") {
    late_release_probe.arm();
    REQUIRE(late_release_probe.armed());
}

TEST_CASE("MemoryMapper handles zero and overflowing requests", "[memory]") {
    REQUIRE(MemoryMapper::get_memory(0) == nullptr);
    REQUIRE_THROWS_AS(MemoryMapper::get_memory(std::numeric_limits<size_t>::max()), std::bad_alloc);
    REQUIRE_THROWS_AS(
        MemoryMapper::get_memory(std::numeric_limits<size_t>::max() - alignof(std::max_align_t)),
        std::bad_alloc);
}

TEST_CASE("MemoryMapper phase profile accounts live and retained slabs",
          "[memory][memory-phase-profile]") {
    if (!MemoryMapper::phase_profile_enabled())
        SKIP("requires MEMORY_MAPPER_PHASE_PROFILE=1 at process start");

    constexpr std::size_t direct_requested = 257U;
    constexpr std::size_t large_requested = 3U * 1024U * 1024U + 123U;
    constexpr std::size_t alignment = alignof(std::max_align_t);
    constexpr std::size_t direct_capacity =
        (direct_requested + alignment - 1U) & ~(alignment - 1U);
    constexpr std::size_t large_capacity =
        (large_requested + alignment - 1U) & ~(alignment - 1U);
    constexpr std::size_t requested_total =
        direct_requested + large_requested;
    constexpr std::size_t capacity_total =
        direct_capacity + large_capacity;

    const MemoryMapperPhaseSnapshot before = MemoryMapper::phase_snapshot();
    void* direct_first = MemoryMapper::get_memory(direct_requested);
    void* large_first = MemoryMapper::get_memory(large_requested);
    const MemoryMapperPhaseSnapshot after_get = MemoryMapper::phase_snapshot();
    const std::uint64_t miss_delta =
        after_get.pool_misses - before.pool_misses;
    const std::uint64_t hit_delta =
        after_get.pool_hits - before.pool_hits;
    const std::uint64_t reserved_delta =
        after_get.capacity_reserved_bytes - before.capacity_reserved_bytes;

    REQUIRE(after_get.enabled);
    REQUIRE(after_get.get_calls == before.get_calls + 2U);
    REQUIRE(hit_delta + miss_delta == 2U);
    REQUIRE(after_get.system_malloc_calls ==
            before.system_malloc_calls + miss_delta);
    REQUIRE(after_get.system_malloc_payload_bytes ==
            before.system_malloc_payload_bytes + reserved_delta);
    REQUIRE(reserved_delta <= capacity_total);
    if (miss_delta == 0U) {
        REQUIRE(after_get.header_reserved_bytes ==
                before.header_reserved_bytes);
    } else {
        REQUIRE(after_get.header_reserved_bytes >
                before.header_reserved_bytes);
    }
    REQUIRE(after_get.requested_live_bytes ==
            before.requested_live_bytes + requested_total);
    REQUIRE(after_get.capacity_live_bytes ==
            before.capacity_live_bytes + capacity_total);
    REQUIRE(after_get.live_blocks == before.live_blocks + 2U);
    REQUIRE(after_get.total_blocks == before.total_blocks + miss_delta);
    REQUIRE(after_get.capacity_reserved_bytes ==
            after_get.capacity_live_bytes + after_get.capacity_free_bytes);
    REQUIRE(after_get.total_blocks ==
            after_get.live_blocks + after_get.free_blocks);
    REQUIRE(after_get.peak_requested_live_bytes >=
            after_get.requested_live_bytes);
    REQUIRE(after_get.peak_capacity_live_bytes >=
            after_get.capacity_live_bytes);
    REQUIRE(after_get.peak_capacity_reserved_bytes >=
            after_get.capacity_reserved_bytes);

    MemoryMapper::release_memory(direct_first, direct_requested);
    MemoryMapper::release_memory(large_first, large_requested);
    const MemoryMapperPhaseSnapshot after_release =
        MemoryMapper::phase_snapshot();
    REQUIRE(after_release.release_calls == before.release_calls + 2U);
    REQUIRE(after_release.requested_live_bytes == before.requested_live_bytes);
    REQUIRE(after_release.capacity_live_bytes == before.capacity_live_bytes);
    REQUIRE(after_release.capacity_free_bytes ==
            before.capacity_free_bytes + reserved_delta);
    REQUIRE(after_release.live_blocks == before.live_blocks);
    REQUIRE(after_release.free_blocks == before.free_blocks + miss_delta);

    void* direct_reused = MemoryMapper::get_memory(direct_requested);
    void* large_reused = MemoryMapper::get_memory(large_requested);
    const MemoryMapperPhaseSnapshot after_reuse =
        MemoryMapper::phase_snapshot();
    REQUIRE(direct_reused == direct_first);
    REQUIRE(large_reused == large_first);
    REQUIRE(after_reuse.pool_hits == after_release.pool_hits + 2U);
    REQUIRE(after_reuse.pool_misses == after_release.pool_misses);
    REQUIRE(after_reuse.system_malloc_calls == after_release.system_malloc_calls);
    REQUIRE(after_reuse.capacity_reserved_bytes ==
            after_release.capacity_reserved_bytes);
    REQUIRE(after_reuse.capacity_live_bytes ==
            before.capacity_live_bytes + capacity_total);
    REQUIRE(after_reuse.capacity_free_bytes + capacity_total ==
            after_release.capacity_free_bytes);

    MemoryMapper::release_memory(direct_reused, direct_requested);
    MemoryMapper::release_memory(large_reused, large_requested);
    const MemoryMapperPhaseSnapshot final = MemoryMapper::phase_snapshot();
    REQUIRE(final.requested_live_bytes == before.requested_live_bytes);
    REQUIRE(final.capacity_live_bytes == before.capacity_live_bytes);
    REQUIRE(final.capacity_reserved_bytes ==
            final.capacity_live_bytes + final.capacity_free_bytes);
    REQUIRE(final.total_blocks == final.live_blocks + final.free_blocks);
}

TEST_CASE("Small coordinate objects bypass MemoryMapper with heap fallback counts",
          "[memory][memory-phase-profile][inline-storage]") {
    if (!MemoryMapper::phase_profile_enabled())
        SKIP("requires MEMORY_MAPPER_PHASE_PROFILE=1 at process start");

    auto require_call_delta = [](const MemoryMapperPhaseSnapshot& before,
                                 const MemoryMapperPhaseSnapshot& after,
                                 std::uint64_t expected_gets,
                                 std::uint64_t expected_releases) {
        REQUIRE(after.get_calls - before.get_calls == expected_gets);
        REQUIRE(after.release_calls - before.release_calls ==
                expected_releases);
    };

    MemoryMapperPhaseSnapshot before = MemoryMapper::phase_snapshot();
    {
        Dim_array dimensions(0);
    }
    MemoryMapperPhaseSnapshot after = MemoryMapper::phase_snapshot();
    require_call_delta(before, after, 0U, 0U);

    Dim_array inline_sizes(3);
    for (int i = 0; i < inline_sizes.get_ndim(); ++i)
        inline_sizes.set(i) = 2;
    before = MemoryMapper::phase_snapshot();
    {
        Index index(inline_sizes);
    }
    after = MemoryMapper::phase_snapshot();
    require_call_delta(before, after, 0U, 0U);

    before = MemoryMapper::phase_snapshot();
    {
        Point point(3);
    }
    after = MemoryMapper::phase_snapshot();
    require_call_delta(before, after, 0U, 0U);

    before = MemoryMapper::phase_snapshot();
    {
        Dim_array dimensions(4);
    }
    after = MemoryMapper::phase_snapshot();
    require_call_delta(before, after, 1U, 1U);

    Dim_array fallback_sizes(4);
    for (int i = 0; i < fallback_sizes.get_ndim(); ++i)
        fallback_sizes.set(i) = 2;
    before = MemoryMapper::phase_snapshot();
    {
        Index index(fallback_sizes);
    }
    after = MemoryMapper::phase_snapshot();
    require_call_delta(before, after, 2U, 2U);

    before = MemoryMapper::phase_snapshot();
    {
        Point point(4);
    }
    after = MemoryMapper::phase_snapshot();
    require_call_delta(before, after, 1U, 1U);

    before = MemoryMapper::phase_snapshot();
    {
        Array<int> empty(0);
        Array<int> scalar(1);
        Array<int> pair(2);
    }
    after = MemoryMapper::phase_snapshot();
    require_call_delta(before, after, 0U, 0U);

    before = MemoryMapper::phase_snapshot();
    {
        Array<int> fallback_three(3);
        Array<int> fallback_four(4);
    }
    after = MemoryMapper::phase_snapshot();
    require_call_delta(before, after, 2U, 2U);

    before = MemoryMapper::phase_snapshot();
    {
        Array<double> generic_small(2);
    }
    after = MemoryMapper::phase_snapshot();
    require_call_delta(before, after, 1U, 1U);
}

TEST_CASE("MemoryMapper phase profile accounts cross-thread release",
          "[memory][memory-phase-profile][memory-threading]") {
    if (!MemoryMapper::phase_profile_enabled())
        SKIP("requires MEMORY_MAPPER_PHASE_PROFILE=1 at process start");

    constexpr std::size_t requested = 60013U;
    constexpr std::size_t alignment = alignof(std::max_align_t);
    constexpr std::size_t capacity =
        (requested + alignment - 1U) & ~(alignment - 1U);
    const MemoryMapperPhaseSnapshot before = MemoryMapper::phase_snapshot();

    void* allocation = nullptr;
    std::thread producer([&] {
        allocation = MemoryMapper::get_memory(requested);
    });
    producer.join();
    const MemoryMapperPhaseSnapshot after_get = MemoryMapper::phase_snapshot();

    std::thread releaser([&] {
        MemoryMapper::release_memory(allocation, requested);
    });
    releaser.join();
    const MemoryMapperPhaseSnapshot after_release =
        MemoryMapper::phase_snapshot();

    REQUIRE(after_get.get_calls == before.get_calls + 1U);
    REQUIRE(after_get.requested_live_bytes ==
            before.requested_live_bytes + requested);
    REQUIRE(after_get.capacity_live_bytes ==
            before.capacity_live_bytes + capacity);
    REQUIRE(after_get.live_blocks == before.live_blocks + 1U);
    REQUIRE(after_release.release_calls == before.release_calls + 1U);
    REQUIRE(after_release.requested_live_bytes ==
            before.requested_live_bytes);
    REQUIRE(after_release.capacity_live_bytes == before.capacity_live_bytes);
    REQUIRE(after_release.live_blocks == before.live_blocks);
    REQUIRE(after_release.total_blocks ==
            after_release.live_blocks + after_release.free_blocks);
}

TEST_CASE("MemoryMapper persistent cross-thread traffic reaches a reserve plateau",
          "[memory][memory-phase-profile][memory-threading]") {
    if (!MemoryMapper::phase_profile_enabled())
        SKIP("requires MEMORY_MAPPER_PHASE_PROFILE=1 at process start");

    constexpr std::size_t rounds = 4U;
    constexpr std::size_t blocks_per_round = 16U;
    constexpr std::size_t requested = 61237U;
    std::array<std::array<void*, blocks_per_round>, rounds> allocations{};
    std::array<MemoryMapperPhaseSnapshot, rounds> snapshots{};
    std::barrier phase_boundary(3);
    const MemoryMapperPhaseSnapshot before = MemoryMapper::phase_snapshot();

    std::thread producer([&] {
        for (std::size_t round = 0; round < rounds; ++round) {
            for (void*& allocation : allocations[round])
                allocation = MemoryMapper::get_memory(requested);
            phase_boundary.arrive_and_wait();
            phase_boundary.arrive_and_wait();
            phase_boundary.arrive_and_wait();
        }
    });
    std::thread consumer([&] {
        for (std::size_t round = 0; round < rounds; ++round) {
            phase_boundary.arrive_and_wait();
            for (void* allocation : allocations[round])
                MemoryMapper::release_memory(allocation, requested);
            phase_boundary.arrive_and_wait();
            phase_boundary.arrive_and_wait();
        }
    });
    for (std::size_t round = 0; round < rounds; ++round) {
        phase_boundary.arrive_and_wait();
        phase_boundary.arrive_and_wait();
        snapshots[round] = MemoryMapper::phase_snapshot();
        phase_boundary.arrive_and_wait();
    }
    producer.join();
    consumer.join();

    for (const auto& snapshot : snapshots) {
        REQUIRE(snapshot.requested_live_bytes ==
                before.requested_live_bytes);
        REQUIRE(snapshot.capacity_live_bytes == before.capacity_live_bytes);
        REQUIRE(snapshot.live_blocks == before.live_blocks);
    }
    REQUIRE(snapshots[2].system_malloc_calls ==
            snapshots[3].system_malloc_calls);
    REQUIRE(snapshots[2].capacity_reserved_bytes ==
            snapshots[3].capacity_reserved_bytes);
}
