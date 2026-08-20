//
// Created by sauliac on 14/04/2020.
//

#include "For_Kadath/Array/memory.hpp"

#include <algorithm>
#include <mutex>

namespace Kadath
{
    struct MemoryMapper::CentralPool {
        std::mutex free_mutex;
        std::array<void*, direct_slab_count> direct_free{};
        large_free_map_t large_free;
        BlockHeader* blocks = nullptr;

        std::mutex profile_mutex;
        MemoryMapperPhaseSnapshot phase{};
        std::vector<ThreadState*> thread_states;
        MemoryMapperTrafficSnapshot exited_traffic{};
        std::uint64_t traffic_epoch = 0U;
        bool traffic_active = false;
    };

    constinit MemoryMapper::ThreadState MemoryMapper::initial_thread_state{};
    constinit thread_local MemoryMapper::ThreadState MemoryMapper::thread_state{};
    thread_local MemoryMapper::ThreadStateRetirer MemoryMapper::thread_state_retirer{};
    const void* const MemoryMapper::initial_thread_identity =
        current_thread_identity();
    alignas(std::atomic_ref<std::uint64_t>::required_alignment)
        constinit std::uint64_t MemoryMapper::allocator_control = 0U;
    const bool MemoryMapper::initial_thread_state_registered = [] {
        // This state and the central pool outlive ordinary static destructors,
        // so late initial-thread releases never need an already-retired TLS
        // object or a process-wide teardown mode.
        ThreadState& state = initial_thread_state;
        state.initial_thread = true;
        initialize_thread_state(state);
        return true;
    }();

    MemoryMapper::ThreadStateRetirer::~ThreadStateRetirer() noexcept
    {
        MemoryMapper::retire_thread_state(MemoryMapper::thread_state);
    }

    MemoryMapper::CentralPool& MemoryMapper::central_pool()
    {
        // Never destroy the mutexes or retained slabs: releases from static or
        // thread-local objects in other translation units may arrive during
        // shutdown. The OS reclaims the process-lifetime allocation.
        static auto* pool = new CentralPool;
        return *pool;
    }

    void MemoryMapper::initialize_thread_state(ThreadState& state)
    {
        if (state.initialized || state.retired)
            return;

        if (!state.initial_thread) {
            // Force this worker's non-trivial guard to register after the POD
            // state has begun its lifetime. Later calls touch only the state.
            ThreadStateRetirer* volatile guard = &thread_state_retirer;
            static_cast<void>(guard);
        }

        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.profile_mutex);
        if (std::find(pool.thread_states.begin(), pool.thread_states.end(),
                      &state) == pool.thread_states.end()) {
            pool.thread_states.push_back(&state);
        }
        state.initialized = true;
    }

    void MemoryMapper::retire_thread_state(ThreadState& state) noexcept
    {
        if (state.initial_thread)
            return;

        if (!state.initialized) {
            state.retired = true;
            return;
        }

        flush_thread_state(state);

        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.profile_mutex);
        if (pool.traffic_active &&
            state.traffic_epoch == pool.traffic_epoch) {
            pool.exited_traffic.get_calls += state.traffic.get_calls;
            pool.exited_traffic.release_calls += state.traffic.release_calls;
            pool.exited_traffic.requested_get_bytes +=
                state.traffic.requested_get_bytes;
            pool.exited_traffic.requested_release_bytes +=
                state.traffic.requested_release_bytes;
        }
        const auto position = std::find(pool.thread_states.begin(),
                                        pool.thread_states.end(), &state);
        if (position != pool.thread_states.end())
            pool.thread_states.erase(position);
        state.retired = true;
    }

    void* MemoryMapper::acquire_central_slab(ThreadState* const state,
                                             size_t const capacity,
                                             bool& pool_hit)
    {
        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.free_mutex);

        void* raw_mem_ptr = nullptr;
        if (capacity <= direct_slab_limit) {
            const size_t index = capacity / slab_alignment;
            if (state != nullptr && !state->retired) {
                std::uint8_t& target = state->direct_cache_target[index];
                if (target == 0U)
                    target = direct_cache_min_per_class;
                const std::uint8_t maximum = direct_cache_maximum(capacity);
                if (target < maximum) {
                    target = std::min<std::uint8_t>(
                        maximum, static_cast<std::uint8_t>(target * 2U));
                }
            }
            raw_mem_ptr = pop_free(pool.direct_free[index]);
            if (raw_mem_ptr != nullptr && state != nullptr && !state->retired) {
                const std::uint8_t cache_limit =
                    direct_cache_limit(*state, index);
                std::uint8_t transferred = 1U;
                while (transferred < direct_transfer_batch &&
                       state->direct_free_count[index] < cache_limit &&
                       state->direct_cached_bytes <=
                           direct_cache_total_budget - capacity) {
                    void* prefetched = pop_free(pool.direct_free[index]);
                    if (prefetched == nullptr)
                        break;
                    push_free(state->direct_free[index], prefetched);
                    ++state->direct_free_count[index];
                    state->direct_cached_bytes += capacity;
                    ++transferred;
                }
            }
        } else {
            auto [position, inserted] =
                pool.large_free.emplace(capacity, nullptr);
            static_cast<void>(inserted);
            raw_mem_ptr = pop_free(position->second);
        }
        if (raw_mem_ptr != nullptr) {
            pool_hit = true;
            return raw_mem_ptr;
        }

        if (capacity > std::numeric_limits<size_t>::max() - sizeof(BlockHeader))
            throw std::bad_alloc();
        void* allocation = std::malloc(sizeof(BlockHeader) + capacity);
        if (allocation == nullptr)
            throw std::bad_alloc();

        auto* block = static_cast<BlockHeader*>(allocation);
        block->next = pool.blocks;
        pool.blocks = block;
        pool_hit = false;
        return block + 1;
    }

    void MemoryMapper::release_large_slab(void* raw_mem_ptr,
                                          size_t const capacity)
    {
        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.free_mutex);
        push_free(pool.large_free[capacity], raw_mem_ptr);
    }

    void MemoryMapper::release_direct_central(void* raw_mem_ptr,
                                              size_t const capacity)
    {
        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.free_mutex);
        push_free(pool.direct_free[capacity / slab_alignment], raw_mem_ptr);
    }

    void MemoryMapper::release_direct_overflow(ThreadState& state,
                                               void* raw_mem_ptr,
                                               size_t const capacity)
    {
        const size_t index = capacity / slab_alignment;
        const std::uint8_t cached = state.direct_free_count[index];
        const std::uint8_t drain_count = std::min<std::uint8_t>(
            direct_transfer_batch, std::max<std::uint8_t>(1U, cached / 2U));

        void* drained = nullptr;
        std::uint8_t drained_count = 0U;
        while (drained_count < drain_count) {
            void* block = pop_free(state.direct_free[index]);
            if (block == nullptr)
                break;
            push_free(drained, block);
            ++drained_count;
        }
        state.direct_free_count[index] -= drained_count;
        state.direct_cached_bytes -=
            static_cast<size_t>(drained_count) * capacity;

        if (state.direct_free_count[index] <
                direct_cache_limit(state, index) &&
            state.direct_cached_bytes <= direct_cache_total_budget - capacity) {
            push_free(state.direct_free[index], raw_mem_ptr);
            ++state.direct_free_count[index];
            state.direct_cached_bytes += capacity;
        } else {
            push_free(drained, raw_mem_ptr);
        }

        if (drained == nullptr)
            return;
        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.free_mutex);
        while (drained != nullptr) {
            void* block = pop_free(drained);
            push_free(pool.direct_free[index], block);
        }
    }

    void MemoryMapper::flush_thread_state(ThreadState& state) noexcept
    {
        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.free_mutex);
        for (size_t index = 0; index < direct_slab_count; ++index) {
            void*& local_head = state.direct_free[index];
            while (local_head != nullptr) {
                void* raw_mem_ptr = pop_free(local_head);
                push_free(pool.direct_free[index], raw_mem_ptr);
            }
            state.direct_free_count[index] = 0U;
        }
        state.direct_cached_bytes = 0U;
    }

    void MemoryMapper::record_profiled_get(ThreadState& state,
                                           size_t const requested,
                                           size_t const capacity,
                                           bool const pool_hit,
                                           std::uint64_t const traffic_epoch)
    {
        const bool record_phase = memory_mapper_phase_profile_enabled();
        const bool record_retired_traffic =
            traffic_epoch != 0U && state.retired;
        if (traffic_epoch != 0U && !record_retired_traffic) {
            if (state.traffic_epoch != traffic_epoch) {
                state.traffic = {};
                state.traffic_epoch = traffic_epoch;
            }
            ++state.traffic.get_calls;
            state.traffic.requested_get_bytes += requested;
        }
        if (!record_phase && !record_retired_traffic)
            return;

        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.profile_mutex);
        if (record_retired_traffic && pool.traffic_active &&
            pool.traffic_epoch == traffic_epoch) {
            ++pool.exited_traffic.get_calls;
            pool.exited_traffic.requested_get_bytes += requested;
        }
        if (!record_phase)
            return;

        auto& phase = pool.phase;
        phase.enabled = true;
        ++phase.get_calls;
        if (pool_hit) {
            ++phase.pool_hits;
        } else {
            ++phase.pool_misses;
            ++phase.system_malloc_calls;
            phase.system_malloc_payload_bytes += capacity;
            phase.capacity_reserved_bytes += capacity;
            phase.header_reserved_bytes += sizeof(BlockHeader);
            ++phase.total_blocks;
            phase.peak_capacity_reserved_bytes = std::max(
                phase.peak_capacity_reserved_bytes,
                phase.capacity_reserved_bytes);
        }
        phase.requested_live_bytes += requested;
        phase.capacity_live_bytes += capacity;
        ++phase.live_blocks;
        phase.peak_requested_live_bytes = std::max(
            phase.peak_requested_live_bytes,
            phase.requested_live_bytes);
        phase.peak_capacity_live_bytes = std::max(
            phase.peak_capacity_live_bytes,
            phase.capacity_live_bytes);
    }

    void MemoryMapper::record_profiled_release(
        ThreadState& state,
        size_t const requested,
        size_t const capacity,
        std::uint64_t const traffic_epoch)
    {
        const bool record_phase = memory_mapper_phase_profile_enabled();
        const bool record_retired_traffic =
            traffic_epoch != 0U && state.retired;
        if (traffic_epoch != 0U && !record_retired_traffic) {
            if (state.traffic_epoch != traffic_epoch) {
                state.traffic = {};
                state.traffic_epoch = traffic_epoch;
            }
            ++state.traffic.release_calls;
            state.traffic.requested_release_bytes += requested;
        }
        if (!record_phase && !record_retired_traffic)
            return;

        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.profile_mutex);
        if (record_retired_traffic && pool.traffic_active &&
            pool.traffic_epoch == traffic_epoch) {
            ++pool.exited_traffic.release_calls;
            pool.exited_traffic.requested_release_bytes += requested;
        }
        if (!record_phase)
            return;

        auto& phase = pool.phase;
        phase.enabled = true;
        ++phase.release_calls;
        phase.requested_live_bytes -= requested;
        phase.capacity_live_bytes -= capacity;
        --phase.live_blocks;
    }

    MemoryMapperPhaseSnapshot MemoryMapper::phase_snapshot()
    {
        if (!memory_mapper_phase_profile_enabled())
            return {};
        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.profile_mutex);
        MemoryMapperPhaseSnapshot snapshot = pool.phase;
        snapshot.enabled = true;
        snapshot.capacity_free_bytes =
            snapshot.capacity_reserved_bytes - snapshot.capacity_live_bytes;
        snapshot.free_blocks = snapshot.total_blocks - snapshot.live_blocks;
        return snapshot;
    }

    void MemoryMapper::begin_jacobian_traffic_profile(bool const enabled)
    {
        if (!enabled)
            return;

        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.profile_mutex);
        if (pool.traffic_active)
            throw std::logic_error(
                "overlapping MemoryMapper Jacobian traffic scopes are unsupported");

        std::uint64_t next_epoch = pool.traffic_epoch + 1U;
        if (next_epoch == 0U)
            next_epoch = 1U;
        pool.traffic_epoch = next_epoch;
        pool.exited_traffic = {};
        pool.traffic_active = true;

        auto control = std::atomic_ref<std::uint64_t>{allocator_control};
        control.store(next_epoch, std::memory_order_release);
    }

    MemoryMapperTrafficSnapshot MemoryMapper::end_jacobian_traffic_profile()
    {
        // Scope owners must first establish a happens-before quiescence (the
        // production assembler is single-threaded per MPI rank; threaded users
        // join or park workers). That lets each shard use plain counters while
        // keeping allocation recording free of per-call locks and RMWs.
        auto control = std::atomic_ref<std::uint64_t>{allocator_control};
        const std::uint64_t active_epoch =
            control.exchange(0U, std::memory_order_acq_rel);

        CentralPool& pool = central_pool();
        std::lock_guard lock(pool.profile_mutex);
        if (!pool.traffic_active || active_epoch != pool.traffic_epoch)
            return {};

        MemoryMapperTrafficSnapshot result = pool.exited_traffic;
        for (const ThreadState* state : pool.thread_states) {
            if (state->traffic_epoch != active_epoch)
                continue;
            result.get_calls += state->traffic.get_calls;
            result.release_calls += state->traffic.release_calls;
            result.requested_get_bytes += state->traffic.requested_get_bytes;
            result.requested_release_bytes +=
                state->traffic.requested_release_bytes;
        }
        pool.traffic_active = false;
        return result;
    }

} // namespace Kadath
