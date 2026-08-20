/*
    Copyright 2019 Ludwig Jens Papenfort

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

//
// Created by sauliac on 14/04/2020, based on a previous work from Ludwig Jens Papenfort.
//

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <map>
#include <queue>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <memory>
#include <algorithm>

#include "For_Kadath/Diagnostics/kernel_profile.hpp"

// only do this if really necessary, e.g. using an Intel compiler not capable of compiling the flat_hash_map below
// this is slower than the hash map
// #define KADATH_VECTORMAP

#ifndef KADATH_VECTORMAP
#include "For_Kadath/Third_Party/flat_hash_map.hpp"
#endif

namespace Kadath
{
    struct MemoryMapperPhaseSnapshot {
        bool enabled = false;
        std::uint64_t requested_live_bytes = 0;
        std::uint64_t capacity_live_bytes = 0;
        std::uint64_t capacity_free_bytes = 0;
        std::uint64_t capacity_reserved_bytes = 0;
        std::uint64_t header_reserved_bytes = 0;
        std::uint64_t peak_requested_live_bytes = 0;
        std::uint64_t peak_capacity_live_bytes = 0;
        std::uint64_t peak_capacity_reserved_bytes = 0;
        std::uint64_t live_blocks = 0;
        std::uint64_t free_blocks = 0;
        std::uint64_t total_blocks = 0;
        std::uint64_t get_calls = 0;
        std::uint64_t release_calls = 0;
        std::uint64_t pool_hits = 0;
        std::uint64_t pool_misses = 0;
        std::uint64_t system_malloc_calls = 0;
        std::uint64_t system_malloc_payload_bytes = 0;
    };

    struct MemoryMapperTrafficSnapshot {
        std::uint64_t get_calls = 0;
        std::uint64_t release_calls = 0;
        std::uint64_t requested_get_bytes = 0;
        std::uint64_t requested_release_bytes = 0;
    };

    /*
      These classes define a thin wrapper around memory allocation calls,
      which are abundant throughout the Kadath library.
      By redirecting these calls and keeping the allocated memory by
      circumventing deallocations, significant runtime can be saved.

      Max-align-sized slabs use an intrusive free list. Common requests use a
      bounded direct size-class table; large requests use a hash map.
    */

    class coef_mem
    {
      private:
        // Kadath's production solvers are one compute thread per MPI rank.
        // Keep these rank-local transform buffers in the single live
        // coef_1d translation unit: thread_local resolution was measurable in
        // the first-J hot path and provided no isolation to the pure-MPI
        // execution model.
        static std::array<std::unique_ptr<double[]>, 2> mem_ptrs;
        static std::array<size_t, 2> lengths;

      public:
        static double* get_mem(size_t const num, size_t const len)
        {
            if (len > lengths[num]) {
                mem_ptrs[num] = std::make_unique<double[]>(len);

                lengths[num] = len;
            }
            return mem_ptrs[num].get();
        }
    };

    class MemoryMapper
    {
      private:
        // Round requests into max-align-sized slabs.  A 64 KiB direct table
        // covers the short-lived Array/Index/Val_domain buffers used by the
        // transform layer without an exact-size hash lookup. Worker tables use
        // about 40--82 KiB of metadata (depending on max alignment) and cap
        // cached slabs at 256 KiB per thread. The process-initial table retains
        // direct slabs for the rank lifetime, as the former process-global
        // table did. Larger cold slabs use a process-wide flat hash map.
        static constexpr size_t slab_alignment = alignof(std::max_align_t);
        static constexpr size_t direct_slab_limit = 64U * 1024U;
        static constexpr size_t direct_slab_count = direct_slab_limit / slab_alignment + 1U;
        static constexpr std::uint8_t direct_cache_min_per_class = 4U;
        static constexpr std::uint8_t direct_cache_max_per_class = 32U;
        static constexpr std::uint8_t direct_transfer_batch = 16U;
        static constexpr size_t direct_cache_class_target_bytes = 128U * 1024U;
        static constexpr size_t direct_cache_total_budget = 256U * 1024U;
        struct alignas(std::max_align_t) BlockHeader {
            BlockHeader* next;
        };
        static_assert(sizeof(BlockHeader) % slab_alignment == 0);

#ifdef KADATH_VECTORMAP
        using large_free_map_t = std::unordered_map<size_t, void*>;
#else
        using large_free_map_t = ska::flat_hash_map<size_t, void*>;
#endif

        struct ThreadState {
            std::array<void*, direct_slab_count> direct_free{};
            std::array<std::uint8_t, direct_slab_count> direct_free_count{};
            std::array<std::uint8_t, direct_slab_count> direct_cache_target{};
            size_t direct_cached_bytes = 0U;
            MemoryMapperTrafficSnapshot traffic{};
            std::uint64_t traffic_epoch = 0U;
            bool initialized = false;
            bool retired = false;
            bool initial_thread = false;
        };
        static_assert(std::is_trivially_destructible_v<ThreadState>);

        struct ThreadStateRetirer {
            ~ThreadStateRetirer() noexcept;
        };

        struct CentralPool;

        // The process-initial thread owns a process-lifetime direct cache. Its
        // one-thread-per-rank hot path therefore needs only an identity compare
        // and an intrusive-list operation. Other threads use bounded TLS
        // caches; their separate retire guards are touched only on first use
        // and flush before the POD TLS state reaches the end of its lifetime.
        static ThreadState initial_thread_state;
        static thread_local ThreadState thread_state;
        static thread_local ThreadStateRetirer thread_state_retirer;
        static const void* const initial_thread_identity;
        alignas(std::atomic_ref<std::uint64_t>::required_alignment)
            static std::uint64_t allocator_control;
        static_assert(std::atomic_ref<std::uint64_t>::is_always_lock_free);
        static const bool initial_thread_state_registered;

        [[nodiscard]] static const void* current_thread_identity() noexcept
        {
#if defined(__has_builtin)
#  if defined(__APPLE__) && defined(__aarch64__) && \
      __has_builtin(__builtin_arm_rsr64)
            // On Darwin arm64 __builtin_thread_pointer reads TPIDR_EL0,
            // whose value is neither stable nor thread-unique. TPIDRRO_EL0
            // contains the stable pthread pointer used by the platform ABI.
            return reinterpret_cast<const void*>(
                __builtin_arm_rsr64("TPIDRRO_EL0"));
#  elif !defined(__APPLE__) && __has_builtin(__builtin_thread_pointer)
            return __builtin_thread_pointer();
#  else
            return static_cast<const void*>(&thread_state);
#  endif
#else
            return static_cast<const void*>(&thread_state);
#endif
        }

        [[nodiscard]] static size_t slab_size(size_t const sz)
        {
            constexpr size_t mask = slab_alignment - 1U;
            static_assert((slab_alignment & mask) == 0U);
            if (sz > std::numeric_limits<size_t>::max() - mask)
                throw std::bad_alloc();
            return (sz + mask) & ~mask;
        }

        [[nodiscard]] static void* pop_free(void*& head)
        {
            void* result = head;
            if (result != nullptr)
                std::memcpy(&head, result, sizeof(head));
            return result;
        }

        static void push_free(void*& head, void* raw_mem_ptr)
        {
            std::memcpy(raw_mem_ptr, &head, sizeof(head));
            head = raw_mem_ptr;
        }

        [[nodiscard]] static CentralPool& central_pool();
        static void initialize_thread_state(ThreadState& state);
        static void retire_thread_state(ThreadState& state) noexcept;
        [[nodiscard]] static void* acquire_central_slab(ThreadState* state,
                                                        size_t capacity,
                                                        bool& pool_hit);
        static void release_direct_overflow(ThreadState& state,
                                            void* raw_mem_ptr,
                                            size_t capacity);
        static void release_direct_central(void* raw_mem_ptr, size_t capacity);
        static void release_large_slab(void* raw_mem_ptr, size_t capacity);
        static void flush_thread_state(ThreadState& state) noexcept;
        static void record_profiled_get(ThreadState& state,
                                        size_t requested,
                                        size_t capacity,
                                        bool pool_hit,
                                        std::uint64_t traffic_epoch);
        static void record_profiled_release(ThreadState& state,
                                            size_t requested,
                                            size_t capacity,
                                            std::uint64_t traffic_epoch);

        [[nodiscard]] static std::uint8_t direct_cache_maximum(
            size_t const capacity)
        {
            const size_t budget_limit =
                direct_cache_class_target_bytes / capacity;
            return static_cast<std::uint8_t>(std::clamp(
                budget_limit,
                static_cast<size_t>(direct_cache_min_per_class),
                static_cast<size_t>(direct_cache_max_per_class)));
        }

        [[nodiscard]] static std::uint8_t direct_cache_limit(
            ThreadState& state, size_t const index)
        {
            std::uint8_t& target = state.direct_cache_target[index];
            if (target == 0U)
                target = direct_cache_min_per_class;
            return target;
        }

      public:
        [[nodiscard]] static void* get_memory(size_t const sz)
        {
            if (sz == 0)
                return nullptr;

            const size_t capacity = slab_size(sz);
            const std::uint64_t control =
                std::atomic_ref<std::uint64_t>{allocator_control}.load(
                    std::memory_order_relaxed);
            const bool initial_thread =
                current_thread_identity() == initial_thread_identity;
            ThreadState* state = &initial_thread_state;
            if (!initial_thread) {
                state = &thread_state;
                if (!state->initialized)
                    initialize_thread_state(*state);
            }

            void* raw_mem_ptr = nullptr;
            if (capacity <= direct_slab_limit) {
                const size_t index = capacity / slab_alignment;
                if (initial_thread) {
                    raw_mem_ptr = pop_free(state->direct_free[index]);
                } else if (!state->retired) {
                    raw_mem_ptr = pop_free(state->direct_free[index]);
                    if (raw_mem_ptr != nullptr) {
                        --state->direct_free_count[index];
                        state->direct_cached_bytes -= capacity;
                    }
                }
            }
            const bool pool_hit = raw_mem_ptr != nullptr;
            bool effective_pool_hit = pool_hit;
            if (!pool_hit)
                raw_mem_ptr = acquire_central_slab(
                    initial_thread ? nullptr : state, capacity,
                    effective_pool_hit);

            // Counter-only probes: the kernel profile records calling-thread
            // traffic in its active FirstJ / ExactJv context, not cross-thread
            // ownership. The phase profile maintains process-wide live and
            // reserved state. The combined gate keeps record-process calls off
            // the default path (the old unconditional hook was otherwise a top
            // first-J leaf).
            const std::uint64_t traffic_epoch = control;
            if (memory_mapper_any_profile_enabled() || traffic_epoch != 0U) {
                if (memory_mapper_profile_enabled())
                    memory_mapper_record_get(sz, effective_pool_hit);
                record_profiled_get(*state, sz, capacity, effective_pool_hit,
                                    traffic_epoch);
            }

            return raw_mem_ptr;
        }

        template <typename T> [[nodiscard]] static T* get_memory(size_t const sz)
        {
            return static_cast<T*>(get_memory(sz * sizeof(T)));
        }

        static void release_memory(void* raw_mem_ptr, size_t const sz)
        {
            if (raw_mem_ptr == nullptr)
                return;

            const size_t capacity = slab_size(sz);
            const std::uint64_t control =
                std::atomic_ref<std::uint64_t>{allocator_control}.load(
                    std::memory_order_relaxed);
            const bool initial_thread =
                current_thread_identity() == initial_thread_identity;
            ThreadState* state = &initial_thread_state;
            if (!initial_thread) {
                state = &thread_state;
                if (!state->initialized)
                    initialize_thread_state(*state);
            }

            if (capacity <= direct_slab_limit) {
                const size_t index = capacity / slab_alignment;
                if (initial_thread) {
                    push_free(state->direct_free[index], raw_mem_ptr);
                } else if (!state->retired &&
                           state->direct_free_count[index] <
                               direct_cache_limit(*state, index) &&
                           state->direct_cached_bytes <=
                               direct_cache_total_budget - capacity) {
                    push_free(state->direct_free[index], raw_mem_ptr);
                    ++state->direct_free_count[index];
                    state->direct_cached_bytes += capacity;
                } else if (!state->retired) {
                    release_direct_overflow(*state, raw_mem_ptr, capacity);
                } else {
                    release_direct_central(raw_mem_ptr, capacity);
                }
            } else {
                release_large_slab(raw_mem_ptr, capacity);
            }

            // Paired counter probes, gated at the call site; see get_memory.
            const std::uint64_t traffic_epoch = control;
            if (memory_mapper_any_profile_enabled() || traffic_epoch != 0U) {
                if (memory_mapper_profile_enabled())
                    memory_mapper_record_release(sz);
                record_profiled_release(*state, sz, capacity, traffic_epoch);
            }
        }

        template <typename T> static void release_memory(void* raw_mem_ptr, size_t const sz)
        {
            release_memory(raw_mem_ptr, sz * sizeof(T));
        }

        [[nodiscard]] static bool phase_profile_enabled()
        {
            return memory_mapper_phase_profile_enabled();
        }

        [[nodiscard]] static MemoryMapperPhaseSnapshot phase_snapshot();

        // Only one process-wide traffic scope may be active. Callers establish
        // a happens-before boundary before beginning and must join or park any
        // worker threads that recorded traffic before ending it.
        static void begin_jacobian_traffic_profile(bool enabled);

        [[nodiscard]] static MemoryMapperTrafficSnapshot end_jacobian_traffic_profile();
    };

    class MemoryMapperJacobianTrafficScope
    {
      public:
        explicit MemoryMapperJacobianTrafficScope(bool const enabled)
            : active_{enabled}
        {
            MemoryMapper::begin_jacobian_traffic_profile(enabled);
        }

        ~MemoryMapperJacobianTrafficScope()
        {
            if (active_)
                static_cast<void>(MemoryMapper::end_jacobian_traffic_profile());
        }

        MemoryMapperJacobianTrafficScope(const MemoryMapperJacobianTrafficScope&) = delete;
        MemoryMapperJacobianTrafficScope& operator=(const MemoryMapperJacobianTrafficScope&) = delete;

        [[nodiscard]] MemoryMapperTrafficSnapshot finish()
        {
            if (!active_)
                return {};
            active_ = false;
            return MemoryMapper::end_jacobian_traffic_profile();
        }

      private:
        bool active_ = false;
    };

    struct MemoryMappable {
        void* operator new(size_t sz) { return MemoryMapper::get_memory(sz); }

        void operator delete(void* mem_ptr, size_t const sz) { MemoryMapper::release_memory(mem_ptr, sz); }

        void* operator new[](size_t sz) { return MemoryMapper::get_memory(sz); }

        void operator delete[](void* mem_ptr, size_t const sz) { MemoryMapper::release_memory(mem_ptr, sz); }
    };

} // namespace Kadath

// Forward declarations of platform-specific allocator-pressure syscalls.
// Pulled out of the inline body because `extern "C"` is only valid at
// namespace scope. The functions live in libSystem (macOS) or libc
// (glibc); the standard headers (<malloc/malloc.h>, <malloc.h>) are
// avoided to keep this header's include surface minimal.
#if defined(__APPLE__)
extern "C" size_t malloc_zone_pressure_relief(void* zone, size_t goal);
#elif defined(__linux__)
extern "C" int malloc_trim(size_t pad);
#endif

namespace Kadath
{
    // Ask the platform allocator to return its free-list pages to the OS.
    // libmalloc on macOS and glibc on Linux both hold a per-thread heap
    // free-list that does NOT shrink on free() for sub-mmap-threshold
    // blocks. Calling this at known phase boundaries (post-cache-flush,
    // post-MUMPS-factor) lets the OS reclaim those pages so process RSS
    // drops between phases.
    //
    // Default off; enable via RELEASE_ALLOCATOR_PAGES=1. The
    // platform syscall costs are non-trivial (10-100 ms typical) so it
    // is gated to deliberate phase boundaries rather than fired
    // continuously.
    inline void release_allocator_pages()
    {
        static const bool enabled = [] {
            const char* env = std::getenv("RELEASE_ALLOCATOR_PAGES");
            return env != nullptr && env[0] != '\0' && env[0] != '0';
        }();
        if (!enabled)
            return;
#if defined(__APPLE__)
        malloc_zone_pressure_relief(nullptr, 0);
#elif defined(__linux__)
        malloc_trim(0);
#endif
    }
} // namespace Kadath
