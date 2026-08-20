#pragma once

// Lightweight per-kernel timing probe for the J-build / exact-Jv hot path.
//
// Two contexts (FirstJ / ExactJv) toggled by RAII scope. Inside each timed
// kernel, the RAII KernelTimer accumulates wall into the active context's
// per-kernel slot. Default-off; the only runtime cost when disabled is one
// pointer compare per kernel entry.
//
// Env gate: KERNEL_PROFILE=1. Caller drives report at convenient
// boundaries (e.g. end of each Newton step) via report_and_reset().

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <map>
#include <string>

namespace Kadath
{
    enum class KernelContext { None, FirstJ, ExactJv };
    enum class KernelId {
        ComputeDerAbs = 0,
        ComputeDerVar = 1,
        EquationApply = 2,
        Count
    };

    inline const char* kernel_name(KernelId id)
    {
        switch (id) {
            case KernelId::ComputeDerAbs: return "Val_domain::compute_der_abs";
            case KernelId::ComputeDerVar: return "Val_domain::compute_der_var";
            case KernelId::EquationApply: return "Equation::apply";
            case KernelId::Count: break;
        }
        return "unknown";
    }

    // Namespace-scope cache rather than a function-local static. The latter's
    // thread-safe-init guard encouraged Clang to merge the enabled check with
    // the following TLS-state check, resolving the TLV even when profiling was
    // off. This form matches the MemoryMapper probe below: environment state is
    // captured once during process startup and hot call sites see a plain load.
    inline const bool kernel_profile_enabled_flag = []() {
        const char* value = std::getenv("KERNEL_PROFILE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();

    inline bool kernel_profile_enabled()
    {
        return kernel_profile_enabled_flag;
    }

    struct KernelProfileSlot {
        double seconds = 0.0;
        long long calls = 0;
    };

    using KernelProfileContextSlots =
        std::array<KernelProfileSlot, static_cast<std::size_t>(KernelId::Count)>;

    struct KernelProfileState {
        KernelProfileContextSlots first_j{};
        KernelProfileContextSlots exact_jv{};
        KernelContext active = KernelContext::None;
        // Apply-scoped attribution: apply_depth > 0 while inside
        // Equation::apply. The *_in_apply slots accumulate der_abs / der_var
        // wall that occurs nested under apply, answering whether apply's total
        // wall is actually spectral transform vs its own operator/copy work.
        int apply_depth = 0;
        KernelProfileContextSlots first_j_in_apply{};
        KernelProfileContextSlots exact_jv_in_apply{};
    };

    inline KernelProfileState& kernel_profile_state()
    {
        thread_local KernelProfileState state;
        return state;
    }

    // RAII scope that arms the active context for the lifetime of the scope.
    // Nested scopes restore the parent context on destruction.
    class KernelProfileScope
    {
      public:
        explicit KernelProfileScope(KernelContext context)
        {
            if (!kernel_profile_enabled())
                return;
            // Compiler-only barrier: keep the following TLS lookup on the
            // enabled side of the branch (emits no machine instruction).
            std::atomic_signal_fence(std::memory_order_acquire);
            state_ = &kernel_profile_state();
            previous_ = state_->active;
            state_->active = context;
        }
        ~KernelProfileScope()
        {
            if (state_ != nullptr)
                state_->active = previous_;
        }
        KernelProfileScope(const KernelProfileScope&) = delete;
        KernelProfileScope& operator=(const KernelProfileScope&) = delete;

      private:
        KernelProfileState* state_ = nullptr;
        KernelContext previous_ = KernelContext::None;
    };

    // RAII timer for a single kernel invocation. The disabled path is one
    // global-flag branch on construction plus the inactive check on destruction.
    class KernelTimer
    {
      public:
        explicit KernelTimer(KernelId id) : id_(id)
        {
            if (!kernel_profile_enabled())
                return;
            std::atomic_signal_fence(std::memory_order_acquire);
            KernelProfileState& state = kernel_profile_state();
            const KernelContext context = state.active;
            if (context == KernelContext::None)
                return;
            state_ = &state;
            start_ = std::chrono::steady_clock::now();
        }
        ~KernelTimer()
        {
            if (state_ == nullptr)
                return;
            const auto delta = std::chrono::steady_clock::now() - start_;
            const double seconds = std::chrono::duration<double>(delta).count();
            KernelProfileState& state = *state_;
            const KernelContext context = state.active;
            KernelProfileContextSlots& slots =
                (context == KernelContext::FirstJ) ? state.first_j : state.exact_jv;
            KernelProfileSlot& slot = slots[static_cast<std::size_t>(id_)];
            slot.seconds += seconds;
            ++slot.calls;
            // Attribute der_abs / der_var wall that fired nested under apply.
            if (state.apply_depth > 0 && id_ != KernelId::EquationApply) {
                KernelProfileContextSlots& aslots = (context == KernelContext::FirstJ)
                    ? state.first_j_in_apply : state.exact_jv_in_apply;
                KernelProfileSlot& aslot = aslots[static_cast<std::size_t>(id_)];
                aslot.seconds += seconds;
                ++aslot.calls;
            }
        }
        KernelTimer(const KernelTimer&) = delete;
        KernelTimer& operator=(const KernelTimer&) = delete;

      private:
        KernelId id_;
        KernelProfileState* state_ = nullptr;
        std::chrono::steady_clock::time_point start_{};
    };

    // --- MemoryMapper profile (phase 1: counters only, no per-call timing) ---
    //
    // Hooks called from MemoryMapper::get_memory / release_memory in
    // memory.hpp. Counter increments are gated by MEMORY_MAPPER_PROFILE=1
    // AND the active KernelContext != None, so allocator traffic is recorded
    // only while we are inside an instrumented kernel (FirstJ or ExactJv).
    // Phase 1 deliberately omits per-call timing to avoid perturbing the
    // measurement; the per-Newton-step report prints call counts, byte volume,
    // pool hit/miss split, peak active buffer count, and a size histogram.

    struct MemoryMapperStats {
        long long get_calls = 0;
        long long release_calls = 0;
        long long bytes_allocated = 0;
        long long bytes_released = 0;
        long long pool_hits = 0;
        long long pool_misses = 0;
        long long peak_active_count = 0;
        long long current_active_count = 0;
        std::map<std::size_t, long long> get_size_histogram;
    };

    struct MemoryMapperContextStats {
        MemoryMapperStats first_j;
        MemoryMapperStats exact_jv;
    };

    inline MemoryMapperContextStats& memory_mapper_stats()
    {
        thread_local MemoryMapperContextStats stats;
        return stats;
    }

    // Cached as a namespace-scope inline variable (C++17) rather than a
    // function-local static: this predicate is tested on every MemoryMapper
    // alloc/free, and a function-local static carries a thread-safe-init guard
    // check on every call that was itself a hot leaf in the first-J profile.
    // A global inline const reads as a plain load the compiler can fold into
    // the gated call site.
    inline const bool memory_mapper_profile_enabled_flag = []() {
        const char* value = std::getenv("MEMORY_MAPPER_PROFILE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();

    // Process-wide allocator-state accounting used by solver phase snapshots.
    // Keep this separate from the kernel-context profile above: phase accounting
    // follows every MemoryMapper allocation/release, including traffic outside
    // FirstJ and ExactJv. The combined predicate preserves one disabled-path
    // branch at the inlined allocator call sites.
    inline const bool memory_mapper_phase_profile_enabled_flag = []() {
        const char* value = std::getenv("MEMORY_MAPPER_PHASE_PROFILE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();

    inline const bool memory_mapper_any_profile_enabled_flag =
        memory_mapper_profile_enabled_flag ||
        memory_mapper_phase_profile_enabled_flag;

    inline bool memory_mapper_profile_enabled()
    {
        return memory_mapper_profile_enabled_flag;
    }

    inline bool memory_mapper_phase_profile_enabled()
    {
        return memory_mapper_phase_profile_enabled_flag;
    }

    inline bool memory_mapper_any_profile_enabled()
    {
        return memory_mapper_any_profile_enabled_flag;
    }

    inline void memory_mapper_record_get(std::size_t sz, bool pool_hit)
    {
        if (!memory_mapper_profile_enabled())
            return;
        const KernelContext context = kernel_profile_state().active;
        if (context == KernelContext::None)
            return;
        MemoryMapperStats& stats = (context == KernelContext::FirstJ)
            ? memory_mapper_stats().first_j
            : memory_mapper_stats().exact_jv;
        ++stats.get_calls;
        stats.bytes_allocated += static_cast<long long>(sz);
        if (pool_hit)
            ++stats.pool_hits;
        else
            ++stats.pool_misses;
        ++stats.current_active_count;
        if (stats.current_active_count > stats.peak_active_count)
            stats.peak_active_count = stats.current_active_count;
        ++stats.get_size_histogram[sz];
    }

    inline void memory_mapper_record_release(std::size_t sz)
    {
        if (!memory_mapper_profile_enabled())
            return;
        const KernelContext context = kernel_profile_state().active;
        if (context == KernelContext::None)
            return;
        MemoryMapperStats& stats = (context == KernelContext::FirstJ)
            ? memory_mapper_stats().first_j
            : memory_mapper_stats().exact_jv;
        ++stats.release_calls;
        stats.bytes_released += static_cast<long long>(sz);
        --stats.current_active_count;
    }

    // --- Array<T> copy-traffic probe (phase 1: counters only) ---
    //
    // Hooks called from Array<T>::Array(const Array<T>&) and
    // Array<T>::operator=(const Array<T>&) in array.inl. Counter
    // increments are gated by ARRAY_COPY_PROFILE=1 AND the active
    // KernelContext != None, so deep-copy traffic is recorded only inside
    // the instrumented FirstJ / ExactJv scopes. Goal: decide whether
    // Array<T> move-semantics work has a real target before committing
    // ~1-2 days of ownership refactor + four-bar-style validation.

    struct ArrayCopyStats {
        long long copy_ctor_calls = 0;
        long long copy_assign_calls = 0;
        long long bytes_copied = 0;  // sum across ctor + assign
    };

    struct ArrayCopyContextStats {
        ArrayCopyStats first_j;
        ArrayCopyStats exact_jv;
    };

    inline ArrayCopyContextStats& array_copy_stats()
    {
        thread_local ArrayCopyContextStats stats;
        return stats;
    }

    inline const bool array_copy_profile_enabled_flag = []() {
        const char* value = std::getenv("ARRAY_COPY_PROFILE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();

    inline bool array_copy_profile_enabled()
    {
        return array_copy_profile_enabled_flag;
    }

    // Apply-scoped copy buckets: count only the Array copies occurring while
    // inside Equation::apply (apply_depth > 0), so the FirstJ-wide copy traffic
    // can be split into apply vs def/export.
    inline ArrayCopyContextStats& array_copy_apply_stats()
    {
        thread_local ArrayCopyContextStats stats;
        return stats;
    }

    inline bool any_apply_profile_enabled()
    {
        return kernel_profile_enabled() || array_copy_profile_enabled()
               || memory_mapper_profile_enabled();
    }

    // RAII marker armed for the lifetime of one Equation::apply call. A plain
    // depth counter (handles any nesting); cost when all profiles disabled is
    // one static-bool-backed check on enter/exit.
    class ApplyScope
    {
      public:
        ApplyScope()
        {
            if (!any_apply_profile_enabled())
                return;
            std::atomic_signal_fence(std::memory_order_acquire);
            state_ = &kernel_profile_state();
            ++state_->apply_depth;
        }
        ~ApplyScope()
        {
            if (state_ != nullptr)
                --state_->apply_depth;
        }
        ApplyScope(const ApplyScope&) = delete;
        ApplyScope& operator=(const ApplyScope&) = delete;

      private:
        KernelProfileState* state_ = nullptr;
    };

    inline void array_copy_record_ctor(std::size_t bytes)
    {
        if (!array_copy_profile_enabled())
            return;
        const KernelContext context = kernel_profile_state().active;
        if (context == KernelContext::None)
            return;
        ArrayCopyStats& stats = (context == KernelContext::FirstJ)
            ? array_copy_stats().first_j
            : array_copy_stats().exact_jv;
        ++stats.copy_ctor_calls;
        stats.bytes_copied += static_cast<long long>(bytes);
        if (kernel_profile_state().apply_depth > 0) {
            ArrayCopyStats& as = (context == KernelContext::FirstJ)
                ? array_copy_apply_stats().first_j
                : array_copy_apply_stats().exact_jv;
            ++as.copy_ctor_calls;
            as.bytes_copied += static_cast<long long>(bytes);
        }
    }

    inline void array_copy_record_assign(std::size_t bytes)
    {
        if (!array_copy_profile_enabled())
            return;
        const KernelContext context = kernel_profile_state().active;
        if (context == KernelContext::None)
            return;
        ArrayCopyStats& stats = (context == KernelContext::FirstJ)
            ? array_copy_stats().first_j
            : array_copy_stats().exact_jv;
        ++stats.copy_assign_calls;
        stats.bytes_copied += static_cast<long long>(bytes);
        if (kernel_profile_state().apply_depth > 0) {
            ArrayCopyStats& as = (context == KernelContext::FirstJ)
                ? array_copy_apply_stats().first_j
                : array_copy_apply_stats().exact_jv;
            ++as.copy_assign_calls;
            as.bytes_copied += static_cast<long long>(bytes);
        }
    }

    // Dump per-context, per-kernel cumulative wall + call counts to stdout
    // on the given rank. Resets the slots after printing so the next caller
    // observes only deltas since this call. Caller decides cadence (end of
    // each Newton step, end of run, etc).
    inline void kernel_profile_report_and_reset(int rank, int step_index)
    {
        if (!kernel_profile_enabled() || rank != 0)
            return;
        KernelProfileState& state = kernel_profile_state();
        auto dump = [&](const char* label, KernelProfileContextSlots& slots) {
            double total = 0.0;
            long long total_calls = 0;
            for (const KernelProfileSlot& slot : slots) {
                total += slot.seconds;
                total_calls += slot.calls;
            }
            std::cout << "KernelProfile[" << label << "] step=" << step_index
                      << " cumulative_wall=" << total << "s"
                      << " cumulative_calls=" << total_calls;
            for (std::size_t k = 0; k < slots.size(); ++k) {
                const KernelProfileSlot& slot = slots[k];
                const double pct = (total > 0.0) ? (100.0 * slot.seconds / total) : 0.0;
                std::cout << " " << kernel_name(static_cast<KernelId>(k))
                          << "=" << slot.seconds << "s(" << pct << "%)/" << slot.calls;
            }
            std::cout << '\n';
            for (KernelProfileSlot& slot : slots) {
                slot.seconds = 0.0;
                slot.calls = 0;
            }
        };
        dump("FirstJ", state.first_j);
        dump("ExactJv", state.exact_jv);

        if (memory_mapper_profile_enabled()) {
            MemoryMapperContextStats& mstate = memory_mapper_stats();
            auto dump_mem = [&](const char* label, MemoryMapperStats& m) {
                const long long total_gets = m.get_calls;
                const double hit_pct = (total_gets > 0)
                    ? (100.0 * static_cast<double>(m.pool_hits) /
                       static_cast<double>(total_gets))
                    : 0.0;
                std::cout << "MemoryMapperProfile[" << label
                          << "] step=" << step_index
                          << " get_calls=" << m.get_calls
                          << " release_calls=" << m.release_calls
                          << " bytes_alloc=" << m.bytes_allocated
                          << " bytes_release=" << m.bytes_released
                          << " pool_hits=" << m.pool_hits
                          << " pool_misses=" << m.pool_misses
                          << " hit_pct=" << hit_pct
                          << " peak_active=" << m.peak_active_count
                          << '\n';
                if (!m.get_size_histogram.empty()) {
                    std::cout << "MemoryMapperProfile[" << label
                              << "][histogram] step=" << step_index << " (size_bytes:count)";
                    for (const auto& kv : m.get_size_histogram) {
                        std::cout << " " << kv.first << ":" << kv.second;
                    }
                    std::cout << '\n';
                }
                m = MemoryMapperStats{};
            };
            dump_mem("FirstJ", mstate.first_j);
            dump_mem("ExactJv", mstate.exact_jv);
        }

        if (array_copy_profile_enabled()) {
            ArrayCopyContextStats& cstate = array_copy_stats();
            auto dump_copy = [&](const char* label, ArrayCopyStats& c) {
                std::cout << "ArrayCopyProfile[" << label
                          << "] step=" << step_index
                          << " copy_ctor=" << c.copy_ctor_calls
                          << " copy_assign=" << c.copy_assign_calls
                          << " bytes_copied=" << c.bytes_copied
                          << '\n';
                c = ArrayCopyStats{};
            };
            dump_copy("FirstJ", cstate.first_j);
            dump_copy("ExactJv", cstate.exact_jv);
        }

        // Apply-scoped attribution: how much of Equation::apply's wall is
        // spectral transform (der_abs / der_var fired nested under apply), plus
        // the Array-copy traffic occurring inside apply. Lets apply's total
        // wall be split into transform vs own-work + copy.
        auto dump_apply_nested = [&](const char* label, KernelProfileContextSlots& aslots) {
            const KernelProfileSlot& da = aslots[static_cast<std::size_t>(KernelId::ComputeDerAbs)];
            const KernelProfileSlot& dv = aslots[static_cast<std::size_t>(KernelId::ComputeDerVar)];
            std::cout << "ApplyNested[" << label << "] step=" << step_index
                      << " der_abs_in_apply=" << da.seconds << "s/" << da.calls
                      << " der_var_in_apply=" << dv.seconds << "s/" << dv.calls << '\n';
            for (KernelProfileSlot& slot : aslots) {
                slot.seconds = 0.0;
                slot.calls = 0;
            }
        };
        dump_apply_nested("FirstJ", state.first_j_in_apply);
        dump_apply_nested("ExactJv", state.exact_jv_in_apply);

        if (array_copy_profile_enabled()) {
            ArrayCopyContextStats& astate = array_copy_apply_stats();
            auto dump_apply_copy = [&](const char* label, ArrayCopyStats& c) {
                std::cout << "ApplyCopy[" << label << "] step=" << step_index
                          << " copy_ctor=" << c.copy_ctor_calls
                          << " copy_assign=" << c.copy_assign_calls
                          << " bytes_copied=" << c.bytes_copied << '\n';
                c = ArrayCopyStats{};
            };
            dump_apply_copy("FirstJ", astate.first_j);
            dump_apply_copy("ExactJv", astate.exact_jv);
        }
        std::cout.flush();
    }
} // namespace Kadath
