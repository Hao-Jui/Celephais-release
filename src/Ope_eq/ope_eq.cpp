/*
    Copyright 2017 Philippe Grandclement

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

#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#if defined(__GNUG__)
#include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace Kadath
{
    namespace
    {
        struct OpeActionOperandScratch
        {
            std::deque<std::optional<Term_eq>> slots;
            std::size_t next_slot = 0;
            const System_of_eqs* owner = nullptr;
        };

        OpeActionOperandScratch& ope_action_operand_scratch()
        {
            static thread_local OpeActionOperandScratch scratch;
            return scratch;
        }

        struct OpeActionProfileRecord {
            long long calls = 0;
            double inclusive_seconds = 0.0;
            double exclusive_seconds = 0.0;
        };

        struct OpeActionProfileState {
            bool enabled = false;
            bool initialized = false;
            int limit = 30;
            std::unordered_map<const char*, OpeActionProfileRecord> records;
            std::vector<ScopedOpeActionProfile*> stack;

            void init_if_needed()
            {
                if (initialized)
                    return;
                const char* env = std::getenv("OPE_ACTION_PROFILE");
                enabled = (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0);
                initialized = true;
            }
        };

        OpeActionProfileState& ope_action_profile_state()
        {
            static thread_local OpeActionProfileState state;
            return state;
        }

        std::string demangle_ope_type(const char* name)
        {
#if defined(__GNUG__)
            int status = 0;
            std::unique_ptr<char, void (*)(void*)> demangled(abi::__cxa_demangle(name, nullptr, nullptr, &status),
                                                             std::free);
            if (status == 0 && demangled) {
                std::string text(demangled.get());
                const std::string prefix = "Kadath::";
                if (text.compare(0, prefix.size(), prefix) == 0)
                    text.erase(0, prefix.size());
                return text;
            }
#endif
            return name;
        }
    }

    ope_action_detail::OperandScratchLease::OperandScratchLease(const System_of_eqs* owner)
        : slot_(0)
    {
        auto& scratch = ope_action_operand_scratch();
        if (scratch.next_slot != 0)
            assert(scratch.owner == owner);
        if (scratch.next_slot == 0 && scratch.owner != owner) {
            for (auto& slot : scratch.slots)
                slot.reset();
            scratch.owner = owner;
        }
        slot_ = scratch.next_slot++;
        auto& slots = scratch.slots;
        if (slot_ == slots.size())
            slots.emplace_back();
    }

    ope_action_detail::OperandScratchLease::~OperandScratchLease()
    {
        auto& scratch = ope_action_operand_scratch();
        assert(scratch.next_slot == slot_ + 1);
        scratch.next_slot = slot_;
        // Retain the O(depth) slot structure across evaluations, but return
        // Tensor/Val_domain storage to their pools at each outer evaluation
        // edge. Keeping every depth slot live for the System lifetime enlarges
        // the active working set and strands pooled arrays between equations.
        if (scratch.next_slot == 0) {
            for (auto& slot : scratch.slots)
                slot.reset();
            scratch.owner = nullptr;
        }
    }

    std::optional<Term_eq>& ope_action_detail::OperandScratchLease::storage()
    {
        return ope_action_operand_scratch().slots[slot_];
    }

    void ope_action_detail::release_operand_scratch(const System_of_eqs* owner)
    {
        auto& scratch = ope_action_operand_scratch();
        if (scratch.owner != owner)
            return;
        assert(scratch.next_slot == 0);
        for (auto& slot : scratch.slots)
            slot.reset();
        scratch.owner = nullptr;
    }

    ScopedOpeActionProfile::ScopedOpeActionProfile(const Ope_eq& op)
        : active_(false), type_name_(nullptr), child_seconds_(0.0), start_()
    {
        auto& state = ope_action_profile_state();
        state.init_if_needed();
        if (!state.enabled)
            return;
        active_ = true;
        type_name_ = typeid(op).name();
        start_ = std::chrono::steady_clock::now();
        state.stack.push_back(this);
    }

    ScopedOpeActionProfile::~ScopedOpeActionProfile()
    {
        if (!active_)
            return;

        auto& state = ope_action_profile_state();
        const auto end = std::chrono::steady_clock::now();
        const double inclusive = std::chrono::duration<double>(end - start_).count();
        const double exclusive = std::max(0.0, inclusive - child_seconds_);

        if (!state.stack.empty() && state.stack.back() == this) {
            state.stack.pop_back();
        } else {
            const auto it = std::find(state.stack.begin(), state.stack.end(), this);
            if (it != state.stack.end())
                state.stack.erase(it);
        }

        auto& record = state.records[type_name_];
        record.calls += 1;
        record.inclusive_seconds += inclusive;
        record.exclusive_seconds += exclusive;

        if (!state.stack.empty())
            state.stack.back()->add_child_seconds(inclusive);
    }

    void ScopedOpeActionProfile::add_child_seconds(double seconds)
    {
        child_seconds_ += seconds;
    }

    extern "C" void dump_ope_action_profile()
    {
        auto& state = ope_action_profile_state();
        state.init_if_needed();
        if (!state.enabled)
            return;
        if (state.records.empty()) {
            std::fprintf(stderr, "[OPE_ACTION_PROFILE] no calls recorded\n");
            return;
        }

        struct Entry {
            const char* type_name = nullptr;
            OpeActionProfileRecord record;
        };
        std::vector<Entry> entries;
        entries.reserve(state.records.size());
        for (const auto& item : state.records)
            entries.push_back({item.first, item.second});

        std::sort(entries.begin(), entries.end(), [](const Entry& lhs, const Entry& rhs) {
            return lhs.record.exclusive_seconds > rhs.record.exclusive_seconds;
        });

        double total_exclusive = 0.0;
        double total_inclusive = 0.0;
        long long total_calls = 0;
        for (const auto& entry : entries) {
            total_exclusive += entry.record.exclusive_seconds;
            total_inclusive += entry.record.inclusive_seconds;
            total_calls += entry.record.calls;
        }

        const int limit = std::min<int>(state.limit, static_cast<int>(entries.size()));
        std::fprintf(stderr,
                     "[OPE_ACTION_PROFILE] operator_count=%zu calls=%lld "
                     "exclusive=%.6fs inclusive=%.6fs top=%d\n",
                     entries.size(), total_calls, total_exclusive, total_inclusive, limit);
        std::fprintf(stderr,
                     "[OPE_ACTION_PROFILE] rank operator calls exclusive_s inclusive_s "
                     "avg_exclusive_us avg_inclusive_us exclusive_frac\n");
        for (int i = 0; i < limit; ++i) {
            const auto& entry = entries[static_cast<std::size_t>(i)];
            const double calls = static_cast<double>(entry.record.calls);
            const double exclusive_fraction =
                (total_exclusive > 0.0) ? entry.record.exclusive_seconds / total_exclusive : 0.0;
            std::fprintf(stderr,
                         "[OPE_ACTION_PROFILE] %d %s %lld %.6f %.6f %.3f %.3f %.6f\n",
                         i + 1, demangle_ope_type(entry.type_name).c_str(), entry.record.calls,
                         entry.record.exclusive_seconds, entry.record.inclusive_seconds,
                         1e6 * entry.record.exclusive_seconds / calls,
                         1e6 * entry.record.inclusive_seconds / calls, exclusive_fraction);
        }
    }

    extern "C" void reset_ope_action_profile()
    {
        auto& state = ope_action_profile_state();
        state.records.clear();
        state.stack.clear();
    }

    Ope_eq::Ope_eq(const System_of_eqs* zesys, int dd, int nn)
        : syst(zesys), dom(dd), n_ope(nn), borrowed_action_result_(nullptr)
    {
        parts.resize(static_cast<std::size_t>(n_ope));
    }

    Ope_eq::Ope_eq(const System_of_eqs* zesys, int dd)
        : syst(zesys), dom(dd), borrowed_action_result_(nullptr)
    {
        n_ope = 0;
    }

    Ope_eq::~Ope_eq() = default;

    bool ope_action_write_into_enabled()
    {
        static const bool enabled = env_flag_enabled("OPE_ACTION_WRITE_INTO", true);
        return enabled;
    }

    bool Ope_eq::action_into(Term_eq& result) const
    {
        Term_eq computed(action());
        if (result.try_write_action_result(computed))
            return true;
        result = computed;
        return false;
    }

    void Ope_eq::action_into_scratch(std::optional<Term_eq>& result) const
    {
        if (!result.has_value()) {
            result.emplace(action());
            return;
        }
        Term_eq computed(action());
        if (!result->try_write_action_result(computed)) {
            // Scratch has no externally visible layout. Rebuild a depth slot
            // when a different-shaped subtree visits it.
            result.reset();
            result.emplace(std::move(computed));
        }
    }

    const Term_eq& Ope_eq::action_operand(std::optional<Term_eq>& storage) const
    {
        if (borrowed_action_result_ != nullptr)
            return *borrowed_action_result_;
        if (!ope_action_write_into_enabled()) {
            storage.emplace(action());
            return *storage;
        }

        action_into_scratch(storage);
        return *storage;
    }

    void Ope_eq::collect_vars(std::set<std::string>& vars) const
    {
        if ((n_ope <= 0) || parts.empty())
            return;
        for (int i = 0; i < n_ope; i++) {
            if (parts[i] != nullptr)
                parts[i]->collect_vars(vars);
        }
    }

    void Ope_eq::collect_def_targets(std::set<const Term_eq*>& targets) const
    {
        if ((n_ope <= 0) || parts.empty())
            return;
        for (int i = 0; i < n_ope; i++) {
            if (parts[i] != nullptr)
                parts[i]->collect_def_targets(targets);
        }
    }

    void Ope_eq::collect_heavy_node_vals(std::vector<std::unique_ptr<Term_eq>>& holder,
                                         long long& heavy_count,
                                         long long& total_count) const
    {
        ++total_count;
        // Visit every node (order irrelevant for sizing).
        if (!parts.empty()) {
            for (int i = 0; i < n_ope; i++)
                if (parts[i] != nullptr)
                    parts[i]->collect_heavy_node_vals(holder, heavy_count, total_count);
        }
        // Transform-heavy nodes are the ones whose primal recompute the hoisting
        // lever would elide by caching their val. A leaf Ope_id over a term/def
        // result already has persistent val, and the Ope_def top node's res is
        // persistent too -- neither is a new cache cost, so neither is cached here.
        const bool is_heavy = (dynamic_cast<const Ope_mult*>(this) != nullptr) ||
                              (dynamic_cast<const Ope_der*>(this) != nullptr) ||
                              (dynamic_cast<const Ope_der_flat*>(this) != nullptr) ||
                              (dynamic_cast<const Ope_der_background*>(this) != nullptr);
        if (is_heavy) {
            ++heavy_count;
            // action() at assembly entry (pre-seed) yields the primal val with
            // der_t == nullptr -- exactly what a per-node val cache would store.
            holder.push_back(std::make_unique<Term_eq>(action()));
        }
    }
} // namespace Kadath
