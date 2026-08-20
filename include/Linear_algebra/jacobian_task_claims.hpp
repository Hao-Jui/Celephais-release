#pragma once

#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace Kadath
{

    enum class JacobianTaskState : std::uint8_t {
        Pending,
        Claimed,
        Complete,
    };

    template <typename Index>
    concept JacobianTaskIndex = std::integral<Index> && !std::same_as<std::remove_cv_t<Index>, bool>;

    namespace jacobian_task_claim_detail
    {

        inline constexpr std::size_t state_required_alignment = std::atomic_ref<JacobianTaskState>::required_alignment;

        struct alignas(state_required_alignment) Slot {
            mutable JacobianTaskState state = JacobianTaskState::Pending;
        };

        static_assert(std::is_trivially_copyable_v<JacobianTaskState>);
        static_assert(alignof(Slot) >= state_required_alignment);
        static_assert(offsetof(Slot, state) % state_required_alignment == 0);

    } // namespace jacobian_task_claim_detail

    // Atomic state overlay for experiments that schedule fixed-width Jacobian
    // column groups. Storage never moves after construction, and every access to
    // a slot after construction goes through std::atomic_ref.
    class JacobianTaskClaims
    {
      public:
        static constexpr std::size_t required_alignment = jacobian_task_claim_detail::state_required_alignment;

        explicit JacobianTaskClaims(std::size_t task_count) : slots_(task_count) {}

        JacobianTaskClaims(const JacobianTaskClaims&) = delete;
        JacobianTaskClaims& operator=(const JacobianTaskClaims&) = delete;
        JacobianTaskClaims(JacobianTaskClaims&&) = delete;
        JacobianTaskClaims& operator=(JacobianTaskClaims&&) = delete;

        [[nodiscard]] std::size_t size() const noexcept { return slots_.size(); }

        [[nodiscard]] JacobianTaskState state(std::size_t index) const
        {
            if (index >= slots_.size())
                throw std::out_of_range("Jacobian task claim index out of range");
            return atomic_state(slots_[index]).load(std::memory_order_acquire);
        }

        template <JacobianTaskIndex Index, std::size_t Extent>
            requires(Extent != std::dynamic_extent)
        [[nodiscard]] bool try_claim(std::span<const Index, Extent> indices) noexcept
        {
            return transition_group(indices, JacobianTaskState::Pending, JacobianTaskState::Claimed);
        }

        // Only the successful claimant may release or complete its group. This
        // ownership rule keeps the multi-slot transition rollback-safe while
        // other workers concurrently try to claim Pending slots.
        template <JacobianTaskIndex Index, std::size_t Extent>
            requires(Extent != std::dynamic_extent)
        [[nodiscard]] bool release(std::span<const Index, Extent> indices) noexcept
        {
            return transition_group(indices, JacobianTaskState::Claimed, JacobianTaskState::Pending);
        }

        template <JacobianTaskIndex Index, std::size_t Extent>
            requires(Extent != std::dynamic_extent)
        [[nodiscard]] bool complete(std::span<const Index, Extent> indices) noexcept
        {
            return transition_group(indices, JacobianTaskState::Claimed, JacobianTaskState::Complete);
        }

      private:
        using Slot = jacobian_task_claim_detail::Slot;

        static std::atomic_ref<JacobianTaskState> atomic_state(const Slot& slot) noexcept
        {
            return std::atomic_ref<JacobianTaskState>(slot.state);
        }

        template <JacobianTaskIndex Index, std::size_t Extent>
        [[nodiscard]] bool valid_group(std::span<const Index, Extent> indices) const noexcept
        {
            for (std::size_t i = 0; i < Extent; ++i) {
                if (!std::in_range<std::size_t>(indices[i]))
                    return false;
                const std::size_t index = static_cast<std::size_t>(indices[i]);
                if (index >= slots_.size())
                    return false;
                for (std::size_t j = 0; j < i; ++j) {
                    if (index == static_cast<std::size_t>(indices[j]))
                        return false;
                }
            }
            return true;
        }

        template <JacobianTaskIndex Index, std::size_t Extent>
        [[nodiscard]] bool transition_group(std::span<const Index, Extent> indices, JacobianTaskState from,
                                            JacobianTaskState to) noexcept
        {
            if (!valid_group(indices))
                return false;

            std::size_t transitioned = 0;
            for (; transitioned < Extent; ++transitioned) {
                const std::size_t index = static_cast<std::size_t>(indices[transitioned]);
                JacobianTaskState expected = from;
                if (!atomic_state(slots_[index])
                         .compare_exchange_strong(expected, to, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    while (transitioned > 0) {
                        --transitioned;
                        const std::size_t rollback_index = static_cast<std::size_t>(indices[transitioned]);
                        JacobianTaskState rollback_expected = to;
                        const bool restored =
                            atomic_state(slots_[rollback_index])
                                .compare_exchange_strong(rollback_expected, from, std::memory_order_release,
                                                         std::memory_order_relaxed);
                        assert(restored && "Jacobian task group ownership violated");
                        (void)restored;
                    }
                    return false;
                }
            }
            return true;
        }

        std::vector<Slot> slots_;
    };

} // namespace Kadath
