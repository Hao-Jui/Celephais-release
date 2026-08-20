#pragma once

#include <cstddef>
#include <memory>
#include <span>

namespace Kadath
{
    class Domain;
    class Val_domain;

    /**
     * Non-owning view of compatible numerical-coordinate derivative lanes.
     *
     * Val_domain owns the input derivative caches and the committed Cartesian
     * derivative caches. This object owns only staged results, so a failed
     * Domain adapter cannot leave a partially updated source field.
     */
    class DerAbsLaneBatch final
    {
      public:
        ~DerAbsLaneBatch();
        DerAbsLaneBatch(DerAbsLaneBatch&&) noexcept;
        DerAbsLaneBatch& operator=(DerAbsLaneBatch&&) noexcept;
        DerAbsLaneBatch(const DerAbsLaneBatch&) = delete;
        DerAbsLaneBatch& operator=(const DerAbsLaneBatch&) = delete;

        std::size_t lane_count() const noexcept;
        int dimension() const noexcept;
        const Domain& domain() const noexcept;

        /// Zero-based numerical-coordinate derivative for one lane.
        const Val_domain& der_var(std::size_t lane, int axis) const;

        /// Stage one zero-based Cartesian derivative result for atomic commit.
        void set_der_abs(std::size_t lane, int axis, Val_domain value);

      private:
        struct Impl;
        std::unique_ptr<Impl> impl;

        explicit DerAbsLaneBatch(std::span<const Val_domain* const> fields);
        void adopt_legacy_result(std::size_t lane, int axis, Val_domain* value);
        void commit();

        friend class Domain;
        friend class Val_domain;
    };
} // namespace Kadath
