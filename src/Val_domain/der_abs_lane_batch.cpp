#include "For_Kadath/Val_domain/der_abs_lane_batch.hpp"

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <utility>
#include <vector>

namespace Kadath
{
    struct DerAbsLaneBatch::Impl
    {
        const Domain* domain = nullptr;
        int dimension = 0;
        std::vector<const Val_domain*> owners;
        std::vector<const Val_domain*> derivatives;
        std::vector<std::unique_ptr<Val_domain>> staged;

        std::size_t offset(std::size_t lane, int axis) const noexcept
        {
            return lane * static_cast<std::size_t>(dimension) + static_cast<std::size_t>(axis);
        }
    };

    DerAbsLaneBatch::DerAbsLaneBatch(std::span<const Val_domain* const> fields)
        : impl(std::make_unique<Impl>())
    {
        if (fields.empty())
            KADATH_THROW("DerAbsLaneBatch requires at least one field");
        if (fields.front() == nullptr)
            KADATH_THROW("DerAbsLaneBatch received a null field");

        impl->domain = fields.front()->zone;
        impl->dimension = impl->domain->get_ndim();
        impl->owners.reserve(fields.size());
        impl->derivatives.reserve(fields.size() * static_cast<std::size_t>(impl->dimension));
        impl->staged.resize(fields.size() * static_cast<std::size_t>(impl->dimension));

        for (const Val_domain* field : fields) {
            if (field == nullptr)
                KADATH_THROW("DerAbsLaneBatch received a null field");
            if (field->zone != impl->domain)
                KADATH_THROW("DerAbsLaneBatch fields must belong to one Domain");

            impl->owners.push_back(field);
            for (int axis = 0; axis < impl->dimension; ++axis) {
                if (field->p_der_var[axis] == nullptr)
                    KADATH_THROW("DerAbsLaneBatch requires prepared numerical derivatives");
                impl->derivatives.push_back(field->p_der_var[axis]);
            }
        }
    }

    DerAbsLaneBatch::~DerAbsLaneBatch() = default;
    DerAbsLaneBatch::DerAbsLaneBatch(DerAbsLaneBatch&&) noexcept = default;
    DerAbsLaneBatch& DerAbsLaneBatch::operator=(DerAbsLaneBatch&&) noexcept = default;

    std::size_t DerAbsLaneBatch::lane_count() const noexcept
    {
        return impl->owners.size();
    }

    int DerAbsLaneBatch::dimension() const noexcept
    {
        return impl->dimension;
    }

    const Domain& DerAbsLaneBatch::domain() const noexcept
    {
        return *impl->domain;
    }

    const Val_domain& DerAbsLaneBatch::der_var(std::size_t lane, int axis) const
    {
        if (lane >= impl->owners.size())
            KADATH_THROW("DerAbsLaneBatch lane index is out of range");
        if (axis < 0 || axis >= impl->dimension)
            KADATH_THROW("DerAbsLaneBatch derivative axis is out of range");
        return *impl->derivatives[impl->offset(lane, axis)];
    }

    void DerAbsLaneBatch::set_der_abs(std::size_t lane, int axis, Val_domain value)
    {
        if (value.get_domain() != impl->domain)
            KADATH_THROW("DerAbsLaneBatch result belongs to a different Domain");
        adopt_legacy_result(lane, axis, new Val_domain(std::move(value)));
    }

    void DerAbsLaneBatch::adopt_legacy_result(std::size_t lane, int axis, Val_domain* value)
    {
        std::unique_ptr<Val_domain> owned(value);
        if (lane >= impl->owners.size())
            KADATH_THROW("DerAbsLaneBatch lane index is out of range");
        if (axis < 0 || axis >= impl->dimension)
            KADATH_THROW("DerAbsLaneBatch derivative axis is out of range");
        if (owned == nullptr)
            KADATH_THROW("DerAbsLaneBatch adapter produced a null result");
        if (owned->get_domain() != impl->domain)
            KADATH_THROW("DerAbsLaneBatch adapter produced a result for a different Domain");

        std::unique_ptr<Val_domain>& slot = impl->staged[impl->offset(lane, axis)];
        if (slot != nullptr)
            KADATH_THROW("DerAbsLaneBatch adapter produced one axis more than once");
        slot = std::move(owned);
    }

    void DerAbsLaneBatch::commit()
    {
        for (std::size_t lane = 0; lane < impl->owners.size(); ++lane) {
            for (int axis = 0; axis < impl->dimension; ++axis) {
                const std::size_t offset = impl->offset(lane, axis);
                if (impl->staged[offset] == nullptr)
                    KADATH_THROW("DerAbsLaneBatch adapter did not produce every axis");
                if (impl->owners[lane]->p_der_abs[axis] != nullptr)
                    KADATH_THROW("DerAbsLaneBatch cannot commit over an existing derivative cache");
            }
        }

        for (std::size_t lane = 0; lane < impl->owners.size(); ++lane)
            for (int axis = 0; axis < impl->dimension; ++axis)
                impl->owners[lane]->p_der_abs[axis] = impl->staged[impl->offset(lane, axis)].release();
    }
} // namespace Kadath
