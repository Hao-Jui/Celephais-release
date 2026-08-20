// Batched collocation import for the axisymmetric XCTS workflows.
#pragma once

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

#include <cstddef>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

namespace Kadath::ns_2d_xcts_import {

struct scalar_import_field
{
    Scalar* target;
    const Scalar* source;
};

inline scalar_import_field import_field(Scalar& target, const Scalar& source)
{
    return {&target, &source};
}

template <typename source_point_mapper_t>
inline void import_scalar_batch(std::span<const scalar_import_field> fields,
                                source_point_mapper_t&& source_point_mapper)
{
    if (fields.empty())
        return;

    if (fields.front().target == nullptr || fields.front().source == nullptr)
        KADATH_THROW("import_scalar_batch requires non-null fields");

    const Space& target_space = fields.front().target->get_space();
    const Space& source_space = fields.front().source->get_space();
    const int target_ndim = fields.front().target->get_ndim();
    const int source_ndim = fields.front().source->get_ndim();

    for (const scalar_import_field& field : fields) {
        if (field.target == nullptr || field.source == nullptr)
            KADATH_THROW("import_scalar_batch requires non-null fields");
        if (&field.target->get_space() != &target_space || field.target->get_ndim() != target_ndim)
            KADATH_THROW("import_scalar_batch targets must share one space");
        if (&field.source->get_space() != &source_space || field.source->get_ndim() != source_ndim)
            KADATH_THROW("import_scalar_batch sources must share one space");
    }

    for (const scalar_import_field& field : fields) {
        field.target->set_in_conf();
        field.target->allocate_conf();
        field.source->coef();
    }

    const int target_ndom = target_space.get_nbr_domains();
    const int source_ndom = source_space.get_nbr_domains();
    for (int target_dom = 0; target_dom < target_ndom; ++target_dom) {
        const Domain& target_domain = *target_space.get_domain(target_dom);
        const Dim_array points = target_domain.get_nbr_points();

        std::vector<Val_domain> coordinates;
        coordinates.reserve(static_cast<std::size_t>(target_ndim));
        for (int component = 1; component <= target_ndim; ++component)
            coordinates.emplace_back(target_domain.get_cart(component));

        Index pos(points);
        do {
            // The last radial collocation point of a compactified outer domain
            // is spatial infinity. Scalar::import skips that point, so the
            // batch path does too.
            if (target_dom == target_ndom - 1 && pos(0) == points(0) - 1)
                continue;

            const Point source_point = source_point_mapper(
                std::span<const Val_domain>(coordinates.data(), coordinates.size()), pos);

            // Match Scalar::val_point's default sens=-1 selection: when a
            // collocation point lies on an interface, choose the outermost
            // source domain containing it.
            int source_dom = -1;
            for (int candidate = source_ndom - 1; candidate >= 0; --candidate) {
                if (source_space.get_domain(candidate)->is_in(source_point)) {
                    source_dom = candidate;
                    break;
                }
            }
            if (source_dom < 0) {
                std::ostringstream message;
                message << "Point " << source_point << "not found in the computational space...\n";
                KADATH_THROW(message.str());
            }

            bool needs_numerical_point = false;
            for (const scalar_import_field& field : fields)
                needs_numerical_point = needs_numerical_point || !field.source->at(source_dom).check_if_zero();

            Point numerical(source_ndim);
            if (needs_numerical_point)
                numerical = source_space.get_domain(source_dom)->absol_to_num(source_point);

            for (const scalar_import_field& field : fields) {
                const Val_domain& source_values = field.source->at(source_dom);
                const double value = source_values.check_if_zero()
                                         ? 0.
                                         : source_values.get_base().summation(
                                               numerical, source_values.get_coef_ref());
                field.target->set_domain(target_dom).set(pos) = value;
            }
        } while (pos.inc());
    }
}

inline void import_scalar_batch(std::span<const scalar_import_field> fields)
{
    import_scalar_batch(fields, [](const std::span<const Val_domain> coordinates, const Index& pos) {
        Point source_point(static_cast<int>(coordinates.size()));
        for (std::size_t component = 0; component < coordinates.size(); ++component)
            source_point.set(static_cast<int>(component) + 1) = coordinates[component](pos);
        return source_point;
    });
}

} // namespace Kadath::ns_2d_xcts_import
