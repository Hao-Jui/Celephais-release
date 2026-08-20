// Shared physical-point scalar batching for BNS setup and regrid transfers.
#pragma once

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <sstream>
#include <vector>

namespace Kadath::bns_field_transfer {

struct located_source_point
{
    explicit located_source_point(int ndim)
        : numerical(ndim)
    {
    }

    int domain = -1;
    Point numerical;
    bool has_numerical = false;
};

// A fixed group of scalar fields on one source space. Domain selection and the
// physical-to-numerical map are shared by every field in the group; each field
// still uses its own spectral basis and coefficient storage for summation.
class scalar_source_batch
{
  public:
    explicit scalar_source_batch(std::span<const Scalar* const> fields)
        : fields_(fields), coef_prepared_(fields.size(), false)
    {
        if (fields_.empty() || fields_.front() == nullptr)
            KADATH_THROW("scalar_source_batch requires non-null fields");

        source_space_ = &fields_.front()->get_space();
        source_ndim_ = fields_.front()->get_ndim();
        for (const Scalar* field : fields_) {
            if (field == nullptr)
                KADATH_THROW("scalar_source_batch requires non-null fields");
            if (&field->get_space() != source_space_ || field->get_ndim() != source_ndim_)
                KADATH_THROW("scalar_source_batch fields must share one source space");
        }
    }

    [[nodiscard]] const Space& source_space() const
    {
        return *source_space_;
    }

    [[nodiscard]] int source_ndim() const
    {
        return source_ndim_;
    }

    [[nodiscard]] std::size_t size() const
    {
        return fields_.size();
    }

    // Call once from a serial region before sharing this batch between worker
    // threads. Afterwards value()/values() only read coefficient state.
    void prepare_coefficients() const
    {
        for (std::size_t lane = 0; lane < fields_.size(); ++lane) {
            fields_[lane]->coef();
            coef_prepared_[lane] = true;
        }
    }

    [[nodiscard]] located_source_point locate(const Point& point) const
    {
        return locate_impl(point, [](std::size_t) { return true; });
    }

    [[nodiscard]] located_source_point locate(
        const Point& point, std::span<const std::size_t> active_lanes) const
    {
        for (const std::size_t lane : active_lanes) {
            if (lane >= fields_.size())
                KADATH_THROW("scalar_source_batch active lane is out of range");
        }
        return locate_impl(point, [&](std::size_t lane) {
            for (const std::size_t active : active_lanes)
                if (active == lane)
                    return true;
            return false;
        });
    }

    [[nodiscard]] double value(const located_source_point& point, std::size_t lane) const
    {
        if (lane >= fields_.size())
            KADATH_THROW("scalar_source_batch value lane is out of range");
        if (point.domain < 0 || point.domain >= source_space_->get_nbr_domains())
            KADATH_THROW("scalar_source_batch received an invalid located point");

        const Val_domain& values = fields_[lane]->at(point.domain);
        if (values.check_if_zero())
            return 0.;
        if (!point.has_numerical)
            KADATH_THROW("scalar_source_batch nonzero lane needs a numerical point");
        // Match Scalar::val_point: a logical-zero sample does not require a
        // spectral basis, and coefficients are prepared only when the first
        // nonzero sample from this field is actually consumed.
        if (!coef_prepared_[lane]) {
            fields_[lane]->coef();
            coef_prepared_[lane] = true;
        }
        return values.get_base().summation(point.numerical, values.get_coef_ref());
    }

    void values(const located_source_point& point, std::span<double> output) const
    {
        if (output.size() != fields_.size())
            KADATH_THROW("scalar_source_batch output size must match its field count");
        for (std::size_t lane = 0; lane < fields_.size(); ++lane)
            output[lane] = value(point, lane);
    }

    // Evaluate one field at four points. The spectral four-lane kernel is used
    // only when every point belongs to the same source domain; mixed-domain
    // tiles retain the scalar path and its domain-specific bases.
    void value_points4(
        std::span<const located_source_point* const, 4> points,
        std::size_t lane, std::span<double, 4> output) const
    {
        if (lane >= fields_.size())
            KADATH_THROW("scalar_source_batch value_points4 lane is out of range");
        for (const located_source_point* point : points)
            if (point == nullptr)
                KADATH_THROW("scalar_source_batch value_points4 requires non-null points");

        const int domain = points.front()->domain;
        bool same_domain = domain >= 0 && domain < source_space_->get_nbr_domains();
        for (const located_source_point* point : points)
            same_domain = same_domain && point->domain == domain;
        if (!same_domain) {
            for (std::size_t point = 0; point < points.size(); ++point)
                output[point] = value(*points[point], lane);
            return;
        }

        const Val_domain& values = fields_[lane]->at(domain);
        if (values.check_if_zero()) {
            std::fill(output.begin(), output.end(), 0.);
            return;
        }
        std::array<const Point*, 4> numerical{};
        for (std::size_t point = 0; point < points.size(); ++point) {
            if (!points[point]->has_numerical)
                KADATH_THROW("scalar_source_batch nonzero lane needs numerical points");
            numerical[point] = &points[point]->numerical;
        }
        if (!coef_prepared_[lane]) {
            fields_[lane]->coef();
            coef_prepared_[lane] = true;
        }
        values.get_base().summation_points4(
            numerical, values.get_coef_ref(), output);
    }

  private:
    template <typename lane_active_t>
    [[nodiscard]] located_source_point locate_impl(
        const Point& point, lane_active_t&& lane_active) const
    {
        located_source_point result(source_ndim_);
        for (int candidate = source_space_->get_nbr_domains() - 1;
             candidate >= 0; --candidate) {
            if (source_space_->get_domain(candidate)->is_in(point)) {
                result.domain = candidate;
                break;
            }
        }
        if (result.domain < 0) {
            std::ostringstream message;
            message << "Point " << point << "not found in the computational space...\n";
            KADATH_THROW(message.str());
        }

        bool needs_numerical = false;
        for (std::size_t lane = 0; lane < fields_.size(); ++lane) {
            if (lane_active(lane) && !fields_[lane]->at(result.domain).check_if_zero()) {
                needs_numerical = true;
                break;
            }
        }
        if (needs_numerical) {
            result.numerical = source_space_->get_domain(result.domain)->absol_to_num(point);
            result.has_numerical = true;
        }
        return result;
    }

    std::span<const Scalar* const> fields_;
    mutable std::vector<bool> coef_prepared_;
    const Space* source_space_ = nullptr;
    int source_ndim_ = 0;
};

struct scalar_import_field
{
    Scalar* target;
    const Scalar* source;
};

inline scalar_import_field import_field(Scalar& target, const Scalar& source)
{
    return {&target, &source};
}

// Exact Scalar::import contract for multiple fields sharing one target/source
// space pair: configuration-space targets, sens=-1 source-domain selection,
// and no write at the compactified infinity collocation point.
inline void import_scalar_batch(std::span<const scalar_import_field> fields)
{
    if (fields.empty())
        return;
    if (fields.front().target == nullptr || fields.front().source == nullptr)
        KADATH_THROW("BNS import_scalar_batch requires non-null fields");

    const Space& target_space = fields.front().target->get_space();
    const int target_ndim = fields.front().target->get_ndim();
    std::vector<const Scalar*> sources;
    sources.reserve(fields.size());
    for (const scalar_import_field& field : fields) {
        if (field.target == nullptr || field.source == nullptr)
            KADATH_THROW("BNS import_scalar_batch requires non-null fields");
        if (&field.target->get_space() != &target_space ||
            field.target->get_ndim() != target_ndim)
            KADATH_THROW("BNS import_scalar_batch targets must share one space");
        field.target->set_in_conf();
        field.target->allocate_conf();
        sources.push_back(field.source);
    }

    scalar_source_batch source_batch(
        std::span<const Scalar* const>(sources.data(), sources.size()));
    if (source_batch.source_ndim() != target_ndim)
        KADATH_THROW("BNS import_scalar_batch source/target dimensions must match");

    std::vector<double> sampled(fields.size());
    const int target_ndom = target_space.get_nbr_domains();
    for (int target_dom = 0; target_dom < target_ndom; ++target_dom) {
        const Domain& target_domain = *target_space.get_domain(target_dom);
        const Dim_array points = target_domain.get_nbr_points();
        Point source_point(target_ndim);
        Index pos(points);
        do {
            if (target_dom == target_ndom - 1 && pos(0) == points(0) - 1)
                continue;
            for (int component = 1; component <= target_ndim; ++component)
                source_point.set(component) = target_domain.get_cart(component)(pos);
            const located_source_point located = source_batch.locate(source_point);
            source_batch.values(located, sampled);
            for (std::size_t lane = 0; lane < fields.size(); ++lane)
                fields[lane].target->set_domain(target_dom).set(pos) = sampled[lane];
        } while (pos.inc());
    }
}

enum class two_source_combination
{
    background,
    weighted_sum,
    accumulate_from_zero,
};

struct two_source_field
{
    Scalar* target;
    const Scalar* source1;
    const Scalar* source2;
    double asymptotic;
    two_source_combination combination;
    bool write_compactified;
};

// Cold binary superposition. The target traversal, translated points, source
// lookups, numerical maps, and decay weights are shared by all output fields.
inline void superpose_two_source_batch(
    std::span<const two_source_field> fields,
    const std::array<double, 2>& centers,
    const std::array<double, 2>& invw4)
{
    if (fields.empty())
        return;
    if (fields.front().target == nullptr)
        KADATH_THROW("superpose_two_source_batch requires non-null targets");

    const Space& target_space = fields.front().target->get_space();
    const int target_ndim = fields.front().target->get_ndim();
    if (target_ndim != 3)
        KADATH_THROW("superpose_two_source_batch requires three-dimensional fields");

    std::vector<const Scalar*> sources1;
    std::vector<const Scalar*> sources2;
    std::vector<int> lanes1(fields.size(), -1);
    std::vector<int> lanes2(fields.size(), -1);
    for (std::size_t lane = 0; lane < fields.size(); ++lane) {
        const two_source_field& field = fields[lane];
        if (field.target == nullptr)
            KADATH_THROW("superpose_two_source_batch requires non-null targets");
        if (&field.target->get_space() != &target_space ||
            field.target->get_ndim() != target_ndim)
            KADATH_THROW("superpose_two_source_batch targets must share one space");
        if (field.source1 != nullptr) {
            lanes1[lane] = static_cast<int>(sources1.size());
            sources1.push_back(field.source1);
        }
        if (field.source2 != nullptr) {
            lanes2[lane] = static_cast<int>(sources2.size());
            sources2.push_back(field.source2);
        }
    }
    if (sources1.empty() || sources2.empty())
        KADATH_THROW("superpose_two_source_batch requires fields from both source spaces");

    scalar_source_batch source_batch1(
        std::span<const Scalar* const>(sources1.data(), sources1.size()));
    scalar_source_batch source_batch2(
        std::span<const Scalar* const>(sources2.data(), sources2.size()));
    if (source_batch1.source_ndim() != 3 || source_batch2.source_ndim() != 3)
        KADATH_THROW("superpose_two_source_batch requires three-dimensional sources");

    std::vector<double> sampled1(sources1.size());
    std::vector<double> sampled2(sources2.size());
    const int target_ndom = target_space.get_nbr_domains();
    for (int target_dom = 0; target_dom < target_ndom; ++target_dom) {
        const Domain& target_domain = *target_space.get_domain(target_dom);
        Index pos(target_domain.get_nbr_points());
        const bool compactified = target_dom == target_ndom - 1;
        do {
            if (compactified) {
                for (const two_source_field& field : fields)
                    if (field.write_compactified)
                        field.target->set_domain(target_dom).set(pos) = field.asymptotic;
                continue;
            }

            const double x = target_domain.get_cart(1)(pos);
            const double y = target_domain.get_cart(2)(pos);
            const double z = target_domain.get_cart(3)(pos);

            Point point1(3);
            point1.set(1) = x - centers[0];
            point1.set(2) = y;
            point1.set(3) = z;
            const double r2 = y * y + z * z;
            const double r2_1 = (x - centers[0]) * (x - centers[0]) + r2;
            const double r4_1 = r2_1 * r2_1;
            const double decay1 = std::exp(-r4_1 * invw4[0]);

            Point point2(3);
            point2.set(1) = x - centers[1];
            point2.set(2) = y;
            point2.set(3) = z;
            const double r2_2 = (x - centers[1]) * (x - centers[1]) + r2;
            const double r4_2 = r2_2 * r2_2;
            const double decay2 = std::exp(-r4_2 * invw4[1]);

            source_batch1.values(source_batch1.locate(point1), sampled1);
            source_batch2.values(source_batch2.locate(point2), sampled2);

            for (std::size_t lane = 0; lane < fields.size(); ++lane) {
                const two_source_field& field = fields[lane];
                const bool has1 = lanes1[lane] >= 0;
                const bool has2 = lanes2[lane] >= 0;
                const double value1 = has1 ? sampled1[static_cast<std::size_t>(lanes1[lane])] : 0.;
                const double value2 = has2 ? sampled2[static_cast<std::size_t>(lanes2[lane])] : 0.;

                double value = field.asymptotic;
                switch (field.combination) {
                case two_source_combination::background:
                    if (has1)
                        value += decay1 * (value1 - field.asymptotic);
                    if (has2)
                        value += decay2 * (value2 - field.asymptotic);
                    break;
                case two_source_combination::weighted_sum:
                    if (has1 && has2)
                        value = decay1 * value1 + decay2 * value2;
                    else if (has1)
                        value = decay1 * value1;
                    else if (has2)
                        value = decay2 * value2;
                    break;
                case two_source_combination::accumulate_from_zero:
                    value = 0.;
                    if (has1)
                        value += decay1 * value1;
                    if (has2)
                        value += decay2 * value2;
                    break;
                }
                field.target->set_domain(target_dom).set(pos) = value;
            }
        } while (pos.inc());
    }
}

} // namespace Kadath::bns_field_transfer
