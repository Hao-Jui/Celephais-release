#pragma once

#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "../../Term_eq/term_eq_derivative_lane_layout.hpp"

#include <algorithm>
#include <array>
#include <span>
#include <utility>
#include <vector>

namespace Kadath
{
    namespace adapted_spherical_detail
    {
        inline void prepare_numerical_derivative_lanes(int domain, const Term_eq& source)
        {
            if (!Val_domain::derivative_lane_tiling_enabled() || source.get_p_der_t() == nullptr)
                return;
            if (!derivative_lane_detail::all_derivative_lanes_match_value_layout(source))
                return;

            const int lane_count = source.get_derivative_lane_count();
            if (lane_count > Term_eq::max_derivative_lanes)
                KADATH_THROW("Adapted derivative lane count exceeds the packed tile bound");

            const Tensor& source_value = source.get_val_t();
            for (int component = 0; component < source_value.get_n_comp(); component++) {
                std::array<const Val_domain*, Term_eq::max_derivative_lanes> fields{};
                std::size_t field_count = 0;
                for (int lane = 0; lane < lane_count; lane++) {
                    const Tensor* derivative = source.get_p_der_t(lane);
                    if (derivative == nullptr)
                        continue;
                    const Array<int> index(derivative->indices(component));
                    fields[field_count++] = &(*derivative)(index)(domain);
                }
                Val_domain::prepare_der_var_batch(
                    std::span<const Val_domain* const>(fields.data(), field_count));
            }
        }

        inline Tensor coordinate_derivative_tensor(int domain, const Tensor& source, int coordinate)
        {
            Tensor derivative_result(
                one_domain_storage, domain, source, false);
            for (int component = 0; component < source.get_n_comp(); component++) {
                Array<int> index(source.indices(component));
                derivative_result.set(index).set_domain(domain) = source(index)(domain).der_var(coordinate);
            }
            return derivative_result;
        }

        inline Term_eq coordinate_derivative_term_eq(int domain, const Term_eq& source, int coordinate)
        {
            const Tensor& source_value = *source.get_p_val_t();
            Tensor value_result(coordinate_derivative_tensor(domain, source_value, coordinate));

            if (source.get_p_der_t() == nullptr)
                return Term_eq(domain, value_result);

            prepare_numerical_derivative_lanes(domain, source);
            Term_eq result(domain, value_result, coordinate_derivative_tensor(domain, source.get_der_t(), coordinate));
            result.set_derivative_lane_count(source.get_derivative_lane_count());
            for (int lane = 1; lane < source.get_derivative_lane_count(); lane++) {
                if (source.has_der_t(lane))
                    result.set_der_t(lane, coordinate_derivative_tensor(domain, source.get_der_t(lane), coordinate));
            }
            return result;
        }

        inline Term_eq sum_one_domain_term_eq(int domain, const Term_eq& source)
        {
            auto derivative_ready_for_summation = [&source](int lane) {
                Tensor derivative(source.get_der_t(lane));
                if (!derivative.is_name_affected() && source.get_val_t().is_name_affected()) {
                    derivative.set_name_affected();
                    for (int index = 0; index < derivative.get_valence(); index++)
                        derivative.set_name_ind(index, source.get_val_t().get_name_ind()[index]);
                }
                return derivative;
            };

            Tensor value_result(source.get_val_t().do_summation_one_dom(domain));
            if (source.get_p_der_t() == nullptr)
                return Term_eq(domain, value_result);

            Tensor primary_derivative(derivative_ready_for_summation(0));
            Term_eq result(domain, value_result, primary_derivative.do_summation_one_dom(domain));
            result.set_derivative_lane_count(source.get_derivative_lane_count());
            for (int lane = 1; lane < source.get_derivative_lane_count(); lane++) {
                if (source.has_der_t(lane)) {
                    Tensor derivative(derivative_ready_for_summation(lane));
                    result.set_der_t(lane, derivative.do_summation_one_dom(domain));
                }
            }
            return result;
        }

        inline Tensor radial_derivative_tensor(
            int domain,
            const Tensor& source_value,
            const Tensor* source_derivative,
            const Term_eq& radial_derivative_term,
            int lane,
            const Val_domain& radial_derivative_squared,
            const std::vector<Val_domain>& source_radial_derivatives)
        {
            const Val_domain& radial_derivative = (*radial_derivative_term.get_p_val_t())()(domain);
            const Tensor* radial_derivative_tangent = radial_derivative_term.get_p_der_t(lane);
            Tensor derivative_result(
                one_domain_storage, domain, source_value, false);

            for (int component = 0; component < source_value.get_n_comp(); component++) {
                Array<int> index(source_value.indices(component));
                // Zero seed without the radial-derivative payload: the copy is
                // discarded by set_zero() anyway.
                Val_domain numerator(radial_derivative, false);
                numerator.set_zero();

                if (source_derivative != nullptr)
                    numerator = (*source_derivative)(index)(domain).der_var(1) * radial_derivative;
                if (radial_derivative_tangent != nullptr)
                    numerator -= source_radial_derivatives[static_cast<std::size_t>(component)] *
                                 (*radial_derivative_tangent)()(domain);

                derivative_result.set(index).set_domain(domain) = numerator / radial_derivative_squared;
            }
            return derivative_result;
        }

        inline Term_eq radial_derivative_term_eq(
            int domain,
            const Term_eq& source,
            const Term_eq& radial_derivative_term)
        {
            const Tensor& source_value = *source.get_p_val_t();
            const Val_domain& radial_derivative = (*radial_derivative_term.get_p_val_t())()(domain);

            bool radial_mapping_has_tangent = false;
            for (int lane = 0; lane < radial_derivative_term.get_derivative_lane_count(); ++lane)
                radial_mapping_has_tangent =
                    radial_mapping_has_tangent || radial_derivative_term.get_p_der_t(lane) != nullptr;

            std::vector<Val_domain> source_radial_derivatives;
            if (radial_mapping_has_tangent)
                source_radial_derivatives.reserve(static_cast<std::size_t>(source_value.get_n_comp()));

            Tensor value_result(
                one_domain_storage, domain, source_value, false);
            for (int component = 0; component < source_value.get_n_comp(); component++) {
                Array<int> index(source_value.indices(component));
                Val_domain source_radial_derivative(source_value(index)(domain).der_var(1));
                value_result.set(index).set_domain(domain) = source_radial_derivative / radial_derivative;
                if (radial_mapping_has_tangent)
                    source_radial_derivatives.emplace_back(std::move(source_radial_derivative));
            }

            const bool has_primary_derivative =
                source.get_p_der_t() != nullptr || radial_derivative_term.get_p_der_t() != nullptr;
            if (!has_primary_derivative)
                return Term_eq(domain, value_result);

            prepare_numerical_derivative_lanes(domain, source);
            // Lane-invariant: the mapping Jacobian squared and the value-field
            // radial derivatives are shared by every tangent lane.
            const Val_domain radial_derivative_squared(radial_derivative * radial_derivative);
            Term_eq result(domain,
                           value_result,
                           radial_derivative_tensor(
                               domain,
                               source_value,
                               source.get_p_der_t(),
                               radial_derivative_term,
                               0,
                               radial_derivative_squared,
                               source_radial_derivatives));

            const int lanes =
                std::max(source.get_derivative_lane_count(), radial_derivative_term.get_derivative_lane_count());
            result.set_derivative_lane_count(lanes);
            for (int lane = 1; lane < lanes; lane++) {
                const Tensor* source_derivative = source.get_p_der_t(lane);
                if (source_derivative == nullptr && radial_derivative_term.get_p_der_t(lane) == nullptr)
                    continue;
                result.set_der_t(lane,
                                 radial_derivative_tensor(domain,
                                                          source_value,
                                                          source_derivative,
                                                          radial_derivative_term,
                                                          lane,
                                                          radial_derivative_squared,
                                                          source_radial_derivatives));
            }
            return result;
        }
    } // namespace adapted_spherical_detail
} // namespace Kadath
