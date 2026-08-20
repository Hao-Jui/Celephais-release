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
    namespace adapted_polar_detail
    {
        inline void prepare_numerical_derivative_lanes(int domain, const Term_eq& source)
        {
            if (!Val_domain::derivative_lane_tiling_enabled() || source.get_p_der_t() == nullptr)
                return;
            if (!derivative_lane_detail::all_derivative_lanes_match_value_layout(source))
                return;

            const int lane_count = source.get_derivative_lane_count();
            if (lane_count > Term_eq::max_derivative_lanes)
                KADATH_THROW("Adapted polar derivative lane count exceeds the packed tile bound");

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

        inline int combined_derivative_lane_count(const Term_eq& first, const Term_eq& second)
        {
            return std::max(first.get_derivative_lane_count(), second.get_derivative_lane_count());
        }

        inline bool has_tensor_derivative_lane(const Term_eq& term, int lane)
        {
            return term.get_type_data() == TERM_T && term.has_der_t(lane);
        }

        inline Tensor theta_prime_derivative_tensor(int domain, const Tensor& derivative)
        {
            Tensor derivative_result(derivative, false);
            for (int component = 0; component < derivative.get_n_comp(); component++) {
                Array<int> index(derivative.indices(component));
                derivative_result.set(index).set_domain(domain) = derivative(index)(domain).der_var(2);
            }
            return derivative_result;
        }

        inline Term_eq theta_prime_derivative_term_eq(int domain, const Term_eq& source)
        {
            const Tensor& source_value = *source.get_p_val_t();
            Tensor value_result(source_value, false);
            for (int component = 0; component < source_value.get_n_comp(); component++) {
                Array<int> index(source_value.indices(component));
                value_result.set(index).set_domain(domain) = source_value(index)(domain).der_var(2);
            }

            if (source.get_p_der_t() == nullptr)
                return Term_eq(domain, value_result);

            // The generic Adapter remains the lane-by-lane der_var calls below.
            // On matching packed layouts, prime every tangent lane through the
            // batched transform Interface before materialising the Tensor lanes.
            prepare_numerical_derivative_lanes(domain, source);
            Term_eq result(domain, value_result, theta_prime_derivative_tensor(domain, source.get_der_t()));
            result.set_derivative_lane_count(source.get_derivative_lane_count());
            for (int lane = 1; lane < source.get_derivative_lane_count(); lane++) {
                if (source.has_der_t(lane))
                    result.set_der_t(lane, theta_prime_derivative_tensor(domain, source.get_der_t(lane)));
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
            Tensor derivative_result(source_value, false);

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

            Tensor value_result(source_value, false);
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

            const Val_domain radial_derivative_squared(radial_derivative * radial_derivative);
            // prepare_der_var_batch computes both adapted coordinates for all
            // active tangent lanes in one traversal. Subsequent der_var calls
            // therefore read the committed caches while preserving the legacy
            // lane materialisation and arithmetic order.
            prepare_numerical_derivative_lanes(domain, source);
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
                result.set_der_t(
                    lane,
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

        /**
         * Shared scalar Laplacian schedule for the three adapted-polar shell
         * domains. Axisymmetric fields have no azimuthal contribution, so do
         * not materialize f/sin(theta), multiply it by zero, and subtract it
         * before the remaining angular division.
         */
        template <typename RadialDerivative,
                  typename ThetaDerivative,
                  typename DivideSinTheta,
                  typename MultiplyCosTheta>
        Term_eq scalar_laplacian_term_eq(const Term_eq& source,
                                         int azimuthal_factor,
                                         const Term_eq& radius,
                                         RadialDerivative&& radial_derivative,
                                         ThetaDerivative&& theta_derivative,
                                         DivideSinTheta&& divide_sin_theta,
                                         MultiplyCosTheta&& multiply_cos_theta)
        {
            Term_eq derivative_theta(theta_derivative(source));
            Term_eq second_derivative_theta(theta_derivative(derivative_theta));

            Term_eq cosine_derivative_theta(multiply_cos_theta(derivative_theta));
            Term_eq cotangent_and_azimuthal = [&]() {
                if (azimuthal_factor == 0)
                    return divide_sin_theta(cosine_derivative_theta);

                Term_eq source_over_sin_theta(divide_sin_theta(source));
                return divide_sin_theta(cosine_derivative_theta - azimuthal_factor * source_over_sin_theta);
            }();

            Term_eq derivative_radius(radial_derivative(source));
            return radial_derivative(derivative_radius) + 2 * derivative_radius / radius +
                   (second_derivative_theta + cotangent_and_azimuthal) / radius / radius;
        }
    } // namespace adapted_polar_detail
} // namespace Kadath
