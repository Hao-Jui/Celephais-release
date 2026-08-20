#pragma once

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"

#include <string>

namespace Kadath
{
    namespace detail
    {
        template <typename DerivativeTransform>
        Tensor apply_scalar_tensor_derivative_lane(
            int domain,
            const Term_eq& target,
            int lane,
            DerivativeTransform derivative_transform)
        {
            const Tensor& derivative = target.get_der_t(lane);
            const Tensor& value_tensor = target.get_val_t();
            if (derivative.get_valence() != value_tensor.get_valence() ||
                derivative.get_n_comp() != value_tensor.get_n_comp()) {
                bool derivative_is_zero = true;
                for (int component = 0; component < derivative.get_n_comp(); ++component) {
                    Array<int> index(derivative.indices(component));
                    if (!derivative(index)(domain).check_if_zero()) {
                        derivative_is_zero = false;
                        break;
                    }
                }
                if (!derivative_is_zero) {
                    KADATH_THROW("scalar unary derivative lane layout does not match scalar value layout");
                }

                Tensor zero_result(
                    one_domain_storage, domain, value_tensor, false);
                for (int component = 0; component < zero_result.get_n_comp(); ++component) {
                    Array<int> index(zero_result.indices(component));
                    zero_result.set(index).set_domain(domain).set_zero();
                }
                return zero_result;
            }

            Tensor result(one_domain_storage, domain, derivative, false);
            for (int component = 0; component < derivative.get_n_comp(); ++component) {
                Array<int> index(derivative.indices(component));
                Val_domain lane_derivative(derivative(index)(domain));
                const Val_domain& value(value_tensor(index)(domain));
                result.set(index).set_domain(domain) = derivative_transform(lane_derivative, value);
            }
            return result;
        }

        template <typename ValueTransform, typename DerivativeTransform>
        Term_eq apply_scalar_unary_operator(
            int domain,
            const Term_eq& target,
            const std::string& operator_name,
            ValueTransform value_transform,
            DerivativeTransform derivative_transform)
        {
            switch (target.get_type_data()) {
                case TERM_T: {
                    if (target.get_val_t().get_valence() != 0) {
                        KADATH_THROW(operator_name + " only defined with respect for a scalar");
                    }

                    Tensor value_result(
                        one_domain_storage, domain, target.get_val_t(), false);
                    for (int component = 0; component < target.get_val_t().get_n_comp(); ++component) {
                        Array<int> index(target.get_val_t().indices(component));
                        const Val_domain& value(target.get_val_t()(index)(domain));
                        value_result.set(index).set_domain(domain) = value_transform(value);
                    }

                    Term_eq result(domain, value_result);
                    if (target.has_der_t(0)) {
                        Tensor derivative_result(
                            apply_scalar_tensor_derivative_lane(domain, target, 0, derivative_transform));
                        result.set_der_t(derivative_result);
                    }
                    for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane) {
                        if (target.has_der_t(lane)) {
                            result.set_der_t(
                                lane,
                                apply_scalar_tensor_derivative_lane(domain, target, lane, derivative_transform));
                        }
                    }
                    return result;
                }
                case TERM_D: {
                    Term_eq result(domain, value_transform(target.get_val_d()));
                    if (target.has_der_d(0))
                        result.set_der_d(derivative_transform(target.get_der_d(), target.get_val_d()));
                    for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane) {
                        if (target.has_der_d(lane)) {
                            result.set_der_d(
                                lane,
                                derivative_transform(target.get_der_d(lane), target.get_val_d()));
                        }
                    }
                    return result;
                }
                default:
                    KADATH_THROW("Unknown storage in Term_eq");
            }
        }
    } // namespace detail
} // namespace Kadath
