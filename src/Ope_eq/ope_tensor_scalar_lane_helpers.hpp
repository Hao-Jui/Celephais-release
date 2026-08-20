#pragma once

#include "For_Kadath/Term_eq/term_eq.hpp"

namespace Kadath
{
    namespace ope_tensor_scalar_lane_detail
    {
        template <typename EvaluateDerivative>
        Term_eq make_double_term_from_tensor_lanes(int domain,
                                                   double value,
                                                   const Term_eq& target,
                                                   EvaluateDerivative evaluate_derivative)
        {
            Term_eq result(domain, value);
            if (target.has_der_t(0))
                result.set_der_d(evaluate_derivative(target.get_der_t(0)));
            result.set_derivative_lane_count(target.get_derivative_lane_count());
            for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane) {
                if (target.has_der_t(lane))
                    result.set_der_d(lane, evaluate_derivative(target.get_der_t(lane)));
            }
            return result;
        }
    } // namespace ope_tensor_scalar_lane_detail
} // namespace Kadath
