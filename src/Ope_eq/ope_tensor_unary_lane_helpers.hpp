#pragma once

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"

#include <string>

namespace Kadath
{
    namespace ope_tensor_unary_lane_detail
    {
        inline void copy_index_names(const Tensor& source, Tensor& target)
        {
            if (!source.is_name_affected())
                return;

            target.set_name_affected();
            const char* names = source.get_name_ind();
            for (int index = 0; index < source.get_valence(); ++index)
                target.set_name_ind(index, names[index]);
        }

        template <typename Transform>
        Tensor apply_componentwise(int domain, const Tensor& source, Transform transform)
        {
            Tensor result(one_domain_storage, domain, source, false);
            for (int component = 0; component < source.get_n_comp(); ++component) {
                Array<int> index(source.indices(component));
                const Val_domain& value(source(index)(domain));
                if (value.check_if_zero())
                    result.set(index).set_domain(domain).set_zero();
                else
                    result.set(index).set_domain(domain) = transform(value);
            }
            return result;
        }

        template <typename Transform>
        Term_eq apply_tensor_unary_operator(int domain,
                                            const Term_eq& target,
                                            const std::string& operator_name,
                                            Transform transform)
        {
            if (target.get_type_data() != TERM_T)
                KADATH_THROW(operator_name + " only defined with respect to a tensor");

            Tensor value_result(apply_componentwise(domain, target.get_val_t(), transform));
            Term_eq result(domain, value_result);
            result.set_derivative_lane_count(target.get_derivative_lane_count());

            if (target.has_der_t(0))
                result.set_der_t(apply_componentwise(domain, target.get_der_t(0), transform));
            for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane)
                if (target.has_der_t(lane))
                    result.set_der_t(lane, apply_componentwise(domain, target.get_der_t(lane), transform));

            return result;
        }
    } // namespace ope_tensor_unary_lane_detail
} // namespace Kadath
