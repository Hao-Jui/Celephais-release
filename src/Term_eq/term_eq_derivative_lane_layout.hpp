#pragma once

#include "For_Kadath/Term_eq/term_eq.hpp"

namespace Kadath::derivative_lane_detail
{
    inline bool same_tensor_component_layout(const Tensor& lhs, const Tensor& rhs)
    {
        if (&lhs.get_space() != &rhs.get_space() || lhs.get_valence() != rhs.get_valence() ||
            lhs.get_n_comp() != rhs.get_n_comp() || lhs.get_ndim() != rhs.get_ndim() ||
            !(lhs.get_basis() == rhs.get_basis())) {
            return false;
        }

        for (int index = 0; index < lhs.get_valence(); ++index) {
            if (lhs.get_index_type(index) != rhs.get_index_type(index))
                return false;
        }

        for (int component = 0; component < lhs.get_n_comp(); ++component) {
            const Array<int> lhs_indices(lhs.indices(component));
            const Array<int> rhs_indices(rhs.indices(component));
            for (int index = 0; index < lhs.get_valence(); ++index) {
                if (lhs_indices(index) != rhs_indices(index))
                    return false;
            }
        }
        return true;
    }

    inline bool all_derivative_lanes_match_value_layout(const Term_eq& source)
    {
        const Tensor& value = source.get_val_t();
        for (int lane = 0; lane < source.get_derivative_lane_count(); ++lane) {
            const Tensor* derivative = source.get_p_der_t(lane);
            if (derivative != nullptr && !same_tensor_component_layout(value, *derivative))
                return false;
        }
        return true;
    }
} // namespace Kadath::derivative_lane_detail
