/*
    Copyright 2017 Philippe Grandclement

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 * Modifications (Celephais):
 *   2026-08-09  Reduced scalar product-rule temporaries.
 *   2026-08-10  Preserved logical-zero bases in direct scalar sums.
 */

#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "term_eq_derivative_lane_layout.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <span>
#include <utility>

namespace Kadath
{
    namespace
    {
        double integer_power(double value, int exponent)
        {
            assert(exponent >= 0);
            double result = 1.0;
            double base = value;
            while (exponent > 0) {
                if ((exponent & 1) != 0)
                    result *= base;
                exponent >>= 1;
                if (exponent > 0)
                    base *= base;
            }
            return result;
        }

        Tensor tensor_power_one_dom(int dom, const Tensor& value, int exponent)
        {
            assert(exponent > 0);
            Tensor result(mult_one_dom(dom, value, 1));
            for (int i = 1; i < exponent; ++i)
                result = mult_one_dom(dom, result, value);
            return result;
        }

        Term_eq repeated_integer_power(const Term_eq& so, int exponent)
        {
            if (exponent > 0) {
                Term_eq res(so);
                for (int i = 1; i < exponent; i++)
                    res = res * so;
                return res;
            } else {
                Scalar one_scal(so.get_val_t(), false);
                one_scal = 1.;
                one_scal.std_base();
                Scalar zero_scal(one_scal);
                zero_scal = 0.;
                Term_eq one(so.get_dom(), one_scal, zero_scal);
                Term_eq res(one);
                for (int i = 0; i < -exponent; i++)
                    res = res / so;
                return res;
            }
        }

        Term_eq constant_one_like(const Term_eq& so)
        {
            if (so.get_type_data() == TERM_D) {
                return Term_eq(so.get_dom(), 1.0, 0.0);
            }

            Scalar one_scal(so.get_val_t(), false);
            one_scal = 1.;
            one_scal.std_base();
            Scalar zero_scal(one_scal);
            zero_scal = 0.;
            return Term_eq(so.get_dom(), one_scal, zero_scal);
        }

        int merged_derivative_lane_count(const Term_eq& lhs, const Term_eq& rhs)
        {
            return std::max(lhs.get_derivative_lane_count(), rhs.get_derivative_lane_count());
        }

        double derivative_or_zero(const Term_eq& term, int lane)
        {
            return term.has_der_d(lane) ? term.get_der_d(lane) : 0.0;
        }

        Tensor copy_tensor_derivative_domain(const Term_eq& term, int lane)
        {
            return Tensor(one_domain_storage, term.get_dom(),
                          term.get_der_t(lane));
        }

        bool tensor_domain_is_zero(const Tensor& tensor, int domain)
        {
            for (int component = 0; component < tensor.get_n_comp(); ++component) {
                Array<int> index(tensor.indices(component));
                if (!tensor(index)(domain).check_if_zero())
                    return false;
            }
            return true;
        }

        void zero_tensor_domain(Tensor& tensor, int domain)
        {
            for (int component = 0; component < tensor.get_n_comp(); ++component) {
                Array<int> index(tensor.indices(component));
                tensor.set(index).set_domain(domain).set_zero();
            }
        }

        void clear_tensor_parameters(Tensor& tensor)
        {
            delete tensor.set_parameters();
            tensor.set_parameters() = nullptr;
        }

        void set_tensor_m_quant_only(Tensor& tensor, int m_quant)
        {
            clear_tensor_parameters(tensor);
            if (m_quant == 0)
                return;

            tensor.affect_parameters();
            tensor.set_parameters()->set_m_quant() = m_quant;
        }

        bool tensor_component_layout_and_names_match(const Tensor& lhs, const Tensor& rhs)
        {
            if (!derivative_lane_detail::same_tensor_component_layout(lhs, rhs) ||
                lhs.is_name_affected() != rhs.is_name_affected()) {
                return false;
            }
            if (!lhs.is_name_affected())
                return true;

            for (int index = 0; index < lhs.get_valence(); ++index) {
                if (lhs.get_name_ind()[index] != rhs.get_name_ind()[index])
                    return false;
            }
            return true;
        }

        bool tensor_uses_standard_component_layout(const Tensor& tensor)
        {
            int standard_component_count = 1;
            for (int index = 0; index < tensor.get_valence(); ++index)
                standard_component_count *= tensor.get_ndim();
            if (tensor.get_n_comp() != standard_component_count)
                return false;

            for (int component = 0; component < tensor.get_n_comp(); ++component) {
                const Array<int> indices(tensor.indices(component));
                int place = component;
                for (int index = tensor.get_valence() - 1; index >= 0; --index) {
                    if (indices(index) != place % tensor.get_ndim() + 1)
                        return false;
                    place /= tensor.get_ndim();
                }
            }
            return true;
        }

        void set_sum_parameters(Tensor& result, const Param_tensor* lhs, const Param_tensor* rhs)
        {
            const int m_quant = add_m_quant(lhs, rhs);
            set_tensor_m_quant_only(result, m_quant);
        }

        void set_product_sum_parameters(Tensor& result,
                                        const Param_tensor* rhs_lhs,
                                        const Param_tensor* rhs_rhs)
        {
            const Param_tensor* lhs_parameters = result.get_parameters();
            const int rhs_m_quant = mult_m_quant(rhs_lhs, rhs_rhs);
            int result_m_quant = 0;
            if (lhs_parameters != nullptr && rhs_m_quant != 0) {
                result_m_quant = std::min(lhs_parameters->get_m_quant(), rhs_m_quant);
            }
            set_tensor_m_quant_only(result, result_m_quant);
        }

        bool accumulate_scaled_tensor_domain(Tensor& result,
                                             const Tensor& source,
                                             int domain,
                                             double scalar,
                                             bool scalar_on_left,
                                             bool subtract)
        {
            if (!tensor_component_layout_and_names_match(result, source))
                return false;

            for (int component = 0; component < result.get_n_comp(); ++component) {
                const Array<int> index(result.indices(component));
                Val_domain& destination = result.set(index).set_domain(domain);
                const Val_domain& source_value = source(index)(domain);
                const Val_domain scaled = scalar_on_left ? scalar * source_value
                                                         : source_value * scalar;
                if (subtract) {
                    if (destination.check_if_zero())
                        destination = -scaled;
                    else
                        destination -= scaled;
                } else {
                    destination += scaled;
                }
            }
            return true;
        }

        bool accumulate_tensor_domain(Tensor& result, const Tensor& source, int domain, bool subtract)
        {
            if (!tensor_component_layout_and_names_match(result, source))
                return false;

            for (int component = 0; component < result.get_n_comp(); ++component) {
                const Array<int> index(result.indices(component));
                Val_domain& destination = result.set(index).set_domain(domain);
                destination = subtract ? destination - source(index)(domain)
                                       : destination + source(index)(domain);
            }
            return true;
        }

        bool accumulate_tensor_product_domain(Tensor& result,
                                              const Tensor& lhs,
                                              const Tensor& rhs,
                                              int domain,
                                              bool subtract)
        {
            if (lhs.get_valence() == 0) {
                if (!tensor_component_layout_and_names_match(result, rhs))
                    return false;

                const Val_domain& scalar = lhs()(domain);
                for (int component = 0; component < result.get_n_comp(); ++component) {
                    const Array<int> index(result.indices(component));
                    Val_domain& destination = result.set(index).set_domain(domain);
                    const Val_domain product = scalar * rhs(index)(domain);
                    if (subtract) {
                        if (destination.check_if_zero())
                            destination = -product;
                        else
                            destination -= product;
                    } else {
                        destination += product;
                    }
                }
                return true;
            }
            if (rhs.get_valence() != 0 ||
                !tensor_component_layout_and_names_match(result, lhs)) {
                return false;
            }

            const Val_domain& scalar = rhs()(domain);
            for (int component = 0; component < result.get_n_comp(); ++component) {
                const Array<int> index(result.indices(component));
                Val_domain& destination = result.set(index).set_domain(domain);
                const Val_domain product = lhs(index)(domain) * scalar;
                if (subtract) {
                    if (destination.check_if_zero())
                        destination = -product;
                    else
                        destination -= product;
                } else {
                    destination += product;
                }
            }
            return true;
        }

        bool divide_tensor_domain_by_square(Tensor& numerator,
                                            const Val_domain& denominator_square,
                                            int domain)
        {
            if (!tensor_uses_standard_component_layout(numerator))
                return false;

            for (int component = 0; component < numerator.get_n_comp(); ++component) {
                const Array<int> index(numerator.indices(component));
                Val_domain& destination = numerator.set(index).set_domain(domain);
                destination = destination / denominator_square;
            }
            clear_tensor_parameters(numerator);
            return true;
        }

        bool divide_tensor_domain_by_square(Tensor& numerator, double denominator_square, int domain)
        {
            if (!tensor_uses_standard_component_layout(numerator))
                return false;

            for (int component = 0; component < numerator.get_n_comp(); ++component) {
                const Array<int> index(numerator.indices(component));
                Val_domain& destination = numerator.set(index).set_domain(domain);
                destination = destination / denominator_square;
            }
            return true;
        }

        // Read-only operand view onto one derivative lane. A present lane is
        // borrowed; a zero tensor is materialized only when the lane is absent.
        // Every consumer takes const Tensor&, so this is the same contract the
        // primary derivative already uses when it hands *aa.der_t straight to
        // those helpers - the borrow just extends it to the packed lanes.
        // Construct only as a temporary in the consumer's argument list: the
        // borrow stays live to the end of that full expression.
        class Derivative_lane_operand
        {
          public:
            Derivative_lane_operand(const Term_eq& term, int lane)
            {
                if (term.has_der_t(lane)) {
                    borrowed = &term.get_der_t(lane);
                    return;
                }
                materialized.emplace(one_domain_storage, term.get_dom(), term.get_val_t(), false);
                zero_tensor_domain(*materialized, term.get_dom());
                borrowed = &(*materialized);
            }

            Derivative_lane_operand(const Derivative_lane_operand&) = delete;
            Derivative_lane_operand& operator=(const Derivative_lane_operand&) = delete;

            operator const Tensor&() const { return *borrowed; }

          private:
            std::optional<Tensor> materialized;
            const Tensor* borrowed;
        };

        template <typename Formula>
        void propagate_tensor_lanes(Term_eq& result, const Term_eq& lhs, const Term_eq& rhs, Formula formula)
        {
            const int lanes = merged_derivative_lane_count(lhs, rhs);
            result.set_derivative_lane_count(lanes);
            for (int lane = 1; lane < lanes; ++lane) {
                result.set_der_t(lane, formula(lane));
            }
        }

        template <typename Formula>
        void propagate_unary_tensor_lanes(Term_eq& result, const Term_eq& source, Formula formula)
        {
            const int lanes = source.get_derivative_lane_count();
            result.set_derivative_lane_count(lanes);
            for (int lane = 1; lane < lanes; ++lane) {
                if (source.has_der_t(lane))
                    result.set_der_t(lane, formula(lane));
            }
        }

        template <typename Formula>
        void propagate_unary_double_lanes(Term_eq& result, const Term_eq& source, Formula formula)
        {
            const int lanes = source.get_derivative_lane_count();
            result.set_derivative_lane_count(lanes);
            for (int lane = 1; lane < lanes; ++lane) {
                if (source.has_der_d(lane))
                    result.set_der_d(lane, formula(lane));
            }
        }

        void prepare_absolute_derivative_lanes(const Term_eq& source)
        {
            if (!Val_domain::derivative_lane_tiling_enabled() ||
                source.get_p_der_t() == nullptr)
                return;
            if (!derivative_lane_detail::all_derivative_lanes_match_value_layout(source))
                return;

            const Tensor& value = source.get_val_t();
            for (int component = 0; component < value.get_n_comp(); ++component) {
                std::array<const Val_domain*, Term_eq::max_derivative_lanes> fields{};
                std::size_t field_count = 0;
                for (int lane = 0; lane < source.get_derivative_lane_count(); ++lane) {
                    const Tensor* derivative = source.get_p_der_t(lane);
                    if (derivative == nullptr)
                        continue;
                    const Array<int> index(derivative->indices(component));
                    fields[field_count++] = &(*derivative)(index)(source.get_dom());
                }
                Val_domain::prepare_der_abs_batch(
                    std::span<const Val_domain* const>(fields.data(), field_count));
            }
        }

        void propagate_double_sum_lanes(Term_eq& result, const Term_eq& lhs, const Term_eq& rhs, double rhs_sign)
        {
            const int lanes = merged_derivative_lane_count(lhs, rhs);
            result.set_derivative_lane_count(lanes);
            for (int lane = 1; lane < lanes; ++lane) {
                result.set_der_d(lane, derivative_or_zero(lhs, lane) + rhs_sign * derivative_or_zero(rhs, lane));
            }
        }

        void propagate_double_product_lanes(Term_eq& result, const Term_eq& lhs, const Term_eq& rhs)
        {
            const int lanes = merged_derivative_lane_count(lhs, rhs);
            result.set_derivative_lane_count(lanes);
            for (int lane = 1; lane < lanes; ++lane) {
                result.set_der_d(lane, derivative_or_zero(lhs, lane) * rhs.get_val_d() +
                                           lhs.get_val_d() * derivative_or_zero(rhs, lane));
            }
        }

        void propagate_double_quotient_lanes(Term_eq& result, const Term_eq& lhs, const Term_eq& rhs)
        {
            const int lanes = merged_derivative_lane_count(lhs, rhs);
            result.set_derivative_lane_count(lanes);
            const double denominator = rhs.get_val_d() * rhs.get_val_d();
            for (int lane = 1; lane < lanes; ++lane) {
                result.set_der_d(lane, (derivative_or_zero(lhs, lane) * rhs.get_val_d() -
                                        lhs.get_val_d() * derivative_or_zero(rhs, lane)) /
                                           denominator);
            }
        }

        Tensor tensor_scalar_product_derivative(
            int domain,
            const Tensor& tensor_value,
            const Tensor& tensor_derivative,
            double scalar_value,
            double scalar_derivative,
            bool simplify_zeros = true)
        {
            if (simplify_zeros && scalar_derivative == 0.0)
                return mult_one_dom(domain, tensor_derivative, scalar_value);

            Tensor result(mult_one_dom(domain, tensor_derivative, scalar_value));
            if (!accumulate_scaled_tensor_domain(
                    result, tensor_value, domain, scalar_derivative, false, false)) {
                return add_one_dom(domain,
                                   result,
                                   mult_one_dom(domain, tensor_value, scalar_derivative));
            }
            set_sum_parameters(result, tensor_derivative.get_parameters(), tensor_value.get_parameters());
            return result;
        }

        Tensor scalar_tensor_product_derivative(
            int domain,
            double scalar_value,
            double scalar_derivative,
            const Tensor& tensor_value,
            const Tensor& tensor_derivative,
            bool simplify_zeros = true)
        {
            if (simplify_zeros && scalar_derivative == 0.0)
                return mult_one_dom(domain, scalar_value, tensor_derivative);

            Tensor result(mult_one_dom(domain, scalar_derivative, tensor_value));
            if (!accumulate_scaled_tensor_domain(
                    result, tensor_derivative, domain, scalar_value, true, false)) {
                return add_one_dom(domain,
                                   result,
                                   mult_one_dom(domain, scalar_value, tensor_derivative));
            }
            set_sum_parameters(result, tensor_value.get_parameters(), tensor_derivative.get_parameters());
            return result;
        }

        Tensor tensor_tensor_product_derivative(
            int domain,
            const Tensor& lhs_value,
            const Tensor& lhs_derivative,
            const Tensor& rhs_value,
            const Tensor& rhs_derivative,
            bool simplify_zeros = true)
        {
            const bool lhs_derivative_is_zero = simplify_zeros && tensor_domain_is_zero(lhs_derivative, domain);
            const bool rhs_derivative_is_zero = simplify_zeros && tensor_domain_is_zero(rhs_derivative, domain);

            if (lhs_derivative_is_zero && rhs_derivative_is_zero) {
                Tensor zero_product(mult_one_dom(domain, lhs_value, rhs_value));
                zero_tensor_domain(zero_product, domain);
                return zero_product;
            }
            if (lhs_derivative_is_zero)
                return mult_one_dom(domain, lhs_value, rhs_derivative);
            if (rhs_derivative_is_zero)
                return mult_one_dom(domain, lhs_derivative, rhs_value);

            Tensor result(mult_one_dom(domain, lhs_derivative, rhs_value));
            if (accumulate_tensor_product_domain(
                    result, lhs_value, rhs_derivative, domain, false)) {
                set_product_sum_parameters(
                    result, lhs_value.get_parameters(), rhs_derivative.get_parameters());
                return result;
            }

            Tensor rhs_product(mult_one_dom(domain, lhs_value, rhs_derivative));
            if (!accumulate_tensor_domain(result, rhs_product, domain, false))
                return add_one_dom(domain, result, rhs_product);
            set_sum_parameters(result, result.get_parameters(), rhs_product.get_parameters());
            return result;
        }

        Tensor scalar_tensor_quotient_numerator(
            int domain,
            double scalar_value,
            double scalar_derivative,
            const Tensor& tensor_value,
            const Tensor& tensor_derivative,
            bool simplify_zeros = true)
        {
            const bool tensor_derivative_is_zero = simplify_zeros && tensor_domain_is_zero(tensor_derivative, domain);
            if (simplify_zeros && scalar_derivative == 0.0 && tensor_derivative_is_zero) {
                Tensor zero_numerator(
                    one_domain_storage, domain, tensor_value, false);
                zero_tensor_domain(zero_numerator, domain);
                return zero_numerator;
            }
            if (simplify_zeros && scalar_derivative == 0.0)
                return mult_one_dom(domain, -scalar_value, tensor_derivative);
            if (tensor_derivative_is_zero)
                return mult_one_dom(domain, scalar_derivative, tensor_value);

            Tensor result(mult_one_dom(domain, scalar_derivative, tensor_value));
            if (!accumulate_scaled_tensor_domain(
                    result, tensor_derivative, domain, scalar_value, true, true)) {
                return sub_one_dom(domain,
                                   result,
                                   mult_one_dom(domain, scalar_value, tensor_derivative));
            }
            set_sum_parameters(result, tensor_value.get_parameters(), tensor_derivative.get_parameters());
            return result;
        }

        Tensor tensor_scalar_quotient_numerator(
            int domain,
            const Tensor& tensor_value,
            const Tensor& tensor_derivative,
            double scalar_value,
            double scalar_derivative,
            bool simplify_zeros = true)
        {
            const bool tensor_derivative_is_zero = simplify_zeros && tensor_domain_is_zero(tensor_derivative, domain);
            if (simplify_zeros && tensor_derivative_is_zero && scalar_derivative == 0.0) {
                Tensor zero_numerator(
                    one_domain_storage, domain, tensor_value, false);
                zero_tensor_domain(zero_numerator, domain);
                return zero_numerator;
            }
            if (tensor_derivative_is_zero)
                return mult_one_dom(domain, tensor_value, -scalar_derivative);
            if (simplify_zeros && scalar_derivative == 0.0)
                return mult_one_dom(domain, tensor_derivative, scalar_value);

            Tensor result(mult_one_dom(domain, tensor_derivative, scalar_value));
            if (!accumulate_scaled_tensor_domain(
                    result, tensor_value, domain, scalar_derivative, false, true)) {
                return sub_one_dom(domain,
                                   result,
                                   mult_one_dom(domain, tensor_value, scalar_derivative));
            }
            set_sum_parameters(result, tensor_derivative.get_parameters(), tensor_value.get_parameters());
            return result;
        }

        Tensor tensor_tensor_quotient_numerator(
            int domain,
            const Tensor& lhs_value,
            const Tensor& lhs_derivative,
            const Tensor& rhs_value,
            const Tensor& rhs_derivative,
            bool simplify_zeros = true)
        {
            const bool lhs_derivative_is_zero = simplify_zeros && tensor_domain_is_zero(lhs_derivative, domain);
            const bool rhs_derivative_is_zero = simplify_zeros && tensor_domain_is_zero(rhs_derivative, domain);
            if (lhs_derivative_is_zero && rhs_derivative_is_zero) {
                Tensor zero_numerator(mult_one_dom(domain, lhs_value, rhs_value));
                zero_tensor_domain(zero_numerator, domain);
                return zero_numerator;
            }
            if (lhs_derivative_is_zero)
                return mult_one_dom(domain, mult_one_dom(domain, lhs_value, rhs_derivative), -1.0);
            if (rhs_derivative_is_zero)
                return mult_one_dom(domain, lhs_derivative, rhs_value);

            Tensor result(mult_one_dom(domain, lhs_derivative, rhs_value));
            if (accumulate_tensor_product_domain(
                    result, lhs_value, rhs_derivative, domain, true)) {
                set_product_sum_parameters(
                    result, lhs_value.get_parameters(), rhs_derivative.get_parameters());
                return result;
            }

            Tensor rhs_product(mult_one_dom(domain, lhs_value, rhs_derivative));
            if (!accumulate_tensor_domain(result, rhs_product, domain, true))
                return sub_one_dom(domain, result, rhs_product);
            set_sum_parameters(result, result.get_parameters(), rhs_product.get_parameters());
            return result;
        }

        Tensor scalar_tensor_quotient_derivative(
            int domain,
            double scalar_value,
            double scalar_derivative,
            const Tensor& tensor_value,
            const Tensor& tensor_derivative,
            const Val_domain& denominator_square,
            bool simplify_zeros = true)
        {
            Tensor numerator(scalar_tensor_quotient_numerator(
                domain, scalar_value, scalar_derivative, tensor_value, tensor_derivative, simplify_zeros));
            if (divide_tensor_domain_by_square(numerator, denominator_square, domain))
                return numerator;
            Tensor denominator_square_tensor(
                one_domain_storage, domain, tensor_value, false);
            denominator_square_tensor.set().set_domain(domain) = denominator_square;
            return div_one_dom(domain, numerator, denominator_square_tensor);
        }

        Tensor tensor_scalar_quotient_derivative(
            int domain,
            const Tensor& tensor_value,
            const Tensor& tensor_derivative,
            double scalar_value,
            double scalar_derivative,
            double denominator_square,
            bool simplify_zeros = true)
        {
            Tensor numerator(tensor_scalar_quotient_numerator(
                domain, tensor_value, tensor_derivative, scalar_value, scalar_derivative, simplify_zeros));
            if (divide_tensor_domain_by_square(numerator, denominator_square, domain))
                return numerator;
            return div_one_dom(domain, numerator, denominator_square);
        }

        Tensor tensor_tensor_quotient_derivative(
            int domain,
            const Tensor& lhs_value,
            const Tensor& lhs_derivative,
            const Tensor& rhs_value,
            const Tensor& rhs_derivative,
            const Val_domain& denominator_square,
            bool simplify_zeros = true)
        {
            Tensor numerator(tensor_tensor_quotient_numerator(
                domain, lhs_value, lhs_derivative, rhs_value, rhs_derivative, simplify_zeros));
            if (divide_tensor_domain_by_square(numerator, denominator_square, domain))
                return numerator;
            Tensor denominator_square_tensor(
                one_domain_storage, domain, rhs_value, false);
            denominator_square_tensor.set().set_domain(domain) = denominator_square;
            return div_one_dom(domain, numerator, denominator_square_tensor);
        }

        bool is_scalar_sum_tensor(const Tensor& tensor)
        {
            return tensor.get_valence() == 0 && tensor.get_n_comp() == 1;
        }

        bool scalar_sum_tensor_matches(const Tensor& tensor, const Tensor& reference)
        {
            return is_scalar_sum_tensor(tensor) &&
                tensor_component_layout_and_names_match(tensor, reference);
        }

        void consume_scalar_sum_domain(Val_domain& lhs,
                                       const Val_domain& rhs,
                                       bool subtract)
        {
            // The rvalue subtraction overload has a deliberate logical-zero
            // branch that raw operator-= does not reproduce. Returning through
            // move assignment consumes nonzero lhs storage while retaining the
            // legacy zero/base behaviour for both operations.
            if (subtract)
                lhs = std::move(lhs) - rhs;
            else
                lhs = std::move(lhs) + rhs;
        }

        void consume_missing_scalar_sum_domain(Val_domain& lhs,
                                               const Val_domain& rhs_value,
                                               bool subtract)
        {
            if (!lhs.check_if_zero())
                return;

            // Derivative_lane_operand would build this same logical zero from
            // rhs_value. In supported spaces (at most four dimensions),
            // Val_domain's derivative-pointer storage is inline, so reproducing
            // that state avoids a Tensor/Scalar shell and MemoryMapper traffic.
            Val_domain rhs_zero(rhs_value, false);
            rhs_zero.set_zero();
            consume_scalar_sum_domain(lhs, rhs_zero, subtract);
        }

        void write_scalar_tensor_sum_domain(Tensor& destination,
                                            const Tensor& lhs,
                                            const Tensor& rhs,
                                            int domain,
                                            bool subtract)
        {
            Val_domain& destination_domain = destination.set().set_domain(domain);
            destination_domain = lhs()(domain);
            consume_scalar_sum_domain(destination_domain, rhs()(domain), subtract);
            set_sum_parameters(destination, lhs.get_parameters(), rhs.get_parameters());
        }
    } // namespace

    bool Term_eq::try_write_sum_action_result(const Term_eq& lhs,
                                              const Term_eq& rhs,
                                              bool subtract)
    {
        // An action target can also be an exact Ope_id leaf. Refuse before
        // touching it so both operands remain readable through the fallback.
        if (this == &lhs || this == &rhs)
            return false;
        if (dom != lhs.dom || dom != rhs.dom ||
            type_data != lhs.type_data || type_data != rhs.type_data)
            return false;

        const int lanes = merged_derivative_lane_count(lhs, rhs);
        if (type_data == TERM_D) {
            if (val_d == nullptr || lhs.val_d == nullptr || rhs.val_d == nullptr)
                return false;

            *val_d = subtract ? *lhs.val_d - *rhs.val_d : *lhs.val_d + *rhs.val_d;
            if (lhs.der_d != nullptr && rhs.der_d != nullptr) {
                const double derivative = subtract
                    ? *lhs.der_d - *rhs.der_d
                    : *lhs.der_d + *rhs.der_d;
                set_der_d(derivative);
            } else {
                clear_der(0);
            }

            set_derivative_lane_count(lanes);
            const double rhs_sign = subtract ? -1.0 : 1.0;
            for (int lane = 1; lane < lanes; ++lane) {
                // Match propagate_double_sum_lanes exactly, including the
                // multiply-before-add order used by subtraction.
                set_der_d(lane,
                          derivative_or_zero(lhs, lane) +
                              rhs_sign * derivative_or_zero(rhs, lane));
            }
            return true;
        }

        if (type_data != TERM_T || val_t == nullptr || lhs.val_t == nullptr || rhs.val_t == nullptr ||
            !is_scalar_sum_tensor(*val_t) ||
            !scalar_sum_tensor_matches(*lhs.val_t, *val_t) ||
            !scalar_sum_tensor_matches(*rhs.val_t, *val_t)) {
            return false;
        }

        const bool do_primary_derivative = lhs.der_t != nullptr && rhs.der_t != nullptr;
        if (do_primary_derivative &&
            (!scalar_sum_tensor_matches(*lhs.der_t, *val_t) ||
             !scalar_sum_tensor_matches(*rhs.der_t, *val_t) ||
             (der_t != nullptr && !scalar_sum_tensor_matches(*der_t, *val_t)))) {
            return false;
        }

        // Preflight every packed source and active destination lane before the
        // value write. A false return therefore never leaves a partial result.
        for (int lane = 1; lane < lanes; ++lane) {
            const Tensor* lhs_lane = lhs.get_p_der_t(lane);
            const Tensor* rhs_lane = rhs.get_p_der_t(lane);
            const Tensor* destination_lane = get_p_der_t(lane);
            if ((lhs_lane != nullptr && !scalar_sum_tensor_matches(*lhs_lane, *val_t)) ||
                (rhs_lane != nullptr && !scalar_sum_tensor_matches(*rhs_lane, *val_t)) ||
                (destination_lane != nullptr &&
                 !scalar_sum_tensor_matches(*destination_lane, *val_t))) {
                return false;
            }
        }

        write_scalar_tensor_sum_domain(*val_t, *lhs.val_t, *rhs.val_t, dom, subtract);

        if (do_primary_derivative) {
            if (der_t == nullptr)
                set_der_t(*lhs.der_t);
            write_scalar_tensor_sum_domain(*der_t, *lhs.der_t, *rhs.der_t, dom, subtract);
        } else {
            clear_der(0);
        }

        set_derivative_lane_count(lanes);
        for (int lane = 1; lane < lanes; ++lane) {
            const Tensor* lhs_lane = lhs.get_p_der_t(lane);
            const Tensor* rhs_lane = rhs.get_p_der_t(lane);
            const Tensor& lhs_storage = lhs_lane == nullptr ? *lhs.val_t : *lhs_lane;
            const Tensor& rhs_storage = rhs_lane == nullptr ? *rhs.val_t : *rhs_lane;

            set_der_t(lane, lhs_storage);
            Tensor* destination_lane = set_der_t(lane);
            assert(destination_lane != nullptr);
            if (lhs_lane == nullptr)
                destination_lane->set().set_domain(dom).set_zero();

            Val_domain& destination_domain = destination_lane->set().set_domain(dom);
            if (rhs_lane != nullptr) {
                consume_scalar_sum_domain(
                    destination_domain, (*rhs_lane)()(dom), subtract);
            } else {
                consume_missing_scalar_sum_domain(
                    destination_domain, (*rhs.val_t)()(dom), subtract);
            }
            set_sum_parameters(
                *destination_lane, lhs_storage.get_parameters(), rhs_storage.get_parameters());
        }
        return true;
    }

    Term_eq operator+(const Term_eq& aa, double xx)
    {
        Term_eq auxi(aa.dom, xx, 0.);
        return aa + auxi;
    }

    Term_eq operator+(double xx, const Term_eq& aa)
    {
        Term_eq auxi(aa.dom, xx, 0.);
        return aa + auxi;
    }

    Term_eq operator-(const Term_eq& aa)
    {
        return (-1. * aa);
    }

    Term_eq Term_eq::der_abs(int i) const
    {

        Term_eq res(dom, type_data);
        Index pos(*val_t);
        switch (type_data) {
            case TERM_D:
                KADATH_THROW("derivative only defined with respect to tensors");
            case TERM_T:
                prepare_absolute_derivative_lanes(*this);
                res.val_t = new Tensor(
                    one_domain_storage, dom, *val_t, false);
                if (der_t != nullptr)
                    res.der_t = new Tensor(
                        one_domain_storage, dom, *der_t, false);
                do {
                    res.val_t->set(pos).set_domain(dom) = (*val_t)(pos)(dom).der_abs(i);
                    if (der_t != nullptr)
                        res.der_t->set(pos).set_domain(dom) = (*der_t)(pos)(dom).der_abs(i);
                } while (pos.inc());
                if (der_t != nullptr)
                    propagate_unary_tensor_lanes(res, *this, [&](int lane) {
                        // The only site that needs an owned copy: the loop below
                        // overwrites each component of it in place.
                        Tensor derivative(copy_tensor_derivative_domain(*this, lane));
                        Index derivative_position(derivative);
                        do {
                            derivative.set(derivative_position).set_domain(dom) =
                                derivative(derivative_position)(dom).der_abs(i);
                        } while (derivative_position.inc());
                        return derivative;
                    });
                break;
            default:
                KADATH_THROW("Unknown data storage in operator+");
        }
        return res;
    }

    Term_eq operator+(const Term_eq& aa, const Term_eq& bb)
    {

        bool do_der = true;

        int data_res;
        assert(aa.dom == bb.dom);
        switch (aa.type_data) {
            case (TERM_D):
                if (aa.der_d == nullptr)
                    do_der = false;
                switch (bb.type_data) {
                    case (TERM_D):
                        data_res = TERM_D;
                        if (bb.der_d == nullptr)
                            do_der = false;
                        break;
                    case (TERM_T):
                        data_res = TERM_T;
                        if (bb.der_t == nullptr)
                            do_der = false;
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator+");
                }
                break;
            case (TERM_T):
                if (aa.der_t == nullptr)
                    do_der = false;
                switch (bb.type_data) {
                    case (TERM_D):
                        if (bb.der_d == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    case (TERM_T):
                        if (bb.der_t == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator+");
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator+");
        }

        Term_eq res(aa.dom, data_res);
        switch (aa.type_data) {
            case (TERM_D):
                switch (bb.type_data) {
                    case (TERM_D):
                        res.val_d = new double(*aa.val_d + *bb.val_d);
                        res.der_d = (do_der) ? new double(*aa.der_d + *bb.der_d) : nullptr;
                        propagate_double_sum_lanes(res, aa, bb, 1.0);
                        break;
                    case (TERM_T):
                        res.val_t = new Tensor(add_one_dom(res.dom, *aa.val_d, *bb.val_t));
                        res.der_t = (do_der) ? new Tensor(add_one_dom(res.dom, *aa.der_d, *bb.der_t)) : nullptr;
                        propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                            return add_one_dom(res.dom,
                                               derivative_or_zero(aa, lane),
                                               Derivative_lane_operand(bb, lane));
                        });
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator+");
                }
                break;
            case (TERM_T):
                switch (bb.type_data) {
                    case (TERM_D):
                        res.val_t = new Tensor(add_one_dom(res.dom, *aa.val_t, *bb.val_d));
                        res.der_t = (do_der) ? new Tensor(add_one_dom(res.dom, *aa.der_t, *bb.der_d)) : nullptr;
                        propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                            return add_one_dom(res.dom,
                                               Derivative_lane_operand(aa, lane),
                                               derivative_or_zero(bb, lane));
                        });
                        break;
                    case (TERM_T):
                        res.val_t = new Tensor(add_one_dom(res.dom, *aa.val_t, *bb.val_t));
                        res.der_t = (do_der) ? new Tensor(add_one_dom(res.dom, *aa.der_t, *bb.der_t)) : nullptr;
                        propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                            return add_one_dom(res.dom,
                                               Derivative_lane_operand(aa, lane),
                                               Derivative_lane_operand(bb, lane));
                        });
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator+");
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator+");
        }
        return res;
    }

    Term_eq operator-(const Term_eq& aa, const Term_eq& bb)
    {
        assert(aa.dom == bb.dom);
        int data_res;
        bool do_der = true;
        switch (aa.type_data) {
            case (TERM_D):
                if (aa.der_d == nullptr)
                    do_der = false;
                switch (bb.type_data) {
                    case (TERM_D):
                        if (bb.der_d == nullptr)
                            do_der = false;
                        data_res = TERM_D;
                        break;
                    case (TERM_T):
                        if (bb.der_t == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator-");
                }
                break;
            case (TERM_T):
                if (aa.der_t == nullptr)
                    do_der = false;
                switch (bb.type_data) {
                    case (TERM_D):
                        if (bb.der_d == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    case (TERM_T):
                        if (bb.der_t == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator-");
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator-");
        }

        Term_eq res(aa.dom, data_res);
        switch (aa.type_data) {
            case (TERM_D):
                switch (bb.type_data) {
                    case (TERM_D):
                        res.val_d = new double(*aa.val_d - *bb.val_d);
                        res.der_d = (do_der) ? new double(*aa.der_d - *bb.der_d) : nullptr;
                        propagate_double_sum_lanes(res, aa, bb, -1.0);
                        break;
                    case (TERM_T):
                        res.val_t = new Tensor(sub_one_dom(res.dom, *aa.val_d, *bb.val_t));
                        res.der_t = (do_der) ? new Tensor(sub_one_dom(res.dom, *aa.der_d, *bb.der_t)) : nullptr;
                        propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                            return sub_one_dom(res.dom,
                                               derivative_or_zero(aa, lane),
                                               Derivative_lane_operand(bb, lane));
                        });
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator+");
                }
                break;
            case (TERM_T):
                switch (bb.type_data) {
                    case (TERM_D):
                        res.val_t = new Tensor(sub_one_dom(res.dom, *aa.val_t, *bb.val_d));
                        res.der_t = (do_der) ? new Tensor(sub_one_dom(res.dom, *aa.der_t, *bb.der_d)) : nullptr;
                        propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                            return sub_one_dom(res.dom,
                                               Derivative_lane_operand(aa, lane),
                                               derivative_or_zero(bb, lane));
                        });
                        break;
                    case (TERM_T):
                        res.val_t = new Tensor(sub_one_dom(res.dom, *aa.val_t, *bb.val_t));
                        res.der_t = (do_der) ? new Tensor(sub_one_dom(res.dom, *aa.der_t, *bb.der_t)) : nullptr;
                        propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                            return sub_one_dom(res.dom,
                                               Derivative_lane_operand(aa, lane),
                                               Derivative_lane_operand(bb, lane));
                        });
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator+");
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator+");
        }
        return res;
    }

    Term_eq operator*(const Term_eq& aa, const Term_eq& bb)
    {

        assert(aa.dom == bb.dom);
        int data_res;
        bool do_der = true;
        switch (aa.type_data) {
            case (TERM_D):
                if (aa.der_d == nullptr)
                    do_der = false;
                switch (bb.type_data) {
                    case (TERM_D):
                        if (bb.der_d == nullptr)
                            do_der = false;
                        data_res = TERM_D;
                        break;
                    case (TERM_T):
                        if (bb.der_t == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator*");
                }
                break;
            case (TERM_T):
                if (aa.der_t == nullptr)
                    do_der = false;
                switch (bb.type_data) {
                    case (TERM_D):
                        if (bb.der_d == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    case (TERM_T):
                        if (bb.der_t == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator*");
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator*");
        }

        Term_eq res(aa.dom, data_res);
        switch (aa.type_data) {
            case (TERM_D):
                switch (bb.type_data) {
                    case (TERM_D):
                        res.val_d = new double((*aa.val_d) * (*bb.val_d));
                        res.der_d =
                            (do_der) ? new double((*aa.der_d) * (*bb.val_d) + (*aa.val_d) * (*bb.der_d)) : nullptr;
                        propagate_double_product_lanes(res, aa, bb);
                        break;
                    case (TERM_T):
                        res.val_t = new Tensor(mult_one_dom(res.dom, *aa.val_d, *bb.val_t));
                        res.der_t = (do_der)
                                        ? new Tensor(scalar_tensor_product_derivative(res.dom,
                                                                                      *aa.val_d,
                                                                                      *aa.der_d,
                                                                                      *bb.val_t,
                                                                                      *bb.der_t,
                                                                                      false))
                                        : nullptr;
                        propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                            return scalar_tensor_product_derivative(res.dom,
                                                                    *aa.val_d,
                                                                    derivative_or_zero(aa, lane),
                                                                    *bb.val_t,
                                                                    Derivative_lane_operand(bb, lane));
                        });
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator+");
                }
                break;
            case (TERM_T):
                switch (bb.type_data) {
                    case (TERM_D):
                        res.val_t = new Tensor(mult_one_dom(res.dom, *aa.val_t, *bb.val_d));
                        res.der_t = (do_der)
                                        ? new Tensor(tensor_scalar_product_derivative(res.dom,
                                                                                      *aa.val_t,
                                                                                      *aa.der_t,
                                                                                      *bb.val_d,
                                                                                      *bb.der_d,
                                                                                      false))
                                        : nullptr;
                        propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                            return tensor_scalar_product_derivative(res.dom,
                                                                    *aa.val_t,
                                                                    Derivative_lane_operand(aa, lane),
                                                                    *bb.val_d,
                                                                    derivative_or_zero(bb, lane));
                        });
                        break;
                    case (TERM_T):
                        res.val_t = new Tensor(mult_one_dom(res.dom, *aa.val_t, *bb.val_t));
                        res.der_t = (do_der)
                                        ? new Tensor(tensor_tensor_product_derivative(res.dom,
                                                                                      *aa.val_t,
                                                                                      *aa.der_t,
                                                                                      *bb.val_t,
                                                                                      *bb.der_t,
                                                                                      false))
                                        : nullptr;
                        propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                            return tensor_tensor_product_derivative(res.dom,
                                                                    *aa.val_t,
                                                                    Derivative_lane_operand(aa, lane),
                                                                    *bb.val_t,
                                                                    Derivative_lane_operand(bb, lane));
                        });
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator*");
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator*");
        }

        return res;
    }

    Term_eq operator/(const Term_eq& aa, const Term_eq& bb)
    {

        assert(aa.dom == bb.dom);
        int data_res;
        bool do_der = true;
        switch (aa.type_data) {
            case (TERM_D):
                if (aa.der_d == nullptr)
                    do_der = false;
                switch (bb.type_data) {
                    case (TERM_D):
                        if (bb.der_d == nullptr)
                            do_der = false;
                        data_res = TERM_D;
                        break;
                    case (TERM_T):
                        if (bb.der_t == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator*");
                }
                break;
            case (TERM_T):
                if (aa.der_t == nullptr)
                    do_der = false;
                switch (bb.type_data) {
                    case (TERM_D):
                        if (bb.der_d == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    case (TERM_T):
                        if (bb.der_t == nullptr)
                            do_der = false;
                        data_res = TERM_T;
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator*");
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator*");
        }

        Term_eq res(aa.dom, data_res);
        switch (aa.type_data) {
            case (TERM_D):
                switch (bb.type_data) {
                    case (TERM_D):
                        res.val_d = new double((*aa.val_d) / (*bb.val_d));
                        res.der_d = (do_der) ? new double(((*aa.der_d) * (*bb.val_d) - (*aa.val_d) * (*bb.der_d)) /
                                                          ((*bb.val_d) * (*bb.val_d)))
                                             : nullptr;
                        propagate_double_quotient_lanes(res, aa, bb);
                        break;
                    case (TERM_T): {
                        res.val_t = new Tensor(div_one_dom(res.dom, *aa.val_d, *bb.val_t));
                        if (do_der || merged_derivative_lane_count(aa, bb) > 1) {
                            const Val_domain denominator_square =
                                (*bb.val_t)()(res.dom) * (*bb.val_t)()(res.dom);
                            res.der_t = (do_der) ? new Tensor(scalar_tensor_quotient_derivative(res.dom,
                                                                                               *aa.val_d,
                                                                                               *aa.der_d,
                                                                                               *bb.val_t,
                                                                                               *bb.der_t,
                                                                                               denominator_square,
                                                                                               false))
                                                 : nullptr;
                            propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                                return scalar_tensor_quotient_derivative(res.dom,
                                                                         *aa.val_d,
                                                                         derivative_or_zero(aa, lane),
                                                                         *bb.val_t,
                                                                         Derivative_lane_operand(bb, lane),
                                                                         denominator_square);
                            });
                        }
                        break;
                    }
                    default:
                        KADATH_THROW("Unknown data storage in operator+");
                }
                break;
            case (TERM_T):
                switch (bb.type_data) {
                    case (TERM_D): {
                        res.val_t = new Tensor(div_one_dom(res.dom, *aa.val_t, *bb.val_d));
                        if (do_der || merged_derivative_lane_count(aa, bb) > 1) {
                            const double denominator_square = (*bb.val_d) * (*bb.val_d);
                            res.der_t = (do_der) ? new Tensor(tensor_scalar_quotient_derivative(res.dom,
                                                                                               *aa.val_t,
                                                                                               *aa.der_t,
                                                                                               *bb.val_d,
                                                                                               *bb.der_d,
                                                                                               denominator_square,
                                                                                               false))
                                                 : nullptr;
                            propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                                return tensor_scalar_quotient_derivative(res.dom,
                                                                         *aa.val_t,
                                                                         Derivative_lane_operand(aa, lane),
                                                                         *bb.val_d,
                                                                         derivative_or_zero(bb, lane),
                                                                         denominator_square);
                            });
                        }
                        break;
                    }
                    case (TERM_T): {
                        res.val_t = new Tensor(div_one_dom(res.dom, *aa.val_t, *bb.val_t));
                        if (do_der || merged_derivative_lane_count(aa, bb) > 1) {
                            const Val_domain denominator_square =
                                (*bb.val_t)()(res.dom) * (*bb.val_t)()(res.dom);
                            res.der_t = (do_der) ? new Tensor(tensor_tensor_quotient_derivative(res.dom,
                                                                                               *aa.val_t,
                                                                                               *aa.der_t,
                                                                                               *bb.val_t,
                                                                                               *bb.der_t,
                                                                                               denominator_square,
                                                                                               false))
                                                 : nullptr;
                            propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                                return tensor_tensor_quotient_derivative(res.dom,
                                                                         *aa.val_t,
                                                                         Derivative_lane_operand(aa, lane),
                                                                         *bb.val_t,
                                                                         Derivative_lane_operand(bb, lane),
                                                                         denominator_square);
                            });
                        }
                        break;
                    }
                    default:
                        KADATH_THROW("Unknown data storage in operator+");
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator+");
        }
        return res;
    }

    Term_eq scalar_product(const Term_eq& aa, const Term_eq& bb)
    {

        assert(aa.dom == bb.dom);
        bool do_der = true;
        switch (aa.type_data) {
            case (TERM_D):
                KADATH_THROW("scalar_product only defined with tensors");
                break;
            case (TERM_T):
                if (aa.der_t == nullptr)
                    do_der = false;
                switch (bb.type_data) {
                    case (TERM_D):
                        KADATH_THROW("scalar_product only defined with tensors");
                        break;
                    case (TERM_T):
                        if (bb.der_t == nullptr)
                            do_der = false;
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator*");
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator*");
        }

        // Try to see the parameter :

        bool param_val = false;
        int mval = 0;

        bool param_der = false;
        int mder = 0;
        if (aa.type_data == TERM_T) {

            if (aa.val_t->is_m_quant_affected())
                param_val = true;
            if (bb.val_t->is_m_quant_affected())
                param_val = true;
            if (param_val) {
                if (aa.val_t->is_m_quant_affected())
                    mval += aa.val_t->get_parameters()->get_m_quant();
                if (bb.val_t->is_m_quant_affected())
                    mval += bb.val_t->get_parameters()->get_m_quant();
            }

            if (do_der) {
                if (aa.der_t->is_m_quant_affected())
                    param_der = true;
                if (bb.der_t->is_m_quant_affected())
                    param_der = true;
                if (param_der) {
                    if (aa.der_t->is_m_quant_affected())
                        mder += aa.der_t->get_parameters()->get_m_quant();
                    if (bb.der_t->is_m_quant_affected())
                        mder += bb.der_t->get_parameters()->get_m_quant();
                }
            }
        }

        Term_eq res(aa.dom, TERM_T);
        switch (aa.type_data) {
            case (TERM_D):
                KADATH_THROW("scalar_product only defined with tensors");
                break;
            case (TERM_T):
                switch (bb.type_data) {
                    case (TERM_D):
                        KADATH_THROW("scalar_product only defined with tensors");
                        break;
                    case (TERM_T):
                        res.val_t = new Tensor(scal_one_dom(res.dom, *aa.val_t, *bb.val_t));
                        if (mval != 0) {
                            res.val_t->affect_parameters();
                            res.val_t->set_parameters()->set_m_quant() = mval;
                        }

                        res.der_t = (do_der)
                                        ? new Tensor(add_one_dom(res.dom, scal_one_dom(res.dom, *aa.der_t, *bb.val_t),
                                                                 scal_one_dom(res.dom, *aa.val_t, *bb.der_t)))
                                        : nullptr;
                        propagate_tensor_lanes(res, aa, bb, [&](int lane) {
                            return add_one_dom(
                                res.dom,
                                scal_one_dom(res.dom, Derivative_lane_operand(aa, lane), *bb.val_t),
                                scal_one_dom(res.dom, *aa.val_t, Derivative_lane_operand(bb, lane)));
                        });

                        if (mder != 0 && res.der_t != nullptr) {
                            res.der_t->affect_parameters();
                            res.der_t->set_parameters()->set_m_quant() = mder;
                        }
                        break;
                    default:
                        KADATH_THROW("Unknown data storage in operator+");
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator+");
        }
        return res;
    }

    Term_eq pow(const Term_eq& so, int nn)
    {
        if (nn == 0)
            return constant_one_like(so);

        if (nn == 1)
            return Term_eq(so);

        if (nn <= std::numeric_limits<int>::min() + 1)
            KADATH_THROW("Term_eq integer power exponent is too negative");

        const int exponent = (nn > 0) ? nn : -nn;
        Term_eq res(so.dom, so.type_data);

        switch (so.type_data) {
            case (TERM_D): {
                const double value = *so.val_d;
                if (nn > 0) {
                    res.val_d = new double(integer_power(value, exponent));
                    if (so.der_d != nullptr) {
                        const double derivative_factor = integer_power(value, exponent - 1);
                        res.der_d = new double(nn * derivative_factor * (*so.der_d));
                        propagate_unary_double_lanes(res, so, [&](int lane) {
                            return nn * derivative_factor * derivative_or_zero(so, lane);
                        });
                    }
                } else {
                    res.val_d = new double(1.0 / integer_power(value, exponent));
                    if (so.der_d != nullptr) {
                        const double derivative_denominator = integer_power(value, exponent + 1);
                        res.der_d = new double(-exponent * (*so.der_d) / derivative_denominator);
                        propagate_unary_double_lanes(res, so, [&](int lane) {
                            return -exponent * derivative_or_zero(so, lane) / derivative_denominator;
                        });
                    }
                }
                break;
            }
            case (TERM_T): {
                if (so.val_t->get_valence() != 0)
                    return repeated_integer_power(so, nn);

                if (nn > 0) {
                    res.val_t = new Tensor(tensor_power_one_dom(so.dom, *so.val_t, exponent));
                    if (so.der_t != nullptr) {
                        if (exponent == 1) {
                            res.der_t = new Tensor(mult_one_dom(so.dom, *so.der_t, 1));
                            propagate_unary_tensor_lanes(res, so, [&](int lane) {
                                return mult_one_dom(so.dom, so.get_der_t(lane), 1);
                            });
                        } else {
                            Tensor value_factor(
                                tensor_power_one_dom(so.dom, *so.val_t, exponent - 1));
                            res.der_t = new Tensor(mult_one_dom(
                                so.dom,
                                exponent,
                                mult_one_dom(so.dom, value_factor, *so.der_t)));
                            propagate_unary_tensor_lanes(res, so, [&](int lane) {
                                return mult_one_dom(
                                    so.dom,
                                    exponent,
                                    mult_one_dom(so.dom, value_factor, so.get_der_t(lane)));
                            });
                        }
                    }
                } else {
                    Tensor denominator(
                        tensor_power_one_dom(so.dom, *so.val_t, exponent));
                    res.val_t = new Tensor(div_one_dom(so.dom, 1.0, denominator));
                    if (so.der_t != nullptr) {
                        Tensor derivative_denominator(
                            tensor_power_one_dom(so.dom, *so.val_t, exponent + 1));
                        res.der_t = new Tensor(mult_one_dom(
                            so.dom,
                            -exponent,
                            div_one_dom(so.dom, *so.der_t, derivative_denominator)));
                        propagate_unary_tensor_lanes(res, so, [&](int lane) {
                            return mult_one_dom(
                                so.dom,
                                -exponent,
                                div_one_dom(so.dom, so.get_der_t(lane), derivative_denominator));
                        });
                    }
                }
                break;
            }
            default:
                KADATH_THROW("Unknown data storage in pow");
        }
        return res;
    }

    Term_eq operator*(int nn, const Term_eq& so)
    {

        Term_eq res(so.dom, so.type_data);

        switch (so.type_data) {
            case (TERM_D):
                res.val_d = new double(double(nn) * (*so.val_d));
                if (so.der_d != nullptr) {
                    res.der_d = new double(double(nn) * (*so.der_d));
                    propagate_unary_double_lanes(res, so, [&](int lane) {
                        return double(nn) * derivative_or_zero(so, lane);
                    });
                }
                break;
            case (TERM_T):
                res.val_t = new Tensor(mult_one_dom(so.dom, nn, *so.val_t));
                if (so.der_t != nullptr) {
                    res.der_t = new Tensor(mult_one_dom(so.dom, nn, (*so.der_t)));
                    propagate_unary_tensor_lanes(res, so, [&](int lane) {
                        return mult_one_dom(so.dom, nn, so.get_der_t(lane));
                    });
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator+");
        }
        return res;
    }

    Term_eq operator*(const Term_eq& so, int nn)
    {
        return nn * so;
    }

    Term_eq operator*(double xx, const Term_eq& so)
    {

        Term_eq res(so.dom, so.type_data);

        switch (so.type_data) {
            case (TERM_D):
                res.val_d = new double(xx * (*so.val_d));
                if (so.der_d != nullptr) {
                    res.der_d = new double(xx * (*so.der_d));
                    propagate_unary_double_lanes(res, so, [&](int lane) {
                        return xx * derivative_or_zero(so, lane);
                    });
                }
                break;
            case (TERM_T):
                res.val_t = new Tensor(mult_one_dom(so.dom, xx, (*so.val_t)));
                if (so.der_t != nullptr) {
                    res.der_t = new Tensor(mult_one_dom(so.dom, xx, (*so.der_t)));
                    propagate_unary_tensor_lanes(res, so, [&](int lane) {
                        return mult_one_dom(so.dom, xx, so.get_der_t(lane));
                    });
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator+");
        }
        return res;
    }

    Term_eq operator*(const Scalar& fact, const Term_eq& so)
    {

        Term_eq res(so.dom, so.type_data);

        switch (so.type_data) {
            case (TERM_D):
                KADATH_THROW("Multiplication of a Term_èeq by a Scalar onlyu defined for Tensorial type");
                break;
            case (TERM_T):
                res.val_t = new Tensor(mult_one_dom(so.dom, fact, *so.val_t));
                if (so.der_t != nullptr) {
                    res.der_t = new Tensor(mult_one_dom(so.dom, fact, *so.der_t));
                    propagate_unary_tensor_lanes(res, so, [&](int lane) {
                        return mult_one_dom(so.dom, fact, so.get_der_t(lane));
                    });
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator+");
        }
        return res;
    }

    Term_eq operator*(const Term_eq& so, double xx)
    {
        return xx * so;
    }

    Term_eq operator/(const Term_eq& so, double xx)
    {

        Term_eq res(so.dom, so.type_data);

        switch (so.type_data) {
            case (TERM_D):
                res.val_d = new double((*so.val_d) / xx);
                if (so.der_d != nullptr) {
                    res.der_d = new double((*so.der_d) / xx);
                    propagate_unary_double_lanes(res, so, [&](int lane) {
                        return derivative_or_zero(so, lane) / xx;
                    });
                }
                break;
            case (TERM_T):
                res.val_t = new Tensor(div_one_dom(so.dom, (*so.val_t), xx));
                if (so.der_t != nullptr) {
                    res.der_t = new Tensor(div_one_dom(so.dom, (*so.der_t), xx));
                    propagate_unary_tensor_lanes(res, so, [&](int lane) {
                        return div_one_dom(so.dom, so.get_der_t(lane), xx);
                    });
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator/");
        }
        return res;
    }

    Term_eq sqrt(const Term_eq& so)
    {

        Term_eq res(so.dom, so.type_data);

        switch (so.type_data) {
            case (TERM_D):
                res.val_d = new double(sqrt(*so.val_d));
                if (so.der_d != nullptr) {
                    res.der_d = new double((*so.der_d) / 2. / sqrt(*so.val_d));
                    propagate_unary_double_lanes(res, so, [&](int lane) {
                        return derivative_or_zero(so, lane) / 2. / sqrt(*so.val_d);
                    });
                }
                break;
            case (TERM_T):
                res.val_t = new Tensor(sqrt_one_dom(so.dom, *so.val_t));
                if (so.der_t != nullptr) {
                    res.der_t = new Tensor(div_one_dom(so.dom, (*so.der_t), 2 * sqrt_one_dom(so.dom, *so.val_t)));
                    propagate_unary_tensor_lanes(res, so, [&](int lane) {
                        return div_one_dom(
                            so.dom,
                            so.get_der_t(lane),
                            2 * sqrt_one_dom(so.dom, *so.val_t));
                    });
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in operator/");
        }
        return res;
    }

    Term_eq partial(const Term_eq& so, char ind)
    {
        Term_eq res(so.dom, so.type_data);
        switch (so.type_data) {
            case (TERM_D):
                KADATH_THROW("Partial only defined with respect to tensors");
            case (TERM_T):
                res.val_t = new Tensor(partial_one_dom(so.dom, ind, *so.val_t));
                if (so.der_t != nullptr) {
                    res.der_t = new Tensor(partial_one_dom(so.dom, ind, (*so.der_t)));
                    propagate_unary_tensor_lanes(res, so, [&](int lane) {
                        return partial_one_dom(so.dom, ind, so.get_der_t(lane));
                    });
                }
                break;
            default:
                KADATH_THROW("Unknown data storage in partial");
        }
        return res;
    }

    Term_eq div_1mx2(const Term_eq& so)
    {
        assert(so.type_data == TERM_T);

        return so.val_t->get_space().get_domain(so.dom)->div_1mx2_term_eq(so);
    }
} // namespace Kadath
