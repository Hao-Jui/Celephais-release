#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Tensor/metric_tensor.hpp"

#include "../../src/Ope_eq/ope_scalar_unary_operator.hpp"

using namespace Kadath;
using Catch::Matchers::WithinAbs;

TEST_CASE("Metric permutation sign matches three-dimensional parity", "[metric-tensor][permutation]")
{
    REQUIRE(sign(std::vector<int>{0, 1, 2}) == 1.0);
    REQUIRE(sign(std::vector<int>{0, 2, 1}) == -1.0);
    REQUIRE(sign(std::vector<int>{1, 0, 2}) == -1.0);
    REQUIRE(sign(std::vector<int>{1, 2, 0}) == 1.0);
    REQUIRE(sign(std::vector<int>{2, 0, 1}) == 1.0);
    REQUIRE(sign(std::vector<int>{2, 1, 0}) == -1.0);
}

namespace {

class FixedActionOpe : public Ope_eq
{
  public:
    FixedActionOpe(const Term_eq& result, int& calls)
        : Ope_eq(nullptr, result.get_dom()), result_(result), calls_(calls)
    {
    }

    Term_eq action() const override
    {
        ++calls_;
        return result_;
    }

  private:
    Term_eq result_;
    int& calls_;
};

class ThrowingActionOpe : public Ope_eq
{
  public:
    explicit ThrowingActionOpe(int domain)
        : Ope_eq(nullptr, domain)
    {
    }

    Term_eq action() const override
    {
        throw std::runtime_error("intentional scratch-unwind probe");
    }
};

class ScratchTestOpeAdd : public Ope_add
{
  public:
    using Ope_add::Ope_add;

    void evaluate_into(std::optional<Term_eq>& result) const
    {
        action_into_scratch(result);
    }
};

class ScratchTestOpeSub : public Ope_sub
{
  public:
    using Ope_sub::Ope_sub;

    void evaluate_into(std::optional<Term_eq>& result) const
    {
        action_into_scratch(result);
    }
};

class OneDomainScalarTensorProbe : public Tensor
{
  public:
    OneDomainScalarTensorProbe(int domain, const Space& space)
        : Tensor(one_domain_storage, domain, space)
    {
    }
};

Space_spheric make_lane_test_space()
{
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;

    Dim_array resolution(3);
    resolution.set(0) = 5;
    resolution.set(1) = 5;
    resolution.set(2) = 4;

    Dim_array bounds_dimension(1);
    bounds_dimension.set(0) = 1;
    Array<double> bounds(bounds_dimension);
    bounds.set(0) = 1.0;

    return Space_spheric(CHEB_TYPE, center, resolution, bounds);
}

Scalar constant_scalar(Space_spheric& space, double value)
{
    Scalar scalar(space);
    scalar = value;
    scalar.std_base();
    return scalar;
}

double scalar_lane_value(const Tensor& tensor)
{
    const Val_domain& domain_value = tensor()(0);
    domain_value.coef_i();
    Index first_point(domain_value.get_conf().get_dimensions());
    return domain_value(first_point);
}

void require_spectral_base_matches(const Base_spectral& actual,
                                   const Base_spectral& expected)
{
    REQUIRE(actual.is_def() == expected.is_def());
    if (expected.is_def())
        REQUIRE(actual == expected);
}

Tensor legacy_scalar_tensor_product(int domain, const Tensor& lhs, const Tensor& rhs)
{
    Scalar result(one_domain_storage, domain, lhs.get_space());
    result.set_domain(domain) = lhs()(domain) * rhs()(domain);
    const int m_quant = mult_m_quant(lhs.get_parameters(), rhs.get_parameters());
    if (m_quant != 0) {
        result.affect_parameters();
        result.set_parameters()->set_m_quant() = m_quant;
    }
    return result;
}

double tensor_component_lane_value(const Tensor& tensor, int component)
{
    Array<int> indices(tensor.indices(component));
    const Val_domain& domain_value = tensor(indices)(0);
    domain_value.coef_i();
    Index first_point(domain_value.get_conf().get_dimensions());
    return domain_value(first_point);
}

Array<int> checked_symmetric_indices(int component, int valence, int dimension)
{
    if (valence != 2 || component < 0 || component >= dimension * (dimension + 1) / 2) {
        KADATH_THROW("batch prefetch indexed past the auxiliary tensor layout");
    }

    Array<int> indices(valence);
    int current = 0;
    for (int first = 1; first <= dimension; ++first) {
        for (int second = first; second <= dimension; ++second) {
            if (current++ != component)
                continue;
            indices.set(0) = first;
            indices.set(1) = second;
            return indices;
        }
    }
    KADATH_THROW("unreachable symmetric tensor component");
}

class CheckedMetricTensor : public Metric_tensor
{
  public:
    using Metric_tensor::operator=;

    CheckedMetricTensor(const Space& space, int index_type, const Base_tensor& basis)
        : Metric_tensor(space, index_type, basis)
    {
        give_indices = checked_symmetric_indices;
    }
};

Term_eq scalar_field_term(
    Space_spheric& space,
    double value,
    double first_derivative,
    double second_derivative)
{
    Scalar value_field = constant_scalar(space, value);
    Scalar first_derivative_field = constant_scalar(space, first_derivative);
    Scalar second_derivative_field = constant_scalar(space, second_derivative);

    Term_eq term(0, value_field, first_derivative_field);
    term.set_der_t(1, second_derivative_field);
    return term;
}

Scalar radial_scalar(Space_spheric& space)
{
    Scalar scalar(space);
    scalar = 0.0;
    scalar.set_domain(0) = space.get_domain(0)->get_radius();
    scalar.std_base();
    return scalar;
}

Scalar cartesian_x_scalar(Space_spheric& space)
{
    Scalar scalar(space);
    scalar = 0.0;
    scalar.set_domain(0) = space.get_domain(0)->get_cart(1);
    scalar.std_base();
    return scalar;
}

} // namespace

TEST_CASE("One-domain tensor add and subtract expand mixed layouts to standard storage",
          "[tensor][one-domain][layout]")
{
    Space_spheric space = make_lane_test_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Tensor full(space, 2, COV, basis);
    Metric_tensor symmetric(space, COV, basis);
    full = 7.0;
    symmetric = 2.0;
    full.std_base();
    symmetric.std_base();

    const Tensor sum = add_one_dom(0, full, symmetric);
    const Tensor difference = sub_one_dom(0, full, symmetric);

    REQUIRE(sum.get_n_comp() == 9);
    REQUIRE(difference.get_n_comp() == 9);
    for (int component = 0; component < 9; ++component) {
        REQUIRE_THAT(tensor_component_lane_value(sum, component), WithinAbs(9.0, 1e-10));
        REQUIRE_THAT(tensor_component_lane_value(difference, component), WithinAbs(5.0, 1e-10));
    }
}

TEST_CASE("Tensor in-place subtraction matches the materialized assignment path",
          "[tensor][in-place]")
{
    Space_spheric space = make_lane_test_space();
    Scalar value = constant_scalar(space, 7.0);
    Scalar correction = constant_scalar(space, 2.0);
    Tensor materialized(value);
    Tensor in_place(value);
    Tensor delta(correction);

    materialized = materialized - delta;
    in_place -= delta;

    REQUIRE(materialized.get_basis() == in_place.get_basis());
    REQUIRE_THAT(scalar_lane_value(in_place),
                 WithinAbs(scalar_lane_value(materialized), 0.0));
}

TEST_CASE("Ope_eq action_into reuses a scalar result with all sparse lanes", "[term-eq][action-into]")
{
    Term_eq expected(0, 2.0, 3.0);
    expected.set_der_d(1, 5.0);
    expected.set_der_d(31, 7.0);

    int calls = 0;
    FixedActionOpe operation(expected, calls);
    Term_eq target(0, -1.0, -2.0);
    target.set_der_d(2, -3.0);

    REQUIRE(operation.action_into(target));
    REQUIRE(calls == 1);
    REQUIRE(target.get_derivative_lane_count() == 32);
    REQUIRE_THAT(target.get_val_d(), WithinAbs(2.0, 1e-14));
    REQUIRE_THAT(target.get_der_d(0), WithinAbs(3.0, 1e-14));
    REQUIRE_THAT(target.get_der_d(1), WithinAbs(5.0, 1e-14));
    REQUIRE_FALSE(target.has_der_d(2));
    REQUIRE_THAT(target.get_der_d(31), WithinAbs(7.0, 1e-14));
}

TEST_CASE("Binary operators preserve exact double values in reusable depth scratch",
          "[ope][action-operand][scratch][double]")
{
    Term_eq lhs_source(0, 2.0, 3.0);
    lhs_source.set_der_d(1, 5.0);
    lhs_source.set_der_d(31, 7.0);
    Term_eq rhs_source(0, 11.0, 13.0);
    rhs_source.set_der_d(1, 17.0);
    rhs_source.set_der_d(31, 19.0);

    int lhs_action_calls = 0;
    int rhs_action_calls = 0;
    Ope_add operation(
        nullptr,
        new FixedActionOpe(lhs_source, lhs_action_calls),
        new FixedActionOpe(rhs_source, rhs_action_calls));

    const Term_eq expected(lhs_source + rhs_source);
    const Term_eq first(operation.action());
    const Term_eq second(operation.action());

    REQUIRE(lhs_action_calls == 2);
    REQUIRE(rhs_action_calls == 2);
    for (const Term_eq* result : {&first, &second}) {
        REQUIRE(result->get_val_d() == expected.get_val_d());
        REQUIRE(result->get_der_d(0) == expected.get_der_d(0));
        REQUIRE(result->get_der_d(1) == expected.get_der_d(1));
        REQUIRE(result->get_der_d(31) == expected.get_der_d(31));
    }
}

TEST_CASE("Add and subtract write doubles directly with exact primary and packed lanes",
          "[ope][action-into][sum][double]")
{
    Term_eq lhs(0, 2.0, 3.0);
    lhs.set_der_d(1, 5.0);
    lhs.set_der_d(31, 7.0);
    Term_eq rhs(0, 11.0, 13.0);
    rhs.set_der_d(31, 19.0);

    SECTION("addition") {
        const Term_eq expected(lhs + rhs);
        Term_eq target(0, -2.0, -3.0);
        target.set_der_d(1, -5.0);
        target.set_der_d(2, -7.0);
        target.set_der_d(31, -11.0);
        int lhs_calls = 0;
        int rhs_calls = 0;
        Ope_add operation(nullptr,
                          new FixedActionOpe(lhs, lhs_calls),
                          new FixedActionOpe(rhs, rhs_calls));

        REQUIRE(operation.action_into(target));
        REQUIRE(lhs_calls == 1);
        REQUIRE(rhs_calls == 1);
        REQUIRE(target.get_val_d() == expected.get_val_d());
        REQUIRE(target.get_der_d(0) == expected.get_der_d(0));
        REQUIRE(target.get_der_d(1) == expected.get_der_d(1));
        REQUIRE(target.get_der_d(2) == expected.get_der_d(2));
        REQUIRE(target.get_der_d(31) == expected.get_der_d(31));
    }

    SECTION("subtraction") {
        const Term_eq expected(lhs - rhs);
        Term_eq target(0, -2.0, -3.0);
        target.set_der_d(1, -5.0);
        target.set_der_d(2, -7.0);
        target.set_der_d(31, -11.0);
        int lhs_calls = 0;
        int rhs_calls = 0;
        Ope_sub operation(nullptr,
                          new FixedActionOpe(lhs, lhs_calls),
                          new FixedActionOpe(rhs, rhs_calls));

        REQUIRE(operation.action_into(target));
        REQUIRE(lhs_calls == 1);
        REQUIRE(rhs_calls == 1);
        REQUIRE(target.get_val_d() == expected.get_val_d());
        REQUIRE(target.get_der_d(0) == expected.get_der_d(0));
        REQUIRE(target.get_der_d(1) == expected.get_der_d(1));
        REQUIRE(target.get_der_d(2) == expected.get_der_d(2));
        REQUIRE(target.get_der_d(31) == expected.get_der_d(31));
    }

    Term_eq alias_target(lhs);
    const Term_eq alias_before(alias_target);
    REQUIRE_FALSE(alias_target.try_write_sum_action_result(alias_target, rhs, false));
    REQUIRE(alias_target.get_val_d() == alias_before.get_val_d());
    REQUIRE(alias_target.get_der_d(0) == alias_before.get_der_d(0));
    REQUIRE(alias_target.get_der_d(1) == alias_before.get_der_d(1));
    REQUIRE(alias_target.get_der_d(31) == alias_before.get_der_d(31));
}

TEST_CASE("Add and subtract reuse scalar tensor domains with exact derivative lanes",
          "[ope][action-into][sum][tensor]")
{
    Space_spheric space = make_lane_test_space();
    Term_eq lhs(0, constant_scalar(space, 2.0), constant_scalar(space, 3.0));
    lhs.set_der_t(1, constant_scalar(space, 5.0));
    lhs.set_der_t(31, constant_scalar(space, 7.0));
    Term_eq rhs(0, constant_scalar(space, 11.0), constant_scalar(space, 13.0));
    rhs.set_der_t(31, constant_scalar(space, 19.0));

    auto make_target = [&] {
        Term_eq target(0, constant_scalar(space, -2.0), constant_scalar(space, -3.0));
        target.set_der_t(1, constant_scalar(space, -5.0));
        target.set_der_t(2, constant_scalar(space, -7.0));
        target.set_der_t(31, constant_scalar(space, -11.0));
        return target;
    };

    SECTION("addition") {
        const Term_eq expected(lhs + rhs);
        Term_eq target(make_target());
        const Tensor* const value_storage = target.get_p_val_t();
        const Tensor* const primary_storage = target.get_p_der_t(0);
        const Tensor* const lane_one_storage = target.get_p_der_t(1);
        const Tensor* const lane_two_storage = target.get_p_der_t(2);
        const Tensor* const lane_thirty_one_storage = target.get_p_der_t(31);
        int lhs_calls = 0;
        int rhs_calls = 0;
        Ope_add operation(nullptr,
                          new FixedActionOpe(lhs, lhs_calls),
                          new FixedActionOpe(rhs, rhs_calls));

        REQUIRE(operation.action_into(target));
        REQUIRE(lhs_calls == 1);
        REQUIRE(rhs_calls == 1);
        REQUIRE(target.get_p_val_t() == value_storage);
        REQUIRE(target.get_p_der_t(0) == primary_storage);
        REQUIRE(target.get_p_der_t(1) == lane_one_storage);
        REQUIRE(target.get_p_der_t(2) == lane_two_storage);
        REQUIRE(target.get_p_der_t(31) == lane_thirty_one_storage);
        REQUIRE(scalar_lane_value(target.get_val_t()) == scalar_lane_value(expected.get_val_t()));
        REQUIRE(scalar_lane_value(target.get_der_t(0)) == scalar_lane_value(expected.get_der_t(0)));
        REQUIRE(scalar_lane_value(target.get_der_t(1)) == scalar_lane_value(expected.get_der_t(1)));
        REQUIRE(target.get_der_t(2)()(0).check_if_zero());
        REQUIRE(expected.get_der_t(2)()(0).check_if_zero());
        REQUIRE(scalar_lane_value(target.get_der_t(31)) == scalar_lane_value(expected.get_der_t(31)));
    }

    SECTION("subtraction") {
        const Term_eq expected(lhs - rhs);
        Term_eq target(make_target());
        const Tensor* const value_storage = target.get_p_val_t();
        const Tensor* const primary_storage = target.get_p_der_t(0);
        const Tensor* const lane_one_storage = target.get_p_der_t(1);
        const Tensor* const lane_two_storage = target.get_p_der_t(2);
        const Tensor* const lane_thirty_one_storage = target.get_p_der_t(31);
        int lhs_calls = 0;
        int rhs_calls = 0;
        Ope_sub operation(nullptr,
                          new FixedActionOpe(lhs, lhs_calls),
                          new FixedActionOpe(rhs, rhs_calls));

        REQUIRE(operation.action_into(target));
        REQUIRE(lhs_calls == 1);
        REQUIRE(rhs_calls == 1);
        REQUIRE(target.get_p_val_t() == value_storage);
        REQUIRE(target.get_p_der_t(0) == primary_storage);
        REQUIRE(target.get_p_der_t(1) == lane_one_storage);
        REQUIRE(target.get_p_der_t(2) == lane_two_storage);
        REQUIRE(target.get_p_der_t(31) == lane_thirty_one_storage);
        REQUIRE(scalar_lane_value(target.get_val_t()) == scalar_lane_value(expected.get_val_t()));
        REQUIRE(scalar_lane_value(target.get_der_t(0)) == scalar_lane_value(expected.get_der_t(0)));
        REQUIRE(scalar_lane_value(target.get_der_t(1)) == scalar_lane_value(expected.get_der_t(1)));
        REQUIRE(target.get_der_t(2)()(0).check_if_zero());
        REQUIRE(expected.get_der_t(2)()(0).check_if_zero());
        REQUIRE(scalar_lane_value(target.get_der_t(31)) == scalar_lane_value(expected.get_der_t(31)));
    }

    Term_eq alias_target(lhs);
    const Term_eq alias_before(alias_target);
    REQUIRE_FALSE(alias_target.try_write_sum_action_result(alias_target, rhs, true));
    REQUIRE(scalar_lane_value(alias_target.get_val_t()) ==
            scalar_lane_value(alias_before.get_val_t()));
    REQUIRE(scalar_lane_value(alias_target.get_der_t(0)) ==
            scalar_lane_value(alias_before.get_der_t(0)));
    REQUIRE(scalar_lane_value(alias_target.get_der_t(1)) ==
            scalar_lane_value(alias_before.get_der_t(1)));
    REQUIRE(scalar_lane_value(alias_target.get_der_t(31)) ==
            scalar_lane_value(alias_before.get_der_t(31)));
}

TEST_CASE("Direct scalar sum write preserves logical-zero bases for missing lanes",
          "[term-eq][action-into][sum][zero]")
{
    Space_spheric space = make_lane_test_space();
    Scalar lhs_zero = constant_scalar(space, 0.0);
    Scalar rhs_zero = constant_scalar(space, 0.0);
    rhs_zero.std_anti_base();
    REQUIRE(lhs_zero(0).check_if_zero());
    REQUIRE(rhs_zero(0).check_if_zero());
    REQUIRE_FALSE(lhs_zero(0).get_base() == rhs_zero(0).get_base());

    Term_eq lhs(0, lhs_zero, lhs_zero);
    lhs.set_derivative_lane_count(5);
    lhs.set_der_t(1, lhs_zero);
    lhs.set_der_t(2, lhs_zero);
    Term_eq rhs(0, rhs_zero, rhs_zero);
    rhs.set_derivative_lane_count(5);
    rhs.set_der_t(1, rhs_zero);
    rhs.set_der_t(3, rhs_zero);

    auto require_direct_matches = [&](const Term_eq& expected, bool subtract) {
        Term_eq direct(lhs);
        REQUIRE(direct.try_write_sum_action_result(lhs, rhs, subtract));
        REQUIRE(direct.get_derivative_lane_count() ==
                expected.get_derivative_lane_count());

        require_spectral_base_matches(direct.get_val_t()()(0).get_base(),
                                      expected.get_val_t()()(0).get_base());
        require_spectral_base_matches(direct.get_der_t(0)()(0).get_base(),
                                      expected.get_der_t(0)()(0).get_base());
        for (int lane = 1; lane < direct.get_derivative_lane_count(); ++lane) {
            REQUIRE(direct.has_der_t(lane) == expected.has_der_t(lane));
            REQUIRE(direct.get_der_t(lane)()(0).check_if_zero() ==
                    expected.get_der_t(lane)()(0).check_if_zero());
            require_spectral_base_matches(
                direct.get_der_t(lane)()(0).get_base(),
                expected.get_der_t(lane)()(0).get_base());
        }
    };

    SECTION("addition") {
        require_direct_matches(lhs + rhs, false);
    }

    SECTION("subtraction") {
        require_direct_matches(lhs - rhs, true);
    }
}

TEST_CASE("Add write-into falls back for non-scalar and mixed tensor operands",
          "[ope][action-into][sum][fallback]")
{
    Space_spheric space = make_lane_test_space();
    Base_tensor basis(space, CARTESIAN_BASIS);

    SECTION("non-scalar tensor") {
        Tensor lhs_value(space, 2, COV, basis);
        Tensor rhs_value(space, 2, COV, basis);
        Tensor target_value(space, 2, COV, basis);
        lhs_value = 2.0;
        rhs_value = 11.0;
        target_value = -3.0;
        lhs_value.std_base();
        rhs_value.std_base();
        target_value.std_base();
        Term_eq lhs(0, lhs_value);
        Term_eq rhs(0, rhs_value);
        Term_eq target(0, target_value);
        const Term_eq target_before(target);

        REQUIRE_FALSE(target.try_write_sum_action_result(lhs, rhs, false));
        REQUIRE(tensor_component_lane_value(target.get_val_t(), 8) ==
                tensor_component_lane_value(target_before.get_val_t(), 8));

        const Term_eq expected(lhs + rhs);
        int lhs_calls = 0;
        int rhs_calls = 0;
        Ope_add operation(nullptr,
                          new FixedActionOpe(lhs, lhs_calls),
                          new FixedActionOpe(rhs, rhs_calls));
        REQUIRE(operation.action_into(target));
        REQUIRE(lhs_calls == 1);
        REQUIRE(rhs_calls == 1);
        REQUIRE(tensor_component_lane_value(target.get_val_t(), 8) ==
                tensor_component_lane_value(expected.get_val_t(), 8));
    }

    SECTION("mixed double and scalar tensor") {
        Term_eq lhs(0, 2.0, 3.0);
        Term_eq rhs(0, constant_scalar(space, 11.0), constant_scalar(space, 13.0));
        Term_eq target(0, constant_scalar(space, -3.0), constant_scalar(space, -5.0));
        const Term_eq target_before(target);

        REQUIRE_FALSE(target.try_write_sum_action_result(lhs, rhs, false));
        REQUIRE(scalar_lane_value(target.get_val_t()) ==
                scalar_lane_value(target_before.get_val_t()));
        REQUIRE(scalar_lane_value(target.get_der_t(0)) ==
                scalar_lane_value(target_before.get_der_t(0)));

        const Term_eq expected(lhs + rhs);
        int lhs_calls = 0;
        int rhs_calls = 0;
        Ope_add operation(nullptr,
                          new FixedActionOpe(lhs, lhs_calls),
                          new FixedActionOpe(rhs, rhs_calls));
        REQUIRE(operation.action_into(target));
        REQUIRE(lhs_calls == 1);
        REQUIRE(rhs_calls == 1);
        REQUIRE(scalar_lane_value(target.get_val_t()) ==
                scalar_lane_value(expected.get_val_t()));
        REQUIRE(scalar_lane_value(target.get_der_t(0)) ==
                scalar_lane_value(expected.get_der_t(0)));
    }
}

TEST_CASE("Add and subtract scratch fallbacks rebuild incompatible result layouts",
          "[ope][action-into][sum][scratch][fallback]")
{
    Space_spheric space = make_lane_test_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Tensor lhs_value(space, 2, COV, basis);
    Tensor rhs_value(space, 2, COV, basis);
    lhs_value = 11.0;
    rhs_value = 2.0;
    lhs_value.std_base();
    rhs_value.std_base();
    const Term_eq lhs(0, lhs_value);
    const Term_eq rhs(0, rhs_value);

    SECTION("addition") {
        int lhs_calls = 0;
        int rhs_calls = 0;
        ScratchTestOpeAdd operation(nullptr,
                                    new FixedActionOpe(lhs, lhs_calls),
                                    new FixedActionOpe(rhs, rhs_calls));
        std::optional<Term_eq> result(
            Term_eq(0, constant_scalar(space, -3.0), constant_scalar(space, -5.0)));

        operation.evaluate_into(result);

        REQUIRE(result.has_value());
        REQUIRE(result->get_val_t().get_valence() == 2);
        REQUIRE_FALSE(result->has_der_t(0));
        REQUIRE(lhs_calls == 1);
        REQUIRE(rhs_calls == 1);
        for (int component = 0; component < 9; ++component) {
            REQUIRE_THAT(tensor_component_lane_value(result->get_val_t(), component),
                         WithinAbs(13.0, 0.0));
        }
    }

    SECTION("subtraction") {
        int lhs_calls = 0;
        int rhs_calls = 0;
        ScratchTestOpeSub operation(nullptr,
                                    new FixedActionOpe(lhs, lhs_calls),
                                    new FixedActionOpe(rhs, rhs_calls));
        std::optional<Term_eq> result(
            Term_eq(0, constant_scalar(space, -3.0), constant_scalar(space, -5.0)));

        operation.evaluate_into(result);

        REQUIRE(result.has_value());
        REQUIRE(result->get_val_t().get_valence() == 2);
        REQUIRE_FALSE(result->has_der_t(0));
        REQUIRE(lhs_calls == 1);
        REQUIRE(rhs_calls == 1);
        for (int component = 0; component < 9; ++component) {
            REQUIRE_THAT(tensor_component_lane_value(result->get_val_t(), component),
                         WithinAbs(9.0, 0.0));
        }
    }
}

TEST_CASE("Operand scratch leases are LIFO, shape guarded, and exception safe",
          "[ope][action-operand][scratch][layout]")
{
    Term_eq double_source(0, 2.0, 3.0);
    double_source.set_der_d(31, 5.0);
    int double_calls = 0;
    FixedActionOpe double_operation(double_source, double_calls);

    const std::optional<Term_eq>* first_slot = nullptr;
    const std::optional<Term_eq>* second_slot = nullptr;
    {
        ope_action_detail::OperandScratchLease first_lease;
        first_slot = &first_lease.storage();
        const Term_eq& first = double_operation.action_operand(first_lease.storage());
        REQUIRE(first.get_val_d() == 2.0);

        ope_action_detail::OperandScratchLease second_lease;
        second_slot = &second_lease.storage();
        const Term_eq& second = double_operation.action_operand(second_lease.storage());
        REQUIRE(second.get_der_d(31) == 5.0);
        REQUIRE(first_slot != second_slot);
    }

    Space_spheric space = make_lane_test_space();
    Term_eq tensor_source = scalar_field_term(space, 7.0, 11.0, 13.0);
    tensor_source.set_der_t(31, constant_scalar(space, 17.0));
    int tensor_calls = 0;
    FixedActionOpe tensor_operation(tensor_source, tensor_calls);
    {
        ope_action_detail::OperandScratchLease reused_lease;
        REQUIRE(&reused_lease.storage() == first_slot);
        const Term_eq& rebuilt = tensor_operation.action_operand(reused_lease.storage());
        REQUIRE(scalar_lane_value(rebuilt.get_val_t()) == 7.0);
        REQUIRE(scalar_lane_value(rebuilt.get_der_t(0)) == 11.0);
        REQUIRE(scalar_lane_value(rebuilt.get_der_t(1)) == 13.0);
        REQUIRE(scalar_lane_value(rebuilt.get_der_t(31)) == 17.0);
    }

    ThrowingActionOpe throwing_operation(0);
    const std::optional<Term_eq>* throwing_slot = nullptr;
    REQUIRE_THROWS_AS(
        [&] {
            ope_action_detail::OperandScratchLease throwing_lease;
            throwing_slot = &throwing_lease.storage();
            throwing_operation.action_operand(throwing_lease.storage());
        }(),
        std::runtime_error);
    {
        ope_action_detail::OperandScratchLease recovered_lease;
        REQUIRE(&recovered_lease.storage() == throwing_slot);
    }
}

TEST_CASE("Nested binary scratch preserves borrowed tensor leaves and every derivative lane",
          "[ope][action-operand][scratch][tensor]")
{
    Space_spheric space = make_lane_test_space();
    Term_eq lhs_source = scalar_field_term(space, 2.0, 3.0, 5.0);
    Term_eq rhs_source = scalar_field_term(space, 7.0, 11.0, 13.0);
    Term_eq factor_source = scalar_field_term(space, 17.0, 19.0, 23.0);
    lhs_source.set_der_t(31, constant_scalar(space, 29.0));
    rhs_source.set_der_t(31, constant_scalar(space, 31.0));
    factor_source.set_der_t(31, constant_scalar(space, 37.0));

    int lhs_action_calls = 0;
    Ope_mult operation(
        nullptr,
        new Ope_add(nullptr,
                    new FixedActionOpe(lhs_source, lhs_action_calls),
                    new Ope_id(nullptr, &rhs_source)),
        new Ope_id(nullptr, &factor_source));

    const Term_eq expected((lhs_source + rhs_source) * factor_source);
    const Term_eq first(operation.action());
    const Term_eq second(operation.action());

    REQUIRE(lhs_action_calls == 2);
    REQUIRE(scalar_lane_value(rhs_source.get_val_t()) == 7.0);
    REQUIRE(scalar_lane_value(rhs_source.get_der_t(0)) == 11.0);
    REQUIRE(scalar_lane_value(rhs_source.get_der_t(1)) == 13.0);
    REQUIRE(scalar_lane_value(rhs_source.get_der_t(31)) == 31.0);
    REQUIRE(scalar_lane_value(factor_source.get_val_t()) == 17.0);
    REQUIRE(scalar_lane_value(factor_source.get_der_t(31)) == 37.0);
    for (const Term_eq* result : {&first, &second}) {
        REQUIRE(scalar_lane_value(result->get_val_t()) ==
                scalar_lane_value(expected.get_val_t()));
        REQUIRE(scalar_lane_value(result->get_der_t(0)) ==
                scalar_lane_value(expected.get_der_t(0)));
        REQUIRE(scalar_lane_value(result->get_der_t(1)) ==
                scalar_lane_value(expected.get_der_t(1)));
        REQUIRE(scalar_lane_value(result->get_der_t(31)) ==
                scalar_lane_value(expected.get_der_t(31)));
    }
}

TEST_CASE("Ope_eq action_into preserves persistent tensor shape and auxiliary lane layout",
          "[term-eq][action-into][layout]")
{
    Space_spheric space = make_lane_test_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Tensor value(space, 2, COV, basis);
    Tensor primary_derivative(space, 2, COV, basis);
    CheckedMetricTensor auxiliary_derivative(space, COV, basis);
    Tensor target_value(space, 2, COV, basis);
    Tensor target_derivative(space, 2, COV, basis);

    value = 2.0;
    primary_derivative = 3.0;
    auxiliary_derivative = 7.0;
    target_value = -1.0;
    target_derivative = -2.0;
    value.std_base();
    primary_derivative.std_base();
    auxiliary_derivative.std_base();
    target_value.std_base();
    target_derivative.std_base();

    Term_eq expected(0, value, primary_derivative);
    expected.set_der_t(31, auxiliary_derivative);
    Term_eq target(0, target_value, target_derivative);
    target.set_der_t(1, target_derivative);
    const Tensor* target_value_storage = target.get_p_val_t();
    const Tensor* target_primary_derivative_storage = target.get_p_der_t(0);

    int calls = 0;
    FixedActionOpe operation(expected, calls);
    REQUIRE(operation.action_into(target));
    REQUIRE(calls == 1);
    REQUIRE(target.get_p_val_t() == target_value_storage);
    REQUIRE(target.get_p_der_t(0) == target_primary_derivative_storage);
    REQUIRE(target.get_val_t().get_n_comp() == 9);
    REQUIRE_THAT(tensor_component_lane_value(target.get_val_t(), 8), WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(tensor_component_lane_value(target.get_der_t(0), 8), WithinAbs(3.0, 1e-10));
    REQUIRE_FALSE(target.has_der_t(1));
    REQUIRE(target.get_der_t(31).get_n_comp() == 6);
    REQUIRE_THAT(tensor_component_lane_value(target.get_der_t(31), 5), WithinAbs(7.0, 1e-10));
}

TEST_CASE("Term_eq action-result reuse refuses incompatibility and action_into falls back on layout",
          "[term-eq][action-into][fallback]")
{
    Term_eq target(0, -1.0, -2.0);
    target.set_der_d(1, -3.0);
    Term_eq wrong_domain(1, 2.0, 3.0);
    wrong_domain.set_der_d(1, 5.0);

    REQUIRE_FALSE(target.try_write_action_result(wrong_domain));
    REQUIRE_THAT(target.get_val_d(), WithinAbs(-1.0, 1e-14));
    REQUIRE_THAT(target.get_der_d(0), WithinAbs(-2.0, 1e-14));
    REQUIRE_THAT(target.get_der_d(1), WithinAbs(-3.0, 1e-14));

    Space_spheric space = make_lane_test_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Tensor expected_value(space, 2, COV, basis);
    Tensor target_value(space, 2, COV, basis);
    expected_value = 7.0;
    target_value = -7.0;
    expected_value.std_base();
    target_value.std_base();
    expected_value.set_name_affected();
    expected_value.set_name_ind(0, 'i');
    expected_value.set_name_ind(1, 'j');
    target_value.set_name_affected();
    target_value.set_name_ind(0, 'j');
    target_value.set_name_ind(1, 'i');

    Term_eq expected(0, expected_value);
    Term_eq tensor_target(0, target_value);
    const Tensor* target_storage = tensor_target.get_p_val_t();
    int calls = 0;
    FixedActionOpe operation(expected, calls);

    REQUIRE_FALSE(operation.action_into(tensor_target));
    REQUIRE(calls == 1);
    REQUIRE(tensor_target.get_p_val_t() == target_storage);
    REQUIRE(tensor_target.get_val_t().get_name_ind()[0] == 'j');
    REQUIRE(tensor_target.get_val_t().get_name_ind()[1] == 'i');
    REQUIRE_THAT(tensor_component_lane_value(tensor_target.get_val_t(), 8), WithinAbs(7.0, 1e-10));
}

TEST_CASE("Term_eq action-result reuse preserves compatible auxiliary lane storage",
          "[term-eq][action-into][lane-storage]")
{
    Space_spheric space = make_lane_test_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Tensor source_value(space, 2, COV, basis);
    Tensor source_lane(space, 2, COV, basis);
    Tensor target_value(space, 2, COV, basis);
    Tensor target_lane(space, 2, COV, basis);
    source_value = 2.0;
    source_lane = 7.0;
    target_value = -2.0;
    target_lane = -7.0;
    source_value.std_base();
    source_lane.std_base();
    target_value.std_base();
    target_lane.std_base();

    Term_eq source(0, source_value);
    source.set_der_t(31, source_lane);
    Term_eq legacy_target(0, target_value);
    legacy_target.set_der_t(31, target_lane);
    Term_eq fast_target(legacy_target);
    Term_eq fast_source(source);
    const Tensor* legacy_storage = legacy_target.get_p_der_t(31);
    const Tensor* fast_storage = fast_target.get_p_der_t(31);

    legacy_target = source;
    REQUIRE(legacy_target.get_p_der_t(31) == legacy_storage);
    REQUIRE(fast_target.try_write_action_result(fast_source));
    REQUIRE(fast_target.get_p_der_t(31) == fast_storage);
    REQUIRE_THAT(tensor_component_lane_value(fast_target.get_der_t(31), 8),
                 WithinAbs(7.0, 1e-10));
}

TEST_CASE("Term_eq double derivative lanes preserve scalar lane zero", "[term-eq][w-lane]")
{
    STATIC_REQUIRE(sizeof(Term_eq) < 552);

    Term_eq term(0, 2.0, 3.0);
    REQUIRE(term.get_derivative_lane_count() == 1);
    REQUIRE_THAT(term.get_der_d(), WithinAbs(3.0, 1e-14));
    REQUIRE_THAT(term.get_der_d(0), WithinAbs(3.0, 1e-14));

    term.set_der_d(1, 7.0);
    REQUIRE(term.get_derivative_lane_count() == 2);
    REQUIRE_THAT(term.get_der_d(0), WithinAbs(3.0, 1e-14));
    REQUIRE_THAT(term.get_der_d(1), WithinAbs(7.0, 1e-14));

    Term_eq copied(term);
    REQUIRE(copied.get_derivative_lane_count() == 2);
    REQUIRE_THAT(copied.get_der_d(0), WithinAbs(3.0, 1e-14));
    REQUIRE_THAT(copied.get_der_d(1), WithinAbs(7.0, 1e-14));

    copied.set_der_zero();
    REQUIRE_THAT(copied.get_der_d(0), WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(copied.get_der_d(1), WithinAbs(0.0, 1e-14));

    copied.set_derivative_lane_count(1);
    REQUIRE(copied.get_derivative_lane_count() == 1);
    REQUIRE_FALSE(copied.has_der_d(1));
    REQUIRE_THROWS(copied.get_der_d(1));
    copied.set_derivative_lane_count(2);
    REQUIRE_FALSE(copied.has_der_d(1));
}

TEST_CASE("Term_eq tensor derivative lane one has independent storage", "[term-eq][w-lane]")
{
    Space_spheric space = make_lane_test_space();
    Scalar value(space);
    value = 2.0;
    value.std_base();

    Scalar first_derivative(space);
    first_derivative = 3.0;
    first_derivative.std_base();

    Scalar second_derivative(space);
    second_derivative = 7.0;
    second_derivative.std_base();

    Term_eq term(0, value, first_derivative);
    REQUIRE(term.get_derivative_lane_count() == 1);
    REQUIRE(term.get_p_der_t() != nullptr);
    REQUIRE(term.get_p_der_t(0) == term.get_p_der_t());

    term.set_der_t(1, second_derivative);
    REQUIRE(term.get_derivative_lane_count() == 2);
    REQUIRE(term.get_p_der_t(1) != nullptr);
    REQUIRE(term.get_p_der_t(1) != term.get_p_der_t(0));

    Term_eq copied(term);
    REQUIRE(copied.get_derivative_lane_count() == 2);
    REQUIRE(copied.get_p_der_t(0) != nullptr);
    REQUIRE(copied.get_p_der_t(1) != nullptr);

    copied.clear_der(1);
    REQUIRE(copied.get_p_der_t(0) != nullptr);
    REQUIRE(copied.get_p_der_t(1) == nullptr);
}

TEST_CASE("Derivative lane prefetch preserves an auxiliary tensor's independent layout",
          "[term-eq][w-lane][derivative-lanes][layout]")
{
    if (!Val_domain::derivative_lane_tiling_enabled()) {
        SKIP("derivative lane tiling is disabled by the process environment");
    }

    Space_spheric space = make_lane_test_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Tensor value(space, 2, COV, basis);
    Tensor primary_derivative(space, 2, COV, basis);
    CheckedMetricTensor auxiliary_derivative(space, COV, basis);

    value = 2.0;
    primary_derivative = 3.0;
    auxiliary_derivative = 7.0;
    value.std_base();
    primary_derivative.std_base();
    auxiliary_derivative.std_base();

    REQUIRE(value.get_n_comp() == 9);
    REQUIRE(auxiliary_derivative.get_n_comp() == 6);

    Term_eq source(0, value, primary_derivative);
    source.set_der_t(1, auxiliary_derivative);

    const Term_eq differentiated(source.der_abs(1));
    REQUIRE(differentiated.get_p_der_t(1) != nullptr);
    REQUIRE(differentiated.get_der_t(1).get_n_comp() == auxiliary_derivative.get_n_comp());
}

TEST_CASE("Term_eq sparse lane 31 preserves missing and present-zero semantics", "[term-eq][w-lane]")
{
    Term_eq term(0, 2.0, 3.0);
    term.set_derivative_lane_count(32);

    REQUIRE_FALSE(term.has_der_d(1));
    REQUIRE_FALSE(term.has_der_d(30));
    REQUIRE_FALSE(term.has_der_d(31));

    term.set_der_zero(31);
    REQUIRE(term.has_der_d(31));
    REQUIRE_THAT(term.get_der_d(31), WithinAbs(0.0, 1e-14));

    Term_eq copied(term);
    REQUIRE(copied.get_derivative_lane_count() == 32);
    REQUIRE(copied.has_der_d(31));
    copied.set_der_d(31, 7.0);
    REQUIRE_THAT(term.get_der_d(31), WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(copied.get_der_d(31), WithinAbs(7.0, 1e-14));

    term.reset_derivative_tile(4);
    REQUIRE(term.get_derivative_lane_count() == 4);
    REQUIRE(term.has_der_d(0));
    REQUIRE_THAT(term.get_der_d(0), WithinAbs(0.0, 1e-14));
    REQUIRE_FALSE(term.has_der_d(31));
    REQUIRE_THROWS(term.get_der_d(31));

    Term_eq value_only(0, 5.0);
    REQUIRE_FALSE(value_only.has_der_d(0));
    value_only.reset_derivative_tile(8);
    REQUIRE(value_only.has_der_d(0));
    REQUIRE_THAT(value_only.get_der_d(0), WithinAbs(0.0, 1e-14));
}

TEST_CASE("Term_eq tensor lane 31 reuses layout and deep-copies sparse storage", "[term-eq][w-lane]")
{
    Space_spheric space = make_lane_test_space();
    Scalar value = constant_scalar(space, 2.0);
    Scalar first_derivative = constant_scalar(space, 3.0);
    Scalar lane_derivative = constant_scalar(space, 7.0);
    Scalar replacement_derivative = constant_scalar(space, 11.0);

    Term_eq term(0, value, first_derivative);
    term.set_derivative_lane_count(32);
    term.set_der_t(31, lane_derivative);
    REQUIRE_FALSE(term.has_der_t(1));
    REQUIRE_FALSE(term.has_der_t(30));

    const Tensor* lane_storage = term.get_p_der_t(31);
    term.set_der_t(31, replacement_derivative);
    REQUIRE(term.get_p_der_t(31) == lane_storage);
    REQUIRE_THAT(scalar_lane_value(term.get_der_t(31)), WithinAbs(11.0, 1e-10));

    Term_eq copied(term);
    REQUIRE(copied.get_p_der_t(31) != term.get_p_der_t(31));
    copied.set_der_t(31, lane_derivative);
    REQUIRE_THAT(scalar_lane_value(term.get_der_t(31)), WithinAbs(11.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(copied.get_der_t(31)), WithinAbs(7.0, 1e-10));

    const Tensor* primary_storage = term.get_p_der_t(0);
    term.reset_derivative_tile(8);
    REQUIRE(term.get_derivative_lane_count() == 8);
    REQUIRE(term.get_p_der_t(0) == primary_storage);
    REQUIRE(term.get_der_t(0)()(0).check_if_zero());
    REQUIRE(term.get_p_der_t(31) == nullptr);

    term.set_der_t(7, lane_derivative);
    REQUIRE(term.get_p_der_t(7) != nullptr);
    term.reset_derivative_tile(1);
    REQUIRE(term.get_derivative_lane_count() == 1);
    REQUIRE(term.get_p_der_t(0) == primary_storage);
    REQUIRE(term.get_der_t(0)()(0).check_if_zero());
    REQUIRE(term.get_p_der_t(7) == nullptr);

    Term_eq value_only(0, value);
    REQUIRE(value_only.get_p_der_t(0) == nullptr);
    value_only.reset_derivative_tile(1);
    REQUIRE(value_only.get_p_der_t(0) != nullptr);
    REQUIRE(value_only.get_der_t(0)()(0).check_if_zero());
}

TEST_CASE("Term_eq retained tensor tile storage stays inactive until rewritten",
          "[term-eq][w-lane][retained-storage]")
{
    Space_spheric space = make_lane_test_space();
    Scalar value = constant_scalar(space, 2.0);
    Scalar primary_derivative = constant_scalar(space, 3.0);
    Scalar lane_seven = constant_scalar(space, 7.0);
    Scalar lane_thirty_one = constant_scalar(space, 31.0);
    Scalar replacement = constant_scalar(space, 11.0);

    Term_eq term(0, value, primary_derivative);
    term.set_der_t(7, lane_seven);
    term.set_der_t(31, lane_thirty_one);
    const Tensor* const primary_storage = term.get_p_der_t(0);
    const Tensor* const lane_seven_storage = term.get_p_der_t(7);
    const Tensor* const lane_thirty_one_storage = term.get_p_der_t(31);

    term.reset_derivative_tile_retain_storage(4);
    REQUIRE(term.get_derivative_lane_count() == 4);
    REQUIRE(term.get_p_der_t(0) == primary_storage);
    REQUIRE(term.get_der_t(0)()(0).check_if_zero());
    REQUIRE_FALSE(term.has_der_t(7));
    REQUIRE_FALSE(term.has_der_t(31));
    REQUIRE(term.get_p_der_t(7) == nullptr);
    REQUIRE(term.get_p_der_t(31) == nullptr);
    REQUIRE_THROWS(term.get_der_t(7));
    REQUIRE_THROWS(term.get_der_t(31));

    term.set_der_t(7, replacement);
    REQUIRE(term.get_p_der_t(7) == lane_seven_storage);
    REQUIRE_THAT(scalar_lane_value(term.get_der_t(7)), WithinAbs(11.0, 1e-10));
    REQUIRE_FALSE(term.has_der_t(31));

    term.reset_derivative_tile_retain_storage(1);
    REQUIRE_FALSE(term.has_der_t(7));
    term.reset_derivative_tile_retain_storage(32);
    REQUIRE_FALSE(term.has_der_t(7));
    REQUIRE_FALSE(term.has_der_t(31));

    term.set_der_t(31, replacement);
    REQUIRE(term.get_p_der_t(31) == lane_thirty_one_storage);
    REQUIRE_THAT(scalar_lane_value(term.get_der_t(31)), WithinAbs(11.0, 1e-10));

    Term_eq value_only(0, value);
    value_only.reset_derivative_tile_retain_storage(8);
    REQUIRE(value_only.has_der_t(0));
    REQUIRE(value_only.get_der_t(0)()(0).check_if_zero());
}

TEST_CASE("Term_eq retained double tile distinguishes dormant storage from a zero lane",
          "[term-eq][w-lane][retained-storage]")
{
    Term_eq term(0, 2.0, 3.0);
    term.set_der_d(1, 1.0);
    term.set_der_d(7, 7.0);
    term.set_der_d(31, 31.0);

    term.reset_derivative_tile_retain_storage(4);
    REQUIRE(term.get_derivative_lane_count() == 4);
    REQUIRE(term.has_der_d(0));
    REQUIRE_THAT(term.get_der_d(0), WithinAbs(0.0, 1e-14));
    REQUIRE_FALSE(term.has_der_d(1));
    REQUIRE_FALSE(term.has_der_d(7));
    REQUIRE_FALSE(term.has_der_d(31));
    REQUIRE_THROWS(term.get_der_d(1));
    REQUIRE_THROWS(term.get_der_d(7));
    REQUIRE_THROWS(term.get_der_d(31));

    term.set_der_zero(7);
    REQUIRE(term.has_der_d(7));
    REQUIRE_THAT(term.get_der_d(7), WithinAbs(0.0, 1e-14));
    REQUIRE_FALSE(term.has_der_d(1));
    REQUIRE_FALSE(term.has_der_d(31));

    term.reset_derivative_tile_retain_storage(1);
    term.reset_derivative_tile_retain_storage(32);
    REQUIRE_FALSE(term.has_der_d(7));
    term.set_der_d(31, 13.0);
    REQUIRE(term.has_der_d(31));
    REQUIRE_THAT(term.get_der_d(31), WithinAbs(13.0, 1e-14));

    Term_eq value_only(0, 5.0);
    value_only.reset_derivative_tile_retain_storage(8);
    REQUIRE(value_only.has_der_d(0));
    REQUIRE_THAT(value_only.get_der_d(0), WithinAbs(0.0, 1e-14));
}

TEST_CASE("Term_eq action-result reuse cannot reactivate dormant tensor lanes",
          "[term-eq][action-into][retained-storage]")
{
    Space_spheric space = make_lane_test_space();
    Scalar value = constant_scalar(space, 2.0);
    Scalar primary_derivative = constant_scalar(space, 3.0);
    Scalar obsolete = constant_scalar(space, -7.0);
    Scalar source_lane = constant_scalar(space, 17.0);

    Term_eq target(0, value, primary_derivative);
    target.set_der_t(1, obsolete);
    target.set_der_t(31, obsolete);
    const Tensor* const lane_thirty_one_storage = target.get_p_der_t(31);
    target.reset_derivative_tile_retain_storage(32);

    Term_eq source(0, value, primary_derivative);
    source.set_der_t(31, source_lane);
    REQUIRE(target.try_write_action_result(source));
    REQUIRE_FALSE(target.has_der_t(1));
    REQUIRE(target.get_p_der_t(1) == nullptr);
    REQUIRE(target.has_der_t(31));
    REQUIRE(target.get_p_der_t(31) == lane_thirty_one_storage);
    REQUIRE_THAT(scalar_lane_value(target.get_der_t(31)), WithinAbs(17.0, 1e-10));

    target.reset_derivative_tile_retain_storage(32);
    Term_eq copied(target);
    REQUIRE_FALSE(copied.has_der_t(1));
    REQUIRE_FALSE(copied.has_der_t(31));
    REQUIRE(copied.get_p_der_t(1) == nullptr);
    REQUIRE(copied.get_p_der_t(31) == nullptr);
}

TEST_CASE("Term_eq retained tensor storage replaces an incompatible dormant layout",
          "[term-eq][w-lane][retained-storage][layout]")
{
    Space_spheric space = make_lane_test_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Tensor value(space, 2, COV, basis);
    Tensor primary_derivative(space, 2, COV, basis);
    CheckedMetricTensor symmetric_lane(space, COV, basis);
    Tensor full_lane(space, 2, COV, basis);
    value = 2.0;
    primary_derivative = 3.0;
    symmetric_lane = 7.0;
    full_lane = 11.0;
    value.std_base();
    primary_derivative.std_base();
    symmetric_lane.std_base();
    full_lane.std_base();

    Term_eq term(0, value, primary_derivative);
    term.set_der_t(31, symmetric_lane);
    term.reset_derivative_tile_retain_storage(32);
    term.set_der_t(31, full_lane);

    REQUIRE(term.has_der_t(31));
    REQUIRE(term.get_der_t(31).get_n_comp() == 9);
    REQUIRE_THAT(tensor_component_lane_value(term.get_der_t(31), 8), WithinAbs(11.0, 1e-10));
}

TEST_CASE("Term_eq sparse tensor assignment reuses storage and handles self-assignment", "[term-eq][w-lane]")
{
    Space_spheric space = make_lane_test_space();
    Scalar value = constant_scalar(space, 2.0);
    Scalar first_derivative = constant_scalar(space, 3.0);
    Scalar source_lane = constant_scalar(space, 7.0);
    Scalar obsolete_lane = constant_scalar(space, 11.0);

    Term_eq source(0, value, first_derivative);
    source.set_derivative_lane_count(32);
    source.set_der_t(31, source_lane);

    Term_eq target(0, value, first_derivative);
    target.set_der_t(1, obsolete_lane);
    target.set_der_t(31, obsolete_lane);
    const Tensor* target_lane_storage = target.get_p_der_t(31);

    target = source;
    REQUIRE(target.get_derivative_lane_count() == 32);
    REQUIRE_FALSE(target.has_der_t(1));
    REQUIRE(target.get_p_der_t(31) == target_lane_storage);
    REQUIRE_THAT(scalar_lane_value(target.get_der_t(31)), WithinAbs(7.0, 1e-10));

    target = target;
    REQUIRE(target.get_derivative_lane_count() == 32);
    REQUIRE_FALSE(target.has_der_t(1));
    REQUIRE(target.get_p_der_t(31) == target_lane_storage);
    REQUIRE_THAT(scalar_lane_value(target.get_der_t(31)), WithinAbs(7.0, 1e-10));
}

TEST_CASE("Term_eq value refresh preserves storage for an unnamed compatible tensor",
          "[term-eq][value-storage]")
{
    Space_spheric space = make_lane_test_space();
    Scalar initial = constant_scalar(space, 2.0);
    Scalar replacement = constant_scalar(space, 7.0);
    Term_eq term(0, initial);
    const Tensor* storage = term.get_p_val_t();

    term.set_val_t(replacement);

    REQUIRE(term.get_p_val_t() == storage);
    REQUIRE_THAT(scalar_lane_value(term.get_val_t()), WithinAbs(7.0, 1e-10));

    term.set_val_t(term.get_val_t());
    REQUIRE(term.get_p_val_t() == storage);
    REQUIRE_THAT(scalar_lane_value(term.get_val_t()), WithinAbs(7.0, 1e-10));
}

TEST_CASE("Term_eq derivative tile reset rejects invalid widths without mutation", "[term-eq][w-lane]")
{
    Term_eq term(0, 2.0, 3.0);
    term.set_der_d(31, 7.0);

    REQUIRE_THROWS(term.reset_derivative_tile_retain_storage(0));
    REQUIRE_THROWS(term.reset_derivative_tile_retain_storage(Term_eq::max_derivative_lanes + 1));
    REQUIRE_THROWS(term.reset_derivative_tile(0));
    REQUIRE_THROWS(term.reset_derivative_tile(Term_eq::max_derivative_lanes + 1));
    REQUIRE(term.get_derivative_lane_count() == 32);
    REQUIRE_THAT(term.get_der_d(0), WithinAbs(3.0, 1e-14));
    REQUIRE_THAT(term.get_der_d(31), WithinAbs(7.0, 1e-14));
}

TEST_CASE("Term_eq double arithmetic propagates second derivative lane", "[term-eq][w-lane]")
{
    Term_eq left(0, 2.0, 3.0);
    left.set_der_d(1, 5.0);

    Term_eq right(0, 7.0, 11.0);
    right.set_der_d(1, 13.0);

    Term_eq sum(left + right);
    REQUIRE(sum.get_derivative_lane_count() == 2);
    REQUIRE_THAT(sum.get_val_d(), WithinAbs(9.0, 1e-14));
    REQUIRE_THAT(sum.get_der_d(0), WithinAbs(14.0, 1e-14));
    REQUIRE_THAT(sum.get_der_d(1), WithinAbs(18.0, 1e-14));

    Term_eq difference(left - right);
    REQUIRE(difference.get_derivative_lane_count() == 2);
    REQUIRE_THAT(difference.get_val_d(), WithinAbs(-5.0, 1e-14));
    REQUIRE_THAT(difference.get_der_d(0), WithinAbs(-8.0, 1e-14));
    REQUIRE_THAT(difference.get_der_d(1), WithinAbs(-8.0, 1e-14));

    Term_eq product(left * right);
    REQUIRE(product.get_derivative_lane_count() == 2);
    REQUIRE_THAT(product.get_val_d(), WithinAbs(14.0, 1e-14));
    REQUIRE_THAT(product.get_der_d(0), WithinAbs(43.0, 1e-14));
    REQUIRE_THAT(product.get_der_d(1), WithinAbs(61.0, 1e-14));

    Term_eq quotient(left / right);
    REQUIRE(quotient.get_derivative_lane_count() == 2);
    REQUIRE_THAT(quotient.get_val_d(), WithinAbs(2.0 / 7.0, 1e-14));
    REQUIRE_THAT(quotient.get_der_d(0), WithinAbs((3.0 * 7.0 - 2.0 * 11.0) / 49.0, 1e-14));
    REQUIRE_THAT(quotient.get_der_d(1), WithinAbs((5.0 * 7.0 - 2.0 * 13.0) / 49.0, 1e-14));
}

TEST_CASE("Term_eq double lane arithmetic treats missing lanes as zero", "[term-eq][w-lane]")
{
    Term_eq variable(0, 2.0, 3.0);
    variable.set_der_d(1, 5.0);

    Term_eq constant(0, 7.0, 0.0);

    Term_eq sum(variable + constant);
    REQUIRE(sum.get_derivative_lane_count() == 2);
    REQUIRE_THAT(sum.get_der_d(1), WithinAbs(5.0, 1e-14));

    Term_eq product(variable * constant);
    REQUIRE(product.get_derivative_lane_count() == 2);
    REQUIRE_THAT(product.get_der_d(1), WithinAbs(35.0, 1e-14));

    Term_eq quotient(variable / constant);
    REQUIRE(quotient.get_derivative_lane_count() == 2);
    REQUIRE_THAT(quotient.get_der_d(1), WithinAbs(5.0 / 7.0, 1e-14));
}

TEST_CASE("Term_eq tensor arithmetic propagates second derivative lane", "[term-eq][w-lane]")
{
    Space_spheric space = make_lane_test_space();

    Term_eq left = scalar_field_term(space, 2.0, 3.0, 5.0);
    Term_eq right = scalar_field_term(space, 7.0, 11.0, 13.0);

    Term_eq sum(left + right);
    REQUIRE(sum.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(sum.get_der_t(1)), WithinAbs(18.0, 1e-10));

    Term_eq difference(left - right);
    REQUIRE(difference.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(difference.get_der_t(1)), WithinAbs(-8.0, 1e-10));

    Term_eq product(left * right);
    REQUIRE(product.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(product.get_der_t(1)), WithinAbs(61.0, 1e-10));

    Term_eq quotient(left / right);
    REQUIRE(quotient.get_derivative_lane_count() == 2);
    REQUIRE_THAT(
        scalar_lane_value(quotient.get_der_t(1)),
        WithinAbs((5.0 * 7.0 - 2.0 * 13.0) / 49.0, 1e-10));
}

TEST_CASE("Scalar tensor multiplication preserves parameters and non-finite zero semantics",
          "[term-eq][scalar-mult][fused-product-quotient]")
{
    Space_spheric space = make_lane_test_space();

    Scalar lhs = constant_scalar(space, 2.0);
    lhs.affect_parameters();
    lhs.set_parameters()->set_m_quant() = 2;
    Scalar rhs = constant_scalar(space, 7.0);
    rhs.affect_parameters();
    rhs.set_parameters()->set_m_quant() = 3;

    const Term_eq product(Term_eq(0, lhs) * Term_eq(0, rhs));
    REQUIRE(product.get_val_t().get_parameters() != nullptr);
    REQUIRE(product.get_val_t().get_parameters()->get_m_quant() == 5);
    REQUIRE_THAT(scalar_lane_value(product.get_val_t()), WithinAbs(14.0, 1e-10));

    const Scalar infinity = constant_scalar(
        space, std::numeric_limits<double>::infinity());
    const Scalar zero = constant_scalar(space, 0.0);
    const Term_eq zero_product(Term_eq(0, infinity) * Term_eq(0, zero));
    REQUIRE(zero_product.get_val_t()()(0).check_if_zero());

    const Scalar nan = constant_scalar(
        space, std::numeric_limits<double>::quiet_NaN());
    const Term_eq nan_product(Term_eq(0, nan) * Term_eq(0, rhs));
    REQUIRE(std::isnan(scalar_lane_value(nan_product.get_val_t())));
}

TEST_CASE("Direct scalar tensor result matches legacy Scalar slicing",
          "[term-eq][scalar-mult][fused-product-quotient]")
{
    Space_spheric space = make_lane_test_space();

    REQUIRE_THROWS(OneDomainScalarTensorProbe(space.get_nbr_domains(), space));

    Scalar lhs = constant_scalar(space, 2.0);
    lhs.affect_parameters();
    lhs.set_parameters()->set_m_quant() = -2;
    Scalar rhs = constant_scalar(space, 7.0);
    rhs.affect_parameters();
    rhs.set_parameters()->set_m_quant() = 3;

    const Tensor direct = mult_one_dom(0, lhs, rhs);
    const Tensor legacy = legacy_scalar_tensor_product(0, lhs, rhs);
    REQUIRE(direct.get_valence() == legacy.get_valence());
    REQUIRE(direct.get_n_comp() == legacy.get_n_comp());
    REQUIRE(direct.is_name_affected() == legacy.is_name_affected());
    REQUIRE(direct.get_basis() == legacy.get_basis());
    REQUIRE(direct.get_parameters() != nullptr);
    REQUIRE(legacy.get_parameters() != nullptr);
    REQUIRE(direct.get_parameters()->get_m_quant() ==
            legacy.get_parameters()->get_m_quant());
    REQUIRE(diffmax(direct()(0).get_conf(), legacy()(0).get_conf()) == 0.0);

    const Scalar plain_lhs = constant_scalar(space, 1.0);
    const Scalar subnormal = constant_scalar(
        space, std::numeric_limits<double>::denorm_min());
    const Tensor direct_subnormal = mult_one_dom(0, plain_lhs, subnormal);
    const Tensor legacy_subnormal =
        legacy_scalar_tensor_product(0, plain_lhs, subnormal);
    REQUIRE(direct_subnormal.get_parameters() == nullptr);
    REQUIRE(legacy_subnormal.get_parameters() == nullptr);
    const Array<double>& direct_subnormal_conf =
        direct_subnormal()(0).get_conf();
    const Array<double>& legacy_subnormal_conf =
        legacy_subnormal()(0).get_conf();
    REQUIRE(direct_subnormal_conf.get_nbr() == legacy_subnormal_conf.get_nbr());
    REQUIRE(std::memcmp(
                direct_subnormal_conf.get_data(),
                legacy_subnormal_conf.get_data(),
                direct_subnormal_conf.get_nbr() * sizeof(double)) == 0);

    Scalar negative_zero = constant_scalar(space, 1.0);
    Val_domain& negative_zero_domain = negative_zero.set_domain(0);
    Index point(negative_zero_domain.get_conf().get_dimensions());
    do {
        negative_zero_domain.set(point) = -0.0;
    } while (point.inc());
    const Tensor direct_negative_zero = mult_one_dom(0, negative_zero, plain_lhs);
    const Tensor legacy_negative_zero =
        legacy_scalar_tensor_product(0, negative_zero, plain_lhs);
    const Array<double>& direct_negative_zero_conf =
        direct_negative_zero()(0).get_conf();
    const Array<double>& legacy_negative_zero_conf =
        legacy_negative_zero()(0).get_conf();
    REQUIRE(direct_negative_zero_conf.get_nbr() ==
            legacy_negative_zero_conf.get_nbr());
    REQUIRE(std::memcmp(
                direct_negative_zero_conf.get_data(),
                legacy_negative_zero_conf.get_data(),
                direct_negative_zero_conf.get_nbr() * sizeof(double)) == 0);
    REQUIRE(std::signbit(direct_negative_zero_conf.get_data()[0]));

    const Scalar infinity = constant_scalar(
        space, std::numeric_limits<double>::infinity());
    const Scalar zero = constant_scalar(space, 0.0);
    const Tensor direct_zero = mult_one_dom(0, infinity, zero);
    const Tensor legacy_zero = legacy_scalar_tensor_product(0, infinity, zero);
    REQUIRE(direct_zero()(0).check_if_zero() == legacy_zero()(0).check_if_zero());

    const Scalar nan = constant_scalar(
        space, std::numeric_limits<double>::quiet_NaN());
    const Tensor direct_nan = mult_one_dom(0, nan, rhs);
    const Tensor legacy_nan = legacy_scalar_tensor_product(0, nan, rhs);
    REQUIRE(std::isnan(scalar_lane_value(direct_nan)));
    REQUIRE(std::isnan(scalar_lane_value(legacy_nan)));
}

TEST_CASE("Term_eq mixed double tensor arithmetic propagates second derivative lane", "[term-eq][w-lane]")
{
    Space_spheric space = make_lane_test_space();

    Term_eq double_factor(0, 2.0, 3.0);
    double_factor.set_der_d(1, 5.0);
    Term_eq tensor_factor = scalar_field_term(space, 7.0, 11.0, 13.0);

    Term_eq double_times_tensor(double_factor * tensor_factor);
    REQUIRE(double_times_tensor.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(double_times_tensor.get_der_t(1)), WithinAbs(61.0, 1e-10));

    Term_eq tensor_times_double(tensor_factor * double_factor);
    REQUIRE(tensor_times_double.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(tensor_times_double.get_der_t(1)), WithinAbs(61.0, 1e-10));

    Term_eq tensor_over_double(tensor_factor / double_factor);
    REQUIRE(tensor_over_double.get_derivative_lane_count() == 2);
    REQUIRE_THAT(
        scalar_lane_value(tensor_over_double.get_der_t(1)),
        WithinAbs((13.0 * 2.0 - 7.0 * 5.0) / 4.0, 1e-10));

    Term_eq double_over_tensor(double_factor / tensor_factor);
    REQUIRE(double_over_tensor.get_derivative_lane_count() == 2);
    REQUIRE_THAT(
        scalar_lane_value(double_over_tensor.get_der_t(1)),
        WithinAbs((5.0 * 7.0 - 2.0 * 13.0) / 49.0, 1e-10));
}

TEST_CASE("Term_eq fused product and quotient arithmetic preserves lane 31 across storage combinations",
          "[term-eq][w-lane][fused-product-quotient]")
{
    Space_spheric space = make_lane_test_space();

    Term_eq double_term(0, 2.0, 3.0);
    double_term.set_der_d(31, 5.0);
    Term_eq tensor_term = scalar_field_term(space, 7.0, 11.0, 13.0);
    tensor_term.set_der_t(31, constant_scalar(space, 13.0));

    const Term_eq double_times_tensor(double_term * tensor_term);
    const Term_eq tensor_times_double(tensor_term * double_term);
    const Term_eq tensor_times_tensor(tensor_term * tensor_term);
    const Term_eq double_over_tensor(double_term / tensor_term);
    const Term_eq tensor_over_double(tensor_term / double_term);
    const Term_eq tensor_over_tensor(tensor_term / tensor_term);

    REQUIRE(double_times_tensor.get_derivative_lane_count() == 32);
    REQUIRE(tensor_times_double.get_derivative_lane_count() == 32);
    REQUIRE(tensor_times_tensor.get_derivative_lane_count() == 32);
    REQUIRE(double_over_tensor.get_derivative_lane_count() == 32);
    REQUIRE(tensor_over_double.get_derivative_lane_count() == 32);
    REQUIRE(tensor_over_tensor.get_derivative_lane_count() == 32);

    REQUIRE_THAT(scalar_lane_value(double_times_tensor.get_der_t(0)), WithinAbs(43.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(double_times_tensor.get_der_t(31)), WithinAbs(61.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(tensor_times_double.get_der_t(0)), WithinAbs(43.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(tensor_times_double.get_der_t(31)), WithinAbs(61.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(tensor_times_tensor.get_der_t(0)), WithinAbs(154.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(tensor_times_tensor.get_der_t(31)), WithinAbs(182.0, 1e-10));

    REQUIRE_THAT(
        scalar_lane_value(double_over_tensor.get_der_t(0)),
        WithinAbs((3.0 * 7.0 - 2.0 * 11.0) / 49.0, 1e-10));
    REQUIRE_THAT(
        scalar_lane_value(double_over_tensor.get_der_t(31)),
        WithinAbs((5.0 * 7.0 - 2.0 * 13.0) / 49.0, 1e-10));
    REQUIRE_THAT(
        scalar_lane_value(tensor_over_double.get_der_t(0)),
        WithinAbs((11.0 * 2.0 - 7.0 * 3.0) / 4.0, 1e-10));
    REQUIRE_THAT(
        scalar_lane_value(tensor_over_double.get_der_t(31)),
        WithinAbs((13.0 * 2.0 - 7.0 * 5.0) / 4.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(tensor_over_tensor.get_der_t(0)), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(tensor_over_tensor.get_der_t(31)), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Term_eq fused product and quotient arithmetic preserves missing and explicit zero lanes",
          "[term-eq][w-lane][fused-product-quotient]")
{
    Space_spheric space = make_lane_test_space();
    const Scalar zero = constant_scalar(space, 0.0);

    Term_eq missing_double(0, 2.0, 3.0);
    missing_double.set_derivative_lane_count(32);
    Term_eq active_tensor = scalar_field_term(space, 7.0, 11.0, 13.0);
    active_tensor.set_der_t(31, constant_scalar(space, 13.0));

    const Term_eq product(missing_double * active_tensor);
    const Term_eq quotient(missing_double / active_tensor);
    REQUIRE(product.has_der_t(31));
    REQUIRE(quotient.has_der_t(31));
    REQUIRE_THAT(scalar_lane_value(product.get_der_t(31)), WithinAbs(26.0, 1e-10));
    REQUIRE_THAT(
        scalar_lane_value(quotient.get_der_t(31)),
        WithinAbs(-26.0 / 49.0, 1e-10));

    Term_eq missing_tensor(0, constant_scalar(space, 2.0), constant_scalar(space, 3.0));
    missing_tensor.set_derivative_lane_count(32);
    Term_eq explicit_zero(0, constant_scalar(space, 7.0), constant_scalar(space, 11.0));
    explicit_zero.set_derivative_lane_count(32);
    explicit_zero.set_der_t(31, zero);

    const Term_eq zero_product(missing_tensor * explicit_zero);
    const Term_eq zero_quotient(missing_tensor / explicit_zero);
    REQUIRE(zero_product.has_der_t(31));
    REQUIRE(zero_quotient.has_der_t(31));
    REQUIRE(zero_product.get_der_t(31)()(0).check_if_zero());
    REQUIRE(zero_quotient.get_der_t(31)()(0).check_if_zero());
}

TEST_CASE("Term_eq fused mixed arithmetic falls back for independent derivative layouts",
          "[term-eq][w-lane][fused-product-quotient][layout]")
{
    Space_spheric space = make_lane_test_space();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Tensor value(space, 2, COV, basis);
    CheckedMetricTensor derivative(space, COV, basis);
    value = 7.0;
    derivative = 11.0;
    value.std_base();
    derivative.std_base();

    Term_eq tensor_term(0, value, derivative);
    Term_eq scalar_term(0, 2.0, 3.0);
    const Term_eq product(tensor_term * scalar_term);
    const Term_eq quotient(tensor_term / scalar_term);

    REQUIRE(product.get_der_t(0).get_n_comp() == 9);
    REQUIRE(quotient.get_der_t(0).get_n_comp() == 9);
    for (int component = 0; component < 9; ++component) {
        REQUIRE_THAT(tensor_component_lane_value(product.get_der_t(0), component),
                     WithinAbs(43.0, 1e-10));
        REQUIRE_THAT(tensor_component_lane_value(quotient.get_der_t(0), component),
                     WithinAbs((11.0 * 2.0 - 7.0 * 3.0) / 4.0, 1e-10));
    }
}

TEST_CASE("Term_eq tensor scalar transforms propagate second derivative lane", "[term-eq][w-lane]")
{
    Space_spheric space = make_lane_test_space();

    Term_eq term = scalar_field_term(space, 4.0, 3.0, 5.0);

    Term_eq squared = pow(term, 2);
    REQUIRE(squared.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(squared.get_der_t(1)), WithinAbs(40.0, 1e-10));

    Term_eq inverse = pow(term, -1);
    REQUIRE(inverse.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(inverse.get_der_t(1)), WithinAbs(-5.0 / 16.0, 1e-10));

    Term_eq scaled = 3.0 * term;
    REQUIRE(scaled.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(scaled.get_der_t(1)), WithinAbs(15.0, 1e-10));

    Term_eq divided = term / 2.0;
    REQUIRE(divided.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(divided.get_der_t(1)), WithinAbs(2.5, 1e-10));

    Term_eq square_root = sqrt(term);
    REQUIRE(square_root.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(square_root.get_der_t(1)), WithinAbs(5.0 / 4.0, 1e-10));
}

TEST_CASE("Term_eq integer powers preserve sparse tensor derivative lanes",
          "[term-eq][w-lane][integer-power]")
{
    Space_spheric space = make_lane_test_space();
    Term_eq term = scalar_field_term(space, 2.0, 3.0, 5.0);
    term.set_der_t(31, constant_scalar(space, 7.0));

    const Term_eq positive = pow(term, 8);
    REQUIRE(positive.get_derivative_lane_count() == 32);
    REQUIRE_FALSE(positive.has_der_t(2));
    REQUIRE_THAT(scalar_lane_value(positive.get_val_t()), WithinAbs(256.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(positive.get_der_t(0)), WithinAbs(3072.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(positive.get_der_t(1)), WithinAbs(5120.0, 1e-10));
    REQUIRE_THAT(scalar_lane_value(positive.get_der_t(31)), WithinAbs(7168.0, 1e-10));

    const Term_eq negative = pow(term, -3);
    REQUIRE(negative.get_derivative_lane_count() == 32);
    REQUIRE_FALSE(negative.has_der_t(2));
    REQUIRE_THAT(scalar_lane_value(negative.get_val_t()), WithinAbs(0.125, 1e-10));
    REQUIRE_THAT(scalar_lane_value(negative.get_der_t(0)), WithinAbs(-0.5625, 1e-10));
    REQUIRE_THAT(scalar_lane_value(negative.get_der_t(1)), WithinAbs(-0.9375, 1e-10));
    REQUIRE_THAT(scalar_lane_value(negative.get_der_t(31)), WithinAbs(-1.3125, 1e-10));
}

TEST_CASE("Term_eq integer powers preserve sparse double derivative lanes",
          "[term-eq][w-lane][integer-power]")
{
    Term_eq term(0, 2.0, 3.0);
    term.set_der_d(1, 5.0);
    term.set_der_d(31, 7.0);

    const Term_eq positive = pow(term, 8);
    REQUIRE(positive.get_derivative_lane_count() == 32);
    REQUIRE_FALSE(positive.has_der_d(2));
    REQUIRE_THAT(positive.get_val_d(), WithinAbs(256.0, 0.0));
    REQUIRE_THAT(positive.get_der_d(0), WithinAbs(3072.0, 0.0));
    REQUIRE_THAT(positive.get_der_d(1), WithinAbs(5120.0, 0.0));
    REQUIRE_THAT(positive.get_der_d(31), WithinAbs(7168.0, 0.0));

    const Term_eq negative = pow(term, -3);
    REQUIRE(negative.get_derivative_lane_count() == 32);
    REQUIRE_FALSE(negative.has_der_d(2));
    REQUIRE_THAT(negative.get_val_d(), WithinAbs(0.125, 0.0));
    REQUIRE_THAT(negative.get_der_d(0), WithinAbs(-0.5625, 0.0));
    REQUIRE_THAT(negative.get_der_d(1), WithinAbs(-0.9375, 0.0));
    REQUIRE_THAT(negative.get_der_d(31), WithinAbs(-1.3125, 0.0));
}

TEST_CASE("Term_eq integer powers retain finite and IEEE edge behavior",
          "[term-eq][w-lane][integer-power]")
{
    Term_eq negative_base(0, -2.0, 3.0);
    REQUIRE_THAT(pow(negative_base, 8).get_val_d(), WithinAbs(256.0, 0.0));
    REQUIRE_THAT(pow(negative_base, 8).get_der_d(0), WithinAbs(-3072.0, 0.0));
    REQUIRE_THAT(pow(negative_base, -3).get_val_d(), WithinAbs(-0.125, 0.0));
    REQUIRE_THAT(pow(negative_base, -3).get_der_d(0), WithinAbs(-0.5625, 0.0));

    Term_eq zero_base(0, 0.0, 1.0);
    const Term_eq zero_positive = pow(zero_base, 8);
    REQUIRE_THAT(zero_positive.get_val_d(), WithinAbs(0.0, 0.0));
    REQUIRE_THAT(zero_positive.get_der_d(0), WithinAbs(0.0, 0.0));
    const Term_eq zero_negative = pow(zero_base, -3);
    REQUIRE(std::isinf(zero_negative.get_val_d()));
    REQUIRE_FALSE(std::signbit(zero_negative.get_val_d()));
    REQUIRE(std::isinf(zero_negative.get_der_d(0)));
    REQUIRE(std::signbit(zero_negative.get_der_d(0)));

    Term_eq nan_base(0, std::numeric_limits<double>::quiet_NaN(), 1.0);
    REQUIRE(std::isnan(pow(nan_base, 8).get_val_d()));
    REQUIRE(std::isnan(pow(nan_base, 8).get_der_d(0)));
    REQUIRE(std::isnan(pow(nan_base, -3).get_val_d()));
    REQUIRE(std::isnan(pow(nan_base, -3).get_der_d(0)));

    Term_eq infinite_base(0, std::numeric_limits<double>::infinity(), 1.0);
    const Term_eq infinite_positive = pow(infinite_base, 8);
    REQUIRE(std::isinf(infinite_positive.get_val_d()));
    REQUIRE(std::isinf(infinite_positive.get_der_d(0)));
    const Term_eq infinite_negative = pow(infinite_base, -3);
    REQUIRE_THAT(infinite_negative.get_val_d(), WithinAbs(0.0, 0.0));
    REQUIRE_THAT(infinite_negative.get_der_d(0), WithinAbs(0.0, 0.0));
    REQUIRE(std::signbit(infinite_negative.get_der_d(0)));

    Term_eq sparse(0, 2.0, 3.0);
    sparse.set_der_d(31, 7.0);
    const Term_eq identity = pow(sparse, 1);
    REQUIRE(identity.has_der_d(31));
    REQUIRE_THAT(identity.get_der_d(31), WithinAbs(7.0, 0.0));
    const Term_eq constant = pow(sparse, 0);
    REQUIRE(constant.get_derivative_lane_count() == 1);
    REQUIRE_THAT(constant.get_val_d(), WithinAbs(1.0, 0.0));
    REQUIRE_THAT(constant.get_der_d(0), WithinAbs(0.0, 0.0));

    REQUIRE_THROWS(pow(sparse, std::numeric_limits<int>::min()));
    REQUIRE_THROWS(pow(sparse, std::numeric_limits<int>::min() + 1));
}

TEST_CASE("Scalar unary operator helper propagates second derivative lane", "[term-eq][w-lane]")
{
    Space_spheric space = make_lane_test_space();
    Term_eq tensor_term = scalar_field_term(space, 2.0, 3.0, 5.0);

    Term_eq tensor_exp = detail::apply_scalar_unary_operator(
        0,
        tensor_term,
        "test_exp",
        [](const auto& value) { return exp(value); },
        [](const auto& derivative, const auto& value) { return derivative * exp(value); });
    REQUIRE(tensor_exp.get_derivative_lane_count() == 2);
    REQUIRE_THAT(scalar_lane_value(tensor_exp.get_der_t(1)), WithinAbs(5.0 * exp(2.0), 1e-10));

    Term_eq double_term(0, 2.0, 3.0);
    double_term.set_der_d(1, 5.0);
    Term_eq double_exp = detail::apply_scalar_unary_operator(
        0,
        double_term,
        "test_exp",
        [](const auto& value) { return exp(value); },
        [](const auto& derivative, const auto& value) { return derivative * exp(value); });
    REQUIRE(double_exp.get_derivative_lane_count() == 2);
    REQUIRE_THAT(double_exp.get_der_d(1), WithinAbs(5.0 * exp(2.0), 1e-14));
}

TEST_CASE("Generic domain derivative helper preserves second tensor lane", "[term-eq][w-lane]")
{
    Space_spheric space = make_lane_test_space();
    Scalar value(space);
    value = 2.0;
    value.std_base();

    Scalar first_derivative(space);
    first_derivative = 3.0;
    first_derivative.std_base();

    Scalar second_derivative(space);
    second_derivative = 7.0;
    second_derivative.std_base();

    Term_eq term(0, value, first_derivative);
    term.set_der_t(1, second_derivative);

    Term_eq radial_derivative(space.get_domain(0)->dr_term_eq(term));
    REQUIRE(radial_derivative.get_derivative_lane_count() == 2);
    REQUIRE(radial_derivative.get_p_der_t(0) != nullptr);
    REQUIRE(radial_derivative.get_p_der_t(1) != nullptr);
}

TEST_CASE("Domain partial derivatives propagate second tensor lane", "[term-eq][w-lane]")
{
    Space_spheric space = make_lane_test_space();

    Scalar value = constant_scalar(space, 1.0);
    Scalar first_derivative = radial_scalar(space);
    Scalar second_derivative = cartesian_x_scalar(space);

    Term_eq packed(0, value, first_derivative);
    packed.set_der_t(1, second_derivative);
    Term_eq reference(0, value, second_derivative);

    Term_eq packed_cartesian(space.get_domain(0)->partial_cart(packed));
    Term_eq reference_cartesian(space.get_domain(0)->partial_cart(reference));
    REQUIRE(packed_cartesian.get_derivative_lane_count() == 2);
    REQUIRE(packed_cartesian.get_p_der_t(1) != nullptr);
    REQUIRE_THAT(
        tensor_component_lane_value(packed_cartesian.get_der_t(1), 0),
        WithinAbs(tensor_component_lane_value(reference_cartesian.get_der_t(), 0), 1e-10));

    Term_eq packed_spherical(space.get_domain(0)->partial_spher(packed));
    Term_eq reference_spherical(space.get_domain(0)->partial_spher(reference));
    REQUIRE(packed_spherical.get_derivative_lane_count() == 2);
    REQUIRE(packed_spherical.get_p_der_t(1) != nullptr);
    REQUIRE_THAT(
        tensor_component_lane_value(packed_spherical.get_der_t(1), 0),
        WithinAbs(tensor_component_lane_value(reference_spherical.get_der_t(), 0), 1e-10));
}

TEST_CASE("Flat metric derivatives preserve second tensor lane", "[term-eq][w-lane]")
{
    Space_spheric space = make_lane_test_space();
    Base_tensor basis(space, SPHERICAL_BASIS);
    Metric_flat metric(space, basis);

    Scalar value = constant_scalar(space, 1.0);
    Scalar first_derivative = radial_scalar(space);
    Scalar second_derivative = cartesian_x_scalar(space);

    Term_eq packed(0, value, first_derivative);
    packed.set_der_t(1, second_derivative);
    Term_eq reference(0, value, second_derivative);

    Term_eq packed_derivative(metric.derive(COV, 'i', packed));
    Term_eq reference_derivative(metric.derive(COV, 'i', reference));
    REQUIRE(packed_derivative.get_derivative_lane_count() == 2);
    REQUIRE(packed_derivative.get_p_der_t(1) != nullptr);
    REQUIRE_THAT(
        tensor_component_lane_value(packed_derivative.get_der_t(1), 0),
        WithinAbs(tensor_component_lane_value(reference_derivative.get_der_t(), 0), 1e-10));
}
