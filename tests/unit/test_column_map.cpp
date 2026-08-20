#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "Linear_algebra/jacobian_assembler.hpp"

#include <cstring>

using namespace Kadath;

namespace
{
Space_spheric make_structural_plan_cache_space()
{
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;
    Dim_array resolution(3);
    resolution.set(0) = 5;
    resolution.set(1) = 5;
    resolution.set(2) = 4;
    Dim_array bounds_shape(1);
    bounds_shape.set(0) = 1;
    Array<double> bounds(bounds_shape);
    bounds.set(0) = 1.0;
    return Space_spheric(CHEB_TYPE, center, resolution, bounds);
}

void require_same_structural_plan(
    const JacobianAssemblerStructuralPlan& expected,
    const JacobianAssemblerStructuralPlan& actual)
{
    REQUIRE(actual.direct_singleton_plan.columns.size() ==
            expected.direct_singleton_plan.columns.size());
    REQUIRE(actual.direct_singleton_plan.entries.size() ==
            expected.direct_singleton_plan.entries.size());
    for (std::size_t index = 0;
         index < expected.direct_singleton_plan.columns.size(); ++index) {
        const DirectJacobianColumn& expected_column =
            expected.direct_singleton_plan.columns[index];
        const DirectJacobianColumn& actual_column =
            actual.direct_singleton_plan.columns[index];
        REQUIRE(actual_column.first_entry_index ==
                expected_column.first_entry_index);
        REQUIRE(actual_column.entry_count == expected_column.entry_count);
    }
    for (std::size_t index = 0;
         index < expected.direct_singleton_plan.entries.size(); ++index) {
        const DirectJacobianEntry& expected_entry =
            expected.direct_singleton_plan.entries[index];
        const DirectJacobianEntry& actual_entry =
            actual.direct_singleton_plan.entries[index];
        REQUIRE(actual_entry.row == expected_entry.row);
        REQUIRE(std::memcmp(&actual_entry.value, &expected_entry.value,
                            sizeof(double)) == 0);
    }
    REQUIRE(actual.column_metadata.size() == expected.column_metadata.size());
    for (std::size_t index = 0; index < expected.column_metadata.size(); ++index) {
        const ColumnMetadata& expected_column = expected.column_metadata[index];
        const ColumnMetadata& actual_column = actual.column_metadata[index];
        REQUIRE(actual_column.column == expected_column.column);
        REQUIRE(actual_column.column_class == expected_column.column_class);
        REQUIRE(actual_column.domain == expected_column.domain);
        REQUIRE(actual_column.term_idx == expected_column.term_idx);
        REQUIRE(actual_column.var_idx == expected_column.var_idx);
        REQUIRE(actual_column.var_double_idx == expected_column.var_double_idx);
        REQUIRE(actual_column.vardom_param == expected_column.vardom_param);
        REQUIRE(actual_column.var_name == expected_column.var_name);
        REQUIRE(actual_column.basis_mode == expected_column.basis_mode);
        REQUIRE(actual_column.domain_type_id == expected_column.domain_type_id);
        REQUIRE(actual_column.tensor_component == expected_column.tensor_component);
        REQUIRE(actual_column.coefficient_i == expected_column.coefficient_i);
        REQUIRE(actual_column.coefficient_j == expected_column.coefficient_j);
        REQUIRE(actual_column.coefficient_k == expected_column.coefficient_k);
        REQUIRE(actual_column.coefficient_nr == expected_column.coefficient_nr);
        REQUIRE(actual_column.coefficient_nt == expected_column.coefficient_nt);
        REQUIRE(actual_column.coefficient_np == expected_column.coefficient_np);
        REQUIRE(actual_column.reason == expected_column.reason);
    }
}
} // namespace

TEST_CASE("column map value types keep default contracts", "[column-map]")
{
    ColumnInfo column_info;
    CHECK(column_info.var_idx == -1);
    CHECK(column_info.var_double_idx == -1);
    CHECK(column_info.domain == -1);
    CHECK(column_info.term_idx == -1);
    CHECK(column_info.basis_mode == -1);
    CHECK(column_info.domain_type_id == static_cast<int>(ColumnDomainType::Unknown));
    CHECK(column_info.tensor_component == -1);
    CHECK(column_info.coefficient_i == -1);
    CHECK(column_info.coefficient_j == -1);
    CHECK(column_info.coefficient_k == -1);
    CHECK(column_info.coefficient_nr == -1);
    CHECK(column_info.coefficient_nt == -1);
    CHECK(column_info.coefficient_np == -1);
    CHECK(column_info.field_class == ColumnClass::FieldUnknown);
    CHECK(column_info.var_name.empty());
    CHECK_FALSE(column_info.is_var_domain);

    ColumnMetadata metadata;
    CHECK(metadata.column == -1);
    CHECK(metadata.column_class == ColumnClass::Unknown);
    CHECK(metadata.domain == -1);
    CHECK(metadata.domain_type_id == static_cast<int>(ColumnDomainType::Unknown));
    CHECK(metadata.tensor_component == -1);
    CHECK(metadata.coefficient_i == -1);
    CHECK(metadata.coefficient_j == -1);
    CHECK(metadata.coefficient_k == -1);
    CHECK(metadata.coefficient_nr == -1);
    CHECK(metadata.coefficient_nt == -1);
    CHECK(metadata.coefficient_np == -1);
    CHECK(metadata.reason == "unknown");
}

TEST_CASE("cold local COO reserve stages capacity before the final density sample",
          "[jacobian-assembler][coo-reserve]")
{
    using namespace jacobian_assembler_detail;

    const LocalNnzReserveSamplePlan empty =
        make_local_nnz_reserve_sample_plan(0);
    CHECK(empty.provisional_columns == 0);
    CHECK(empty.final_columns == 0);

    const LocalNnzReserveSamplePlan small =
        make_local_nnz_reserve_sample_plan(7);
    CHECK(small.provisional_columns == 1);
    CHECK(small.final_columns == 1);

    const LocalNnzReserveSamplePlan ns2d_sized =
        make_local_nnz_reserve_sample_plan(4375);
    CHECK(ns2d_sized.provisional_columns == 136);
    CHECK(ns2d_sized.final_columns == 546);
    CHECK(ns2d_sized.provisional_columns < ns2d_sized.final_columns);

    const std::size_t full_estimate =
        extrapolate_local_nnz_reserve(1000, 10, 100, 20000);
    CHECK(full_estimate == 11250);
    const std::size_t provisional =
        provisional_local_nnz_reserve(1000, full_estimate);
    CHECK(provisional == 3563);
    CHECK(provisional > 1000);
    CHECK(provisional < full_estimate);

    // A packed group may advance the accounted-column count past a sampling
    // boundary. The estimator uses the actual accounted count, not the nominal
    // threshold, so nnz accounting remains exact for such a leap.
    CHECK(extrapolate_local_nnz_reserve(3200, 32, 100, 20000) == 11250);
    CHECK(extrapolate_local_nnz_reserve(25000, 32, 100, 20000) == 20000);
    CHECK(provisional_local_nnz_reserve(17, 17) == 17);
}

TEST_CASE("column map production methods stay available", "[column-map]")
{
    using BuildColumnMap =
        void (System_of_eqs::*)(std::vector<ColumnInfo>&, bool) const;
    using BuildColumnRowIncidence =
        void (System_of_eqs::*)(const std::vector<ColumnInfo>&,
                                std::vector<std::set<int>>&) const;

    BuildColumnMap build_column_map = &System_of_eqs::build_column_map;
    BuildColumnRowIncidence build_column_row_incidence =
        &System_of_eqs::build_column_row_incidence;

    CHECK(build_column_map != nullptr);
    CHECK(build_column_row_incidence != nullptr);
}

TEST_CASE("assembler structural plan cache hits and preserves uncached output",
          "[column-map][jacobian-structural-plan-cache]")
{
    Space_spheric space = make_structural_plan_cache_space();
    System_of_eqs system(space, 0, 0);
    Scalar unknown(space);
    unknown = 1.0;
    unknown.std_base();
    unknown.coef();
    system.add_var("u", unknown);
    system.add_eq_full(0, "u = 0");
    (void)system.sec_member();

    JacobianAssemblerStructuralPlanAccess first_access;
    const JacobianAssemblerStructuralPlan& first =
        system.get_jacobian_assembler_structural_plan(
            true, true, first_access);
    REQUIRE_FALSE(first_access.cache_hit);
    REQUIRE(first_access.cache_miss_build_seconds >= 0.0);
    const JacobianAssemblerStructuralPlan expected = first;

    JacobianAssemblerStructuralPlanAccess second_access;
    const JacobianAssemblerStructuralPlan& second =
        system.get_jacobian_assembler_structural_plan(
            true, true, second_access);
    REQUIRE(second_access.cache_hit);
    REQUIRE(second_access.cache_check_seconds >= 0.0);
    REQUIRE(second_access.cache_miss_build_seconds == 0.0);
    REQUIRE(&second == &first);

    JacobianAssemblerStructuralPlanAccess uncached_access;
    const JacobianAssemblerStructuralPlan& uncached =
        system.get_jacobian_assembler_structural_plan(
            true, false, uncached_access);
    REQUIRE_FALSE(uncached_access.cache_hit);
    REQUIRE(uncached_access.cache_check_seconds == 0.0);
    require_same_structural_plan(expected, uncached);

    JacobianAssemblerStructuralPlanAccess after_disable_access;
    (void)system.get_jacobian_assembler_structural_plan(
        true, true, after_disable_access);
    REQUIRE_FALSE(after_disable_access.cache_hit);
}

TEST_CASE("assembler structural plan cache rejects a spectral basis change",
          "[column-map][jacobian-structural-plan-cache]")
{
    Space_spheric space = make_structural_plan_cache_space();
    System_of_eqs system(space, 0, 0);
    Scalar unknown(space);
    unknown = 1.0;
    unknown.std_base();
    unknown.coef();
    system.add_var("u", unknown);
    system.add_eq_full(0, "u = 0");
    (void)system.sec_member();

    JacobianAssemblerStructuralPlanAccess initial_access;
    (void)system.get_jacobian_assembler_structural_plan(
        true, true, initial_access);
    REQUIRE_FALSE(initial_access.cache_hit);

    Tensor* term_value = system.give_term(0, 0)->set_val_t();
    REQUIRE(term_value != nullptr);
    term_value->set().set_domain(0).std_anti_base();

    JacobianAssemblerStructuralPlanAccess changed_basis_access;
    (void)system.get_jacobian_assembler_structural_plan(
        true, true, changed_basis_access);
    REQUIRE_FALSE(changed_basis_access.cache_hit);

    JacobianAssemblerStructuralPlanAccess stable_basis_access;
    (void)system.get_jacobian_assembler_structural_plan(
        true, true, stable_basis_access);
    REQUIRE(stable_basis_access.cache_hit);
}
