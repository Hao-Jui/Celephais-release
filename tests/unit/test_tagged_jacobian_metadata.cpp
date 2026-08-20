#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <type_traits>

using namespace Kadath;

TEST_CASE("tagged Jacobian metadata aliases preserve System_of_eqs API", "[tagged-jacobian-metadata]")
{
    static_assert(std::is_same_v<System_of_eqs::RowClass, RowClass>);
    static_assert(std::is_same_v<System_of_eqs::RowTaxonomy, RowTaxonomy>);
    static_assert(std::is_same_v<System_of_eqs::RowMetadata, RowMetadata>);
    static_assert(std::is_same_v<System_of_eqs::TaggedJacobianMetadata, TaggedJacobianMetadata>);
    static_assert(std::is_same_v<System_of_eqs::TaggedJacobianMetadataValidation,
                                 TaggedJacobianMetadataValidation>);
    static_assert(std::is_same_v<System_of_eqs::IncidenceColumnPartition, IncidenceColumnPartition>);

    RowMetadata row;
    CHECK(row.row == -1);
    CHECK(row.legacy_class == System_of_eqs::RowClass::Constraint);
    CHECK(row.taxonomy == System_of_eqs::RowTaxonomy::Unknown);

    TaggedJacobianMetadata metadata;
    CHECK(metadata.nrows == 0);
    CHECK(metadata.ncols == 0);
    CHECK(metadata.rows.empty());
    CHECK(metadata.columns.empty());
    CHECK(metadata.rows_per_column.empty());

    TaggedJacobianMetadataValidation validation;
    CHECK_FALSE(validation.ok);
    CHECK(validation.errors.empty());
}

TEST_CASE("ColumnClass::FieldOuterShellTau enum value exists",
          "[tagged-jacobian-metadata][outer-shell-tau]")
{
    // Sanity: enum lists the new class. promote_outer_shell_tau_columns is
    // file-local to tagged_jacobian_metadata.cpp; coverage of its behavior is
    // exercised via the BH2d/BNS integration run plus the synthetic case in
    // test_interface_partition.cpp (`FieldOuterShellTau col follows incidence transfer role`).
    Kadath::ColumnMetadata col;
    col.column_class = Kadath::ColumnClass::FieldOuterShellTau;
    CHECK(col.column_class == Kadath::ColumnClass::FieldOuterShellTau);
}

TEST_CASE("tagged Jacobian metadata methods stay available", "[tagged-jacobian-metadata]")
{
    using BuildTaggedMetadata =
        void (System_of_eqs::*)(TaggedJacobianMetadata&, bool) const;
    using ValidateTaggedMetadata =
        TaggedJacobianMetadataValidation (System_of_eqs::*)(const TaggedJacobianMetadata&) const;
    using BuildPartition =
        void (System_of_eqs::*)(const std::vector<RowMetadata>&,
                                const std::vector<ColumnMetadata>&,
                                const std::vector<std::set<int>>&,
                                IncidenceColumnPartition&) const;

    BuildTaggedMetadata build_metadata = &System_of_eqs::build_tagged_jacobian_metadata;
    ValidateTaggedMetadata validate_metadata = &System_of_eqs::validate_tagged_jacobian_metadata;
    BuildPartition build_partition = &System_of_eqs::build_incidence_column_partition;

    CHECK(build_metadata != nullptr);
    CHECK(validate_metadata != nullptr);
    CHECK(build_partition != nullptr);
}
