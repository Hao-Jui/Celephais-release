#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

using namespace Kadath;

namespace {
Space_spheric make_two_domain_space()
{
    Point center(3);
    center.set(1) = 0.0;
    center.set(2) = 0.0;
    center.set(3) = 0.0;

    Dim_array res(3);
    res.set(0) = 5;
    res.set(1) = 5;
    res.set(2) = 4;

    Dim_array bounds_dim(1);
    bounds_dim.set(0) = 2;
    Array<double> bounds(bounds_dim);
    bounds.set(0) = 1.0;
    bounds.set(1) = 10.0;

    return Space_spheric(CHEB_TYPE, center, res, bounds);
}

bool is_explicit_field_column(ColumnClass column_class)
{
    return column_class == ColumnClass::FieldInteriorVol ||
           column_class == ColumnClass::FieldBoundaryTau ||
           column_class == ColumnClass::FieldMatching ||
           column_class == ColumnClass::FieldGauge;
}
} // namespace

TEST_CASE("System_of_eqs classifies DSL equation row ranges independently of columns",
          "[system_of_eqs][row-classification]")
{
    Space_spheric space = make_two_domain_space();
    System_of_eqs sys(space, 0, 1);
    Scalar u(space);
    u = 0.0;
    u.std_base();
    u.coef();
    sys.add_var("u", u);

    sys.add_eq_inside(0, "u=0");
    sys.add_eq_order(1, 2, "u=0");
    sys.add_eq_bc(1, OUTER_BC, "u=0");
    sys.add_eq_matching(0, OUTER_BC, "u");
    (void)sys.sec_member();

    std::vector<System_of_eqs::RowClass> rows;
    sys.classify_equation_rows(rows);

    REQUIRE(static_cast<int>(rows.size()) == sys.get_nbr_conditions());
    const int n_galerkin = static_cast<int>(std::count(
        rows.begin(), rows.end(), System_of_eqs::RowClass::Galerkin));
    const int n_constraint = static_cast<int>(std::count(
        rows.begin(), rows.end(), System_of_eqs::RowClass::Constraint));

    REQUIRE(n_galerkin > 0);
    REQUIRE(n_constraint > 0);
    REQUIRE(n_galerkin + n_constraint == sys.get_nbr_conditions());

    std::vector<System_of_eqs::RowMetadata> row_metadata;
    sys.classify_equation_row_metadata(row_metadata);
    REQUIRE(static_cast<int>(row_metadata.size()) == sys.get_nbr_conditions());
    REQUIRE(std::count_if(row_metadata.begin(), row_metadata.end(), [](const auto& row) {
        return row.taxonomy == System_of_eqs::RowTaxonomy::Vol;
    }) > 0);
    REQUIRE(std::count_if(row_metadata.begin(), row_metadata.end(), [](const auto& row) {
        return row.taxonomy == System_of_eqs::RowTaxonomy::TauBc;
    }) > 0);
    REQUIRE(std::count_if(row_metadata.begin(), row_metadata.end(), [](const auto& row) {
        return row.taxonomy == System_of_eqs::RowTaxonomy::TauMatch;
    }) > 0);

    std::vector<ColumnMetadata> column_metadata;
    sys.classify_columns(column_metadata);
    REQUIRE(static_cast<int>(column_metadata.size()) == sys.get_nbr_unknowns());
    REQUIRE(std::count_if(column_metadata.begin(), column_metadata.end(), [](const auto& col) {
        return is_explicit_field_column(col.column_class) && col.domain >= 0;
    }) == sys.get_nbr_unknowns());
    REQUIRE(std::count_if(column_metadata.begin(), column_metadata.end(), [](const auto& col) {
        return col.column_class == ColumnClass::FieldInteriorVol && col.domain >= 0;
    }) > 0);
    REQUIRE(std::count_if(column_metadata.begin(), column_metadata.end(), [](const auto& col) {
        return col.column_class == ColumnClass::FieldUnknown && col.domain >= 0;
    }) == 0);
}

TEST_CASE("System_of_eqs builds validated tagged Jacobian metadata from existing classifiers",
          "[system_of_eqs][tagged-jacobian-metadata]")
{
    Space_spheric space = make_two_domain_space();
    System_of_eqs sys(space, 0, 1);
    Scalar u(space);
    u = 0.0;
    u.std_base();
    u.coef();
    sys.add_var("u", u);

    sys.add_eq_inside(0, "u=0");
    sys.add_eq_order(1, 2, "u=0");
    sys.add_eq_bc(1, OUTER_BC, "u=0");
    sys.add_eq_matching(0, OUTER_BC, "u");
    (void)sys.sec_member();

    System_of_eqs::TaggedJacobianMetadata metadata;
    sys.build_tagged_jacobian_metadata(metadata);

    REQUIRE(metadata.nrows == sys.get_nbr_conditions());
    REQUIRE(metadata.ncols == sys.get_nbr_unknowns());
    REQUIRE(static_cast<int>(metadata.rows.size()) == sys.get_nbr_conditions());
    REQUIRE(static_cast<int>(metadata.columns.size()) == sys.get_nbr_unknowns());
    REQUIRE(static_cast<int>(metadata.rows_per_column.size()) == sys.get_nbr_unknowns());

    for (int row = 0; row < sys.get_nbr_conditions(); ++row) {
        REQUIRE(metadata.rows[static_cast<std::size_t>(row)].row == row);
    }
    for (int col = 0; col < sys.get_nbr_unknowns(); ++col) {
        REQUIRE(metadata.columns[static_cast<std::size_t>(col)].column == col);
    }

    const auto validation = sys.validate_tagged_jacobian_metadata(metadata);
    REQUIRE(validation.ok);
    REQUIRE(validation.expected_rows == sys.get_nbr_conditions());
    REQUIRE(validation.expected_cols == sys.get_nbr_unknowns());
    REQUIRE(validation.actual_rows == sys.get_nbr_conditions());
    REQUIRE(validation.actual_cols == sys.get_nbr_unknowns());
    REQUIRE(validation.row_support_columns == sys.get_nbr_unknowns());
    REQUIRE(validation.row_support_entries > 0);

    REQUIRE(std::count_if(metadata.rows.begin(), metadata.rows.end(), [](const auto& row) {
        return row.taxonomy == System_of_eqs::RowTaxonomy::Vol;
    }) > 0);
    REQUIRE(std::count_if(metadata.rows.begin(), metadata.rows.end(), [](const auto& row) {
        return row.taxonomy == System_of_eqs::RowTaxonomy::TauBc;
    }) > 0);
    REQUIRE(std::count_if(metadata.rows.begin(), metadata.rows.end(), [](const auto& row) {
        return row.taxonomy == System_of_eqs::RowTaxonomy::TauMatch;
    }) > 0);
    REQUIRE(std::count_if(metadata.columns.begin(), metadata.columns.end(), [](const auto& col) {
        return is_explicit_field_column(col.column_class) && col.domain >= 0;
    }) == sys.get_nbr_unknowns());
    REQUIRE(std::count_if(metadata.columns.begin(), metadata.columns.end(), [](const auto& col) {
        return col.column_class == ColumnClass::FieldInteriorVol && col.domain >= 0;
    }) > 0);
    REQUIRE(std::count_if(metadata.columns.begin(), metadata.columns.end(), [](const auto& col) {
        return col.column_class == ColumnClass::FieldUnknown && col.domain >= 0;
    }) == 0);

    std::ostringstream summary;
    sys.dump_tagged_jacobian_metadata_summary(summary);
    REQUIRE(summary.str().find("Validation: ok") != std::string::npos);
    REQUIRE(summary.str().find("Vol=") != std::string::npos);
    REQUIRE(summary.str().find("FieldInteriorVol=") != std::string::npos);

    std::ostringstream row_csv;
    std::ostringstream col_csv;
    sys.dump_tagged_jacobian_metadata_csv(row_csv, col_csv);
    REQUIRE(row_csv.str().find("row,taxonomy,legacy_class") == 0);
    REQUIRE(row_csv.str().find("TauMatch") != std::string::npos);
    REQUIRE(col_csv.str().find("column,column_class,domain") == 0);
    REQUIRE(col_csv.str().find("FieldInteriorVol") != std::string::npos);
}
