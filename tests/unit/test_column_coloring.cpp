#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

using namespace Kadath;

TEST_CASE("column coloring cache type keeps default contract", "[column-coloring]")
{
    ColumnColoring coloring;

    CHECK(coloring.color.empty());
    CHECK(coloring.groups.empty());
    CHECK(coloring.num_colors == 0);
    CHECK(coloring.column_map.empty());
    CHECK_FALSE(coloring.initialized);
    CHECK(coloring.system_ptr == nullptr);
    CHECK(coloring.rows_per_column.empty());
    CHECK(coloring.col_to_domain.empty());
}

TEST_CASE("column coloring public methods stay available", "[column-coloring]")
{
    using ComputeColumnColoring =
        void (System_of_eqs::*)(ColumnColoring&) const;
    using GetColumnColoring =
        const ColumnColoring& (System_of_eqs::*)();
    using ResetColumnColoring =
        void (System_of_eqs::*)();
    using ValidateColumnColoring =
        bool (System_of_eqs::*)(int, double, double, std::ostream*);

    ComputeColumnColoring compute_column_coloring =
        &System_of_eqs::compute_column_coloring;
    GetColumnColoring get_column_coloring =
        &System_of_eqs::get_column_coloring;
    ResetColumnColoring reset_column_coloring =
        &System_of_eqs::reset_column_coloring;
    ValidateColumnColoring validate_column_coloring =
        &System_of_eqs::validate_column_coloring;

    CHECK(compute_column_coloring != nullptr);
    CHECK(get_column_coloring != nullptr);
    CHECK(reset_column_coloring != nullptr);
    CHECK(validate_column_coloring != nullptr);
}
