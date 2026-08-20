#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

using namespace Kadath;

TEST_CASE("column taxonomy public entry points stay available", "[column-taxonomy]")
{
    using ClassifyFieldColumn =
        ColumnClass (System_of_eqs::*)(int, int, int) const;
    using VolumeSelection =
        bool (System_of_eqs::*)(int, int, int, int) const;
    using EquationExport =
        bool (System_of_eqs::*)(int, int, int, int, bool) const;

    ClassifyFieldColumn classify_field_column =
        &System_of_eqs::classify_field_column_from_equations;
    VolumeSelection selected_by_volume =
        &System_of_eqs::field_coefficient_selected_by_volume_basis;
    EquationExport exported_by_equation =
        &System_of_eqs::field_coefficient_exported_by_equation;

    CHECK(classify_field_column != nullptr);
    CHECK(selected_by_volume != nullptr);
    CHECK(exported_by_equation != nullptr);
}
