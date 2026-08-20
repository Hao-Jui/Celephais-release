#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "For_Kadath/Matrice/matrice.hpp"

using namespace Kadath;
using Catch::Matchers::WithinAbs;

TEST_CASE("Matrice LU solve", "[matrice]") {
    // 2x2 system: [[2,1],[1,3]] * [x,y] = [5, 10] -> x=1, y=3
    Matrice m(2, 2);
    m.set(0,0) = 2; m.set(0,1) = 1;
    m.set(1,0) = 1; m.set(1,1) = 3;

    Dim_array dims(1); dims.set(0) = 2;
    Array<double> rhs(dims);
    rhs.set(0) = 5; rhs.set(1) = 10;

    m.set_lu();
    Array<double> x = m.solve(rhs);
    REQUIRE_THAT(x(0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(x(1), WithinAbs(3.0, 1e-12));
}

TEST_CASE("Matrice 3x3 solve", "[matrice]") {
    // 3x3 system: identity matrix * x = [1,2,3] -> x = [1,2,3]
    Matrice m(3, 3);
    m.set(0,0) = 1; m.set(0,1) = 0; m.set(0,2) = 0;
    m.set(1,0) = 0; m.set(1,1) = 1; m.set(1,2) = 0;
    m.set(2,0) = 0; m.set(2,1) = 0; m.set(2,2) = 1;

    Dim_array dims(1); dims.set(0) = 3;
    Array<double> rhs(dims);
    rhs.set(0) = 1; rhs.set(1) = 2; rhs.set(2) = 3;

    m.set_lu();
    Array<double> x = m.solve(rhs);
    REQUIRE_THAT(x(0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(x(1), WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(x(2), WithinAbs(3.0, 1e-12));
}
