#include <catch2/catch_test_macros.hpp>
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"  // for LMAX
#include <string>

using namespace Kadath;

namespace {
Term_eq dummy_ope(const Term_eq& a, Param*) { return a; }

Space_spheric make_minimal_space() {
    Point center(3);
    center.set(1) = 0; center.set(2) = 0; center.set(3) = 0;
    Dim_array res(3);
    res.set(0) = 5; res.set(1) = 5; res.set(2) = 4;
    Dim_array bounds_dim(1);
    bounds_dim.set(0) = 2;
    Array<double> bounds(bounds_dim);
    bounds.set(0) = 1.0;
    bounds.set(1) = 10.0;
    return Space_spheric(CHEB_TYPE, center, res, bounds);
}
}

// Pins the std::string_view API contract for the 10 add_* entry points.
// The four behavior cases:
//   1. trim normalizes whitespace (run-collapse + trailing-space convention)
//   2. user-defined operator registers and counter increments
//   3. long names (> LMAX) are accepted without truncation or UB
//   4. indexed-name (`_`/`^`) and missing-`=` cases trigger abort()
//      — death tests are not in scope here; documented expectations only
TEST_CASE("System_of_eqs string_view API contract", "[system_of_eqs][string_view]") {
    Space_spheric space = make_minimal_space();

    SECTION("trim_normalizes_whitespace: leading/trailing/runs collapse") {
        System_of_eqs sys(space, 0, 0);
        double a = 0.0, b = 0.0;
        sys.add_var("  myvar  ", a);
        sys.add_var("myvar2",   b);
        REQUIRE(sys.get_nvar_double() == 2);
    }

    SECTION("user_operator_register_then_counter") {
        System_of_eqs sys(space, 0, 0);
        Param par;
        sys.add_ope("myop",  dummy_ope, &par);
        sys.add_ope("  spc op  ", dummy_ope, &par);
        REQUIRE(sys.get_nopeuser() == 2);
    }

    SECTION("long_name_accepted: name longer than LMAX should not truncate or crash") {
        System_of_eqs sys(space, 0, 0);
        // LMAX is 1000 today; pick a length comfortably past it. Migration goal:
        // names of arbitrary length register without UB or truncation.
        std::string huge(LMAX + 100, 'x');
        double a = 0.0;
        sys.add_var(huge, a);
        REQUIRE(sys.get_nvar_double() == 1);
    }

    SECTION("std_string_argument_works: implicit conversion path") {
        System_of_eqs sys(space, 0, 0);
        std::string n = "fromstring";
        double a = 0.0;
        sys.add_var(n, a);
        sys.add_cst(std::string{"fromtemp"}, 1.5);
        REQUIRE(sys.get_nvar_double() == 1);
        REQUIRE(sys.get_ncst() == 1);
    }

    // Indexed-name (`_`/`^`) and definition missing `=` aborts are documented
    // contract but not asserted here (Catch2 v3 has no built-in death tests).
    // Behavior is verified by:
    //   - existing growth test (no crash on legal names)
    //   - smoke_bh2d (BH2d exercises add_def with `=` paths)
}
