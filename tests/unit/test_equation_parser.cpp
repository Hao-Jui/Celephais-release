#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Param/param.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Ope_eq/ope_eq.hpp"

#include <string>

using Catch::Matchers::ContainsSubstring;
using namespace Kadath;

namespace {
Term_eq id_user_ope(const Term_eq& a, Param*) { return a; }

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
} // namespace

// parse_eq_trim is the natural parser entry point — it normalizes whitespace
// via trim_spaces before recursing into give_ope/find_ope. Calling give_ope
// directly requires a space-padded buffer (parser internal precondition); the
// public `add_eq_*` family always goes through trim. Use parse_eq_trim here so
// we exercise the same code path the rest of the codebase uses.

TEST_CASE("EquationParser legal binary parse — Ope_add tree built",
          "[equation_parser]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);
    sys.add_cst("f", 1.0);
    sys.add_cst("g", 2.0);

    Ope_eq* tree = sys.parse_eq_trim(0, "f + g", -1, true);
    REQUIRE(tree != nullptr);
    delete tree;
}

TEST_CASE("EquationParser user-operator parse — Ope_user tree built",
          "[equation_parser]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);
    Param par;
    sys.add_ope("myop", id_user_ope, &par);
    sys.add_cst("f", 1.0);

    Ope_eq* tree = sys.parse_eq_trim(0, "myop(f)", -1, true);
    REQUIRE(tree != nullptr);
    delete tree;
}

TEST_CASE("EquationParser unknown operator throws KadathError with token in message",
          "[equation_parser]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);

    REQUIRE_THROWS_AS(
        sys.parse_eq_trim(0, "completely_unknown_xyz", -1, true),
        Kadath::KadathError);
    try {
        (void)sys.parse_eq_trim(0, "completely_unknown_xyz", -1, true);
    } catch (const Kadath::KadathError& e) {
        REQUIRE_THAT(std::string{e.what()}, ContainsSubstring("completely_unknown_xyz"));
    }
}

TEST_CASE("EquationParser is_ope_minus on bare '-' input does not crash",
          "[equation_parser]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);
    // Bare '-' is not a valid expression; parser should throw KadathError
    // (unknown operator) rather than UB-write past occi[-1] inside is_ope_minus.
    REQUIRE_THROWS_AS(
        sys.parse_eq_trim(0, "-", -1, true),
        Kadath::KadathError);
}
