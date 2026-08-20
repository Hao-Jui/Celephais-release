#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/polar.hpp"
#include "For_Kadath/Array/point.hpp"

using namespace Kadath;

namespace {
Domain_polar_compact make_polar_compact() {
    Point center(2);
    center.set(1) = 0;
    center.set(2) = 0;
    Dim_array nbr(2);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    return Domain_polar_compact(0, CHEB_TYPE, 1.0, center, nbr);
}
} // namespace

TEST_CASE("Domain_polar_compact::find_other_dom rejects unknown bound", "[polar_failures]") {
    auto dom = make_polar_compact();
    int other_dom = -1;
    int other_bound = -1;
    REQUIRE_THROWS_AS(dom.find_other_dom(0, /*bound=*/-999, other_dom, other_bound),
                      Kadath::KadathError);
}

TEST_CASE("Domain_polar_compact::find_other_dom accepts INNER_BC", "[polar_failures]") {
    auto dom = make_polar_compact();
    int other_dom = -1;
    int other_bound = -1;
    REQUIRE_NOTHROW(dom.find_other_dom(3, INNER_BC, other_dom, other_bound));
    REQUIRE(other_dom == 2);
    REQUIRE(other_bound == OUTER_BC);
}
