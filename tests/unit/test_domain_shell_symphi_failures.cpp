#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/spheric_symphi.hpp"
#include "For_Kadath/Array/point.hpp"

using namespace Kadath;

TEST_CASE("Domain_shell_symphi::find_other_dom rejects unknown bound",
          "[symphi_failures]") {
    Point center(3);
    center.set(1) = 0;
    center.set(2) = 0;
    center.set(3) = 0;
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    nbr.set(2) = 4;
    Domain_shell_symphi dom(0, CHEB_TYPE, 1.0, 2.0, center, nbr);
    int other_dom = -1;
    int other_bound = -1;
    REQUIRE_THROWS_AS(dom.find_other_dom(0, /*bound=*/-999, other_dom, other_bound),
                      Kadath::KadathError);
}
