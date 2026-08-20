#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/polar_periodic.hpp"

using namespace Kadath;

TEST_CASE("Domain_polar_periodic_nucleus::find_other_dom rejects unknown bound",
          "[polar_periodic_failures]") {
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    nbr.set(2) = 4;
    Domain_polar_periodic_nucleus dom(0, CHEB_TYPE, /*radius=*/1.0, /*ome=*/0.5, nbr);
    int other_dom = -1;
    int other_bound = -1;
    REQUIRE_THROWS_AS(dom.find_other_dom(0, /*bound=*/-999, other_dom, other_bound),
                      Kadath::KadathError);
}
