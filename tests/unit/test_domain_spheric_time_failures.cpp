#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/spheric_time.hpp"

using namespace Kadath;

TEST_CASE("Domain_spheric_time_nucleus::find_other_dom rejects unknown bound",
          "[spheric_time_failures]") {
    Dim_array nbr(2);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    Domain_spheric_time_nucleus dom(0, CHEB_TYPE, /*tmmin=*/0.0, /*tmmax=*/1.0,
                                    /*radius=*/1.0, nbr);
    int other_dom = -1;
    int other_bound = -1;
    REQUIRE_THROWS_AS(dom.find_other_dom(0, /*bound=*/-999, other_dom, other_bound),
                      Kadath::KadathError);
}
