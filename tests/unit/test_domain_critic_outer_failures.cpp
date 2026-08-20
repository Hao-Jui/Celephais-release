#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/critic.hpp"

using namespace Kadath;

TEST_CASE("Domain_critic_outer::find_other_dom rejects unknown bound",
          "[critic_outer_failures]") {
    Dim_array nbr(2);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    Domain_critic_outer dom(0, CHEB_TYPE, nbr, /*xl=*/1.0);
    int other_dom = -1;
    int other_bound = -1;
    REQUIRE_THROWS_AS(dom.find_other_dom(0, /*bound=*/-999, other_dom, other_bound),
                      Kadath::KadathError);
}
