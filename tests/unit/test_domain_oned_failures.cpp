#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/oned.hpp"

using namespace Kadath;

TEST_CASE("Domain_oned_ori::find_other_dom rejects unknown bound",
          "[oned_failures]") {
    Dim_array nbr(1);
    nbr.set(0) = 9;
    Domain_oned_ori dom(0, CHEB_TYPE, /*radius=*/1.0, nbr);
    int other_dom = -1;
    int other_bound = -1;
    REQUIRE_THROWS_AS(dom.find_other_dom(0, /*bound=*/-999, other_dom, other_bound),
                      Kadath::KadathError);
}

TEST_CASE("Domain_oned_qcq::find_other_dom rejects unknown bound",
          "[oned_failures]") {
    Dim_array nbr(1);
    nbr.set(0) = 9;
    Domain_oned_qcq dom(0, CHEB_TYPE, /*x_int=*/1.0, /*x_ext=*/2.0, nbr);
    int other_dom = -1;
    int other_bound = -1;
    REQUIRE_THROWS_AS(dom.find_other_dom(0, /*bound=*/-999, other_dom, other_bound),
                      Kadath::KadathError);
}

TEST_CASE("Domain_oned_inf::find_other_dom rejects unknown bound",
          "[oned_failures]") {
    Dim_array nbr(1);
    nbr.set(0) = 9;
    Domain_oned_inf dom(0, CHEB_TYPE, /*radius=*/1.0, nbr);
    int other_dom = -1;
    int other_bound = -1;
    REQUIRE_THROWS_AS(dom.find_other_dom(0, /*bound=*/-999, other_dom, other_bound),
                      Kadath::KadathError);
}
