#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Param/param.hpp"

using namespace Kadath;

TEST_CASE("Param::add_int rejects duplicate position", "[param_failures]") {
    Param p;
    p.add_int(42, /*position=*/0);
    REQUIRE_THROWS_AS(p.add_int(99, /*position=*/0), Kadath::KadathError);
}

TEST_CASE("Param::add_double rejects duplicate position", "[param_failures]") {
    Param p;
    p.add_double(3.14, /*position=*/0);
    REQUIRE_THROWS_AS(p.add_double(2.71, /*position=*/0), Kadath::KadathError);
}
