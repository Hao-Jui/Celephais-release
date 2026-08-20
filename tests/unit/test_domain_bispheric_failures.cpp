#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

using namespace Kadath;

namespace {
Domain_bispheric_chi_first make_chi_first() {
    Dim_array nbr(3);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    nbr.set(2) = 5;
    return Domain_bispheric_chi_first(0, CHEB_TYPE, /*aa=*/1.0, /*etalim=*/0.5,
                                      /*rr=*/2.0, /*chi_max=*/1.0, nbr);
}
} // namespace

TEST_CASE("Domain_bispheric_chi_first::find_other_dom rejects unknown bound",
          "[bispheric_failures]") {
    auto dom = make_chi_first();
    int other_dom = -1;
    int other_bound = -1;
    REQUIRE_THROWS_AS(dom.find_other_dom(0, /*bound=*/-999, other_dom, other_bound),
                      Kadath::KadathError);
}

TEST_CASE("Domain_bispheric_chi_first rejects an even folding line",
          "[bispheric_failures][parity]") {
    REQUIRE_THROWS_AS(([] {
        Dim_array nbr(3);
        nbr.set(0) = 9;
        nbr.set(1) = 5;
        nbr.set(2) = 4;
        Domain_bispheric_chi_first dom(0, CHEB_TYPE, /*aa=*/1.0, /*etalim=*/0.5,
                                       /*rr=*/2.0, /*chi_max=*/1.0, nbr);
        Val_domain field(&dom);
        field.std_base();
    }()), Kadath::KadathError);
}
