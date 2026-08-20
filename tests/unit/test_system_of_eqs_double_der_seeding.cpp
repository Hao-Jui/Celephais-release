#include <catch2/catch_test_macros.hpp>
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include <vector>

using namespace Kadath;

namespace {
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

// Regression guard for the xx_to_ders double-variable seeding off-by-one
// (loop ran dd < dom_max instead of dd <= dom_max). add_var(double) lays out
// term_double as which*ndom + (dd - dom_min) with ndom slots per variable;
// xx_to_ders must seed every domain's slot of variable i with the same
// xx(conte). The exclusive loop advanced pos_term only ndom-1 times per
// variable, shifting later variables' seeds into earlier variables' high-domain
// slots and leaving the trailing slots unseeded — corrupting the matrix-free
// Jv (do_JX) for any numeric unknown referenced on high domains.
TEST_CASE("xx_to_ders seeds every domain slot of each double variable",
          "[system_of_eqs][xx_to_ders]") {
    Space_spheric space = make_minimal_space();
    const int ndom = space.get_nbr_domains();
    REQUIRE(ndom >= 3); // need a multi-domain range to expose the off-by-one

    System_of_eqs sys(space, 0, ndom - 1);

    std::vector<double> storage = {1.0, 2.0, 3.0};
    sys.add_var("va", storage[0]);
    sys.add_var("vb", storage[1]);
    sys.add_var("vc", storage[2]);

    const int nvar = static_cast<int>(storage.size());
    REQUIRE(sys.get_nbr_unknowns() == nvar);

    Array<double> xx(nvar);
    xx.set(0) = 10.0;
    xx.set(1) = 20.0;
    xx.set(2) = 30.0;

    sys.xx_to_ders(xx);

    for (int vv = 0; vv < nvar; vv++)
        for (int dd = 0; dd < ndom; dd++) {
            const Term_eq* slot = sys.give_term_double(vv, dd);
            REQUIRE(slot != nullptr);
            // Pre-fix: high-domain slots held the NEXT variable's seed and the
            // last variable's slots were never seeded (get_der_d throws).
            REQUIRE(slot->get_der_d() == xx(vv));
        }
}
