#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Tensor/vector.hpp"

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
} // namespace

TEST_CASE("System_of_eqs::add_var rejects names with index marker '_'",
          "[system_of_eqs_failures]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);
    double a = 0.0;
    REQUIRE_THROWS_AS(sys.add_var("bad_name", a), Kadath::KadathError);
}

TEST_CASE("System_of_eqs::add_var rejects names with index marker '^'",
          "[system_of_eqs_failures]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);
    double a = 0.0;
    REQUIRE_THROWS_AS(sys.add_var("bad^name", a), Kadath::KadathError);
}

TEST_CASE("Domain-local scalar definition view matches materialization and last registration",
          "[system_of_eqs_failures][definition-domain-view]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);
    Scalar one(space);
    Scalar two(space);
    one = 1.0;
    two = 2.0;
    one.std_base();
    two.std_base();
    sys.add_cst("one", one);
    sys.add_cst("two", two);
    sys.add_def(0, "duplicate = one");
    sys.add_def(0, "duplicate = two");
    sys.give_def(0)->compute_res();
    sys.give_def(1)->compute_res();

    const Tensor materialized_tensor = sys.give_val_def("duplicate");
    const Val_domain& materialized = materialized_tensor()(0);
    // Acquire the view only after the materializing definition evaluation: a
    // view must be consumed before another operation can refresh definitions.
    const Val_domain& view = sys.give_val_def_scalar_domain(" duplicate ", 0);
    CHECK(view.integ_volume() == materialized.integ_volume());
    CHECK(view.get_base() == materialized.get_base());
}

TEST_CASE("Domain-local scalar definition view refuses absent domains and tensor definitions",
          "[system_of_eqs_failures][definition-domain-view]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);
    Base_tensor basis(space, CARTESIAN_BASIS);
    Vector vector(space, CON, basis);
    vector = 1.0;
    vector.std_base();
    sys.add_cst("vector", vector);
    sys.add_def(0, "vectorcopy^i = vector^i");
    sys.give_def(0)->compute_res();

    CHECK_THROWS_AS(sys.give_val_def_scalar_domain("missing", 0), Kadath::KadathError);
    CHECK_THROWS_AS(sys.give_val_def_scalar_domain("vectorcopy", 0), Kadath::KadathError);
    CHECK_THROWS_AS(sys.give_val_def_scalar_domain("vectorcopy", 1), Kadath::KadathError);
}
