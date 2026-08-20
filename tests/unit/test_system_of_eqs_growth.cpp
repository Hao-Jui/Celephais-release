#include <catch2/catch_test_macros.hpp>
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Param/param.hpp"
#include <cstdio>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

using namespace Kadath;

namespace {
// Trivial user-defined operator for add_ope growth test.
Term_eq dummy_ope(const Term_eq& a, Param*) { return a; }

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

// Proves the old VARMAX=1000 compile-time cap is gone: register more than 1000
// entries of each kind and confirm no abort/UB/crash.
TEST_CASE("System_of_eqs grows past old VARMAX=1000 cap", "[system_of_eqs][growth]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);

    constexpr int N = 1500;

    SECTION("add_var(double) past 1000") {
        std::vector<double> storage(N, 0.0);
        for (int i = 0; i < N; ++i) {
            char name[32];
            std::snprintf(name, sizeof(name), "v%d", i);
            sys.add_var(name, storage[i]);
        }
        REQUIRE(sys.get_nvar_double() == N);
    }

    SECTION("add_cst(double) past 1000") {
        for (int i = 0; i < N; ++i) {
            char name[32];
            std::snprintf(name, sizeof(name), "c%d", i);
            sys.add_cst(name, static_cast<double>(i));
        }
        REQUIRE(sys.get_ncst() == N);
    }

    SECTION("add_ope past 1000") {
        Param par;
        for (int i = 0; i < N; ++i) {
            char name[32];
            std::snprintf(name, sizeof(name), "op%d", i);
            sys.add_ope(name, dummy_ope, &par);
        }
        REQUIRE(sys.get_nopeuser() == N);
    }
}

TEST_CASE("System_of_eqs invalidates forwarded residuals when state changes",
          "[system_of_eqs][forwarded_residual]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);
    double unknown = 1.0;
    sys.add_var("unknown", unknown);

    auto seed_forwarded = [&sys]() {
        Array<double> residual(1);
        residual.set(0) = 42.0;
        sys.store_forwarded_residual(std::move(residual));
    };
    auto replacement = []() {
        Array<double> values(1);
        values.set(0) = 0.5;
        return values;
    };

    SECTION("forwarded norm observation is non-consuming") {
        Array<double> residual(3);
        residual.set(0) = -42.0;
        residual.set(1) = 7.0;
        residual.set(2) = 19.0;
        sys.store_forwarded_residual(std::move(residual));

        double norm = -1.0;
        REQUIRE(sys.forwarded_residual_infinity_norm(norm));
        REQUIRE(norm == 42.0);

        std::unique_ptr<Array<double>> forwarded = sys.take_forwarded_residual();
        REQUIRE(forwarded);
        REQUIRE((*forwarded)(0) == -42.0);
        REQUIRE((*forwarded)(1) == 7.0);
        REQUIRE((*forwarded)(2) == 19.0);
        REQUIRE_FALSE(sys.forwarded_residual_infinity_norm(norm));
    }

    SECTION("forwarded norm fails closed for non-finite values and accepts empty input") {
        Array<double> non_finite(3);
        non_finite.set(0) = 1.0;
        non_finite.set(1) = std::numeric_limits<double>::quiet_NaN();
        non_finite.set(2) = 2.0;
        sys.store_forwarded_residual(std::move(non_finite));

        double norm = -1.0;
        REQUIRE(sys.forwarded_residual_infinity_norm(norm));
        REQUIRE(std::isinf(norm));

        Array<double> infinite(1);
        infinite.set(0) = -std::numeric_limits<double>::infinity();
        sys.store_forwarded_residual(std::move(infinite));
        REQUIRE(sys.forwarded_residual_infinity_norm(norm));
        REQUIRE(std::isinf(norm));

        Array<double> empty(0);
        sys.store_forwarded_residual(std::move(empty));
        REQUIRE(sys.forwarded_residual_infinity_norm(norm));
        REQUIRE(norm == 0.0);
    }

    SECTION("independent solver reset") {
        seed_forwarded();
        sys.reset_solver_runtime_state();
        REQUIRE_FALSE(sys.take_forwarded_residual());
    }

    SECTION("absolute unknown replacement") {
        seed_forwarded();
        Array<double> values = replacement();
        int offset = 0;
        sys.xx_to_vars(values, offset);
        REQUIRE_FALSE(sys.take_forwarded_residual());
    }

    SECTION("Newton unknown correction") {
        seed_forwarded();
        Array<double> values = replacement();
        int offset = 0;
        sys.xx_to_vars_delta(values, offset);
        REQUIRE_FALSE(sys.take_forwarded_residual());
    }

    SECTION("snapshot restore") {
        const System_of_eqs::State_snapshot snapshot = sys.snapshot_state();
        seed_forwarded();
        sys.restore_state(snapshot);
        REQUIRE_FALSE(sys.take_forwarded_residual());
    }
}

TEST_CASE("System_of_eqs snapshots remain independent and restore exact unknown values",
          "[system_of_eqs][state_snapshot]") {
    Space_spheric space = make_minimal_space();
    System_of_eqs sys(space, 0, 0);
    double numeric = 1.25;
    Scalar field(space);
    field = 3.5;
    field.std_base();
    sys.add_var("numeric", numeric);
    sys.add_var("field", field);

    const System_of_eqs::State_snapshot first = sys.snapshot_state();
    numeric = -7.0;
    field = 4.75;
    field.std_base();
    const System_of_eqs::State_snapshot second = sys.snapshot_state();

    sys.restore_state(first);
    CHECK(numeric == 1.25);
    CHECK(maxval(field) == 3.5);

    sys.restore_state(second);
    CHECK(numeric == -7.0);
    CHECK(maxval(field) == 4.75);
}
