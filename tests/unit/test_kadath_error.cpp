#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "Apps/Startup/solver_startup.hpp"

#include <string>

using Catch::Matchers::ContainsSubstring;

TEST_CASE("KadathError what() contains message and file:line", "[kadath_error]") {
    Kadath::KadathError err("path/to/x.cpp", 17, "tensor required");
    const std::string what = err.what();
    REQUIRE_THAT(what, ContainsSubstring("tensor required"));
    REQUIRE_THAT(what, ContainsSubstring("path/to/x.cpp:17"));
}

TEST_CASE("KADATH_THROW captures __FILE__ and __LINE__", "[kadath_error]") {
    try {
        KADATH_THROW("oops");
        FAIL("KADATH_THROW did not throw");
    } catch (const Kadath::KadathError& e) {
        const std::string what = e.what();
        REQUIRE_THAT(what, ContainsSubstring("oops"));
        REQUIRE_THAT(what, ContainsSubstring("test_kadath_error.cpp"));
    }
}

TEST_CASE("guarded_run with non-throwing callable returns 0", "[kadath_error]") {
    int marker = 0;
    int rc = KadathApps::guarded_run([&] { marker = 1; },
                                 [](const Kadath::KadathError&) { return 99; });
    REQUIRE(rc == 0);
    REQUIRE(marker == 1);
}

TEST_CASE("guarded_run with throwing callable invokes injected handler", "[kadath_error]") {
    std::string captured;
    int rc = KadathApps::guarded_run(
        [] { KADATH_THROW("explode"); },
        [&](const Kadath::KadathError& e) {
            captured = e.what();
            return 7;
        });
    REQUIRE(rc == 7);
    REQUIRE_THAT(captured, ContainsSubstring("explode"));
}
