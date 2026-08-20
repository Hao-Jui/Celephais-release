#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Forward-declare the function (defined in src/Coef/leg.cpp)
namespace Kadath {
void legendre(int n, double& poly, double& pder,
              double& polym1, double& pderm1,
              double& polym2, double& pderm2, double x);
}

using Catch::Matchers::WithinAbs;

TEST_CASE("Legendre P_0 = 1", "[legendre]") {
    double poly, pder, pm1, pdm1, pm2, pdm2;
    Kadath::legendre(0, poly, pder, pm1, pdm1, pm2, pdm2, 0.5);
    REQUIRE_THAT(poly, WithinAbs(1.0, 1e-14));
}

TEST_CASE("Legendre P_1(x) = x", "[legendre]") {
    double poly, pder, pm1, pdm1, pm2, pdm2;
    Kadath::legendre(1, poly, pder, pm1, pdm1, pm2, pdm2, 0.7);
    REQUIRE_THAT(poly, WithinAbs(0.7, 1e-14));
}

TEST_CASE("Legendre Bonnet recursion", "[legendre]") {
    // (n+1)*P_{n+1} = (2n+1)*x*P_n - n*P_{n-1}
    double x = 0.3;
    double pn, pdn, pnm1, pdnm1, pnm2, pdnm2;
    for (int n = 2; n <= 10; n++) {
        Kadath::legendre(n, pn, pdn, pnm1, pdnm1, pnm2, pdnm2, x);
        // Check derivative: P'_n(x) = n*(x*P_n - P_{n-1})/(x^2-1)
        double expected_der = n * (x * pn - pnm1) / (x*x - 1.0);
        REQUIRE_THAT(pdn, WithinAbs(expected_der, 1e-10));
    }
}
