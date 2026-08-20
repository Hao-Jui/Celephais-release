#include <algorithm>
#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Hydro/Margherita/tov_mass_search.hh"

using Catch::Matchers::WithinRel;

namespace
{

    struct GammaTwoPolytrope {
        struct error_t {
        };

        static double press_cold_eps_cold__rho(double& eps_cold, double& density, error_t&)
        {
            eps_cold = 100.0 * density;
            return 100.0 * density * density;
        }

        static double rho_energy_dedp__press_cold(double& energy_density, double& energy_derivative_by_pressure,
                                                  double& pressure, error_t&)
        {
            const double density = std::sqrt(pressure / 100.0);
            energy_density = density + 100.0 * density * density;
            energy_derivative_by_pressure = (1.0 + 200.0 * density) / (200.0 * density);
            return density;
        }

        static double rho__press_cold(double& pressure, error_t&) { return std::sqrt(std::max(pressure, 0.0) / 100.0); }
    };

    struct UnitKGammaTwoPolytrope {
        struct error_t {
        };

        inline static constexpr double rhomax = 1.0;

        static double press_cold_eps_cold__rho(double& eps_cold, double& density, error_t&)
        {
            eps_cold = density;
            return density * density;
        }

        static double rho_energy_dedp__press_cold(double& energy_density, double& energy_derivative_by_pressure,
                                                  double& pressure, error_t&)
        {
            const double density = std::sqrt(pressure);
            energy_density = density + density * density;
            energy_derivative_by_pressure = (1.0 + 2.0 * density) / (2.0 * density);
            return density;
        }

        static double rho__press_cold(double& pressure, error_t&) { return std::sqrt(std::max(pressure, 0.0)); }
    };

} // namespace

TEST_CASE("MargheritaTOV solves a gamma=2 polytrope", "[tov]")
{
    Kadath::Margherita::MargheritaTOV<GammaTwoPolytrope> tov;

    tov.solve(1.3e-3);

    REQUIRE(tov.state.size() == 9554);
    REQUIRE_THAT(tov.press_c, WithinRel(1.69e-4, 1e-12));
    REQUIRE_THAT(tov.rhoc, WithinRel(1.3e-3, 1e-12));
    REQUIRE_THAT(tov.arealr, WithinRel(9.5540000000001442, 1e-12));
    REQUIRE_THAT(tov.radius, WithinRel(8.0850219648162938, 1e-12));
    REQUIRE_THAT(tov.mass, WithinRel(1.4077033494536644, 1e-12));
    REQUIRE_THAT(tov.baryon_mass, WithinRel(1.515150322105026, 1e-12));
    REQUIRE_THAT(tov.tidal_love_k2, WithinRel(0.072860989387138725, 1e-12));
    REQUIRE_THAT(tov.state.back()[tov.CONF], WithinRel(1.0870562476873649, 1e-12));
}

TEST_CASE("TOV ADM-mass search stays outside the core integrator", "[tov]")
{
    Kadath::Margherita::MargheritaTOV<GammaTwoPolytrope> tov;

    const bool used_maximum_mass = Kadath::Margherita::solve_tov_for_adm_mass(tov, 1.35);

    REQUIRE_FALSE(used_maximum_mass);
    REQUIRE(std::abs(tov.mass - 1.35) < 1e-3);
    REQUIRE(tov.rhoc > 5e-4);
    REQUIRE(tov.rhoc < 1e-2);
    REQUIRE(tov.radius > 0.0);
    REQUIRE(tov.state.back()[tov.CONF] > 0.0);
}

TEST_CASE("TOV ADM-mass search expands to an analytic EOS density range", "[tov]")
{
    Kadath::Margherita::MargheritaTOV<UnitKGammaTwoPolytrope> tov;

    const bool used_maximum_mass = Kadath::Margherita::solve_tov_for_adm_mass(tov, 0.1266961451);

    REQUIRE_FALSE(used_maximum_mass);
    REQUIRE(std::abs(tov.mass - 0.1266961451) < 1e-3);
    REQUIRE(tov.rhoc > 1e-2);
    REQUIRE(tov.rhoc < UnitKGammaTwoPolytrope::rhomax);
}
