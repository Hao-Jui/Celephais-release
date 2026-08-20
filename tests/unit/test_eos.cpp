#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include "Hydro/EOS.hh"
#include "celephais_paths.h"
#include "Apps/Diagnostics/configured_eos.hpp"
#include "Apps/Diagnostics/sacra_hydro_conversion.hpp"
#include "For_Kadath/Config/config_bco.hpp"

using namespace Kadath;
using namespace Kadath::Margherita;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// gam2.polytrope: single-piece gamma=2 polytrope, K=100, geometrised units.
// Analytic relations:
//   P   = K * rho^2  = 100 * rho^2
//   eps = K * rho    = 100 * rho          (from eps_tab[0]=0, integral of K*rho^(g-1)/(g-1))
//   h   = 1 + eps + P/rho = 1 + 200*rho
// => rho = (h - 1) / 200

static std::string eos_path() {
    // Prefer runtime env var override; fall back to compile-time CELEPHAIS_DATA_DIR
    const char* home = std::getenv("HOME_CELEPHAIS");
    const std::string data_dir = home ? std::string(home) + "/data" : CELEPHAIS_DATA_DIR;
    return data_dir + "/eos/gam2.polytrope";
}

static std::string cold_table_path() {
    const char* home = std::getenv("HOME_CELEPHAIS");
    const std::string data_dir = home ? std::string(home) + "/data" : CELEPHAIS_DATA_DIR;
    return data_dir + "/eos/dd2.lorene";
}

TEST_CASE("configured cold-table EOS parameters retain mnuc_cgs",
          "[eos][cold-table][config]") {
    kadath_config<BCO_NS_INFO> config;
    config.set_eos(EOSFILE) = cold_table_path();
    config.set_eos(HCUT) = 1.01;
    config.set_eos(INTERP_PTS) = 1234;
    config.set_eos(MNUC_CGS) = 1.6748109230081286e-24;

    const auto parameters = KadathApps::configured_cold_table_parameters(config);

    REQUIRE(parameters.filename == cold_table_path());
    REQUIRE(parameters.h_cut == 1.01);
    REQUIRE(parameters.interpolation_points == 1234);
    REQUIRE(parameters.mnuc_cgs == 1.6748109230081286e-24);

    config.set_eos(HCUT) = 0.0;
    KadathApps::init_configured_cold_table(config);
    const double nondefault_rhomin = Cold_Table::rhomin;

    constexpr double atomic_mass_unit_cgs = 1.660539040e-24;
    config.set_eos(MNUC_CGS) = atomic_mass_unit_cgs;
    KadathApps::init_configured_cold_table(config);
    const double default_rhomin = Cold_Table::rhomin;

    REQUIRE_THAT(nondefault_rhomin / default_rhomin,
                 WithinRel(1.6748109230081286e-24 / atomic_mass_unit_cgs, 1.0e-13));

    config.set_eos(MNUC_CGS) = 0.0;
    KadathApps::init_configured_cold_table(config);
    REQUIRE(Cold_Table::rhomin == default_rhomin);

    config.set_eos(INTERP_PTS) = 0;
    REQUIRE(KadathApps::configured_cold_table_parameters(config).interpolation_points == 2000);
}

TEST_CASE("EOS init and pressure from enthalpy", "[eos]") {
    EOS<Cold_PWPoly, eos_var_t::PRESSURE>::init(eos_path());

    // h = 1.2 => rho = (1.2 - 1) / 200 = 0.001 => P = 100 * 0.001^2 = 1e-4
    double h = 1.2;
    double P = EOS<Cold_PWPoly, eos_var_t::PRESSURE>::get(h);
    REQUIRE(P > 0.0);
    REQUIRE_THAT(P, WithinRel(1.0e-4, 1e-10));
}

TEST_CASE("EOS density from enthalpy", "[eos]") {
    EOS<Cold_PWPoly, eos_var_t::PRESSURE>::init(eos_path());

    // h = 1.2 => rho = 0.001
    double h = 1.2;
    double rho = EOS<Cold_PWPoly, eos_var_t::DENSITY>::get(h);
    REQUIRE(rho > 0.0);
    REQUIRE_THAT(rho, WithinRel(0.001, 1e-10));
}

TEST_CASE("EOS round-trip: h_cold__rho inverts get(DENSITY)", "[eos]") {
    EOS<Cold_PWPoly, eos_var_t::PRESSURE>::init(eos_path());

    double h_in = 1.5;
    double rho = EOS<Cold_PWPoly, eos_var_t::DENSITY>::get(h_in);
    double h_out = EOS<Cold_PWPoly, eos_var_t::PRESSURE>::h_cold__rho(rho);
    REQUIRE_THAT(h_out, WithinRel(h_in, 1e-10));
}

TEST_CASE("EOS surface: enthalpy at h=1 gives zero density", "[eos]") {
    EOS<Cold_PWPoly, eos_var_t::PRESSURE>::init(eos_path());

    // h exactly 1.0 is the surface; rho__h_cold clamps to rhomin and sets error bit
    // get() should still return a value (rhomin, tiny)
    double rho = EOS<Cold_PWPoly, eos_var_t::DENSITY>::get(1.0);
    // rho should be at rhomin = 1e-19, which is extremely small but non-negative
    REQUIRE(rho >= 0.0);
}

TEST_CASE("SACRA hydro guard preserves positive-H EOS round-trip", "[eos][sacra]") {
    EOS<Cold_PWPoly, eos_var_t::PRESSURE>::init(eos_path());

    const double h_in = 1.5;
    const double pressure = EOS<Cold_PWPoly, eos_var_t::PRESSURE>::get(h_in);
    const double rho = EOS<Cold_PWPoly, eos_var_t::DENSITY>::get(h_in);

    const auto guarded = sacra_hydro::zero_outside_matter(
        std::log(h_in), sacra_hydro::ColdHydroState{pressure, rho});

    REQUIRE(guarded.pressure == pressure);
    REQUIRE(guarded.density == rho);
    const double h_out = EOS<Cold_PWPoly, eos_var_t::PRESSURE>::h_cold__rho(guarded.density);
    REQUIRE_THAT(h_out, WithinRel(h_in, 1e-10));
}

TEST_CASE("SACRA hydro guard zeroes pressure and density for nonpositive H", "[eos][sacra]") {
    const sacra_hydro::ColdHydroState nonzero{1.0e-4, 1.0e-3};

    for (const double log_enthalpy : {0.0, -1.0}) {
        const auto guarded = sacra_hydro::zero_outside_matter(log_enthalpy, nonzero);
        REQUIRE(guarded.pressure == 0.0);
        REQUIRE(guarded.density == 0.0);
    }
}
