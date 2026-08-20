#pragma once
/*
 * TOV helpers — space-agnostic, no space type involved.
 *
 * Extracted from seed/legacy/co_solver_utils so that seed/GR/ files have an
 * explicit dependency rather than relying on a transitive include via legacy.
 * These stay in global namespace for backward compatibility with legacy callers.
 */
#include "Hydro/EOS.hh"
#include "Hydro/Margherita/tov_mass_search.hh"
#include <memory>

// Indices into the array returned by setup_interpolator_from_TOV().
// Shared here so the producer (setup_interpolator_from_TOV) and the
// consumer (write_ns_fields) refer to the same ordering by name.
enum class TovField { LAPSE = 0, RHO, CONF };

template <typename eos_t, typename config_t>
auto setup_ns_config_from_TOV(config_t& bconfig)
{
    using namespace Kadath::Margherita;
    auto tov = std::make_unique<MargheritaTOV<eos_t>>();
    bool use_Mmax = solve_tov_for_adm_mass(*tov, bconfig(MADM));
    if (use_Mmax)
        bconfig.set(MADM) = tov->mass;
    bconfig.set(NC)   = tov->rhoc;
    bconfig.set(HC)   = EOS<eos_t, eos_var_t::PRESSURE>::h_cold__rho(bconfig(NC));
    bconfig.set(MB)   = tov->baryon_mass;
    bconfig.set(RMID) = tov->radius;
    bconfig.set(RIN)  = 0.5 * bconfig(RMID);
    bconfig.set(ROUT) = 2.0 * bconfig(RMID);
    return tov;
}

template <typename tov_t>
auto setup_interpolator_from_TOV(tov_t& tov)
{
    const int n = static_cast<int>(tov.state.size());
    auto r_ptr = std::make_unique_for_overwrite<double[]>(n);
    auto lapse_ptr = std::make_unique_for_overwrite<double[]>(n);
    auto rho_ptr = std::make_unique_for_overwrite<double[]>(n);
    auto conf_ptr = std::make_unique_for_overwrite<double[]>(n);
    for (int j = 0; j < n; ++j) {
        lapse_ptr[j] = std::exp(tov.state[j][tov.PHI]);
        conf_ptr[j]  = tov.state[j][tov.CONF];
        r_ptr[j]     = tov.state[j][tov.RISO];
        rho_ptr[j]   = tov.state[j][tov.RHOB];
    }
    // Packed order matches TovField enum: LAPSE=0, RHO=1, CONF=2
    linear_interp_t<double, 3> ltp(n,
        std::move(r_ptr), std::move(lapse_ptr), std::move(rho_ptr), std::move(conf_ptr));
    return ltp;
}
