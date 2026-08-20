#pragma once
/*
 * write_ns_fields<space_t> — space-generic NS field initialiser.
 *
 * TOV helpers (setup_ns_config_from_TOV, setup_interpolator_from_TOV) are
 * intentionally NOT re-declared here; the versions in co_solver_utils.hpp /
 * co_solver_utils_imp.cpp (global namespace) are the single source of truth
 * and are visible to all ObjectSeeder specialisations that also include
 * co_solver_utils.hpp.
 *
 * Only write_ns_fields lives in Kadath::Seed because it needs specialisation
 * by space_t; the rest is space-agnostic and stays global.
 */
#include "Hydro/EOS.hh"
#include "Apps/Seed/tov_utils.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

namespace Kadath::Seed {


// write_ns_fields — templated on space_t.
// Works for Space_spheric_adapted, Space_spheric_adapted_nosym, and any
// future spherical NS space that supports CoordFields<space_t>.
// Calls the global ::setup_interpolator_from_TOV (from co_solver_utils).

template <typename space_t, typename tov_t, typename config_t>
void write_ns_fields(space_t& space, config_t& bconfig, tov_t& tov)
{
    using eos_t = typename tov_t::eos_t;
    const int ndom = space.get_nbr_domains();
    Base_tensor basis(space, CARTESIAN_BASIS);

    Scalar lapse(space);
    lapse = 1.;
    Scalar conf(lapse);
    Scalar logh(space);
    logh.annule_hard();
    Scalar Omg(space);
    Omg.annule_hard();

    // setup_interpolator_from_TOV is the global version from co_solver_utils
    auto lintp = setup_interpolator_from_TOV(tov);
    CoordFields<space_t> cfg(space);
    Scalar r_field(cfg.radius());

    auto update_fields = [&](const size_t dom) {
        Index pos(space.get_domain(dom)->get_nbr_points());
        do {
            double rval   = r_field(dom)(pos);
            auto all_ltp  = lintp.interpolate_all(rval);
            auto rho      = (all_ltp[static_cast<int>(TovField::RHO)] <= 0)
                            ? 1e-15 : all_ltp[static_cast<int>(TovField::RHO)];
            auto h        = EOS<eos_t, eos_var_t::DENSITY>::h_cold__rho(rho);
            if (pos(0) == 0 && pos(1) == 0 && pos(2) == 0)
                bconfig.set(HC) = h;
            logh.set_domain(dom).set(pos)  = (h < 1) ? 0. : std::log(h);
            lapse.set_domain(dom).set(pos) = all_ltp[static_cast<int>(TovField::LAPSE)];
            conf.set_domain(dom).set(pos)  = all_ltp[static_cast<int>(TovField::CONF)];
        } while (pos.inc());
    };
    for (int i = 0; i < 3; ++i)
        update_fields(i);
    for (int d = 2; d < ndom; ++d)
        logh.set_domain(d).annule_hard();

    Vector shift(space, CON, basis);
    for (int i = 1; i <= 3; i++)
        shift.set(i).annule_hard();

    shift.std_base();
    logh.std_base();
    conf.std_base();
    lapse.std_base();
    Omg.std_base();

    {
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh, Omg);
    }
}

} // namespace Kadath::Seed
