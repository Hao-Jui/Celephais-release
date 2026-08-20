#pragma once
// ObjectSeeder specialization: isolated neutron star, nosym spherical space, GR.
#include "../seed_traits.hpp"
#include "Hydro/EOS.hh"
#include "../tov_utils.hpp"
#include "../seed_utils.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"   // Space_spheric_adapted_nosym
#include "Apps/Policy/app_resolution.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include <sstream>
#include <vector>

namespace Kadath::Seed {

template <>
struct ObjectSeeder<NS, Space_spheric_adapted_nosym, GR> {
    template <typename config_t>
    static void execute(config_t& bconfig, bool use_config_vars = false)
    {
        using namespace Kadath::Margherita;

        const int type_coloc = CHEB_TYPE;
        Dim_array res = spheric_res_nosym(static_cast<int>(bconfig(BCO_RES)));

        Point center(3);
        for (int i = 1; i <= 3; i++)
            center.set(i) = 0;

        const int ndom = 4 + static_cast<int>(bconfig(NSHELLS));
        std::vector<double> bounds(ndom - 1);

        const double h_cut          = bconfig.template eos<double>(HCUT);
        const std::string eos_file  = bconfig.template eos<std::string>(EOSFILE);
        const std::string eos_type  = bconfig.template eos<std::string>(EOSTYPE);

        auto gen_NS = [&](auto tov) {
            bco_utils::set_NS_bounds(bounds, bconfig);
            Space_spheric_adapted_nosym space(type_coloc, center, res, bounds);
            Kadath::Seed::write_ns_fields(space, bconfig, *tov);
        };

        if (!use_config_vars) {
            if (eos_type == "Cold_PWPoly") {
                using eos_t = Cold_PWPoly;
                EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
                // setup_ns_config_from_TOV is the global version from co_solver_utils
                gen_NS(::setup_ns_config_from_TOV<eos_t>(bconfig));
            } else if (eos_type == "Cold_Table") {
                using eos_t = Cold_Table;
                const int interp_pts = (bconfig.template eos<int>(INTERP_PTS) == 0)
                    ? 2000 : bconfig.template eos<int>(INTERP_PTS);
                EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut, interp_pts,
                                                     bconfig.template eos<double>(MNUC_CGS));
                gen_NS(::setup_ns_config_from_TOV<eos_t>(bconfig));
            } else {
                std::ostringstream oss;
                oss << eos_type << " is not a recognized EOS type.\n";
                KADATH_THROW(oss.str());
            }
        }
    }
};

} // namespace Kadath::Seed
