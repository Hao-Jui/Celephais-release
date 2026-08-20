#pragma once

#include "../seed_traits.hpp"
#include "bh_seed_utils.hpp"
#include "Apps/Policy/app_resolution.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Kadath_point_h/kadath_trumpet.hpp"

namespace Kadath::Seed {

template <>
struct ObjectSeeder<BH, Space_trumpet, GR> {
    template <typename config_t>
    static void execute(config_t& bconfig, bool use_config_vars = false)
    {
        if (select_bh_seed_kind(bconfig) != BhSeedKind::Trumpet) {
            KADATH_THROW("Space_trumpet requires [bh].trumpet_bh_seed = true.");
        }

        const int type_coloc = CHEB_TYPE;
        validate_resolution(static_cast<int>(bconfig(BCO_RES)));
        Dim_array res = spheric_res(static_cast<int>(bconfig(BCO_RES)));

        Point center(3);
        for (int i = 1; i <= 3; i++)
            center.set(i) = 0;

        std::vector<double> bounds = isolated_trumpet_bh_seed_bounds(bconfig, use_config_vars);
        Space_trumpet space(type_coloc, center, res, bounds);

        write_trumpet_bh_fields(space, bconfig, 0);
    }
};

} // namespace Kadath::Seed
