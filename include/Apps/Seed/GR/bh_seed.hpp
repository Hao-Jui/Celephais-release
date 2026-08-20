#pragma once
// ObjectSeeder specialization: isolated black hole, adapted-BH space, GR.
#include "../seed_traits.hpp"
#include "bh_seed_utils.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted_bh.hpp"   // Space_adapted_bh
#include "Apps/Policy/app_resolution.hpp"
#include "For_Kadath/Array/exceptions.hpp"

namespace Kadath::Seed {

template <>
struct ObjectSeeder<BH, Space_adapted_bh, GR> {
    template <typename config_t>
    static void execute(config_t& bconfig, bool use_config_vars = false)
    {
        const int type_coloc = CHEB_TYPE;
        Dim_array res = spheric_res(static_cast<int>(bconfig(BCO_RES)));

        Point center(3);
        for (int i = 1; i <= 3; i++)
            center.set(i) = 0;

        std::vector<double> bounds = isolated_bh_seed_bounds(bconfig, use_config_vars);

        Space_adapted_bh space(type_coloc, center, res, bounds);
        Kadath::Seed::write_bh_fields(space, bconfig, Kadath::Seed::select_bh_seed_kind(bconfig));
    }
};

} // namespace Kadath::Seed
