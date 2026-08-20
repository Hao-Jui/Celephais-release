#pragma once

#include "Apps/Formalism/Shared/Regrid/ns_regrid.hpp"

namespace Kadath {

template <typename config_t>
int ns_3d_xcts_nosym_interpolate_on_new_grid(config_t& bconfig, const int new_res, std::string outputfile,
                                              bool use_config_vars = false)
{
    return ns_3d_xcts_interpolate_on_new_grid<config_t, Space_spheric_adapted_nosym>(
        bconfig, new_res, outputfile, use_config_vars);
}

} // namespace Kadath
