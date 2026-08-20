#pragma once

#include "Apps/Formalism/Shared/Regrid/bns_regrid_gr.hpp"
#include "Apps/Formalism/Shared/Regrid/bns_regrid_nosym_space.hpp"

namespace Kadath {

template <typename config_t>
int bns_xcts_nosym_regrid(config_t& bconfig, std::string output_fname,
                          const std::vector<Dim_array>& res_per_domain = {})
{
    return bns_xcts_regrid_impl<config_t, Space_bin_ns_nosym>(bconfig, output_fname, res_per_domain);
}

} // namespace Kadath
