#pragma once

// GR binary-NS regrid: thin formalism wrapper over Shared/Regrid/bns_regrid_gr.hpp.
// The shared template owns all geometry/bounds/adapted-mapping plus the GR field
// transfer (conf, lapse, shift, logh, phi); this file supplies only the public
// symmetric-space entry-point name. The nosym space reuses the same template
// (see BNS_XCTS_nosym/regrid.hpp).

#include "Apps/Formalism/Shared/Regrid/bns_regrid_gr.hpp"

namespace Kadath {

template <typename config_t>
int bns_xcts_regrid(config_t& bconfig, std::string output_fname,
                    const std::vector<Dim_array>& res_per_domain = {})
{
    return bns_xcts_regrid_impl<config_t, Space_bin_ns>(bconfig, output_fname, res_per_domain);
}

} // namespace Kadath
