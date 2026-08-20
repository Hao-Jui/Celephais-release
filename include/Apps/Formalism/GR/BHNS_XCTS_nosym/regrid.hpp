#pragma once

#include "For_Kadath/Space/bhns_nosym.hpp"
#include "Apps/Formalism/GR/BHNS_XCTS/regrid.hpp"

namespace Kadath {

template <>
struct bhns_space_traits<Space_bhns_nosym> {
    using outer_adapted = Domain_shell_outer_adapted_nosym;
    using inner_adapted = Domain_shell_inner_adapted_nosym;
    using outer_homothetic = Domain_shell_outer_homothetic_nosym;
    using inner_homothetic = Domain_shell_inner_homothetic_nosym;
};

inline int bhns_xcts_nosym_regrid(config_t& bconfig, std::string output_fname, bool use_config_vars = false)
{
    return bhns_xcts_regrid_impl<Space_bhns_nosym>(bconfig, output_fname, use_config_vars);
}

// Per-domain AMR p-refinement overload (see BHNS_XCTS/regrid.hpp).
inline int bhns_xcts_nosym_regrid(config_t& bconfig, std::string output_fname,
                                  const std::vector<Dim_array>& res_per_domain)
{
    return bhns_xcts_regrid_impl<Space_bhns_nosym>(bconfig, output_fname, /*use_config_vars=*/false,
                                                   res_per_domain);
}

} // namespace Kadath
