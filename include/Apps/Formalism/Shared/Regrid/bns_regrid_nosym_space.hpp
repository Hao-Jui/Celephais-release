#pragma once

#include "Apps/Formalism/Shared/Regrid/bns_regrid.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"

namespace Kadath {

// Geometry trait shared by every formalism that regrids a no-symmetry BNS.
template <>
struct bns_space_traits<Space_bin_ns_nosym> {
    using outer_adapted = Domain_shell_outer_adapted_nosym;
    using inner_adapted = Domain_shell_inner_adapted_nosym;
};

} // namespace Kadath
