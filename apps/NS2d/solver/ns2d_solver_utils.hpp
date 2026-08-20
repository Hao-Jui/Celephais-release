#pragma once
#include <vector>
#include "Hydro/EOS.hh"
#include <functional>
#include <memory>
#include <string>

#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "Hydro/Margherita/tov_mass_search.hh"

// Build a 2D (phi-symmetric) initial configuration and write it to disk.
template <NODES s_type, typename config_t> void setup_co(config_t& bconfig, bool use_config_vars = false);

// Solve the 1D TOV problem and update config quantities (MADM/NC/HC/MB/R*).
template <typename eos_t, typename config_t> auto setup_ns_config_from_TOV(config_t& bconfig);

// Build radial interpolators for lapse/conf/rho from a TOV solution.
template <typename tov_t> auto setup_interpolator_from_TOV(tov_t& tov);

#include "ns2d_solver_utils_imp.ipp"
