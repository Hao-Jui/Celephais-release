
#pragma once
#include <vector>
#include <functional>
#include <memory>
#include <string>

#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Space/adapted_bh_polar.hpp"

// Build a 2D (phi-symmetric) BH initial configuration and write it to disk.
template <NODES s_type, typename config_t> void setup_co(config_t& bconfig, bool use_config_vars = false);

// Write a 2D MSQI BH initial guess on a polar homothetic space.
template <typename config_t> void write_bh2d_init_setup_tofile_msqi(Space_adapted_bh_polar& space, config_t& bconfig);

#include "bh2d_solver_utils_imp.ipp"
