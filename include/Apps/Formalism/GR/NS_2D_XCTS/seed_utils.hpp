#pragma once
#include <vector>
#include "Hydro/EOS.hh"
#include <functional>
#include <memory>
#include <string>

#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "Apps/Seed/tov_utils.hpp"

// Build a 2D (phi-symmetric) initial configuration and write it to disk.
template <NODES s_type, typename config_t> void setup_co(config_t& bconfig, bool use_config_vars = false);

// Write the conformally flat XCTS seed fields generated from the TOV profile.
template <typename tov_t, typename config_t>
void write_ns2d_init_setup_tofile_xcts(Space_polar_adapted& space, config_t& bconfig, tov_t& tov);

#include "seed_utils_imp.ipp"
