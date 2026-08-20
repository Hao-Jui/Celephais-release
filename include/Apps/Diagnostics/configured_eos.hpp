/*
 * Copyright 2026 Celephais contributors
 *
 * This file is part of Celephais and is licensed under the GNU General
 * Public License, version 3 or (at your option) any later version.
 */

#pragma once

#include "For_Kadath/Config/config_enums.hpp"
#include "Hydro/EOS.hh"

#include <string>

namespace KadathApps
{

struct ColdTableEosParameters
{
    std::string filename;
    double h_cut;
    int interpolation_points;
    double mnuc_cgs;
};

template <typename config_t, typename... node_t>
ColdTableEosParameters configured_cold_table_parameters(config_t& config,
                                                        node_t... nodes)
{
    const int configured_points = config.template eos<int>(INTERP_PTS, nodes...);
    return {
        config.template eos<std::string>(EOSFILE, nodes...),
        config.template eos<double>(HCUT, nodes...),
        configured_points == 0 ? 2000 : configured_points,
        config.template eos<double>(MNUC_CGS, nodes...),
    };
}

template <typename config_t, typename... node_t>
void init_configured_cold_table(config_t& config, node_t... nodes)
{
    using eos_t = Kadath::Margherita::Cold_Table;
    const auto parameters = configured_cold_table_parameters(config, nodes...);
    EOS<eos_t, eos_var_t::PRESSURE>::init(
        parameters.filename, parameters.h_cut, parameters.interpolation_points,
        parameters.mnuc_cgs);
}

} // namespace KadathApps
