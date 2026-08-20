/*
 * Copyright 2022
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 *
 * Author:
 * Samuel D. Tootle <tootle@itp.uni-frankfurt.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once
#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted_bh_nosym.hpp"
namespace Kadath {

template <typename config_t, typename space_t = Space_adapted_bh_nosym>
class bh_3d_xcts_nosym_solver : public XCTS_Solver<config_t, space_t>
{
  public:
    using typename XCTS_Solver<config_t, space_t>::base_config_t;
    using typename XCTS_Solver<config_t, space_t>::base_space_t;

    using scalar_t = Scalar&;
    using vector_t = Vector&;

  private:
    scalar_t conf;
    scalar_t lapse;
    vector_t shift;
    Metric_flat fmet;
    std::array<int, 2> excluded_doms{0, 1};

    // Specify base class members used to avoid this->
    using XCTS_Solver<config_t, space_t>::space;
    using XCTS_Solver<config_t, space_t>::bconfig;
    using XCTS_Solver<config_t, space_t>::basis;
    using XCTS_Solver<config_t, space_t>::cfields;
    using XCTS_Solver<config_t, space_t>::coord_vectors;
    using XCTS_Solver<config_t, space_t>::ndom;
    using XCTS_Solver<config_t, space_t>::check_max_iter_exceeded;
    using XCTS_Solver<config_t, space_t>::solution_exists;

  public:
    // solver is not trivially constructable since Kadath containers are not
    // trivially constructable
    bh_3d_xcts_nosym_solver() = delete;

    bh_3d_xcts_nosym_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in, Scalar& conf_in,
                            Scalar& lapse_in, Vector& shift_in);

    // syst always requires the same initialization for the stages
    void syst_init(System_of_eqs& syst);

    // diagnostics at runtime
    void print_diagnostics_norot(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const;
    void print_diagnostics(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const override;

    std::string converged_filename(const std::string& stage = "") const override;
    // void update_stages(config_t& old_config);

    // solve driver
    int solve();

    // solver stages
    int fixed_lapse_stage();
    int von_Neumann_stage();

    /**
     * binary_boost_stage
     *
     * based on an input binary Configurator file, we boost the BH accordingly
     *
     * @param[input] binconfig: binary Configurator file
     * @param[input] bco: index of BCO - needed to determine coordinate shift based on BCO location in binary space
     */
    int binary_boost_stage(kadath_config<BIN_INFO>& binconfig, NODES bco);
};

/**
 * bh_3d_xcts_nosym_driver
 *
 * Control computation of 3D BH from setup config/dat file combination
 * filename is pulled from bconfig which should pair with <filename>.dat
 *
 * This code exists to allow the required  BHs to be computed during a BBH
 * or BHNS solver while also simplifying the code for the isolated solver.
 *
 * Also, KADATH at the time of writing, was strict on not allowing trivial
 * construction of base types (Base_tensor, Scalar, Tensor, Space, etc) which
 * means a driver is required to populate the related Solver class.
 *
 * @tparam[int] bconfig - Configurator file
 * @return Success/failure
 */
template <typename config_t>
inline int bh_3d_xcts_nosym_driver(config_t& bconfig, std::string outputdir,
                                   kadath_config<BIN_INFO> binconfig = kadath_config<BIN_INFO>{},
                                   NODES bco = BCO1);

template <typename config_t>
inline int bh_3d_xcts_nosym_bin_boost_driver(config_t& bconfig, kadath_config<BIN_INFO> binconfig, NODES bco);


} // namespace Kadath
#include "solver_imp.ipp"
#include "stages.ipp"
