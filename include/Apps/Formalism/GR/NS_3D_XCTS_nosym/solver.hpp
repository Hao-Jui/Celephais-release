/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once
#include "Apps/Helper/solver_base.hpp"
#include "Apps/Formalism/GR/syst_init.ipp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/System_of_eqs/solver_runtime_config.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"
#include <string>
namespace Kadath {

template <class eos_t, typename config_t, typename space_t = Space_spheric_adapted_nosym>
class ns_3d_xcts_nosym_solver : public XCTS_Solver<config_t, space_t>
{
  public:
    using typename XCTS_Solver<config_t, space_t>::base_config_t;
    using typename XCTS_Solver<config_t, space_t>::base_space_t;

  private:
    Scalar& conf;
    Scalar& lapse;
    Scalar& logh;
    Vector& shift;
    Metric_flat fmet;

    // Specify base class members used to avoid this->
    using XCTS_Solver<config_t, space_t>::space;
    using XCTS_Solver<config_t, space_t>::bconfig;
    using XCTS_Solver<config_t, space_t>::basis;
    using XCTS_Solver<config_t, space_t>::cfields;
    using XCTS_Solver<config_t, space_t>::coord_vectors;
    using XCTS_Solver<config_t, space_t>::ndom;
    using XCTS_Solver<config_t, space_t>::check_max_iter_exceeded;
    using XCTS_Solver<config_t, space_t>::solution_exists;
    using XCTS_Solver<config_t, space_t>::add_mass_fixing;
    using XCTS_Solver<config_t, space_t>::run_newton_loop;

    // Write the converged dataset for `stage`. The first overload stores a zero
    // velocity-potential placeholder (non-rotating / rotating stages); the second
    // stores the supplied phi (boost stage).
    void save_converged_solution(const std::string& stage);
    void save_converged_solution(const std::string& stage, const Scalar& phi);

    // Shared XCTS equation registration, byte-identical across the stages that
    // call them. Each emits a fixed set of add_cst/add_var/add_def/add_eq calls;
    // callers register them at the point their stage requires.
    void add_spin_constants(System_of_eqs& syst);               // chi, angs, ome
    void add_extrinsic_curvature(System_of_eqs& syst);          // A^ij, intJ
    void add_constraint_equations(System_of_eqs& syst, int d);  // eqP/eqNP/eqbet (matter vs vacuum by domain)
    void add_conformal_field_bcs(System_of_eqs& syst);          // N/P/bet eqs + infinity BCs

    // The matter region is the nucleus + adapted star shell (domains 0,1); every
    // other domain is vacuum. Drives the source terms in add_constraint_equations.
    bool is_vacuum_domain(int d) const { return d > 1; }

  public:
    // solver is not trivially constructable since Kadath containers are not
    // trivially constructable
    ns_3d_xcts_nosym_solver() = delete;

    ns_3d_xcts_nosym_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in, Scalar& conf_in, Scalar& lapse_in,
                      Scalar& logh_in, Vector& shift_in);

    // syst always requires the same initialization for the stages
    void syst_init(System_of_eqs& syst);

    // diagnostics at runtime
    void print_diagnostics(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const override;

    std::string converged_filename(const std::string& stage = "") const override;
    // void update_stages(config_t& old_config);

    // solve driver
    int solve();

    // solver stages
    int norot_stage(bool fixed = false);
    int uniform_rot_stage();

    /**
     * binary_boost_stage
     *
     * based on an input binary Configurator file, we boost the BH accordingly
     *
     * @param[input] binconfig: binary Configurator file
     * @param[input] bco: index of BCO - needed to determine coordinate shift based on BCO location in binary space
     */
    int binary_boost_stage(kadath_config<BIN_INFO>& binconfig, NODES bco);

    // Update bconfig(HC) and bconfig(NC)
    void update_config_quantities(const double& loghc);

    //  template<class eos_t>
    // bool useful_checkpoint_exists();
};

/**
 * ns_3d_xcts_nosym_driver
 *
 * Control computation of 3D NS from setup config/dat file combination
 * filename is pulled from bconfig which should pair with <filename>.dat
 *
 * This code exists to allow the required  NS' to be computed during a BNS solver
 * while also minimizing the code in an isolated solver.
 *
 * Also, KADATH at the time of writing, was strict on not allowing trivial
 * construction of base types (Base_tensor, Scalar, Tensor, Space, etc) which
 * means a driver is required to populate the related Solver class.
 *
 * @tparam[int] bconfig - Configurator file
 * @return Success/failure
 */
template <typename config_t>
inline int ns_3d_xcts_nosym_driver(config_t& bconfig, std::string outputdir,
                             kadath_config<BIN_INFO> binconfig = kadath_config<BIN_INFO>{},
                             NODES bco = BCO1);

} // namespace Kadath
#include "solver_imp.ipp"
#include "stage_helper.ipp"
