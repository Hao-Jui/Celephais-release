/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once
#include "Apps/Helper/solver_base.hpp"
#include "Apps/Formalism/GR/syst_init.ipp"
#include "Apps/Formalism/Shared/Stages/ns_3d_xcts_binary_boost_stage.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"

namespace Kadath {

template <class eos_t, typename config_t, typename space_t = Space_spheric_adapted>
class ns_3d_xcts_solver : public XCTS_Solver<config_t, space_t>
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

  public:
    // solver is not trivially constructable since Kadath containers are not
    // trivially constructable
    ns_3d_xcts_solver() = delete;

    ns_3d_xcts_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in, Scalar& conf_in, Scalar& lapse_in,
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

    // Shared binary-boost body needs the derived-private field references and
    // the base-protected solution_exists / run_newton_loop helpers.
    template <bool WithScalar, typename SolverT, typename AdvectionDefs, typename ScalarSetup,
              typename FieldEqs, typename ScalarGlobalEqs>
    friend int ns_3d_xcts_run_binary_boost_stage(SolverT& solver, kadath_config<BIN_INFO>& binconfig,
                                                 NODES bco, const char* banner,
                                                 AdvectionDefs&& emit_advection_defs,
                                                 ScalarSetup&& emit_scalar_setup, FieldEqs&& emit_field_equations,
                                                 ScalarGlobalEqs&& emit_scalar_global_eqs);

    // Shared norot / uniform_rot stage bodies and their save helper need the
    // same access (see Apps/Formalism/Shared/Stages/ns_3d_xcts_solve_stages.hpp).
    template <bool WithScalar, typename SolverT, typename ScalarSetup, typename FieldEqs, typename ScalarGlobalEqs>
    friend int ns_3d_xcts_run_norot_stage(SolverT& solver, bool fixed, const char* banner_head,
                                          const SolverRuntimeConfig& solver_config, ScalarSetup&& emit_scalar_setup,
                                          FieldEqs&& emit_field_equations, ScalarGlobalEqs&& emit_scalar_global_eqs);
    template <bool WithScalar, typename SolverT, typename ScalarSetup, typename FieldEqs, typename ScalarGlobalEqs>
    friend int ns_3d_xcts_run_uniform_rot_stage(SolverT& solver, const char* banner_head, const char* stilde_def,
                                                bool print_initial_residual_norm,
                                                const SolverRuntimeConfig& solver_config,
                                                ScalarSetup&& emit_scalar_setup, FieldEqs&& emit_field_equations,
                                                ScalarGlobalEqs&& emit_scalar_global_eqs);
    template <bool WithScalar, typename SolverT>
    friend void ns_3d_xcts_save_stage_fields(SolverT& solver);

    //  template<class eos_t>
    // bool useful_checkpoint_exists();
};

// The NS_3D_XCTS from-file driver is now NsPolicyCommon::solve_ns_xcts<GrNsDriverTraits>
// (Apps/Policy/ns_policy_common.hpp); GrNsDriverTraits is defined in solver_imp.ipp.

} // namespace Kadath
#include "solver_imp.ipp"
#include "stage_helper.ipp"
