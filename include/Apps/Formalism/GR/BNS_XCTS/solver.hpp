#pragma once
#include "Apps/Helper/solver_base.hpp"
#include "Apps/Formalism/GR/syst_init.ipp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"
namespace Kadath {

template <class eos_t, typename config_t, typename space_t = Space_bin_ns>
class bns_xcts_solver : public XCTS_Solver<config_t, space_t>
{
  public:
    using typename XCTS_Solver<config_t, space_t>::base_config_t;
    using typename XCTS_Solver<config_t, space_t>::base_space_t;

  private:
    Scalar& conf;
    Scalar& lapse;
    Scalar& logh;
    Scalar& phi;
    Vector& shift;
    Metric_flat fmet;

    const double xc1;
    const double xc2;
    const double xo{0.};

    // Specify base class members used to avoid this->
    using XCTS_Solver<config_t, space_t>::space;
    using XCTS_Solver<config_t, space_t>::bconfig;
    using XCTS_Solver<config_t, space_t>::basis;
    using XCTS_Solver<config_t, space_t>::cfields;
    using XCTS_Solver<config_t, space_t>::coord_vectors;
    using XCTS_Solver<config_t, space_t>::ndom;
    using XCTS_Solver<config_t, space_t>::check_max_iter_exceeded;
    using XCTS_Solver<config_t, space_t>::run_newton_loop;
    using XCTS_Solver<config_t, space_t>::solution_exists;

    // Shared stage bodies (Apps/Formalism/Shared/Stages/bns_stages_common.ipp) need the
    // derived-private field references and base-protected helpers (run_newton_loop).
    template <typename SolverT, typename MatterEqbetDef, typename PzVelocity, typename PzConstraint>
    friend int bns_run_force_balance_stage(SolverT& solver, MatterEqbetDef&& matter_eqbet_def,
                                           PzVelocity&& pz_velocity, PzConstraint&& pz_constraint);
    template <typename SolverT, typename MatterEqbetDef, typename PzVelocity, typename PzConstraint>
    friend int bns_run_hydro_rescaling_stage(SolverT& solver, STAGE stage, const std::string& stage_text,
                                             MatterEqbetDef&& matter_eqbet_def, PzVelocity&& pz_velocity,
                                             PzConstraint&& pz_constraint);

  public:
    // solver is not trivially constructable since Kadath containers are not
    // trivially constructable
    bns_xcts_solver() = delete;

    bns_xcts_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in, Scalar& conf_in, Scalar& lapse_in,
                    Vector& shift_in, Scalar& logh_in, Scalar& phi_in);

    // syst always requires the same initialization for the stages
    void syst_init(System_of_eqs& syst);

    // diagnostics at runtime
    void print_diagnostics(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const override;

    std::string converged_filename(const std::string& stage = "") const override;
    // void update_stages(config_t& old_config);

    /**
     * hydrostatic_equilibrium_stage — fallback when ecc_red = false.
     *
     * Omega is a Newton unknown closed by the force-balance point conditions;
     * first integral + mass integrals fix the matter.
     */
    int hydrostatic_equilibrium_stage();

    // solve driver for the shared binary workflow.
    int solve();

    /**
     * hydro_rescaling_stage
     *
     * QUASI_EQUIL fixes the PN orbital velocity and rescales the matter to the
     * target baryonic masses. ECC_RED reuses the rescale body with the
     * eccentricity-reduction label and radial-infall contribution.
     */
    int hydro_rescaling_stage(STAGE stage, const std::string& stage_text);

    int ecc_red_stage();

    // Update bconfig(HC) and bconfig(NC)
    void update_config_quantities(const double& loghc);
};

template <typename eos_t, typename NSpaceT = Space_spheric_adapted, typename BinSpaceT = Space_bin_ns,
          typename NSOuterAdaptedT = Domain_shell_outer_adapted,
          typename NSInnerAdaptedT = Domain_shell_inner_adapted,
          typename BinOuterAdaptedT = NSOuterAdaptedT,
          typename BinInnerAdaptedT = NSInnerAdaptedT>
inline void bns_setup_boosted_3d(kadath_config<BCO_NS_INFO>& NS1config,
                                 kadath_config<BCO_NS_INFO>& NS2config, kadath_config<BIN_INFO>& bconfig);


} // namespace Kadath
#include "solver_imp.ipp"
#include "stages.ipp"
