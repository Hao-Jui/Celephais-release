#pragma once
#include "Apps/Helper/solver_base.hpp"
#include "Apps/Formalism/Shared/omega_mode.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Space/bhns.hpp"
namespace Kadath {

template <class eos_t, typename config_t, typename space_t = Space_bhns>
class bhns_xcts_solver : public XCTS_Solver<config_t, space_t>
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
    std::array<int, 2> excluded_doms;

    // Specify base class members used to avoid this->
    using XCTS_Solver<config_t, space_t>::space;
    using XCTS_Solver<config_t, space_t>::bconfig;
    using XCTS_Solver<config_t, space_t>::basis;
    using XCTS_Solver<config_t, space_t>::cfields;
    using XCTS_Solver<config_t, space_t>::coord_vectors;
    using XCTS_Solver<config_t, space_t>::ndom;
    using XCTS_Solver<config_t, space_t>::check_max_iter_exceeded;
    using XCTS_Solver<config_t, space_t>::solution_exists;

    // Shared stage bodies (Apps/Formalism/Shared/Stages/bhns_stages_common.ipp) need the
    // derived-private field references and base-protected helpers.
    template <typename SolverT, typename MatterEqbetDef, typename PzVelocity, typename PzConstraint>
    friend int bhns_run_hydrostatic_equilibrium_stage(SolverT& solver, STAGE stage, OmegaMode omega_mode,
                                                      const std::string stage_text, MatterEqbetDef&& matter_eqbet_def,
                                                      PzVelocity&& pz_velocity, PzConstraint&& pz_constraint);
    template <typename SolverT, typename MatterEqbetDef, typename PzVelocity, typename PzConstraint>
    friend int bhns_run_hydro_rescaling_stages(SolverT& solver, STAGE stage, OmegaMode omega_mode,
                                               std::string stage_text, MatterEqbetDef&& matter_eqbet_def,
                                               PzVelocity&& pz_velocity, PzConstraint&& pz_constraint);

  public:
    // solver is not trivially constructable since Kadath containers are not
    // trivially constructable
    bhns_xcts_solver() = delete;

    bhns_xcts_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in, Scalar& conf_in, Scalar& lapse_in,
                     Vector& shift_in, Scalar& logh_in, Scalar& phi_in);

    // syst always requires the same initialization for the stages
    void syst_init(System_of_eqs& syst);

    // diagnostics at runtime
    void print_diagnostics(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const override;

    std::string converged_filename(const std::string& stage = "") const override;
    // void update_stages(config_t& old_config);

    // solve driver. `want_warmup` is kept for app-boundary compatibility; the
    // binary workflow uses explicit stage gates.
    int solve(bool want_warmup);

    /**
     * hydrostatic_equilibrium stage
     *
     * The binary is solved using the force balance equation
     * in addition to solving the relativisitc Euler equation
     * to obtain a binary in with the NS in hydrostatic equilibrium.
     *
     * @param[input] omega_mode: Fixed holds the orbital frequency (and x-axis)
     *   constant and drops the ADM-momentum integrals; Free solves for them.
     */
    int hydrostatic_equilibrium_stage(STAGE stage, OmegaMode omega_mode, const std::string stage_text);
    /**
     * hydro_rescaling_stage
     *
     * The binary is solved using fixed orbital velocity
     * therefore the matter scalar fields are simply rescaled
     * based on the fixed baryonic mass of the NS.
     *
     * @param[input] stage: Some changes are made based on QUASI_EQUIL or ECC_RED stage.
     * @param[input] omega_mode: Fixed for this stage (the orbital frequency is
     *   always held constant). Carried for a uniform stage signature.
     */
    int hydro_rescaling_stages(STAGE stage, OmegaMode omega_mode, std::string stage_text);

    // Update bconfig(HC) and bconfig(NC)
    void update_config_quantities(const double& loghc);
};


} // namespace Kadath
#include "solver_imp.ipp"
#include "stages.ipp"
