#pragma once
#include "Apps/Helper/solver_base.hpp"
#include "Apps/Formalism/GR/syst_init.ipp"
#include "Apps/Formalism/Shared/omega_mode.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Space/bhns_nosym.hpp"

#include <cstdlib>
#include <string>
namespace Kadath {

// Standalone no-symmetry BHNS XCTS solver. Mirrors bhns_xcts_solver but owns its own
// stages so the no-symmetry rotational-gauge closure can be built directly into the
// equation setup without touching the symmetric solver.
template <class eos_t, typename config_t, typename space_t = Space_bhns_nosym>
class bhns_xcts_nosym_solver : public XCTS_Solver<config_t, space_t>
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

    // Bordered rotational-gauge multiplier (see syst_init / stage eqbet defs).
    // Persists as a member so add_var can bind it across the stage's System_of_eqs.
    // Only the NS side carries a rigid-z-rotation gauge near-kernel; the BH side is
    // excised/von-Neumann and keeps the plain shift equation.
    double muz1_gauge{0.};

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

    // The no-symmetry bispheric basis admits the local NS rigid-z-rotation of the
    // shift (a near-kernel of the XCTS longitudinal operator) that the symmetric parity
    // split structurally forbids, breaking the first Newton step. When this flag is set,
    // the solver closes that gauge with a bordered constraint: curl_z(bet) = 0 pinned at
    // the NS center, paired with a multiplier that forces the rigid-rotation direction
    // (mmz) in eqbet. Toggle via BHNS_NOSYM_ROT_GAUGE (default off so the bare
    // divergence remains reproducible for A/B comparison).
    bool rotational_gauge_enabled() const
    {
        const char* value = std::getenv("BHNS_NOSYM_ROT_GAUGE");
        if (value == nullptr || value[0] == '\0')
            return false;
        const std::string text(value);
        return text != "0" && text != "false" && text != "off" && text != "no";
    }

  public:
    // solver is not trivially constructable since Kadath containers are not
    // trivially constructable
    bhns_xcts_nosym_solver() = delete;

    bhns_xcts_nosym_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in, Scalar& conf_in,
                           Scalar& lapse_in, Vector& shift_in, Scalar& logh_in, Scalar& phi_in);

    // syst always requires the same initialization for the stages
    void syst_init(System_of_eqs& syst);

    // diagnostics at runtime
    void print_diagnostics(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const override;

    std::string converged_filename(const std::string& stage = "") const override;

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

template <typename eos_t>
inline void bhns_setup_boosted_3d_nosym(kadath_config<BCO_NS_INFO>& NSconfig,
                                        kadath_config<BCO_BH_INFO>& BHconfig, kadath_config<BIN_INFO>& bconfig);


} // namespace Kadath
#include "solver_imp.ipp"
#include "stages.ipp"
