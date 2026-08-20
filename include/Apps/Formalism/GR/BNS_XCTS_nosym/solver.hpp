#pragma once
#include "Apps/Helper/solver_base.hpp"
#include "Apps/Formalism/GR/syst_init.ipp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Domain/spheric_adapted_nosym.hpp"

#include <cstdlib>
#include <string>
namespace Kadath {

// Standalone no-symmetry BNS XCTS solver. Mirrors bns_xcts_solver but owns its own
// stages (bns_xcts_nosym_stages.cpp) so the no-symmetry rotational-gauge closure can
// be built directly into the equation setup without touching the symmetric solver.
template <class eos_t, typename config_t, typename space_t = Space_bin_ns_nosym>
class bns_xcts_nosym_solver : public XCTS_Solver<config_t, space_t>
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

    // Bordered rotational-gauge multipliers (see syst_init / stage eqbet defs).
    // Persist as members so add_var can bind them across the stage's System_of_eqs.
    double muz1_gauge{0.};
    double muz2_gauge{0.};

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

    // The no-symmetry bispheric basis admits the local per-star rigid-z-rotation of the
    // shift (a near-kernel of the XCTS longitudinal operator) that the symmetric parity
    // split structurally forbids, breaking the first Newton step. When this flag is set,
    // the solver closes that gauge with a bordered constraint: curl_z(bet) = 0 pinned at
    // each stellar center, paired with a multiplier that forces the rigid-rotation
    // direction (mmz/mpz) in eqbet. Toggle via BNS_NOSYM_ROT_GAUGE (default off so
    // the bare divergence remains reproducible for A/B comparison).
    bool rotational_gauge_enabled() const
    {
        const char* value = std::getenv("BNS_NOSYM_ROT_GAUGE");
        if (value == nullptr || value[0] == '\0')
            return false;
        const std::string text(value);
        return text != "0" && text != "false" && text != "off" && text != "no";
    }

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
    bns_xcts_nosym_solver() = delete;

    bns_xcts_nosym_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in, Scalar& conf_in,
                          Scalar& lapse_in, Vector& shift_in, Scalar& logh_in, Scalar& phi_in);

    // syst always requires the same initialization for the stages
    void syst_init(System_of_eqs& syst);

    // diagnostics at runtime
    void print_diagnostics(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const override;

    std::string converged_filename(const std::string& stage = "") const override;

    /**
     * hydrostatic_equilibrium_stage — fallback when ecc_red = false.
     * Original nosym force-balance solve (free Omega).
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

template <typename eos_t>
inline void bns_setup_boosted_3d_nosym(kadath_config<BCO_NS_INFO>& NS1config,
                                       kadath_config<BCO_NS_INFO>& NS2config,
                                       kadath_config<BIN_INFO>& bconfig)
{
    bns_setup_boosted_3d<eos_t, Space_spheric_adapted_nosym, Space_bin_ns_nosym,
                         Domain_shell_outer_adapted_nosym, Domain_shell_inner_adapted_nosym>(
        NS1config, NS2config, bconfig);
}


} // namespace Kadath
#include "solver_imp.ipp"
#include "stages.ipp"
