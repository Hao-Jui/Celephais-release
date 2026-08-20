#pragma once
#include "Apps/Helper/solver_base.hpp"
#include "Apps/Formalism/Shared/omega_mode.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "Apps/Formalism/GR/BH_3D_XCTS/solver.hpp"
#include "For_Kadath/Kadath_point_h/kadath_bin_bh.hpp"
namespace Kadath {

template <typename config_t, typename space_t = Space_bin_bh>
class bbh_xcts_solver : public XCTS_Solver<config_t, space_t>
{
  public:
    using typename XCTS_Solver<config_t, space_t>::base_config_t;
    using typename XCTS_Solver<config_t, space_t>::base_space_t;

  private:
    Scalar& conf;
    Scalar& lapse;
    Vector& shift;
    Metric_flat fmet;

    const double xc1;
    const double xc2;
    const double xo{0.};
    std::array<int, 4> excluded_doms;

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
    bbh_xcts_solver() = delete;

    bbh_xcts_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in, Scalar& conf_in, Scalar& lapse_in,
                    Vector& shift_in);

    // syst always requires the same initialization for the stages
    void syst_init(System_of_eqs& syst);

    // diagnostics at runtime
    void print_diagnostics(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const override;

    std::string converged_filename(const std::string& stage = "") const override;
    // void update_stages(config_t& old_config);

    // solve driver. `want_warmup` requests the one-shot fixed-Omega warm-up pass
    // (computed and owned by the workflow); the config flag is never mutated.
    int solve(bool want_warmup);

    // One stage method for both passes; `omega_mode` selects fixed vs free Omega
    // (add_cst vs add_var "ome", and whether the quasi-equilibrium constraint is
    // added). Replaces the in-body config-flag read.
    int solve_stage(STAGE stage, OmegaMode omega_mode, std::string stage_text);
};

inline void bbh_xcts_setup_boosted_3d(kadath_config<BCO_BH_INFO>& BH1config,
                                      kadath_config<BCO_BH_INFO>& BH2config,
                                      kadath_config<BIN_INFO>& bconfig);

} // namespace Kadath
#include "solver_imp.ipp"
#include "stages.ipp"
