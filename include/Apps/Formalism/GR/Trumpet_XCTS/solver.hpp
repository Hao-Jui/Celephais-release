#pragma once

#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_trumpet.hpp"

namespace Kadath {

template <typename config_t, typename space_t = Space_trumpet>
class trumpet_3d_xcts_solver : public XCTS_Solver<config_t, space_t>
{
  public:
    using scalar_t = Scalar&;
    using vector_t = Vector&;

  private:
    scalar_t conf;
    scalar_t lapse;
    vector_t shift;
    Metric_flat fmet;

    using XCTS_Solver<config_t, space_t>::space;
    using XCTS_Solver<config_t, space_t>::bconfig;
    using XCTS_Solver<config_t, space_t>::basis;
    using XCTS_Solver<config_t, space_t>::cfields;
    using XCTS_Solver<config_t, space_t>::coord_vectors;
    using XCTS_Solver<config_t, space_t>::ndom;
    using XCTS_Solver<config_t, space_t>::check_max_iter_exceeded;
    using XCTS_Solver<config_t, space_t>::run_newton_loop;
    using XCTS_Solver<config_t, space_t>::solution_exists;

  public:
    trumpet_3d_xcts_solver() = delete;

    trumpet_3d_xcts_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in, Scalar& conf_in,
                           Scalar& lapse_in, Vector& shift_in);

    void syst_init(System_of_eqs& syst, Scalar& trumpet_conf, Scalar& trumpet_lapse,
                   Scalar& trumpet_lapse_conf, Vector& trumpet_shift);

    void print_diagnostics(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const override;

    std::string converged_filename(const std::string& stage = "") const override;

    int solve();
    int trumpet_stage();
};

template <typename config_t>
inline int trumpet_3d_xcts_driver(config_t& bconfig, std::string outputdir);

} // namespace Kadath

#include "solver_imp.ipp"
#include "stages.ipp"
