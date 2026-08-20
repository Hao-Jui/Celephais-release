// Specialized 2D XCTS differential-rotation solver.
#pragma once
#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
namespace Kadath {

template <class eos_t, typename config_t, typename space_t = Space_polar_adapted>
class ns_2d_msqi_diff_solver : public Solver<config_t, space_t>
{
  public:
    using typename Solver<config_t, space_t>::base_config_t;
    using typename Solver<config_t, space_t>::base_space_t;

  private:
    Scalar& conf;
    Scalar& lapse;
    Scalar& logh;
    Scalar& shift;
    Scalar& metQ;
    Scalar& Omg;

    // Specify base class members used to avoid this->
    using Solver<config_t, space_t>::space;
    using Solver<config_t, space_t>::bconfig;
    using Solver<config_t, space_t>::ndom;
    using Solver<config_t, space_t>::check_max_iter_exceeded;
    using Solver<config_t, space_t>::solution_exists;

    int maybe_skip_stage(int rank, const char* stage_name);
    void init_stage_fields(double& loghc);
    void run_newton_loop(System_of_eqs& syst, double& loghc, int rank);
    void add_common_defs(System_of_eqs& syst);
    void add_domain_equations(System_of_eqs& syst, bool include_intf);
    void add_omega_constraints(System_of_eqs& syst);
    void add_boundary_conditions(System_of_eqs& syst);

  public:
    ns_2d_msqi_diff_solver() = delete;
    ns_2d_msqi_diff_solver(config_t& config_in, space_t& space_in, Scalar& conf_in, Scalar& lapse_in, Scalar& logh_in,
                           Scalar& shift_in, Scalar& metQ_in, Scalar& Omg_in);

    void syst_init(System_of_eqs& syst);

    void print_diagnostics(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const override;

    std::string converged_filename(const std::string& stage = "") const override;

    int solve();

    int keh_stage();
    int uniform_stage();

    void update_config_quantities(const double& loghc);
};

template <typename config_t> inline int ns_2d_msqi_driver(config_t& bconfig, std::string outputdir);


} // namespace Kadath
#include "ns_2d_msqi_solver_imp.ipp"
#include "keh_stages.ipp"
#include "uniform_stages.ipp"
#include "ns_2d_msqi_driver.ipp"
