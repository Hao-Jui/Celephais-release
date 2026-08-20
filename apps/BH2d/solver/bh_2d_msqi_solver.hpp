#pragma once
#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Space/adapted_bh_polar.hpp"
#include "coord_fields_2d.hpp"
namespace Kadath {

template <typename config_t, typename space_t = Space_adapted_bh_polar>
class bh_2d_msqi_solver : public Solver<config_t, space_t>
{
  public:
    using typename Solver<config_t, space_t>::base_config_t;
    using typename Solver<config_t, space_t>::base_space_t;

  private:
    Scalar& lapse;
    Scalar& metA;
    Scalar& metB;
    Scalar& beta;
    Scalar& varscal;
    CoordFields2D<space_t> cfields;
    std::array<int, 2> excluded_doms{0, 1};

    // Specify base class members used to avoid this->
    using Solver<config_t, space_t>::space;
    using Solver<config_t, space_t>::bconfig;
    using Solver<config_t, space_t>::ndom;
    using Solver<config_t, space_t>::check_max_iter_exceeded;
    using Solver<config_t, space_t>::solution_exists;

    double compute_max_sphi() const;

    void add_common_defs(System_of_eqs& syst, bool fixed_lapse);
    void add_domain_equations(System_of_eqs& syst, bool fixed_lapse);
    void add_boundary_conditions(System_of_eqs& syst, bool fixed_lapse);
    int run_stage(bool fixed_lapse, const char* resume_tag, const char* banner, const char* fname_tag,
                  bool update_fixed_lapse);

  public:
    bh_2d_msqi_solver() = delete;
    bh_2d_msqi_solver(config_t& config_in, space_t& space_in, Scalar& lapse_in, Scalar& metA_in, Scalar& metB_in,
                      Scalar& beta_in, Scalar& varscal_in);

    void syst_init(System_of_eqs& syst);
    void debug_operators(System_of_eqs& syst_dbg);
    void print_diagnostics(const System_of_eqs& syst, const int ite = 0, const double conv = 0) const override;

    std::string converged_filename(const std::string& stage = "") const override;

    int solve();

    // solver stages
    int fixed_lapse_stage();
    int von_Neumann_stage();
};

template <typename config_t> inline int bh_2d_msqi_driver(config_t& bconfig, std::string outputdir);


} // namespace Kadath
#include "bh_2d_msqi_solver_imp.ipp"
#include "bh_2d_msqi_stages.ipp"
#include "bh_2d_msqi_driver.ipp"
