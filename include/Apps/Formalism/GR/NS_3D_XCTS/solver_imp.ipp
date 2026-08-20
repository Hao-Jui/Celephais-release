#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Kadath_point_h/kadath_spheric_homothetic.hpp"
#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include <sstream>
#include "mpi.h"
#include "regrid.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include <cmath>
#include <utility>
#include "Apps/Formalism/Shared/ns_driver_common.hpp"
#include "Apps/Workflow/ns_solve_driver.hpp"
#include "Apps/Formalism/Shared/ns_solver_instantiate.hpp"
#include "Apps/Formalism/Shared/ns_2d_seed.hpp"
#include "Apps/Formalism/Shared/xcts_syst_init.ipp"
#include "Apps/Workflow/ns_resolution_sequence.hpp"
#include "Apps/Formalism/GR/NS_2D_XCTS/solver.hpp"

using namespace Kadath::Margherita;

namespace Kadath {

template <class eos_t, typename config_t, typename space_t>
ns_3d_xcts_solver<eos_t, config_t, space_t>::ns_3d_xcts_solver(config_t& config_in, space_t& space_in,
                                                               Base_tensor& base_in, Scalar& conf_in, Scalar& lapse_in,
                                                               Scalar& logh_in, Vector& shift_in)
    : XCTS_Solver<config_t, space_t>(config_in, space_in, base_in), conf(conf_in), lapse(lapse_in), logh(logh_in),
      shift(shift_in), fmet(Metric_flat(space_in, base_in))
{
    coord_vectors = default_co_vector_ary(space);
    update_fields_co(cfields, coord_vectors, {}, 0.);
}

template <class eos_t, typename config_t, typename space_t>
std::string ns_3d_xcts_solver<eos_t, config_t, space_t>::converged_filename(const std::string& stage) const
{
    return converged_gr_ns_filename(bconfig, space, stage);
}

template <class eos_t, typename config_t, typename space_t>
int ns_3d_xcts_solver<eos_t, config_t, space_t>::solve()
{
    return ns_3d_xcts_solve_common(*this, bconfig);
}

// ---------------------------------------------------------------------------
// Formalism driver traits consumed by NsPolicyCommon::solve_ns_xcts. Describes the
// GR NS solver to the shared load -> construct -> drive machinery: the solver
// template, its checkpoint layout (no trailing scalar field), and its spaces
// (homothetic for the NOROT-stage solve and the resolution ladder, adapted for
// rotating stages + binary boost). A distinct type per formalism, so
// solve_ns_xcts<GrNsDriverTraits> is a distinct symbol (the previous shared-name
// ns_3d_xcts_driver collided under ODR).
// ---------------------------------------------------------------------------
struct GrNsDriverTraits
{
    template <typename eos_t, typename config_t, typename space_t>
    using solver_t = ns_3d_xcts_solver<eos_t, config_t, space_t>;

    static constexpr bool has_scalar = false;

    using ladder_space   = Space_spheric_homothetic;    // isolated-ladder primary
    using rotating_space = Space_spheric_adapted;       // non-NOROT solve + boost
    using norot_space    = Space_spheric_homothetic;    // NOROT solve

    // GR carries no field beyond the common checkpoint prefix.
    template <typename config_t, typename Fields>
    static int load_extra(config_t& /*bconfig*/, Fields& /*fields*/) { return 0; }

    template <typename config_t>
    static int regrid(config_t& bc, int new_res, std::string filename)
    {
        const auto stages = bc.return_stages();
        if (stages[to_int(NOROT)])
            return ns_3d_xcts_interpolate_on_new_grid<config_t, Space_spheric_homothetic>(bc, new_res, filename);

        return ns_3d_xcts_interpolate_on_new_grid<config_t, Space_spheric_adapted>(bc, new_res, filename);
    }

    template <typename config_t>
    static void seed_initial_data(config_t& bc, int current_res, int final_res)
    {
        ns_2d_xcts_seed_initial_data<rotating_space>(
            bc, current_res, final_res,
            [](auto& config, const std::string& outputdir, const std::string& basename) {
                return ns_2d_xcts_driver(config, outputdir, basename);
            },
            "NS", "initns");
    }
};

// ---------------------------------------------------------------------------
// System initialization
// ---------------------------------------------------------------------------
template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_solver<eos_t, config_t, space_t>::syst_init(System_of_eqs& syst)
{
    using namespace Kadath::Margherita;

    auto& space = syst.get_space();
    const int ndom = space.get_nbr_domains();
    xcts::add_flat_metric(fmet, syst);

    xcts::add_four_pi_g(syst);

    xcts::add_global_rot_and_surface_coords(syst, coord_vectors);

    xcts::add_conformal_lapse_vars(syst, conf, lapse);

    xcts::add_conformal_combinations_and_adm_integrands(syst, ndom);

    Param p;
    xcts::add_eos_operators<eos_t>(syst, p);

    xcts::add_thermodynamic_defs(syst);
}

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_solver<eos_t, config_t, space_t>::print_diagnostics(System_of_eqs const& syst, const int ite,
                                                                    const double conv) const
{
    ns_3d_xcts_print_diagnostics(space, bconfig, syst, ite, conv);
    std::cout << "=======================================" << "\n\n";
}

template <class eos_t, typename config_t, typename space_t>
void ns_3d_xcts_solver<eos_t, config_t, space_t>::update_config_quantities(const double& loghc)
{
    xcts::update_config_quantities<eos_t>(bconfig, loghc);
}

} // namespace Kadath
