// Axisymmetric conformally-flat XCTS neutron-star solver.
// The shift is represented by brsint = r sin(theta) beta^phi, matching NS2d.

#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include "mpi.h"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "For_Kadath/Array/memory.hpp"
#include "Apps/Helper/newton_loop_runner.hpp"
#include "Apps/Formalism/Shared/NS_2D_XCTS/core.hpp"
#include <cmath>
#include <cstdlib>

using namespace Kadath::Margherita;

namespace Kadath {


template <class eos_t, typename config_t, typename space_t>
ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::ns_2d_xcts_diff_solver(config_t& config_in, space_t& space_in,
                                                                         Scalar& conf_in, Scalar& lapse_in,
                                                                         Scalar& logh_in, Scalar& shift_in,
                                                                         Scalar& Omg_in)
    : Solver<config_t, space_t>(config_in, space_in), conf(conf_in), lapse(lapse_in), logh(logh_in), shift(shift_in),
      Omg(Omg_in)
{
    shift.affect_parameters();
    shift.set_parameters()->set_m_quant() = 1;
    shift.std_base();
}

template <class eos_t, typename config_t, typename space_t>
int ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::maybe_skip_stage(int rank, const char* stage_name)
{
    const auto resume = this->load_existing_solution(stage_name);
    if (!resume.found()) {
        return -1;
    }
    if (rank == 0) {
        std::cout << "Solved previously: " << bconfig.config_filename_abs() << std::endl;
    }
    return legacy_status_from_dataset_resume(resume);
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::init_stage_fields(double& loghc)
{
    loghc = std::log(bconfig(HC));
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::syst_init(System_of_eqs& syst)
{
    ns_2d_xcts_core::add_base_system<eos_t>(syst, bconfig, conf, lapse, logh, shift, Omg);
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::add_common_defs(System_of_eqs& syst)
{
    ns_2d_xcts_core::add_common_definitions(syst, ndom);
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::add_domain_equations(System_of_eqs& syst, bool include_intf)
{
    const std::string firstint_expr = include_intf ? "H + log(N) - log(W) + intF" : "H + log(N) - log(W)";

    const int adapted_inner = space.ADAPTED_INNER;
    for (int d = 0; d < ndom; d++) {
        if (d < adapted_inner) {
            syst.add_def(d, "U = multrsint(P^2 / N * (Omg + bet))"); // amplitude of spatial velocity
            syst.add_def(d, "Usq = U*U");
            syst.add_def(d, "Wsquare = 1 / (1 - Usq)");
            syst.add_def(d, "W = sqrt(Wsquare)");

            // XCTS matter sources, projected to the axisymmetric scalar shift.
            syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
            syst.add_def(d, "Stilde = 3. * press * delta + press * h * Wsquare * Usq");
            syst.add_def(d, "Jphi = P^2 * (Etilde + press * delta) * U"); // J_φ / (r*sin(θ))

            syst.add_def(d, "eqP = delta * lap(P) + Asquare / P^7 / 8. * delta + 4piG / 2. * P^5 * Etilde");
            syst.add_def(d, "eqNP = delta * lap(NP) - 7. / 8. * NP / P^8 * delta * Asquare"
                            " - 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
            syst.add_def(d, "eqbrsint = lap(brsint) * delta - 4. * 4piG * N * Jphi"
                            " - delta * multrsint(scal(gradbet, grad(log(Ntilde))))");
            syst.add_def(d, "intMb = W * rho * P^6 * 4piG / 2");
            const std::string firstint_def = std::string("firstint = ") + firstint_expr;
            syst.add_def(d, firstint_def.c_str());
            syst.add_eq_full(d, "rotlaw = 0");
        } else {
            syst.add_eq_full(d, "H = 0");
            syst.add_def(d, "eqP = lap(P) + Asquare / P^7 / 8.");
            syst.add_def(d, "eqNP = lap(NP) - 7. / 8. * NP / P^8 * Asquare");
            syst.add_def(d, "eqbrsint = lap(brsint) - multrsint(scal(gradbet, grad(log(Ntilde))))");
        }
    }
    space.add_eq(syst, "eqNP= 0", "N", "dn(N)");
    space.add_eq(syst, "eqP= 0", "P", "dn(P)");
    space.add_eq(syst, "eqbrsint= 0", "brsint", "dn(brsint)");
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::add_omega_constraints(System_of_eqs& syst)
{
    ns_2d_xcts_core::add_rotation_constraints(space, syst, ndom);
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::add_boundary_conditions(System_of_eqs& syst)
{
    ns_2d_xcts_core::add_global_conditions(space, syst, ndom);
}

template <class eos_t, typename config_t, typename space_t>
int ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::solve()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exit_status = EXIT_SUCCESS;
    std::string law = bconfig.template diffrot<std::string>(DIFF_LAW);
    std::array<bool, NUM_STAGES_V> stage_enabled = bconfig.return_stages();
    auto [last_stage, last_stage_idx] = get_last_enabled(MSTAGE(), stage_enabled);
    if (rank == 0) {
        std::cout << "Last stage: " << last_stage << "\n";
    }

    if (stage_enabled[to_int(UNIROT)]) {
        bconfig.set_stage(NOROT) = false;
        if (law != "uniform") {
            KADATH_THROW("NS2d_xcts currently supports only uniform rotation.");
        }
        exit_status = uniform_stage();
        if (status_requires_field_reload(exit_status)) {
            return exit_status;
        }
    }
    // Barrier needed in case we need to read from the previous output.
    MPI_Barrier(MPI_COMM_WORLD);
    return exit_status;
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::run_newton_loop(System_of_eqs& syst, double& loghc, int rank)
{
    bool endloop = false;
    int ite = 1;
    double conv = 0.;
    SolverRuntimeConfig solver_config = SolverRuntimeConfig::from_environment();
    solver_config = this->with_stage_mumps_tree_cache(
        solver_config, "UNIROT");
    if (solver_config.backend == NewtonBackend::JfnkMumps) {
        // Small, JFNK-sensitive 2D XCTS solve: use the same sparse Jacobian with
        // a direct MUMPS Newton step (matches the NS_3D_XCTS boost stage).
        solver_config.backend = NewtonBackend::Mumps;
    }
    while (!endloop) {
        endloop = newton_step_with_consensus(syst, bconfig.template seq_setting_as<double>(PREC), conv, solver_config, ite == 1);
        update_config_quantities(loghc);
        std::stringstream ss;
        ss << "rot_3d_total" << ite - 1;
        bconfig.set(QLMADM) = bconfig(MADM);
        bconfig.set_filename(ss.str());
        if (rank == 0) {
            print_diagnostics(syst, ite, conv);
            if (bconfig.control(CHECKPOINT))
                bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh, Omg);
        }
        ite++;
        check_max_iter_exceeded(rank, ite, conv);
    }
    bconfig.set_filename(converged_filename("UNIROT"));
    if (rank == 0)
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh, Omg);
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::print_diagnostics(System_of_eqs const& syst, const int ite,
                                                                         const double conv) const
{

    int ndom = space.get_nbr_domains();

    constexpr double M2Hz = 2.029739818539300e+05 / 2. / M_PI;
    const int adapted_outer = space.ADAPTED_OUTER;
    double baryonic_mass = 0.;
    for (int d = 0; d <= adapted_outer; ++d)
        baryonic_mass += syst.give_val_def_scalar_domain("intMb", d).integ_volume();
    double Madm = space.get_domain(ndom - 1)->integ(
        syst.give_val_def_scalar_domain("intMadm", ndom - 1), OUTER_BC);
    double Mk = space.get_domain(ndom - 1)->integ(
        syst.give_val_def_scalar_domain("intMk", ndom - 1), OUTER_BC);
    double J = space.get_domain(ndom - 1)->integ(
        syst.give_val_def_scalar_domain("intJ", ndom - 1), OUTER_BC);

    auto rs = bco_utils::get_rmin_rmax(space, adapted_outer);

    auto this_domain = Omg(adapted_outer).get_domain();
    Index pos(this_domain->get_nbr_points());
    pos.set(0) = this_domain->get_nbr_points()(0) - 1;
    pos.set(1) = this_domain->get_nbr_points()(1) - 1;

    // A logically-zero Val_domain keeps no configuration array, so sampling it would
    // dereference a null buffer. Its value is zero everywhere, so report that directly.
    auto sample_omega = [&pos](const Val_domain& omega) {
        return omega.check_if_zero() ? 0. : omega(pos);
    };
    double equator_omg = sample_omega(Omg(adapted_outer));
    double max_omg = sample_omega(Omg(0));

#define FORMAT std::setw(13) << std::left << std::showpos
    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << "=======================================" << std::endl
              << FORMAT << "Iter: " << ite << std::endl
              << FORMAT << "Error: " << conv << std::endl
              << FORMAT << "Mb: " << baryonic_mass << std::endl
              << FORMAT << "Madm_ql: " << Mk << " [" << std::abs(Madm - Mk) / Madm << "]" << std::endl
              << FORMAT << "R: " << rs[0] << " " << rs[1] << " [" << rs[0] / rs[1] << "]" << std::endl
              << FORMAT << "J: " << J << std::endl
              << FORMAT << "Chi: " << J / Madm / Madm << " [" << bconfig(CHI) << "]" << std::endl
              << FORMAT << "Omega: " << bconfig(OMEGA) << " [" << bconfig(OMEGA) * M2Hz << " Hz]" << std::endl
              << FORMAT << "Max/Equator: [" << max_omg << ", " << equator_omg << "]" << std::endl;
    std::cout.flags(f);
    std::cout << "=======================================" << "\n\n";
#undef FORMAT
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::update_config_quantities(const double& loghc)
{
    bconfig.set(HC) = std::exp(loghc);
    bconfig.set(NC) = EOS<eos_t, eos_var_t::DENSITY>::get(bconfig(HC));
}

template <class eos_t, typename config_t, typename space_t>
std::string ns_2d_xcts_diff_solver<eos_t, config_t, space_t>::converged_filename(const std::string& stage) const
{
    return converged_ns_diffrot_filename(bconfig, space, stage);
}

} // namespace Kadath
