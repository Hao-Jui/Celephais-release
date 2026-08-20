// The formalism follows from Shibata's book (2015)
/**
 * Metric:
 * ds^2 = -α^2 dt^2
 *      + ψ^4 [ e^{2q}(dr^2 + r^2 dθ^2) + r^2 sin^2θ (β dt + dφ)^2 ]
 */

#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include "mpi.h"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "For_Kadath/Array/memory.hpp"
#include "Apps/Helper/newton_loop_runner.hpp"
#include <cmath>
#include <cstdlib>

using namespace Kadath::Margherita;

namespace Kadath {


template <class eos_t, typename config_t, typename space_t>
ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::ns_2d_msqi_diff_solver(config_t& config_in, space_t& space_in,
                                                                         Scalar& conf_in, Scalar& lapse_in,
                                                                         Scalar& logh_in, Scalar& shift_in,
                                                                         Scalar& metQ_in, Scalar& Omg_in)
    : Solver<config_t, space_t>(config_in, space_in), conf(conf_in), lapse(lapse_in), logh(logh_in), shift(shift_in),
      metQ(metQ_in), Omg(Omg_in)
{
    shift.affect_parameters();
    shift.set_parameters()->set_m_quant() = 1;
    shift.std_base();
}

template <class eos_t, typename config_t, typename space_t>
int ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::maybe_skip_stage(int rank, const char* stage_name)
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
void ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::init_stage_fields(double& loghc)
{
    loghc = std::log(bconfig(HC));
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::syst_init(System_of_eqs& syst)
{
    using namespace Kadath::Margherita;


    auto& space = syst.get_space();
    [[maybe_unused]] const int ndom = space.get_nbr_domains();

    // define numerical constants
    Scalar oneField(space);
    oneField = 1.;
    oneField.std_base();
    syst.add_cst("ones", oneField);
    syst.add_cst("4piG", 4.0 * M_PI);

    // the basic fields, conformal factor, lapse and (log) enthalpy
    syst.add_var("P", conf);
    syst.add_var("N", lapse);
    syst.add_var("H", logh);
    syst.add_var("Omg", Omg);
    syst.add_var("brsint", shift);
    syst.add_var("Q", metQ);

    // define the EOS operators
    Param p;
    syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
    syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
    syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    syst.add_ope("dHdlnrho", &EOS<eos_t, eos_var_t::DHDRHO>::action, &p);

    syst.add_cst("chi", bconfig(CHI));
    syst.add_var("ome", bconfig(OMEGA));

    if (bconfig.control(MB_FIXING)) {
        syst.add_cst("Mb", bconfig(MB));
        syst.add_var("Madm", bconfig(MADM));
    } else {
        syst.add_var("Mb", bconfig(MB));
        syst.add_cst("Madm", bconfig(MADM));
    }
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::add_common_defs(System_of_eqs& syst)
{
    // define common combinations of conformal factor and lapse
    syst.add_def("NP = P*N");
    syst.add_def("bet = divrsint(brsint)");
    syst.add_def("Asquare = multrsint(multrsint(scal(grad(bet), grad(bet)))) / ( 4 * N^2 )");
    syst.add_def("NPP = NP*P");

    // define quantity to be integrated at infinity
    // two (in this case) equivalent definitions of ADM mass
    // as well as the Komar mass
    syst.add_def(ndom - 1, "intMadm = -dr(P) * 2 / 4piG");
    syst.add_def(ndom - 1, "intMk = dr(N)  / 4piG");
    syst.add_def(ndom - 1, "intJ = multrsint(multrsint(dr(bet))) / 4 / 4piG");

    // enthalpy from the logarithmic enthalpy, the latter is the actual variable in this system
    syst.add_def("h = exp(H)");

    // define rest-mass density, internal energy and pressure through the enthalpy
    syst.add_def("rho = rho(h)");
    syst.add_def("eps = eps(h)");
    syst.add_def("press = press(h)");
    syst.add_def("dHdlnrho = dHdlnrho(h)");

    // definition to rescale the equations
    // delta = p / rho
    syst.add_def("delta = h - eps - 1.");
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::add_domain_equations(System_of_eqs& syst, bool include_intf)
{
    const std::string firstint_expr = include_intf ? "H + log(N) - log(W) + intF" : "H + log(N) - log(W)";

    const int adapted_inner = space.ADAPTED_INNER;
    for (int d = 0; d < ndom; d++) {
        if (d < adapted_inner) {
            syst.add_def(d, "U = multrsint(P^2 / N * (Omg + bet))"); // amplitude of spatial velocity
            syst.add_def(d, "Usq = U*U");
            syst.add_def(d, "Wsquare = 1 / (1 - Usq)");
            syst.add_def(d, "W = sqrt(Wsquare)");

            // sources rescaled by P/rho as in Papenfort2021
            syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
            syst.add_def(d, "Spp = Etilde - press * h + 2 * press * delta");
            syst.add_def(d, "Stilde = 2 * press * delta + Spp");
            syst.add_def(d, "Jphi = P^2 * (Etilde + press * delta) * U"); // J_φ / (r*sin(θ))

            syst.add_def(d, "Sq = - 2 * 4piG * P^4 * exp(2 * Q) * ( Stilde - 2 * Spp )"
                            "+ 3 * Asquare * P^4 * delta + 2 * ( lap(NPP) - lap2(NPP) ) / NPP * delta"
                            "+ 4 * scal(grad(NP), grad(P)) / NPP * delta");
            syst.add_def(d, "Rtilde = - 2 * exp(- 2 * Q) * Sq");

            syst.add_def(d, "eqP    = delta * lap(P)  -  P * exp(2 * Q) * Rtilde / 8 "
                            "+ 4piG / 2 * P^5 * Etilde * exp(2 * Q) + Asquare * delta * P^5 / 4");
            syst.add_def(d, "eqNP   = delta * lap(NP) - NP * exp(2 * Q) * Rtilde / 8 "
                            "- 4piG / 2 * N * P^5 * (Etilde + 2 * Stilde) * exp(2 * Q) "
                            "- 7 * Asquare * N * P^5 / 4 * delta");
            syst.add_def(d, "eqbrsint  = lap(brsint) * delta - 4 * 4piG * N * exp(2 * Q) * Jphi"
                            "- delta * multrsint(scal(grad(bet), grad( log(N) - 6 * log(P) ) ))");
            syst.add_def(d, "eqQ    = lap2(Q) * delta - Sq");
            // QB = W * rho(h) * exp(2 * Q) * P^6
            syst.add_def(d, "intMb = W * rho * exp(2 * Q) * P^6 * 4piG / 2");
            const std::string firstint_def = std::string("firstint = ") + firstint_expr;
            syst.add_def(d, firstint_def.c_str());
            syst.add_eq_full(d, "rotlaw = 0");
        } else {
            syst.add_eq_full(d, "H = 0");
            syst.add_def(d, "Sq = 3 * Asquare * P^4 + 2 * ( lap(NPP) - lap2(NPP) ) / NPP"
                            "+ 4 * scal(grad(NP), grad(P)) / NPP");
            syst.add_def(d, "Ricci = - 2 * exp(- 2 * Q) * Sq");

            syst.add_def(d, "eqP   = lap(P)  -  P * exp(2 * Q) * Ricci / 8 + Asquare * P^5 / 4");
            syst.add_def(d, "eqNP  = lap(NP) - NP * exp(2 * Q) * Ricci / 8 - 7 * Asquare * N * P^5 / 4");
            syst.add_def(d, "eqbrsint = lap(brsint) - multrsint(scal(grad(bet), grad( log(N) - 6 * log(P) ) ))");
            syst.add_def(d, "eqQ   = lap2(Q) - Sq");
        }
    }
    space.add_eq(syst, "eqNP= 0", "N", "dn(N)");
    space.add_eq(syst, "eqP= 0", "P", "dn(P)");
    space.add_eq(syst, "eqbrsint= 0", "brsint", "dn(brsint)");
    space.add_eq(syst, "eqQ= 0", "Q", "dn(Q)");
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::add_omega_constraints(System_of_eqs& syst)
{
    const int adapted_inner = space.ADAPTED_INNER;
    syst.add_eq_matching(adapted_inner, INNER_BC, "Omg");
    syst.add_eq_matching(adapted_inner, INNER_BC, "dn(Omg)");
    syst.add_eq_inside(adapted_inner, "Omg = 0");
    for (int d = adapted_inner + 1; d < ndom; ++d)
        syst.add_eq_full(d, "Omg = 0");
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::add_boundary_conditions(System_of_eqs& syst)
{

    const int adapted_outer = space.ADAPTED_OUTER;
    syst.add_eq_bc(ndom - 1, OUTER_BC, "N=1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "P=1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "brsint=0");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "Q=0");

    syst.add_eq_bc(adapted_outer, OUTER_BC, "H = 0");
    syst.add_eq_first_integral(0, adapted_outer, "firstint", "H - Hc");

    space.add_eq_int_volume(syst, adapted_outer + 1, "integvolume(intMb) = Mb");
    space.add_eq_int_inf(syst, "integ(intJ) - chi * Madm * Madm = 0");
    space.add_eq_int_inf(syst, "integ(intMadm) = Madm");
}

template <class eos_t, typename config_t, typename space_t>
int ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::solve()
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
        if (law == "uniform") {
            exit_status = uniform_stage();
        } else if (law == "keh") {
            exit_status = keh_stage();
        } else {
            KADATH_THROW("The rotational law has not been implemented!");
        }
        if (status_requires_field_reload(exit_status)) {
            return exit_status;
        }
    }
    // Barrier needed in case we need to read from the previous output.
    MPI_Barrier(MPI_COMM_WORLD);
    return exit_status;
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::run_newton_loop(System_of_eqs& syst, double& loghc, int rank)
{
    bool endloop = false;
    int ite = 1;
    double conv = 0.;
    const SolverRuntimeConfig solver_config = this->with_stage_mumps_tree_cache(
        SolverRuntimeConfig::from_environment(), "UNIROT");
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
                bco_utils::save_to_file(space, bconfig, conf, lapse, shift, metQ, logh, Omg);
        }
        ite++;
        check_max_iter_exceeded(rank, ite, conv);
    }
    bconfig.set_filename(converged_filename("UNIROT"));
    if (rank == 0)
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift, metQ, logh, Omg);
}

template <class eos_t, typename config_t, typename space_t>
void ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::print_diagnostics(System_of_eqs const& syst, const int ite,
                                                                         const double conv) const
{

    int ndom = space.get_nbr_domains();

    Val_domain integMadm(syst.give_val_def("intMadm")()(ndom - 1));
    Val_domain integMk(syst.give_val_def("intMk")()(ndom - 1));
    Val_domain integJ(syst.give_val_def("intJ")()(ndom - 1));

    constexpr double M2Hz = 2.029739818539300e+05 / 2. / M_PI;
    const int adapted_outer = space.ADAPTED_OUTER;
    double baryonic_mass = 0.;
    for (int d = 0; d <= adapted_outer; ++d)
        baryonic_mass += syst.give_val_def("intMb")()(d).integ_volume();
    double Madm = space.get_domain(ndom - 1)->integ(integMadm, OUTER_BC);
    double Mk = space.get_domain(ndom - 1)->integ(integMk, OUTER_BC);
    double J = space.get_domain(ndom - 1)->integ(integJ, OUTER_BC);

    auto rs = bco_utils::get_rmin_rmax(space, adapted_outer);

    auto this_domain = Omg(adapted_outer).get_domain();
    Index pos(this_domain->get_nbr_points());
    pos.set(0) = this_domain->get_nbr_points()(0) - 1;
    pos.set(1) = this_domain->get_nbr_points()(1) - 1;

    double equator_omg = Omg(adapted_outer)(pos);
    double max_omg = Omg(0)(pos);

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
void ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::update_config_quantities(const double& loghc)
{
    bconfig.set(HC) = std::exp(loghc);
    bconfig.set(NC) = EOS<eos_t, eos_var_t::DENSITY>::get(bconfig(HC));
}

template <class eos_t, typename config_t, typename space_t>
std::string ns_2d_msqi_diff_solver<eos_t, config_t, space_t>::converged_filename(const std::string& stage) const
{
    return converged_ns_diffrot_filename(bconfig, space, stage);
}

} // namespace Kadath
