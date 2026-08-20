#include "mpi.h"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"
#include "Apps/Formalism/Shared/PreBinary/bco_binary_setup.hpp"
#include "Apps/Formalism/Shared/xcts_syst_init.ipp"
#include "Apps/Formalism/GR/BH_3D_XCTS/solver.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include <string>
#include <iostream>
#include <utility>

namespace Kadath {

template <typename config_t, typename space_t>
bbh_xcts_solver<config_t, space_t>::bbh_xcts_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in,
                                                    Scalar& conf_in, Scalar& lapse_in, Vector& shift_in)
    : XCTS_Solver<config_t, space_t>(config_in, space_in, base_in), conf(conf_in), lapse(lapse_in), shift(shift_in),
      fmet(Metric_flat(space_in, base_in)), xc1(bco_utils::get_center(space_in, space.BH1)),
      xc2(bco_utils::get_center(space_in, space.BH2)),
      excluded_doms({space.BH1, space.BH1 + 1, space.BH2, space.BH2 + 1})

{
    // initialize coordinate vector fields
    coord_vectors = default_binary_vector_ary(space);

    update_fields(cfields, coord_vectors, {}, xo, xc1, xc2);
}

// standardized filename for each converged dataset at the end of each stage.
template <typename config_t, typename space_t>
std::string bbh_xcts_solver<config_t, space_t>::converged_filename(const std::string& stage) const
{
    return converged_gr_bbh_filename(bconfig, space, stage);
}

template <typename config_t, typename space_t> int bbh_xcts_solver<config_t, space_t>::solve(bool want_warmup)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exit_status = EXIT_SUCCESS;

    std::array<bool, NUM_STAGES_V> stage_enabled = bconfig.return_stages();
    auto [last_stage, last_stage_idx] = get_last_enabled(MSTAGE(), stage_enabled);

    // check if we have solved this before til the last stage
    const auto resume = this->load_existing_solution(last_stage);
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Solved previously: " << bconfig.config_filename() << std::endl;
        return EXIT_SUCCESS;
    }

    if (stage_enabled[to_int(QUASI_EQUIL)]) {
        // Optional fixed-Omega warm-up (one-shot, caller-owned) then the free-Omega
        // pass; both are the same QUASI_EQUIL stage distinguished by OmegaMode. When
        // no warm-up is requested only the free pass runs; if the warm-up pass fails
        // the free pass is skipped. The config flag is never mutated.
        if (want_warmup)
            exit_status = solve_stage(QUASI_EQUIL, OmegaMode::Fixed, "QUASI_EQUIL");
        if (exit_status == EXIT_SUCCESS)
            exit_status = solve_stage(QUASI_EQUIL, OmegaMode::Free, "QUASI_EQUIL");
    }

    if (stage_enabled[to_int(ECC_RED)]) {
        // ECC_RED is inherently fixed-Omega (constant ome, no quasi-equilibrium
        // constraint), independent of the warm-up decision.
        exit_status = solve_stage(ECC_RED, OmegaMode::Fixed, "ECC_RED");
    }

    // Barrier needed in case we need to read from the previous output
    MPI_Barrier(MPI_COMM_WORLD);

    return exit_status;
}

template <typename config_t, typename space_t> void bbh_xcts_solver<config_t, space_t>::syst_init(System_of_eqs& syst)
{

    const int ndom = space.get_nbr_domains();
    // call the (flat) conformal metric "f"
    xcts::add_flat_metric(fmet, syst);

    // define numerical constants
    xcts::add_four_pi_g(syst);
    syst.add_cst("PI", M_PI);

    // BH1 fixing parameters
    syst.add_cst("Mm", bconfig(MIRR, BCO1));
    syst.add_cst("chim", bconfig(CHI, BCO1));
    syst.add_cst("CMm", bconfig(MCH, BCO1));

    // BH2 fixing parameters
    syst.add_cst("Mp", bconfig(MIRR, BCO2));
    syst.add_cst("chip", bconfig(CHI, BCO2));
    syst.add_cst("CMp", bconfig(MCH, BCO2));

    // constant vector fields
    // Global-rotation field + flat-space surface normals "sm"/"einf" (shared trio);
    // the per-object rotation vectors "mm"/"mp" and the BCO2 normal "sp" follow.
    xcts::add_global_rot_and_surface_coords(syst, coord_vectors);

    syst.add_cst("mm", *coord_vectors[BCO1_ROT]);
    syst.add_cst("mp", *coord_vectors[BCO2_ROT]);
    syst.add_cst("sp", *coord_vectors[to_int(coord_vector::S_BCO2)]);

    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);
    syst.add_cst("ez", *coord_vectors[to_int(coord_vector::EZ)]);

    // the basic fields, conformal factor, lapse and (log) enthalpy
    xcts::add_conformal_lapse_vars(syst, conf, lapse);
    syst.add_var("bet", shift);

    // setup parameters accordingly for a corotating binary
    if (bconfig.control(COROT_BIN)) {
        bconfig.set(OMEGA, BCO1) = 0.;
        bconfig.set(OMEGA, BCO2) = 0.;
    } else {
        // otherwise we setup the system for arbitrarily rotating BHs
        // based on either fixed rotation frequency or variable
        // rotation frequency fixed by a dimensionless
        // spin parameter CHI.
        if (!std::isnan(bconfig.set(FIXED_BCOMEGA, BCO1))) {
            syst.add_cst("omesm", bconfig(FIXED_BCOMEGA, BCO1));
            bconfig.set(OMEGA, BCO1) = bconfig(FIXED_BCOMEGA, BCO1);
        } else
            syst.add_var("omesm", bconfig(OMEGA, BCO1));

        syst.add_def(space.BH1 + 2, "ExOme^i = omesm * mm^i");

        if (!std::isnan(bconfig.set(FIXED_BCOMEGA, BCO2))) {
            syst.add_cst("omesp", bconfig(FIXED_BCOMEGA, BCO2));
            bconfig.set(OMEGA, BCO2) = bconfig(FIXED_BCOMEGA, BCO2);
        } else
            syst.add_var("omesp", bconfig(OMEGA, BCO2));
        syst.add_def(space.BH2 + 2, "ExOme^i = omesp * mp^i");
    }

    // define common combinations of conformal factor and lapse
    syst.add_def("NP = P*N");
    syst.add_def("Ntilde = N / P^6");

    // definition describing the binary rotation
    syst.add_def("Morb^i  = mg^i + xaxis * ey^i + yaxis * ex^i");

    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / Ntilde");

    // ADM linear momenta
    syst.add_def("intPx   = A_ij * ex^j * einf^i / 8 / PI");
    syst.add_def("intPy   = A_ij * ey^j * einf^i / 8 / PI");
    syst.add_def("intPz   = A_ij * ez^j * einf^i / 8 / PI");

    // spin definition on the apparent horizons
    syst.add_def("intSm   = A_ij * mm^i * sm^j   / 8 / PI");
    syst.add_def("intSp   = A_ij * mp^i * sp^j   / 8 / PI");

    // diagnostic from https://arxiv.org/abs/1506.01689
    syst.add_def(ndom - 1, "COMx  = -3 * P^4 * einf^i * ex_i / 8. / PI");
    syst.add_def(ndom - 1, "COMy  = 3 * P^4 * einf^i * ey_i / 8. / PI");
    syst.add_def(ndom - 1, "COMz  = 3 * P^4 * einf^i * ez_i / 8. / PI");

    // BH irreducible mass
    syst.add_def("intMsq  = P^4 / 16. / PI");

    // define quantity to be integrated at infinity
    // two (in this case) equivalent definitions of ADM mass
    // as well as the Komar mass
    syst.add_def(ndom - 1, "intMadm = - einf^i * D_i P / 4piG * 2");
    syst.add_def(ndom - 1, "intMk = einf^i * D_i N / 4piG");
}

template <typename config_t, typename space_t>
void bbh_xcts_solver<config_t, space_t>::print_diagnostics(const System_of_eqs& syst, const int ite,
                                                           const double conv) const
{

    std::ios_base::fmtflags f(std::cout.flags());
    int ndom = space.get_nbr_domains();

    // local spin angular momentum BH1
    Val_domain integSm(syst.give_val_def("intSm")()(space.BH1 + 2));
    double Sm = space.get_domain(space.BH1 + 2)->integ(integSm, INNER_BC);

    // local spin angular momentum BH2
    Val_domain integSp(syst.give_val_def("intSp")()(space.BH2 + 2));
    double Sp = space.get_domain(space.BH2 + 2)->integ(integSp, INNER_BC);

    // irreducible mass BH1
    Val_domain integMmsq(syst.give_val_def("intMsq")()(space.BH1 + 2));
    double Mirrmsq = space.get_domain(space.BH1 + 2)->integ(integMmsq, INNER_BC);
    double Mirrm = std::sqrt(Mirrmsq);
    // MCH
    double Mchm = std::sqrt(Mirrmsq + Sm * Sm / 4. / Mirrmsq);

    // irreducible mass BH2
    Val_domain integMpsq(syst.give_val_def("intMsq")()(space.BH2 + 2));
    double Mirrpsq = space.get_domain(space.BH2 + 2)->integ(integMpsq, INNER_BC);
    double Mirrp = std::sqrt(Mirrpsq);
    // MCH
    double Mchp = std::sqrt(Mirrpsq + Sp * Sp / 4. / Mirrpsq);

    // ADM Linear Momenta
    Val_domain integPx(syst.give_val_def("intPx")()(ndom - 1));
    double Px = space.get_domain(ndom - 1)->integ(integPx, OUTER_BC);

    Val_domain integPy(syst.give_val_def("intPy")()(ndom - 1));
    double Py = space.get_domain(ndom - 1)->integ(integPy, OUTER_BC);

    Val_domain integPz(syst.give_val_def("intPz")()(ndom - 1));
    double Pz = space.get_domain(ndom - 1)->integ(integPz, OUTER_BC);

    cout << "=======================================" << endl;
    std::cout << FORMAT << "Iter: " << ite << std::endl << FORMAT << "Error: " << conv << std::endl;
    std::cout << FORMAT << "Omega: " << bconfig(GOMEGA) << std::endl;
    if (!std::isnan(bconfig.set(ADOT)))
        std::cout << FORMAT << "Adot: " << bconfig(ADOT) << std::endl;
    std::cout << FORMAT << "Axis: " << bconfig(COM) << std::endl;
    std::cout << FORMAT << "P: " << "[" << Px << ", " << Py << ", " << Pz << "]\n\n";

    std::cout << FORMAT << "BH1-Mirr: " << Mirrm << std::endl;
    std::cout << FORMAT << "BH1-Mch: " << Mchm << std::endl;
    std::cout << FORMAT << "BH1-R: " << bco_utils::get_radius(space.get_domain(space.BH1 + 1), EQUI) << std::endl;
    std::cout << FORMAT << "BH1-S: " << Sm << std::endl;
    std::cout << FORMAT << "BH1-Chi: " << Sm / Mchm / Mchm << std::endl;
    std::cout << FORMAT << "BH1-Omega: " << bconfig(OMEGA, BCO1) << "\n\n";

    std::cout << FORMAT << "BH2-Mirr: " << Mirrp << std::endl;
    std::cout << FORMAT << "BH2-Mch: " << Mchp << std::endl;
    std::cout << FORMAT << "BH2-R: " << bco_utils::get_radius(space.get_domain(space.BH2 + 1), EQUI) << std::endl;
    std::cout << FORMAT << "BH2-S: " << Sp << std::endl;
    std::cout << FORMAT << "BH2-Chi: " << Sp / Mchp / Mchp << std::endl;
    std::cout << FORMAT << "BH2-Omega: " << bconfig(OMEGA, BCO2) << "\n\n";
    std::cout.flags(f);

    std::cout << "=======================================" << std::endl;
}

inline void bbh_xcts_setup_boosted_3d(kadath_config<BCO_BH_INFO>& BH1config,
                                      kadath_config<BCO_BH_INFO>& BH2config,
                                      kadath_config<BIN_INFO>& bconfig)
{

    std::string in_spacefile = BH1config.space_filename();
    BeFileSource ff1(in_spacefile);
    Space_adapted_bh spacein1(ff1);
    Scalar confin1(spacein1, ff1);
    Scalar lapsein1(spacein1, ff1);
    Vector shiftin1(spacein1, ff1);
    bco_utils::update_config_BH_radii(spacein1, BH1config, 1, confin1);

    in_spacefile = BH2config.space_filename();
    BeFileSource ff2(in_spacefile);
    Space_adapted_bh spacein2(ff2);
    Scalar confin2(spacein2, ff2);
    Scalar lapsein2(spacein2, ff2);
    Vector shiftin2(spacein2, ff2);
    bco_utils::update_config_BH_radii(spacein2, BH2config, 1, confin2);
    // end opening old solutions

    // update binary parameters - but save shell input
    const int nshells1 = bconfig(NSHELLS, BCO1);
    const int nshells2 = bconfig(NSHELLS, BCO2);
    for (int i = 0; i < NUM_BCO_PARAMS_V; ++i) {
        bconfig.set(i, BCO1) = BH1config.set(i);
        bconfig.set(i, BCO2) = BH2config.set(i);
    }
    bconfig.set(NSHELLS, BCO1) = nshells1;
    bconfig.set(NSHELLS, BCO2) = nshells2;

    const double r_max_tot = (bconfig(RMID, BCO1) > bconfig(RMID, BCO2)) ? bconfig(RMID, BCO1) : bconfig(RMID, BCO2);

    const double rout_sep_est = (bconfig(DIST) / 2. - r_max_tot) / 3. + r_max_tot;
    const double rout_max_est = 2. * bco_utils::gold_ratio * r_max_tot;

    bconfig.set(ROUT, BCO1) = (rout_sep_est > rout_max_est) ? rout_max_est : rout_sep_est;
    bconfig.set(ROUT, BCO2) = bconfig(ROUT, BCO1);

    std::string ifilename{"./initbin.toml"};
    bconfig.set_filename(ifilename);

    std::vector<double> out_bounds(1 + bconfig(OUTER_SHELLS));
    std::vector<double> BH1_bounds(3 + bconfig(NSHELLS, BCO1));
    std::vector<double> BH2_bounds(3 + bconfig(NSHELLS, BCO2));

    // for out_bounds.size > 1 - add equi-distant shells
    for (int e = 0; e < out_bounds.size(); ++e)
        out_bounds[e] = bconfig(REXT) + e * 0.25 * bconfig(REXT);

    bco_utils::set_BH_bounds(BH1_bounds, bconfig, BCO1, true);
    bco_utils::set_BH_bounds(BH2_bounds, bconfig, BCO2, false);
#ifdef DEBUG
    bco_utils::print_bounds("bh1", BH1_bounds);
    bco_utils::print_bounds("bh2", BH2_bounds);
#endif
    // create space containing the domain decomposition
    int type_coloc = CHEB_TYPE;
    Space_bin_bh space(type_coloc, bconfig(DIST), BH1_bounds, BH2_bounds, out_bounds, bconfig(BIN_RES));
    Base_tensor basis(space, CARTESIAN_BASIS);

    auto interp_BH_fields = [&](auto& bhconf, auto& bhlapse, auto& bhshift, auto& bhspacein) {
        const Domain_shell_outer_homothetic* old_bh_outer =
            dynamic_cast<const Domain_shell_outer_homothetic*>(bhspacein.get_domain(1));
        // Update BH fields based to help with interpolation later
        bco_utils::update_adapted_field(bhconf, 2, 1, old_bh_outer, OUTER_BC);
        bco_utils::update_adapted_field(bhlapse, 2, 1, old_bh_outer, OUTER_BC);
        for (int i = 1; i <= 3; ++i)
            bco_utils::update_adapted_field(bhshift.set(i), 2, 1, old_bh_outer, OUTER_BC);
    };

    interp_BH_fields(confin1, lapsein1, shiftin1, spacein1);
    interp_BH_fields(confin2, lapsein2, shiftin2, spacein2);

    double xc1 = bco_utils::get_center(space, space.BH1);
    double xc2 = bco_utils::get_center(space, space.BH2);

    Scalar conf(space);
    conf.annule_hard();

    Scalar lapse(space);
    lapse.annule_hard();

    Vector shift(space, CON, basis);
    for (int i = 1; i <= 3; i++)
        shift.set(i).annule_hard();

    const double bh1_invw4 = bco_utils::set_decay(bconfig, BCO1);
    const double bh2_invw4 = bco_utils::set_decay(bconfig, BCO2);

    // start importing the fields from the single star
    int ndom = space.get_nbr_domains();
    for (int dom = 0; dom < ndom; dom++) {
        // get an index in each domain to iterate over all colocation points
        Index new_pos(space.get_domain(dom)->get_nbr_points());

        do {
            if (dom <= space.BH1 + 1 || dom == space.BH2 || dom == space.BH2 + 1)
                continue;
            // get cartesian coordinates of the current colocation point
            double x = space.get_domain(dom)->get_cart(1)(new_pos);
            double y = space.get_domain(dom)->get_cart(2)(new_pos);
            double z = space.get_domain(dom)->get_cart(3)(new_pos);

            // define a point shifted suitably to the stellar centers in the binary
            Point absol1(3);
            absol1.set(1) = (x - xc1);
            absol1.set(2) = y;
            absol1.set(3) = z;
            double r2 = y * y + z * z;
            double r2_1 = (x - xc1) * (x - xc1) + r2;
            double r4_1 = r2_1 * r2_1;
            double r4_invw4_1 = r4_1 * bh1_invw4;
            double decay_1 = std::exp(-r4_invw4_1);

            Point absol2(3);
            absol2.set(1) = (x - xc2);
            absol2.set(2) = y;
            absol2.set(3) = z;
            double r2_2 = (x - xc2) * (x - xc2) + r2;
            double r4_2 = r2_2 * r2_2;
            double r4_invw4_2 = r4_2 * bh2_invw4;
            double decay_2 = std::exp(-r4_invw4_2);

            if (dom < ndom - 1) {
                conf.set_domain(dom).set(new_pos) =
                    1. + decay_1 * (confin1.val_point(absol1) - 1.) + decay_2 * (confin2.val_point(absol2) - 1.);
                lapse.set_domain(dom).set(new_pos) =
                    1. + decay_1 * (lapsein1.val_point(absol1) - 1.) + decay_2 * (lapsein2.val_point(absol2) - 1.);
                for (int i = 1; i <= 3; i++)
                    shift.set(i).set_domain(dom).set(new_pos) =
                        decay_1 * shiftin1(i).val_point(absol1) + decay_2 * shiftin2(i).val_point(absol2);

            } else {
                // We have to set the compactified domain manually since the outer collocation point is always
                // at inf which is undefined numerically
                conf.set_domain(dom).set(new_pos) = 1.;
                lapse.set_domain(dom).set(new_pos) = 1.;
            }
            // loop over all colocation points
        } while (new_pos.inc());
    } // end importing fields

    // safety to ensure excision region is empty
    auto clear_excision_region = [&](int nuc) {
        for (int d = nuc; d < nuc + 2; ++d) {
            conf.set_domain(d).annule_hard();
            lapse.set_domain(d).annule_hard();
            for (int i = 1; i <= 3; i++)
                shift.set(i).set_domain(d).annule_hard();
        }
    };
    clear_excision_region(space.BH1);
    clear_excision_region(space.BH2);

    conf.std_base();
    lapse.std_base();
    shift.std_base();

    bco_utils::save_to_file(space, bconfig, conf, lapse, shift);
}


} // namespace Kadath
