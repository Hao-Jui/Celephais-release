/*
 * Copyright 2022
 * This file is part of the KADATH library and published under
 * https://arxiv.org/abs/2103.09911
 *
 * Author:
 * Samuel D. Tootle <tootle@itp.uni-frankfurt.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#include "mpi.h"
#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include "Apps/Workflow/solver_sequence.hpp"
#include "Apps/Workflow/resume_ladder.hpp"
#include "Apps/Workflow/isolated_driver_loop.hpp"
#include <sstream>
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "solver.hpp"
#include "regrid.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "Apps/Formalism/Shared/xcts_syst_init.ipp"

using namespace Kadath::Margherita;

namespace Kadath {

template <typename config_t, typename space_t>
bh_3d_xcts_solver<config_t, space_t>::bh_3d_xcts_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in,
                                                        Scalar& conf_in, Scalar& lapse_in, Vector& shift_in)
    : XCTS_Solver<config_t, space_t>(config_in, space_in, base_in), conf(conf_in), lapse(lapse_in), shift(shift_in),
      fmet(Metric_flat(space_in, base_in))
{
    // initialize only the vector fields we need
    coord_vectors = default_co_vector_ary(space);

    update_fields_co(cfields, coord_vectors, {}, 0.);
}

// standardized filename for each converged dataset at the end of each stage.
template <typename config_t, typename space_t>
std::string bh_3d_xcts_solver<config_t, space_t>::converged_filename(const std::string& stage) const
{
    return converged_gr_bh_filename(bconfig, space, stage);
}

template <typename config_t, typename space_t> int bh_3d_xcts_solver<config_t, space_t>::solve()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int exit_status = EXIT_SUCCESS;

    std::array<bool, NUM_STAGES_V> stage_enabled = bconfig.return_stages();
    auto [last_stage, last_stage_idx] = get_last_enabled(MSTAGE(), stage_enabled);

    const double current_chi = bconfig.template value_as<double>(CHI);
    const double final_chi = bconfig.has_seq_setting(FINAL_CHI)
                                 ? bconfig.template seq_setting_as<double>(FINAL_CHI)
                                 : current_chi;
    bool iterative_chi = std::abs(final_chi) > 0.5;

    if (stage_enabled[to_int(DIRICHLET_LAPSE)]) {
        exit_status = fixed_lapse_stage();
        if (status_requires_field_reload(exit_status)) {
            // The fixed-lapse stage was satisfied by resuming an existing
            // dataset (a different file than the one in memory), so the driver
            // must reload fields before continuing. Consume the stage so the
            // re-entry advances to VON_NEUMANN instead of resuming DIRICHLET
            // again -- otherwise, when both the DIRICHLET and VON_NEUMANN caches
            // exist, the two stages ping-pong the active dataset forever. This
            // mirrors the NS driver, which disables NOROT once it is consumed.
            bconfig.set_stage(DIRICHLET_LAPSE) = false;
            return exit_status;
        }
    }

    if (stage_enabled[to_int(VON_NEUMANN)]) {
        if (iterative_chi) {
            while (iterative_chi) {

#ifdef DEBUG
                if (rank == 0)
                    cout << bconfig << endl;
#endif

                exit_status = von_Neumann_stage();
                if (status_requires_field_reload(exit_status))
                    return exit_status;

                if ((std::abs(final_chi) <= 0.8) || (std::abs(bconfig(CHI)) >= 0.8)) {
                    iterative_chi = false;
                } else if (std::abs(bconfig(CHI)) < 0.8)
                    bconfig(CHI) = std::copysign(0.8, final_chi);
            }
            bconfig(CHI) = final_chi;
        }
        exit_status = von_Neumann_stage();
        if (status_requires_field_reload(exit_status))
            return exit_status;
    }

    // Barrier needed in case we need to read from the previous output
    MPI_Barrier(MPI_COMM_WORLD);
    if (last_stage_idx == to_int(BIN_BOOST))
        exit_status = legacy_status_from_stage_outcome(KadathApps::request_binary_boost());
    return exit_status;
}

template <typename config_t>
inline int bh_3d_xcts_driver(config_t& bconfig, std::string outputdir, kadath_config<BIN_INFO> binconfig,
                             NODES bco)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (std::abs(bconfig(CHI)) > 0.85) {
        std::cerr << "Unable to handle chi > 0.85\n";
        return EXIT_FAILURE;
    }
    if (!bconfig.has_seq_setting(INIT_RES))
        bconfig.seq_setting(INIT_RES) = Kadath::kDefaultInitialResolution;

    // Continuation ladder: solve at init_res, then step the resolution up to
    // final_res in +2 rungs (regrid + re-solve each rung), identical to the
    // NS / BNS stepping policy. Warm starts floor init at the loaded grid's
    // resolution so a continuation never restarts below (and regrids away) the
    // saved solution.
    const ResolutionLadder ladder = KadathApps::make_resume_ladder<Space_adapted_bh>(
        bconfig, bconfig.template seq_setting_as<int>(INIT_RES), bconfig(BCO_RES),
        /*warm_start=*/fs::exists(bconfig.space_filename()));
    const int final_res = ladder.final;
    int current_res = ladder.init;
    bool res_inc = (current_res < final_res);
    bconfig.set(BCO_RES) = current_res;

    bool iterative_chi = std::abs(bconfig(CHI)) > 0.5;

    bconfig.seq_setting(FINAL_CHI) = bconfig(CHI);
    if (iterative_chi)
        bconfig(CHI) = std::copysign(0.5, bconfig.template seq_setting_as<double>(FINAL_CHI));

    if (rank == 0)
        std::cout << "Solutions will be stored in: " << outputdir << "\n"
                  << "Directory will be created if it doesn't exist.\n";
    fs::create_directory(outputdir);

    if (!fs::exists(bconfig.space_filename())) {
        bconfig.set_filename("initbh");
        if (rank == 0) {
            Kadath::Seed::setup_co<BH, Space_adapted_bh>(bconfig);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        bconfig.open_config();
    }

    std::array<bool, NUM_STAGES_V> stage_enabled = bconfig.return_stages();
    auto [last_stage, last_stage_idx] = get_last_enabled(MSTAGE(), stage_enabled);

    std::string spacein = bconfig.space_filename();
    if (!fs::exists(spacein)) {
        // mainly for debugging MPI bugs
        if (rank == 0) {
            std::cerr << "File: " << spacein << " not found.\n\n";
        } else {
            std::cerr << "File: " << spacein << " not found for another rank.\n\n";
        }
        KADATH_THROW("Stage failed");
    }

    // BH continuation loop: binary boost (when requested by the last stage) is
    // handled INLINE in the same loop, so allow_binary_boost = true. The body
    // below is the per-rung step (unchanged from the hand-written loop).
    return KadathApps::run_isolated_driver_loop(
        bconfig, /*allow_binary_boost=*/true,
        [&](config_t& bconfig, int exit_status) -> int {
            std::string spacein = bconfig.space_filename();
            // just so you really know
            if (rank == 0) {
                std::cout << "Config File: " << bconfig.config_outputdir() + bconfig.config_filename() << std::endl
                          << "Fields File: " << spacein << std::endl
                          << bconfig << std::endl;
            }
            BeFileSource ff1(spacein);
            Space_adapted_bh space(ff1);
            Scalar conf(space, ff1);
            Scalar lapse(space, ff1);
            Vector shift(space, ff1);
            Base_tensor basis(space, CARTESIAN_BASIS);

            if (outputdir != "")
                bconfig.set_outputdir(outputdir);

            bh_3d_xcts_solver<decltype(bconfig), decltype(space)> bh_solver(bconfig, space, basis, conf, lapse, shift);
            exit_status = bh_solver.solve();
            if (status_requests_binary_boost(exit_status)) {
                exit_status = bh_solver.binary_boost_stage(binconfig, bco);
            } else if (res_inc && status_completed_without_reload(exit_status)) {
                const int next_res = KadathApps::next_resolution(current_res, final_res);

                if (rank == 0)
                    exit_status = bh_3d_xcts_regrid(bconfig, next_res, "initbh");
                current_res = next_res;
                res_inc = (current_res < final_res);
                bconfig.set_filename("initbh");
                iterative_chi = false;
                MPI_Barrier(MPI_COMM_WORLD);
                bconfig.open_config();
                bconfig.set(BCO_RES) = current_res;
                exit_status = legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
            }
            return exit_status;
        });
}

template <typename config_t, typename space_t> void bh_3d_xcts_solver<config_t, space_t>::syst_init(System_of_eqs& syst)
{

    const int ndom = space.get_nbr_domains();
    // Metric
    xcts::add_flat_metric(fmet, syst);

    // define numerical constants
    xcts::add_four_pi_g(syst);
    syst.add_cst("PI", M_PI);
    syst.add_cst("M", bconfig(MIRR));
    syst.add_cst("CM", bconfig(MCH));
    syst.add_cst("chi", bconfig(CHI));
    syst.add_var("ome", bconfig(OMEGA));

    // Global-rotation field + flat-space surface normals (shared trio: mg, sm, einf)
    xcts::add_global_rot_and_surface_coords(syst, coord_vectors);

    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);

    // the basic fields, conformal factor, lapse and (log) enthalpy
    xcts::add_conformal_lapse_vars(syst, conf, lapse);
    syst.add_var("bet", shift);

    // define common combinations of conformal factor and lapse
    syst.add_def("NP = P*N");
    syst.add_def("Ntilde = N / P^6");
    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / Ntilde");

    // definitions of integrals on the excision surface
    syst.add_def("intMsq= P^4 / 16. / PI");

    // define quantity to be integrated at infinity
    // two (in this case) equivalent definitions of ADM mass
    // as well as the Komar mass
    syst.add_def(ndom - 1, "intMadm = - einf^i * D_i P / 4piG * 2");
    syst.add_def(ndom - 1, "intMk = einf^i * D_i N / 4piG");
}

template <typename config_t, typename space_t>
void bh_3d_xcts_solver<config_t, space_t>::print_diagnostics_norot(const System_of_eqs& syst, const int ite,
                                                                   const double conv) const
{

    int ndom = space.get_nbr_domains();
    double r = bco_utils::get_radius(space.get_domain(1), OUTER_BC);

    Val_domain integMsq(syst.give_val_def("intMsq")()(2));
    double Mirrsq = space.get_domain(2)->integ(integMsq, INNER_BC);
    double Mirr = std::sqrt(Mirrsq);

    // compute the ADM mass as surface integral at infinity
    Val_domain integMadm(syst.give_val_def("intMadm")()(ndom - 1));
    double Madm = space.get_domain(ndom - 1)->integ(integMadm, OUTER_BC);

    // compute the Komar mass as surface integral at infinity
    Val_domain integMk(syst.give_val_def("intMk")()(ndom - 1));
    double Mk = space.get_domain(ndom - 1)->integ(integMk, OUTER_BC);

    // output to standard output
    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << "=======================================" << std::endl
              << FORMAT << "Iter: " << ite << std::endl
              << FORMAT << "Error: " << conv << std::endl
              << FORMAT << "Madm: " << Madm << std::endl
              << FORMAT << "Mk: " << Mk << " [" << std::abs(Madm - Mk) / Madm << "]" << std::endl
              << FORMAT << "Mirr: " << Mirr << std::endl;
    std::cout << FORMAT << "R: " << r << std::endl;
    std::cout.flags(f);
} // end print diagnostics norot

// runtime diagnostics specific for rotating solutions
template <typename config_t, typename space_t>
void bh_3d_xcts_solver<config_t, space_t>::print_diagnostics(System_of_eqs const& syst, const int ite,
                                                             const double conv) const
{

    // print all the diagnostics as in the non-rotating case first
    print_diagnostics_norot(syst, ite, conv);

    Val_domain integS(syst.give_val_def("intS")()(2));
    double S = space.get_domain(2)->integ(integS, INNER_BC);

    Val_domain integMsq(syst.give_val_def("intMsq")()(2));
    double Mirrsq = space.get_domain(2)->integ(integMsq, INNER_BC);
    double Mch = std::sqrt(Mirrsq + S * S / 4. / Mirrsq);

    // output the dimensionless spin and angular frequency parameter
    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << FORMAT << "Omega: " << bconfig(OMEGA) << std::endl
              << FORMAT << "S: " << S << std::endl
              << FORMAT << "Mch: " << Mch << std::endl
              << FORMAT << "Chi: " << S / Mch / Mch << std::endl;
    std::cout.flags(f);
    std::cout << "=======================================" << "\n\n";
} // end print diagnostics rot


} // namespace Kadath
