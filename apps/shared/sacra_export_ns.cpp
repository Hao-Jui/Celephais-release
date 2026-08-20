
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mpi.h"

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "For_Kadath/Utilities/exporter_utilities.hpp"
#include "Apps/Diagnostics/configured_eos.hpp"
#include "Apps/Diagnostics/sacra_export_utils.hpp"
#include "Apps/Diagnostics/sacra_format.hpp"
#include "Apps/Diagnostics/sacra_hydro_conversion.hpp"
#include "Hydro/EOS.hh"

namespace fs = std::filesystem;
using namespace Kadath;
using namespace export_utils;
using sacra_export::compute_level_range;
using sacra_export::write_concat_level_files;
using sacra_fmt::eWpDe3;
using sacra_fmt::iW;

int main(int argc, char** argv) {

    int rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        std::cerr << "Error starting MPI\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    int rank = 0, nb_procs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);

    std::string const usage =
        "Usage: ./2sacra input.toml outdir "
        "[--depth-max D] [--flvmax F] [--N N]\n"
        "Example: ./2sacra input.toml outdir --depth-max 9 --flvmax 5 --N 80\n"
        "Defaults: --depth-max 6 --flvmax 6 --N 80; jd_org is set to -N.";
    if (argc < 3) {
        if (rank == 0)
            std::cerr << "Missing required arguments.\n" << usage << "\n";
        MPI_Finalize();
        return 1;
    }

    sacra_export::GridOptions grid_options{6, 6, 80};
    try {
        grid_options = sacra_export::parse_grid_options(argc, argv, 3, grid_options, usage);
    } catch (const std::exception& e) {
        if (rank == 0)
            std::cerr << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    std::string input_file_path = argv[1];
    kadath_config<BCO_NS_INFO> bconfig(input_file_path);
    std::string kadath_filename = bconfig.space_filename();

    if (!fs::exists(kadath_filename)) {
        std::ostringstream oss;
        oss << "File: " << kadath_filename << " not found.\n\n";
        KADATH_THROW(oss.str());
    }

    // Read Kadath fields
    BeFileSource fin(kadath_filename);
    Space_spheric_adapted space(fin);
    Scalar conf(space, fin);
    Scalar lapse(space, fin);
    Vector shift_con(space, fin);
    Scalar logh(space, fin);

    // Modern NS checkpoints carry {conf, lapse, shift, logh}, then phi for
    // boosted datasets (gated on the file's PHI flag; discarded — the export
    // velocity is the rigid rotation below), then the formalism scalar slot.
    // The rigid rotation rate is config data, not a field; synthesize it like
    // the reader does. Legacy FUKa-v2 NS datasets with an on-disk Omg field
    // are not supported.
    if (bconfig.field(PHI) == true) {
        Scalar discarded_phi(space, fin);
        (void)discarded_phi;
    }
#if defined(KADATH_SACRA_EXPORT_DEF)
    Scalar xiscal(space, fin);
    Scalar varscal(xiscal);
    if (bconfig.field(SWEIGHT) == true) {
        Scalar Sweight(space, fin);
        for (int d = 0; d < space.get_nbr_domains(); ++d) {
            Index pos(space.get_domain(d)->get_nbr_points());
            do {
                varscal.set_domain(d).set(pos) = xiscal(d)(pos) * Sweight(d)(pos);
            } while (pos.inc());
        }
        varscal.std_base();
    }
#elif defined(KADATH_SACRA_EXPORT_SGB)
    Scalar varscal(space, fin);
#endif
    Scalar Omg(space);
    Omg = bconfig(OMEGA);
    Omg.std_base();

    if (shift_con.get_valence() != 1 || shift_con.get_index_type() != CON)
        KADATH_THROW("2sacra: shift field is not a contravariant rank-1 tensor");

    int ndoms = space.get_nbr_domains();

    Base_tensor basis(shift_con.get_basis());
    Metric_flat fmet(space, basis);

    System_of_eqs syst(space, 0, ndoms - 1);
    fmet.set_system(syst, "f");

    // Derived fields
    syst.add_cst("P", conf);
    syst.add_def("logP = log(P)");
    Scalar log_conf(syst.give_val_def("logP"));

    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift_con);
    // \tilde{A}_ij = \Psi^{-4} A = \Psi^{-6} \hat{A} used by SACRA
    syst.add_def("AKevolScalCov_ij = (D_i bet_j + D_j bet_i - 2. / 3.* D^k bet_k * f_ij) /2. / N");
    Tensor AK_evolScalCov(syst.give_val_def("AKevolScalCov"));

    if (AK_evolScalCov.get_valence() != 2 || AK_evolScalCov.get_index_type(0) != COV ||
        AK_evolScalCov.get_index_type(1) != COV)
        KADATH_THROW("2sacra: AKevolScalCov is not a covariant rank-2 tensor");

    // Coordinate fields
    CoordFields<Space_spheric_adapted> cf_generator(space);
    vec_ary_t coord_vectors{default_co_vector_ary(space)};

    double xo = bco_utils::get_center(space, 0);
    update_fields_co(cf_generator, coord_vectors, {}, xo);

    syst.add_cst("H", logh);
    syst.add_def("h = exp(H)");
    syst.add_cst("Omg", Omg);

    syst.add_cst("mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_def("omega^i = bet^i + Omg * mg^i");

    // derived matter quantities (the star occupies domains 0 and 1)
    for (int d = 0; d < ndoms; d++) {
        switch (d) {
            case 0:
            case 1:
                syst.add_def(d, "U^i = omega^i / N");
                syst.add_def(d, "Usquare = P^4 * U_i * U^i");
                syst.add_def(d, "Wsquare = 1. / (1. - Usquare)");
                syst.add_def(d, "W = sqrt(Wsquare)");
                syst.add_def(d, "spatCpntfourVelCov_i = U_i * P^4  * W");
                break;
            default:
                syst.add_def(d, "spatCpntfourVelCov_i = 0*bet_i");
        }
    }

    Vector velocity_cov(syst.give_val_def("spatCpntfourVelCov"));

    if (velocity_cov.get_valence() != 1 || velocity_cov.get_index_type() != COV)
        KADATH_THROW("2sacra: spatCpntfourVelCov is not a covariant rank-1 tensor");

    // EOS and density
    const double h_cut = bconfig.eos<double>(HCUT);
    const std::string eos_file = bconfig.eos<std::string>(EOSFILE);
    const std::string eos_type = bconfig.eos<std::string>(EOSTYPE);

    Param p;
    if (eos_type == "Cold_Table") {
        using namespace Kadath::Margherita;
        using eos_t = Cold_Table;
        KadathApps::init_configured_cold_table(bconfig);
        syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
        syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
        syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    } else if (eos_type == "Cold_PWPoly") {
        using namespace Kadath::Margherita;
        using eos_t = Cold_PWPoly;
        EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
        syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
        syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
        syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    }
    syst.add_def("rho = rho(h)");
    Scalar density(syst.give_val_def("rho"));

    Scalar EK(space); // extrinsic curvature trace (zero on the XCTS slice)
    EK.annule_hard();
    EK.std_base();

    std::vector<const Scalar*> field({
        &lapse, &shift_con(1), &shift_con(2), &shift_con(3),
        &log_conf, &EK,
        &AK_evolScalCov(1,1), &AK_evolScalCov(2,2), &AK_evolScalCov(3,3),
        &AK_evolScalCov(1,2), &AK_evolScalCov(1,3), &AK_evolScalCov(2,3),
        &density, &velocity_cov(1), &velocity_cov(2), &velocity_cov(3)
    });
    constexpr std::size_t density_field_index = 12; // &density above
#if defined(KADATH_SACRA_EXPORT_DEF) || defined(KADATH_SACRA_EXPORT_SGB)
    field.push_back(&varscal); 
#endif
#if defined(KADATH_SACRA_EXPORT_DEF)
    field.push_back(&xiscal);  
#endif

    // AMR grid parameters.
    // Illustration:
    // dep0: __________________________lv0
    // dep1:  _____________lv1
    // dep2:  ____lv2  ____lv4
    // dep3:   __lv3    __lv5
    double const time = 0; // initial time of simulation
    int const depth_max = grid_options.depth_max; // depth ranges from 0 to depth_max
    int const flvmax = grid_options.flvmax;
    int const lvmax = flvmax + 2 * (depth_max - flvmax);
    int const lvmax0 = lvmax + 1;

    int const doubWid = 17, doubDec = 8; // (21, 12) or (17, 8)
    int const Digit = 12;                // header x/y/z record precision: 8 or 12

    bool const is_cell_centered = true;
    int const cell_cent_shift = is_cell_centered ? 1 : 0;
    int const hllc = 0;

    int const jd_org = -grid_options.grid_half_width;
    int const ju_org = -jd_org - cell_cent_shift;
    int const kd_org = jd_org;
    int const ku_org = ju_org;
    int const ld_org = 0;
    int const lu_org = ju_org;
    int const total_nb_lg = lu_org - ld_org + 13 + hllc * 2; // = lu_org+6 - (ld_org-6) + 1

    if (nb_procs > lvmax0 * total_nb_lg) {
        std::cerr << "Number of procs exceeds vmax0*total_nb_lg = " << lvmax0 * total_nb_lg << "." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Grid spacing: finest level covers the star with margin
    std::vector<double> ts_xd(lvmax0, 0.);
    std::vector<double> ts_xu(lvmax0, 0.);
    std::vector<double> ts_yd(lvmax0, 0.);
    std::vector<double> ts_yu(lvmax0, 0.);
    std::vector<double> ts_zd(lvmax0, 0.);
    std::vector<double> ts_zu(lvmax0, 0.);
    std::vector<double> ts_dx(lvmax0, -1000.);
    std::vector<double> ts_dy(lvmax0, -1000.);
    std::vector<double> ts_dz(lvmax0, -1000.);
    std::vector<int> ts_jc_amr(lvmax0, -1000);
    std::vector<int> ts_kc_amr(lvmax0, -1000);
    std::vector<int> ts_lc_amr(lvmax0, -1000);

    std::cout << std::setprecision(15);

    double rns = bconfig(ROUT);

    ts_dx[lvmax] = rns * 1.3 * 2. / (ju_org - jd_org + cell_cent_shift);
    if (rank == 0) {
        std::cout << "rns = " << rns << std::endl;
        std::cout << "Before adjustment, ts_dx[lvmax] = " << ts_dx[lvmax] << std::endl;
    }
    ts_dy[lvmax] = ts_dx[lvmax];
    ts_dz[lvmax] = ts_dx[lvmax];

    // Single object exported in the BNS SACRA layout: the trailer carries one
    // (duplicated) puncture record built from the NS data.
    double BH_1_ahmass = bconfig(MADM);
    double BH_1_x_punc = 0, BH_1_y_punc = 0, BH_1_z_punc = 0;
    double BH_1_lvt = depth_max;

    // Output path setup
    std::string output_dir_path = argv[2];
    if (output_dir_path.back() != '/')
        output_dir_path += '/';
    if (rank == 0)
        std::cout << "output_dir_path = " << output_dir_path << std::endl;

    // NOTE: longstanding naming quirk preserved — for paths with directories
    // the count argument exceeds the remainder, so the ".toml" suffix stays in
    // the output basename (downstream scripts expect these names).
    int const input_file_path_length(input_file_path.length());
    int index_char(input_file_path_length);
    while (index_char > 0 && input_file_path[index_char - 1] != '/')
        index_char--;

    std::string partial_output_file_name = input_file_path.substr(index_char, input_file_path_length - 5)
        + "_FUKaID2SACRA_dp" + std::to_string(depth_max) + "_flv" + std::to_string(flvmax)
        + "_ju" + std::to_string(ju_org);
    if (is_cell_centered)
        partial_output_file_name += "_cell-centered";

    std::string partial_output_file_path = output_dir_path + partial_output_file_name;
    std::string output_file_path("unaffected");

    std::ofstream fout;

    // Write AMR header (rank 0)
    if (rank == 0) {
        output_file_path = partial_output_file_path + "_top_info.d";
        std::cout << "Rank " << rank << " writing top info in output_file_path = " << output_file_path
                  << std::endl << std::endl << std::endl;

        fout.open(output_file_path, std::ios::out | std::ios::trunc);

        fout << " time=" << eWpDe3(time, doubWid, doubDec) << "\n";
        fout << "AMR grid setting" << "\n";
        fout << "lvmax0=" << iW(lvmax0, 2) << "\n";
    }

    std::string str_mvpm("unaffected");

    //l213: do lv=0,lvmax
    for (int lv = 0; lv <= lvmax; lv++) {
        int const current_depth = (lv <= depth_max) ? lv : lv - (depth_max - flvmax);
        int const p_lv = (lv == depth_max + 1) ? flvmax : lv - 1;

        ts_dx[lv] = ts_dx[lvmax] * pow(2, depth_max - current_depth);
        ts_dy[lv] = ts_dy[lvmax] * pow(2, depth_max - current_depth);
        ts_dz[lv] = ts_dz[lvmax] * pow(2, depth_max - current_depth);

        ts_xd[lv] = (jd_org + 0.5 * cell_cent_shift) * ts_dx[lv];
        ts_xu[lv] = (ju_org + 0.5 * cell_cent_shift) * ts_dx[lv];
        ts_yd[lv] = (kd_org + 0.5 * cell_cent_shift) * ts_dy[lv];
        ts_yu[lv] = (ku_org + 0.5 * cell_cent_shift) * ts_dy[lv];
        ts_zd[lv] = (ld_org + 0.5 * cell_cent_shift) * ts_dz[lv];
        ts_zu[lv] = (lu_org + 0.5 * cell_cent_shift) * ts_dz[lv];
        ts_jc_amr[lv] = 0;
        ts_kc_amr[lv] = 0;
        ts_lc_amr[lv] = 0;

        str_mvpm = (lv <= flvmax) ? "N" : "Y";

        if (rank == 0) {
            std::cout << "Rank " << rank << " writing top info for level " << lv
                      << " in output_file_path = " << output_file_path << std::endl << std::endl << std::endl;

            //l220: read(401,"(4X,i3,8X,i3,7X,i3)") lv_t,ts%lv_depth,ts%lv_parent
            fout << " lv=" << iW(lv, 3) << "  depth=" << iW(current_depth, 3) << "  p_lv=" << iW(p_lv, 3) << "\n";

            //l222/l226: read(401,"(5X,2e<W>.<D>e3,2X,2i7)") ts%xd,ts%xu,jd_org,ju_org
            int const hw = (Digit == 8) ? 17 : 21;
            int const hd = Digit;
            auto axis_line = [&](const char* name, double lo, double hi, int lo_idx, int hi_idx) {
                fout << "   " << name << ":" << eWpDe3(lo, hw, hd) << eWpDe3(hi, hw, hd) << "  "
                     << iW(lo_idx, 7) << iW(hi_idx, 7) << "\n";
            };
            axis_line("x", ts_xd[lv], ts_xu[lv], jd_org, ju_org);
            axis_line("y", ts_yd[lv], ts_yu[lv], kd_org, ku_org);
            axis_line("z", ts_zd[lv], ts_zu[lv], ld_org, lu_org);

            //l230: read(401,"(8X,i3,8X,i3,8X,i3)") ts%jc_amr,ts%kc_amr,ts%lc_amr
            //l231: read(401,"(6X,a1)") ts%mvpm
            fout << " jc_amr=" << iW(ts_jc_amr[lv], 3) << " kc_amr=" << iW(ts_kc_amr[lv], 3)
                 << " lc_amr=" << iW(ts_lc_amr[lv], 3) << "\n";
            fout << " mvpm=" << str_mvpm << "\n";
        }
    }

    if (rank == 0) {
        //l663: read(401,"(11X)")
        fout << "metric data" << "\n";
        fout.close();
    }

    // MPI work dispatch
    sacra_export::LevelRange lr = compute_level_range(rank, nb_procs, lvmax0, total_nb_lg, ld_org, hllc);

    MPI_Barrier(MPI_COMM_WORLD);
    if (nb_procs <= lvmax0)
        std::cout << "Rank " << rank << " will write data of levels " << lr.first_level
                  << " to " << lr.last_level << "." << std::endl;
    else
        std::cout << "Rank " << rank << " will write data for level " << lr.first_level
                  << " from lg = " << lr.first_lg << " to lg = " << lr.last_lg << ", both included." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    // Interpolate and write field data
    Point current_point(3);

    //l665: do lv=0,lvmax
    for (int lv = lr.first_level; lv <= lr.last_level; lv++) {
        output_file_path = partial_output_file_path + "_fields_data_lv" + std::to_string(lv)
            + "_lg" + std::to_string(lr.first_lg) + "to" + std::to_string(lr.last_lg) + ".d";
        fout.open(output_file_path, std::ios::out | std::ios::trunc);

        //l697: do lg=ld_org-6,lu_org+6
        for (int lg = lr.first_lg; lg <= lr.last_lg; lg++) {
            std::cout << "Rank " << rank << " currently writing fields data at level " << lv
                      << " in output_file_path = " << output_file_path << " at lg = " << lg
                      << ", noting that this rank writes data for level " << lr.first_level
                      << " from lg = " << lr.first_lg << " to lg = " << lr.last_lg << ", both included."
                      << std::endl;
            current_point.set(3) = ts_zd[lv] + ts_dz[lv] * (lg - ld_org);

            //l698: do kg=kd_org-6,ku_org+6
            for (int kg = kd_org - (6 + hllc); kg <= ku_org + (6 + hllc); kg++) {
                current_point.set(2) = ts_yd[lv] + ts_dy[lv] * (kg - kd_org);

                //l699: do jg=jd_org-6,ju_org+6
                for (int jg = jd_org - (6 + hllc); jg <= ju_org + (6 + hllc); jg++) {
                    current_point.set(1) = ts_xd[lv] + ts_dx[lv] * (jg - jd_org);

                    // The spectral rho(h) evaluation is meaningless outside the
                    // star (h <= 1); zero it there like the BNS_nosym exporter.
                    const double H_at_point = logh.val_point(current_point);
                    for (std::size_t ff = 0; ff < field.size(); ++ff) {
                        double value = field[ff]->val_point(current_point);
                        if (ff == density_field_index)
                            value = sacra_hydro::zero_outside_matter(H_at_point, value);
                        fout << eWpDe3(value, doubWid, doubDec);
                    }
                    fout << "\n";
                }
            }
        }
        fout.close();
    }

    // Bottom info + concatenation script (rank 0)
    if (rank == 0) {
        output_file_path = partial_output_file_path + "_bottom_info.d";
        std::cout << "Rank " << rank << " writing bottom info in output_file_path = " << output_file_path
                  << std::endl << std::endl << std::endl;

        fout.open(output_file_path, std::ios::out | std::ios::trunc);

        //l982: read(401,"(7X)")
        fout << "BH data" << "\n"; // SACRA trailer label, also for NS data
        //l983: read(401,"(5X,4e17.8e3,i3)") BH_1%ahmass, BH_1%x_punc,BH_1%y_punc,BH_1%z_punc, BH_1%lvt
        // The record is duplicated so a single object can be evolved by the
        // binary version of SACRA.
        for (const char* label : {"BH_1:", "BH_2:"}) {
            fout << label << eWpDe3(BH_1_ahmass, doubWid, doubDec) << eWpDe3(BH_1_x_punc, doubWid, doubDec)
                 << eWpDe3(BH_1_y_punc, doubWid, doubDec);
            fout << eWpDe3(BH_1_z_punc, doubWid, doubDec) << iW(BH_1_lvt, 3) << "\n";
        }
        fout.close();

        // concatenation script
        output_file_path = output_dir_path + "concatenate_data_files_" + partial_output_file_name + ".sh";
        fout.open(output_file_path, std::ios::out | std::ios::trunc);
        fout << "cat " + partial_output_file_name + "_top_info.d \\" << "\n";

        if (nb_procs <= lvmax0) {
            for (int lv = 0; lv <= lvmax; lv++)
                fout << partial_output_file_name + "_fields_data_lv" + std::to_string(lv)
                        + "_lg" + std::to_string(lr.first_lg) + "to" + std::to_string(lr.last_lg) + ".d \\"
                     << "\n";
        } else {
            int const remain = nb_procs % lvmax0;
            int const quotient = nb_procs / lvmax0;
            for (int lv = 0; lv <= lvmax; lv++)
                write_concat_level_files(fout, partial_output_file_name, lv,
                                         (lv < remain) ? quotient + 1 : quotient, total_nb_lg, ld_org, hllc);

            // cross-check listing built from the per-rank dispatch
            std::ofstream fout_check(output_dir_path + "check_names_data_files_" + partial_output_file_name + ".txt",
                                     std::ios::out | std::ios::trunc);
            for (int proc_rk = 0; proc_rk < nb_procs; proc_rk++) {
                sacra_export::LevelRange pr =
                    compute_level_range(proc_rk, nb_procs, lvmax0, total_nb_lg, ld_org, hllc);
                fout_check << partial_output_file_name + "_fields_data_lv" + std::to_string(pr.first_level)
                              + "_lg" + std::to_string(pr.first_lg) + "to" + std::to_string(pr.last_lg) + ".d \\"
                           << "\n";
            }
            fout_check.close();
        }

        fout << partial_output_file_name + "_bottom_info.d \\" << "\n";
        fout << "> concatenated_" + partial_output_file_name + ".d" << "\n";
        fout.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << std::endl << std::endl << std::endl
                  << "All data written in the separate files in folder " + output_dir_path + "." << std::endl;
        std::cout << "In this folder, execute chmod 744 concatenate_data_files_" + partial_output_file_name + ".sh"
                  << std::endl;
        std::cout << "Finally exectute ./concatenate_data_files_" + partial_output_file_name + ".sh" << std::endl;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
