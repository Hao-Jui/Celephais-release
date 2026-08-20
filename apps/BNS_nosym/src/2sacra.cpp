#include <string>
#include "For_Kadath/Array/exceptions.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <exception>
#include <vector>
#include "mpi.h"
#include "For_Kadath/Kadath_point_h/kadath_bin_ns_nosym.hpp"
#include "Hydro/EOS.hh"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Utilities/exporter_utilities.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "Apps/Diagnostics/configured_eos.hpp"
#include "Apps/Diagnostics/sacra_export_utils.hpp"
#include "Apps/Diagnostics/sacra_hydro_conversion.hpp"
#include "Apps/Diagnostics/sacra_format.hpp"
#include <cmath>
#include <cstddef>

namespace fs = std::filesystem;
using namespace Kadath;
using namespace export_utils;
using sacra_fmt::eWpDe3;
using sacra_fmt::iW;

namespace {

int snap_to_parent_grid(double center, double origin, double parent_dx)
{
    double ratio = std::abs(center - origin) / parent_dx;
    int steps = (ratio - std::floor(ratio) < 0.5)
        ? static_cast<int>(std::floor(ratio))
        : static_cast<int>(std::floor(ratio)) + 1;
    if (center < origin)
        steps = -steps;
    return steps;
}

struct LevelRange {
    int first_level;
    int last_level;
    int first_lg;
    int last_lg;
};

LevelRange compute_level_range(int rank, int nb_procs, int lvmax0, int total_nb_lg,
                               int ld_org, int hllc)
{
    LevelRange r{};
    if (nb_procs <= lvmax0) {
        r.first_lg = ld_org - (6 + hllc);
        r.last_lg = -ld_org + (6 + hllc) - 1 + total_nb_lg + ld_org - (6 + hllc);

        int remain = lvmax0 % nb_procs;
        int quotient = lvmax0 / nb_procs;
        if (rank < remain) {
            r.first_level = rank * (quotient + 1);
            r.last_level = r.first_level + quotient;
        } else {
            r.first_level = remain * (quotient + 1) + (rank - remain) * quotient;
            r.last_level = r.first_level + quotient - 1;
        }
        r.first_lg = ld_org - (6 + hllc);
        r.last_lg = ld_org - (6 + hllc) + total_nb_lg - 1;
    } else {
        int remain = nb_procs % lvmax0;
        int quotient = nb_procs / lvmax0;
        int smallest_rank;
        int procs_on_level;

        if (rank < remain * (quotient + 1)) {
            r.first_level = rank / (quotient + 1);
            smallest_rank = r.first_level * (quotient + 1);
            procs_on_level = quotient + 1;
        } else {
            r.first_level = remain + (rank - remain * (quotient + 1)) / quotient;
            smallest_rank = remain * (quotient + 1) + (r.first_level - remain) * quotient;
            procs_on_level = quotient;
        }
        r.last_level = r.first_level;

        int remain_lg = total_nb_lg % procs_on_level;
        int quotient_lg = total_nb_lg / procs_on_level;
        int local_rank = rank - smallest_rank;

        if (local_rank < remain_lg) {
            r.first_lg = ld_org - (6 + hllc) + local_rank * (quotient_lg + 1);
            r.last_lg = r.first_lg + quotient_lg;
        } else {
            r.first_lg = ld_org - (6 + hllc) + remain_lg * (quotient_lg + 1) + (local_rank - remain_lg) * quotient_lg;
            r.last_lg = r.first_lg + quotient_lg - 1;
        }
    }
    return r;
}

void write_concat_level_files(std::ofstream& fout, const std::string& base,
                              int lv, int procs_on_level, int total_nb_lg,
                              int ld_org, int hllc)
{
    int remain_lg = total_nb_lg % procs_on_level;
    int quotient_lg = total_nb_lg / procs_on_level;

    for (int p = 0; p < procs_on_level; ++p) {
        int flg, llg;
        if (p < remain_lg) {
            flg = ld_org - (6 + hllc) + p * (quotient_lg + 1);
            llg = flg + quotient_lg;
        } else {
            flg = ld_org - (6 + hllc) + remain_lg * (quotient_lg + 1) + (p - remain_lg) * quotient_lg;
            llg = flg + quotient_lg - 1;
        }
        fout << base + "_fields_data_lv" + std::to_string(lv) + "_lg" + std::to_string(flg) + "to" + std::to_string(llg) + ".d \\" << std::endl;
    }
}

} // anonymous namespace


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
        "Defaults: --depth-max 9 --flvmax 5 --N 80; jd_org is set to -N.";
    if (argc < 3) {
        if (rank == 0)
            std::cerr << "Missing required arguments.\n" << usage << "\n";
        MPI_Finalize();
        return 1;
    }

    sacra_export::GridOptions grid_options{9, 5, 80};
    try {
        grid_options = sacra_export::parse_grid_options(argc, argv, 3, grid_options, usage);
    } catch (const std::exception& e) {
        if (rank == 0)
            std::cerr << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    std::string input_file_path = argv[1];
    kadath_config<BIN_INFO> bconfig(input_file_path);
    std::string kadath_filename = bconfig.space_filename();

    if (!fs::exists(kadath_filename)) {
        std::ostringstream oss;
        oss << "File: " << kadath_filename << " not found.\n\n";
        KADATH_THROW(oss.str());
    }

    // AMR grid parameters
    double const time = 0;
    int const depth_max = grid_options.depth_max;
    int const flvmax = grid_options.flvmax;
    int const lvmax = flvmax + 2 * (depth_max - flvmax);
    int const lvmax0 = lvmax + 1;

    bool const is_cell_centered = true;
    int const cell_cent_shift = is_cell_centered ? 1 : 0;
    int const hllc = 0;

    int const jd_org = -grid_options.grid_half_width;
    int const ju_org = -jd_org - cell_cent_shift;
    int const kd_org = jd_org;
    int const ku_org = ju_org;
    int const ld_org = kd_org;
    int const lu_org = ku_org;
    int const total_nb_lg = lu_org - ld_org + 13 + hllc * 2;

    if (nb_procs > lvmax0 * total_nb_lg) {
        std::cerr << "Should call with nb_procs <= lvmax0*total_nb_lg = " << lvmax0 * total_nb_lg << "\n";
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    // Grid spacing: finest level sets the scale
    std::vector<double> ts_dx(lvmax0, -1000.);
    std::vector<double> ts_dy(lvmax0, -1000.);
    std::vector<double> ts_dz(lvmax0, -1000.);

    ts_dx[lvmax] = 10. * 2. / (ju_org - jd_org + cell_cent_shift);

    if (rank == 0)
        std::cout << std::setprecision(15) << "ts_dx[lvmax] = " << ts_dx[lvmax] << "\n";
    ts_dy[lvmax] = ts_dx[lvmax];
    ts_dz[lvmax] = ts_dx[lvmax];

    // Read Kadath fields
    BeFileSource fin(kadath_filename);
    Space_bin_ns_nosym space(fin);
    Scalar conf(space, fin);
    Scalar lapse(space, fin);
    Vector shift_con(space, fin);
    Scalar logh(space, fin);
    Scalar velocity_potential(space, fin);

    Base_tensor basis(shift_con.get_basis());
    Metric_flat fmet(space, basis);

    int ndoms = space.get_nbr_domains();
    System_of_eqs syst(space, 0, ndoms - 1);
    fmet.set_system(syst, "f");

    // Derived fields
    syst.add_cst("P", conf);
    syst.add_def("logP = log(P)");
    Scalar log_conf(syst.give_val_def("logP"));

    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift_con);
    syst.add_def("AKevolScalCov_ij = (D_i bet_j + D_j bet_i - 2. / 3.* D^k bet_k * f_ij) /2. / N");
    Tensor AK_evolScalCov(syst.give_val_def("AKevolScalCov"));

    // Coordinate fields
    double xc1 = bco_utils::get_center(space, space.NS1);
    double xc2 = bco_utils::get_center(space, space.NS2);
    double xo = bco_utils::get_center(space, ndoms - 1);

    CoordFields<Space_bin_ns_nosym> cfields(space);
    vec_ary_t coord_vectors{default_binary_vector_ary(space)};
    update_fields(cfields, coord_vectors, {}, xo, xc1, xc2);

    // NS spins and velocity
    syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
    syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
    syst.add_cst("mpx", *coord_vectors[to_int(coord_vector::BCO2_ROTx)]);
    syst.add_cst("mpz", *coord_vectors[to_int(coord_vector::BCO2_ROTz)]);
    syst.add_cst("omes1", bconfig(OMEGA, BCO1));
    syst.add_cst("angs1", bconfig(DEG, BCO1) * std::acos(-1.) / 180.);
    syst.add_cst("omes2", bconfig(OMEGA, BCO2));
    syst.add_cst("angs2", bconfig(DEG, BCO2) * std::acos(-1.) / 180.);
    syst.add_def("mm^i = cos(angs1) * mmz^i + sin(angs1) * mmx^i ");
    syst.add_def("mp^i = cos(angs2) * mpz^i + sin(angs2) * mpx^i ");

    for (int d = space.NS1; d <= space.ADAPTED1; ++d)
        syst.add_def(d, "s^i  = omes1 * mm^i");
    for (int d = space.NS2; d <= space.ADAPTED2; ++d)
        syst.add_def(d, "s^i  = omes2 * mp^i");

    syst.add_cst("velocPot", velocity_potential);
    for (int d = 0; d < ndoms; ++d) {
        if ((d <= space.ADAPTED1) || (d <= space.ADAPTED2 && d >= space.NS2))
            syst.add_def(d, "eta_i = D_i velocPot + P^4 * s_i");
        else
            syst.add_def(d, "eta_i = D_i velocPot");
    }
    syst.add_cst("H", logh);
    syst.add_def("h = exp(H)");
    syst.add_def("spatCpntfourVelCov_i = eta_i/h");
    Vector velocity_cov(syst.give_val_def("spatCpntfourVelCov"));

    // EOS and density
    const double h_cut = bconfig.eos<double>(HCUT, BCO1);
    const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
    const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);

    Param p;
    if (eos_type == "Cold_Table") {
        using namespace Kadath::Margherita;
        using eos_t = Cold_Table;
        KadathApps::init_configured_cold_table(bconfig, BCO1);
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

    // SACRA field vector: ALPHA, BX, BY, BZ, phitmp, EK, AKXX..AKYZ, rho, ux, uy, uz
    Scalar EK(space);
    EK.annule_hard();
    EK.std_base();

    std::vector<const Scalar*> field({
        &lapse, &shift_con(1), &shift_con(2), &shift_con(3),
        &log_conf, &EK,
        &AK_evolScalCov(1,1), &AK_evolScalCov(2,2), &AK_evolScalCov(3,3),
        &AK_evolScalCov(1,2), &AK_evolScalCov(1,3), &AK_evolScalCov(2,3),
        &density, &velocity_cov(1), &velocity_cov(2), &velocity_cov(3)
    });
    constexpr std::size_t density_field_index = 12;

    // Output path setup
    std::string output_dir_path = argv[2];
    if (output_dir_path.back() != '/')
        output_dir_path += '/';
    if (rank == 0)
        std::cout << "output_dir_path = " << output_dir_path << "\n";

    std::string basename = fs::path(input_file_path).stem().string();
    std::string partial_name = basename + "_FUKaID2SACRA_dp" + std::to_string(depth_max)
        + "_flv" + std::to_string(flvmax) + "_ju" + std::to_string(ju_org);
    if (is_cell_centered)
        partial_name += "_cell-centered";
    std::string partial_path = output_dir_path + partial_name;

    // Write AMR header (rank 0)
    int const wid = 17, dec = 8;
    std::ofstream fout;

    if (rank == 0) {
        fout.open(partial_path + "_top_info.d", std::ios::out | std::ios::trunc);
        fout << " time=" << eWpDe3(time, wid, dec) << "\n";
        fout << "AMR grid setting\n";
        fout << "lvmax0=" << iW(lvmax0, 2) << "\n";
    }

    // NS centers in COM frame
    Point center_NS1(space.get_domain(space.NS1)->get_center());
    Point center_NS2(space.get_domain(space.NS2)->get_center());
    double COMx = -bconfig(COM);
    double COMy = 0.;
    if (!std::isnan(bconfig.set(COMY)))
        COMy = -bconfig(COMY);
    center_NS1.set(1) -= COMx;
    center_NS2.set(1) -= COMx;
    center_NS1.set(2) -= COMy;
    center_NS2.set(2) -= COMy;

    if (rank == 0)
        std::cout << "center of NS1 = " << eWpDe3(center_NS1(1), wid, dec) << "\n";

    // Build AMR level hierarchy
    std::vector<double> ts_xd(lvmax0, 0.);
    std::vector<double> ts_xu(lvmax0, 0.);
    std::vector<double> ts_yd(lvmax0, 0.);
    std::vector<double> ts_yu(lvmax0, 0.);
    std::vector<double> ts_zd(lvmax0, 0.);
    std::vector<double> ts_zu(lvmax0, 0.);
    std::vector<int> ts_jc_amr(lvmax0, 0);
    std::vector<int> ts_kc_amr(lvmax0, 0);
    std::vector<int> ts_lc_amr(lvmax0, 0);

    Point center_NS_auxi(center_NS1);
    Point origin_coord(3);
    origin_coord.set(1) = 0; origin_coord.set(2) = 0; origin_coord.set(3) = 0;

    for (int lv = 0; lv <= lvmax; lv++) {
        int current_depth = (lv <= depth_max) ? lv : lv - (depth_max - flvmax);
        double scale = std::pow(2, depth_max - current_depth);
        ts_dx[lv] = ts_dx[lvmax] * scale;
        ts_dy[lv] = ts_dy[lvmax] * scale;
        ts_dz[lv] = ts_dz[lvmax] * scale;

        int p_lv = (lv == depth_max + 1) ? flvmax : lv - 1;
        std::string str_mvpm;

        if (lv <= flvmax) {
            double half = 0.5 * cell_cent_shift;
            ts_xd[lv] = (jd_org + half) * ts_dx[lv];
            ts_xu[lv] = (ju_org + half) * ts_dx[lv];
            ts_yd[lv] = (kd_org + half) * ts_dy[lv];
            ts_yu[lv] = (ku_org + half) * ts_dy[lv];
            ts_zd[lv] = (ld_org + half) * ts_dz[lv];
            ts_zu[lv] = (lu_org + half) * ts_dz[lv];
            ts_jc_amr[lv] = 0;
            ts_kc_amr[lv] = 0;
            ts_lc_amr[lv] = 0;
            str_mvpm = "N";
        } else {
            if (lv == depth_max + 1) {
                center_NS_auxi = center_NS2;
                origin_coord.set(1) = 0;
                origin_coord.set(2) = 0;
                origin_coord.set(3) = 0;
            }

            double half = 0.5 * cell_cent_shift;
            double parent_dx[] = {ts_dx[p_lv], ts_dy[p_lv], ts_dz[p_lv]};
            int* amr[] = {&ts_jc_amr[lv], &ts_kc_amr[lv], &ts_lc_amr[lv]};
            int lo[] = {jd_org, kd_org, ld_org};
            int hi[] = {ju_org, ku_org, lu_org};
            double* xd[] = {&ts_xd[lv], &ts_yd[lv], &ts_zd[lv]};
            double* xu[] = {&ts_xu[lv], &ts_yu[lv], &ts_zu[lv]};

            for (int axis = 0; axis < 3; ++axis) {
                int dim = axis + 1;
                *amr[axis] = snap_to_parent_grid(center_NS_auxi(dim), origin_coord(dim), parent_dx[axis]);
                origin_coord.set(dim) += (*amr[axis]) * parent_dx[axis];
                *xd[axis] = origin_coord(dim) + (lo[axis] + half) * ts_dx[lv]; // all dx=dy=dz
                *xu[axis] = origin_coord(dim) + (hi[axis] + half) * ts_dx[lv];
            }
            str_mvpm = "Y";
        }

        if (rank == 0) {
            fout << " lv=" << iW(lv, 3) << "  depth=" << iW(current_depth, 3) << "  p_lv=" << iW(p_lv, 3) << "\n";
            //l226: read(401,"(5X,2e21.12e3,2X,2i7)") ts%xd,ts%xu,jd_org,ju_org
            fout << "   x:" << eWpDe3(ts_xd[lv], 21, 12) << eWpDe3(ts_xu[lv], 21, 12) << "  " << iW(jd_org, 7) << iW(ju_org, 7) << "\n";
            fout << "   y:" << eWpDe3(ts_yd[lv], 21, 12) << eWpDe3(ts_yu[lv], 21, 12) << "  " << iW(kd_org, 7) << iW(ku_org, 7) << "\n";
            fout << "   z:" << eWpDe3(ts_zd[lv], 21, 12) << eWpDe3(ts_zu[lv], 21, 12) << "  " << iW(ld_org, 7) << iW(lu_org, 7) << "\n";
            fout << " jc_amr=" << iW(ts_jc_amr[lv], 3) << " kc_amr=" << iW(ts_kc_amr[lv], 3) << " lc_amr=" << iW(ts_lc_amr[lv], 3) << "\n";
            fout << " mvpm=" + str_mvpm << "\n";
        }
    }

    if (rank == 0) {
        fout << "metric data\n";
        fout.close();
    }

    // MPI work dispatch
    LevelRange lr = compute_level_range(rank, nb_procs, lvmax0, total_nb_lg, ld_org, hllc);

    // Interpolate and write field data
    Point current_point(3);
    for (int lv = lr.first_level; lv <= lr.last_level; lv++) {
        std::string path = partial_path + "_fields_data_lv" + std::to_string(lv)
            + "_lg" + std::to_string(lr.first_lg) + "to" + std::to_string(lr.last_lg) + ".d";
        fout.open(path, std::ios::out | std::ios::trunc);

        for (int lg = lr.first_lg; lg <= lr.last_lg; lg++) {
            current_point.set(3) = ts_zd[lv] + ts_dz[lv] * (lg - ld_org);

            for (int kg = kd_org - 6; kg <= ku_org + 6; kg++) {
                current_point.set(2) = ts_yd[lv] + ts_dy[lv] * (kg - kd_org) + COMy;

                for (int jg = jd_org - 6; jg <= ju_org + 6; jg++) {
                    current_point.set(1) = ts_xd[lv] + ts_dx[lv] * (jg - jd_org) + COMx;

                    //l738: read(401,"(16e17.8e3)") ALPHA, BX,BY,BZ, phitmp, EK, AKXX..AKYZ, rho,ux,uy,uz
                    const double H_at_point = logh.val_point(current_point);
                    for (std::size_t ff = 0; ff < field.size(); ++ff) {
                        double value = field[ff]->val_point(current_point);
                        if (ff == density_field_index)
                            value = sacra_hydro::zero_outside_matter(H_at_point, value);
                        fout << eWpDe3(value, wid, dec);
                    }
                    fout << "\n";
                }
            }
        }
        fout.close();
    }

    // Bottom info: NS center data (rank 0)
    if (rank == 0) {
        fout.open(partial_path + "_bottom_info.d", std::ios::out | std::ios::trunc);
        fout << "BH data\n";

        const Point* centers[] = {&center_NS1, &center_NS2};
        NODES bco_keys[] = {BCO1, BCO2};
        int lvt_vals[] = {depth_max, lvmax};

        for (int i = 0; i < 2; ++i) {
            fout << "BH_" << (i + 1) << ":"
                 << eWpDe3(bconfig(MADM, bco_keys[i]), wid, dec)
                 << eWpDe3((*centers[i])(1), wid, dec)
                 << eWpDe3((*centers[i])(2), wid, dec)
                 << eWpDe3((*centers[i])(3), wid, dec)
                 << iW(lvt_vals[i], 3) << "\n";
        }
        fout.close();

        // Concatenation script
        std::string concat_path = output_dir_path + "concatenate_data_files_" + partial_name + ".sh";
        fout.open(concat_path, std::ios::out | std::ios::trunc);
        fout << "cat " + partial_name + "_top_info.d \\\n";

        if (nb_procs <= lvmax0) {
            for (int lv = 0; lv <= lvmax; lv++) {
                fout << partial_name + "_fields_data_lv" + std::to_string(lv)
                    + "_lg" + std::to_string(lr.first_lg) + "to" + std::to_string(lr.last_lg) + ".d \\\n";
            }
        } else {
            int remain = nb_procs % lvmax0;
            int quotient = nb_procs / lvmax0;
            for (int lv = 0; lv <= lvmax; lv++) {
                int procs = (lv < remain) ? quotient + 1 : quotient;
                write_concat_level_files(fout, partial_name, lv, procs, total_nb_lg, ld_org, hllc);
            }
        }

        fout << partial_name + "_bottom_info.d \\\n";
        fout << "> concatenated_" + partial_name + ".d\n";
        fout.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n\nAll data written in " << output_dir_path << "\n";
        std::cout << "Run: chmod 744 concatenate_data_files_" << partial_name << ".sh\n";
        std::cout << "Then: ./concatenate_data_files_" << partial_name << ".sh\n";
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
