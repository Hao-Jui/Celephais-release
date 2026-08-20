#pragma once

#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Formalism/GR/syst_init.ipp"
#include "Apps/Formalism/Shared/xcts_syst_init.ipp"
#include "Apps/Seed/GR/trumpet_bh_seed.hpp"
#include "Apps/Workflow/solver_sequence.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "mpi.h"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace Kadath {
namespace {

template <typename space_t, typename config_t>
void fill_trumpet_background(space_t& space, config_t& bconfig, Scalar& trumpet_conf, Scalar& trumpet_lapse,
                             Scalar& trumpet_lapse_conf, Vector& trumpet_shift)
{
    CoordFields<space_t> coords(space);
    Scalar radius(coords.radius());
    Vector radial_unit(coords.template e_rad<CON>());
    Seed::NonRotatingTrumpetProfile profile(static_cast<double>(bconfig(MCH)));

    trumpet_conf.annule_hard();
    trumpet_lapse.annule_hard();
    trumpet_lapse_conf.annule_hard();
    for (int i = 1; i <= 3; ++i)
        trumpet_shift.set(i).annule_hard();

    const int ndom = space.get_nbr_domains();
    for (int dom = 0; dom < ndom; ++dom) {
        Index pos(space.get_domain(dom)->get_nbr_points());
        do {
            const double r = radius(dom)(pos);
            trumpet_conf.set_domain(dom).set(pos) = profile.conformal_factor(r);
            trumpet_lapse.set_domain(dom).set(pos) = profile.lapse(r);
            trumpet_lapse_conf.set_domain(dom).set(pos) = profile.lapse_times_conformal_factor(r);
            const double beta_r = profile.radial_shift(r);
            for (int i = 1; i <= 3; ++i)
                trumpet_shift.set(i).set_domain(dom).set(pos) = beta_r * radial_unit(i)(dom)(pos);
        } while (pos.inc());
    }

    trumpet_conf.std_base();
    trumpet_lapse.std_base();
    trumpet_lapse_conf.std_base();
    trumpet_shift.std_base();
}

template <typename space_t, typename config_t>
void fill_paper_horizon_level(space_t& space, config_t& bconfig, Scalar& horizon_level)
{
    CoordFields<space_t> coords(space);
    Scalar radius(coords.radius());
    const Seed::NonRotatingTrumpetProfile profile(static_cast<double>(bconfig(MCH)));
    const double horizon_radius = profile.horizon_isotropic_radius();
    horizon_level = radius * radius - horizon_radius * horizon_radius;
    horizon_level.std_base();
}

inline void copy_trumpet_shift(const Vector& trumpet_shift, Vector& shift)
{
    for (int i = 1; i <= 3; ++i)
        shift.set(i) = trumpet_shift(i);
    shift.std_base();
}

template <typename space_t>
void copy_trumpet_interior_background(space_t& space, const Scalar& trumpet_conf, const Scalar& trumpet_lapse,
                                      Scalar& conf, Scalar& lapse, int first_exterior_domain)
{
    for (int dom = 0; dom < first_exterior_domain; ++dom) {
        Index pos(space.get_domain(dom)->get_nbr_points());
        do {
            conf.set_domain(dom).set(pos) = trumpet_conf(dom)(pos);
            lapse.set_domain(dom).set(pos) = trumpet_lapse(dom)(pos);
        } while (pos.inc());
    }
    conf.std_base();
    lapse.std_base();
}

inline void refresh_trumpet_background_constants(System_of_eqs& syst, Scalar& trumpet_conf, Scalar& trumpet_lapse,
                                                 Scalar& trumpet_lapse_conf, Vector& trumpet_shift,
                                                 int first_active_domain)
{
    const int ndom = trumpet_conf.get_space().get_nbr_domains();
    for (int dom = first_active_domain; dom < ndom; ++dom) {
        update_field(syst, dom, "P0", trumpet_conf);
        update_field(syst, dom, "N0", trumpet_lapse);
        update_field(syst, dom, "NP0", trumpet_lapse_conf);
        update_field(syst, dom, "bet0^i", trumpet_shift);
    }
    syst.sec_member();
}

template <typename vector_array_t>
void refresh_trumpet_coordinate_constants(System_of_eqs& syst, vector_array_t& coord_vectors,
                                          int first_active_domain, int ndom)
{
    for (int dom = first_active_domain; dom < ndom; ++dom) {
        update_field(syst, dom, "mg^i", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
        update_field(syst, dom, "sm^i", *coord_vectors[to_int(coord_vector::S_BCO1)]);
        update_field(syst, dom, "einf^i", *coord_vectors[to_int(coord_vector::S_INF)]);
    }
    syst.sec_member();
}

inline void refresh_trumpet_scalar_constant(System_of_eqs& syst, Scalar& field, const char* name,
                                            int first_active_domain, int ndom)
{
    for (int dom = first_active_domain; dom < ndom; ++dom)
        update_field(syst, dom, name, field);
    syst.sec_member();
}

} // namespace

template <typename config_t, typename space_t>
trumpet_3d_xcts_solver<config_t, space_t>::trumpet_3d_xcts_solver(config_t& config_in, space_t& space_in,
                                                                  Base_tensor& base_in, Scalar& conf_in,
                                                                  Scalar& lapse_in, Vector& shift_in)
    : XCTS_Solver<config_t, space_t>(config_in, space_in, base_in), conf(conf_in), lapse(lapse_in), shift(shift_in),
      fmet(Metric_flat(space_in, base_in))
{
    coord_vectors = default_co_vector_ary(space);
    update_fields_co(cfields, coord_vectors, {}, 0.);
}

template <typename config_t, typename space_t>
std::string trumpet_3d_xcts_solver<config_t, space_t>::converged_filename(const std::string& stage) const
{
    return converged_trumpet_filename(bconfig, space, stage);
}

template <typename config_t, typename space_t>
int trumpet_3d_xcts_solver<config_t, space_t>::solve()
{
    if (std::abs(static_cast<double>(bconfig(CHI))) > 1e-14)
        KADATH_THROW("Trumpet_XCTS currently supports only non-rotating BHs (CHI = 0).");

    std::array<bool, NUM_STAGES_V> stage_enabled = bconfig.return_stages();
    int exit_status = EXIT_SUCCESS;
    if (stage_enabled[to_int(TRUMPET)])
        exit_status = trumpet_stage();

    MPI_Barrier(MPI_COMM_WORLD);
    return exit_status;
}

template <typename config_t>
inline int trumpet_3d_xcts_driver(config_t& bconfig, std::string outputdir)
{
    int exit_status = legacy_status_from_stage_outcome(KadathApps::loaded_different_dataset());
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (std::abs(static_cast<double>(bconfig(CHI))) > 1e-14)
        KADATH_THROW("Trumpet_XCTS currently supports only non-rotating BHs (CHI = 0).");

    bconfig.set(TRUMPET_BH_SEED) = true;
    bconfig.set_stage(TRUMPET) = true;

    if (!bconfig.has_seq_setting(INIT_RES))
        bconfig.seq_setting(INIT_RES) = bconfig(BCO_RES);
    validate_resolution(static_cast<int>(bconfig(BCO_RES)));
    validate_resolution(bconfig.template seq_setting_as<int>(INIT_RES));

    if (rank == 0)
        std::cout << "Solutions will be stored in: " << outputdir << "\n"
                  << "Directory will be created if it doesn't exist.\n";
    fs::create_directory(outputdir);

    if (!fs::exists(bconfig.space_filename())) {
        bconfig.set_filename("inittrumpet");
        if (rank == 0)
            Seed::setup_co<BH, Space_trumpet>(bconfig);
        MPI_Barrier(MPI_COMM_WORLD);
        bconfig.open_config();
    }

    std::string spacein = bconfig.space_filename();
    if (!fs::exists(spacein)) {
        std::ostringstream oss;
        oss << "File: " << spacein << " not found.";
        KADATH_THROW(oss.str());
    }

    while (status_requires_field_reload(exit_status)) {
        spacein = bconfig.space_filename();
        if (rank == 0) {
            std::cout << "Config File: " << bconfig.config_outputdir() + bconfig.config_filename() << std::endl
                      << "Fields File: " << spacein << std::endl
                      << bconfig << std::endl;
        }

        BeFileSource source(spacein);
        Space_trumpet space(source);
        Scalar conf(space, source);
        Scalar lapse(space, source);
        Vector shift(space, source);
        Base_tensor basis(space, CARTESIAN_BASIS);

        if (outputdir != "")
            bconfig.set_outputdir(outputdir);

        trumpet_3d_xcts_solver<decltype(bconfig), decltype(space)> solver(bconfig, space, basis, conf, lapse, shift);
        exit_status = solver.solve();
        MPI_Barrier(MPI_COMM_WORLD);
    }

    return exit_status;
}

template <typename config_t, typename space_t>
void trumpet_3d_xcts_solver<config_t, space_t>::syst_init(System_of_eqs& syst, Scalar& trumpet_conf,
                                                          Scalar& trumpet_lapse, Scalar& trumpet_lapse_conf,
                                                          Vector& trumpet_shift)
{
    // Metric
    xcts::add_flat_metric(fmet, syst);

    // Physical constants
    xcts::add_four_pi_g(syst);
    syst.add_cst("PI", M_PI);
    syst.add_cst("M", bconfig(MIRR));
    syst.add_cst("CM", bconfig(MCH));

    // Global-rotation field + flat-space surface normals (shared trio).
    xcts::add_global_rot_and_surface_coords(syst, coord_vectors);

    syst.add_cst("P0", trumpet_conf);
    syst.add_cst("N0", trumpet_lapse);
    syst.add_cst("NP0", trumpet_lapse_conf);
    syst.add_cst("bet0", trumpet_shift);

    // Fundamental fields
    xcts::add_conformal_lapse_vars(syst, conf, lapse);
    syst.add_def("NP = P * N");
    syst.add_def("bet^i = bet0^i");
    syst.add_def("Ntilde = N / P^6");
    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3. * D_k bet^k * f^ij) / 2. / Ntilde");
    syst.add_def("intMsq = P^4 / 16. / PI");

    syst.add_def(ndom - 1, "intMadm = - einf^i * D_i P / 4piG * 2");
    syst.add_def(ndom - 1, "intMk = einf^i * D_i N / 4piG");
}

template <typename config_t, typename space_t>
void trumpet_3d_xcts_solver<config_t, space_t>::print_diagnostics(const System_of_eqs& syst, const int ite,
                                                                  const double conv) const
{
    Val_domain integMsq(syst.give_val_def("intMsq")()(2));
    const double Mirrsq = space.get_domain(2)->integ(integMsq, INNER_BC);
    const double Mirr = std::sqrt(Mirrsq);

    Val_domain integMadm(syst.give_val_def("intMadm")()(ndom - 1));
    const double Madm = space.get_domain(ndom - 1)->integ(integMadm, OUTER_BC);

    Val_domain integMk(syst.give_val_def("intMk")()(ndom - 1));
    const double Mk = space.get_domain(ndom - 1)->integ(integMk, OUTER_BC);

    auto rs = bco_utils::get_rmin_rmax(space, 1);

    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << "=======================================" << std::endl
              << FORMAT << "Iter: " << ite << std::endl
              << FORMAT << "Error: " << conv << std::endl
              << FORMAT << "Madm: " << Madm << std::endl
              << FORMAT << "Mk: " << Mk << " [" << std::abs(Madm - Mk) / Madm << "]" << std::endl
              << FORMAT << "Mirr: " << Mirr << std::endl
              << FORMAT << "Horizon R: " << rs[0] << " " << rs[1] << "\n";
    std::cout.flags(f);
}

} // namespace Kadath
