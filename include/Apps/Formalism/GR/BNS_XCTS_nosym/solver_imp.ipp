#include "mpi.h"
#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"

#include <memory>
#include "Apps/Formalism/Shared/binary_driver_common.hpp"
#include "Apps/Workflow/binary_solve_driver.hpp"
#include "Apps/Formalism/Shared/PreBinary/bns_boosted_setup.hpp"
#include "Apps/Formalism/Shared/xcts_syst_init.ipp"

using namespace Kadath::Margherita;

// =============================================================================
// Constructor
// =============================================================================

namespace Kadath {


template <class eos_t, typename config_t, typename space_t>
bns_xcts_nosym_solver<eos_t, config_t, space_t>::bns_xcts_nosym_solver(config_t& config_in, space_t& space_in, Base_tensor& base_in,
                                                           Scalar& conf_in, Scalar& lapse_in, Vector& shift_in,
                                                           Scalar& logh_in, Scalar& phi_in)
    : XCTS_Solver<config_t, space_t>(config_in, space_in, base_in), conf(conf_in), lapse(lapse_in), logh(logh_in),
      phi(phi_in), shift(shift_in), fmet(Metric_flat(space_in, base_in)), xc1(bco_utils::get_center(space_in, space.NS1)),
      xc2(bco_utils::get_center(space_in, space.NS2)), xo(bco_utils::get_center(space, ndom - 1))
{
    coord_vectors = default_binary_vector_ary(space);
    update_fields(cfields, coord_vectors, {}, xo, xc1, xc2);
}

// =============================================================================
// Filename generation
// =============================================================================

template <class eos_t, typename config_t, typename space_t>
std::string bns_xcts_nosym_solver<eos_t, config_t, space_t>::converged_filename(const std::string& stage) const
{
    return converged_gr_bns_nosym_filename(bconfig, space, stage);
}

// =============================================================================
// Main solve driver
// =============================================================================

template <class eos_t, typename config_t, typename space_t>
int bns_xcts_nosym_solver<eos_t, config_t, space_t>::solve()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::string last_stage = binary_workflow_last_stage(bconfig.return_stages());
    const auto resume = this->load_existing_solution(last_stage);
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Solved previously: " << bconfig.config_filename_abs() << std::endl;
        return legacy_status_from_dataset_resume(resume);
    }

    return bns_xcts_solve_common(*this, bconfig, [&]() {
        std::cout << "=================================" << endl;
        std::cout << "BNS input" << endl;
        std::cout << "Distance: " << bconfig(DIST) << endl;
        std::cout << "Omega guess: " << bconfig(GOMEGA) << endl;
        std::cout << "Units: " << (4.0 * M_PI) << endl;
        std::cout << "=================================" << endl;
    });
}

// =============================================================================
// System initialization - common setup for all stages
// =============================================================================

template <class eos_t, typename config_t, typename space_t>
void bns_xcts_nosym_solver<eos_t, config_t, space_t>::syst_init(System_of_eqs& syst)
{
    using namespace Kadath::Margherita;


    // Metric
    xcts::add_flat_metric(fmet, syst);

    // EOS operators
    Param p;
    xcts::add_eos_operators<eos_t>(syst, p);

    // Physical constants
    xcts::add_four_pi_g(syst);

    // Star 1 parameters
    syst.add_cst("Madm1", bconfig(MADM, BCO1));
    syst.add_cst("Mb1", bconfig(MB, BCO1));
    syst.add_cst("chi1", bconfig(CHI, BCO1));
    syst.add_cst("angs1", bconfig(DEG, BCO1) * std::acos(-1.) / 180.);

    // Star 2 parameters
    syst.add_cst("Madm2", bconfig(MADM, BCO2));
    syst.add_cst("Mb2", bconfig(MB, BCO2));
    syst.add_cst("chi2", bconfig(CHI, BCO2));
    syst.add_cst("angs2", bconfig(DEG, BCO2) * std::acos(-1.) / 180.);

    // Coordinate vector fields (global rotation + BCO1/inf surface normals)
    xcts::add_global_rot_and_surface_coords(syst, coord_vectors);
    syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
    syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
    syst.add_cst("mpx", *coord_vectors[to_int(coord_vector::BCO2_ROTx)]);
    syst.add_cst("mpz", *coord_vectors[to_int(coord_vector::BCO2_ROTz)]);

    // Cartesian basis vectors
    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);
    syst.add_cst("ez", *coord_vectors[to_int(coord_vector::EZ)]);

    // Surface normal (BCO2; mg/sm/einf handled by the shared helper above)
    syst.add_cst("sp", *coord_vectors[to_int(coord_vector::S_BCO2)]);

    // Fundamental fields
    xcts::add_conformal_lapse_vars(syst, conf, lapse);
    syst.add_var("bet", shift);

    // Quasi-local ADM masses
    syst.add_var("qlMadm1", bconfig(QLMADM, BCO1));
    syst.add_var("qlMadm2", bconfig(QLMADM, BCO2));

    // Rotation setup: corotating vs irrotational/spinning
    if (bconfig.control(COROT_BIN)) {
        bconfig.set(OMEGA, BCO1) = 0.;
        syst.add_cst("omes1", bconfig(OMEGA, BCO1));
        bconfig.set(OMEGA, BCO2) = 0.;
        syst.add_cst("omes2", bconfig(OMEGA, BCO2));
    } else {
        syst.add_var("phi", phi);

        // Star 1 spin frequency
        if (!std::isnan(bconfig.set(FIXED_BCOMEGA, BCO1))) {
            syst.add_cst("omes1", bconfig(FIXED_BCOMEGA, BCO1));
            bconfig.set(OMEGA, BCO1) = bconfig(FIXED_BCOMEGA, BCO1);
        } else {
            syst.add_var("omes1", bconfig(OMEGA, BCO1));
        }

        // Star 2 spin frequency
        if (!std::isnan(bconfig.set(FIXED_BCOMEGA, BCO2))) {
            syst.add_cst("omes2", bconfig(FIXED_BCOMEGA, BCO2));
            bconfig.set(OMEGA, BCO2) = bconfig(FIXED_BCOMEGA, BCO2);
        } else {
            syst.add_var("omes2", bconfig(OMEGA, BCO2));
        }

        // Spin axis definitions
        syst.add_def("mm^i = cos(angs1) * mmz^i + sin(angs1) * mmx^i");
        syst.add_def("mp^i = cos(angs2) * mpz^i + sin(angs2) * mpx^i");

        // Fluid velocity: irrotational + spinning parts
        for (int d = space.NS1; d <= space.ADAPTED1; ++d) {
            syst.add_def(d, "s^i = omes1 * mm^i");
            syst.add_def(d, "eta_i = D_i phi + P^4 * s_i");
        }
        for (int d = space.NS2; d <= space.ADAPTED2; ++d) {
            syst.add_def(d, "s^i = omes2 * mp^i");
            syst.add_def(d, "eta_i = D_i phi + P^4 * s_i");
        }
    }

    // Conformal combinations (NP, Ntilde) + ADM mass integrands at infinity
    xcts::add_conformal_combinations_and_adm_integrands(syst, ndom);

    // Thermodynamic quantities
    xcts::add_thermodynamic_defs(syst);

    // Extrinsic curvature + ADM linear-momentum integrands (binary-shared)
    xcts::add_extrinsic_curvature_and_adm_momenta(syst, ndom);

    // Quasi-local spin integrands (binary-shared; domains differ per app)
    xcts::add_quasilocal_spin_integrands(syst, space.ADAPTED1 + 1, space.ADAPTED2 + 1);

    // -------------------------------------------------------------------------
    // No-symmetry rotational-gauge closure (bordered). muz1/muz2 force the rigid-
    // z-rotation directions mmz/mpz in eqbet (see the stage eqbet defs); the two
    // point conditions pin curl_z(bet) = (D^i bet^j - D^j bet^i) ex_i ey_j = 0 at
    // each stellar center. +2 unknowns balanced by +2 conditions keeps the
    // Jacobian square. Toggle via BNS_NOSYM_ROT_GAUGE.
    // -------------------------------------------------------------------------
    if (rotational_gauge_enabled()) {
        syst.add_var("muz1", muz1_gauge);
        syst.add_var("muz2", muz2_gauge);

        Index pos_center1(space.get_domain(space.NS1)->get_nbr_points());
        Index pos_center2(space.get_domain(space.NS2)->get_nbr_points());
        syst.add_eq_val(space.NS1, "(D^i bet^j - D^j bet^i) * ex_i * ey_j", pos_center1);
        syst.set_last_eq_int_reflection_sector(+1);
        syst.add_eq_val(space.NS2, "(D^i bet^j - D^j bet^i) * ex_i * ey_j", pos_center2);
        syst.set_last_eq_int_reflection_sector(+1);
    }
}

// =============================================================================
// Runtime diagnostics
// =============================================================================

template <class eos_t, typename config_t, typename space_t>
void bns_xcts_nosym_solver<eos_t, config_t, space_t>::print_diagnostics(System_of_eqs const& syst, const int ite,
                                                                  const double conv) const
{
    bns_xcts_print_diagnostics(space, bconfig, syst, ite, conv);
}

// =============================================================================
// Configuration update
// =============================================================================

template <class eos_t, typename config_t, typename space_t>
void bns_xcts_nosym_solver<eos_t, config_t, space_t>::update_config_quantities(const double& loghc)
{
    xcts::update_config_quantities<eos_t>(bconfig, loghc);
}

// =============================================================================
// Boosted 3D setup utility
// =============================================================================

template <typename eos_t, typename NSpaceT, typename BinSpaceT, typename NSOuterAdaptedT, typename NSInnerAdaptedT,
          typename BinOuterAdaptedT, typename BinInnerAdaptedT>
inline void bns_setup_boosted_3d(kadath_config<BCO_NS_INFO>& NS1config,
                                 kadath_config<BCO_NS_INFO>& NS2config, kadath_config<BIN_INFO>& bconfig)
{
    // nosym (vacuum) binary superposition — shared 5-field core (WithScalar=false).
    // nosym space + domain types arrive via bns_setup_boosted_3d_nosym (solver.hpp).
    bns_setup_boosted_3d_impl<false, eos_t, NSpaceT, BinSpaceT, NSOuterAdaptedT, NSInnerAdaptedT,
                              BinOuterAdaptedT, BinInnerAdaptedT>(
        NS1config, NS2config, bconfig,
        [](kadath_config<BCO_NS_INFO>&, kadath_config<BCO_NS_INFO>&) {});
}

} // namespace Kadath
