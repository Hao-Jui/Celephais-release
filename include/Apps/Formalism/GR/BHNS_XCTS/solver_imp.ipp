#include "mpi.h"
#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bco_regrid.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"
#include "Apps/Formalism/Shared/PreBinary/bco_binary_setup.hpp"
#include "Apps/Formalism/Shared/PreBinary/bhns_boosted_setup.hpp"
#include "Apps/Workflow/binary_solve_driver.hpp"
#include "Apps/Formalism/GR/syst_init.ipp"
#include "Apps/Formalism/Shared/xcts_syst_init.ipp"
#include "Apps/Formalism/GR/BH_3D_XCTS/solver.hpp"
#include "Apps/Formalism/GR/NS_3D_XCTS/solver.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/Kadath_point_h/kadath.hpp"

using namespace Kadath::Margherita;

namespace Kadath {


template <class eos_t, typename config_t, typename space_t>
bhns_xcts_solver<eos_t, config_t, space_t>::bhns_xcts_solver(config_t& config_in, space_t& space_in,
                                                             Base_tensor& base_in, Scalar& conf_in, Scalar& lapse_in,
                                                             Vector& shift_in, Scalar& logh_in, Scalar& phi_in)
    : XCTS_Solver<config_t, space_t>(config_in, space_in, base_in), conf(conf_in), lapse(lapse_in), logh(logh_in),
      phi(phi_in), shift(shift_in), fmet(Metric_flat(space_in, base_in)), xc1(bco_utils::get_center(space_in, space.NS)),
      xc2(bco_utils::get_center(space_in, space.BH)), xo(bco_utils::get_center(space, ndom - 1)),
      excluded_doms({space.BH, space.BH + 1})
{
    // initialize coordinate vector fields
    coord_vectors = default_binary_vector_ary(space);

    update_fields(cfields, coord_vectors, {}, xo, xc1, xc2);
}

// standardized filename for each converged dataset at the end of each stage.
template <class eos_t, typename config_t, typename space_t>
std::string bhns_xcts_solver<eos_t, config_t, space_t>::converged_filename(const std::string& stage) const
{
    return converged_gr_bhns_filename(bconfig, space, stage);
}

template <class eos_t, typename config_t, typename space_t>
int bhns_xcts_solver<eos_t, config_t, space_t>::solve(bool want_warmup)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto stage_enabled = bconfig.return_stages();
    const std::string last_stage = binary_workflow_last_stage(stage_enabled);

    // check if we have solved this before
    const auto resume = this->load_existing_solution(last_stage);
    if (resume.found()) {
        if (rank == 0)
            std::cout << "Solved previously: " << bconfig.config_filename_abs() << std::endl;
        return legacy_status_from_dataset_resume(resume);
    }

    // Overview + QUASI_EQUIL / FORCE_BALANCE / ECC_RED dispatch are identical
    // across the sym and nosym BHNS solvers (Apps/Formalism/Shared).
    return bhns_xcts_solve_common(*this, bconfig, want_warmup);
}

template <class eos_t, typename config_t, typename space_t>
void bhns_xcts_solver<eos_t, config_t, space_t>::syst_init(System_of_eqs& syst)
{
    using namespace Kadath::Margherita;


    const int ndom = space.get_nbr_domains();
    // call the (flat) conformal metric "f"
    xcts::add_flat_metric(fmet, syst);

    // define the EOS operators
    Param p;
    xcts::add_eos_operators<eos_t>(syst, p);

    // define numerical constants
    xcts::add_four_pi_g(syst);

    // baryonic mass and dimensionless spin are fixed input parameters
    // along with the ADM of each NS at infinite separation
    syst.add_cst("Madm1", bconfig(MADM, BCO1));
    syst.add_cst("Mb1", bconfig(MB, BCO1));
    syst.add_cst("chi1", bconfig(CHI, BCO1));
    syst.add_cst("angs1", bconfig(DEG, BCO1) * std::acos(-1.) / 180.);

    syst.add_cst("Mch", bconfig(MCH, BCO2));
    syst.add_cst("Mirr", bconfig(MIRR, BCO2));
    syst.add_cst("chi2", bconfig(CHI, BCO2));
    syst.add_cst("angs2", bconfig(DEG, BCO2) * std::acos(-1.) / 180.);
    syst.add_cst("n0", bconfig(FIXED_LAPSE, BCO2));

    // global-rotation field + flat-space surface normals (shared trio); the
    // per-object rotation / Cartesian vectors and the BCO2 normal "sp" follow.
    // COO-safe reorder vs the previous interleaved spelling.
    xcts::add_global_rot_and_surface_coords(syst, coord_vectors);

    // coordinate dependent (rotational) vector fields
    syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
    syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
    syst.add_cst("mpx", *coord_vectors[to_int(coord_vector::BCO2_ROTx)]);
    syst.add_cst("mpz", *coord_vectors[to_int(coord_vector::BCO2_ROTz)]);

    syst.add_def("mm^i = cos(angs1) * mmz^i + sin(angs1) * mmx^i ");
    syst.add_def("mp^i = cos(angs2) * mpz^i + sin(angs2) * mpx^i ");

    // cartesianbasis vector fields
    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);
    syst.add_cst("ez", *coord_vectors[to_int(coord_vector::EZ)]);

    // BCO2 surface normal ("sm"/"einf" emitted above with the shared trio)
    syst.add_cst("sp", *coord_vectors[to_int(coord_vector::S_BCO2)]);

    // the basic fields, conformal factor, lapse and (log) enthalpy
    xcts::add_conformal_lapse_vars(syst, conf, lapse);
    syst.add_var("bet", shift);

    // quasi-local measurement of the component ADM masses
    syst.add_var("qlMadm1", bconfig(QLMADM, BCO1));

    // check if corotation is considered and adjust
    // rotational velocity components accordingly
    if (bconfig.control(COROT_BIN)) {
        // for a corotating binary, there is no angular frequency parameter
        // since stellar rotation is fixed by the orbital frequency
        bconfig.set(OMEGA, BCO1) = 0.;
        syst.add_cst("omes1", bconfig(OMEGA, BCO1));

        bconfig.set(OMEGA, BCO2) = 0.;
        syst.add_cst("omes2", bconfig(OMEGA, BCO2));
    } else {
        // in the arbitrary spinning / irrotational case
        // the irrotational part is defined by the velocity potential
        syst.add_var("phi", phi);

        // check if stellar omega is fixed or has to be solved
        // for a specific dimensionless spin
        if (!std::isnan(bconfig.set(FIXED_BCOMEGA, BCO1))) {
            syst.add_cst("omes1", bconfig(FIXED_BCOMEGA, BCO1));
            bconfig.set(OMEGA, BCO1) = bconfig(FIXED_BCOMEGA, BCO1);
        } else {
            syst.add_var("omes1", bconfig(OMEGA, BCO1));
        }

        if (!std::isnan(bconfig.set(FIXED_BCOMEGA, BCO2))) {
            syst.add_cst("omes2", bconfig(FIXED_BCOMEGA, BCO2));
            bconfig.set(OMEGA, BCO2) = bconfig(FIXED_BCOMEGA, BCO2);
        } else {
            syst.add_var("omes2", bconfig(OMEGA, BCO2));
        }

        // for mixed spins, we have additional definitions
        // that are required for both objects that describe
        // the local rotation field
        for (int d = space.NS; d <= space.ADAPTEDNS; ++d) {
            syst.add_def(d, "s^i  = omes1 * mm^i");
            syst.add_def(d, "eta_i  = D_i phi + P^4 * s_i");
        }
        syst.add_def(space.ADAPTEDBH + 1, "s^i = omes2 * mp^i");
    }

    // define common combinations of conformal factor and lapse, and the
    // quantities integrated at infinity (two equivalent ADM-mass definitions
    // plus the Komar mass)
    xcts::add_conformal_combinations_and_adm_integrands(syst, ndom);
    syst.add_def("intMsq  = P^4 / 4. / 4piG");

    // enthalpy, rest-mass density, internal energy, pressure (functions of the
    // logarithmic enthalpy H, the actual variable), and the rescaling delta
    xcts::add_thermodynamic_defs(syst);

    // the conformal extrinsic curvature + ADM linear-momentum integrands
    // (binary-shared)
    xcts::add_extrinsic_curvature_and_adm_momenta(syst, ndom);

    // quasi-local spin, surface integral outside the stellar matter distribution
    // (binary-shared; domains differ per app)
    xcts::add_quasilocal_spin_integrands(syst, space.ADAPTEDNS + 1, space.ADAPTEDBH + 1);
}

// runtime diagnostics specific for rotating solutions
template <class eos_t, typename config_t, typename space_t>
void bhns_xcts_solver<eos_t, config_t, space_t>::print_diagnostics(System_of_eqs const& syst, const int ite,
                                                                   const double conv) const
{
    std::ios_base::fmtflags f(std::cout.flags());

    double baryonic_massNS = 0.;
    double ql_massNS = 0.;
    for (int d = space.NS; d <= space.ADAPTEDNS; ++d) {
        baryonic_massNS += syst.give_val_def("intMb")()(d).integ_volume();
        ql_massNS += syst.give_val_def("intM")()(d).integ_volume();
    }

    auto [rmin, rmax] = bco_utils::get_rmin_rmax(space, space.ADAPTEDNS);
    double r_bh = bco_utils::get_radius(space.get_domain(space.ADAPTEDBH), EQUI);

    double mirrsq = space.get_domain(space.BH + 2)->integ(syst.give_val_def("intMsq")()(space.BH + 2), INNER_BC);
    double Mirr = std::sqrt(mirrsq);

    Val_domain integS1(syst.give_val_def("intS1")()(space.ADAPTEDNS + 1));
    double S1 = space.get_domain(space.ADAPTEDNS + 1)->integ(integS1, OUTER_BC);
    double chi1 = S1 / bconfig(MADM, BCO1) / bconfig(MADM, BCO1);
    double chiql1 = S1 / ql_massNS / ql_massNS;

    Val_domain integS2(syst.give_val_def("intS2")()(space.ADAPTEDBH + 1));
    double S2 = space.get_domain(space.ADAPTEDBH + 1)->integ(integS2, OUTER_BC);
    double Mch = std::sqrt(mirrsq + S2 * S2 / 4. / mirrsq);
    double chi2 = S2 / Mch / Mch;

    Val_domain integPx(syst.give_val_def("intPx")()(ndom - 1));
    double Px = space.get_domain(ndom - 1)->integ(integPx, OUTER_BC);

    Val_domain integPy(syst.give_val_def("intPy")()(ndom - 1));
    double Py = space.get_domain(ndom - 1)->integ(integPy, OUTER_BC);

    Val_domain integPz(syst.give_val_def("intPz")()(ndom - 1));
    double Pz = space.get_domain(ndom - 1)->integ(integPz, OUTER_BC);

    std::cout << "=======================================" << endl;
    std::cout << FORMAT << "Iter: " << ite << std::endl << FORMAT << "Error: " << conv << "\n";
    std::cout << FORMAT << "Omega: " << bconfig(GOMEGA) << endl;
    std::cout << FORMAT << "Axis: " << bconfig(COM) << endl;
    std::cout << FORMAT << "Padm: " << "[" << Px << ", " << Py << ", " << Pz << "]\n\n";

    std::cout << FORMAT << "NS-Mb: " << baryonic_massNS << "[" << bconfig(MB, BCO1) << "]\n";
    std::cout << FORMAT << "NS-Madm_ql: " << ql_massNS << "[" << bconfig(MADM, BCO1) << "]\n";
    std::cout << FORMAT << "NS-R: " << rmin << " " << rmax << std::endl << FORMAT << "NS-S: " << S1 << std::endl;
    std::cout << FORMAT << "NS-Chi: " << chi1 << "[" << bconfig(CHI, BCO1) << "]\n";
    std::cout << FORMAT << "NS-qlChi: " << chiql1 << std::endl;
    std::cout << FORMAT << "NS-Omega: " << bconfig(OMEGA, BCO1) << "\n\n";

    std::cout << FORMAT << "BH-Mirr: " << Mirr << "[" << bconfig(MIRR, BCO2) << "]\n";
    std::cout << FORMAT << "BH-Mch: " << Mch << "[" << bconfig(MCH, BCO2) << "]\n";
    std::cout << FORMAT << "BH-R: " << r_bh << std::endl;
    std::cout << FORMAT << "BH-S: " << S2 << std::endl;
    std::cout << FORMAT << "BH-Chi: " << chi2 << "[" << bconfig(CHI, BCO2) << "]\n";
    std::cout << FORMAT << "BH-Omega: " << bconfig(OMEGA, BCO2) << std::endl;
    std::cout.flags(f);
    std::cout << "=======================================" << endl;
} // end print_diagnostics

template <typename eos_t>
inline void bhns_setup_boosted_3d(kadath_config<BCO_NS_INFO>& NSconfig,
                                  kadath_config<BCO_BH_INFO>& BHconfig, kadath_config<BIN_INFO>& bconfig)
{
    // symmetric BH-NS superposition — shared body, instantiated with the
    // symmetric space + domain types (the only difference from the nosym variant).
    bhns_setup_boosted_3d_impl<eos_t, Space_spheric_adapted, Space_adapted_bh, Space_bhns,
                               Domain_shell_outer_adapted, Domain_shell_inner_adapted,
                               Domain_shell_outer_homothetic>(NSconfig, BHconfig, bconfig);
}

} // namespace Kadath
