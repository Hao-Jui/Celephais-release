#pragma once

#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Config/config_three_body.hpp"
#include "For_Kadath/Array/index.hpp"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "Hydro/EOS.hh"

#include "For_Kadath/IO/be_file_source.hpp"

#include "Apps/Diagnostics/configured_eos.hpp"
#include "mpi.h"
#include "Apps/Startup/solver_startup.hpp"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

namespace Kadath
{
class Space_spheric_adapted_nosym;
} // namespace Kadath

namespace KadathApps
{
namespace xz_snapshot_detail
{
// Collocation points whose |y| falls below this threshold are treated as
// lying in the xz-plane. The threshold is generous because collocation nodes
// are not guaranteed to land exactly on y == 0 in every domain.
constexpr double xz_plane_tolerance = 1e-10;

inline std::filesystem::path xz_snapshot_output_path()
{
    const char* runtime_root = std::getenv("HOME_CELEPHAIS");
    const std::filesystem::path repo_root = (runtime_root != nullptr && runtime_root[0] != '\0')
                                                ? std::filesystem::path{runtime_root}
                                                : std::filesystem::path{CELEPHAIS_ROOT_DIR};
    const std::filesystem::path output_dir = repo_root / "matlab" / "snapshot";
    std::filesystem::create_directories(output_dir);

    return output_dir / "xz_snapshot.dat";
}

template <typename space_t>
constexpr bool has_bns_domains_v = requires(const space_t& space) {
    space.NS1;
    space.ADAPTED1;
    space.NS2;
    space.ADAPTED2;
};

template <typename space_t>
constexpr bool has_bhns_domains_v = requires(const space_t& space) {
    space.NS;
    space.ADAPTEDNS;
    space.BH;
};

template <typename space_t>
Kadath::Vector zero_sacra_velocity_cov(space_t& space, Kadath::Base_tensor& basis)
{
    Kadath::Vector velocity_cov(space, COV, basis);
    for (int i = 1; i <= 3; ++i) {
        velocity_cov.set(i).annule_hard();
    }
    velocity_cov.std_base();
    return velocity_cov;
}

template <typename space_t, typename config_t>
Kadath::Vector binary_sacra_velocity_cov(space_t& space, config_t& bconfig,
                                         Kadath::Base_tensor& basis, Kadath::Scalar& conf,
                                         Kadath::Scalar& logh, Kadath::Scalar& phi)
{
    if constexpr (!has_bns_domains_v<space_t> && !has_bhns_domains_v<space_t>) {
        return zero_sacra_velocity_cov(space, basis);
    } else {
        using namespace Kadath;

        const int ndom = space.get_nbr_domains();
        Metric_flat fmet(space, basis);
        CoordFields<space_t> cfields(space);
        vec_ary_t coord_vectors{default_binary_vector_ary(space)};

        const double xo = bco_utils::get_center(space, ndom - 1);
        double xc1 = 0.0;
        double xc2 = 0.0;
        if constexpr (has_bns_domains_v<space_t>) {
            xc1 = bco_utils::get_center(space, space.NS1);
            xc2 = bco_utils::get_center(space, space.NS2);
        } else {
            xc1 = bco_utils::get_center(space, space.NS);
            xc2 = bco_utils::get_center(space, space.BH);
        }
        update_fields(cfields, coord_vectors, {}, xo, xc1, xc2);

        System_of_eqs syst(space, 0, ndom - 1);
        fmet.set_system(syst, "f");

        syst.add_cst("P", conf);
        syst.add_cst("H", logh);
        syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
        syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
        syst.add_cst("mpx", *coord_vectors[to_int(coord_vector::BCO2_ROTx)]);
        syst.add_cst("mpz", *coord_vectors[to_int(coord_vector::BCO2_ROTz)]);
        syst.add_cst("omes1", bconfig(OMEGA, BCO1));
        syst.add_cst("angs1", bconfig(DEG, BCO1) * std::acos(-1.) / 180.);
        syst.add_cst("omes2", bconfig(OMEGA, BCO2));
        syst.add_cst("angs2", bconfig(DEG, BCO2) * std::acos(-1.) / 180.);
        syst.add_def("m1^i = cos(angs1) * mmz^i + sin(angs1) * mmx^i");
        syst.add_def("m2^i = cos(angs2) * mpz^i + sin(angs2) * mpx^i");

        if constexpr (has_bns_domains_v<space_t>) {
            for (int d = space.NS1; d <= space.ADAPTED1; ++d)
                syst.add_def(d, "s^i = omes1 * m1^i");
            for (int d = space.NS2; d <= space.ADAPTED2; ++d)
                syst.add_def(d, "s^i = omes2 * m2^i");
        } else {
            for (int d = space.NS; d <= space.ADAPTEDNS; ++d)
                syst.add_def(d, "s^i = omes1 * m1^i");
        }

        syst.add_cst("velocPot", phi);
        for (int d = 0; d < ndom; ++d) {
            bool star_domain = false;
            if constexpr (has_bns_domains_v<space_t>) {
                star_domain = (d <= space.ADAPTED1) ||
                              (d >= space.NS2 && d <= space.ADAPTED2);
            } else {
                star_domain = (d <= space.ADAPTEDNS);
            }

            if (star_domain)
                syst.add_def(d, "eta_i = D_i velocPot + P^4 * s_i");
            else
                syst.add_def(d, "eta_i = D_i velocPot");
        }

        syst.add_def("h = exp(H)");
        syst.add_def("spatCpntfourVelCov_i = eta_i / h");
        return Vector(syst.give_val_def("spatCpntfourVelCov"));
    }
}

template <typename space_t, typename config_t>
Kadath::Vector ns_sacra_velocity_cov(space_t& space, config_t& bconfig,
                                     Kadath::Base_tensor& basis, Kadath::Scalar& conf,
                                     Kadath::Scalar& lapse, Kadath::Vector& shift,
                                     Kadath::Scalar& logh)
{
    using namespace Kadath;

    const int ndom = space.get_nbr_domains();
    Metric_flat fmet(space, basis);
    CoordFields<space_t> cfields(space);
    vec_ary_t coord_vectors{default_co_vector_ary(space)};
    const double xo = bco_utils::get_center(space, 0);
    update_fields_co(cfields, coord_vectors, {}, xo);

    Scalar Omg(space);
    Omg = bconfig(OMEGA);
    Omg.std_base();

    System_of_eqs syst(space, 0, ndom - 1);
    fmet.set_system(syst, "f");

    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);
    syst.add_cst("H", logh);
    syst.add_cst("Omg", Omg);
    // NS_nosym solves tilt the spin axis by DEG (degrees) in the xz-plane and
    // rotate rigidly about mm^i = cos(angs)*mmz^i + sin(angs)*mmx^i
    // (NS_3D_XCTS_nosym stage_helper/uniform_rot_stage); the snapshot velocity
    // must use the same generator or a tilted dataset gets ux/uy/uz for a
    // z-axis rotation. Sym spaces keep the z generator unchanged.
    if constexpr (std::is_same_v<space_t, Kadath::Space_spheric_adapted_nosym>) {
        double deg = bconfig(DEG);
        if (std::isnan(deg))
            deg = 0.;
        const double spin_axis_angle = deg * std::acos(-1.) / 180.;
        syst.add_cst("angs", spin_axis_angle);
        syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
        syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
        syst.add_def("mm^i = cos(angs) * mmz^i + sin(angs) * mmx^i");
        syst.add_def("omega^i = bet^i + Omg * mm^i");
    } else {
        syst.add_cst("mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
        syst.add_def("omega^i = bet^i + Omg * mg^i");
    }

    for (int d = 0; d < ndom; ++d) {
        if (d == 0 || d == 1) {
            syst.add_def(d, "U^i = omega^i / N");
            syst.add_def(d, "Usquare = P^4 * U_i * U^i");
            syst.add_def(d, "Wsquare = 1. / (1. - Usquare)");
            syst.add_def(d, "W = sqrt(Wsquare)");
            syst.add_def(d, "spatCpntfourVelCov_i = U_i * P^4 * W");
        } else {
            syst.add_def(d, "spatCpntfourVelCov_i = 0 * bet_i");
        }
    }

    return Vector(syst.give_val_def("spatCpntfourVelCov"));
}
} // namespace xz_snapshot_detail


struct GrTraits
{
    static constexpr bool has_varscal = false;
    static constexpr bool has_xiscal = false;
    // Header fragments inserted after the SACRA velocity columns. Empty for GR
    // so the header reads "... ux uy uz".
    static const char* varscal_header_fragment() { return ""; }
    static const char* xiscal_header_fragment() { return ""; }

    template <typename config_t, typename space_t>
    static void convert_loaded_scalar_slot(const config_t&, space_t&, Kadath::BeFileSource&,
                                           Kadath::Scalar&, Kadath::Scalar&)
    {
    }
};


// Binary compact-object snapshot (Space_bin_ns / Space_bin_ns_nosym /
// Space_bhns / Space_bhns_nosym). Loads the field set saved by the binary XCTS
// stages (conf, lapse, shift, logh, phi[, scalar]) and emits, for every
// xz-plane collocation point, the columns:
//   dom x y z P N H rho phi bet_x bet_y bet_z ux uy uz [varscal [xiScal]]
// with rho evaluated from the EOS at the local enthalpy and ux/uy/uz matching
// the covariant spatial four-velocity components emitted by 2sacra.
template <typename space_t, typename FieldTraits>
int xz_snapshot_binary_main(int argc, char** argv)
{
    using namespace Kadath;
    using xz_snapshot_detail::xz_plane_tolerance;

    KadathApps::init_mpi(argc, argv);

    return KadathApps::guarded_run([&] {
        if (argc < 2) {
            KADATH_THROW(
                "Usage: ./xz_snapshot /<path>/<ID base name>.toml|.dat [xz|all]  "
                "e.g. ./xz_snapshot converged.toml xz");
        }

        const bool xz_only = (argc < 3) || (std::string{argv[2]} != "all");

        // Load the configuration. Accept either the .toml or the .dat path.
        const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<BIN_INFO> bconfig{ifilename};

        // Reconstruct the space and the solved fields from the binary dataset.
        // The field set and order match the final converged save in the binary
        // XCTS stages: conf, lapse, shift, logh, phi[, scalar slot[, Sweight]].
        const std::string spacein = bconfig.space_filename();
        BeFileSource ff1(spacein);
        space_t space(ff1);
        Scalar conf(space, ff1);  // P : conformal factor
        Scalar lapse(space, ff1); // N : lapse
        Vector shift(space, ff1); // bet : shift (Cartesian basis, 3 components)
        Scalar logh(space, ff1);  // H : log enthalpy
        Scalar phi(space, ff1);   // phi : velocity potential

        Scalar scalar_slot(space); 
        if constexpr (FieldTraits::has_varscal) {
            scalar_slot = Scalar(space, ff1);
        }
        Scalar xiscal_slot(space); 

        const int ndom = space.get_nbr_domains();
        if constexpr (FieldTraits::has_varscal) {
            FieldTraits::convert_loaded_scalar_slot(bconfig, space, ff1, scalar_slot, xiscal_slot);
        }
        Base_tensor basis(shift.get_basis());
        Vector velocity_cov =
            xz_snapshot_detail::binary_sacra_velocity_cov(space, bconfig, basis, conf, logh, phi);

        const std::filesystem::path output_path = xz_snapshot_detail::xz_snapshot_output_path();

        std::ofstream out{output_path};
        if (!out) {
            KADATH_THROW("xz_snapshot: cannot open output file: " + output_path.string());
        }

        std::cout << "xz_snapshot: writing to " << output_path << "\n";

        out << "% dom x y z P N H rho phi bet_x bet_y bet_z ux uy uz"
            << FieldTraits::varscal_header_fragment() << FieldTraits::xiscal_header_fragment()
            << "\n";
        out << std::scientific << std::setprecision(16);

        // EOS initialisation — needed to evaluate density at each point.
        const double h_cut = bconfig.eos<double>(HCUT, BCO1);
        const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
        const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);

        // Per-domain output loop, templated on the EOS type so that
        // EOS<eos_t, eos_var_t::DENSITY>::get() resolves at compile time.
        auto write_domains = [&]<class eos_t>() {
            for (int d = 0; d < ndom; ++d) {
                const Domain* dom = space.get_domain(d);

                const Val_domain cart_x = dom->get_cart(1);
                const Val_domain cart_y = dom->get_cart(2);
                const Val_domain cart_z = dom->get_cart(3);

                Index pos(dom->get_nbr_points());
                do {
                    const double x = cart_x(pos);
                    const double y = cart_y(pos);
                    const double z = cart_z(pos);

                    if (xz_only && std::fabs(y) > xz_plane_tolerance) {
                        continue;
                    }

                    const double val_P     = conf(d)(pos);
                    const double val_N     = lapse(d)(pos);
                    const double val_H     = logh(d)(pos);
                    const double val_rho   = EOS<eos_t, eos_var_t::DENSITY>::get(std::exp(val_H));
                    const double val_phi   = phi(d)(pos);
                    const double val_bet_x = shift(1)(d)(pos);
                    const double val_bet_y = shift(2)(d)(pos);
                    const double val_bet_z = shift(3)(d)(pos);
                    const double val_ux    = velocity_cov(1)(d)(pos);
                    const double val_uy    = velocity_cov(2)(d)(pos);
                    const double val_uz    = velocity_cov(3)(d)(pos);

                    out << d << ' ' << x << ' ' << y << ' ' << z
                        << ' ' << val_P << ' ' << val_N << ' ' << val_H
                        << ' ' << val_rho << ' ' << val_phi
                        << ' ' << val_bet_x << ' ' << val_bet_y << ' ' << val_bet_z
                        << ' ' << val_ux << ' ' << val_uy << ' ' << val_uz;
                    if constexpr (FieldTraits::has_varscal) {
                        out << ' ' << scalar_slot(d)(pos);
                    }
                    if constexpr (FieldTraits::has_xiscal) {
                        out << ' ' << xiscal_slot(d)(pos);
                    }
                    out << '\n';
                } while (pos.inc());
            }
        };

        using namespace Kadath::Margherita;
        if (eos_type == "Cold_PWPoly") {
            EOS<Cold_PWPoly, eos_var_t::PRESSURE>::init(eos_file, h_cut);
            write_domains.template operator()<Cold_PWPoly>();
        } else if (eos_type == "Cold_Table") {
            KadathApps::init_configured_cold_table(bconfig, BCO1);
            write_domains.template operator()<Cold_Table>();
        } else {
            KADATH_THROW("xz_snapshot: unrecognised EOS type: " + eos_type);
        }

        out.close();
        std::cout << "xz_snapshot: done.\n";

        MPI_Finalize();
    });
}

// Isolated neutron-star snapshot (Space_spheric_adapted /
// Space_spheric_adapted_nosym). Loads the field set saved by the isolated NS
// stages (conf, lapse, shift, logh[, scalar]); phi is loaded only when the
// PHI field flag is set (binary-boosted output) and is otherwise treated as
// identically zero. Emits, for every xz-plane collocation point, the columns:
//   dom x y z P N H rho phi bet_x bet_y bet_z ux uy uz [varscal [xiScal]]
template <typename space_t, typename FieldTraits>
int xz_snapshot_ns_main(int argc, char** argv)
{
    using namespace Kadath;
    using xz_snapshot_detail::xz_plane_tolerance;

    KadathApps::init_mpi(argc, argv);

    return KadathApps::guarded_run([&] {
        if (argc < 2) {
            KADATH_THROW(
                "Usage: ./xz_snapshot /<path>/<ID base name>.toml|.dat [xz|all]  "
                "e.g. ./xz_snapshot converged.toml xz");
        }

        const bool xz_only = (argc < 3) || (std::string{argv[2]} != "all");

        // Load the configuration. Accept either the .toml or the .dat path.
        const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<BCO_NS_INFO> bconfig{ifilename};

        // Reconstruct the space and the solved fields from the isolated-NS dataset.
        const std::string spacein = bconfig.space_filename();
        BeFileSource ff1(spacein);
        space_t space(ff1);
        Scalar conf(space, ff1);    // P : conformal factor
        Scalar lapse(space, ff1);   // N : lapse
        Vector shift(space, ff1);   // bet : shift (Cartesian basis, 3 components)
        Scalar logh(space, ff1);    // H : log enthalpy

        // phi is only present when the binary_boost_stage final output was saved.
        // norot_stage and uniform_rot_stage are corotating / non-irrotational
        // models and do not save phi; treat it as zero in those cases.
        Scalar phi(space);
        if (bconfig.field(PHI)) {
            phi = Scalar(space, ff1);
        } else {
            phi.annule_hard();
            phi.std_base();
        }

        Scalar scalar_slot(space); 
        if constexpr (FieldTraits::has_varscal) {
            scalar_slot = Scalar(space, ff1);
        }
        Scalar xiscal_slot(space); 

        const int ndom = space.get_nbr_domains();
        if constexpr (FieldTraits::has_varscal) {
            FieldTraits::convert_loaded_scalar_slot(bconfig, space, ff1, scalar_slot, xiscal_slot);
        }
        Base_tensor basis(shift.get_basis());
        Vector velocity_cov =
            xz_snapshot_detail::ns_sacra_velocity_cov(space, bconfig, basis, conf, lapse, shift, logh);

        const std::filesystem::path output_path = xz_snapshot_detail::xz_snapshot_output_path();

        std::ofstream out{output_path};
        if (!out) {
            KADATH_THROW("xz_snapshot: cannot open output file: " + output_path.string());
        }

        std::cout << "xz_snapshot: writing to " << output_path << "\n";

        out << "% dom x y z P N H rho phi bet_x bet_y bet_z ux uy uz"
            << FieldTraits::varscal_header_fragment() << FieldTraits::xiscal_header_fragment()
            << "\n";
        out << std::scientific << std::setprecision(16);

        // EOS initialisation — needed to evaluate density at each point.
        const double h_cut = bconfig.eos<double>(HCUT);
        const std::string eos_file = bconfig.eos<std::string>(EOSFILE);
        const std::string eos_type = bconfig.eos<std::string>(EOSTYPE);

        // Per-domain output loop, templated on the EOS type so that
        // EOS<eos_t, eos_var_t::DENSITY>::get() resolves at compile time.
        auto write_domains = [&]<class eos_t>() {
            for (int d = 0; d < ndom; ++d) {
                const Domain* dom = space.get_domain(d);

                // Cartesian coordinates of the collocation points in this domain.
                const Val_domain cart_x = dom->get_cart(1);
                const Val_domain cart_y = dom->get_cart(2);
                const Val_domain cart_z = dom->get_cart(3);

                Index pos(dom->get_nbr_points());
                do {
                    const double x = cart_x(pos);
                    const double y = cart_y(pos);
                    const double z = cart_z(pos);

                    if (xz_only && std::fabs(y) > xz_plane_tolerance) {
                        continue;
                    }

                    const double val_P     = conf(d)(pos);
                    const double val_N     = lapse(d)(pos);
                    const double val_H     = logh(d)(pos);
                    const double val_rho   = EOS<eos_t, eos_var_t::DENSITY>::get(std::exp(val_H));
                    const double val_phi   = phi(d)(pos);
                    const double val_bet_x = shift(1)(d)(pos);
                    const double val_bet_y = shift(2)(d)(pos);
                    const double val_bet_z = shift(3)(d)(pos);
                    const double val_ux    = velocity_cov(1)(d)(pos);
                    const double val_uy    = velocity_cov(2)(d)(pos);
                    const double val_uz    = velocity_cov(3)(d)(pos);

                    out << d << ' ' << x << ' ' << y << ' ' << z
                        << ' ' << val_P << ' ' << val_N << ' ' << val_H
                        << ' ' << val_rho << ' ' << val_phi
                        << ' ' << val_bet_x << ' ' << val_bet_y << ' ' << val_bet_z
                        << ' ' << val_ux << ' ' << val_uy << ' ' << val_uz;
                    if constexpr (FieldTraits::has_varscal) {
                        out << ' ' << scalar_slot(d)(pos);
                    }
                    if constexpr (FieldTraits::has_xiscal) {
                        out << ' ' << xiscal_slot(d)(pos);
                    }
                    out << '\n';
                } while (pos.inc());
            }
        };

        using namespace Kadath::Margherita;
        if (eos_type == "Cold_PWPoly") {
            EOS<Cold_PWPoly, eos_var_t::PRESSURE>::init(eos_file, h_cut);
            write_domains.template operator()<Cold_PWPoly>();
        } else if (eos_type == "Cold_Table") {
            KadathApps::init_configured_cold_table(bconfig);
            write_domains.template operator()<Cold_Table>();
        } else {
            KADATH_THROW("xz_snapshot: unrecognised EOS type: " + eos_type);
        }

        out.close();
        std::cout << "xz_snapshot: done.\n";

        MPI_Finalize();
    });
}

// Three-body snapshots keep the binary/NS column contract for downstream tools.
// There is no 2sacra velocity analogue for this static family, so ux/uy/uz are
// emitted as zero.
template <typename space_t, typename FieldTraits>
int xz_snapshot_three_body_main(int argc, char** argv)
{
    using namespace Kadath;
    using xz_snapshot_detail::xz_plane_tolerance;

    KadathApps::init_mpi(argc, argv);

    return KadathApps::guarded_run([&] {
        if (argc < 2) {
            KADATH_THROW(
                "Usage: ./xz_snapshot /<path>/<ID base name>.toml|.dat [xz|all]  "
                "e.g. ./xz_snapshot converged.toml xz");
        }

        const bool xz_only = (argc < 3) || (std::string{argv[2]} != "all");

        // Load the configuration. Accept either the .toml or the .dat path.
        const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<TRI_INFO> bconfig{ifilename};

        // Reconstruct the space and the solved fields from the three-body dataset.
        const std::string spacein = bconfig.space_filename();
        BeFileSource ff1(spacein);
        space_t space(ff1);
        Scalar conf(space, ff1);    // P : conformal factor
        Scalar lapse(space, ff1);   // N : lapse
        Vector shift(space, ff1);   // bet : shift (Cartesian basis, 3 components)
        Scalar logh(space, ff1);    // H : log enthalpy

        // The static three-body stage structurally saves only {P, N, bet, H} —
        // there is no velocity potential. Do NOT consult the PHI field flag (a
        // config copied from a binary/NS setup may carry a stale phi=true, which
        // would drive a read past EOF -> BeFileSource short read). phi is always
        // identically zero here; the column is kept for header parity with the
        // binary/NS snapshots that downstream MATLAB tooling expects.
        Scalar phi(space);
        phi.annule_hard();
        phi.std_base();

        Scalar scalar_slot(space); 
        if constexpr (FieldTraits::has_varscal) {
            scalar_slot = Scalar(space, ff1);
        }
        Scalar xiscal_slot(space); 

        const int ndom = space.get_nbr_domains();
        if constexpr (FieldTraits::has_varscal) {
            FieldTraits::convert_loaded_scalar_slot(bconfig, space, ff1, scalar_slot, xiscal_slot);
        }

        const std::filesystem::path output_path = xz_snapshot_detail::xz_snapshot_output_path();

        std::ofstream out{output_path};
        if (!out) {
            KADATH_THROW("xz_snapshot: cannot open output file: " + output_path.string());
        }

        std::cout << "xz_snapshot: writing to " << output_path << "\n";

        Base_tensor basis(shift.get_basis());
        Vector velocity_cov = xz_snapshot_detail::zero_sacra_velocity_cov(space, basis);

        out << "% dom x y z P N H rho phi bet_x bet_y bet_z ux uy uz"
            << FieldTraits::varscal_header_fragment() << FieldTraits::xiscal_header_fragment()
            << "\n";
        out << std::scientific << std::setprecision(16);

        // EOS initialisation — all three bodies share one EOS (BCO1 slot).
        const double h_cut = bconfig.eos<double>(HCUT, BCO1);
        const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
        const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);

        // Per-domain output loop, templated on the EOS type so that
        // EOS<eos_t, eos_var_t::DENSITY>::get() resolves at compile time.
        auto write_domains = [&]<class eos_t>() {
            for (int d = 0; d < ndom; ++d) {
                const Domain* dom = space.get_domain(d);

                // Cartesian coordinates of the collocation points in this domain.
                const Val_domain cart_x = dom->get_cart(1);
                const Val_domain cart_y = dom->get_cart(2);
                const Val_domain cart_z = dom->get_cart(3);

                Index pos(dom->get_nbr_points());
                do {
                    const double x = cart_x(pos);
                    const double y = cart_y(pos);
                    const double z = cart_z(pos);

                    if (xz_only && std::fabs(y) > xz_plane_tolerance) {
                        continue;
                    }

                    const double val_P     = conf(d)(pos);
                    const double val_N     = lapse(d)(pos);
                    const double val_H     = logh(d)(pos);
                    const double val_rho   = EOS<eos_t, eos_var_t::DENSITY>::get(std::exp(val_H));
                    const double val_phi   = phi(d)(pos);
                    const double val_bet_x = shift(1)(d)(pos);
                    const double val_bet_y = shift(2)(d)(pos);
                    const double val_bet_z = shift(3)(d)(pos);
                    const double val_ux    = velocity_cov(1)(d)(pos);
                    const double val_uy    = velocity_cov(2)(d)(pos);
                    const double val_uz    = velocity_cov(3)(d)(pos);

                    out << d << ' ' << x << ' ' << y << ' ' << z
                        << ' ' << val_P << ' ' << val_N << ' ' << val_H
                        << ' ' << val_rho << ' ' << val_phi
                        << ' ' << val_bet_x << ' ' << val_bet_y << ' ' << val_bet_z
                        << ' ' << val_ux << ' ' << val_uy << ' ' << val_uz;
                    if constexpr (FieldTraits::has_varscal) {
                        out << ' ' << scalar_slot(d)(pos);
                    }
                    if constexpr (FieldTraits::has_xiscal) {
                        out << ' ' << xiscal_slot(d)(pos);
                    }
                    out << '\n';
                } while (pos.inc());
            }
        };

        using namespace Kadath::Margherita;
        if (eos_type == "Cold_PWPoly") {
            EOS<Cold_PWPoly, eos_var_t::PRESSURE>::init(eos_file, h_cut);
            write_domains.template operator()<Cold_PWPoly>();
        } else if (eos_type == "Cold_Table") {
            KadathApps::init_configured_cold_table(bconfig, BCO1);
            write_domains.template operator()<Cold_Table>();
        } else {
            KADATH_THROW("xz_snapshot: unrecognised EOS type: " + eos_type);
        }

        out.close();
        std::cout << "xz_snapshot: done.\n";

        MPI_Finalize();
    });
}

} // namespace KadathApps
