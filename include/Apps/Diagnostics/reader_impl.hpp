#pragma once

#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Config/config_three_body.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "For_Kadath/Utilities/levi_civita.hpp"
#include "Hydro/EOS.hh"

#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/System_of_eqs/system_dof_record.hpp"

#include "Apps/Startup/solver_startup.hpp"
#include "Apps/Formalism/GR/syst_init.ipp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// Forward declarations of the SCAFFOLD space types. The BNS thin main only
// includes kadath_bin_ns.hpp (Space_bin_ns); the co_traits specializations for
// the other binary/single spaces are declared here so the header is well-formed
// in every translation unit, but their member-accessing bodies are only
// instantiated by an app that includes the matching Space header and selects
// that Space type. Forming a specialization needs only an (incomplete) class
// declaration; completeness is required solely at instantiation time.
namespace Kadath
{
class Space_bin_ns_nosym;
class Space_bhns;
class Space_bhns_nosym;
class Space_bin_bh;
class Space_spheric_adapted_nosym;
class Space_adapted_bh;
class Space_three_body;
} // namespace Kadath

namespace KadathApps
{

// Conversion from solar mass to km, shared by every reader variant.
inline constexpr double reader_M2km = 1.4769994423016508;

// Optional app-local diagnostic extension for the axisymmetric reader.
// Production readers use this no-op; validation-only targets may provide a
// compatible type without placing experiment-specific output in this header.
struct NoReaderValidation
{
    template <typename... Args>
    static void emit_2d(Args&&...)
    {
    }
};

// Per-domain resolution along one spectral axis (0 = radial r, 1 = polar
// theta, 2 = azimuthal phi), comma-separated in domain order. Each axis is
// listed independently because per-direction AMR (p-refinement) can raise the
// radial, polar, and azimuthal collocation counts of a domain independently,
// so a single domain-0 tuple no longer captures the grid.
template <typename space_t>
std::string reader_resolution_by_domain(const space_t& space, int axis)
{
    std::ostringstream output;
    for (int d = 0; d < space.get_nbr_domains(); ++d) {
        if (d != 0)
            output << ", ";
        output << space.get_domain(d)->get_nbr_points()(axis);
    }
    return output.str();
}

template <typename space_t>
std::string reader_radial_resolution_by_domain(const space_t& space)
{
    return reader_resolution_by_domain(space, 0);
}

template <typename space_t>
std::string reader_polar_resolution_by_domain(const space_t& space)
{
    return reader_resolution_by_domain(space, 1);
}

template <typename space_t>
std::string reader_azimuthal_resolution_by_domain(const space_t& space)
{
    return reader_resolution_by_domain(space, 2);
}

// Format the system DOF stamped into a dataset by the solver (read via
// read_system_dof_from_file): the count, or "n/a" for a dataset saved before the
// DOF stamp existed (read_system_dof_from_file returns -1).
inline std::string reader_format_system_dof(int system_dof)
{
    return system_dof >= 0 ? std::to_string(system_dof) : std::string("n/a (pre-DOF-stamp dataset)");
}

// ---------------------------------------------------------------------------
// Compact-object descriptor and per-space traits.
// ---------------------------------------------------------------------------

// One compact object inside a (possibly binary) space. nucleus_dom / adapted_dom
// are the starting domain indices of that object's nucleus and adapted shells;
// center_x is the x-coordinate of its center (filled at runtime from
// bco_utils::get_center); label is the banner fragment (with surrounding spaces
// matching the original readers, e.g. " NS_MINUS ").
struct CompactObject
{
    enum class Kind
    {
        NS,
        BH
    };

    Kind kind;
    int nucleus_dom;
    int adapted_dom;
    double center_x;
    const char* label;
};

// Primary template: every concrete space specializes this to enumerate its
// compact objects and to declare whether it is a binary (object count > 1).
template <typename space_t>
struct co_traits;

// Space_bin_ns / Space_bin_ns_nosym: two neutron stars (NS_MINUS, NS_PLUS).
template <typename space_t>
struct bin_ns_co_traits
{
    static constexpr bool is_binary = true;

    static std::vector<CompactObject> objects(const space_t& space)
    {
        return {
            {CompactObject::Kind::NS, space.NS1, space.ADAPTED1,
             bco_utils::get_center(space, space.NS1), " NS_MINUS "},
            {CompactObject::Kind::NS, space.NS2, space.ADAPTED2,
             bco_utils::get_center(space, space.NS2), " NS_PLUS "},
        };
    }
};

template <>
struct co_traits<Kadath::Space_bin_ns> : bin_ns_co_traits<Kadath::Space_bin_ns>
{
};
template <>
struct co_traits<Kadath::Space_bin_ns_nosym> : bin_ns_co_traits<Kadath::Space_bin_ns_nosym>
{
};

// Space_three_body: three neutron stars on the nested bispheric layout. The
// suffix order matches the TRI_INFO branch convention (1 = parent-level direct
// body, 2/3 = the child-aggregate pair). Kept a template (like
// bin_ns_co_traits) so the member body only instantiates in a TU that includes
// the Space header.
template <typename space_t>
struct three_body_co_traits
{
    static constexpr bool is_binary = false;

    static std::vector<CompactObject> objects(const space_t& space)
    {
        return {
            {CompactObject::Kind::NS, space.BODY, space.ADAPTED_BODY,
             bco_utils::get_center(space, space.BODY), " NS1 "},
            {CompactObject::Kind::NS, space.CHILD1, space.ADAPTED_CHILD1,
             bco_utils::get_center(space, space.CHILD1), " NS2 "},
            {CompactObject::Kind::NS, space.CHILD2, space.ADAPTED_CHILD2,
             bco_utils::get_center(space, space.CHILD2), " NS3 "},
        };
    }
};

template <>
struct co_traits<Kadath::Space_three_body> : three_body_co_traits<Kadath::Space_three_body>
{
};

// SCAFFOLD: Space_bhns / Space_bhns_nosym — neutron star + black hole. Structured
// here for the binary-BHNS port; not yet wired to an app this turn.
template <typename space_t>
struct bhns_co_traits
{
    static constexpr bool is_binary = true;

    static std::vector<CompactObject> objects(const space_t& space)
    {
        return {
            {CompactObject::Kind::NS, space.NS, space.ADAPTEDNS,
             bco_utils::get_center(space, space.NS), " Neutron Star "},
            {CompactObject::Kind::BH, space.BH, space.ADAPTEDBH,
             bco_utils::get_center(space, space.BH), " Black Hole "},
        };
    }
};

template <>
struct co_traits<Kadath::Space_bhns> : bhns_co_traits<Kadath::Space_bhns>
{
};
template <>
struct co_traits<Kadath::Space_bhns_nosym> : bhns_co_traits<Kadath::Space_bhns_nosym>
{
};

// SCAFFOLD: Space_bin_bh — two black holes. Structured for the binary-BH port;
// not yet wired to an app this turn.
template <>
struct co_traits<Kadath::Space_bin_bh>
{
    static constexpr bool is_binary = true;

    static std::vector<CompactObject> objects(const Kadath::Space_bin_bh& space)
    {
        return {
            {CompactObject::Kind::BH, space.BH1, space.BH1 + 1,
             bco_utils::get_center(space, space.BH1), " Black Hole 1 "},
            {CompactObject::Kind::BH, space.BH2, space.BH2 + 1,
             bco_utils::get_center(space, space.BH2), " Black Hole 2 "},
        };
    }
};

// Single isolated objects. The nucleus is domain 0 and the adapted shell domain
// 1, mirroring the isolated NS / BH readers. Isolated NS readers use
// reader_single_output; the distinct horizon geometry uses reader_single_bh_output.
template <typename space_t>
struct single_co_traits
{
    static constexpr bool is_binary = false;
};

template <>
struct co_traits<Kadath::Space_spheric_adapted> : single_co_traits<Kadath::Space_spheric_adapted>
{
};
template <>
struct co_traits<Kadath::Space_spheric_adapted_nosym>
    : single_co_traits<Kadath::Space_spheric_adapted_nosym>
{
};
template <>
struct co_traits<Kadath::Space_adapted_bh> : single_co_traits<Kadath::Space_adapted_bh>
{
};


// GR: no scalar field. Universal scalar diagnostic rows are printed as n/a at
// the call sites.
struct GrFormalism
{
    static constexpr bool has_varscal = false;

    template <typename config_t, typename space_t>
    static void convert_loaded_scalar_slot(const config_t&, space_t&, BeFileSource&, Scalar&)
    {
    }
};


inline std::string reader_na_text()
{
    return "n/a";
}

inline std::string reader_format_universal_number(double value)
{
    std::ostringstream out;
    out << std::setprecision(5) << std::scientific << std::showpos << value;
    return out.str();
}

inline bool reader_domain_is_excluded(int domain, const std::vector<int>& excluded_domains)
{
    for (int excluded : excluded_domains) {
        if (domain == excluded)
            return true;
    }
    return false;
}

inline std::vector<int> reader_diagnostic_domains(int ndom,
                                                  const std::vector<int>& excluded_domains = {})
{
    std::vector<int> domains;
    domains.reserve(static_cast<std::size_t>(ndom));
    for (int d = 0; d < ndom; ++d) {
        if (!reader_domain_is_excluded(d, excluded_domains))
            domains.push_back(d);
    }
    return domains;
}

inline std::array<double, 2> reader_2d_boundary_min_max(
    const Scalar& field, int domain, int radial_index)
{
    const Dim_array& points = field.get_domain(domain)->get_nbr_points();
    Index position(points);
    Index boundary_position(points);
    boundary_position.set(0) = radial_index;
    for (int axis = 1; axis < points.get_ndim(); ++axis)
        boundary_position.set(axis) = position(axis);

    double minimum = field(domain)(boundary_position);
    double maximum = minimum;
    do {
        boundary_position.set(0) = radial_index;
        for (int axis = 1; axis < points.get_ndim(); ++axis)
            boundary_position.set(axis) = position(axis);
        const double value = field(domain)(boundary_position);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    } while (position.inc());
    return {minimum, maximum};
}

inline double reader_hamiltonian_l2(System_of_eqs& syst,
                                    const std::vector<int>& diagnostic_domains)
{
    double hcon_L2 = 0.;
    for (int d : diagnostic_domains) {
        // Proper-volume L2: integrate Hcon^2 against the PHYSICAL volume element
        // sqrt(gamma) = P^6 (conformally flat XCTS, gamma_ij = P^4 f_ij), not the
        // bare coordinate measure that integ_volume() carries.
        syst.add_def(d, "Hcon2pv = Hcon2 * P^6");
        hcon_L2 += syst.give_val_def("Hcon2pv")()(d).integ_volume();
    }
    return std::sqrt(hcon_L2);
}

inline double reader_hamiltonian_l2(System_of_eqs& syst, int ndom)
{
    return reader_hamiltonian_l2(syst, reader_diagnostic_domains(ndom));
}

inline double reader_hamiltonian_l2_axisym(
    System_of_eqs& syst, const std::vector<int>& diagnostic_domains)
{
    double hcon_l2 = 0.;
    for (int d : diagnostic_domains) {
        // Space_polar_adapted integrates the meridional half-plane. Restore the
        // suppressed azimuthal integral (2*pi = 4piG/2 in G=1 units) so this is
        // the same physical proper-volume L2 used by the 3D readers.
        syst.add_def(d, "Hcon2pv2d = Hcon2 * P^6 * 4piG / 2.");
        hcon_l2 += syst.give_val_def("Hcon2pv2d")()(d).integ_volume();
    }
    return std::sqrt(hcon_l2);
}

inline std::array<double, 2> reader_xcts_constraint_rms(System_of_eqs& syst,
                                                       const std::vector<int>& diagnostic_domains)
{
    // Proper-volume-weighted RMS of the physical Hamiltonian / momentum
    // constraints (Hdiag = Hcon/(8 P^5), Mdiag): average the squared constraint
    // over the PHYSICAL volume sqrt(gamma) = P^6 (gamma_ij = P^4 f_ij), i.e.
    // sqrt( int q^2 P^6 dV / int P^6 dV ), rather than the grid-dependent
    // unweighted per-point mean.
    double h_num = 0., m_num = 0., proper_vol = 0.;
    for (int d : diagnostic_domains) {
        syst.add_def(d, "Hdiag2pv = Hdiag2 * P^6");
        syst.add_def(d, "Mdiag2pv = Mdiag2 * P^6");
        syst.add_def(d, "PvolElem = P^6");
        h_num      += syst.give_val_def("Hdiag2pv")()(d).integ_volume();
        m_num      += syst.give_val_def("Mdiag2pv")()(d).integ_volume();
        proper_vol += syst.give_val_def("PvolElem")()(d).integ_volume();
    }
    const double inv_vol = (proper_vol > 0.) ? 1. / proper_vol : 0.;
    return {std::sqrt(h_num * inv_vol), std::sqrt(m_num * inv_vol)};
}

inline std::array<double, 2> reader_xcts_constraint_rms(System_of_eqs& syst,
                                                       int ndom)
{
    return reader_xcts_constraint_rms(syst, reader_diagnostic_domains(ndom));
}

inline void reader_add_gr_xcts_constraint_rms_defs(System_of_eqs& syst, int domain,
                                                  bool has_matter,
                                                  const std::string& energy_source = {},
                                                  const std::string& momentum_source = {})
{
    std::string hdiag =
        "Hdiag = (D^i D_i P + A_ij * A^ij / P^7 / 8";
    if (has_matter) {
        syst.add_def(domain, ("Ediag = " + energy_source).c_str());
        syst.add_def(domain, ("Jdiag^i = " + momentum_source).c_str());
        hdiag += " + 4piG / 2. * P^5 * Ediag";
    }
    hdiag += ") / 8. / P^5";

    syst.add_def(domain, hdiag.c_str());
    syst.add_def(domain, "Hdiag2 = Hdiag * Hdiag");

    std::string mdiag =
        "Rbeta^i = D_j D^j bet^i + D^i D_j bet^j / 3. "
        "- 2. * A^ij * D_j Ntilde";
    if (has_matter)
        mdiag += " - 4. * 4piG * N * P^4 * Jdiag^i";
    syst.add_def(domain, mdiag.c_str());
    syst.add_def(domain, "Mdiag2 = Rbeta_i * Rbeta^i / 4. / N^2 / P^8");
}

// ---------------------------------------------------------------------------
// Shared XCTS diagnostic core.
// ---------------------------------------------------------------------------
//
// Binary and isolated NS readers differ in geometry and in their global
// observables, but the gravity fields, formalism sources, per-domain XCTS
// residuals, and final constraint rows are the same. Keep that common layer
// here; the two reader bodies only supply their geometry-specific velocity and
// matter definitions before calling it.

template <typename Formalism, typename config_t, typename space_t>
void reader_add_formalism_field_constants(const config_t& bconfig,
                                          System_of_eqs& syst,
                                          space_t& space,
                                          const Scalar& scalar_slot)
{
    if constexpr (Formalism::has_varscal)
        syst.add_cst("varscal", scalar_slot);

}

template <typename Formalism, typename config_t>
void reader_add_formalism_xcts_sources(const config_t& bconfig,
                                       System_of_eqs& syst)
{
    (void)bconfig;
    (void)syst;
}

template <typename Formalism>
void reader_add_xcts_constraint_defs(System_of_eqs& syst, int domain,
                                     bool has_matter)
{
    if constexpr (std::is_same_v<Formalism, GrFormalism>) {
        Kadath::gr_xcts::add_xcts_field_equations(
            syst, domain, has_matter, /*has_shift=*/true);
        syst.add_def(domain, "Hcon = eqP");
        reader_add_gr_xcts_constraint_rms_defs(
            syst, domain, has_matter,
            "rho * h * Wsquare - press",
            "rho * h * Wsquare * U^i");
    }



    syst.add_def(domain, "Hcon2 = Hcon * Hcon");
}

struct ReaderXctsConstraintNorms
{
    std::string hamiltonian_l2;
    std::string hamiltonian_rms;
    std::string momentum_rms;
};

template <typename Formalism>
ReaderXctsConstraintNorms reader_xcts_constraint_norms(
    System_of_eqs& syst, const std::vector<int>& diagnostic_domains)
{
    ReaderXctsConstraintNorms norms{
        reader_format_universal_number(
            reader_hamiltonian_l2(syst, diagnostic_domains)),
        reader_na_text(), reader_na_text()};

    if constexpr (std::is_same_v<Formalism, GrFormalism>) {
        const auto [h_rms, m_rms] =
            reader_xcts_constraint_rms(syst, diagnostic_domains);
        norms.hamiltonian_rms = reader_format_universal_number(h_rms);
        norms.momentum_rms = reader_format_universal_number(m_rms);
    }
    return norms;
}

inline void reader_print_xcts_constraint_norms(
    const ReaderXctsConstraintNorms& norms)
{
    const auto label_format = []() -> std::ostream& {
        return std::cout << std::setw(25) << std::right
                         << std::setprecision(5) << std::scientific
                         << std::showpos;
    };
    label_format() << "Hamiltonian H (L2) = " << norms.hamiltonian_l2 << '\n';
    label_format() << "Hamiltonian H (RMS) = " << norms.hamiltonian_rms << '\n';
    label_format() << "Momentum M (RMS) = " << norms.momentum_rms << '\n';
}

template <typename Formalism, typename config_t>
std::string reader_gravity_theory(const config_t& bconfig)
{
    (void)bconfig;
    return "GR";
}


template <typename Formalism, typename config_t>
void reader_print_universal_gravity_block(const config_t& bconfig)
{
    std::cout << std::setw(25) << std::right << std::setprecision(12) << std::fixed << std::showpos
              << "Gravity Theory = " << reader_gravity_theory<Formalism>(bconfig) << "\n";
}

// ---------------------------------------------------------------------------
// Axisymmetric XCTS diagnostic branch.
// ---------------------------------------------------------------------------
//
// Space_polar_adapted stores the rotational shift as the regular scalar
// amplitude brsint, rather than the 3D Cartesian vector bet^i. Keep its field
// and equation reduction separate, but feed the resulting norms and observables
// into the same report schema and universal gravity helpers used above.

template <typename Formalism, typename config_t, typename space_t>
void reader_add_2d_formalism_defs(const config_t& bconfig,
                                  System_of_eqs& syst,
                                  space_t& space,
                                  const Scalar& scalar_slot)
{
    if constexpr (Formalism::has_varscal)
        syst.add_cst("varscal", scalar_slot);

}

template <typename Formalism>
void reader_add_2d_hamiltonian_def(System_of_eqs& syst, int domain,
                                   bool has_matter)
{
    if constexpr (std::is_same_v<Formalism, GrFormalism>) {
        if (has_matter) {
            syst.add_def(domain,
                         "eqP = delta*lap(P) + Asquare/P^7/8.*delta "
                         "+ 4piG/2.*P^5*Etilde");
        } else {
            syst.add_def(domain, "eqP = lap(P) + Asquare/P^7/8.");
        }
    }


    syst.add_def(domain, "Hcon = eqP");
    syst.add_def(domain, "Hcon2 = Hcon*Hcon");
}

template <class eos_t, typename space_t, typename Formalism>
void reader_binary_output(kadath_config<BIN_INFO> bconfig)
{
    using namespace Kadath;
    using namespace Kadath::Margherita;

    using vec_d = std::vector<double>;
    using ary_d = std::array<double, 2>;
    using ary_i = std::array<int, 2>;

    if (std::isnan(bconfig.set(MADM, BCO1)) || std::isnan(bconfig.set(MADM, BCO2))) {
        std::cerr << "Missing \"Madm\" in config file\n"
                     "Setting to quasi-local \"Madm\"! \n";
        bconfig.set(MADM, BCO1) = bconfig(QLMADM, BCO1);
        bconfig.set(MADM, BCO2) = bconfig(QLMADM, BCO2);
    }
    if (std::isnan(bconfig.set(COMY))) bconfig.set(COMY) = 0.;

    // read domain decomposition and fields from binary file
    std::string in_spacefile = bconfig.space_filename();
    BeFileSource fich(in_spacefile);

    space_t space(fich);
    Scalar conf(space, fich);
    Scalar lapse(space, fich);
    Vector shift(space, fich);
    Scalar logh(space, fich);
    Scalar phi(space, fich);

    Scalar scalar_slot(space);
    if constexpr (Formalism::has_varscal) {
        scalar_slot = Scalar(space, fich);
    }

    // define basic fields
    Base_tensor basis(shift.get_basis());
    Metric_flat fmet(space, basis);

    // coordinate dependent fields
    CoordFields<space_t> cfields(space);

    // enumerate the compact objects of this space (2 NS / NS+BH / 2 BH)
    std::vector<CompactObject> objects = co_traits<space_t>::objects(space);

    // get center of each star
    const std::array<double, 2> x_nuc{objects[0].center_x, objects[1].center_x};
    if constexpr (Formalism::has_varscal) {
        Formalism::convert_loaded_scalar_slot(bconfig, space, fich, scalar_slot);
    }

    // coordinate origin
    double xo = 0;

    // coordinate dependent vector fields
    vec_ary_t coord_vectors = default_binary_vector_ary(space);

    // vector field for expansion factor
    Vector CART(space, CON, basis);
    CART = cfields.cart();

    // initialize all coordinate dependent fields with the given centers
    update_fields(cfields, coord_vectors, {}, xo, x_nuc[0], x_nuc[1]);

    // compute the location of the maximum densities
    std::array<double, 2> x_max;
    {
        System_of_eqs syst_H(space);
        fmet.set_system(syst_H, "f");

        syst_H.add_cst("H", logh);
        syst_H.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);

        syst_H.add_def("dH = (ex^i * D_i H) / H");
        syst_H.add_def("dH2 = ex^i * D_i dH");

        Scalar dHdx = syst_H.give_val_def("dH");
        Scalar dHdx2 = syst_H.give_val_def("dH2");

        for (int j : {0, 1}) {
            double xc = x_nuc[j];
            double err = 1;

            Point absol(3);
            absol.set(1) = xc;
            absol.set(2) = 0;
            absol.set(3) = 0;

            while (err > 1e-14) {
                xc = xc - dHdx.val_point(absol) / dHdx2.val_point(absol);
                absol.set(1) = xc;
                err = std::abs(dHdx.val_point(absol));
            }
            x_max[j] = xc;
        }
    }

    // setup a system of equations to compute derived quantities
    // using the spectral representation of the fields
    int ndom = space.get_nbr_domains();
    const int system_dof = read_system_dof_from_file(in_spacefile);
    System_of_eqs syst(space);
    cout << "Number of domains: " << ndom << "\n"
         << "System DOF: " << reader_format_system_dof(system_dof) << "\n"
         << "NS1 contains: " << space.NS1 << " " << space.ADAPTED1 << "\n"
         << "NS2 contains: " << space.NS2 << " " << space.ADAPTED2 << endl;

    // Outer radius of every domain in a star's block, plus the last one
    // repeated as the bispheric matching radius r_bisph (mirrors reader_bhns_output).
    cout << "Bounds: [rin, rmid, rout, shells] + r_bisph" << endl;
    auto print_co_bounds = [&](const char* label, int dom_first, int dom_last) {
        cout << label << ": [";
        double r_bisph = 0.;
        for (int dom = dom_first; dom <= dom_last; ++dom) {
            const double rout = bco_utils::get_radius(space.get_domain(dom), OUTER_BC);
            cout << (dom > dom_first ? " " : "") << rout;
            r_bisph = rout;
        }
        cout << "] + " << r_bisph << endl;
    };
    print_co_bounds("NS1", space.NS1, space.NS2 - 1);
    print_co_bounds("NS2", space.NS2, space.OUTER - 1);

    // conformally flat background metric
    fmet.set_system(syst, "f");

    // setup the eos operators
    Param p;
    syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
    syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
    syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    syst.add_ope("dHdlnrho", &EOS<eos_t, eos_var_t::DHDRHO>::action, &p);

    // constants in the system
    syst.add_cst("4piG", 4.0 * M_PI);

    syst.add_cst("Madm1", bconfig(MADM, BCO1));
    syst.add_cst("Mb1", bconfig(MB, BCO1));
    syst.add_cst("chi1", bconfig(CHI, BCO1));

    syst.add_cst("Madm2", bconfig(MADM, BCO2));
    syst.add_cst("Mb2", bconfig(MB, BCO2));
    syst.add_cst("chi2", bconfig(CHI, BCO2));

    // these names are hardcoded in the coord_fields.hpp!
    syst.add_cst("mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
    syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
    syst.add_cst("mpx", *coord_vectors[to_int(coord_vector::BCO2_ROTx)]);
    syst.add_cst("mpz", *coord_vectors[to_int(coord_vector::BCO2_ROTz)]);

    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);
    syst.add_cst("ez", *coord_vectors[to_int(coord_vector::EZ)]);

    syst.add_cst("sm", *coord_vectors[to_int(coord_vector::S_BCO1)]);
    syst.add_cst("sp", *coord_vectors[to_int(coord_vector::S_BCO2)]);
    syst.add_cst("einf", *coord_vectors[to_int(coord_vector::S_INF)]);

    // "center of mass" and orbital angular frequency parameter
    syst.add_cst("xaxis", bconfig(COM));
    syst.add_cst("yaxis", bconfig(COMY));
    syst.add_cst("ome", bconfig(GOMEGA));

    syst.add_cst("omes1", bconfig(OMEGA, BCO1));
    syst.add_cst("angs1", bconfig(DEG, BCO1) * std::acos(-1.) / 180.);
    syst.add_cst("omes2", bconfig(OMEGA, BCO2));
    syst.add_cst("angs2", bconfig(DEG, BCO2) * std::acos(-1.) / 180.);

    syst.add_def("mm^i = cos(angs1) * mmz^i + sin(angs1) * mmx^i ");
    syst.add_def("mp^i = cos(angs2) * mpz^i + sin(angs2) * mpx^i ");

    // the actual fields representing the solution
    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);
    syst.add_cst("H", logh);
    syst.add_cst("phi", phi);
    reader_add_formalism_field_constants<Formalism>(
        bconfig, syst, space, scalar_slot);

    // different definitions for the velocity
    // between the corotation and the irrotational / spinning cases
    if (bconfig.control(COROT_BIN)) {
        syst.add_cst("omes1", bconfig(OMEGA, BCO1));
        syst.add_cst("omes2", bconfig(OMEGA, BCO2));
    } else {
        for (int d = space.NS1; d <= space.ADAPTED1; ++d) {
            syst.add_def(d, "s^i  = omes1 * mm^i");
            syst.add_def(d, "eta_i  = D_i phi + P^4 * s_i");
        }
        for (int d = space.NS2; d <= space.ADAPTED2; ++d) {
            syst.add_def(d, "s^i  = omes2 * mp^i");
            syst.add_def(d, "eta_i  = D_i phi + P^4 * s_i");
        }
    }

    // common definitions
    syst.add_def("NP     = P*N");
    syst.add_def("Ntilde = N / P^6");

    // definitions of derived matter quantities
    syst.add_def("h = exp(H)");
    syst.add_def("press = press(h)");
    syst.add_def("eps = eps(h)");
    syst.add_def("rho = rho(h)");
    syst.add_def("dHdlnrho = dHdlnrho(h)");
    syst.add_def("delta = h - eps - 1.");

    // orbital vector field and resulting "total" shift
    // check for ADOT so we don't get errors.
    std::string eccstr{};
    if (!std::isnan(bconfig.set(ADOT))) {
        syst.add_cst("adot", bconfig(ADOT));
        syst.add_cst("r", CART);
        syst.add_def("comr^i = r^i - xaxis * ex^i + yaxis * ey^i");
        eccstr += " + adot * comr^i";
    }
    std::string omegastr{"omega^i = bet^i + ome * Morb^i" + eccstr};
    syst.add_def("Morb^i = mg^i + xaxis * ey^i");
    syst.add_def(omegastr.c_str());

    // normalized derivative of the enthalpy along binary axis
    syst.add_def("dH = (ex^i * D_i H) / H");
    syst.add_def("dH2 = ex^i * D_i dH");

    // conformal extrinsic curvature
    syst.add_def("A^ij   = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / Ntilde");
    reader_add_formalism_xcts_sources<Formalism>(bconfig, syst);

    // integrant of the ADM linear momentum at infinity
    syst.add_def(ndom - 1, "intPx = A_i^j * ex_j * einf^i");
    syst.add_def(ndom - 1, "intPy = A_i^j * ey_j * einf^i");
    syst.add_def(ndom - 1, "intPz = A_i^j * ez_j * einf^i");

    // definitions from https://arxiv.org/abs/1506.01689
    syst.add_def(ndom - 1, "intCOMx = 3. / 2. / 4piG * P^4 * ex_i * einf^i");
    syst.add_def(ndom - 1, "intCOMy = 3. / 2. / 4piG * P^4 * ey_i * einf^i");
    syst.add_def(ndom - 1, "intCOMz = 3. / 2. / 4piG * P^4 * ez_i * einf^i");

    // quasi-local spin angular momentum surface integral integrants for both stars
    syst.add_def(space.ADAPTED1 + 1, "intS = A_ij * mm^i * sm^j / 2. / 4piG");
    syst.add_def(space.ADAPTED2 + 1, "intS = A_ij * mp^i * sp^j / 2. / 4piG");

    for (int d = 0; d < ndom; d++) {
        const bool is_star_domain = (d >= space.NS1 && d <= space.ADAPTED1) ||
                                    (d >= space.NS2 && d <= space.ADAPTED2);
        if (!is_star_domain) {
            reader_add_xcts_constraint_defs<Formalism>(
                syst, d, /*has_matter=*/false);
        } else {
            if (bconfig.control(COROT_BIN)) {
                syst.add_def(d, "U^i    = omega^i / N");
                syst.add_def(d, "Usquare= P^4 * U_i * U^i");
                syst.add_def(d, "Wsquare= 1 / (1 - Usquare)");
                syst.add_def(d, "W      = sqrt(Wsquare)");
                syst.add_def(d, "firstint = log(h * N / W)");
            } else {
                syst.add_def(d, "Wsquare= eta^i * eta_i / h^2 / P^4 + 1.");
                syst.add_def(d, "W      = sqrt(Wsquare)");
                syst.add_def(d, "U^i    = eta^i / P^4 / h / W");
                syst.add_def(d, "Usquare= P^4 * U_i * U^i");
                syst.add_def(d, "V^i    = N * U^i - omega^i");
                syst.add_def(d, "firstint = log(h * N / W + D_i phi * V^i)");
            }

            syst.add_def(d, "intMb  = P^6 * rho * W");
            syst.add_def(d, "intM   = - D_i D^i P * 2. / 4piG");

            syst.add_def(d, "intH  = P^6 * H * W");
            syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
            syst.add_def(d, "Stilde = 3 * press * delta + (Etilde + press * delta) * Usquare");
            syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");
            reader_add_xcts_constraint_defs<Formalism>(
                syst, d, /*has_matter=*/true);
        }
    }

    // ADM and Komar integrants at infinity
    syst.add_def(ndom - 1, "intJ = multr(A_ij * Morb^j * einf^i) / 2. / 4piG");
    syst.add_def(ndom - 1, "Madm = -dr(P) * 2 / 4piG");
    syst.add_def(ndom - 1, "Mk   =  dr(N) / 4piG");
    // integrant to compute proper area over a spherical surface
    syst.add_def("intMsq = P^4");

    // BNS component quantities
    ary_i nuc_doms{space.NS1, space.NS2};
    ary_i adapted_doms{space.ADAPTED1, space.ADAPTED2};
    ary_i shells{};
    ary_d areal_r;
    ary_d central_logh;
    ary_d central_rho;
    ary_d central_P;
    ary_d central_dHdx;
    ary_d central_euler;
    ary_d H_int;
    ary_d ql_madm;
    ary_d ql_spin;
    ary_d madm;
    ary_d mbs;
    ary_d xcom;

    std::vector<ary_d> r_extrema;
    std::vector<vec_d> mb_distro;

    // compute everything for both stars
    for (int i : {0, 1}) {
        int dom = adapted_doms[i];
        shells[i] = dom - nuc_doms[i] - 1;

        // compute area and areal radius from that
        double A = space.get_domain(dom + 1)->integ(syst.give_val_def("intMsq")()(dom + 1), INNER_BC);
        areal_r[i] = sqrt(A / 4. / acos(-1.));

        // point defining the center of the star
        Point ns_c(3);
        ns_c.set(1) = x_nuc[i];
        ns_c.set(2) = 0;
        ns_c.set(3) = 0;

        // central matter quantities
        central_logh[i] = logh.val_point(ns_c);
        central_rho[i] = EOS<eos_t, eos_var_t::DENSITY>::get(std::exp(central_logh[i]));
        central_P[i] = EOS<eos_t, eos_var_t::PRESSURE>::get(std::exp(central_logh[i]));
        central_dHdx[i] = syst.give_val_def("dH")().val_point(ns_c);
        central_euler[i] = syst.give_val_def("firstint")().val_point(ns_c);

        // volume integrated quantities, log enthalpy, quasi-local ADM mass and baryonic mass
        H_int[i] = 0;
        vec_d ql_m{};
        vec_d baryonic_mass{};

        for (int d = nuc_doms[i]; d <= adapted_doms[i]; ++d) {
            H_int[i] += syst.give_val_def("intH")()(d).integ_volume();
            ql_m.push_back(syst.give_val_def("intM")()(d).integ_volume());
            baryonic_mass.push_back(syst.give_val_def("intMb")()(d).integ_volume());
        }

        mb_distro.push_back(baryonic_mass);
        mbs[i] = std::accumulate(baryonic_mass.begin(), baryonic_mass.end(), 0.);

        ql_madm[i] = std::accumulate(ql_m.begin(), ql_m.end(), 0.);
        // get quasi-local spin as surface integral outside of the star
        ql_spin[i] = space.get_domain(dom + 1)->integ(syst.give_val_def("intS")()(dom + 1), OUTER_BC);

        // the correct ADM mass (at infinite separation), is given from the single star solution
        // and thus fixed
        madm[i] = bconfig(MADM, i);

        // minimal and maximal (coordinate) radius across the adapted surface
        r_extrema.push_back(bco_utils::get_rmin_rmax(space, dom));

        // center-of-mass shifted center
        xcom[i] = x_nuc[i] + bconfig(COM);
    }

    // binary quantities
    double adm_inf = space.get_domain(ndom - 1)->integ(syst.give_val_def("Madm")()(ndom - 1), OUTER_BC);
    double komar = space.get_domain(ndom - 1)->integ(syst.give_val_def("Mk")()(ndom - 1), OUTER_BC);
    double Jinf = space.get_domain(ndom - 1)->integ(syst.give_val_def("intJ")()(ndom - 1), OUTER_BC);
    double Px = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPx")()(ndom - 1), OUTER_BC);
    double Py = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPy")()(ndom - 1), OUTER_BC);
    double Pz = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPz")()(ndom - 1), OUTER_BC);

    // center of mass defined like in https://arxiv.org/abs/1506.01689
    double COMx = space.get_domain(ndom - 1)->integ(syst.give_val_def("intCOMx")()(ndom - 1), OUTER_BC) / adm_inf;
    double COMy = space.get_domain(ndom - 1)->integ(syst.give_val_def("intCOMy")()(ndom - 1), OUTER_BC) / adm_inf;
    double COMz = space.get_domain(ndom - 1)->integ(syst.give_val_def("intCOMz")()(ndom - 1), OUTER_BC) / adm_inf;

    // ADM mass at infinite separation
    double Minf = std::accumulate(madm.begin(), madm.end(), 0.);
    // binding energy
    double e_bind = adm_inf - Minf;
    // end binary quantities

    const auto constraint_norms = reader_xcts_constraint_norms<Formalism>(
        syst, reader_diagnostic_domains(ndom));

#define FORMAT1 std::setw(25) << std::right << std::setprecision(12) << std::fixed << std::showpos
#define FORMAT std::setw(25) << std::right << std::setprecision(5) << std::scientific << std::showpos

    const double M2km = reader_M2km;

    // helper lambda to print shell radii
    auto print_shells = [&](int dom_min, int dom_max) {
        int cnt = 1;
        for (int i = dom_min; i < dom_max; ++i) {
            std::string shell{"SHELL" + std::to_string(cnt) + " = "};
            std::cout << FORMAT1 << shell << "["
                      << bco_utils::get_radius(space.get_domain(i), INNER_BC) << ", "
                      << bco_utils::get_radius(space.get_domain(i), EQUI) << "]" << std::endl;
            cnt++;
        }
    };

    // helper to print contents of a collection of values
    auto print_vec = [&](auto& vec) {
        for (auto& e : vec) {
            std::cout << e << ",";
        }
    };

    // string to identify the stars
    std::array<std::string, 2> ns_str{objects[0].label, objects[1].label};
    ary_i bounds{space.NS2 - 1, space.OUTER - 1};
    std::string header(22, '#');
    for (int i = 0; i <= 1; ++i) {
        std::cout << header + ns_str[i] + header + "\n";
        std::cout << FORMAT1 << "Center_COM = " << "(" << xcom[i] << ", 0, 0)\n"
                  << FORMAT1 << "Coord R_IN = " << bco_utils::get_radius(space.get_domain(nuc_doms[i]), EQUI) << '\n'
                  << FORMAT1 << "Coord R = " << "[" << r_extrema[i][0] << "," << r_extrema[i][1] << "] ("
                  << "[" << r_extrema[i][0] * M2km << "," << r_extrema[i][1] * M2km << "] km)\n";
        print_shells(adapted_doms[i] + 1, bounds[i]);
        std::cout << FORMAT1 << "Coord R_OUT = " << bco_utils::get_radius(space.get_domain(adapted_doms[i] + 1), EQUI) << "\n"
                  << FORMAT1 << "Areal R = " << areal_r[i] << " [" << areal_r[i] * M2km << "km]\n"
                  // baryonic mass and fractions per stellar domain covering the star
                  << FORMAT1 << "Baryonic Mass = " << mbs[i] << " (";
        print_vec(mb_distro[i]);
        std::cout << ")\n"
                  << FORMAT1 << "Isolated ADM Mass = " << bconfig(MADM, i) << "\n"
                  << FORMAT1 << "Quasi-local Madm = " << ql_madm[i]
                  << " Diff:" << std::fabs(1. - ql_madm[i] / bconfig(MADM, i)) << std::endl
                  // quasi-local spin angular momentum
                  << FORMAT1 << "Quasi-local S = " << ql_spin[i] << std::endl
                  // dimensionless spin (constant, given by the imported single star!)
                  // angular frequency paramter of the star, describing the magnitude of the spin component of the velocity field
                  << FORMAT1 << "Chi = " << ql_spin[i] / bconfig(MADM, i) / bconfig(MADM, i) << " [" << bconfig(CHI, i) << "]\n"
                  << FORMAT1 << "Omega = " << bconfig(OMEGA, i) << std::endl
                  // location of the maximal density (and compared to the nucleus' center)
                  << FORMAT1 << "x(max(Density)) = " << x_max[i] << " (" << (x_nuc[i] - x_max[i]) / x_nuc[i] << ")\n"
                  << FORMAT << "Central Density = " << central_rho[i] << std::endl
                  << FORMAT << "Central log(h) = " << central_logh[i] << std::endl
                  << FORMAT << "Central Pressure = " << central_P[i] << std::endl;
        std::string central_varscal = reader_na_text();
        if constexpr (Formalism::has_varscal) {
            Point ns_c(3);
            ns_c.set(1) = x_nuc[i];
            ns_c.set(2) = 0;
            ns_c.set(3) = 0;
            central_varscal = reader_format_universal_number(scalar_slot.val_point(ns_c));
        }
        std::cout << FORMAT << "Central varscal = " << central_varscal << std::endl;
        std::cout << FORMAT << "Central dlog(h)/dx = " << central_dHdx[i] << std::endl
                  // central values of the Euler constant, log enthalpy and its derivative
                  << FORMAT1 << "Central Euler Constant = " << central_euler[i] << std::endl
                  // integrated log enthalpy
                  << FORMAT1 << "Integrated log(h) = " << H_int[i] << "\n\n";
    }

    std::cout << "Outer bounds: "
              << bco_utils::get_radius(space.get_domain(space.OUTER + 5), INNER_BC)
              << std::endl;
    auto M1 = bconfig(MADM, BCO1);
    auto M2 = bconfig(MADM, BCO2);
    if (M2 > M1) std::swap(M1, M2);
    // mass ratio, ratio of the ADM masses at infinity
    std::cout << header + " Binary " + header + '\n'
              << FORMAT1 << "Radial RES = " << reader_radial_resolution_by_domain(space) << '\n'
              << FORMAT1 << "Theta RES = " << reader_polar_resolution_by_domain(space) << '\n'
              << FORMAT1 << "Phi RES = " << reader_azimuthal_resolution_by_domain(space) << '\n';
    reader_print_universal_gravity_block<Formalism>(bconfig);
    std::string varscal_outer_min = reader_na_text();
    std::string varscal_outer_max = reader_na_text();
    if constexpr (Formalism::has_varscal) {
        auto [scalar_min, scalar_max] = bco_utils::get_field_min_max(scalar_slot, ndom - 1, OUTER_BC);
        varscal_outer_min = reader_format_universal_number(scalar_min);
        varscal_outer_max = reader_format_universal_number(scalar_max);
    }
    std::cout << FORMAT << "Outer varscal = " << "[" << varscal_outer_min << ", " << varscal_outer_max << "]"
              << std::endl;
    std::cout << FORMAT1 << "Q = " << M2 / M1 << std::endl
              // distance between the stellar centers, defined at the setup stage
              << FORMAT1 << std::setprecision(2) << "Separation = " << bconfig(DIST) << " [" << bconfig(DIST) / Minf << "] (" << bconfig(DIST) * M2km << "km)" << std::endl
              // orbital angular frequency parameter
              << FORMAT1 << "Orbital Omega = " << bconfig(GOMEGA) << std::endl
              // Komar and ADM mass of the binary
              << FORMAT1 << "Komar mass = " << komar << std::endl
              << FORMAT1 << "Adm mass = " << adm_inf << ", Diff: " << fabs(adm_inf - komar) / (adm_inf + komar) << std::endl
              << FORMAT1 << "Total mass (Minf) = " << Minf << std::endl
              // ADM angular momentum of the binary
              << FORMAT1 << "Adm moment. = " << Jinf << std::endl
              // binding energy, defined by the gravitational mass difference at finite separation
              << FORMAT << "Binding energy = " << e_bind << std::endl
              // dimensionless orbital frequency
              << FORMAT << "Minf * Ome = " << Minf * bconfig(GOMEGA) << std::endl
              // dimensionless binding energy
              << FORMAT << "E_b / Minf = " << e_bind / Minf << std::endl
              // ADM linear momentum
              << FORMAT << "Px = " << Px << std::endl
              << FORMAT << "Py = " << Py << std::endl
              << FORMAT << "Pz = " << Pz << std::endl
              // "center of mass" defined by a vanishing ADM momentum at infinity
              // With an analytical estimate of the center of mass estimate from Osokine+
              << FORMAT1 << "COMx = " << bconfig(COM) << ", A-COMx = " << COMx << std::endl
              << FORMAT1 << "COMy = " << bconfig(COMY) << ", A-COMy = " << COMy << std::endl
              << FORMAT1 << "A-COMz = " << COMz << std::endl;
    reader_print_xcts_constraint_norms(constraint_norms);

#undef FORMAT1
#undef FORMAT
}

// Binary reader entry point. The EOS-type dispatch (Cold_PWPoly / Cold_Table)
// lives in the thin per-app main (mirroring the original BNS main lines 33-73),
// which selects eos_t and forwards to reader_binary_output<eos_t, ...>. This
// function is the single instantiation site for a chosen (eos_t, space_t,
// Formalism) triple, kept thin so the thin main reads like the original.
template <class eos_t, typename space_t, typename Formalism>
int reader_binary_main(kadath_config<BIN_INFO> bconfig)
{
    reader_binary_output<eos_t, space_t, Formalism>(bconfig);
    return 0;
}

// ===========================================================================
// Per-object diagnostics for the binary BH branch (SCAFFOLD).
// ===========================================================================
//
// Ported from apps/BHNS/src/reader.cpp (the BH branch, lines ~206-220 and the
// intS/intMsq defs). It expects the BHNS-style definitions
// (intS2/intMsq/intAsq) to already be present in `syst`. Structured here for the
// binary-BHNS / binary-BH port; not yet wired to an app this turn. Emits the
// " Black Hole " block (ql_spin, mirr, mch, areal_r, lapse/conf horizon
// extrema). No EOS / matter.
template <typename space_t, typename config_t, typename FmtFn1>
void reader_bh_object_block(const CompactObject& obj, space_t& space, System_of_eqs& syst,
                            const Scalar& lapse, const Scalar& conf, config_t& bconfig,
                            int bco_index, FmtFn1&& fmt1)
{
    using namespace Kadath;
    const double M2km = reader_M2km;
    std::string header(22, '#');

    const int adapted = obj.adapted_dom;
    const int bh_nucleus = obj.nucleus_dom;

    double ql_spinbh = space.get_domain(adapted + 1)->integ(syst.give_val_def("intS2")()(adapted + 1), OUTER_BC);

    // mirr from the proper-area integral over the apparent horizon (BH+2 = the
    // inner-homothetic excision domain carrying the horizon inner BC, in the
    // BHNS layout — one domain per BH, not two).
    double mirrsq = space.get_domain(bh_nucleus + 2)->integ(syst.give_val_def("intMsq")()(bh_nucleus + 2), INNER_BC);
    double mirr = sqrt(mirrsq);
    double mch = sqrt(mirrsq + ql_spinbh * ql_spinbh / 4. / mirrsq);

    double rin_bh = bco_utils::get_radius(space.get_domain(bh_nucleus), EQUI);
    double r_bh = bco_utils::get_radius(space.get_domain(bh_nucleus + 1), EQUI);
    double rout_bh = bco_utils::get_radius(space.get_domain(space.OUTER - 1), EQUI);
    double A = space.get_domain(adapted + 1)->integ(syst.give_val_def("intAsq")()(adapted + 1), INNER_BC);
    double areal_rbh = sqrt(A);
    auto [lapsemin, lapsemax] = bco_utils::get_field_min_max(lapse, bh_nucleus + 2, INNER_BC);
    auto [confmin, confmax] = bco_utils::get_field_min_max(conf, bh_nucleus + 2, INNER_BC);
    double BH_x_com = obj.center_x + bconfig(COM);

    std::cout << header + obj.label + header + "\n"
              << fmt1() << "Center_COM = " << "(" << BH_x_com << ", 0, 0)\n"
              << fmt1() << "Coord R_IN = " << rin_bh << std::endl
              << fmt1() << "Coord R = " << r_bh << " [" << r_bh * M2km << "km]\n";
    {
        int cnt = 1;
        for (int i = bh_nucleus + 2; i < space.OUTER - 1; ++i) {
            std::string shell{"SHELL" + std::to_string(cnt) + " = "};
            Index pos(space.get_domain(i + 1)->get_radius().get_conf().get_dimensions());
            std::cout << fmt1() << shell << space.get_domain(i + 1)->get_radius()(pos) << std::endl;
            cnt++;
        }
    }
    std::cout << fmt1() << "Coord R_OUT = " << rout_bh << std::endl
              << fmt1() << "Areal R = " << areal_rbh << std::endl
              << fmt1() << " LAPSE = [" << lapsemin << ", " << lapsemax << "]\n"
              << fmt1() << " PSI = [" << confmin << ", " << confmax << "]\n"
              << fmt1() << "Mirr = " << mirr << "\n"
              << fmt1() << "Mch = " << mch << " [" << bconfig(MCH, bco_index) << "]\n"
              << fmt1() << "Chi = " << ql_spinbh / (mch * mch) << " [" << bconfig(CHI, bco_index) << "]\n"
              << fmt1() << "S = " << ql_spinbh << std::endl
              << fmt1() << "Omega = " << bconfig(OMEGA, bco_index) << "\n\n";
}

template <class eos_t, typename space_t, typename Formalism>
void reader_single_output(kadath_config<BCO_NS_INFO> bconfig)
{
    using namespace Kadath;
    using namespace Kadath::Margherita;

    constexpr double M2km = reader_M2km;
    constexpr double M2Hz = 2.029739818539300e+05 / 2. / M_PI;

    const std::string spacein = bconfig.space_filename();

    // load domain decomposition and fields from binary file
    BeFileSource ff1(spacein);
    space_t space(ff1);
    Scalar conf(space, ff1);
    Scalar lapse(space, ff1);
    Vector shift(space, ff1);
    Scalar logh(space, ff1);
    const bool has_phi = bconfig.field(PHI);
    std::unique_ptr<Scalar> phi;
    if (has_phi) phi = std::make_unique<Scalar>(space, ff1);
    Scalar scalar_slot(space);
    if constexpr (Formalism::has_varscal) {
        scalar_slot = Scalar(space, ff1);
    }

    Scalar Omg(space);
    Omg = bconfig(OMEGA);
    Omg.std_base();

    // number of domains, a cartesian type of basis and the flat bg metric
    int ndom = space.get_nbr_domains();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Metric_flat fmet(space, basis);
    if constexpr (Formalism::has_varscal) {
        Formalism::convert_loaded_scalar_slot(bconfig, space, ff1, scalar_slot);
    }

    // fields depending on the coords
    CoordFields<space_t> cf_generator(space);
    vec_ary_t coord_vectors{default_co_vector_ary(space)};

    // get origin of the system and initialize coordinate fields
    double xo = bco_utils::get_center(space, 0);
    update_fields_co(cf_generator, coord_vectors, {}, xo);

    // central values of the matter fields
    double loghc = bco_utils::get_boundary_val(0, logh, INNER_BC);
    double hc = std::exp(loghc);
    double nc = EOS<eos_t, eos_var_t::DENSITY>::get(hc);
    double pc = EOS<eos_t, eos_var_t::PRESSURE>::get(hc);

    // minimal and maximal radius of the adapted surface domain
    auto [rmin, rmax] = bco_utils::get_rmin_rmax(space, 1);
    // inner radius of the nucleus
    double rin1 = bco_utils::get_radius(space.get_domain(0), OUTER_BC);

    // setup the system of equations to define derived quantities
    System_of_eqs syst(space, 0, ndom - 1);

    // flat background metric
    fmet.set_system(syst, "f");

    // constants
    syst.add_cst("4piG", 4.0 * M_PI);
    syst.add_cst("PI", M_PI);

    syst.add_cst("Mb", bconfig(MB));
    syst.add_cst("chi", bconfig(CHI));
    syst.add_cst("Hc", loghc);
    syst.add_cst("ome", bconfig(OMEGA));
    syst.add_cst("Madm", bconfig(MADM));
    syst.add_cst("cstA", bconfig(CSTA));

    syst.add_cst("mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);
    syst.add_cst("ez", *coord_vectors[to_int(coord_vector::EZ)]);
    syst.add_cst("einf", *coord_vectors[to_int(coord_vector::S_INF)]);

    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);

    syst.add_cst("Omg", Omg);
    syst.add_cst("H", logh);
    reader_add_formalism_field_constants<Formalism>(
        bconfig, syst, space, scalar_slot);

    syst.add_def("NP = P*N");
    syst.add_def("Ntilde = N / P^6");

    if (std::isnan(bconfig.set(BVELY))) bconfig.set(BVELY) = 0.;

    // NS_nosym solves tilt the spin axis by DEG in the xz-plane and pin J about
    // that generator (NS_3D_XCTS_nosym uniform_rot_stage: mm^i = cos(angs)*mmz^i
    // + sin(angs)*mmx^i). Diagnostics must integrate about the SAME axis: with
    // the bare z generator a tilted dataset reads back only the projection
    // (Jadm = J cos deg, Chi likewise) and a wrong fluid velocity/Lorentz factor
    // (Mb drift). Sym spaces keep the z-axis generator unchanged.
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
        syst.add_def("omega^i = bet^i + Omg * mg^i");
    }

    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / Ntilde");
    reader_add_formalism_xcts_sources<Formalism>(bconfig, syst);
    syst.add_def(ndom - 1, "intPx = A_i^j * ex_j * einf^i");
    syst.add_def(ndom - 1, "intPy = A_i^j * ey_j * einf^i");
    syst.add_def(ndom - 1, "intPz = A_i^j * ez_j * einf^i");

    if constexpr (std::is_same_v<space_t, Kadath::Space_spheric_adapted_nosym>)
        syst.add_def(ndom - 1, "intJ = multr(A_ij * mm^j * einf^i) / 8. / PI");
    else
        syst.add_def(ndom - 1, "intJ = multr(A_ij * mg^j * einf^i) / 8. / PI");
    syst.add_def(ndom - 1, "intMadmalt = -dr(P) / 2 / PI");
    syst.add_def(ndom - 1, "intMadm = - einf^i * D_i P / 2 / PI");
    syst.add_def(ndom - 1, "intMk = (einf^i * D_i N - A_ij * einf^i * bet^j) / 4 / PI");

    syst.add_def("dH = (ex^i * D_i H) / H");

    Param p;
    syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
    syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
    syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    syst.add_def("h = exp(H)");
    syst.add_def("press = press(h)");
    syst.add_def("eps = eps(h)");
    syst.add_def("rho = rho(h)");
    syst.add_def("delta = h - eps - 1.");

    for (int d = 0; d < ndom; d++) {
        const bool has_matter = d == 0 || d == 1;
        if (has_matter) {
            syst.add_def(d, "U^i = omega^i / N");
            syst.add_def(d, "Usquare = P^4 * U_i * U^i");
            syst.add_def(d, "Wsquare = 1. / (1. - Usquare)");
            syst.add_def(d, "W = sqrt(Wsquare)");

            syst.add_def(d, "intMb = P^6 * rho * W");
            syst.add_def(d, "firstint = H + log(N) - log(W) - cstA^2 * (Omg-ome)^2 / 2.");
            syst.add_def(d, "intH  = P^6 * H * W");
            syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
            syst.add_def(d, "Stilde = 3 * press * delta + (Etilde + press * delta) * Usquare");
            syst.add_def(d, "ptilde^i = press * h * Wsquare * U^i");
        }
        reader_add_xcts_constraint_defs<Formalism>(syst, d, has_matter);
    }
    double Px = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPx")()(ndom - 1), OUTER_BC);
    double Py = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPy")()(ndom - 1), OUTER_BC);
    double Pz = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPz")()(ndom - 1), OUTER_BC);

    Val_domain integJ(syst.give_val_def("intJ")()(ndom - 1));
    double J = space.get_domain(ndom - 1)->integ(integJ, OUTER_BC);

    Val_domain integMadm(syst.give_val_def("intMadm")()(ndom - 1));
    double Madm = space.get_domain(ndom - 1)->integ(integMadm, OUTER_BC);

    Val_domain integMk(syst.give_val_def("intMk")()(ndom - 1));
    double Mk = space.get_domain(ndom - 1)->integ(integMk, OUTER_BC);

    double baryonic_mass = syst.give_val_def("intMb")()(0).integ_volume() +
                           syst.give_val_def("intMb")()(1).integ_volume();
    double integrated_logh = syst.give_val_def("intH")()(0).integ_volume() +
                             syst.give_val_def("intH")()(1).integ_volume();
    const auto constraint_norms = reader_xcts_constraint_norms<Formalism>(
        syst, reader_diagnostic_domains(ndom));

    syst.add_def("intMsq = P^4");
    double A = space.get_domain(2)->integ(syst.give_val_def("intMsq")()(2), INNER_BC);
    double AR = sqrt(A / 4. / acos(-1.));

    Point P(3); // [x, y, z] =[0, 0, 0]
    double central_dHdx = syst.give_val_def("dH")().val_point(P);
    double central_euler = syst.give_val_def("firstint")().val_point(P);
    double central_omg = bconfig(OMEGA);
    double equator_omg = bconfig(OMEGA);

#define FORMAT1 std::setw(25) << std::right << std::setprecision(12) << std::fixed << std::showpos
#define FORMAT std::setw(25) << std::right << std::setprecision(5) << std::scientific << std::showpos

    auto print_shells = [&](int dom_min, int dom_max) {
        int cnt = 1;
        for (int i = dom_min; i < dom_max; ++i) {
            std::string shell{"SHELL" + std::to_string(cnt) + " = "};
            std::cout << FORMAT1 << shell << bco_utils::get_radius(space.get_domain(i), OUTER_BC) << std::endl;
            cnt++;
        }
    };

    auto res_r = space.get_domain(0)->get_nbr_points()(0);
    auto res_t = space.get_domain(0)->get_nbr_points()(1);
    auto res_p = space.get_domain(0)->get_nbr_points()(2);

    std::string header(22, '#');
    std::cout << "Number of domains: " << ndom << "\n"
              << "System DOF: " << reader_format_system_dof(read_system_dof_from_file(spacein)) << "\n"
              << "NS1 contains: 0 1" << std::endl;

    std::cout << header + " Neutron Star " + header + "\n"
              << FORMAT1 << "Center_COM = " << "(" << xo << ", 0, 0)\n"
              << FORMAT1 << "Coord R_IN = " << rin1 << std::endl
              << FORMAT1 << "Coord R = " << "[" << rmin << "," << rmax << "] ("
              << "[" << rmin * M2km << "," << rmax * M2km << "] km)\n";
    print_shells(2, ndom - 2);
    std::cout << FORMAT1 << "Coord R_OUT = " << bco_utils::get_radius(space.get_domain(ndom - 2), OUTER_BC) << "\n"
              << FORMAT1 << "Areal R = " << AR << " [" << AR * M2km << "km]\n"
              << FORMAT1 << "Baryonic Mass = " << baryonic_mass << std::endl
              << FORMAT1 << "Isolated ADM Mass = " << bconfig(MADM) << "\n"
              << FORMAT1 << "Quasi-local Madm = " << Madm
              << " Diff:" << std::fabs(1. - Madm / bconfig(MADM)) << std::endl
              << FORMAT1 << "Quasi-local S = " << J << std::endl
              << FORMAT1 << "Chi = " << J / Madm / Madm << " [" << bconfig(CHI) << "]\n"
              << FORMAT1 << "Omega = " << central_omg << std::endl
              << FORMAT1 << "x(max(Density)) = " << xo << " (" << 0. << ")\n"
              << FORMAT << "Central Density = " << nc << std::endl
              << FORMAT << "Central log(h) = " << loghc << std::endl
              << FORMAT << "Central Pressure = " << pc << std::endl;
    std::string central_varscal = reader_na_text();
    std::string varscal_outer_min = reader_na_text();
    std::string varscal_outer_max = reader_na_text();
    if constexpr (Formalism::has_varscal) {
        central_varscal = reader_format_universal_number(scalar_slot.val_point(P));
        auto [scalar_min, scalar_max] = bco_utils::get_field_min_max(scalar_slot, ndom - 1, OUTER_BC);
        varscal_outer_min = reader_format_universal_number(scalar_min);
        varscal_outer_max = reader_format_universal_number(scalar_max);
    }
    std::cout << FORMAT << "Central varscal = " << central_varscal << std::endl
              << FORMAT << "Central dlog(h)/dx = " << central_dHdx << std::endl
              << FORMAT1 << "Central Euler Constant = " << central_euler << std::endl
              << FORMAT1 << "Integrated log(h) = " << integrated_logh << "\n\n";

    std::cout << header + " Single " + header + "\n"
              << FORMAT1 << std::fixed << "RES = " << "[" << res_r << "," << res_t << "," << res_p << "]\n";
    reader_print_universal_gravity_block<Formalism>(bconfig);
    std::cout << FORMAT << "Outer varscal = " << "[" << varscal_outer_min << ", " << varscal_outer_max << "]"
              << std::endl
              << FORMAT1 << "Velocity Potential = " << (has_phi ? "present" : "absent") << "\n"
              << FORMAT1 << "Central Omega = " << central_omg << " [" << bconfig(OMEGA) * M2Hz << "Hz]\n"
              << FORMAT1 << "Equator Omega = " << equator_omg << " [" << equator_omg * M2Hz << "Hz]\n"
              << FORMAT1 << "Komar mass = " << Mk
              << ", Diff: " << 2. * fabs(Madm - Mk) / (Madm + Mk) << std::endl
              << FORMAT1 << "Adm mass = " << Madm << " [" << bconfig(MADM) << "]\n"
              << FORMAT1 << "Adm moment. = " << J << std::endl
              << FORMAT << "Px = " << Px << std::endl
              << FORMAT << "Py = " << Py << std::endl
              << FORMAT << "Pz = " << Pz << std::endl;
    reader_print_xcts_constraint_norms(constraint_norms);

#undef FORMAT1
#undef FORMAT
}

template <class eos_t, typename space_t, typename Formalism>
int reader_single_main(kadath_config<BCO_NS_INFO> bconfig)
{
    reader_single_output<eos_t, space_t, Formalism>(bconfig);
    return 0;
}

// ===========================================================================
// Axisymmetric isolated neutron-star reader.
// ===========================================================================
//
// The polar space saves a scalar rotational-shift amplitude and has no
// azimuthal spectral axis. Its loader and XCTS reduction therefore branch here,
// while the labels, gravity block, and three constraint rows match the common
// isolated/binary reader contract.
template <class eos_t, typename space_t, typename Formalism,
          typename Validation = NoReaderValidation>
void reader_2d_output(kadath_config<BCO_NS_INFO> bconfig)
{
    using namespace Kadath;
    using namespace Kadath::Margherita;

    constexpr double M2km = reader_M2km;
    constexpr double M2Hz = 2.029739818539300e+05 / 2. / M_PI;

    const std::string spacein = bconfig.space_filename();
    BeFileSource source(spacein);
    space_t space(source);
    Scalar conf(space, source);
    Scalar lapse(space, source);
    Scalar shift(space, source);
    Scalar logh(space, source);
    Scalar Omg(space, source);
    Scalar scalar_slot(space);
    if constexpr (Formalism::has_varscal)
        scalar_slot = Scalar(space, source);
    if constexpr (Formalism::has_varscal)
        Formalism::convert_loaded_scalar_slot(bconfig, space, source, scalar_slot);

    const int ndom = space.get_nbr_domains();
    const int adapted_inner = space.ADAPTED_INNER;
    const int adapted_outer = space.ADAPTED_OUTER;
    const double loghc = bco_utils::get_boundary_val(0, logh, INNER_BC);
    const double hc = std::exp(loghc);
    const double nc = EOS<eos_t, eos_var_t::DENSITY>::get(hc);
    const double pc = EOS<eos_t, eos_var_t::PRESSURE>::get(hc);
    const auto [rmin, rmax] = bco_utils::get_rmin_rmax(space, adapted_outer);
    const double rin = bco_utils::get_radius(space.get_domain(0), OUTER_BC);
    const double rout =
        bco_utils::get_radius(space.get_domain(ndom - 2), OUTER_BC);

    System_of_eqs syst(space, 0, ndom - 1);
    syst.add_cst("4piG", 4. * M_PI);
    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("brsint", shift);
    syst.add_cst("H", logh);
    syst.add_cst("Omg", Omg);
    syst.add_def("NP = P*N");
    syst.add_def("Ntilde = N/P^6");
    syst.add_def("bet = divrsint(brsint)");
    syst.add_def("Asquare = multrsint(multrsint("
                 "scal(grad(bet),grad(bet))))/2./Ntilde^2");
    syst.add_def(ndom - 1, "intMadm = -dr(P)*2./4piG");
    syst.add_def(ndom - 1, "intMk = dr(N)/4piG");
    syst.add_def(ndom - 1,
                 "intJ = multrsint(multrsint(dr(bet)))/4./4piG");
    syst.add_def("dH = dr(H)/H");

    Param eos_parameters;
    syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action,
                 &eos_parameters);
    syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action,
                 &eos_parameters);
    syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action,
                 &eos_parameters);
    syst.add_def("h = exp(H)");
    syst.add_def("rho = rho(h)");
    syst.add_def("eps = eps(h)");
    syst.add_def("press = press(h)");
    syst.add_def("delta = h-eps-1.");
    reader_add_2d_formalism_defs<Formalism>(
        bconfig, syst, space, scalar_slot);

    for (int d = 0; d < ndom; ++d) {
        const bool has_matter = d < adapted_inner;
        if (has_matter) {
            syst.add_def(d, "U = multrsint(P^2/N*(Omg+bet))");
            syst.add_def(d, "Usq = U*U");
            syst.add_def(d, "Wsquare = 1./(1.-Usq)");
            syst.add_def(d, "W = sqrt(Wsquare)");
            syst.add_def(d,
                         "Etilde = press*h*Wsquare-press*delta");
            syst.add_def(d,
                         "Stilde = 3.*press*delta+"
                         "(Etilde+press*delta)*Usq");
            syst.add_def(d, "intMb = W*rho*P^6*4piG/2.");
            syst.add_def(d, "intH = H*W*P^6*4piG/2.");
            syst.add_def(d, "firstint = H+log(N)-log(W)");
        }
        reader_add_2d_hamiltonian_def<Formalism>(syst, d, has_matter);
    }

    const double J = space.get_domain(ndom - 1)->integ(
        syst.give_val_def("intJ")()(ndom - 1), OUTER_BC);
    const double Madm = space.get_domain(ndom - 1)->integ(
        syst.give_val_def("intMadm")()(ndom - 1), OUTER_BC);
    const double Mk = space.get_domain(ndom - 1)->integ(
        syst.give_val_def("intMk")()(ndom - 1), OUTER_BC);
    double baryonic_mass = 0.;
    double integrated_logh = 0.;
    for (int d = 0; d < adapted_inner; ++d) {
        baryonic_mass += syst.give_val_def("intMb")()(d).integ_volume();
        integrated_logh += syst.give_val_def("intH")()(d).integ_volume();
    }
    const ReaderXctsConstraintNorms constraint_norms{
        reader_format_universal_number(reader_hamiltonian_l2_axisym(
            syst, reader_diagnostic_domains(ndom))),
        reader_na_text(), reader_na_text()};

    Point center(2);
    const double central_dhdx = syst.give_val_def("dH")().val_point(center);
    const double central_euler =
        syst.give_val_def("firstint")().val_point(center);
    const double central_omg = Omg.val_point(center);

    const auto equator_domain = Omg(adapted_outer).get_domain();
    Index equator(equator_domain->get_nbr_points());
    equator.set(0) = equator_domain->get_nbr_points()(0) - 1;
    equator.set(1) = equator_domain->get_nbr_points()(1) - 1;
    const double equator_omg = Omg(adapted_outer)(equator);
    const double psi_equator = conf(adapted_outer)(equator);
    const double areal_radius = rmax * psi_equator * psi_equator;

    std::string central_varscal = reader_na_text();
    std::string varscal_outer_min = reader_na_text();
    std::string varscal_outer_max = reader_na_text();
    if constexpr (Formalism::has_varscal) {
        central_varscal =
            reader_format_universal_number(scalar_slot.val_point(center));
        const int outer_radial_index =
            scalar_slot.get_domain(ndom - 1)->get_nbr_points()(0) - 1;
        const auto [scalar_min, scalar_max] = reader_2d_boundary_min_max(
            scalar_slot, ndom - 1, outer_radial_index);
        varscal_outer_min = reader_format_universal_number(scalar_min);
        varscal_outer_max = reader_format_universal_number(scalar_max);
    }

#define FORMAT1 std::setw(25) << std::right << std::setprecision(12) << std::fixed << std::showpos
#define FORMAT std::setw(25) << std::right << std::setprecision(5) << std::scientific << std::showpos

    auto print_shells = [&](int dom_min, int dom_max) {
        int count = 1;
        for (int d = dom_min; d < dom_max; ++d) {
            const std::string label{"SHELL" + std::to_string(count) + " = "};
            std::cout << FORMAT1 << label
                      << bco_utils::get_radius(space.get_domain(d), OUTER_BC)
                      << '\n';
            ++count;
        }
    };

    const int res_r = space.get_domain(0)->get_nbr_points()(0);
    const int res_t = space.get_domain(0)->get_nbr_points()(1);
    const std::string header(22, '#');
    std::cout << "Number of domains: " << ndom << '\n'
              << "System DOF: "
              << reader_format_system_dof(read_system_dof_from_file(spacein))
              << '\n'
              << "NS1 contains: 0 " << adapted_outer << '\n';

    std::cout << header + " Neutron Star " + header + "\n"
              << FORMAT1 << "Center_COM = " << "(0, 0, 0)\n"
              << FORMAT1 << "Coord R_IN = " << rin << '\n'
              << FORMAT1 << "Coord R = " << '[' << rmin << ',' << rmax
              << "] ([" << rmin * M2km << ',' << rmax * M2km << "] km)\n";
    print_shells(2, ndom - 2);
    std::cout << FORMAT1 << "Coord R_OUT = " << rout << '\n'
              << FORMAT1 << "Areal R = " << areal_radius << " ["
              << areal_radius * M2km << "km]\n"
              << FORMAT1 << "Baryonic Mass = " << baryonic_mass << '\n'
              << FORMAT1 << "Isolated ADM Mass = " << bconfig(MADM) << '\n'
              << FORMAT1 << "Quasi-local Madm = " << Madm << " Diff:"
              << std::fabs(1. - Madm / bconfig(MADM)) << '\n'
              << FORMAT1 << "Quasi-local S = " << J << '\n'
              << FORMAT1 << "Chi = " << J / Madm / Madm << " ["
              << bconfig(CHI) << "]\n"
              << FORMAT1 << "Omega = " << central_omg << '\n'
              << FORMAT1 << "x(max(Density)) = " << 0. << " (" << 0.
              << ")\n"
              << FORMAT << "Central Density = " << nc << '\n'
              << FORMAT << "Central log(h) = " << loghc << '\n'
              << FORMAT << "Central Pressure = " << pc << '\n'
              << FORMAT << "Central varscal = " << central_varscal << '\n'
              << FORMAT << "Central dlog(h)/dx = " << central_dhdx << '\n'
              << FORMAT1 << "Central Euler Constant = " << central_euler
              << '\n'
              << FORMAT1 << "Integrated log(h) = " << integrated_logh
              << "\n\n";

    std::cout << header + " Single " + header + "\n"
              << FORMAT1 << "RES = " << '[' << res_r << ',' << res_t
              << "]\n";
    reader_print_universal_gravity_block<Formalism>(bconfig);
    std::cout << FORMAT << "Outer varscal = " << '[' << varscal_outer_min
              << ", " << varscal_outer_max << "]\n"
              << FORMAT1 << "Velocity Potential = " << "absent\n"
              << FORMAT1 << "Central Omega = " << central_omg << " ["
              << bconfig(OMEGA) * M2Hz << "Hz]\n"
              << FORMAT1 << "Equator Omega = " << equator_omg << " ["
              << equator_omg * M2Hz << "Hz]\n"
              << FORMAT1 << "Komar mass = " << Mk << ", Diff: "
              << 2. * std::fabs(Madm - Mk) / (Madm + Mk) << '\n'
              << FORMAT1 << "Adm mass = " << Madm << " [" << bconfig(MADM)
              << "]\n"
              << FORMAT1 << "Adm moment. = " << J << '\n'
              << FORMAT << "Px = " << reader_na_text() << '\n'
              << FORMAT << "Py = " << reader_na_text() << '\n'
              << FORMAT << "Pz = " << reader_na_text() << '\n';
    reader_print_xcts_constraint_norms(constraint_norms);
    Validation::emit_2d(
        bconfig, space, conf, lapse, scalar_slot);

#undef FORMAT1
#undef FORMAT
}

template <class eos_t, typename space_t, typename Formalism,
          typename Validation = NoReaderValidation>
int reader_2d_main(kadath_config<BCO_NS_INFO> bconfig)
{
    reader_2d_output<eos_t, space_t, Formalism, Validation>(bconfig);
    return 0;
}

template <class eos_t, typename space_t>
void reader_bhns_output(kadath_config<BIN_INFO> bconfig)
{
    using namespace Kadath;
    using namespace Kadath::Margherita;
    const double M2km = reader_M2km;

    if (std::isnan(bconfig.set(MADM, BCO1))) {
        std::cerr << "Missing \"fixed_madm\" in config file\n"
                     "Setting to \"madm\"! \n";
        bconfig.set(MADM, BCO1) = bconfig(QLMADM, BCO1);
    }

    std::string in_spacefile = bconfig.space_filename();
    BeFileSource fich(in_spacefile);
    space_t space(fich);
    Scalar conf(space, fich);
    Scalar lapse(space, fich);
    Vector shift(space, fich);
    Scalar logh(space, fich);
    Scalar phi(space, fich);

    Base_tensor basis(shift.get_basis());
    Metric_flat fmet(space, basis);
    CoordFields<space_t> cfields(space);

    int ndom = space.get_nbr_domains();
    const int system_dof = read_system_dof_from_file(in_spacefile);
    std::cout << "Number of domains: " << ndom << "\n"
              << "System DOF: " << reader_format_system_dof(system_dof) << "\n"
              << "Bounds: [rin, rmid, rout, shells] + r_bisph" << std::endl;

    // Outer radius of every domain in a compact object's block, plus the last
    // one repeated as the bispheric matching radius r_bisph.
    auto print_co_bounds = [&](const char* label, int dom_first, int dom_last) {
        std::cout << label << ": [";
        double r_bisph = 0.;
        for (int dom = dom_first; dom <= dom_last; ++dom) {
            const double rout = bco_utils::get_radius(space.get_domain(dom), OUTER_BC);
            std::cout << (dom > dom_first ? " " : "") << rout;
            r_bisph = rout;
        }
        std::cout << "] + " << r_bisph << std::endl;
    };
    print_co_bounds("NS", space.NS, space.BH - 1);
    print_co_bounds("BH", space.BH, space.OUTER - 1);
    std::cout << "Outer bounds: "
              << bco_utils::get_radius(space.get_domain(space.OUTER + 5), INNER_BC)
              << std::endl;

    double xc1 = bco_utils::get_center(space, space.NS);
    double xc2 = bco_utils::get_center(space, space.BH);
    double xo = bco_utils::get_center(space, ndom - 1);

    vec_ary_t coord_vectors = default_binary_vector_ary(space);
    update_fields(cfields, coord_vectors, {}, xo, xc1, xc2);

    System_of_eqs syst(space);
    fmet.set_system(syst, "f");

    Param p;
    syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
    syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
    syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);

    syst.add_cst("4piG", 4.0 * M_PI);
    syst.add_cst("PI", M_PI);

    syst.add_cst("Mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
    syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
    syst.add_cst("mpx", *coord_vectors[to_int(coord_vector::BCO2_ROTx)]);
    syst.add_cst("mpz", *coord_vectors[to_int(coord_vector::BCO2_ROTz)]);

    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);
    syst.add_cst("ez", *coord_vectors[to_int(coord_vector::EZ)]);

    syst.add_cst("s1", *coord_vectors[to_int(coord_vector::S_BCO1)]);
    syst.add_cst("s2", *coord_vectors[to_int(coord_vector::S_BCO2)]);
    syst.add_cst("esurf", *coord_vectors[to_int(coord_vector::S_INF)]);

    syst.add_cst("xaxis", bconfig(COM));
    syst.add_cst("yaxis", bconfig(COMY));
    syst.add_cst("ome", bconfig(GOMEGA));
    syst.add_cst("omes1", bconfig(OMEGA, BCO1));
    syst.add_cst("angs1", bconfig(DEG, BCO1) * std::acos(-1.) / 180.);
    syst.add_cst("omes2", bconfig(OMEGA, BCO2));
    syst.add_cst("angs2", bconfig(DEG, BCO2) * std::acos(-1.) / 180.);

    syst.add_def("m1^i = cos(angs1) * mmz^i + sin(angs1) * mmx^i ");
    syst.add_def("m2^i = cos(angs2) * mpz^i + sin(angs2) * mpx^i ");

    for (int d = space.NS; d <= space.ADAPTEDNS; ++d) {
        syst.add_def(d, "s^i  = omes1 * m1^i");
    }

    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);
    syst.add_cst("H", logh);
    syst.add_cst("phi", phi);

    syst.add_def("NP     = P*N");
    syst.add_def("Ntilde = N / P^6");

    syst.add_def("Morb^i = Mg^i + xaxis * ey^i + yaxis * ex^i");
    syst.add_def("B^i = bet^i + ome * Morb^i");

    syst.add_def("dH = (ex^i * D_i H) / H");
    syst.add_def("dH2 = ex^i * D_i dH");

    syst.add_def("A^ij   = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / Ntilde");

    syst.add_def(ndom - 1, "intPx = A_i^j * ex_j * esurf^i / 8 / PI");
    syst.add_def(ndom - 1, "intPy = A_i^j * ey_j * esurf^i / 8 / PI");
    syst.add_def(ndom - 1, "intPz = A_i^j * ez_j * esurf^i / 8 / PI");
    syst.add_def(space.BH + 2, "h^ij = f^ij - s2^i * s2^j");

    syst.add_def(space.ADAPTEDNS + 1, "intS1 = A_ij * m1^i * s1^j / 8. / PI");
    syst.add_def(space.ADAPTEDBH + 1, "intS2 = A_ij * m2^i * s2^j / 8. / PI");
    syst.add_def("intAsq = P^4 / 4 / PI");
    syst.add_def("intMsq = intAsq / 4");
    syst.add_def(space.ADAPTEDBH + 1, "intMsq = P^4 / 16. / PI");
    syst.add_def(space.ADAPTEDNS + 1, "intMsq = P^4 / 4.  / PI");

    syst.add_def(ndom - 1, "COMx  = -3 * P^4 *esurf^i * ex_i / 8. / PI");
    syst.add_def(ndom - 1, "COMy  = 3 * P^4 * esurf^i * ey_i / 8. / PI");
    syst.add_def(ndom - 1, "COMz  = 3 * P^4 * esurf^i * ez_i / 8. / PI");
    syst.add_def("dtgamma = D_k bet^k + 6 / P * B^k * D_k P");

    for (int d = 0; d < ndom; d++) {
        // Vacuum outside the NS (matches BHNS_XCTS/stages.ipp gating). The old
        // `(d >= BH) || d == ADAPTEDNS+1` form leaked inter-object domains into
        // the in-star branch; harmless here (eta_i is defined inline) but brittle.
        if (d >= space.ADAPTEDNS + 1) {
            syst.add_def(d, "eqP     = D^i D_i P + A_ij * A^ij / P^7 / 8");
            syst.add_def(d, "Hcon2   = eqP * eqP");
            syst.add_def(d, "eqNP    = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
            syst.add_def(d, "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");
            reader_add_gr_xcts_constraint_rms_defs(syst, d, /*has_matter=*/false);
        } else {
            syst.add_def(d, "h      = exp(H)");
            syst.add_def(d, "s^i  = omes1 * m1^i");
            syst.add_def(d, "eta_i  = D_i phi + P^4 * s_i");
            syst.add_def(d, "Wsquare= eta^i * eta_i / h^2 / P^4 + 1.");
            syst.add_def(d, "W      = sqrt(Wsquare)");
            syst.add_def(d, "U^i    = eta^i / P^4 / h / W");
            syst.add_def(d, "V^i    = N * U^i - B^i");
            syst.add_def(d, "Usquare= P^4 * U_i * U^i");
            syst.add_def(d, "E      = rho(h) * h * Wsquare - press(h)");
            syst.add_def(d, "S      = 3 * press(h) + (E + press(h)) * Usquare");
            syst.add_def(d, "p^i    = rho(h) * h * Wsquare * U^i");
            syst.add_def(d, "intMb  = P^6 * rho(h) * W");
            syst.add_def(d, "intM   = - D_i D^i P / 2. / PI");
            syst.add_def(d, "firstint = h * N / W + eta_i * V^i");
            syst.add_def(d, "eqphi  = D_i (P^6 * W * rho(h) * V^i)");
            syst.add_def(d, "eqP    = D^i D_i P + A_ij * A^ij / P^7 / 8 + 4piG / 2. * P^5 * E");
            syst.add_def(d, "Hcon2  = eqP * eqP");
            syst.add_def(d, "eqNP   = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij *A^ij "
                            "- 4piG / 2. * N * P^5 * (E + 2. * S)");
            syst.add_def(d, "eqbet^i= D_j D^j bet^i + D^i D_j bet^j / 3. "
                            "- 2. * A^ij * D_j Ntilde - 4. * 4piG * N * P^4 * p^i");
            reader_add_gr_xcts_constraint_rms_defs(
                syst, d, /*has_matter=*/true,
                "rho(h) * h * Wsquare - press(h)",
                "rho(h) * h * Wsquare * U^i");
            syst.add_def(d, "intH  = P^6 * H * W");
        }
    }
    syst.add_def(ndom - 1, "intJ = multr(A_ij * Morb^j * esurf^i) / 8 / PI");
    syst.add_def(ndom - 1, "Madm = -dr(P) / 2 / PI");
    syst.add_def(ndom - 1, "Mk   =  dr(N) / 4 / PI");

    // NS quantities
    double loghc1 = bco_utils::get_boundary_val(space.NS, logh, INNER_BC);
    double pressc1 = EOS<eos_t, eos_var_t::PRESSURE>::get(std::exp(loghc1));
    double rhoc1 = EOS<eos_t, eos_var_t::DENSITY>::get(std::exp(loghc1));

    std::vector<double> baryonic_mass1{};
    std::vector<double> int_H1{};
    std::vector<double> adm_mass1{};
    for (int d = space.NS; d <= space.ADAPTEDNS; ++d) {
        baryonic_mass1.push_back(syst.give_val_def("intMb")()(d).integ_volume());
        int_H1.push_back(syst.give_val_def("intH")()(d).integ_volume());
        adm_mass1.push_back(syst.give_val_def("intM")()(d).integ_volume());
    }
    double MB1 = std::accumulate(baryonic_mass1.begin(), baryonic_mass1.end(), 0.);
    double ql_madm1 = std::accumulate(adm_mass1.begin(), adm_mass1.end(), 0.);
    double intH1 = std::accumulate(int_H1.begin(), int_H1.end(), 0.);
    double ql_spinns =
        space.get_domain(space.ADAPTEDNS + 1)->integ(syst.give_val_def("intS1")()(space.ADAPTEDNS + 1), OUTER_BC);

    auto [rmin, rmax] = bco_utils::get_rmin_rmax(space, space.ADAPTEDNS);
    double A = space.get_domain(space.ADAPTEDNS + 1)->integ(syst.give_val_def("intAsq")()(space.ADAPTEDNS + 1), INNER_BC);
    double areal_rns = sqrt(A);

    double rin_ns = bco_utils::get_radius(space.get_domain(space.NS), EQUI);
    double rout_ns = bco_utils::get_radius(space.get_domain(space.BH - 1), EQUI);

    auto dHdx = syst.give_val_def("dH")();
    double dHdx1 = bco_utils::get_boundary_val(space.NS, dHdx, INNER_BC);
    double euler1 = bco_utils::get_boundary_val(space.NS, syst.give_val_def("firstint")(), INNER_BC);
    double NS_x_com = xc1 + bconfig(COM);
    double x_max = xc1;
    {
        double err = 1.;
        Point absol(3);
        absol.set(1) = x_max;
        absol.set(2) = 0.;
        absol.set(3) = 0.;

        Scalar dHdx2 = syst.give_val_def("dH2");
        while (err > 1e-14) {
            x_max = x_max - dHdx.val_point(absol) / dHdx2.val_point(absol);
            absol.set(1) = x_max;
            err = std::abs(dHdx.val_point(absol));
        }
    }

    // BH quantities
    double ql_spinbh =
        space.get_domain(space.ADAPTEDBH + 1)->integ(syst.give_val_def("intS2")()(space.ADAPTEDBH + 1), OUTER_BC);
    double mirrsq = space.get_domain(space.BH + 2)->integ(syst.give_val_def("intMsq")()(space.BH + 2), INNER_BC);
    double mirr = sqrt(mirrsq);
    double mch = sqrt(mirrsq + ql_spinbh * ql_spinbh / 4. / mirrsq);
    double rin_bh = bco_utils::get_radius(space.get_domain(space.BH), EQUI);
    double r_bh = bco_utils::get_radius(space.get_domain(space.BH + 1), EQUI);
    double rout_bh = bco_utils::get_radius(space.get_domain(space.OUTER - 1), EQUI);
    A = space.get_domain(space.ADAPTEDBH + 1)->integ(syst.give_val_def("intAsq")()(space.ADAPTEDBH + 1), INNER_BC);
    double areal_rbh = sqrt(A);
    auto [lapsemin, lapsemax] = bco_utils::get_field_min_max(lapse, space.BH + 2, INNER_BC);
    auto [confmin, confmax] = bco_utils::get_field_min_max(conf, space.BH + 2, INNER_BC);
    double BH_x_com = xc2 + bconfig(COM);

    // binary quantities
    double adm_inf = space.get_domain(ndom - 1)->integ(syst.give_val_def("Madm")()(ndom - 1), OUTER_BC);
    double komar = space.get_domain(ndom - 1)->integ(syst.give_val_def("Mk")()(ndom - 1), OUTER_BC);
    double e_diff = fabs(adm_inf - komar) / (adm_inf + komar);
    double Jinf = space.get_domain(ndom - 1)->integ(syst.give_val_def("intJ")()(ndom - 1), OUTER_BC);
    double Px = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPx")()(ndom - 1), OUTER_BC);
    double Py = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPy")()(ndom - 1), OUTER_BC);
    double Pz = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPz")()(ndom - 1), OUTER_BC);

    double& Madm1 = bconfig(MADM, BCO1);
    double Minf = Madm1 + mch;
    double e_bind = adm_inf - Minf;

    double COMx = space.get_domain(ndom - 1)->integ(syst.give_val_def("COMx")()(ndom - 1), OUTER_BC) / adm_inf;
    double COMy = space.get_domain(ndom - 1)->integ(syst.give_val_def("COMy")()(ndom - 1), OUTER_BC) / adm_inf;
    double COMz = space.get_domain(ndom - 1)->integ(syst.give_val_def("COMz")()(ndom - 1), OUTER_BC) / adm_inf;

    const auto diagnostic_domains = reader_diagnostic_domains(ndom, {space.BH, space.BH + 1});
    double hcon_L2 = reader_hamiltonian_l2(syst, diagnostic_domains);
    const auto [h_rms, m_rms] = reader_xcts_constraint_rms(syst, diagnostic_domains);
    const std::string hcon_rms = reader_format_universal_number(h_rms);
    const std::string momcon_rms = reader_format_universal_number(m_rms);

#define FORMAT1 std::setw(25) << std::right << std::setprecision(12) << std::fixed << std::showpos
#define FORMAT std::setw(25) << std::right << std::setprecision(5) << std::scientific << std::showpos
    auto print_shells = [&](int dom_min, int dom_max) {
        int cnt = 1;
        for (int i = dom_min; i < dom_max; ++i) {
            std::string shell{"SHELL" + std::to_string(cnt) + " = "};
            Index pos(space.get_domain(i + 1)->get_radius().get_conf().get_dimensions());
            std::cout << FORMAT1 << shell << space.get_domain(i + 1)->get_radius()(pos) << std::endl;
            cnt++;
        }
    };
    auto print_shell_mb = [&](auto& vec) {
        for (auto& e : vec) {
            std::cout << e << ",";
        }
    };

    std::string header(22, '#');
    std::cout << header + " Neutron Star " + header + "\n";
    std::cout << FORMAT1 << "Center_COM = " << "(" << NS_x_com << ", 0, 0)\n"
              << FORMAT1 << "Coord R_IN = " << rin_ns << std::endl;
    print_shells(space.NS + 1, space.ADAPTEDNS);
    std::cout << FORMAT1 << "Coord R = " << "[" << rmin << "," << rmax << "] ("
              << "[" << rmin * M2km << "," << rmax * M2km << "] km)" << std::endl;
    print_shells(space.ADAPTEDNS + 1, space.BH - 1);
    std::cout << FORMAT1 << "Coord R_OUT = " << rout_ns << std::endl
              << FORMAT1 << "Areal R = " << areal_rns << " [" << areal_rns * M2km << "km]\n"
              << FORMAT1 << "Baryonic Mass = " << MB1 << " (";
    print_shell_mb(baryonic_mass1);
    std::cout << ")\n"
              << FORMAT1 << "Isolated ADM Mass = " << bconfig(MADM, BCO1) << "\n"
              << FORMAT1 << "Quasi-local Madm = " << ql_madm1 << " Diff:"
              << std::fabs(1. - ql_madm1 / bconfig(MADM, BCO1)) << std::endl
              << FORMAT1 << "Quasi-local S = " << ql_spinns << std::endl
              << FORMAT1 << "Chi = " << ql_spinns / Madm1 / Madm1 << " [" << bconfig(CHI, BCO1) << "]\n"
              << FORMAT1 << "Omega = " << bconfig(OMEGA, BCO1) << std::endl
              << FORMAT1 << "x(max(Density)) = " << x_max << " (" << (xc1 - x_max) / xc1 << ")\n"
              << FORMAT << "Central Density = " << rhoc1 << std::endl
              << FORMAT << "Central log(h) = " << loghc1 << std::endl
              << FORMAT << "Central Pressure = " << pressc1 << std::endl
              << FORMAT << "Central varscal = " << reader_na_text() << std::endl
              << FORMAT << "Central dlog(h)/dx = " << dHdx1 << std::endl
              << FORMAT1 << "Central Euler Constant = " << euler1 << std::endl
              << FORMAT1 << "Integrated log(h) = " << intH1 << "\n\n";

    std::cout << header + " Black Hole " + header + "\n"
              << FORMAT1 << "Center_COM = " << "(" << BH_x_com << ", 0, 0)\n"
              << FORMAT1 << "Coord R_IN = " << rin_bh << std::endl
              << FORMAT1 << "Coord R = " << r_bh << " [" << r_bh * M2km << "km]\n";
    print_shells(space.BH + 2, space.OUTER - 1);
    std::cout << FORMAT1 << "Coord R_OUT = " << rout_bh << std::endl
              << FORMAT1 << "Areal R = " << areal_rbh << std::endl
              << FORMAT1 << " LAPSE = [" << lapsemin << ", " << lapsemax << "]\n"
              << FORMAT1 << " PSI = [" << confmin << ", " << confmax << "]\n"
              << FORMAT1 << "Mirr = " << mirr << "\n"
              << FORMAT1 << "Mch = " << mch << " [" << bconfig(MCH, BCO2) << "]\n"
              << FORMAT1 << "Chi = " << ql_spinbh / (mch * mch) << " [" << bconfig(CHI, BCO2) << "]\n"
              << FORMAT1 << "S = " << ql_spinbh << std::endl
              << FORMAT1 << "Omega = " << bconfig(OMEGA, BCO2) << "\n\n";

    auto outer_shells = space.get_n_shells_outer();
    auto M1 = bconfig(MADM, BCO1);
    auto M2 = bconfig(MCH, BCO2);
    if (M2 > M1) std::swap(M1, M2);
    if (outer_shells > 0) {
        std::cout << FORMAT1 << "Outer shell bounds\n";
        print_shells(ndom - 1 - outer_shells, ndom - 1);
    }
    std::cout << header + " Binary " + header + "\n"
              << FORMAT1 << "Radial RES = " << reader_radial_resolution_by_domain(space) << '\n'
              << FORMAT1 << "Theta RES = " << reader_polar_resolution_by_domain(space) << '\n'
              << FORMAT1 << "Phi RES = " << reader_azimuthal_resolution_by_domain(space) << '\n';
    reader_print_universal_gravity_block<GrFormalism>(bconfig);
    std::cout << FORMAT << "Outer varscal = " << "[" << reader_na_text() << ", " << reader_na_text() << "]"
              << std::endl;
    std::cout << FORMAT1 << "Q = " << M2 / M1 << std::endl
              << FORMAT1 << std::setprecision(2) << "Separation = " << bconfig(DIST) << " [" << bconfig(DIST) / Minf
              << "] (" << bconfig(DIST) * M2km << "km)" << std::endl
              << FORMAT1 << "Orbital Omega = " << bconfig(GOMEGA) << std::endl
              << FORMAT1 << "Komar mass = " << komar << std::endl
              << FORMAT1 << "Adm mass = " << adm_inf << ", Diff: " << e_diff << std::endl
              << FORMAT1 << "Total mass (Minf) = " << Minf << " [" << Madm1 + bconfig(MCH, BCO2) << "]\n"
              << FORMAT1 << "Adm moment. = " << Jinf << std::endl
              << FORMAT << "Binding energy = " << e_bind << std::endl
              << FORMAT << "Minf * Ome = " << Minf * bconfig(GOMEGA) << std::endl
              << FORMAT << "E_b / Minf = " << e_bind / Minf << std::endl
              << FORMAT << "Px = " << Px << std::endl
              << FORMAT << "Py = " << Py << std::endl
              << FORMAT << "Pz = " << Pz << std::endl
              << FORMAT1 << "COMx = " << bconfig(COM) << ", A-COMx = " << COMx << std::endl
              << FORMAT1 << "COMy = " << bconfig(COMY) << ", A-COMy = " << COMy << std::endl
              << FORMAT1 << "A-COMz = " << COMz << std::endl
              << FORMAT << "Hamiltonian H (L2) = " << hcon_L2
              << std::endl
              << FORMAT << "Hamiltonian H (RMS) = " << hcon_rms
              << std::endl
              << FORMAT << "Momentum M (RMS) = " << momcon_rms
              << std::endl;
#undef FORMAT1
#undef FORMAT
}

// BHNS reader entry point. EOS-type dispatch lives in the thin per-app main.
template <class eos_t, typename space_t>
int reader_bhns_main(kadath_config<BIN_INFO> bconfig)
{
    reader_bhns_output<eos_t, space_t>(bconfig);
    return 0;
}

// ===========================================================================
// Isolated black-hole reader.
// ===========================================================================
//
// The old apps/BH/src/reader.cpp was fundamentally broken: it loaded a
// Space_adapted_bh dataset as Space_spheric_adapted and printed neutron-star
// matter diagnostics (no horizon mass). This is a correct replacement, ported
// from the BH solver's own diagnostics (include/Apps/Formalism/GR/BH_3D_XCTS/
// solver_imp.ipp, syst_init + print_diagnostics_norot): the irreducible mass
// Mirr from the proper-area integral over the apparent horizon (domain 2,
// INNER_BC), the Christodoulou mass, and the ADM/Komar masses at infinity. The
// BH solver saves exactly conf/lapse/shift (no matter/EOS). BUILD/PORT-ONLY: no
// converged BH fixture, so not bit-identical A/B-gated.
template <typename space_t>
void reader_single_bh_output(kadath_config<BCO_NS_INFO> bconfig)
{
    using namespace Kadath;
    const double M2km = reader_M2km;

    const std::string spacein = bconfig.space_filename();
    BeFileSource ff1(spacein);
    space_t space(ff1);
    Scalar conf(space, ff1);
    Scalar lapse(space, ff1);
    Vector shift(space, ff1);

    int ndom = space.get_nbr_domains();
    Base_tensor basis(space, CARTESIAN_BASIS);
    Metric_flat fmet(space, basis);

    CoordFields<space_t> cf_generator(space);
    vec_ary_t coord_vectors{default_co_vector_ary(space)};
    double xo = bco_utils::get_center(space, 0);
    update_fields_co(cf_generator, coord_vectors, {}, xo);

    std::cout << "Number of domains: " << ndom << "\n"
              << "System DOF: " << reader_format_system_dof(read_system_dof_from_file(spacein))
              << "\n";
    System_of_eqs syst(space, 0, ndom - 1);
    fmet.set_system(syst, "f");

    syst.add_cst("4piG", 4.0 * M_PI);
    syst.add_cst("PI", M_PI);

    syst.add_cst("mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_cst("sm", *coord_vectors[to_int(coord_vector::S_BCO1)]);
    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);
    syst.add_cst("einf", *coord_vectors[to_int(coord_vector::S_INF)]);

    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);

    syst.add_def("NP = P*N");
    syst.add_def("Ntilde = N / P^6");
    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / Ntilde");

    // excision-surface (horizon) + at-infinity integrands (from the BH solver's syst_init)
    syst.add_def("intMsq= P^4 / 16. / PI");
    syst.add_def("intS  = A_ij * mg^i * sm^j / 8. / PI");
    syst.add_def(ndom - 1, "intMadm = - einf^i * D_i P / 4piG * 2");
    syst.add_def(ndom - 1, "intMk = einf^i * D_i N / 4piG");

    // irreducible mass from the proper-area integral over the apparent horizon
    // (domain 2, INNER_BC), the spin, and the Christodoulou mass
    double Mirrsq = space.get_domain(2)->integ(syst.give_val_def("intMsq")()(2), INNER_BC);
    double Mirr = std::sqrt(Mirrsq);
    double S = space.get_domain(2)->integ(syst.give_val_def("intS")()(2), INNER_BC);
    double Mch = std::sqrt(Mirrsq + S * S / 4. / Mirrsq);

    // ADM and Komar masses as surface integrals at infinity
    double Madm = space.get_domain(ndom - 1)->integ(syst.give_val_def("intMadm")()(ndom - 1), OUTER_BC);
    double Mk = space.get_domain(ndom - 1)->integ(syst.give_val_def("intMk")()(ndom - 1), OUTER_BC);

    // horizon coordinate radius + lapse/conformal-factor extrema on the excision
    double r_hor = bco_utils::get_radius(space.get_domain(1), OUTER_BC);
    auto [lapsemin, lapsemax] = bco_utils::get_field_min_max(lapse, 2, INNER_BC);
    auto [confmin, confmax] = bco_utils::get_field_min_max(conf, 2, INNER_BC);

    auto res_r = space.get_domain(0)->get_nbr_points()(0);
    auto res_t = space.get_domain(0)->get_nbr_points()(1);
    auto res_p = space.get_domain(0)->get_nbr_points()(2);

#define FORMAT std::setw(25) << std::right << std::setprecision(5) << std::scientific << std::showpos
    std::string header(22, '#');
    std::cout << header + " Black Hole " + header + "\n"
              << FORMAT << "RES = " << "[" << res_r << "," << res_t << "," << res_p << "]\n"
              << FORMAT << "Horizon Coord R = " << r_hor << " [" << r_hor * M2km << "km]\n"
              << FORMAT << "Mirr = " << Mirr << std::endl
              << FORMAT << "Mch = " << Mch << std::endl
              << FORMAT << "S = " << S << std::endl
              << FORMAT << "Chi = " << S / (Mch * Mch) << std::endl
              << FORMAT << "Madm = " << Madm << std::endl
              << FORMAT << "Mk = " << Mk << ", Diff: " << std::abs(Madm - Mk) / Madm << std::endl
              << FORMAT << " LAPSE = [" << lapsemin << ", " << lapsemax << "]\n"
              << FORMAT << " PSI = [" << confmin << ", " << confmax << "]\n";
#undef FORMAT
}

// Isolated-BH reader entry point. The BH solver carries no matter/EOS, so the
// thin main needs no EOS dispatch.
template <typename space_t>
int reader_single_bh_main(kadath_config<BCO_NS_INFO> bconfig)
{
    reader_single_bh_output<space_t>(bconfig);
    return 0;
}

// ===========================================================================
// Three-body reader: three TOV stars on Space_three_body. The static stage
// solves the corotating form at fixed (zero) frequency without a velocity
// potential, so the diagnostics use U^i = omega^i / N and skip the spin
// blocks. GR-only for now (the three-body apps carry no scalar field).
// ===========================================================================
template <class eos_t, typename space_t, typename Formalism>
void reader_three_body_output(kadath_config<TRI_INFO> bconfig)
{
    using namespace Kadath;
    using namespace Kadath::Margherita;

    static_assert(!Formalism::has_varscal,
                  "three-body reader: unsupported field set");

    using vec_d = std::vector<double>;
    using ary_d = std::array<double, 3>;
    using ary_i = std::array<int, 3>;

    constexpr std::array<NODES, 3> bcos = {BCO1, BCO2, BCO3};
    for (auto bco : bcos) {
        if (std::isnan(bconfig.set(MADM, bco))) {
            std::cerr << "Missing \"Madm\" in config file\n"
                         "Setting to quasi-local \"Madm\"! \n";
            bconfig.set(MADM, bco) = bconfig(QLMADM, bco);
        }
    }

    // read domain decomposition and fields from binary file
    std::string in_spacefile = bconfig.space_filename();
    BeFileSource fich(in_spacefile);

    space_t space(fich);
    Scalar conf(space, fich);
    Scalar lapse(space, fich);
    Vector shift(space, fich);
    Scalar logh(space, fich);

    // define basic fields
    Base_tensor basis(shift.get_basis());
    Metric_flat fmet(space, basis);

    // coordinate dependent fields
    CoordFields<space_t> cfields(space);

    // enumerate the three stars of this space
    std::vector<CompactObject> objects = co_traits<space_t>::objects(space);
    const ary_d x_nuc{objects[0].center_x, objects[1].center_x, objects[2].center_x};

    // coordinate origin
    double xo = 0;

    // coordinate dependent vector fields (single-origin set: mg, ex/ey/ez, einf)
    vec_ary_t coord_vectors = default_co_vector_ary(space);
    update_fields_co(cfields, coord_vectors, {}, xo);

    // compute the location of the maximum densities
    ary_d x_max;
    {
        System_of_eqs syst_H(space);
        fmet.set_system(syst_H, "f");

        syst_H.add_cst("H", logh);
        syst_H.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);

        syst_H.add_def("dH = (ex^i * D_i H) / H");
        syst_H.add_def("dH2 = ex^i * D_i dH");

        Scalar dHdx = syst_H.give_val_def("dH");
        Scalar dHdx2 = syst_H.give_val_def("dH2");

        for (int j : {0, 1, 2}) {
            double xc = x_nuc[j];
            double err = 1;

            Point absol(3);
            absol.set(1) = xc;
            absol.set(2) = 0;
            absol.set(3) = 0;

            while (err > 1e-14) {
                xc = xc - dHdx.val_point(absol) / dHdx2.val_point(absol);
                absol.set(1) = xc;
                err = std::abs(dHdx.val_point(absol));
            }
            x_max[j] = xc;
        }
    }

    // setup a system of equations to compute derived quantities
    int ndom = space.get_nbr_domains();
    const int system_dof = read_system_dof_from_file(in_spacefile);
    System_of_eqs syst(space);
    cout << "Number of domains: " << ndom << "\n"
         << "System DOF: " << reader_format_system_dof(system_dof) << "\n"
         << "NS1 contains: " << space.BODY << " " << space.ADAPTED_BODY << "\n"
         << "NS2 contains: " << space.CHILD1 << " " << space.ADAPTED_CHILD1 << "\n"
         << "NS3 contains: " << space.CHILD2 << " " << space.ADAPTED_CHILD2 << endl;

    fmet.set_system(syst, "f");

    // setup the eos operators
    Param p;
    syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
    syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
    syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    syst.add_ope("dHdlnrho", &EOS<eos_t, eos_var_t::DHDRHO>::action, &p);

    // constants in the system
    syst.add_cst("4piG", 4.0 * M_PI);

    // these names are hardcoded in the coord_fields.hpp!
    syst.add_cst("mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);
    syst.add_cst("ez", *coord_vectors[to_int(coord_vector::EZ)]);
    syst.add_cst("einf", *coord_vectors[to_int(coord_vector::S_INF)]);

    // global frequency parameter (zero for the static configuration)
    syst.add_cst("ome", bconfig(TRI_GOMEGA));

    // the actual fields representing the solution
    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);
    syst.add_cst("H", logh);

    // matter quantities from the EOS
    syst.add_def("h = exp(H)");
    syst.add_def("press = press(h)");
    syst.add_def("eps = eps(h)");
    syst.add_def("rho = rho(h)");
    syst.add_def("delta = h - eps - 1.");
    syst.add_def("dH = (ex^i * D_i H) / H");

    // conformal quantities and the extrinsic curvature
    syst.add_def("NP = P*N");
    syst.add_def("Ntilde = N / P^6");
    syst.add_def("omega^i = bet^i + ome * mg^i");
    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3.* D_k bet^k * f^ij) / 2. / Ntilde");

    // ADM momenta and center-of-mass integrants at infinity
    syst.add_def(ndom - 1, "intPx = A_i^j * ex_j * einf^i");
    syst.add_def(ndom - 1, "intPy = A_i^j * ey_j * einf^i");
    syst.add_def(ndom - 1, "intPz = A_i^j * ez_j * einf^i");

    // definitions from https://arxiv.org/abs/1506.01689
    syst.add_def(ndom - 1, "intCOMx = 3. / 2. / 4piG * P^4 * ex_i * einf^i");
    syst.add_def(ndom - 1, "intCOMy = 3. / 2. / 4piG * P^4 * ey_i * einf^i");
    syst.add_def(ndom - 1, "intCOMz = 3. / 2. / 4piG * P^4 * ez_i * einf^i");

    for (int d = 0; d < ndom; ++d) {
        bool is_star_domain = false;
        for (const auto& obj : objects) {
            if (d >= obj.nucleus_dom && d <= obj.adapted_dom) {
                is_star_domain = true;
                break;
            }
        }
        if (!is_star_domain) {
            syst.add_def(d, "Hcon   = D^i D_i P + A_ij * A^ij / P^7 / 8");
            syst.add_def(d, "Hcon2  = Hcon * Hcon");
        }
    }

    for (const auto& obj : objects) {
        for (int d = obj.nucleus_dom; d <= obj.adapted_dom; ++d) {
            // static / corotating velocity form
            syst.add_def(d, "U^i    = omega^i / N");
            syst.add_def(d, "Usquare= P^4 * U_i * U^i");
            syst.add_def(d, "Wsquare= 1 / (1 - Usquare)");
            syst.add_def(d, "W      = sqrt(Wsquare)");
            syst.add_def(d, "firstint = log(h * N / W)");

            syst.add_def(d, "intMb  = P^6 * rho * W");
            syst.add_def(d, "intM   = - D_i D^i P * 2. / 4piG");
            syst.add_def(d, "intH  = P^6 * H * W");

            // Hamiltonian-constraint residual (matter eqP form, same as the
            // solver's gr_xcts::add_xcts_field_equations). Vacuum domains use the
            // matter-free form above.
            syst.add_def(d, "Etilde = press * h * Wsquare - press * delta");
            syst.add_def(d, "Hcon   = delta * D^i D_i P + delta * A_ij * A^ij / P^7 / 8 + 4piG / 2. * P^5 * Etilde");
            syst.add_def(d, "Hcon2  = Hcon * Hcon");
        }
    }

    // ADM and Komar integrants at infinity
    syst.add_def(ndom - 1, "intJ = multr(A_ij * mg^j * einf^i) / 2. / 4piG");
    syst.add_def(ndom - 1, "Madm = -dr(P) * 2 / 4piG");
    syst.add_def(ndom - 1, "Mk   =  dr(N) / 4piG");
    // integrant to compute proper area over a spherical surface
    syst.add_def("intMsq = P^4");

    // per-star quantities
    ary_d areal_r;
    ary_d central_logh;
    ary_d central_rho;
    ary_d central_P;
    ary_d central_dHdx;
    ary_d central_euler;
    ary_d H_int;
    ary_d ql_madm;
    ary_d madm;
    ary_d mbs;
    ary_i nuc_doms;
    ary_i adapted_doms;

    std::vector<std::array<double, 2>> r_extrema;
    std::vector<vec_d> mb_distro;

    for (int i : {0, 1, 2}) {
        nuc_doms[i] = objects[i].nucleus_dom;
        adapted_doms[i] = objects[i].adapted_dom;
        int dom = adapted_doms[i];

        // compute area and areal radius from that
        double A = space.get_domain(dom + 1)->integ(syst.give_val_def("intMsq")()(dom + 1), INNER_BC);
        areal_r[i] = sqrt(A / 4. / acos(-1.));

        // point defining the center of the star
        Point ns_c(3);
        ns_c.set(1) = x_nuc[i];
        ns_c.set(2) = 0;
        ns_c.set(3) = 0;

        // central matter quantities
        central_logh[i] = logh.val_point(ns_c);
        central_rho[i] = EOS<eos_t, eos_var_t::DENSITY>::get(std::exp(central_logh[i]));
        central_P[i] = EOS<eos_t, eos_var_t::PRESSURE>::get(std::exp(central_logh[i]));
        central_dHdx[i] = syst.give_val_def("dH")().val_point(ns_c);
        central_euler[i] = syst.give_val_def("firstint")().val_point(ns_c);

        // volume integrated quantities
        H_int[i] = 0;
        vec_d ql_m{};
        vec_d baryonic_mass{};

        for (int d = nuc_doms[i]; d <= adapted_doms[i]; ++d) {
            H_int[i] += syst.give_val_def("intH")()(d).integ_volume();
            ql_m.push_back(syst.give_val_def("intM")()(d).integ_volume());
            baryonic_mass.push_back(syst.give_val_def("intMb")()(d).integ_volume());
        }

        mb_distro.push_back(baryonic_mass);
        mbs[i] = std::accumulate(baryonic_mass.begin(), baryonic_mass.end(), 0.);
        ql_madm[i] = std::accumulate(ql_m.begin(), ql_m.end(), 0.);
        madm[i] = bconfig(MADM, bcos[i]);

        // minimal and maximal (coordinate) radius across the adapted surface
        r_extrema.push_back(bco_utils::get_rmin_rmax(space, dom));
    }

    // system quantities
    double adm_inf = space.get_domain(ndom - 1)->integ(syst.give_val_def("Madm")()(ndom - 1), OUTER_BC);
    double komar = space.get_domain(ndom - 1)->integ(syst.give_val_def("Mk")()(ndom - 1), OUTER_BC);
    double Jinf = space.get_domain(ndom - 1)->integ(syst.give_val_def("intJ")()(ndom - 1), OUTER_BC);
    double Px = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPx")()(ndom - 1), OUTER_BC);
    double Py = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPy")()(ndom - 1), OUTER_BC);
    double Pz = space.get_domain(ndom - 1)->integ(syst.give_val_def("intPz")()(ndom - 1), OUTER_BC);

    double COMx = space.get_domain(ndom - 1)->integ(syst.give_val_def("intCOMx")()(ndom - 1), OUTER_BC) / adm_inf;
    double COMy = space.get_domain(ndom - 1)->integ(syst.give_val_def("intCOMy")()(ndom - 1), OUTER_BC) / adm_inf;
    double COMz = space.get_domain(ndom - 1)->integ(syst.give_val_def("intCOMz")()(ndom - 1), OUTER_BC) / adm_inf;

    // ADM mass at infinite separation and binding energy
    double Minf = std::accumulate(madm.begin(), madm.end(), 0.);
    double e_bind = adm_inf - Minf;

    // Proper-volume L2 norm of the Hamiltonian-constraint violation summed over
    // every domain: integrate Hcon^2 against the physical measure sqrt(gamma) =
    // P^6 (conformally flat XCTS), not the bare coordinate volume. All domain
    // types, including the bispherical zones, implement integ_volume.
    double hcon_L2 = 0.;
    for (int d = 0; d < ndom; ++d) {
        syst.add_def(d, "Hcon2pv = Hcon2 * P^6");
        hcon_L2 += syst.give_val_def("Hcon2pv")()(d).integ_volume();
    }
    hcon_L2 = sqrt(hcon_L2);

#define FORMAT1 std::setw(25) << std::right << std::setprecision(12) << std::fixed << std::showpos
#define FORMAT std::setw(25) << std::right << std::setprecision(5) << std::scientific << std::showpos

    const double M2km = reader_M2km;

    auto print_vec = [&](auto& vec) {
        for (auto& e : vec) {
            std::cout << e << ",";
        }
    };

    auto res_r = space.get_domain(0)->get_nbr_points()(0);
    auto res_t = space.get_domain(0)->get_nbr_points()(1);
    auto res_p = space.get_domain(0)->get_nbr_points()(2);

    std::string header(22, '#');
    for (int i = 0; i <= 2; ++i) {
        std::cout << header + objects[i].label + header + "\n";
        std::cout << FORMAT1 << "Center = " << "(" << x_nuc[i] << ", 0, 0)\n"
                  << FORMAT1 << "Coord R_IN = " << bco_utils::get_radius(space.get_domain(nuc_doms[i]), EQUI) << '\n'
                  << FORMAT1 << "Coord R = " << "[" << r_extrema[i][0] << "," << r_extrema[i][1] << "] ("
                  << "[" << r_extrema[i][0] * M2km << "," << r_extrema[i][1] * M2km << "] km)\n";
        std::cout << FORMAT1 << "Coord R_OUT = " << bco_utils::get_radius(space.get_domain(adapted_doms[i] + 1), EQUI) << "\n"
                  << FORMAT1 << "Areal R = " << areal_r[i] << " [" << areal_r[i] * M2km << "km]\n"
                  << FORMAT1 << "Baryonic Mass = " << mbs[i] << " (";
        print_vec(mb_distro[i]);
        std::cout << ")\n"
                  << FORMAT1 << "Isolated ADM Mass = " << madm[i] << "\n"
                  << FORMAT1 << "Quasi-local Madm = " << ql_madm[i]
                  << " Diff:" << std::fabs(1. - ql_madm[i] / madm[i]) << std::endl
                  << FORMAT1 << "x(max(Density)) = " << x_max[i] << " (" << (x_nuc[i] - x_max[i]) / x_nuc[i] << ")\n"
                  << FORMAT << "Central Density = " << central_rho[i] << std::endl
                  << FORMAT << "Central log(h) = " << central_logh[i] << std::endl
                  << FORMAT << "Central Pressure = " << central_P[i] << std::endl
                  << FORMAT << "Central varscal = " << reader_na_text() << std::endl
                  << FORMAT << "Central dlog(h)/dx = " << central_dHdx[i] << std::endl
                  << FORMAT1 << "Central Euler Constant = " << central_euler[i] << std::endl
                  << FORMAT1 << "Integrated log(h) = " << H_int[i] << "\n\n";
    }

    std::cout << header + " Three-body " + header + '\n'
              << FORMAT1 << std::fixed << "RES = " << "[" << res_r << "," << res_t << "," << res_p << "]\n";
    reader_print_universal_gravity_block<Formalism>(bconfig);
    std::cout << FORMAT << "Outer varscal = " << "[" << reader_na_text() << ", " << reader_na_text() << "]"
              << std::endl
              << FORMAT1 << std::setprecision(2) << "Parent separation = " << bconfig(PARENT_DIST)
              << " (" << bconfig(PARENT_DIST) * M2km << "km)" << std::endl
              << FORMAT1 << std::setprecision(2) << "Child separation = " << bconfig(CHILD_DIST)
              << " (" << bconfig(CHILD_DIST) * M2km << "km)" << std::endl
              << FORMAT1 << "Global Omega = " << bconfig(TRI_GOMEGA) << std::endl
              << FORMAT1 << "Komar mass = " << komar << std::endl
              << FORMAT1 << "Adm mass = " << adm_inf << ", Diff: " << fabs(adm_inf - komar) / (adm_inf + komar) << std::endl
              << FORMAT1 << "Total mass (Minf) = " << Minf << std::endl
              << FORMAT1 << "Adm moment. = " << Jinf << std::endl
              << FORMAT << "Binding energy = " << e_bind << std::endl
              << FORMAT << "E_b / Minf = " << e_bind / Minf << std::endl
              << FORMAT << "Px = " << Px << std::endl
              << FORMAT << "Py = " << Py << std::endl
              << FORMAT << "Pz = " << Pz << std::endl
              << FORMAT1 << "A-COMx = " << COMx << std::endl
              << FORMAT1 << "A-COMy = " << COMy << std::endl
              << FORMAT1 << "A-COMz = " << COMz << std::endl
              << FORMAT << "Hamiltonian H (L2) = " << hcon_L2
              << std::endl;

#undef FORMAT1
#undef FORMAT
}

// Three-body reader entry point: single instantiation site for a chosen
// (eos_t, space_t, Formalism) triple.
template <class eos_t, typename space_t, typename Formalism>
int reader_three_body_main(kadath_config<TRI_INFO> bconfig)
{
    reader_three_body_output<eos_t, space_t, Formalism>(bconfig);
    return 0;
}

} // namespace KadathApps
