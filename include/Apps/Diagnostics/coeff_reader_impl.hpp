#pragma once

#include "Apps/Diagnostics/xz_snapshot_impl.hpp" 
#include "Apps/AMR/bns_hp_indicators.hpp"
#include "Apps/Startup/solver_startup.hpp"

#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/Config/config_three_body.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"

#include "mpi.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace KadathApps
{
namespace coeff_reader_detail
{
using Kadath::Dim_array;
using Kadath::Index;
using Kadath::Scalar;
using Kadath::Val_domain;
using Kadath::Vector;
using Kadath::bns_hp::SpectralTailOptions;
using Kadath::bns_hp::SpectralTailRatio;
using Kadath::bns_hp::TailAxisSelection;

// Production AMR default tail width (top-w modes). Matches bns_hp_indicators and
// the live refinement indicator, so the ratios printed here equal the trigger.
constexpr int tail_width = 2;

// One solved field to be dumped. Exactly one of scalar / vector is non-null.
struct NamedField
{
    const char* name = nullptr;
    const Scalar* scalar = nullptr;
    const Vector* vector = nullptr;
};

inline const char* axis_meaning_comment()
{
    return "#   spheric / compactified domain : 0=r    1=theta  2=phi\n"
           "#   bispheric domain              : 0=chi  1=eta    2=phi";
}

// The Val_domain of every component of a field in one domain (one entry for a
// Scalar, three for a Vector). The returned pointers alias the field objects,
// which the caller keeps alive for the whole dump.
inline std::vector<const Val_domain*> field_components(const NamedField& field, int domain)
{
    std::vector<const Val_domain*> components;
    if (field.scalar != nullptr) {
        components.push_back(&(*field.scalar)(domain));
    } else {
        for (int component = 1; component <= field.vector->get_n_comp(); ++component)
            components.push_back(&(*field.vector)(component)(domain));
    }
    return components;
}

// Per-axis coefficient energy in one domain: energy[axis][mode] is the square
// root of the sum of coef^2 over the OTHER axes (and, for a vector, over its
// components) at the given mode along `axis`. This is the per-mode marginal the
// AMR spectral-tail ratio integrates — a flat tail here is what the indicator
// flags. Structural cos/sin-packing zeros stay 0 (downstream tooling masks them).
inline std::array<std::vector<double>, 3> field_axis_energy(const NamedField& field, int domain)
{
    const std::vector<const Val_domain*> components = field_components(field, domain);
    const Dim_array dimensions = components.front()->get_domain()->get_nbr_coefs();
    const int naxes = dimensions.get_ndim();

    std::array<std::vector<double>, 3> energy;
    for (int axis = 0; axis < 3; ++axis)
        energy[static_cast<std::size_t>(axis)].assign(axis < naxes ? dimensions(axis) : 0, 0.0);

    for (const Val_domain* value : components) {
        value->coef();
        Index position(dimensions);
        do {
            const double coefficient = value->get_coef(position);
            const double square = coefficient * coefficient;
            for (int axis = 0; axis < naxes; ++axis)
                energy[static_cast<std::size_t>(axis)][static_cast<std::size_t>(position(axis))] += square;
        } while (position.inc());
    }

    for (int axis = 0; axis < naxes; ++axis)
        for (double& value : energy[static_cast<std::size_t>(axis)])
            value = std::sqrt(value);
    return energy;
}

// Spectral-tail ratio (the live AMR indicator value) for one field, axis, domain.
// Scalars use the scalar indicator; vectors use the component-aggregate one — the
// same calls the production refinement trigger makes.
inline SpectralTailRatio field_axis_ratio(const NamedField& field, int axis, int domain)
{
    SpectralTailOptions options;
    options.tail_width = tail_width;
    options.norm_floor = 0.0;
    options.axes = axis == 0   ? TailAxisSelection::RadialOnly
                   : axis == 1 ? TailAxisSelection::ThetaOnly
                               : TailAxisSelection::PhiOnly;

    const std::vector<SpectralTailRatio> ratios =
        field.scalar != nullptr
            ? Kadath::bns_hp::scalar_spectral_tail_ratios(*field.scalar, options)
            : Kadath::bns_hp::tensor_aggregate_spectral_tail_ratios(*field.vector, options);

    for (const SpectralTailRatio& ratio : ratios)
        if (ratio.domain == domain)
            return ratio;
    return SpectralTailRatio{};
}

// Emit the whole report to stdout: a self-describing banner, then the RATIO
// summary section, then the COEF spectrum section. Both record types are
// comma-separated and greppable by their leading tag, so a downstream awk filter
// (e.g. field==shift && domain==8) reproduces a single series exactly.
inline void emit_report(int ndom, const std::vector<NamedField>& fields,
                        const std::string& spacefile)
{
    std::cout << "# ============================================================================\n"
              << "#  coeff_read — spectral coefficient spectrum of every solved field, every domain\n"
              << "# ============================================================================\n"
              << "#  dataset : " << spacefile << "\n"
              << "#  domains : " << ndom << "      fields :";
    for (const NamedField& field : fields)
        std::cout << ' ' << field.name;
    std::cout << "\n#\n"
              << "#  Spectral axes are each domain's own collocation directions, index 0,1,2:\n"
              << axis_meaning_comment() << "\n"
              << "#  energy(field,dom,axis,m) = sqrt( sum of coef^2 over the OTHER axes and, for a\n"
              << "#    vector field, over its components ) at mode m along `axis`.\n"
              << "#\n"
              << "#  Two comma-separated record types, greppable by their leading tag:\n"
              << "#    RATIO,<field>,<domain>,<axis>,<l2_ratio>,<linf_ratio>,<tail_modes>,<total_modes>\n"
              << "#    COEF,<field>,<domain>,<axis>,<mode>,<energy>\n"
              << "#  RATIO is the per-(field,domain,axis) tail-to-norm summary (the live AMR\n"
              << "#  indicator value); COEF is the full per-mode spectrum it is computed from.\n"
              << "# ============================================================================\n";

    std::cout << "# --- spectral-tail ratio summary (the AMR indicator) ---\n";
    for (const NamedField& field : fields)
        for (int domain = 0; domain < ndom; ++domain)
            for (int axis = 0; axis < 3; ++axis) {
                const SpectralTailRatio ratio = field_axis_ratio(field, axis, domain);
                std::cout << "RATIO," << field.name << ',' << domain << ',' << axis << ','
                          << std::scientific << std::setprecision(6) << ratio.l2_ratio << ','
                          << ratio.linf_ratio << std::defaultfloat << ',' << ratio.tail_modes << ','
                          << ratio.total_modes << '\n';
            }

    std::cout << "# --- coefficient energy spectrum (every field, domain, axis, mode) ---\n";
    for (const NamedField& field : fields)
        for (int domain = 0; domain < ndom; ++domain) {
            const std::array<std::vector<double>, 3> energy = field_axis_energy(field, domain);
            for (int axis = 0; axis < 3; ++axis)
                for (std::size_t mode = 0; mode < energy[static_cast<std::size_t>(axis)].size(); ++mode)
                    std::cout << "COEF," << field.name << ',' << domain << ',' << axis << ',' << mode
                              << ',' << std::scientific << std::setprecision(8)
                              << energy[static_cast<std::size_t>(axis)][mode] << std::defaultfloat << '\n';
        }
}

inline const char* usage()
{
    return "Usage: ./coeff_read /<path>/<ID base name>.toml|.dat  e.g. ./coeff_read converged.9.toml";
}
} // namespace coeff_reader_detail

// Binary compact-object coefficient dump (Space_bin_ns / Space_bin_ns_nosym /
// Space_bhns / Space_bhns_nosym). Field-load order mirrors xz_snapshot_binary_main.
template <typename space_t, typename FieldTraits>
int coeff_reader_binary_main(int argc, char** argv)
{
    using namespace Kadath;
    using namespace KadathApps::coeff_reader_detail;

    KadathApps::init_mpi(argc, argv);

    return KadathApps::guarded_run([&] {
        if (argc < 2)
            KADATH_THROW(usage());

        const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<BIN_INFO> bconfig{ifilename};

        const std::string spacein = bconfig.space_filename();
        BeFileSource ff1(spacein);
        space_t space(ff1);
        Scalar conf(space, ff1);  // P : conformal factor
        Scalar lapse(space, ff1); // N : lapse
        Vector shift(space, ff1); // bet : shift
        Scalar logh(space, ff1);  // H : log enthalpy
        Scalar phi(space, ff1);   // phi : velocity potential

        Scalar scalar_slot(space);
        if constexpr (FieldTraits::has_varscal) {
            scalar_slot = Scalar(space, ff1);
        }
        Scalar xiscal_slot(space);
        if constexpr (FieldTraits::has_varscal) {
            FieldTraits::convert_loaded_scalar_slot(bconfig, space, ff1, scalar_slot, xiscal_slot);
        }

        std::vector<NamedField> fields{
            {"conf", &conf, nullptr},   {"lapse", &lapse, nullptr}, {"shift", nullptr, &shift},
            {"logh", &logh, nullptr},   {"phi", &phi, nullptr}};
        if constexpr (FieldTraits::has_varscal)
            fields.push_back({"varscal", &scalar_slot, nullptr});
        if constexpr (FieldTraits::has_xiscal)
            fields.push_back({"xiScal", &xiscal_slot, nullptr});

        emit_report(space.get_nbr_domains(), fields, spacein);

        MPI_Finalize();
    });
}

// Isolated neutron-star coefficient dump (Space_spheric_adapted /
// Space_spheric_adapted_nosym). phi is read only when the PHI flag is set (the
// binary-boosted output); otherwise it is absent from the dataset and the field
// list. Field-load order mirrors xz_snapshot_ns_main.
template <typename space_t, typename FieldTraits>
int coeff_reader_ns_main(int argc, char** argv)
{
    using namespace Kadath;
    using namespace KadathApps::coeff_reader_detail;

    KadathApps::init_mpi(argc, argv);

    return KadathApps::guarded_run([&] {
        if (argc < 2)
            KADATH_THROW(usage());

        const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<BCO_NS_INFO> bconfig{ifilename};

        const std::string spacein = bconfig.space_filename();
        BeFileSource ff1(spacein);
        space_t space(ff1);
        Scalar conf(space, ff1);
        Scalar lapse(space, ff1);
        Vector shift(space, ff1);
        Scalar logh(space, ff1);

        // phi exists only in the binary-boosted stage output; norot / uniform_rot
        // never save it. Reading a slot the file lacks would run ff1 past EOF.
        const bool has_phi = bconfig.field(PHI);
        Scalar phi(space);
        if (has_phi)
            phi = Scalar(space, ff1);

        Scalar scalar_slot(space);
        if constexpr (FieldTraits::has_varscal) {
            scalar_slot = Scalar(space, ff1);
        }
        Scalar xiscal_slot(space);
        if constexpr (FieldTraits::has_varscal) {
            FieldTraits::convert_loaded_scalar_slot(bconfig, space, ff1, scalar_slot, xiscal_slot);
        }

        std::vector<NamedField> fields{
            {"conf", &conf, nullptr}, {"lapse", &lapse, nullptr}, {"shift", nullptr, &shift},
            {"logh", &logh, nullptr}};
        if (has_phi)
            fields.push_back({"phi", &phi, nullptr});
        if constexpr (FieldTraits::has_varscal)
            fields.push_back({"varscal", &scalar_slot, nullptr});
        if constexpr (FieldTraits::has_xiscal)
            fields.push_back({"xiScal", &xiscal_slot, nullptr});

        emit_report(space.get_nbr_domains(), fields, spacein);

        MPI_Finalize();
    });
}

// Three-body coefficient dump (Space_three_body). The static stage saves only
// {conf, lapse, shift, logh} — no velocity potential — so phi is never read (a
// stale phi=true flag copied from a binary config must not drive a read past
// EOF). Field-load order mirrors xz_snapshot_three_body_main.
template <typename space_t, typename FieldTraits>
int coeff_reader_three_body_main(int argc, char** argv)
{
    using namespace Kadath;
    using namespace KadathApps::coeff_reader_detail;

    KadathApps::init_mpi(argc, argv);

    return KadathApps::guarded_run([&] {
        if (argc < 2)
            KADATH_THROW(usage());

        const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<TRI_INFO> bconfig{ifilename};

        const std::string spacein = bconfig.space_filename();
        BeFileSource ff1(spacein);
        space_t space(ff1);
        Scalar conf(space, ff1);
        Scalar lapse(space, ff1);
        Vector shift(space, ff1);
        Scalar logh(space, ff1);

        Scalar scalar_slot(space);
        if constexpr (FieldTraits::has_varscal) {
            scalar_slot = Scalar(space, ff1);
        }
        Scalar xiscal_slot(space);
        if constexpr (FieldTraits::has_varscal) {
            FieldTraits::convert_loaded_scalar_slot(bconfig, space, ff1, scalar_slot, xiscal_slot);
        }

        std::vector<NamedField> fields{
            {"conf", &conf, nullptr}, {"lapse", &lapse, nullptr}, {"shift", nullptr, &shift},
            {"logh", &logh, nullptr}};
        if constexpr (FieldTraits::has_varscal)
            fields.push_back({"varscal", &scalar_slot, nullptr});
        if constexpr (FieldTraits::has_xiscal)
            fields.push_back({"xiScal", &xiscal_slot, nullptr});

        emit_report(space.get_nbr_domains(), fields, spacein);

        MPI_Finalize();
    });
}

} // namespace KadathApps
