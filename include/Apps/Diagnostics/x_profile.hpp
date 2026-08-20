#pragma once


#include "Apps/Startup/solver_startup.hpp"
#include "For_Kadath/Array/index.hpp"
#include "For_Kadath/Config/config_bco.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "celephais_paths.h"
#include "mpi.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <unistd.h>

namespace KadathApps
{
    namespace x_profile_detail
    {
        using NamedScalar = std::pair<std::string, Kadath::Scalar>;
        using NamedScalars = std::vector<NamedScalar>;

        class TemporaryFileGuard
        {
            std::filesystem::path path;

          public:
            explicit TemporaryFileGuard(std::filesystem::path temporary_path) : path(std::move(temporary_path)) {}

            TemporaryFileGuard(const TemporaryFileGuard&) = delete;
            TemporaryFileGuard& operator=(const TemporaryFileGuard&) = delete;

            ~TemporaryFileGuard()
            {
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
            }
        };

        template <typename space_t>
        NamedScalars load_scalars(space_t& space, Kadath::BeFileSource& source,
                                  std::initializer_list<std::string_view> names)
        {
            NamedScalars fields;
            fields.reserve(names.size());
            for (const std::string_view name : names)
                fields.emplace_back(std::string{name}, Kadath::Scalar(space, source));
            return fields;
        }

        template <typename config_t> void require_gravity_theory(const config_t& config, std::string_view expected)
        {
            if (!config.has_gravity_setting(GRAV_THEORY)) {
                if (expected == "GR")
                    return;
                KADATH_THROW("x_proflle: expected [gravity] theory = \"" + std::string{expected} +
                             "\", but the setting is absent.");
            }

            const std::string actual = config.template gravity<std::string>(GRAV_THEORY);
            if (actual != expected)
                KADATH_THROW("x_proflle: expected [gravity] theory = \"" + std::string{expected} + "\", got \"" +
                             actual + "\".");
        }

        inline std::filesystem::path output_path()
        {
            const char* runtime_root = std::getenv("HOME_CELEPHAIS");
            const std::filesystem::path repository_root = (runtime_root != nullptr && runtime_root[0] != '\0')
                                                              ? std::filesystem::path{runtime_root}
                                                              : std::filesystem::path{CELEPHAIS_ROOT_DIR};
            const std::filesystem::path output_directory = repository_root / "matlab" / "snapshot";

            std::error_code error;
            std::filesystem::create_directories(output_directory, error);
            if (error)
                KADATH_THROW("x_proflle: cannot create output directory: " + output_directory.string() + ": " +
                             error.message());

            return output_directory / "x_profile.dat";
        }

        template <typename space_t>
        void write_profile(const space_t& space, const NamedScalars& fields, int first_physical_domain)
        {
            const int domain_count = space.get_nbr_domains();
            if (fields.empty())
                KADATH_THROW("x_proflle: saved-field layout is empty.");
            if (first_physical_domain < 0 || first_physical_domain >= domain_count)
                KADATH_THROW("x_proflle: invalid first physical domain.");

            const std::filesystem::path path = output_path();
            std::filesystem::path temporary_path = path;
            temporary_path += ".tmp." + std::to_string(static_cast<long long>(::getpid()));
            const TemporaryFileGuard temporary_file{temporary_path};

            std::ofstream output{temporary_path};
            if (!output)
                KADATH_THROW("x_proflle: cannot open temporary output file: " + temporary_path.string());

            output << "% dom x";
            for (const auto& [name, field] : fields) {
                (void)field;
                output << ' ' << name;
            }
            output << '\n' << std::scientific << std::setprecision(16);

            std::size_t rows_written = 0;
            for (int domain = first_physical_domain; domain < domain_count; ++domain) {
                const Kadath::Domain* polar_domain = space.get_domain(domain);
                const Kadath::Dim_array& point_count = polar_domain->get_nbr_points();
                if (point_count.get_ndim() < 2)
                    KADATH_THROW("x_proflle: expected a two-dimensional polar domain.");

                const int radial_count = point_count(0);
                const int theta_count = point_count(1);
                if (radial_count < 2 || theta_count < 2)
                    KADATH_THROW("x_proflle: each polar axis requires at least two collocation points.");

                // Keep one copy of each shared interface. The final point of the final
                // compactified domain represents spatial infinity and has no finite x.
                const int radial_begin = (domain == first_physical_domain) ? 0 : 1;
                const int radial_end = radial_count - ((domain == domain_count - 1) ? 1 : 0);
                const int equator = theta_count - 1;
                const Kadath::Val_domain cartesian_x = polar_domain->get_cart(1);

                for (int radial = radial_begin; radial < radial_end; ++radial) {
                    Kadath::Index position(point_count);
                    position.set_start();
                    position.set(0) = radial;
                    position.set(1) = equator;

                    const double x = cartesian_x(position);
                    if (!std::isfinite(x))
                        KADATH_THROW("x_proflle: encountered a non-finite x coordinate.");

                    output << domain << ' ' << x;
                    for (const auto& [name, field] : fields) {
                        (void)name;
                        output << ' ' << field(domain)(position);
                    }
                    output << '\n';
                    ++rows_written;
                }
            }

            if (rows_written == 0)
                KADATH_THROW("x_proflle: no finite x-axis collocation points found.");

            output.close();
            if (!output)
                KADATH_THROW("x_proflle: failed while writing temporary output file: " + temporary_path.string());

            std::error_code rename_error;
            std::filesystem::rename(temporary_path, path, rename_error);
            if (rename_error)
                KADATH_THROW("x_proflle: cannot replace output file: " + path.string() + ": " + rename_error.message());

            std::cout << "x_proflle: wrote " << rows_written << " points to " << path << '\n';
        }
    } // namespace x_profile_detail

    // Legacy NS2d/MSQI layout: (conf, lapse, shift, metQ, logh, Omg).
    struct Ns2dMsqiProfileFields {
        template <typename config_t> static void validate(const config_t& config)
        {
            x_profile_detail::require_gravity_theory(config, "GR");
        }

        template <typename config_t, typename space_t>
        static x_profile_detail::NamedScalars load(const config_t&, space_t& space, Kadath::BeFileSource& source)
        {
            return x_profile_detail::load_scalars(space, source, {"conf", "lapse", "shift", "metQ", "logh", "Omg"});
        }
    };

    // Modern GR XCTS layout: (conf, lapse, shift, logh, Omg).
    struct Ns2dXctsGrProfileFields {
        template <typename config_t> static void validate(const config_t& config)
        {
            x_profile_detail::require_gravity_theory(config, "GR");
        }

        template <typename config_t, typename space_t>
        static x_profile_detail::NamedScalars load(const config_t&, space_t& space, Kadath::BeFileSource& source)
        {
            return x_profile_detail::load_scalars(space, source, {"conf", "lapse", "shift", "logh", "Omg"});
        }
    };


    // BH2d/MSQI layout: (lapse, metA, metB, beta, varscal).
    struct Bh2dMsqiProfileFields {
        template <typename config_t, typename space_t>
        static x_profile_detail::NamedScalars load(const config_t&, space_t& space, Kadath::BeFileSource& source)
        {
            return x_profile_detail::load_scalars(space, source, {"lapse", "metA", "metB", "beta", "varscal"});
        }
    };

    template <typename space_t, typename FieldLayout> int x_profile_ns_main(int argc, char** argv)
    {
        const int rank = KadathApps::init_mpi(argc, argv);
        const int result = KadathApps::guarded_run([&] {
            if (argc < 2) {
                KADATH_THROW("Usage: ./x_proflle /<path>/<ID base name>.toml|.dat  "
                             "e.g. ./x_proflle converged_NS_2D.toml");
            }
            if (rank != 0)
                return;

            const std::string config_path = KadathApps::toml_config_path_from_reader_input(argv[1]);
            kadath_config<BCO_NS_INFO> config{config_path};
            if (config(DIM) != 2)
                KADATH_THROW("x_proflle: neutron-star input must have dim = 2.");
            FieldLayout::validate(config);

            Kadath::BeFileSource source(config.space_filename());
            space_t space(source);
            const x_profile_detail::NamedScalars fields = FieldLayout::load(config, space, source);
            x_profile_detail::write_profile(space, fields, 0);
        });
        MPI_Finalize();
        return result;
    }

    template <typename space_t, typename FieldLayout> int x_profile_bh_main(int argc, char** argv)
    {
        const int rank = KadathApps::init_mpi(argc, argv);
        const int result = KadathApps::guarded_run([&] {
            if (argc < 2) {
                KADATH_THROW("Usage: ./x_proflle /<path>/<ID base name>.toml|.dat  "
                             "e.g. ./x_proflle converged_BH.toml");
            }
            if (rank != 0)
                return;

            const std::string config_path = KadathApps::toml_config_path_from_reader_input(argv[1]);
            kadath_config<BCO_BH_INFO> config{config_path};
            if (config(DIM) != 2)
                KADATH_THROW("x_proflle: black-hole input must have dim = 2.");

            Kadath::BeFileSource source(config.space_filename());
            space_t space(source);
            const x_profile_detail::NamedScalars fields = FieldLayout::load(config, space, source);
            x_profile_detail::write_profile(space, fields, space.HOMOTHETIC_INNER);
        });
        MPI_Finalize();
        return result;
    }
} // namespace KadathApps
