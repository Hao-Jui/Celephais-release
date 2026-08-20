#pragma once

#include "Apps/Diagnostics/configured_eos.hpp"
#include "Apps/Diagnostics/reader_impl.hpp"
#include "mpi.h"

#include <sstream>
#include <string>

namespace KadathApps
{

template <typename space_t, typename Formalism>
int reader_single_eos_entrypoint(int argc, char** argv, const char* usage)
{
    KadathApps::init_mpi(argc, argv);

    return KadathApps::guarded_run([&] {
        if (argc < 2) {
            KADATH_THROW(usage);
        }

        const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<BCO_NS_INFO> bconfig{ifilename};

        const double h_cut = bconfig.eos<double>(HCUT);
        const std::string eos_file = bconfig.eos<std::string>(EOSFILE);
        const std::string eos_type = bconfig.eos<std::string>(EOSTYPE);

        if (eos_type == "Cold_PWPoly") {
            using eos_t = Kadath::Margherita::Cold_PWPoly;
            EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);

            if (bconfig(DIM) == 3)
                KadathApps::reader_single_main<eos_t, space_t, Formalism>(bconfig);
        } else if (eos_type == "Cold_Table") {
            using eos_t = Kadath::Margherita::Cold_Table;
            KadathApps::init_configured_cold_table(bconfig);

            if (bconfig(DIM) == 3)
                KadathApps::reader_single_main<eos_t, space_t, Formalism>(bconfig);
        } else {
            KADATH_THROW("Unknown EOSTYPE.");
        }

        MPI_Finalize();
    });
}

template <typename space_t, typename Formalism,
          typename Validation = NoReaderValidation>
int reader_2d_eos_entrypoint(int argc, char** argv, const char* usage)
{
    KadathApps::init_mpi(argc, argv);

    return KadathApps::guarded_run([&] {
        if (argc < 2)
            KADATH_THROW(usage);

        const std::string ifilename =
            KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<BCO_NS_INFO> bconfig{ifilename};
        if (bconfig(DIM) != 2)
            KADATH_THROW("Axisymmetric NS reader requires dim = 2.");

        const double h_cut = bconfig.eos<double>(HCUT);
        const std::string eos_file = bconfig.eos<std::string>(EOSFILE);
        const std::string eos_type = bconfig.eos<std::string>(EOSTYPE);

        if (eos_type == "Cold_PWPoly") {
            using eos_t = Kadath::Margherita::Cold_PWPoly;
            EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
            KadathApps::reader_2d_main<
                eos_t, space_t, Formalism, Validation>(bconfig);
        } else if (eos_type == "Cold_Table") {
            using eos_t = Kadath::Margherita::Cold_Table;
            KadathApps::init_configured_cold_table(bconfig);
            KadathApps::reader_2d_main<
                eos_t, space_t, Formalism, Validation>(bconfig);
        } else {
            KADATH_THROW("Unknown EOSTYPE.");
        }

        MPI_Finalize();
    });
}

template <typename space_t, typename Formalism>
int reader_three_body_eos_entrypoint(int argc, char** argv, const char* usage)
{
    KadathApps::init_mpi(argc, argv);

    return KadathApps::guarded_run([&] {
        if (argc < 2) {
            KADATH_THROW(usage);
        }

        const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<TRI_INFO> bconfig{ifilename};

        const double h_cut = bconfig.eos<double>(HCUT, BCO1);
        const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
        const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);

        if (eos_type == "Cold_PWPoly") {
            using eos_t = Kadath::Margherita::Cold_PWPoly;
            EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
            KadathApps::reader_three_body_main<eos_t, space_t, Formalism>(bconfig);
        } else if (eos_type == "Cold_Table") {
            using eos_t = Kadath::Margherita::Cold_Table;
            KadathApps::init_configured_cold_table(bconfig, BCO1);
            KadathApps::reader_three_body_main<eos_t, space_t, Formalism>(bconfig);
        } else {
            KADATH_THROW("Unknown EOSTYPE.");
        }

        MPI_Finalize();
    });
}

template <typename space_t>
int reader_single_bh_entrypoint(int argc, char** argv, const char* usage)
{
    KadathApps::init_mpi(argc, argv);

    return KadathApps::guarded_run([&] {
        if (argc < 2) {
            KADATH_THROW(usage);
        }

        const std::string ifilename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<BCO_NS_INFO> bconfig{ifilename};
        KadathApps::reader_single_bh_main<space_t>(bconfig);

        MPI_Finalize();
    });
}

template <typename space_t, typename Formalism>
int reader_binary_eos_entrypoint(int argc, char** argv, const char* usage)
{
    using namespace Kadath;

    return KadathApps::guarded_run([&] {
        if (argc < 2) {
            KADATH_THROW(usage);
        }

        const std::string in_filename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<BIN_INFO> bconfig{in_filename};

        const double h_cut = bconfig.eos<double>(HCUT, BCO1);
        const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
        const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);

        if (eos_type == "Cold_PWPoly") {
            using eos_t = Kadath::Margherita::Cold_PWPoly;
            EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
            KadathApps::reader_binary_main<eos_t, space_t, Formalism>(bconfig);
        } else if (eos_type == "Cold_Table") {
            using eos_t = Kadath::Margherita::Cold_Table;
            KadathApps::init_configured_cold_table(bconfig, BCO1);
            KadathApps::reader_binary_main<eos_t, space_t, Formalism>(bconfig);
        } else {
            std::ostringstream oss;
            oss << eos_type << " is not recognized.";
            KADATH_THROW(oss.str());
        }
    });
}

template <typename space_t>
int reader_bhns_eos_entrypoint(int argc, char** argv, const char* usage)
{
    using namespace Kadath;

    return KadathApps::guarded_run([&] {
        if (argc < 2) {
            KADATH_THROW(usage);
        }

        const std::string in_filename = KadathApps::toml_config_path_from_reader_input(argv[1]);
        kadath_config<BIN_INFO> bconfig{in_filename};

        const double h_cut = bconfig.eos<double>(HCUT, BCO1);
        const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
        const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);

        if (eos_type == "Cold_PWPoly") {
            using eos_t = Kadath::Margherita::Cold_PWPoly;
            EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
            KadathApps::reader_bhns_main<eos_t, space_t>(bconfig);
        } else if (eos_type == "Cold_Table") {
            using eos_t = Kadath::Margherita::Cold_Table;
            KadathApps::init_configured_cold_table(bconfig, BCO1);
            KadathApps::reader_bhns_main<eos_t, space_t>(bconfig);
        } else {
            std::ostringstream oss;
            oss << eos_type << " is not recognized.";
            KADATH_THROW(oss.str());
        }
    });
}

} // namespace KadathApps
