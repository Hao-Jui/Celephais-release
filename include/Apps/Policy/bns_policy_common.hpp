#pragma once

#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Tensor/vector.hpp"
#include "Hydro/EOS.hh"
#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mpi.h>
#include <string>
#include <utility>

namespace KadathApps::BnsPolicyCommon
{
    template <typename eos_t>
    struct EosType
    {
        using type = eos_t;
    };

    template <typename DistanceCheck>
    void setup_binary_ns_config(kadath_config<BIN_INFO>& bconfig, DistanceCheck&& check_distance)
    {
        // Pass each star's mass AND surface radius (RMID) so the check can use a
        // geometric / Roche criterion instead of a bare mass multiple.
        std::forward<DistanceCheck>(check_distance)(
            bconfig(DIST),
            bconfig(MADM, BCO1), bconfig(RMID, BCO1),
            bconfig(MADM, BCO2), bconfig(RMID, BCO2));

        bconfig.set(REXT) = 2 * bconfig(DIST);
        bconfig.set(Q) = bconfig(MADM, BCO2) / bconfig(MADM, BCO1);
        bconfig.set(COM) =
            bco_utils::com_estimate(bconfig(DIST), bconfig(MADM, BCO1), bconfig(MADM, BCO2));

        const auto pn_orbital_params =
            bco_utils::binary_pn_orbital_seed(bconfig, bconfig(MADM, BCO1), bconfig(MADM, BCO2));
        bconfig.set(ECC_OMEGA) = pn_orbital_params.omega;
        bconfig.set(ADOT) = pn_orbital_params.adot;
        bconfig.set(GOMEGA) = pn_orbital_params.omega;
        bconfig.set(ADOT) = std::nan("1");
    }

    // BNS solvers expose solve(); BHNS keeps the historical solve(bool)
    // signature at the app boundary. The variadic pack forwards either call
    // shape through one definition.
    template <typename SpaceT, template <class, typename, typename> class SolverT, typename... WarmupArg>
    int solve_xcts(kadath_config<BIN_INFO>& bconfig, const std::string& outputdir, WarmupArg... warmup)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        const std::string spacein = bconfig.space_filename();

        if (rank == 0) {
            std::cout << "Config File: " << bconfig.config_outputdir() + bconfig.config_filename() << std::endl
                      << "Fields File: " << spacein << std::endl
                      << bconfig << std::endl;
        }

        Kadath::BeFileSource ff1(spacein);
        SpaceT space(ff1);
        Kadath::Scalar conf(space, ff1);
        Kadath::Scalar lapse(space, ff1);
        Kadath::Vector shift(space, ff1);
        Kadath::Scalar logh(space, ff1);
        Kadath::Scalar phi(space, ff1);
        Kadath::Base_tensor basis(space, CARTESIAN_BASIS);

        if (!outputdir.empty()) {
            std::filesystem::create_directory(outputdir);
            bconfig.set_outputdir(outputdir);
        }

        const double h_cut = bconfig.template eos<double>(HCUT, BCO1);
        const std::string eos_file = bconfig.template eos<std::string>(EOSFILE, BCO1);
        const std::string eos_type = bconfig.template eos<std::string>(EOSTYPE, BCO1);

        auto run_solver = [&](auto eos_tag, auto&&... eos_args) -> int {
            using eos_t = typename decltype(eos_tag)::type;
            ::EOS<eos_t, ::eos_var_t::PRESSURE>::init(
                eos_file, h_cut, std::forward<decltype(eos_args)>(eos_args)...);
            SolverT<eos_t, decltype(bconfig), decltype(space)>
                bns_solver(bconfig, space, basis, conf, lapse, shift, logh, phi);
            return bns_solver.solve(warmup...);
        };

        if (eos_type == "Cold_PWPoly") {
            return run_solver(EosType<Kadath::Margherita::Cold_PWPoly>{});
        } else if (eos_type == "Cold_Table") {
            const int interp_pts = bconfig.template eos<int>(INTERP_PTS, BCO1) == 0
                ? 2000
                : bconfig.template eos<int>(INTERP_PTS, BCO1);
            return run_solver(EosType<Kadath::Margherita::Cold_Table>{}, interp_pts,
                              bconfig.template eos<double>(MNUC_CGS, BCO1));
        } else {
            KADATH_THROW("Unknown EOSTYPE.");
        }

        return EXIT_FAILURE;
    }

    template <typename SpaceT, template <class, typename, typename> class SolverT>
    int solve_xcts_def(kadath_config<BIN_INFO>& bconfig, const std::string& outputdir)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        const std::string spacein = bconfig.space_filename();

        if (rank == 0) {
            std::cout << "Config File: " << bconfig.config_outputdir() + bconfig.config_filename() << std::endl
                      << "Fields File: " << spacein << std::endl
                      << bconfig << std::endl;
        }

        Kadath::BeFileSource ff1(spacein);
        SpaceT space(ff1);
        Kadath::Scalar conf(space, ff1);
        Kadath::Scalar lapse(space, ff1);
        Kadath::Vector shift(space, ff1);
        Kadath::Scalar logh(space, ff1);
        Kadath::Scalar phi(space, ff1);
        Kadath::Scalar xiscal(space, ff1); // = saved xiScal == the Newton unknown
        Kadath::Base_tensor basis(space, CARTESIAN_BASIS);
        // The disk slot IS the unknown xiScal; hand it straight to the solver.
        // The file's SWEIGHT flag only says whether a saved Sweight trails the
        // slot on disk; read it past for stream alignment and discard it (the
        // solver rebuilds its active weight from the current TOML massScal).
        if (bconfig.set_field(SWEIGHT) == true) {
            Kadath::Scalar saved_Sweight(space, ff1);
            (void)saved_Sweight;
        }

        if (!outputdir.empty()) {
            std::filesystem::create_directory(outputdir);
            bconfig.set_outputdir(outputdir);
        }

        const double h_cut = bconfig.template eos<double>(HCUT, BCO1);
        const std::string eos_file = bconfig.template eos<std::string>(EOSFILE, BCO1);
        const std::string eos_type = bconfig.template eos<std::string>(EOSTYPE, BCO1);

        auto run_solver = [&](auto eos_tag, auto&&... eos_args) -> int {
            using eos_t = typename decltype(eos_tag)::type;
            ::EOS<eos_t, ::eos_var_t::PRESSURE>::init(
                eos_file, h_cut, std::forward<decltype(eos_args)>(eos_args)...);
            SolverT<eos_t, decltype(bconfig), decltype(space)>
                bns_solver(bconfig, space, basis, conf, lapse, shift, logh, phi, xiscal);
            return bns_solver.solve();
        };

        if (eos_type == "Cold_PWPoly") {
            return run_solver(EosType<Kadath::Margherita::Cold_PWPoly>{});
        } else if (eos_type == "Cold_Table") {
            const int interp_pts = bconfig.template eos<int>(INTERP_PTS, BCO1) == 0
                ? 2000
                : bconfig.template eos<int>(INTERP_PTS, BCO1);
            return run_solver(EosType<Kadath::Margherita::Cold_Table>{}, interp_pts,
                              bconfig.template eos<double>(MNUC_CGS, BCO1));
        } else {
            KADATH_THROW("Unknown EOSTYPE.");
        }

        return EXIT_FAILURE;
    }

    template <typename BoostedSetup>
    void superimposed_import(kadath_config<BIN_INFO>& bconfig,
                             const std::array<std::string, 2>& ns_filenames,
                             BoostedSetup&& boosted_setup)
    {
        kadath_config<BCO_NS_INFO> ns1_config(ns_filenames[0]);
        kadath_config<BCO_NS_INFO> ns2_config(ns_filenames[1]);

        const double h_cut = ns1_config.eos<double>(HCUT);
        const std::string eos_file = ns1_config.eos<std::string>(EOSFILE);
        const std::string eos_type = ns1_config.eos<std::string>(EOSTYPE);

        auto setup = [&](auto eos_tag, auto&&... eos_args) {
            using eos_t = typename decltype(eos_tag)::type;
            ::EOS<eos_t, ::eos_var_t::PRESSURE>::init(
                eos_file, h_cut, std::forward<decltype(eos_args)>(eos_args)...);
            std::forward<BoostedSetup>(boosted_setup)(eos_tag, ns1_config, ns2_config, bconfig);
        };

        if (eos_type == "Cold_PWPoly") {
            setup(EosType<Kadath::Margherita::Cold_PWPoly>{});
        } else if (eos_type == "Cold_Table") {
            const int interp_pts = ns1_config.eos<int>(INTERP_PTS) == 0
                ? 2000
                : ns1_config.eos<int>(INTERP_PTS);
            setup(EosType<Kadath::Margherita::Cold_Table>{}, interp_pts,
                  ns1_config.eos<double>(MNUC_CGS));
        } else {
            KADATH_THROW("Unknown EOSTYPE.");
        }
    }
} // namespace KadathApps::BnsPolicyCommon
