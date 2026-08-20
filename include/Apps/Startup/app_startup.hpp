#pragma once

#include <cstdlib>
#include "Apps/Startup/solver_startup.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Config/configurator_toml.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

namespace KadathApps
{

    // Set an environment variable only if absent/empty — app policies use this
    // in configure_defaults() to install per-app solver defaults that the user
    // environment always overrides.
    inline void set_default_env(const char* name, const char* value)
    {
        const char* current = std::getenv(name);
        if (current == nullptr || current[0] == '\0') {
            ::setenv(name, value, 0);
        }
    }

    // AMR (adaptive_mesh_refinement) is implemented only by the binary BNS/BHNS
    // workflows (run_bns_amr_gate). Every app's config still serialises an
    // [adaptive_mesh_refinement] block, so setting enabled=true on an app with
    // no AMR gate would otherwise run to completion at fixed resolution with no
    // refinement and no error — a silent no-op. Fail loudly at startup instead.
    template <class RuntimeConfig>
    inline void throw_if_amr_unsupported(const RuntimeConfig& bconfig)
    {
        if (bconfig.template amr_setting_as<bool>(AMR_ENABLED))
            KADATH_THROW(
                "adaptive_mesh_refinement.enabled=true is not supported by this app; "
                "AMR is implemented only by the binary BNS/BHNS workflows");
    }

    template <class RuntimeConfig>
    StartupResult<RuntimeConfig> parse_kadath_config_toml_startup(int argc, char** argv, int rank)
    {
        StartupResult<RuntimeConfig> result{};
        result.outputdir = "./";
        result.example_setup = false;

        const std::string input_config_path{argv[1]};
        result.bconfig.set_filename(input_config_path);
        result.bconfig.open_config();

        if (std::filesystem::exists(result.bconfig.space_filename())) {
            if (rank == 0) {
                std::cout << "Solving based on previous solution: "
                          << input_config_path << std::endl;
            }
            result.setup_first = false;
        } else {
            if (rank == 0) {
                std::cout << "No dat file associated with : "
                          << input_config_path << std::endl;
                std::cout << "Creating a setup based on TOML config file..." << std::endl;
            }
            result.setup_first = true;
        }

        if (argc > 2) {
            result.outputdir = std::string{argv[2]};
        }

        return result;
    }

    template <class RuntimeConfig, class ConfigureExample>
    void write_example_toml(int rank, const std::string& path, ConfigureExample&& configure_example)
    {
        if (rank != 0) {
            return;
        }

        RuntimeConfig example_config;
        std::forward<ConfigureExample>(configure_example)(example_config);
        example_config.write_config(path);

        std::cout << "TOML config file missing - generated example setup: "
                  << path << std::endl
                  << "Modify as needed before rerunning `solve "
                  << path << " <outputdir>`\n";
    }

    template <class RuntimeConfig>
    void write_example_toml(int rank, const std::string& path)
    {
        KadathApps::write_example_toml<RuntimeConfig>(rank, path, [](RuntimeConfig& example_config) {
            example_config.set_defaults();
        });
    }
} // namespace KadathApps
