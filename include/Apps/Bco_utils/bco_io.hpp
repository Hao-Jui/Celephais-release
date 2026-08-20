/*
 * This file is part of the KADATH library.
 * Author: Samuel Tootle
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/System_of_eqs/system_dof_record.hpp"
#include "For_Kadath/System_of_eqs/solver_runtime_config.hpp"
#include "Linear_algebra/mumps_tree_cache.hpp"
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

using namespace Kadath;

namespace bco_utils
{
    template <typename config_t>
    void cleanup_mumps_tree_cache_after_converged_save(const config_t& bconfig) noexcept
    {
        try {
            if (!SolverRuntimeConfig::from_environment().mumps.tree_cache_enabled)
                return;

            std::filesystem::path config_path(bconfig.config_filename_abs());
            const std::string basename = config_path.filename().string();
            if (!basename.starts_with("converged_") ||
                config_path.extension() != ".toml")
                return;

            config_path.replace_extension(".mumpstree");
            const std::filesystem::path active_path =
                active_mumps_tree_cache_path();
            const auto remove_after_save = [](const std::filesystem::path& path) {
                if (path.empty())
                    return;
                try {
                    (void)remove_mumps_tree_cache(path);
                } catch (const std::exception& error) {
                    std::cerr << "MUMPS tree cache cleanup failed for "
                              << path.string() << ": " << error.what()
                              << "; converged solution remains valid" << std::endl;
                }
            };
            remove_after_save(config_path);
            if (active_path != config_path)
                remove_after_save(active_path);
            clear_active_mumps_tree_cache_path();
        } catch (const std::exception& error) {
            std::cerr << "MUMPS tree cache cleanup failed: " << error.what()
                      << "; converged solution remains valid" << std::endl;
        } catch (...) {
            std::cerr << "MUMPS tree cache cleanup failed with an unknown error; "
                         "converged solution remains valid"
                      << std::endl;
        }
    }

    template <typename space_t, typename config_t, typename... fields_t>
    void save_to_file(std::stringstream& base_fname, space_t& space, config_t& bconfig, fields_t&... fields)
    {
        bconfig.set_filename(base_fname.str());
        bconfig.write_config();
        std::string kadath_filename = bconfig.space_filename();
        {
            BeFileSink sink(kadath_filename);
            space.save(sink);
            (fields.save(sink), ...);
            write_system_dof(sink);  // trailing tagged DOF of the solve that produced this dataset
        }
        cleanup_mumps_tree_cache_after_converged_save(bconfig);
    }

    template <typename space_t, typename config_t, typename... fields_t>
    void save_to_file(space_t& space, config_t& bconfig, fields_t&... fields)
    {
        bconfig.write_config();
        std::string kadath_filename = bconfig.space_filename();
        {
            BeFileSink sink(kadath_filename);
            space.save(sink);
            (fields.save(sink), ...);
            write_system_dof(sink);  // trailing tagged DOF of the solve that produced this dataset
        }
        cleanup_mumps_tree_cache_after_converged_save(bconfig);
    }
} // namespace bco_utils
