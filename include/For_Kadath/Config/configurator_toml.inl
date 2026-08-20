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

#include "configurator_toml.hpp"
#include "For_Kadath/Utilities/name_tools.hpp"

#include <algorithm>
#include <string>

// std::string::ends_with is C++20; this free helper keeps the .inl C++17-compatible.
inline bool ends_with(const std::string& str, const std::string& suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

template <typename ParamC> kadath_config<ParamC>::kadath_config()
{
    fields.fill(false);
    stages.fill(false);
    controls.fill(false);
    seq_settings.fill(seq_var_t{std::nan("1")});
    gravity_settings.fill(gravity_var_t{std::nan("1")});
    set_seq_defaults();
    set_amr_defaults();
}

template <typename ParamC> kadath_config<ParamC>::kadath_config(std::string ifile) : container{}
{
    seq_settings.fill(seq_var_t{std::nan("1")});
    gravity_settings.fill(gravity_var_t{std::nan("1")});
    set_seq_defaults();
    set_amr_defaults();
    this->set_filename(ifile);
    this->open_config();
}

template <typename ParamC> int kadath_config<ParamC>::open_config()
{
    tree = read_toml_config_tree(outputdir + filename);
    int status = -1;
    const std::string container_type = container.get_type();
    const bool is_composite = (container_type == "binary" || container_type == "three_body");
    auto key = MBCO().find(container_type);
    if (is_composite || key != MBCO().end()) {
        branch_in = read_branch(tree, container.get_type());
    } else {
        for (auto& e : MBCO()) {
            branch_in = read_branch(tree, e.first);
            if (!branch_in.empty()) {
                status = to_int(e.second);
                break;
            }
        }
    }
    if (branch_in.empty()) {
        std::cout << "No Node found. \n";
        return status;
    }

    container.read_params(branch_in);

    // EOS is a shared top-level [eos] block distributed to every compact object.
    // Legacy configs stored the EOS per-object (already read by read_params
    // above), so this is skipped when [eos] is absent — those remain valid and
    // are migrated to the [eos] block on the next write_config.
    if (tree.find("eos") != tree.not_found()) {
        auto eos_branch = read_branch(tree, "eos");
        container.read_eos_branch(eos_branch);
    }

    read_keys(MBCO_FIELDS(), fields, read_branch(tree, "fields"));

    // Stages: legacy v1 keys (norot_bc/total/total_bc) are app-ambiguous and
    // not aliased. read_keys would silently ignore them (no stage enabled), so
    // reject loudly instead — developer/migrate_stage_filenames.sh migrates a
    // dataset in place (renames artifacts and rewrites [stages] keys).
    {
        const auto stage_branch = read_branch(tree, "stages");
        for (const char* legacy_key : {"norot_bc", "total", "total_bc"}) {
            if (stage_branch.find(legacy_key) != stage_branch.not_found()) {
                KADATH_THROW("Legacy v1 stage key '" + std::string(legacy_key) + "' in " + filename +
                             "; run developer/migrate_stage_filenames.sh on its directory.");
            }
        }
        read_keys(MSTAGE(), stages, stage_branch);
    }

    if (tree.find("sequence_controls") != tree.not_found())
        read_keys(MCONTROLS(), controls, read_branch(tree, "sequence_controls"));

    if (tree.find("sequence_settings") != tree.not_found())
        read_keys(MSEQ_SETTINGS(), seq_settings, read_branch(tree, "sequence_settings"));

    if (tree.find("adaptive_mesh_refinement") != tree.not_found())
        read_keys(MAMR_SETTINGS(), amr_settings, read_branch(tree, "adaptive_mesh_refinement"));

    if (tree.find("gravity") != tree.not_found())
        read_keys(MGRAVITY_PARAMS(), gravity_settings, read_branch(tree, "gravity"));

    return status;
}

template <typename ParamC> void kadath_config<ParamC>::write_config(std::string ofile)
{
    ConfigTree new_tree;
    ConfigTree branch = container.return_branch();

    std::string section{"initial"};
    if (tree.find(section) != tree.not_found()) {
        new_tree.push_back(std::make_pair(section, tree.get_child(section)));
    }

    section = container.get_type();
    for (auto& key : branch) {
        if (key.first == section) {
            new_tree.push_back(std::make_pair(section, key.second));
        } else {
            std::string nested_key = section + "." + key.first;
            new_tree.put(nested_key, key.second.data());
        }
    }

    // Emit the single shared EOS as a top-level [eos] block. Matter-free
    // systems (e.g. BBH) return an empty branch and carry no [eos] section.
    ConfigTree eos_branch = container.return_eos_branch();
    if (!eos_branch.empty())
        new_tree.push_back(std::make_pair("eos", eos_branch));

    new_tree.push_back(std::make_pair("fields", build_branch<ConfigTree>(MBCO_FIELDS(), fields)));

    auto stage_map = append_map(MSTAGE(), container.get_stage_map(), stages);
    new_tree.push_back(std::make_pair("stages", build_branch<ConfigTree>(stage_map, stages, true)));

    new_tree.push_back(std::make_pair("sequence_controls", build_branch<ConfigTree>(MCONTROLS(), controls, true)));
    new_tree.push_back(
        std::make_pair("adaptive_mesh_refinement", build_branch<ConfigTree>(MAMR_SETTINGS(), amr_settings, true)));
    new_tree.push_back(std::make_pair("sequence_settings", build_branch<ConfigTree>(MSEQ_SETTINGS(), seq_settings)));

    if (has_gravity_setting(GRAV_THEORY))
        new_tree.push_back(std::make_pair("gravity", build_branch<ConfigTree>(MGRAVITY_PARAMS(), gravity_settings)));

    if (ofile != "null") {
        set_filename(ofile);
    }

    write_toml_config_tree(outputdir + filename, new_tree);
}

template <typename ParamC> void kadath_config<ParamC>::set_outputdir(std::string dir)
{
    if (dir.back() != '/')
        dir.push_back('/');
    this->outputdir = std::move(dir);
}

template <typename ParamC> void kadath_config<ParamC>::set_filename(std::string fname)
{
    if (fname.find('/') == std::string::npos)
        filename = fname;
    else {
        filename = Kadath::extract_filename(fname);
        outputdir = Kadath::extract_path(fname);
    }

    if (ends_with(filename, ".toml")) {
        return;
    }

    if (ends_with(filename, ".info") || ends_with(filename, ".dat")) {
        KADATH_THROW("Configurator accepts only .toml files: " + filename);
    }

    filename += ".toml";
}

template <typename T> std::ostream& operator<<(std::ostream& out, const kadath_config<T>& config)
{
    out << config.container;
    if (config.has_gravity_setting(GRAV_THEORY)) {
        print_params(MGRAVITY_PARAMS(), config.gravity_settings, out);
    }
    print_params(MAMR_SETTINGS(), config.amr_settings, out);
    return out;
}

template <typename ParamC> const std::string kadath_config<ParamC>::space_filename() const
{
    int idx = filename.rfind(".");
    return std::string{outputdir + filename.substr(0, idx) + ".dat"};
}

template <typename ParamC> const std::string kadath_config<ParamC>::config_filename_abs() const
{
    return std::string{outputdir + filename};
}

template <typename ParamC> inline void kadath_config<ParamC>::set_seq_defaults()
{
    if (!has_seq_setting(PREC))
        seq_settings[to_int(PREC)] = 1e-8;
    if (!has_seq_setting(MAX_ITER))
        seq_settings[to_int(MAX_ITER)] = 50;
}

template <typename ParamC> inline void kadath_config<ParamC>::set_amr_defaults()
{
    amr_settings[to_int(AMR_ENABLED)] = false;
    amr_settings[to_int(AMR_MAX_CYCLES)] = 2;
    amr_settings[to_int(AMR_TAIL_WIDTH)] = 2;
    amr_settings[to_int(AMR_L2_TAIL_THRESHOLD)] = 1e-8;
    amr_settings[to_int(AMR_LINF_TAIL_THRESHOLD)] = 1e-7;
    amr_settings[to_int(AMR_MODE)] = std::string{"hp"};
    // Absolute floor on a domain's total tail norm: ratios report zero when the
    // total norm sits at or below the floor, so near-zero fields cannot trigger
    // refinement on roundoff noise. 0.0 = no floor.
    amr_settings[to_int(AMR_NORM_FLOOR)] = 0.0;
    // mode="hp" refines h-first: a flagged region below max_shells takes an
    // h-move (one more shell); once every flagged region is shell-capped the
    // cycle falls back to per-domain p-refinement, with max_nr as the radial
    // ceiling on the p phase.
    amr_settings[to_int(AMR_MAX_NR)] = 15;
    amr_settings[to_int(AMR_MAX_SHELLS)] = 3;
    // The five bispheric domains share one Dim_array and refine as a locked block
    // in PHI (azimuthal, axis 2) only: all five take the same +2 together. Phi is
    // the same physical coordinate in every bipolar variant, so the bump keeps the
    // conforming internal seams square while the star-side / outer (import) seams
    // absorb the jump to the spherical neighbours. The bipolar chi/eta axes are not
    // refinable (they map to different physical directions across the variants and
    // unbalance the internal seams). Set false to pin the block at the base.
    amr_settings[to_int(AMR_REFINE_BISPHERIC)] = true;
    // h is a measured layout trial: after an h solve the moved-region radial
    // demand must fall by at least this factor before another h-move is allowed.
    amr_settings[to_int(AMR_H_GAIN_MIN)] = 2.0;
    // Legacy/inert compatibility key retained so older TOML files still parse.
    amr_settings[to_int(AMR_H_STALL_CYCLES)] = 1;
    // Legacy/inert compatibility key retained so older TOML files still parse.
    amr_settings[to_int(AMR_H_SATURATION_LINF)] = 0.95;
    // If angular demand is at least this multiple of radial demand, skip h for
    // the current cycle and spend points in the marked p axes instead.
    amr_settings[to_int(AMR_H_ANGULAR_DOMINANCE)] = 1.0;
    // Projected p-growth trust limit. 0 disables the guard. The post-h fallback
    // has stricter staging knobs below because a large p jump immediately after
    // layout changes can leave the nonlinear continuation basin.
    amr_settings[to_int(AMR_P_DOF_GROWTH_LIMIT)] = 2.0;
    amr_settings[to_int(AMR_P_FALLBACK_DOF_GROWTH_LIMIT)] = 1.1;
    amr_settings[to_int(AMR_P_FALLBACK_MAX_CANDIDATES)] = 1;
}
