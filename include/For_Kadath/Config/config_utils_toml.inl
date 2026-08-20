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

#include "config_utils_toml.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include <cmath>
#include <limits>

/** \addtogroup Configurator_Utils
 * @{ */

/// True iff `v` is finite, has zero fractional part, and lies within the
/// signed-int range. Used by build_branch to round-trip integer-semantic
/// double storage (legacy double arrays + ConfigSlot::as_double() values
/// that came in as `9.0` literals) without printing `9.0000000000000000`
/// to TOML. Note: 0.0/-0.0 both pass and round to 0; non-finite, fractional,
/// or out-of-range values fall through to double emission.
inline bool config_value_emits_as_int(double v) noexcept
{
    if (!std::isfinite(v)) {
        return false;
    }
    if (v < static_cast<double>(std::numeric_limits<int>::min()) ||
        v > static_cast<double>(std::numeric_limits<int>::max())) {
        return false;
    }
    return v == std::floor(v);
}

template <typename map_t, typename T> bool check_for_nan(const map_t& map, const T& var, const int idx)
{
    if constexpr (!std::is_floating_point_v<std::decay_t<T>>) {
        return false;
    } else if (!std::isnan(var)) {
        return false;
    } else {
        auto var_name =
            std::find_if(map.begin(), map.end(), [idx](const auto& pair) { return to_int(pair.second) == idx; });

        if (var_name == map.end()) {
            std::cerr << "No var found matching index: " << idx << "\n";
        } else {
            std::ostringstream message;
            message << "Var \"" << var_name->first << "\" is undefined.";
            KADATH_THROW(message.str());
        }
    }
    return true;
}

template <typename tree_t> tree_t read_branch(const tree_t& tree, std::string node)
{
    if (tree.find(node) != tree.not_found()) {
        return tree.get_child(node);
    }

    std::cerr << node << " not found " << std::endl;
    return tree_t{};
}

template <typename ary_t, typename map_t, typename tree_t>
void read_keys(const map_t& storage_map, ary_t& storage, const tree_t& tree) noexcept
{
    for (const auto& ele : storage_map) {
        if (tree.find(ele.first) == tree.not_found()) {
            continue;
        }

        const auto key = tree.get_child(ele.first).data();
        const auto lowered_key = config_tree_detail::lowercase(key);
        int idx = to_int(ele.second);
        using var_t = decltype(storage[idx]);

        if (lowered_key == "on" || lowered_key == "off" || lowered_key == "true" || lowered_key == "false") {
            if (std::is_assignable<var_t, bool>::value) {
                storage[idx] = (lowered_key == "on" || lowered_key == "true");
            } else {
                std::cerr << "Boolean parameter " << ele.first
                          << " encountered, but cannot assign to storage array\n";
            }
            continue;
        }

        try {
            double td = std::stod(key);
            int ti = std::stoi(key);

            if ((td - ti) == 0 && std::is_assignable<var_t, int>::value)
                storage[idx] = ti;
            else if (std::is_assignable<var_t, double>::value)
                storage[idx] = td;
            else
                std::cerr << "int/double parameter " << ele.first
                          << " encountered, but cannot assign to storage array \n";
            continue;
        } catch (...) {
        }

        if constexpr (std::is_assignable<var_t, std::string>::value) {
            storage[idx] = key;
            continue;
        } else {
            std::cerr << "String parameter " << ele.first << " encountered, but cannot assign to storage array \n";
        }
    }
}

template <typename map_t, typename ary_t>
void print_params(const map_t& storage_map, const ary_t& storage, std::ostream& out)
{
    using stored_t = std::remove_cv_t<std::remove_reference_t<decltype(*storage.begin())>>;
    for (const auto& a : storage_map) {
        auto print = [&](auto&& arg) {
            using arg_t = std::remove_cv_t<std::remove_reference_t<decltype(arg)>>;
            if constexpr (std::is_same_v<arg_t, double>) {
                if (std::isnan(arg)) {
                    return;
                }
            } else if constexpr (std::is_same_v<arg_t, bool>) {
                if constexpr (std::is_same_v<stored_t, bool>) {
                    if (storage[to_int(a.second)]) {
                        out << std::setw(20) << a.first << ": on" << std::endl;
                    }
                } else {
                    out << std::setw(20) << a.first << ": " << (arg ? "true" : "false") << std::endl;
                }
                return;
            }
            out << std::setw(20) << a.first << ": " << arg << std::endl;
        };

        const auto& ele = storage.at(to_int(a.second));
        if constexpr (std::is_same_v<stored_t, Kadath::ConfigSlot>) {
            if (!ele.has_value())
                continue;
            if (ele.holds_bool())
                out << std::setw(20) << a.first << ": " << (ele.as_bool() ? "true" : "false") << std::endl;
            else if (ele.holds_int())
                out << std::setw(20) << a.first << ": " << ele.as_int() << std::endl;
            else
                out << std::setw(20) << a.first << ": " << ele.as_double() << std::endl;
        } else if constexpr (!std::is_fundamental_v<stored_t>) {
            std::visit(print, ele);
        } else {
            print(ele);
        }
    }
}

template <typename tree_t, typename map_t, typename ary_t>
tree_t build_branch(const map_t& storage_map, const ary_t& storage, const bool inc_off)
{
    using stored_t = std::remove_cv_t<std::remove_reference_t<decltype(*storage.begin())>>;
    tree_t branch;
    for (const auto& a : storage_map) {
        auto add_key = [&](auto&& arg) -> void {
            using arg_t = std::remove_cv_t<std::remove_reference_t<decltype(arg)>>;
            if constexpr (std::is_same_v<arg_t, double>) {
                if (std::isnan(arg))
                    return;
                if (config_value_emits_as_int(arg))
                    branch.put(a.first, static_cast<int>(arg));
                else
                    branch.put(a.first, arg);
                return;
            } else if constexpr (std::is_same_v<arg_t, bool>) {
                if (arg)
                    branch.put(a.first, true);
                else if (!arg && inc_off)
                    branch.put(a.first, false);
                return;
            } else {
                branch.put(a.first, arg);
            }
        };

        if constexpr (std::is_same_v<stored_t, Kadath::ConfigSlot>) {
            const auto& slot = storage[to_int(a.second)];
            if (!slot.has_value())
                continue;
            if (slot.holds_bool()) {
                branch.put(a.first, slot.as_bool());
            } else if (slot.holds_int()) {
                branch.put(a.first, slot.as_int());
            } else {
                const double v = slot.as_double();
                if (config_value_emits_as_int(v))
                    branch.put(a.first, static_cast<int>(v));
                else
                    branch.put(a.first, v);
            }
        } else if constexpr (!std::is_fundamental_v<stored_t>)
            std::visit(add_key, storage[to_int(a.second)]);
        else
            add_key(storage[to_int(a.second)]);
    }
    return branch;
}

template <typename ary_t, typename map_t> std::tuple<std::string, int> get_last_enabled(map_t enum_map, ary_t toggle)
{
    auto last_enabled = std::distance(std::find(toggle.rbegin(), toggle.rend(), true), toggle.rend() - 1);
    auto name = std::find_if(enum_map.begin(), enum_map.end(),
                             [last_enabled](const auto& map) { return to_int(map.second) == last_enabled; });
    if (name != enum_map.end()) {
        return std::make_tuple(name->first, last_enabled);
    }
    throw std::invalid_argument("\nNo valid stages enabled\n)");
}

template <typename map_t, typename boolary_t>
map_t append_map(const map_t& full_map, const map_t& partial_map, const boolary_t& storage)
{
    map_t map{partial_map};

    for (const auto& [str, i] : full_map) {
        if (map.find(str) == map.end() && storage[to_int(i)])
            map.insert({str, i});
    }
    return map;
}

/**
 * @} */
