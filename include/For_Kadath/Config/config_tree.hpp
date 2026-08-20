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

#include "For_Kadath/Array/exceptions.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#endif
#include "For_Kadath/Third_Party/Tomlplusplus/include/toml.hpp"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

class ConfigTree
{
  public:
    using Child = std::pair<std::string, ConfigTree>;
    using Children = std::vector<Child>;
    using iterator = Children::iterator;
    using const_iterator = Children::const_iterator;

    ConfigTree() = default;
    explicit ConfigTree(std::string value) : value_(std::move(value)) {}

    const std::string& data() const { return value_; }
    std::string& data() { return value_; }
    bool empty() const { return children_.empty(); }

    iterator begin() { return children_.begin(); }
    iterator end() { return children_.end(); }
    const_iterator begin() const { return children_.begin(); }
    const_iterator end() const { return children_.end(); }
    const_iterator cbegin() const { return children_.cbegin(); }
    const_iterator cend() const { return children_.cend(); }

    iterator not_found() { return children_.end(); }
    const_iterator not_found() const { return children_.end(); }

    iterator find(const std::string& key)
    {
        return std::find_if(children_.begin(), children_.end(),
                            [&key](const Child& child) { return child.first == key; });
    }

    const_iterator find(const std::string& key) const
    {
        return std::find_if(children_.begin(), children_.end(),
                            [&key](const Child& child) { return child.first == key; });
    }

    ConfigTree& get_child(const std::string& path)
    {
        if (auto* child = find_path(path)) {
            return *child;
        }
        KADATH_THROW("Missing configuration branch: " + path);
    }

    const ConfigTree& get_child(const std::string& path) const
    {
        if (const auto* child = find_path(path)) {
            return *child;
        }
        KADATH_THROW("Missing configuration branch: " + path);
    }

    template <typename Value> void put(const std::string& path, const Value& value)
    {
        auto& child = ensure_path(path);
        child.value_ = config_value_to_string(value);
    }

    void put_child(const std::string& path, const ConfigTree& child)
    {
        ensure_path(path) = child;
    }

    void add_child(const std::string& path, const ConfigTree& child)
    {
        const auto dot = path.rfind('.');
        if (dot == std::string::npos) {
            children_.push_back({path, child});
            return;
        }

        auto& parent = ensure_path(path.substr(0, dot));
        parent.children_.push_back({path.substr(dot + 1), child});
    }

    void push_back(const Child& child) { children_.push_back(child); }
    void push_back(Child&& child) { children_.push_back(std::move(child)); }

  private:
    std::string value_;
    Children children_;

    ConfigTree* find_direct_child(const std::string& key)
    {
        const auto child = find(key);
        return child == children_.end() ? nullptr : &child->second;
    }

    const ConfigTree* find_direct_child(const std::string& key) const
    {
        const auto child = find(key);
        return child == children_.end() ? nullptr : &child->second;
    }

    ConfigTree* find_path(const std::string& path)
    {
        const auto dot = path.find('.');
        if (dot == std::string::npos) {
            return find_direct_child(path);
        }

        auto* child = find_direct_child(path.substr(0, dot));
        return child ? child->find_path(path.substr(dot + 1)) : nullptr;
    }

    const ConfigTree* find_path(const std::string& path) const
    {
        const auto dot = path.find('.');
        if (dot == std::string::npos) {
            return find_direct_child(path);
        }

        const auto* child = find_direct_child(path.substr(0, dot));
        return child ? child->find_path(path.substr(dot + 1)) : nullptr;
    }

    ConfigTree& ensure_path(const std::string& path)
    {
        const auto dot = path.find('.');
        if (dot == std::string::npos) {
            if (auto* child = find_direct_child(path)) {
                return *child;
            }
            children_.push_back({path, ConfigTree{}});
            return children_.back().second;
        }

        auto& child = ensure_path(path.substr(0, dot));
        return child.ensure_path(path.substr(dot + 1));
    }

    template <typename Value> static std::string config_value_to_string(const Value& value)
    {
        if constexpr (std::is_same_v<std::decay_t<Value>, bool>) {
            return value ? "true" : "false";
        } else if constexpr (std::is_floating_point_v<std::decay_t<Value>>) {
            std::ostringstream output;
            output << std::setprecision(std::numeric_limits<std::decay_t<Value>>::max_digits10) << std::showpoint
                   << value;
            return output.str();
        } else if constexpr (std::is_integral_v<std::decay_t<Value>>) {
            return std::to_string(value);
        } else {
            std::ostringstream output;
            output << value;
            return output.str();
        }
    }
};

namespace config_tree_detail
{

inline std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

inline bool is_toml_boolean_literal(const std::string& value)
{
    const auto lowered = lowercase(value);
    return lowered == "on" || lowered == "off" || lowered == "true" || lowered == "false";
}

inline bool is_toml_number_literal(const std::string& value)
{
    if (value.empty()) {
        return false;
    }
    char* end = nullptr;
    (void)std::strtod(value.c_str(), &end);
    return end != value.c_str() && *end == '\0';
}

inline std::string escape_toml_string(const std::string& value)
{
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

inline void write_toml_value(std::ostream& output, const std::string& value)
{
    const auto lowered = lowercase(value);
    if (lowered == "on" || lowered == "true") {
        output << "true";
    } else if (lowered == "off" || lowered == "false") {
        output << "false";
    } else if (is_toml_number_literal(value)) {
        output << value;
    } else {
        output << '"' << escape_toml_string(value) << '"';
    }
}

inline std::string toml_scalar_to_string(const toml::node& node)
{
    if (const auto* value = node.as_boolean()) {
        return value->get() ? "true" : "false";
    }
    if (const auto* value = node.as_integer()) {
        return std::to_string(value->get());
    }
    if (const auto* value = node.as_floating_point()) {
        std::ostringstream output;
        output << std::setprecision(std::numeric_limits<double>::max_digits10) << std::showpoint << value->get();
        return output.str();
    }
    if (const auto* value = node.as_string()) {
        return value->get();
    }
    KADATH_THROW("Unsupported TOML value in Configurator file.");
}

inline ConfigTree toml_table_to_config_tree(const toml::table& table)
{
    ConfigTree tree;
    for (const auto& [key, value] : table) {
        const std::string name{key.str()};
        if (const auto* nested = value.as_table()) {
            tree.push_back({name, toml_table_to_config_tree(*nested)});
        } else {
            tree.push_back({name, ConfigTree{toml_scalar_to_string(value)}});
        }
    }
    return tree;
}

// Groups preserve ConfigTree insertion order (write_config controls the
// section sequence of emitted TOML); a std::map here would resort them
// alphabetically.
using TomlGroups = std::vector<std::pair<std::string, std::vector<std::pair<std::string, std::string>>>>;

inline std::vector<std::pair<std::string, std::string>>& toml_group_entries(TomlGroups& groups,
                                                                            const std::string& name)
{
    for (auto& [group, entries] : groups) {
        if (group == name)
            return entries;
    }
    groups.push_back({name, {}});
    return groups.back().second;
}

inline void collect_toml_groups(const ConfigTree& tree, const std::string& prefix, TomlGroups& groups)
{
    for (const auto& [name, child] : tree) {
        if (child.empty()) {
            toml_group_entries(groups, prefix).push_back({name, child.data()});
        } else {
            const auto child_prefix = prefix.empty() ? name : prefix + "." + name;
            collect_toml_groups(child, child_prefix, groups);
        }
    }
}

} // namespace config_tree_detail

inline ConfigTree read_toml_config_tree(const std::string& path)
{
    try {
        return config_tree_detail::toml_table_to_config_tree(toml::parse_file(path));
    } catch (const toml::parse_error& error) {
        std::ostringstream message;
        message << "TOML parse error in " << path << ": " << error.description();
        KADATH_THROW(message.str());
    }
}

inline void write_toml_config_tree(const std::string& path, const ConfigTree& tree)
{
    std::ofstream output{path};
    if (!output) {
        KADATH_THROW("Unable to open TOML config for writing: " + path);
    }

    config_tree_detail::TomlGroups groups;
    config_tree_detail::collect_toml_groups(tree, "", groups);

    bool first_group = true;
    for (const auto& [group, entries] : groups) {
        if (!first_group) {
            output << '\n';
        }
        first_group = false;

        if (!group.empty()) {
            output << '[' << group << "]\n";
        }
        for (const auto& [key, value] : entries) {
            output << key << " = ";
            config_tree_detail::write_toml_value(output, value);
            output << '\n';
        }
    }

    if (!output) {
        KADATH_THROW("Unable to finish writing TOML config: " + path);
    }
}
