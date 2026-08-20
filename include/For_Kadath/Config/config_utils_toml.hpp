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

#include "config_enums.hpp"
#include "config_tree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <variant>

/** @defgroup Configurator_Utils
 * @ingroup Configurator
 * Utility functions to parse, store, and write TOML-backed Configurator data.
 * @{
 */

namespace Kadath {

/**
 * Type-aware scalar config slot. Stores a double plus a flag noting whether
 * the value was authored as an integer literal. Read-side conversion is
 * implicit (operator double/int) so existing arithmetic call sites keep
 * working. Toml output dispatches on @ref holds_int() so logically-integer
 * fields emit as "9" instead of "9.0000000000000000".
 */
class ConfigSlot
{
  public:
    ConfigSlot() = default;
    ConfigSlot(double v) : value_{v}, is_int_{false} {}
    ConfigSlot(int v) : value_{static_cast<double>(v)}, is_int_{true}, is_bool_{false} {}

    ConfigSlot& operator=(double v)
    {
        value_ = v;
        is_int_ = false;
        is_bool_ = false;
        return *this;
    }
    ConfigSlot& operator=(int v)
    {
        value_ = static_cast<double>(v);
        is_int_ = true;
        is_bool_ = false;
        return *this;
    }
    ConfigSlot& operator=(bool v)
    {
        value_ = v ? 1.0 : 0.0;
        is_int_ = true;
        is_bool_ = true;
        return *this;
    }

    // Conversions: reference forms make the slot usable as a `double&`
    // (e.g. for add_var(name, var) sinks and rebindings). Plain value
    // conversion to int is explicit to avoid binary-operator ambiguity in
    // arithmetic with literals; callers reading into `int` rely on the
    // standard double-to-int narrowing for initialization contexts.
    operator double&() { return value_; }
    operator const double&() const { return value_; }
    explicit operator int() const { return static_cast<int>(value_); }

    /// True iff the slot has been authored (i.e. value is not NaN).
    bool has_value() const { return !std::isnan(value_); }

    /**
     * True iff the slot still represents an integer value.
     *
     * Newton solvers bind some ConfigSlot values as `double&` unknowns.
     * Mutating through that reference updates `value_` but cannot clear the
     * original integer-literal flag. Recheck the current value so an input
     * such as `omega = 0` does not serialize a solved fractional omega as 0.
     */
    bool holds_int() const
    {
        return !is_bool_ && is_int_ && std::isfinite(value_) &&
               value_ >= static_cast<double>(std::numeric_limits<int>::min()) &&
               value_ <= static_cast<double>(std::numeric_limits<int>::max()) &&
               value_ == std::trunc(value_);
    }

    bool holds_bool() const
    {
        return is_bool_ && std::isfinite(value_) &&
               (value_ == 0.0 || value_ == 1.0);
    }

    double as_double() const { return value_; }
    int as_int() const { return static_cast<int>(value_); }
    bool as_bool() const { return value_ != 0.0; }

  private:
    double value_{std::nan("1")};
    bool is_int_{false};
    bool is_bool_{false};
};

inline bool isnan(const ConfigSlot& slot)
{
    return !slot.has_value();
}

} // namespace Kadath

// Bring isnan into the global namespace so existing `std::isnan(bconfig(KEY))`
// call sites keep compiling; std::isnan resolves via ADL on ConfigSlot.
namespace std {
inline bool isnan(const Kadath::ConfigSlot& slot)
{
    return !slot.has_value();
}
} // namespace std

template <typename map_t, typename T> bool check_for_nan(const map_t& map, const T& var, const int idx);

template <typename tree_t> tree_t read_branch(const tree_t& tree, std::string node);

template <typename ary_t, typename map_t, typename tree_t>
void read_keys(const map_t& storage_map, ary_t& storage, const tree_t& tree) noexcept;

template <typename map_t, typename ary_t>
void print_params(const map_t& storage_map, const ary_t& storage, std::ostream& out = std::cout);

template <typename tree_t, typename map_t, typename ary_t>
tree_t build_branch(const map_t& storage_map, const ary_t& storage, const bool inc_off = false);

template <typename ary_t, typename map_t> std::tuple<std::string, int> get_last_enabled(map_t enum_map, ary_t toggle);

template <typename map_t, typename boolary_t>
map_t append_map(const map_t& full_map, const map_t& partial_map, const boolary_t& storage);

/** @} end config_utils group */

#include "config_utils_toml.inl"
