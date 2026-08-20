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
#include "config_utils_toml.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>

template <typename ParamC> class kadath_config;
template <typename ParamC> std::ostream& operator<<(std::ostream& out, const kadath_config<ParamC>& config);

template <typename ParamC> class kadath_config
{
    using FArray = std::array<bool, NUM_BCO_FIELDS_V>;
    using SArray = std::array<bool, NUM_STAGES_V>;
    using CArray = std::array<bool, NUM_CONTROLS_V>;

  private:
    std::string filename{};
    std::string outputdir{"./"};
    ConfigTree tree;
    ConfigTree branch_in;
    FArray fields{};
    SArray stages{};
    CArray controls{};
  public:
    using seq_var_t = std::variant<double, int>;
    using gravity_var_t = std::variant<double, int, std::string>;
    using amr_var_t = std::variant<bool, int, double, std::string>;
  private:
    std::array<seq_var_t, NUM_SEQ_SETTINGS_V> seq_settings{};
    std::array<gravity_var_t, NUM_GRAVITY_PARAMS_V> gravity_settings{};
    std::array<amr_var_t, NUM_AMR_SETTINGS_V> amr_settings{};
    ParamC container; ///< Parameter container

  public:
    kadath_config();
    explicit kadath_config(std::string ifile);

    int open_config();
    void write_config(std::string ofile = "null");

    const std::string config_filename() const { return filename; }
    const std::string config_filename_abs() const;
    const std::string space_filename() const;
    const std::string config_outputdir() const { return outputdir; }

    FArray const& return_fields() const { return fields; }
    FArray& return_fields() { return fields; }

    SArray const& return_stages() const { return stages; }
    SArray& return_stages() { return stages; }

    template <typename E> auto& set_field(E idx) { return fields[static_cast<int>(idx)]; }
    template <typename E> auto const& field(E idx) const { return fields[static_cast<int>(idx)]; }

    template <typename E> auto& set_stage(E idx) { return stages[static_cast<int>(idx)]; }
    template <typename E> auto const& set_stage(E idx) const { return stages[static_cast<int>(idx)]; }

    template <typename E> auto const& control(E idx) const { return controls[static_cast<int>(idx)]; }
    template <typename E> auto& control(E idx) { return controls[static_cast<int>(idx)]; }

    template <typename E> auto& seq_setting(E idx) { return seq_settings[static_cast<int>(idx)]; }
    template <typename E> auto const& seq_setting(E idx) const { return seq_settings[static_cast<int>(idx)]; }

    template <typename E> auto& set_gravity(E idx) { return gravity_settings[static_cast<int>(idx)]; }
    template <typename E> auto const& gravity_setting(E idx) const { return gravity_settings[static_cast<int>(idx)]; }

    template <typename E> auto& amr_setting(E idx) { return amr_settings[static_cast<int>(idx)]; }
    template <typename E> auto const& amr_setting(E idx) const { return amr_settings[static_cast<int>(idx)]; }

    template <typename T, typename E> T amr_setting_as(E idx) const
    {
        T result{};
        std::visit(
            [&result](auto&& v) {
                using vt = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::string> && std::is_same_v<vt, std::string>) {
                    result = v;
                } else if constexpr (std::is_arithmetic_v<T> && std::is_arithmetic_v<vt>) {
                    result = static_cast<T>(v);
                }
            },
            amr_settings[static_cast<int>(idx)]);
        return result;
    }

    template <typename T, typename E> T gravity(E idx) const
    {
        T var{};
        auto set_var = [&](auto&& v) mutable {
            using v_t = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string> && std::is_same_v<v_t, std::string>) {
                var = v;
            } else if constexpr (std::is_arithmetic_v<T> && std::is_arithmetic_v<v_t>) {
                if constexpr (std::is_floating_point_v<v_t>) {
                    if (!std::isnan(v))
                        var = static_cast<T>(v);
                } else {
                    var = static_cast<T>(v);
                }
            }
        };
        std::visit(set_var, gravity_settings[static_cast<int>(idx)]);
        return var;
    }

    bool has_gravity_setting(GRAVITY_PARAMS idx) const
    {
        return std::visit(
            [](auto&& v) -> bool {
                using vt = std::decay_t<decltype(v)>;
                if constexpr (std::is_floating_point_v<vt>) {
                    return !std::isnan(v);
                } else if constexpr (std::is_same_v<vt, std::string>) {
                    return !v.empty();
                } else {
                    return true;
                }
            },
            gravity_settings[static_cast<int>(idx)]);
    }


    /**
     * Returns true when the slot was explicitly set: int slots are always
     * considered set; double slots are set unless their value is NaN.
     */
    template <typename E> bool has_seq_setting(E idx) const
    {
        return std::visit(
            [](auto&& v) -> bool {
                using vt = std::decay_t<decltype(v)>;
                if constexpr (std::is_floating_point_v<vt>) {
                    return !std::isnan(v);
                } else {
                    return true;
                }
            },
            seq_settings[static_cast<int>(idx)]);
    }

    /**
     * Typed read accessor. Visits the variant and returns the value cast to T.
     * If the slot holds an unset (NaN) double, returns T{}.
     */
    template <typename T, typename E> T seq_setting_as(E idx) const
    {
        static_assert(std::is_arithmetic_v<T>, "seq_setting_as<T>: T must be arithmetic");
        T result{};
        std::visit(
            [&result](auto&& v) {
                using vt = std::decay_t<decltype(v)>;
                if constexpr (std::is_floating_point_v<vt>) {
                    if (!std::isnan(v))
                        result = static_cast<T>(v);
                } else {
                    result = static_cast<T>(v);
                }
            },
            seq_settings[static_cast<int>(idx)]);
        return result;
    }

    void set_outputdir(std::string dir);
    void set_filename(std::string fname);

    void initialize_binary(std::array<std::string, 2> bco_types) { container.init_binary(bco_types); }
    void initialize_three_body(std::array<std::string, 3> bco_types) { container.init_three_body(bco_types); }
    void set_defaults() { container.set_defaults(*this); }

    constexpr auto& get_map() const { return container.get_map(); }

    template <typename E> constexpr auto& get_map(E idx) const { return container.get_map(static_cast<int>(idx)); }

    template <typename E> constexpr auto& operator()(E idx)
    {
        if (!check_for_nan(container.get_map(), container(idx), static_cast<int>(idx))) {
            return container(idx);
        }
        throw std::invalid_argument("\nInvalid Configuration Indicies\n)");
    }

    /**
     * Typed read accessor for container-stored params. Handles plain
     * `double` storage, `Kadath::ConfigSlot` storage, and `std::variant`
     * storage uniformly: NaN double slots return T{}; int slots return
     * T(value).
     */
    template <typename T, typename E> T value_as(E idx)
    {
        static_assert(std::is_arithmetic_v<T>, "value_as<T>: T must be arithmetic");
        T result{};
        auto& slot = container(idx);
        using slot_t = std::decay_t<decltype(slot)>;
        if constexpr (std::is_floating_point_v<slot_t>) {
            if (!std::isnan(slot))
                result = static_cast<T>(slot);
        } else if constexpr (std::is_integral_v<slot_t>) {
            result = static_cast<T>(slot);
        } else if constexpr (std::is_same_v<slot_t, Kadath::ConfigSlot>) {
            if (slot.has_value()) {
                result = slot.holds_int()
                             ? static_cast<T>(slot.as_int())
                             : static_cast<T>(slot.as_double());
            }
        } else {
            std::visit(
                [&result](auto&& v) {
                    using vt = std::decay_t<decltype(v)>;
                    if constexpr (std::is_floating_point_v<vt>) {
                        if (!std::isnan(v))
                            result = static_cast<T>(v);
                    } else if constexpr (std::is_arithmetic_v<vt>) {
                        result = static_cast<T>(v);
                    }
                },
                slot);
        }
        return result;
    }

    /**
     * Returns true if the container slot has been explicitly set.
     * Integer slots are always considered set; double slots only when not NaN.
     * Variadic form forwards extra args (e.g. BCO index) to container().
     */
    template <typename... idx_t> bool has_value(idx_t... idxs)
    {
        auto& slot = container(idxs...);
        using slot_t = std::decay_t<decltype(slot)>;
        if constexpr (std::is_floating_point_v<slot_t>) {
            return !std::isnan(slot);
        } else if constexpr (std::is_integral_v<slot_t>) {
            return true;
        } else if constexpr (std::is_same_v<slot_t, Kadath::ConfigSlot>) {
            return slot.has_value();
        } else {
            return std::visit(
                [](auto&& v) -> bool {
                    using vt = std::decay_t<decltype(v)>;
                    if constexpr (std::is_floating_point_v<vt>) {
                        return !std::isnan(v);
                    } else {
                        return true;
                    }
                },
                slot);
        }
    }

    template <typename E, typename P> constexpr auto& operator()(E idx, P Pidx)
    {
        if (!check_for_nan(container.get_map(static_cast<int>(Pidx)), container(idx, Pidx), static_cast<int>(idx))) {
            return container(idx, Pidx);
        }
        throw std::invalid_argument("\nInvalid Configuration Indicies\n)");
    }

    template <typename... idx_t> constexpr auto& set(const idx_t... idxs) { return container(idxs...); }

    template <typename... idx_t> constexpr auto& set_eos(const idx_t... idxs)
    {
        return container.set_eos_param(idxs...);
    }

    template <typename... idx_t> constexpr auto& set_diffrot(const idx_t... idxs)
    {
        return container.set_diffrot_param(idxs...);
    }

    template <typename T, typename... idx_t, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
    constexpr const T eos(idx_t... idx)
    {
        T var{};
        auto set_var = [&](auto&& v) mutable {
            using v_t = std::decay_t<decltype(v)>;
            if constexpr (std::is_arithmetic_v<v_t>) {
                if (!std::isnan(v))
                    var = v;
            }
        };
        std::visit(set_var, this->set_eos(idx...));
        return var;
    }

    template <typename T, typename... idx_t, std::enable_if_t<std::is_same_v<T, std::string>, bool> = true>
    constexpr const T eos(idx_t... idx)
    {
        T var{};
        auto set_var = [&](auto&& v) mutable {
            using v_t = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<v_t, std::string>) {
                var = v;
            }
        };
        std::visit(set_var, this->set_eos(idx...));
        return var;
    }

    template <typename T, typename... idx_t, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
    constexpr const T diffrot(idx_t... idx)
    {
        T var{};
        auto set_var = [&](auto&& v) mutable {
            using v_t = std::decay_t<decltype(v)>;
            if constexpr (std::is_arithmetic_v<v_t>) {
                if (!std::isnan(v))
                    var = v;
            }
        };
        std::visit(set_var, this->set_diffrot(idx...));
        return var;
    }

    template <typename T, typename... idx_t, std::enable_if_t<std::is_same_v<T, std::string>, bool> = true>
    constexpr const T diffrot(idx_t... idx)
    {
        T var{};
        auto set_var = [&](auto&& v) mutable {
            using v_t = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<v_t, std::string>) {
                var = v;
            }
        };
        std::visit(set_var, this->set_diffrot(idx...));
        return var;
    }

    inline void set_seq_defaults();
    inline void set_amr_defaults();

    friend std::ostream& operator<< <>(std::ostream&, const kadath_config<ParamC>&);
};

#include "configurator_toml.inl"
