/*
    Copyright 2026 Hao-Jui Kuan

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

// Canonical runtime-environment-variable parsers shared by the Newton solver
// backends, the Jacobian assembler, and the assembly/value-domain runtime
// gates. These replace a family of per-translation-unit copies that had
// drifted in their case handling (some checked "FALSE"/"OFF" but not "no",
// others the reverse). All boolean parsing now lowercases the value first and
// then treats it as disabled iff it is empty or one of {0, false, off, no},
// which is a strict superset of every prior copy (so "FALSE", "Off", "No",
// "NO", etc. now uniformly disable).

#pragma once

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace Kadath
{
    namespace runtime_env_detail
    {
        // Lowercase a copy of the value for case-insensitive comparison.
        inline std::string to_lower_copy(const char* value)
        {
            std::string text(value);
            for (char& ch : text)
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            return text;
        }
    } // namespace runtime_env_detail

    // True unless the variable is unset, empty, or (case-insensitively) one of
    // "0", "false", "off", "no". Returns default_val when the variable is unset.
    inline bool env_flag_enabled(const char* name, bool default_val = false)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0')
            return default_val;
        const std::string text = runtime_env_detail::to_lower_copy(value);
        return text != "0" && text != "false" && text != "off" && text != "no";
    }

    // Default-on production controls with a deliberately narrow opt-out:
    // only the exact value "0" disables; unset, empty, and every other value
    // enable. Keep this separate from env_flag_enabled so existing flag
    // semantics do not change.
    inline bool env_flag_zero_opt_out(const char* name)
    {
        const char* value = std::getenv(name);
        return value == nullptr || std::string(value) != "0";
    }

    // Parse a finite double; returns default_val when unset/empty/malformed or
    // non-finite.
    inline double env_double_value(const char* name, double default_val)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0')
            return default_val;
        char* end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || !std::isfinite(parsed))
            return default_val;
        return parsed;
    }

    // Parse a base-10 integer; returns default_val when unset/empty/malformed.
    inline int env_int_value(const char* name, int default_val)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0')
            return default_val;
        errno = 0;
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' ||
            parsed < std::numeric_limits<int>::min() ||
            parsed > std::numeric_limits<int>::max())
            return default_val;
        return static_cast<int>(parsed);
    }

    // Parse a strictly-positive base-10 integer; returns default_val when
    // unset/empty/malformed or when the parsed value is <= 0.
    inline int env_positive_int(const char* name, int default_val)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0')
            return default_val;
        errno = 0;
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' || parsed <= 0 ||
            parsed > std::numeric_limits<int>::max())
            return default_val;
        return static_cast<int>(parsed);
    }

    // Return the raw variable value, or default_val when unset/empty.
    inline std::string env_string_value(const char* name, std::string default_val = "")
    {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0')
            return default_val;
        return std::string(value);
    }

    // Parse a comma-separated list of base-10 integers; returns default_val
    // when the variable is unset or empty. Tokens that fail to parse are
    // silently skipped.
    inline std::vector<int> env_int_list(const char* name, std::vector<int> default_val = {})
    {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0')
            return default_val;
        std::vector<int> values;
        std::stringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ',')) {
            char* end = nullptr;
            const long parsed = std::strtol(token.c_str(), &end, 10);
            if (end != token.c_str())
                values.push_back(static_cast<int>(parsed));
        }
        return values;
    }

} // namespace Kadath
