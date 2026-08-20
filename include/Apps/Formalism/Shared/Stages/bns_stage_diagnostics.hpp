#pragma once

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace Kadath {
namespace bns_diagnostics {

inline bool env_flag(const char* name, bool default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return default_value;
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text != "0" && text != "false" && text != "off" && text != "no";
}

inline double env_double(const char* name, double default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return default_value;
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || !std::isfinite(parsed))
        return default_value;
    return parsed;
}

inline int env_int(const char* name, int default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return default_value;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value)
        return default_value;
    return static_cast<int>(parsed);
}

inline std::vector<int> env_int_list(const char* name)
{
    std::vector<int> result;
    const char* cursor = std::getenv(name);
    if (cursor == nullptr || cursor[0] == '\0')
        return result;

    while (*cursor != '\0') {
        char* end = nullptr;
        const long parsed = std::strtol(cursor, &end, 10);
        if (end != cursor)
            result.push_back(static_cast<int>(parsed));
        cursor = end;
        while (*cursor == ',' || *cursor == ':' || std::isspace(static_cast<unsigned char>(*cursor)))
            ++cursor;
        if (end == cursor && *cursor != '\0')
            ++cursor;
    }
    return result;
}

inline void print_top_equation_errors(System_of_eqs& syst, int limit)
{
    if (limit <= 0)
        return;

    Array<double> residual(syst.sec_member());
    std::vector<System_of_eqs::RowMetadata> metadata;
    syst.classify_equation_row_metadata(metadata);

    std::vector<int> order(residual.get_size(0));
    for (int i = 0; i < residual.get_size(0); ++i)
        order[static_cast<std::size_t>(i)] = i;

    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return std::abs(residual(lhs)) > std::abs(residual(rhs));
    });

    std::cout << "Top residual rows:" << std::endl;
    for (int n = 0; n < limit && n < static_cast<int>(order.size()); ++n) {
        const int idx = order[static_cast<std::size_t>(n)];
        const auto& row = metadata[static_cast<std::size_t>(idx)];
        std::cout << "  [" << idx << "] " << residual(idx) << " abs=" << std::abs(residual(idx))
                  << " eq_index=" << row.eq_index << " type=" << row.equation_type << " taxonomy="
                  << static_cast<int>(row.taxonomy) << " dom=" << row.dom << " owner=" << row.owner_var_name
                  << " local_row=" << row.eq_local_row << " basis=" << row.basis_mode << std::endl;
    }
}

inline void print_selected_equation_rows(System_of_eqs& /*syst*/) {}

inline double array_inf_norm(const Array<double>& values)
{
    double max_abs = 0.0;
    for (int i = 0; i < values.get_size(0); ++i)
        max_abs = std::max(max_abs, std::abs(values(i)));
    return max_abs;
}

inline void probe_jacobian_columns(System_of_eqs& /*syst*/) {}

inline void compare_jx_columns(System_of_eqs& /*syst*/) {}

} // namespace bns_diagnostics
} // namespace Kadath
