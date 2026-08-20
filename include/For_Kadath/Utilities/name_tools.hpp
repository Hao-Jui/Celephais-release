/*
    Copyright 2017 Philippe Grandclement

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

#pragma once

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/array.hpp"
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace Kadath
{
    constexpr int LMAX = 1000;
    void trim_spaces(char* dest, const char* name);
    void get_util(char*, char*);
    int nbr_char(const char*, char);
    void get_term(char*, char*, char);
    void get_parts(const char*, char*, char*, char, int place = 0);
    bool is_tensor(const char*, const char*, int&, char*&, Array<int>*&);
    std::string extract_path(std::string fullvar);
    std::string extract_filename(std::string fullvar);

    // ---- Modern value-returning parser helpers (additive, no caller buffers).
    // All returned strings carry the legacy trim_spaces convention: internal
    // runs of spaces collapsed; single trailing space if non-empty.

    /// Two halves of a binary split, both trim-normalized.
    struct BinarySplit {
        std::string lhs;
        std::string rhs;
    };

    /// Parsed tensor expression. \c base_name retains the trailing space
    /// (matches \c trim_spaces output stored in System_of_eqs name vectors).
    /// \c indices is the raw index character sequence (length == \c valence).
    /// \c types entries are 0 (COV, written with `_`) or 1 (CON, with `^`).
    struct TensorParse {
        std::string base_name;
        int valence = 0;
        std::string indices;
        std::vector<int> types;
    };

    /// Top-level binary operator split, parity with System_of_eqs::is_ope_bin:
    /// scans right-to-left for occurrences of \c op and returns the first
    /// split where both halves have balanced parentheses. \c op must be one
    /// of `+`, `-`, `*`, `/`, `=`. Returns nullopt if no balanced split exists.
    std::optional<BinarySplit> split_binary_operator(std::string_view input, char op);

    /// Split on the first (leftmost) occurrence of \c sep, parity with
    /// \c get_term. Both halves are trim-normalized. If \c sep is not present,
    /// returns {trim(input), ""} (modern guard; legacy \c get_term was UB on
    /// missing separator).
    std::pair<std::string, std::string> split_first(std::string_view input, char sep);

    /// Split at the (occurrence+1)-th separator counted RIGHT-TO-LEFT, parity
    /// with \c get_parts(input, p1, p2, sep, occurrence). occurrence==0 is the
    /// last separator. Returns nullopt if fewer than (occurrence+1) separators
    /// exist or if the matched separator is at position 0 (matches the
    /// pos==-1 fallback in get_parts).
    std::optional<BinarySplit> split_nth(std::string_view input, char sep, int occurrence);

    /// Parse a tensor name like "g_ij " or "v^a ". Returns nullopt for
    /// malformed input (unless \c input contains exactly one space, parity with
    /// \c is_tensor's `nbr_char(input, ' ') == 1` precondition). Does NOT
    /// compare against an expected base name; caller does
    /// `if (auto p = parse_tensor(s); p && p->base_name == expected) ...`.
    std::optional<TensorParse> parse_tensor(std::string_view input);
} // namespace Kadath
