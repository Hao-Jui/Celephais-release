/*
    Copyright 2026 Kadath contributors

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

#include "Linear_algebra/mumps_analysis_hints.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace Kadath
{
    namespace
    {

        constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
        constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

        void hash_u64(std::uint64_t& hash, std::uint64_t value)
        {
            for (int byte = 0; byte < 8; ++byte) {
                hash ^= (value >> (8 * byte)) & 0xffU;
                hash *= kFnvPrime;
            }
        }

        bool known_row_taxonomy(RowTaxonomy value)
        {
            switch (value) {
                case RowTaxonomy::Vol:
                case RowTaxonomy::TauBc:
                case RowTaxonomy::TauMatch:
                case RowTaxonomy::GlobalInt:
                    return true;
                case RowTaxonomy::Unknown:
                    return false;
            }
            return false;
        }

        bool known_column_class(ColumnClass value)
        {
            switch (value) {
                case ColumnClass::FieldUnknown:
                case ColumnClass::FieldInterior:
                case ColumnClass::FieldBoundary:
                case ColumnClass::FieldInteriorVol:
                case ColumnClass::FieldBoundaryTau:
                case ColumnClass::FieldOuterShellTau:
                case ColumnClass::FieldMatching:
                case ColumnClass::FieldGauge:
                case ColumnClass::VarDomain:
                case ColumnClass::ScalarGlobal:
                    return true;
                case ColumnClass::Unknown:
                    return false;
            }
            return false;
        }

        void validate_pattern(const MumpsAnalysisPatternView& pattern)
        {
            if (pattern.n <= 0 || pattern.nnz < 0) {
                throw std::invalid_argument("MUMPS analysis hints require positive n and nonnegative nnz");
            }
            if (static_cast<unsigned long long>(pattern.nnz) >
                static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
                throw std::length_error("MUMPS analysis hint nnz does not fit in addressable memory");
            }
            if (pattern.nnz > 0 && (pattern.irn_1based == nullptr || pattern.jcn_1based == nullptr)) {
                throw std::invalid_argument("MUMPS analysis hints received null non-empty COO pointers");
            }
            if (pattern.rows.size() != static_cast<std::size_t>(pattern.n) ||
                pattern.columns.size() != static_cast<std::size_t>(pattern.n)) {
                throw std::invalid_argument("MUMPS analysis hint metadata must match the factor dimension");
            }
            for (int i = 0; i < pattern.n; ++i) {
                if (pattern.rows[static_cast<std::size_t>(i)].original_row < 0 ||
                    pattern.columns[static_cast<std::size_t>(i)].original_column < 0) {
                    throw std::invalid_argument("MUMPS analysis hint metadata has a negative original index");
                }
                const auto& row = pattern.rows[static_cast<std::size_t>(i)];
                const auto& column = pattern.columns[static_cast<std::size_t>(i)];
                if (row.domain < -1 || row.domain_pair < -1) {
                    throw std::invalid_argument("topology ordering row domains must be -1 or nonnegative");
                }
                if (!known_row_taxonomy(row.taxonomy)) {
                    throw std::invalid_argument("topology ordering refuses an unknown row taxonomy");
                }
                if (!known_column_class(column.column_class)) {
                    throw std::invalid_argument("topology ordering refuses an unknown column classification");
                }
                if (column.incidence_role < -1 || column.incidence_role > 1) {
                    throw std::invalid_argument("topology ordering incidence role must be -1, 0, or 1");
                }
                if (column.domain < -1 || column.term_idx < -1 || column.var_idx < -1 || column.var_double_idx < -1 ||
                    column.vardom_param < -1 || column.basis_mode < -1) {
                    throw std::invalid_argument("topology ordering column numeric fields must be -1 or nonnegative");
                }
                if (column.incidence_role == 0 && column.column_class != ColumnClass::FieldInteriorVol) {
                    throw std::invalid_argument("topology ordering bulk role requires FieldInteriorVol");
                }
            }
            for (long long k = 0; k < pattern.nnz; ++k) {
                const int row = pattern.irn_1based[static_cast<std::size_t>(k)];
                const int col = pattern.jcn_1based[static_cast<std::size_t>(k)];
                if (row < 1 || row > pattern.n || col < 1 || col > pattern.n) {
                    throw std::out_of_range("MUMPS analysis hint COO coordinate lies outside [1,n]");
                }
            }
        }

        std::vector<int> validate_and_invert_permutation(int n, std::span<const int> permutation_1based)
        {
            if (permutation_1based.size() != static_cast<std::size_t>(n)) {
                throw std::invalid_argument("symbolic permutation length does not match matrix dimension");
            }
            std::vector<int> inverse(static_cast<std::size_t>(n), -1);
            for (int original = 0; original < n; ++original) {
                const int position = permutation_1based[static_cast<std::size_t>(original)];
                if (position < 1 || position > n || inverse[static_cast<std::size_t>(position - 1)] != -1) {
                    throw std::invalid_argument("symbolic permutation must be a 1-based bijection");
                }
                inverse[static_cast<std::size_t>(position - 1)] = original;
            }
            return inverse;
        }

        std::vector<std::uint64_t> unique_symmetric_edges(const MumpsAnalysisPatternView& pattern,
                                                          std::span<const int> positions_1based)
        {
            std::vector<std::uint64_t> edges;
            edges.reserve(static_cast<std::size_t>(pattern.nnz));
            for (long long k = 0; k < pattern.nnz; ++k) {
                const int original_row = pattern.irn_1based[static_cast<std::size_t>(k)] - 1;
                const int original_col = pattern.jcn_1based[static_cast<std::size_t>(k)] - 1;
                int row = positions_1based.empty() ? original_row
                                                   : positions_1based[static_cast<std::size_t>(original_row)] - 1;
                int col = positions_1based.empty() ? original_col
                                                   : positions_1based[static_cast<std::size_t>(original_col)] - 1;
                if (row == col)
                    continue;
                const std::uint32_t lo = static_cast<std::uint32_t>(std::min(row, col));
                const std::uint32_t hi = static_cast<std::uint32_t>(std::max(row, col));
                edges.push_back((static_cast<std::uint64_t>(hi) << 32) | lo);
            }
            std::sort(edges.begin(), edges.end());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
            return edges;
        }

        int column_class_rank(ColumnClass value)
        {
            return static_cast<int>(value);
        }

        std::vector<int> set_bits(const std::uint64_t* words, std::size_t word_count, const std::vector<int>& domains)
        {
            std::vector<int> result;
            for (std::size_t word = 0; word < word_count; ++word) {
                std::uint64_t bits = words[word];
                while (bits != 0) {
                    const unsigned bit = std::countr_zero(bits);
                    const std::size_t index = word * 64 + bit;
                    if (index < domains.size())
                        result.push_back(domains[index]);
                    bits &= bits - 1;
                }
            }
            return result;
        }

        int topology_tier(const MumpsAnalysisColumnTag& column, int incidence_role, bool has_tau_match,
                          int support_domain_count)
        {
            if (incidence_role == 0)
                return 0;
            if (column.column_class == ColumnClass::ScalarGlobal)
                return 4;
            if (column.column_class == ColumnClass::FieldGauge || column.column_class == ColumnClass::VarDomain ||
                column.domain < 0)
                return 3;
            if (has_tau_match || support_domain_count > 1 || column.column_class == ColumnClass::FieldMatching)
                return 2;
            return 1;
        }

        struct TopologyWorkspace {
            std::vector<int> domains;
            std::vector<int> domain_rank;
            std::vector<int> tier;
            std::vector<int> incidence_role;
            std::vector<std::vector<int>> support_domains;
            std::vector<unsigned char> has_tau_match;
            int unresolved_matching_rows = 0;
        };

        TopologyWorkspace build_topology_workspace(const MumpsAnalysisPatternView& pattern)
        {
            TopologyWorkspace workspace;
            for (const auto& column : pattern.columns) {
                if (column.domain >= 0)
                    workspace.domains.push_back(column.domain);
            }
            for (const auto& row : pattern.rows) {
                if (row.domain >= 0)
                    workspace.domains.push_back(row.domain);
                if (row.domain_pair >= 0)
                    workspace.domains.push_back(row.domain_pair);
            }
            std::sort(workspace.domains.begin(), workspace.domains.end());
            workspace.domains.erase(std::unique(workspace.domains.begin(), workspace.domains.end()),
                                    workspace.domains.end());
            if (workspace.domains.size() > 4096) {
                throw std::length_error("topology ordering supports at most 4096 physical domains");
            }

            std::unordered_map<int, int> domain_index;
            for (std::size_t i = 0; i < workspace.domains.size(); ++i)
                domain_index.emplace(workspace.domains[i], static_cast<int>(i));
            const std::size_t word_count = (workspace.domains.size() + 63) / 64;
            const std::size_t bit_count = static_cast<std::size_t>(pattern.n) * word_count;
            if (word_count != 0 && bit_count / word_count != static_cast<std::size_t>(pattern.n))
                throw std::length_error("topology incidence workspace size overflow");
            std::vector<std::uint64_t> row_domain_bits(bit_count, 0);
            std::vector<std::uint64_t> column_domain_bits(bit_count, 0);
            std::vector<unsigned char> has_same_domain_vol(static_cast<std::size_t>(pattern.n), 0);
            std::vector<unsigned char> has_cross_domain_vol(static_cast<std::size_t>(pattern.n), 0);
            workspace.has_tau_match.assign(static_cast<std::size_t>(pattern.n), 0);

            for (long long k = 0; k < pattern.nnz; ++k) {
                const int row = pattern.irn_1based[static_cast<std::size_t>(k)] - 1;
                const int col = pattern.jcn_1based[static_cast<std::size_t>(k)] - 1;
                const int domain = pattern.columns[static_cast<std::size_t>(col)].domain;
                const auto found = domain_index.find(domain);
                if (found != domain_index.end()) {
                    const std::size_t bit = static_cast<std::size_t>(found->second);
                    row_domain_bits[static_cast<std::size_t>(row) * word_count + bit / 64] |= std::uint64_t{1}
                                                                                              << (bit % 64);
                }
            }

            for (long long k = 0; k < pattern.nnz; ++k) {
                const int row = pattern.irn_1based[static_cast<std::size_t>(k)] - 1;
                const int col = pattern.jcn_1based[static_cast<std::size_t>(k)] - 1;
                const auto& row_tag = pattern.rows[static_cast<std::size_t>(row)];
                const auto& column = pattern.columns[static_cast<std::size_t>(col)];
                if (row_tag.domain >= 0) {
                    const std::size_t bit = static_cast<std::size_t>(domain_index.at(row_tag.domain));
                    column_domain_bits[static_cast<std::size_t>(col) * word_count + bit / 64] |= std::uint64_t{1}
                                                                                                 << (bit % 64);
                }
                if (row_tag.taxonomy == RowTaxonomy::TauMatch)
                    workspace.has_tau_match[static_cast<std::size_t>(col)] = 1;
                if (row_tag.taxonomy == RowTaxonomy::Vol) {
                    if (row_tag.domain >= 0 && row_tag.domain == column.domain)
                        has_same_domain_vol[static_cast<std::size_t>(col)] = 1;
                    else if (row_tag.domain >= 0)
                        has_cross_domain_vol[static_cast<std::size_t>(col)] = 1;
                }
            }

            const std::size_t domain_count = workspace.domains.size();
            std::vector<std::vector<unsigned char>> adjacency(domain_count,
                                                              std::vector<unsigned char>(domain_count, 0));
            const auto add_clique = [&](const std::vector<int>& domain_ids) {
                for (std::size_t i = 0; i < domain_ids.size(); ++i) {
                    const int left = domain_index.at(domain_ids[i]);
                    for (std::size_t j = i + 1; j < domain_ids.size(); ++j) {
                        const int right = domain_index.at(domain_ids[j]);
                        adjacency[static_cast<std::size_t>(left)][static_cast<std::size_t>(right)] = 1;
                        adjacency[static_cast<std::size_t>(right)][static_cast<std::size_t>(left)] = 1;
                    }
                }
            };

            for (int row = 0; row < pattern.n; ++row) {
                if (pattern.rows[static_cast<std::size_t>(row)].taxonomy != RowTaxonomy::TauMatch)
                    continue;
                const std::uint64_t* row_words =
                    word_count == 0 ? nullptr : row_domain_bits.data() + static_cast<std::size_t>(row) * word_count;
                std::vector<int> endpoints = set_bits(row_words, word_count, workspace.domains);
                const auto& row_tag = pattern.rows[static_cast<std::size_t>(row)];
                if (endpoints.size() < 2) {
                    if (row_tag.domain >= 0)
                        endpoints.push_back(row_tag.domain);
                    if (row_tag.domain_pair >= 0)
                        endpoints.push_back(row_tag.domain_pair);
                    std::sort(endpoints.begin(), endpoints.end());
                    endpoints.erase(std::unique(endpoints.begin(), endpoints.end()), endpoints.end());
                }
                if (endpoints.size() < 2)
                    ++workspace.unresolved_matching_rows;
                else
                    add_clique(endpoints);
            }

            workspace.support_domains.resize(static_cast<std::size_t>(pattern.n));
            workspace.incidence_role.resize(static_cast<std::size_t>(pattern.n), 1);
            workspace.tier.resize(static_cast<std::size_t>(pattern.n), 4);
            std::vector<int> bulk_count(domain_count, 0);
            for (int col = 0; col < pattern.n; ++col) {
                auto& support = workspace.support_domains[static_cast<std::size_t>(col)];
                const std::uint64_t* column_words =
                    word_count == 0 ? nullptr : column_domain_bits.data() + static_cast<std::size_t>(col) * word_count;
                support = set_bits(column_words, word_count, workspace.domains);
                const auto& column = pattern.columns[static_cast<std::size_t>(col)];
                int role = column.incidence_role;
                if (role != 0 && role != 1) {
                    role = column.column_class == ColumnClass::FieldInteriorVol && column.domain >= 0 &&
                                   has_same_domain_vol[static_cast<std::size_t>(col)] != 0 &&
                                   has_cross_domain_vol[static_cast<std::size_t>(col)] == 0
                               ? 0
                               : 1;
                }
                if (role == 0 && column.domain < 0) {
                    throw std::invalid_argument("topology ordering found a bulk column without a domain");
                }
                workspace.incidence_role[static_cast<std::size_t>(col)] = role;
                workspace.tier[static_cast<std::size_t>(col)] =
                    topology_tier(column, role, workspace.has_tau_match[static_cast<std::size_t>(col)] != 0,
                                  static_cast<int>(support.size()));
                if (role == 0)
                    ++bulk_count[static_cast<std::size_t>(domain_index.at(column.domain))];
                if (role == 1 && support.size() > 1)
                    add_clique(support);
            }

            std::vector<unsigned char> alive(domain_count, 1);
            workspace.domain_rank.assign(domain_count, -1);
            for (std::size_t step = 0; step < domain_count; ++step) {
                int best = -1;
                std::tuple<long long, int, int, int> best_key;
                for (std::size_t candidate = 0; candidate < domain_count; ++candidate) {
                    if (!alive[candidate])
                        continue;
                    std::vector<int> neighbours;
                    for (std::size_t other = 0; other < domain_count; ++other) {
                        if (alive[other] && adjacency[candidate][other])
                            neighbours.push_back(static_cast<int>(other));
                    }
                    long long missing = 0;
                    for (std::size_t i = 0; i < neighbours.size(); ++i) {
                        for (std::size_t j = i + 1; j < neighbours.size(); ++j) {
                            if (!adjacency[static_cast<std::size_t>(neighbours[i])]
                                          [static_cast<std::size_t>(neighbours[j])])
                                ++missing;
                        }
                    }
                    const auto key = std::make_tuple(missing, static_cast<int>(neighbours.size()),
                                                     bulk_count[candidate], workspace.domains[candidate]);
                    if (best < 0 || key < best_key) {
                        best = static_cast<int>(candidate);
                        best_key = key;
                    }
                }
                std::vector<int> neighbours;
                for (std::size_t other = 0; other < domain_count; ++other) {
                    if (alive[other] && adjacency[static_cast<std::size_t>(best)][other])
                        neighbours.push_back(static_cast<int>(other));
                }
                for (std::size_t i = 0; i < neighbours.size(); ++i) {
                    for (std::size_t j = i + 1; j < neighbours.size(); ++j) {
                        adjacency[static_cast<std::size_t>(neighbours[i])][static_cast<std::size_t>(neighbours[j])] = 1;
                        adjacency[static_cast<std::size_t>(neighbours[j])][static_cast<std::size_t>(neighbours[i])] = 1;
                    }
                }
                alive[static_cast<std::size_t>(best)] = 0;
                workspace.domain_rank[static_cast<std::size_t>(best)] = static_cast<int>(step);
            }
            return workspace;
        }

        int max_domain_rank(const std::vector<int>& support, const std::vector<int>& domains,
                            const std::vector<int>& ranks, int fallback_domain)
        {
            int result = -1;
            for (int domain : support) {
                const auto found = std::lower_bound(domains.begin(), domains.end(), domain);
                if (found != domains.end() && *found == domain)
                    result = std::max(result, ranks[static_cast<std::size_t>(found - domains.begin())]);
            }
            if (result < 0 && fallback_domain >= 0) {
                const auto found = std::lower_bound(domains.begin(), domains.end(), fallback_domain);
                if (found != domains.end() && *found == fallback_domain)
                    result = ranks[static_cast<std::size_t>(found - domains.begin())];
            }
            return result < 0 ? std::numeric_limits<int>::max() : result;
        }

    } // namespace

    TopologyOrderingHints build_topology_ordering_hints(const MumpsAnalysisPatternView& pattern)
    {
        validate_pattern(pattern);
        TopologyWorkspace workspace = build_topology_workspace(pattern);

        std::vector<int> order(static_cast<std::size_t>(pattern.n));
        std::iota(order.begin(), order.end(), 0);
        std::vector<int> domain_position(static_cast<std::size_t>(pattern.n));
        for (int col = 0; col < pattern.n; ++col) {
            const auto& item = pattern.columns[static_cast<std::size_t>(col)];
            domain_position[static_cast<std::size_t>(col)] =
                max_domain_rank(workspace.support_domains[static_cast<std::size_t>(col)], workspace.domains,
                                workspace.domain_rank, item.domain);
        }
        std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
            const auto left_index = static_cast<std::size_t>(left);
            const auto right_index = static_cast<std::size_t>(right);
            const int left_tier = workspace.tier[left_index];
            const int right_tier = workspace.tier[right_index];
            if (left_tier != right_tier)
                return left_tier < right_tier;
            if (domain_position[left_index] != domain_position[right_index])
                return domain_position[left_index] < domain_position[right_index];
            const auto& left_support = workspace.support_domains[left_index];
            const auto& right_support = workspace.support_domains[right_index];
            if (left_support != right_support)
                return left_support < right_support;
            const auto& left_item = pattern.columns[left_index];
            const auto& right_item = pattern.columns[right_index];
            return std::tie(left_item.column_class, left_item.var_idx, left_item.term_idx, left_item.vardom_param,
                            left_item.var_name_hash, left_item.basis_mode, left_item.original_column, left) <
                   std::tie(right_item.column_class, right_item.var_idx, right_item.term_idx, right_item.vardom_param,
                            right_item.var_name_hash, right_item.basis_mode, right_item.original_column, right);
        });

        TopologyOrderingHints result;
        result.permutation_1based.resize(static_cast<std::size_t>(pattern.n));
        for (int position = 0; position < pattern.n; ++position) {
            result.permutation_1based[static_cast<std::size_t>(order[static_cast<std::size_t>(position)])] =
                position + 1;
        }
        result.domain_elimination_order.resize(workspace.domains.size());
        for (std::size_t domain = 0; domain < workspace.domains.size(); ++domain) {
            const int rank = workspace.domain_rank[domain];
            result.domain_elimination_order[static_cast<std::size_t>(rank)] = workspace.domains[domain];
        }
        result.unresolved_matching_rows = workspace.unresolved_matching_rows;
        for (int tier : workspace.tier) {
            switch (tier) {
                case 0:
                    ++result.bulk_columns;
                    break;
                case 1:
                    ++result.local_boundary_columns;
                    break;
                case 2:
                    ++result.seam_columns;
                    break;
                case 3:
                    ++result.auxiliary_columns;
                    break;
                case 4:
                    ++result.global_columns;
                    break;
                default:
                    throw std::logic_error("invalid topology tier");
            }
        }
        return result;
    }

    ExactBlockAnalysisHints build_exact_topology_blocks(const MumpsAnalysisPatternView& pattern,
                                                        const TopologyOrderingHints& topology)
    {
        validate_pattern(pattern);
        validate_and_invert_permutation(pattern.n, topology.permutation_1based);
        TopologyWorkspace workspace = build_topology_workspace(pattern);
        const std::vector<std::uint64_t> edges = unique_symmetric_edges(pattern, {});

        std::vector<std::size_t> degree(static_cast<std::size_t>(pattern.n), 1);
        for (std::uint64_t edge : edges) {
            ++degree[static_cast<std::size_t>(static_cast<std::uint32_t>(edge))];
            ++degree[static_cast<std::size_t>(static_cast<std::uint32_t>(edge >> 32))];
        }
        std::vector<std::size_t> offsets(static_cast<std::size_t>(pattern.n + 1), 0);
        for (int i = 0; i < pattern.n; ++i)
            offsets[static_cast<std::size_t>(i + 1)] =
                offsets[static_cast<std::size_t>(i)] + degree[static_cast<std::size_t>(i)];
        std::vector<int> closed(offsets.back());
        std::vector<std::size_t> cursor = offsets;
        for (int i = 0; i < pattern.n; ++i)
            closed[cursor[static_cast<std::size_t>(i)]++] = i;
        for (std::uint64_t edge : edges) {
            const int lo = static_cast<int>(static_cast<std::uint32_t>(edge));
            const int hi = static_cast<int>(static_cast<std::uint32_t>(edge >> 32));
            closed[cursor[static_cast<std::size_t>(lo)]++] = hi;
            closed[cursor[static_cast<std::size_t>(hi)]++] = lo;
        }
        std::vector<std::uint64_t> adjacency_hash(static_cast<std::size_t>(pattern.n), 0);
        for (int i = 0; i < pattern.n; ++i) {
            const auto begin = closed.begin() + static_cast<std::ptrdiff_t>(offsets[static_cast<std::size_t>(i)]);
            const auto end = closed.begin() + static_cast<std::ptrdiff_t>(offsets[static_cast<std::size_t>(i + 1)]);
            std::sort(begin, end);
            std::uint64_t hash = kFnvOffset;
            for (auto it = begin; it != end; ++it)
                hash_u64(hash, static_cast<std::uint64_t>(*it));
            adjacency_hash[static_cast<std::size_t>(i)] = hash;
        }

        std::vector<int> variables(static_cast<std::size_t>(pattern.n));
        std::iota(variables.begin(), variables.end(), 0);
        const auto semantic_key = [&](int variable) {
            const auto& column = pattern.columns[static_cast<std::size_t>(variable)];
            return std::tuple{workspace.tier[static_cast<std::size_t>(variable)], column.domain,
                              column_class_rank(column.column_class), degree[static_cast<std::size_t>(variable)],
                              adjacency_hash[static_cast<std::size_t>(variable)]};
        };
        std::sort(variables.begin(), variables.end(), [&](int left, int right) {
            const auto left_key = semantic_key(left);
            const auto right_key = semantic_key(right);
            if (left_key != right_key)
                return left_key < right_key;
            const auto left_begin =
                closed.begin() + static_cast<std::ptrdiff_t>(offsets[static_cast<std::size_t>(left)]);
            const auto left_end =
                closed.begin() + static_cast<std::ptrdiff_t>(offsets[static_cast<std::size_t>(left + 1)]);
            const auto right_begin =
                closed.begin() + static_cast<std::ptrdiff_t>(offsets[static_cast<std::size_t>(right)]);
            const auto right_end =
                closed.begin() + static_cast<std::ptrdiff_t>(offsets[static_cast<std::size_t>(right + 1)]);
            if (!std::equal(left_begin, left_end, right_begin, right_end))
                return std::lexicographical_compare(left_begin, left_end, right_begin, right_end);
            return topology.permutation_1based[static_cast<std::size_t>(left)] <
                   topology.permutation_1based[static_cast<std::size_t>(right)];
        });

        std::vector<std::vector<int>> blocks;
        for (int variable : variables) {
            bool same = false;
            if (!blocks.empty()) {
                const int representative = blocks.back().front();
                same =
                    semantic_key(representative) == semantic_key(variable) &&
                    std::equal(
                        closed.begin() + static_cast<std::ptrdiff_t>(offsets[static_cast<std::size_t>(representative)]),
                        closed.begin() +
                            static_cast<std::ptrdiff_t>(offsets[static_cast<std::size_t>(representative + 1)]),
                        closed.begin() + static_cast<std::ptrdiff_t>(offsets[static_cast<std::size_t>(variable)]),
                        closed.begin() + static_cast<std::ptrdiff_t>(offsets[static_cast<std::size_t>(variable + 1)]));
            }
            if (!same)
                blocks.emplace_back();
            blocks.back().push_back(variable);
        }
        for (auto& block : blocks) {
            std::sort(block.begin(), block.end(), [&](int left, int right) {
                return topology.permutation_1based[static_cast<std::size_t>(left)] <
                       topology.permutation_1based[static_cast<std::size_t>(right)];
            });
        }
        std::sort(blocks.begin(), blocks.end(), [&](const auto& left, const auto& right) {
            return topology.permutation_1based[static_cast<std::size_t>(left.front())] <
                   topology.permutation_1based[static_cast<std::size_t>(right.front())];
        });

        ExactBlockAnalysisHints result;
        result.blkptr_1based.push_back(1);
        for (const auto& block : blocks) {
            if (block.size() > 1) {
                ++result.nonsingleton_blocks;
                result.compressed_variables += static_cast<int>(block.size()) - 1;
            }
            for (int variable : block)
                result.blkvar_1based.push_back(variable + 1);
            result.blkptr_1based.push_back(static_cast<int>(result.blkvar_1based.size()) + 1);
        }
        return result;
    }

    SymbolicEliminationDiagnostics analyze_symbolic_elimination(const MumpsAnalysisPatternView& pattern,
                                                                std::span<const int> permutation_1based)
    {
        validate_pattern(pattern);
        validate_and_invert_permutation(pattern.n, permutation_1based);
        const std::vector<std::uint64_t> edges = unique_symmetric_edges(pattern, permutation_1based);

        std::vector<int> parent(static_cast<std::size_t>(pattern.n), -1);
        std::vector<int> ancestor(static_cast<std::size_t>(pattern.n), -1);
        std::size_t edge_cursor = 0;
        for (int column = 0; column < pattern.n; ++column) {
            while (edge_cursor < edges.size() &&
                   static_cast<int>(static_cast<std::uint32_t>(edges[edge_cursor] >> 32)) == column) {
                int row = static_cast<int>(static_cast<std::uint32_t>(edges[edge_cursor]));
                while (row != -1 && row < column) {
                    const int next = ancestor[static_cast<std::size_t>(row)];
                    ancestor[static_cast<std::size_t>(row)] = column;
                    if (next == -1)
                        parent[static_cast<std::size_t>(row)] = column;
                    row = next;
                }
                ++edge_cursor;
            }
        }

        std::vector<std::uint64_t> column_counts(static_cast<std::size_t>(pattern.n), 1);
        std::vector<int> mark(static_cast<std::size_t>(pattern.n), -1);
        edge_cursor = 0;
        for (int row = 0; row < pattern.n; ++row) {
            mark[static_cast<std::size_t>(row)] = row;
            while (edge_cursor < edges.size() &&
                   static_cast<int>(static_cast<std::uint32_t>(edges[edge_cursor] >> 32)) == row) {
                int column = static_cast<int>(static_cast<std::uint32_t>(edges[edge_cursor]));
                while (column != -1 && mark[static_cast<std::size_t>(column)] != row) {
                    mark[static_cast<std::size_t>(column)] = row;
                    ++column_counts[static_cast<std::size_t>(column)];
                    column = parent[static_cast<std::size_t>(column)];
                }
                ++edge_cursor;
            }
        }

        SymbolicEliminationDiagnostics result;
        result.parent_1based.resize(static_cast<std::size_t>(pattern.n), 0);
        std::vector<int> depth(static_cast<std::size_t>(pattern.n), 0);
        for (int node = pattern.n - 1; node >= 0; --node) {
            const int node_parent = parent[static_cast<std::size_t>(node)];
            result.parent_1based[static_cast<std::size_t>(node)] = node_parent < 0 ? 0 : node_parent + 1;
            depth[static_cast<std::size_t>(node)] =
                node_parent < 0 ? 0 : depth[static_cast<std::size_t>(node_parent)] + 1;
            result.max_tree_depth = std::max(result.max_tree_depth, depth[static_cast<std::size_t>(node)]);
        }
        result.depth_histogram.assign(static_cast<std::size_t>(result.max_tree_depth + 1), 0);
        for (int value : depth)
            ++result.depth_histogram[static_cast<std::size_t>(value)];

        std::uint64_t maximum = 1;
        for (std::uint64_t count : column_counts) {
            if (result.filled_lower_nnz_proxy > std::numeric_limits<std::uint64_t>::max() - count) {
                throw std::overflow_error("symbolic fill count overflow");
            }
            result.filled_lower_nnz_proxy += count;
            maximum = std::max(maximum, count);
        }
        result.max_singleton_front_width = maximum;
        std::uint64_t upper = 1;
        while (true) {
            SymbolicFrontHistogramBin bin;
            bin.lower_exclusive = upper == 1 ? 0 : upper / 2;
            bin.upper_inclusive = upper;
            for (std::uint64_t count : column_counts) {
                if (count > bin.lower_exclusive && count <= bin.upper_inclusive)
                    ++bin.count;
            }
            result.front_width_histogram.push_back(bin);
            if (upper >= maximum || upper > std::numeric_limits<std::uint64_t>::max() / 2)
                break;
            upper *= 2;
        }
        result.parent_hash = kFnvOffset;
        for (int value : result.parent_1based)
            hash_u64(result.parent_hash, static_cast<std::uint64_t>(value));
        return result;
    }

} // namespace Kadath
