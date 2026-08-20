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
    along with Kadath. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "Linear_algebra/captured_linear_system.hpp"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace Kadath
{

    constexpr std::uint32_t kProlongationSemanticSchemaVersion = 2;

    /**
     * Resolution-invariant identity of one captured column.
     *
     * FIELD equality deliberately excludes the flat basis_mode and the
     * resolution-dependent coefficient counts. It uses the exact schema-v2
     * component and admissible coefficient coordinates instead. Non-field
     * equality uses ordinal_within_group inside
     * (var_name_hash, domain, column_class), as required by the archive
     * contract; its field-only members stay at -1.
     */
    struct ProlongationSemanticKey {
        bool is_field = false;
        std::uint64_t var_name_hash = 0;
        std::int32_t domain = -1;
        CapturedColumnClass column_class = CapturedColumnClass::Unknown;
        std::int32_t term_idx = -1;
        std::int32_t var_idx = -1;
        std::int32_t var_double_idx = -1;
        std::int32_t vardom_param = -1;
        std::int32_t domain_type_id = -1;
        std::int32_t tensor_component = -1;
        std::int32_t coefficient_i = -1;
        std::int32_t coefficient_j = -1;
        std::int32_t coefficient_k = -1;
        std::int32_t ordinal_within_group = -1;

        auto operator<=>(const ProlongationSemanticKey&) const = default;
    };

    struct ProlongationSemanticColumn {
        ProlongationSemanticKey key;
        // Full-space diagnostic index from the archive. The emitted MUMPS
        // permutation is indexed by this vector's reduced-column order, not by
        // original_column.
        std::int32_t original_column = -1;
        std::int32_t coefficient_nr = -1;
        std::int32_t coefficient_nt = -1;
        std::int32_t coefficient_np = -1;
    };

    bool captured_column_class_is_field(CapturedColumnClass column_class);

    // Refuses legacy archives with a role-specific message. This is kept
    // separate so the replay arm can fail before launching donor analyses.
    void require_prolongation_semantics(std::uint32_t schema_version,
                                        std::string_view archive_role);

    std::vector<ProlongationSemanticColumn>
    extract_prolongation_semantics(std::uint32_t schema_version,
                                   std::span<const CapturedColumnTag> column_tags,
                                   std::string_view archive_role);

    // A deterministic scalar resolution coordinate for ordering archives when
    // no filename or application-specific resolution is available. It is the
    // largest per-direction coefficient count among FIELD columns.
    double infer_prolongation_resolution(
        std::span<const ProlongationSemanticColumn> columns);

    struct ProlongationOrderingObservation {
        double resolution = 0.0;
        std::vector<ProlongationSemanticColumn> columns;
        // MUMPS convention: entry i is the one-based pivot position of columns[i].
        std::vector<int> permutation_1based;
    };

    struct ProlongationOptions {
        // A key is stable when max(rank_fraction)-min(rank_fraction) does not
        // exceed this absolute tolerance.
        double stable_rank_spread_tolerance = 0.02;
        // FIELD columns that defeat every widening pass are sent to the
        // elimination tail instead of aborting, as long as they stay below
        // this fraction of the target columns. Beyond it the ensemble is
        // presumed mismatched (for example QE coarse archives against an FB
        // target) and the group census throws.
        double maximum_defaulted_column_fraction = 0.001;
    };

    struct ProlongationDiagnostics {
        std::size_t exact_semantic_hits = 0;
        std::size_t interpolated_columns = 0;
        std::size_t single_observation_keys = 0;
        std::size_t stable_keys = 0;
        std::size_t monotone_increasing_keys = 0;
        std::size_t monotone_decreasing_keys = 0;
        std::size_t nonmonotone_keys = 0;
        std::size_t observed_rank_spread_keys = 0;
        double mean_observed_rank_spread = 0.0;
        double maximum_observed_rank_spread = 0.0;
        // Zero-hit FIELD groups whose members matched coarse members after
        // per-axis top anchoring (i -> nr - i). Tail structures such as tau
        // and quotient remainders are keyed relative to the coefficient
        // count, so their absolute indices never coincide across
        // resolutions. The single/stable/monotone key counters above include
        // anchor-recovered keys.
        std::size_t anchor_recovered_groups = 0;
        std::size_t anchor_recovered_columns = 0;
        // New FIELD modes whose whole semantic group is born at the target
        // resolution; placed next to the nearest sibling tensor component of
        // the same variable, domain, and term.
        std::size_t component_widened_columns = 0;
        // Born groups without even a sibling component; placed next to the
        // nearest hit of the same domain and column class.
        std::size_t domain_widened_columns = 0;
        // Columns that defeated every widening pass and were sent to the
        // elimination tail under maximum_defaulted_column_fraction.
        std::size_t defaulted_tail_columns = 0;
    };

    struct ProlongationResult {
        // Indexed by the target's reduced-column order. Entry i is the
        // one-based prolonged pivot position of target_columns[i].
        std::vector<int> permutation_1based;
        std::vector<double> predicted_rank_fractions;
        ProlongationDiagnostics diagnostics;
    };

    /**
     * Learn a deterministic total order from one or more coarse observations.
     *
     * Stable keys use their mean fractional pivot rank. Monotone keys use a
     * least-squares line in inverse resolution, evaluated at target_resolution.
     * Non-monotone keys use the finest observation. Unseen FIELD modes use the
     * nearest exact-hit target neighbor in normalized (i/nr,j/nt,k/np) space;
     * equal-distance ties select the lexicographically smaller semantic key.
     * Non-field groups inherit the observed rank band in ordinal order.
     */
    ProlongationResult build_prolonged_order(
        std::span<const ProlongationOrderingObservation> coarse_observations,
        double target_resolution,
        std::span<const ProlongationSemanticColumn> target_columns,
        const ProlongationOptions& options = {});

    // Throws std::invalid_argument unless permutation is exactly a bijection of
    // [1,n]. Public for replay-side refusal and focused unit tests.
    void validate_prolonged_permutation(std::span<const int> permutation_1based);

} // namespace Kadath
