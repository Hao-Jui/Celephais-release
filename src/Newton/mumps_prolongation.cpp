/*
    Copyright 2026 Kadath contributors

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Linear_algebra/mumps_prolongation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

namespace Kadath
{
    namespace
    {
        struct NonFieldGroupKey {
            std::uint64_t var_name_hash = 0;
            std::int32_t domain = -1;
            CapturedColumnClass column_class = CapturedColumnClass::Unknown;

            auto operator<=>(const NonFieldGroupKey&) const = default;
        };

        struct FieldGroupKey {
            std::uint64_t var_name_hash = 0;
            std::int32_t domain = -1;
            CapturedColumnClass column_class = CapturedColumnClass::Unknown;
            std::int32_t term_idx = -1;
            std::int32_t var_idx = -1;
            std::int32_t domain_type_id = -1;
            std::int32_t tensor_component = -1;

            auto operator<=>(const FieldGroupKey&) const = default;
        };

        struct RankObservation {
            double resolution = 0.0;
            double rank_fraction = 0.0;
        };

        FieldGroupKey field_group(const ProlongationSemanticColumn& column)
        {
            const auto& key = column.key;
            return {key.var_name_hash, key.domain, key.column_class, key.term_idx,
                    key.var_idx, key.domain_type_id, key.tensor_component};
        }

        // Everything in FieldGroupKey except the tensor component: the
        // last-resort neighbor pool for groups born at the target
        // resolution, whose sibling components share the matrix block.
        FieldGroupKey component_widened_group(const FieldGroupKey& group)
        {
            FieldGroupKey widened = group;
            widened.tensor_component = -1;
            return widened;
        }

        NonFieldGroupKey nonfield_group(const ProlongationSemanticColumn& column)
        {
            const auto& key = column.key;
            return {key.var_name_hash, key.domain, key.column_class};
        }

        // Exemplar of a fine semantic group absent from every coarse archive,
        // kept so the failure names the offending domains/components instead
        // of aborting anonymously.
        struct MissingGroupExemplar {
            std::size_t column_count = 0;
            ProlongationSemanticColumn exemplar;
        };

        std::string describe_missing_groups(
            std::string_view field_phrase,
            const std::map<FieldGroupKey, MissingGroupExemplar>& field_missing,
            std::string_view nonfield_phrase,
            const std::map<NonFieldGroupKey, MissingGroupExemplar>& nonfield_missing)
        {
            constexpr std::size_t maximum_reported_groups = 12;
            std::size_t field_columns = 0;
            for (const auto& [group, entry] : field_missing) {
                (void)group;
                field_columns += entry.column_count;
            }
            std::size_t nonfield_columns = 0;
            for (const auto& [group, entry] : nonfield_missing) {
                (void)group;
                nonfield_columns += entry.column_count;
            }
            std::ostringstream message;
            message << "prolongation cannot place " << field_columns
                    << " FIELD columns in " << field_missing.size()
                    << " semantic groups (" << field_phrase << ") and "
                    << nonfield_columns << " non-field columns in "
                    << nonfield_missing.size() << " groups ("
                    << nonfield_phrase << "); exemplars:";
            std::size_t reported = 0;
            for (const auto& [group, entry] : field_missing) {
                if (reported == maximum_reported_groups)
                    break;
                ++reported;
                const auto& key = entry.exemplar.key;
                message << " [field var=0x" << std::hex << key.var_name_hash
                        << std::dec << " dom=" << key.domain
                        << " type=" << key.domain_type_id
                        << " comp=" << key.tensor_component
                        << " class=" << static_cast<int>(key.column_class)
                        << " term=" << key.term_idx << " vidx=" << key.var_idx
                        << " n=" << entry.column_count << " mode=("
                        << key.coefficient_i << ',' << key.coefficient_j << ','
                        << key.coefficient_k << ")/("
                        << entry.exemplar.coefficient_nr << ','
                        << entry.exemplar.coefficient_nt << ','
                        << entry.exemplar.coefficient_np << ")]";
                (void)group;
            }
            for (const auto& [group, entry] : nonfield_missing) {
                if (reported == maximum_reported_groups)
                    break;
                ++reported;
                const auto& key = entry.exemplar.key;
                message << " [nonfield var=0x" << std::hex << key.var_name_hash
                        << std::dec << " dom=" << key.domain << " class="
                        << static_cast<int>(key.column_class)
                        << " n=" << entry.column_count << ']';
                (void)group;
            }
            const std::size_t total_groups =
                field_missing.size() + nonfield_missing.size();
            if (total_groups > reported)
                message << " ... (" << (total_groups - reported)
                        << " more groups)";
            return message.str();
        }

        double rank_fraction(int position_1based, std::size_t column_count)
        {
            if (column_count <= 1)
                return 0.0;
            return static_cast<double>(position_1based - 1) /
                   static_cast<double>(column_count - 1);
        }

        double normalized_coordinate(std::int32_t coordinate, std::int32_t count)
        {
            if (coordinate < 0 || count <= 0 || coordinate >= count)
                throw std::invalid_argument(
                    "prolongation semantic coordinate lies outside its coefficient count");
            // The owner-specified distance is i/nr (and analogously for theta,
            // phi), not i/(nr-1).
            return static_cast<double>(coordinate) / static_cast<double>(count);
        }

        double squared_mode_distance(const ProlongationSemanticColumn& left,
                                     const ProlongationSemanticColumn& right)
        {
            const double di = normalized_coordinate(left.key.coefficient_i, left.coefficient_nr) -
                              normalized_coordinate(right.key.coefficient_i, right.coefficient_nr);
            const double dj = normalized_coordinate(left.key.coefficient_j, left.coefficient_nt) -
                              normalized_coordinate(right.key.coefficient_j, right.coefficient_nt);
            const double dk = normalized_coordinate(left.key.coefficient_k, left.coefficient_np) -
                              normalized_coordinate(right.key.coefficient_k, right.coefficient_np);
            return di * di + dj * dj + dk * dk;
        }

        double clamp_rank(double value)
        {
            return std::clamp(value, 0.0, 1.0);
        }

        // Per-axis anchoring for tail-relative coefficient indices. Bottom
        // keeps the absolute index; top re-keys it as its distance below the
        // coefficient count, which is the invariant coordinate of tau and
        // quotient remainder rows across resolutions.
        struct AnchorCombination {
            bool top_i = false;
            bool top_j = false;
            bool top_k = false;
        };

        std::array<std::int32_t, 3> anchored_mode(
            const ProlongationSemanticColumn& column,
            const AnchorCombination& anchors)
        {
            const auto& key = column.key;
            return {anchors.top_i ? column.coefficient_nr - key.coefficient_i
                                  : key.coefficient_i,
                    anchors.top_j ? column.coefficient_nt - key.coefficient_j
                                  : key.coefficient_j,
                    anchors.top_k ? column.coefficient_np - key.coefficient_k
                                  : key.coefficient_k};
        }

        double inverse_resolution_regression(
            std::span<const RankObservation> observations,
            double target_resolution)
        {
            const double target_x = 1.0 / target_resolution;
            double mean_x = 0.0;
            double mean_y = 0.0;
            for (const auto& observation : observations) {
                mean_x += 1.0 / observation.resolution;
                mean_y += observation.rank_fraction;
            }
            mean_x /= static_cast<double>(observations.size());
            mean_y /= static_cast<double>(observations.size());

            double covariance = 0.0;
            double variance = 0.0;
            for (const auto& observation : observations) {
                const double dx = 1.0 / observation.resolution - mean_x;
                covariance += dx * (observation.rank_fraction - mean_y);
                variance += dx * dx;
            }
            if (!(variance > 0.0))
                throw std::invalid_argument(
                    "prolongation observations do not have distinct resolutions");
            return clamp_rank(mean_y + covariance / variance * (target_x - mean_x));
        }

        void validate_semantic_columns(
            std::span<const ProlongationSemanticColumn> columns,
            std::string_view role)
        {
            std::map<ProlongationSemanticKey, std::size_t> seen;
            for (std::size_t index = 0; index < columns.size(); ++index) {
                const auto& column = columns[index];
                if (column.original_column < 0)
                    throw std::invalid_argument(std::string(role) +
                                                " contains a negative original column");
                if (column.key.is_field) {
                    if (!captured_column_class_is_field(column.key.column_class) ||
                        column.key.var_name_hash == 0 || column.key.domain < 0 ||
                        column.key.term_idx < 0 || column.key.var_idx < 0 ||
                        column.key.domain_type_id < 1 ||
                        column.key.tensor_component < 0) {
                        throw std::invalid_argument(std::string(role) +
                                                    " contains incomplete FIELD semantics");
                    }
                    (void)normalized_coordinate(column.key.coefficient_i,
                                                column.coefficient_nr);
                    (void)normalized_coordinate(column.key.coefficient_j,
                                                column.coefficient_nt);
                    (void)normalized_coordinate(column.key.coefficient_k,
                                                column.coefficient_np);
                } else if (column.key.ordinal_within_group < 0) {
                    throw std::invalid_argument(std::string(role) +
                                                " contains an invalid non-field group ordinal");
                }
                if (!seen.emplace(column.key, index).second)
                    throw std::invalid_argument(std::string(role) +
                                                " contains duplicate semantic column keys");
            }
        }
    } // namespace

    bool captured_column_class_is_field(CapturedColumnClass column_class)
    {
        const auto code = static_cast<std::int32_t>(column_class);
        return code >= static_cast<std::int32_t>(CapturedColumnClass::FieldUnknown) &&
               code <= static_cast<std::int32_t>(CapturedColumnClass::FieldGauge);
    }

    void require_prolongation_semantics(std::uint32_t schema_version,
                                        std::string_view archive_role)
    {
        if (schema_version < kProlongationSemanticSchemaVersion) {
            throw std::invalid_argument(
                "prolongation requires schema v2 semantic column tags; " +
                std::string(archive_role) + " has schema v" +
                std::to_string(schema_version));
        }
        if (schema_version > kProlongationSemanticSchemaVersion) {
            throw std::invalid_argument(
                "prolongation does not support semantic schema v" +
                std::to_string(schema_version) + " in " + std::string(archive_role));
        }
    }

    std::vector<ProlongationSemanticColumn>
    extract_prolongation_semantics(std::uint32_t schema_version,
                                   std::span<const CapturedColumnTag> column_tags,
                                   std::string_view archive_role)
    {
        require_prolongation_semantics(schema_version, archive_role);
        std::map<NonFieldGroupKey, std::int32_t> next_group_ordinal;
        std::vector<ProlongationSemanticColumn> result;
        result.reserve(column_tags.size());
        for (const CapturedColumnTag& tag : column_tags) {
            ProlongationSemanticColumn column;
            column.original_column = tag.original_column;
            auto& key = column.key;
            key.is_field = captured_column_class_is_field(tag.column_class);
            key.var_name_hash = tag.var_name_hash;
            key.domain = tag.domain;
            key.column_class = tag.column_class;
            if (key.is_field) {
                key.term_idx = tag.term_idx;
                key.var_idx = tag.var_idx;
                key.var_double_idx = tag.var_double_idx;
                key.vardom_param = tag.vardom_param;
                key.domain_type_id = tag.domain_type_id;
                key.tensor_component = tag.tensor_component;
                key.coefficient_i = tag.coefficient_i;
                key.coefficient_j = tag.coefficient_j;
                key.coefficient_k = tag.coefficient_k;
                column.coefficient_nr = tag.coefficient_nr;
                column.coefficient_nt = tag.coefficient_nt;
                column.coefficient_np = tag.coefficient_np;
            } else {
                const NonFieldGroupKey group{tag.var_name_hash, tag.domain,
                                             tag.column_class};
                key.ordinal_within_group = next_group_ordinal[group]++;
            }
            result.push_back(column);
        }
        validate_semantic_columns(result, archive_role);
        return result;
    }

    double infer_prolongation_resolution(
        std::span<const ProlongationSemanticColumn> columns)
    {
        std::int32_t resolution = 0;
        for (const auto& column : columns) {
            if (!column.key.is_field)
                continue;
            resolution = std::max(
                {resolution, column.coefficient_nr, column.coefficient_nt,
                 column.coefficient_np});
        }
        if (resolution <= 0)
            throw std::invalid_argument(
                "cannot infer prolongation resolution without FIELD coefficient counts");
        return static_cast<double>(resolution);
    }

    void validate_prolonged_permutation(
        std::span<const int> permutation_1based)
    {
        std::vector<bool> seen(permutation_1based.size(), false);
        for (int position : permutation_1based) {
            if (position < 1 ||
                static_cast<std::size_t>(position) > permutation_1based.size()) {
                throw std::invalid_argument(
                    "prolonged permutation contains an out-of-range position");
            }
            const std::size_t index = static_cast<std::size_t>(position - 1);
            if (seen[index])
                throw std::invalid_argument(
                    "prolonged permutation is not a bijection");
            seen[index] = true;
        }
    }

    ProlongationResult build_prolonged_order(
        std::span<const ProlongationOrderingObservation> coarse_observations,
        double target_resolution,
        std::span<const ProlongationSemanticColumn> target_columns,
        const ProlongationOptions& options)
    {
        if (coarse_observations.empty())
            throw std::invalid_argument(
                "prolongation requires at least one coarse ordering observation");
        if (!std::isfinite(target_resolution) || target_resolution <= 0.0)
            throw std::invalid_argument(
                "prolongation target resolution must be finite and positive");
        if (!std::isfinite(options.stable_rank_spread_tolerance) ||
            options.stable_rank_spread_tolerance < 0.0 ||
            options.stable_rank_spread_tolerance > 1.0) {
            throw std::invalid_argument(
                "prolongation stable-rank tolerance must lie in [0,1]");
        }
        if (target_columns.empty())
            throw std::invalid_argument(
                "prolongation target contains no columns");
        validate_semantic_columns(target_columns, "prolongation target");

        std::map<ProlongationSemanticKey, std::vector<RankObservation>> ranks_by_key;
        std::vector<double> ensemble_resolutions;
        ensemble_resolutions.reserve(coarse_observations.size());
        for (const auto& observation : coarse_observations) {
            if (!std::isfinite(observation.resolution) ||
                observation.resolution <= 0.0) {
                throw std::invalid_argument(
                    "prolongation coarse resolution must be finite and positive");
            }
            if (observation.columns.empty() ||
                observation.columns.size() != observation.permutation_1based.size()) {
                throw std::invalid_argument(
                    "prolongation coarse columns and permutation lengths differ");
            }
            validate_semantic_columns(observation.columns,
                                      "prolongation coarse observation");
            validate_prolonged_permutation(observation.permutation_1based);
            ensemble_resolutions.push_back(observation.resolution);
            for (std::size_t index = 0; index < observation.columns.size(); ++index) {
                ranks_by_key[observation.columns[index].key].push_back(
                    {observation.resolution,
                     rank_fraction(observation.permutation_1based[index],
                                   observation.columns.size())});
            }
        }
        std::sort(ensemble_resolutions.begin(), ensemble_resolutions.end());
        if (std::adjacent_find(ensemble_resolutions.begin(),
                               ensemble_resolutions.end()) !=
            ensemble_resolutions.end()) {
            throw std::invalid_argument(
                "prolongation ensemble contains duplicate resolutions");
        }

        ProlongationResult result;
        result.predicted_rank_fractions.assign(
            target_columns.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<bool> direct_prediction(target_columns.size(), false);
        double observed_rank_spread_sum = 0.0;
        const auto predict_from_observations =
            [&](std::vector<RankObservation> observations) -> double {
            std::sort(observations.begin(), observations.end(),
                      [](const RankObservation& left,
                         const RankObservation& right) {
                          return left.resolution < right.resolution;
                      });
            if (observations.size() == 1) {
                ++result.diagnostics.single_observation_keys;
                return observations.front().rank_fraction;
            }

            const auto [minimum, maximum] = std::minmax_element(
                observations.begin(), observations.end(),
                [](const RankObservation& left,
                   const RankObservation& right) {
                    return left.rank_fraction < right.rank_fraction;
                });
            const double rank_spread =
                maximum->rank_fraction - minimum->rank_fraction;
            ++result.diagnostics.observed_rank_spread_keys;
            observed_rank_spread_sum += rank_spread;
            result.diagnostics.maximum_observed_rank_spread =
                std::max(result.diagnostics.maximum_observed_rank_spread,
                         rank_spread);
            if (rank_spread <=
                options.stable_rank_spread_tolerance) {
                ++result.diagnostics.stable_keys;
                double mean = 0.0;
                for (const auto& observation : observations)
                    mean += observation.rank_fraction;
                return mean / static_cast<double>(observations.size());
            }

            bool nondecreasing = true;
            bool nonincreasing = true;
            for (std::size_t index = 1; index < observations.size(); ++index) {
                nondecreasing =
                    nondecreasing && observations[index - 1].rank_fraction <=
                                         observations[index].rank_fraction;
                nonincreasing =
                    nonincreasing && observations[index - 1].rank_fraction >=
                                         observations[index].rank_fraction;
            }
            if (nondecreasing || nonincreasing) {
                if (nondecreasing)
                    ++result.diagnostics.monotone_increasing_keys;
                else
                    ++result.diagnostics.monotone_decreasing_keys;
                return inverse_resolution_regression(observations,
                                                     target_resolution);
            }
            ++result.diagnostics.nonmonotone_keys;
            return observations.back().rank_fraction;
        };
        for (std::size_t target_index = 0; target_index < target_columns.size();
             ++target_index) {
            const auto found = ranks_by_key.find(target_columns[target_index].key);
            if (found == ranks_by_key.end())
                continue;
            direct_prediction[target_index] = true;
            ++result.diagnostics.exact_semantic_hits;
            result.predicted_rank_fractions[target_index] =
                predict_from_observations(found->second);
        }
        if (result.diagnostics.observed_rank_spread_keys != 0) {
            result.diagnostics.mean_observed_rank_spread =
                observed_rank_spread_sum /
                static_cast<double>(
                    result.diagnostics.observed_rank_spread_keys);
        }

        // Zero-hit FIELD groups: tail structures (tau and quotient
        // remainders) are keyed relative to the coefficient count, so their
        // absolute indices never coincide across resolutions. Re-key such
        // groups per axis with top anchoring and accept the axis combination
        // that matches the most fine members; matched members predict
        // normally and seed the neighbor interpolation below.
        std::map<FieldGroupKey, std::vector<std::size_t>> field_group_members;
        for (std::size_t index = 0; index < target_columns.size(); ++index) {
            if (target_columns[index].key.is_field)
                field_group_members[field_group(target_columns[index])]
                    .push_back(index);
        }
        for (const auto& [group, members] : field_group_members) {
            const bool group_has_hit = std::any_of(
                members.begin(), members.end(),
                [&](std::size_t index) { return direct_prediction[index]; });
            if (group_has_hit)
                continue;
            struct CoarseGroupMember {
                ProlongationSemanticColumn column;
                RankObservation observation;
            };
            std::vector<CoarseGroupMember> coarse_members;
            for (const auto& observation : coarse_observations) {
                for (std::size_t index = 0; index < observation.columns.size();
                     ++index) {
                    const auto& column = observation.columns[index];
                    if (!column.key.is_field || field_group(column) != group)
                        continue;
                    coarse_members.push_back(
                        {column,
                         {observation.resolution,
                          rank_fraction(observation.permutation_1based[index],
                                        observation.columns.size())}});
                }
            }
            if (coarse_members.empty())
                continue;

            constexpr std::array<AnchorCombination, 7> anchor_combinations{{
                {true, false, false},
                {false, true, false},
                {false, false, true},
                {true, true, false},
                {true, false, true},
                {false, true, true},
                {true, true, true},
            }};
            std::size_t best_matched = 0;
            std::map<std::array<std::int32_t, 3>, std::vector<RankObservation>>
                best_coarse_ranks;
            AnchorCombination best_anchors;
            for (const auto& anchors : anchor_combinations) {
                std::map<std::array<std::int32_t, 3>,
                         std::vector<RankObservation>>
                    coarse_ranks;
                for (const auto& member : coarse_members) {
                    coarse_ranks[anchored_mode(member.column, anchors)]
                        .push_back(member.observation);
                }
                std::size_t matched = 0;
                for (std::size_t index : members) {
                    if (coarse_ranks.count(
                            anchored_mode(target_columns[index], anchors)) != 0)
                        ++matched;
                }
                if (matched > best_matched) {
                    best_matched = matched;
                    best_coarse_ranks = std::move(coarse_ranks);
                    best_anchors = anchors;
                }
            }
            if (best_matched == 0)
                continue;
            for (std::size_t index : members) {
                const auto found = best_coarse_ranks.find(
                    anchored_mode(target_columns[index], best_anchors));
                if (found == best_coarse_ranks.end())
                    continue;
                direct_prediction[index] = true;
                result.predicted_rank_fractions[index] =
                    predict_from_observations(found->second);
                ++result.diagnostics.anchor_recovered_columns;
            }
            ++result.diagnostics.anchor_recovered_groups;
        }

        // FIELD new modes use only exact-hit target neighbors, keeping the
        // result independent of target-column traversal order.
        std::map<FieldGroupKey, MissingGroupExemplar> missing_field_groups;
        for (std::size_t target_index = 0; target_index < target_columns.size();
             ++target_index) {
            if (direct_prediction[target_index] ||
                !target_columns[target_index].key.is_field)
                continue;
            const FieldGroupKey group = field_group(target_columns[target_index]);
            std::size_t best_index = target_columns.size();
            double best_distance = std::numeric_limits<double>::infinity();
            // Pass 0 stays inside the semantic group; pass 1 admits sibling
            // tensor components of the same variable and term; pass 2 admits
            // any hit of the same domain and column class, the last
            // structural neighborhood available for groups born at the
            // target resolution.
            int matched_pass = 0;
            for (int pass = 0; pass < 3 && best_index == target_columns.size();
                 ++pass) {
                matched_pass = pass;
                for (std::size_t candidate = 0;
                     candidate < target_columns.size(); ++candidate) {
                    if (!direct_prediction[candidate] ||
                        !target_columns[candidate].key.is_field)
                        continue;
                    const auto& candidate_key = target_columns[candidate].key;
                    if (pass == 0) {
                        if (field_group(target_columns[candidate]) != group)
                            continue;
                    } else if (pass == 1) {
                        if (component_widened_group(
                                field_group(target_columns[candidate])) !=
                            component_widened_group(group))
                            continue;
                    } else if (candidate_key.domain != group.domain ||
                               candidate_key.column_class !=
                                   group.column_class) {
                        continue;
                    }
                    const double distance = squared_mode_distance(
                        target_columns[target_index], target_columns[candidate]);
                    if (distance < best_distance ||
                        (distance == best_distance &&
                         best_index != target_columns.size() &&
                         candidate_key < target_columns[best_index].key)) {
                        best_distance = distance;
                        best_index = candidate;
                    }
                }
            }
            if (best_index == target_columns.size()) {
                auto& entry = missing_field_groups[group];
                if (entry.column_count == 0)
                    entry.exemplar = target_columns[target_index];
                ++entry.column_count;
                continue;
            }
            result.predicted_rank_fractions[target_index] =
                result.predicted_rank_fractions[best_index];
            ++result.diagnostics.interpolated_columns;
            if (matched_pass == 1)
                ++result.diagnostics.component_widened_columns;
            else if (matched_pass == 2)
                ++result.diagnostics.domain_widened_columns;
        }

        // Non-field columns occupy their coarse-observed group rank band in
        // ordinal order. This also assigns fine-only ordinals without allowing
        // them to cross another member of the same group.
        std::map<NonFieldGroupKey, std::vector<std::size_t>> nonfield_groups;
        for (std::size_t index = 0; index < target_columns.size(); ++index) {
            if (!target_columns[index].key.is_field)
                nonfield_groups[nonfield_group(target_columns[index])].push_back(index);
        }
        std::map<NonFieldGroupKey, MissingGroupExemplar> missing_nonfield_groups;
        for (auto& [group, indices] : nonfield_groups) {
            std::sort(indices.begin(), indices.end(), [&](std::size_t left,
                                                          std::size_t right) {
                return target_columns[left].key.ordinal_within_group <
                       target_columns[right].key.ordinal_within_group;
            });
            double band_min = std::numeric_limits<double>::infinity();
            double band_max = -std::numeric_limits<double>::infinity();
            for (std::size_t index : indices) {
                if (!direct_prediction[index])
                    continue;
                band_min = std::min(band_min,
                                    result.predicted_rank_fractions[index]);
                band_max = std::max(band_max,
                                    result.predicted_rank_fractions[index]);
            }
            if (!std::isfinite(band_min)) {
                auto& entry = missing_nonfield_groups[group];
                entry.exemplar = target_columns[indices.front()];
                entry.column_count = indices.size();
                continue;
            }
            for (std::size_t ordinal = 0; ordinal < indices.size(); ++ordinal) {
                const std::size_t index = indices[ordinal];
                if (!direct_prediction[index])
                    ++result.diagnostics.interpolated_columns;
                const double fraction = indices.size() <= 1
                                            ? 0.0
                                            : static_cast<double>(ordinal) /
                                                  static_cast<double>(indices.size() - 1);
                result.predicted_rank_fractions[index] =
                    band_min + fraction * (band_max - band_min);
            }
        }

        std::size_t missing_field_columns = 0;
        for (const auto& [group, entry] : missing_field_groups) {
            (void)group;
            missing_field_columns += entry.column_count;
        }
        if (!missing_nonfield_groups.empty() ||
            static_cast<double>(missing_field_columns) >
                options.maximum_defaulted_column_fraction *
                    static_cast<double>(target_columns.size())) {
            throw std::invalid_argument(describe_missing_groups(
                "no coarse hit for a new FIELD mode", missing_field_groups,
                "absent from every coarse archive", missing_nonfield_groups));
        }
        if (missing_field_columns != 0) {
            for (std::size_t target_index = 0;
                 target_index < target_columns.size(); ++target_index) {
                if (target_columns[target_index].key.is_field &&
                    !std::isfinite(
                        result.predicted_rank_fractions[target_index])) {
                    result.predicted_rank_fractions[target_index] = 1.0;
                    ++result.diagnostics.defaulted_tail_columns;
                }
            }
        }

        for (double prediction : result.predicted_rank_fractions) {
            if (!std::isfinite(prediction))
                throw std::logic_error(
                    "prolongation left a target column without a predicted rank");
        }

        std::vector<std::size_t> elimination_order(target_columns.size());
        std::iota(elimination_order.begin(), elimination_order.end(),
                  std::size_t{0});
        std::stable_sort(
            elimination_order.begin(), elimination_order.end(),
            [&](std::size_t left, std::size_t right) {
                const double left_rank = result.predicted_rank_fractions[left];
                const double right_rank = result.predicted_rank_fractions[right];
                if (left_rank != right_rank)
                    return left_rank < right_rank;
                if (target_columns[left].key != target_columns[right].key)
                    return target_columns[left].key < target_columns[right].key;
                return left < right;
            });
        result.permutation_1based.assign(target_columns.size(), 0);
        for (std::size_t position = 0; position < elimination_order.size();
             ++position) {
            result.permutation_1based[elimination_order[position]] =
                static_cast<int>(position + 1);
        }
        validate_prolonged_permutation(result.permutation_1based);
        return result;
    }

} // namespace Kadath
