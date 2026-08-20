#include "coloring_internals.hpp"

namespace Kadath::coloring_internals
{
    SupportSummary summarize_support(const std::vector<std::set<int>>& rows_per_column)
    {
        SupportSummary summary;
        if (rows_per_column.empty()) {
            return summary;
        }
        summary.min_rows = static_cast<int>(rows_per_column.front().size());
        long long total = 0;
        for (const auto& rows : rows_per_column) {
            const int count = static_cast<int>(rows.size());
            summary.min_rows = std::min(summary.min_rows, count);
            summary.max_rows = std::max(summary.max_rows, count);
            total += count;
        }
        summary.avg_rows = static_cast<double>(total) / static_cast<double>(rows_per_column.size());
        return summary;
    }

    void print_support_summary(std::ostream& os, const char* label, const SupportSummary& summary)
    {
        os << "  " << label << ": min=" << summary.min_rows << " avg=" << summary.avg_rows
           << " max=" << summary.max_rows << std::endl;
    }

    EligibilitySummary summarize_seeded_eligibility(const std::vector<std::vector<int>>& groups,
                                                    const std::vector<ColumnInfo>& column_map)
    {
        EligibilitySummary summary;
        for (const auto& group : groups) {
            if (group.empty()) {
                continue;
            }
            if (group.size() == 1) {
                ++summary.single_groups;
                ++summary.single_columns;
                continue;
            }

            bool has_var_domain = false;
            bool has_var_double = false;
            bool multi_domain = false;
            int first_domain = -1;
            for (int cc : group) {
                const auto& info = column_map[static_cast<std::size_t>(cc)];
                has_var_domain = has_var_domain || info.is_var_domain;
                has_var_double = has_var_double || (info.var_double_idx >= 0);
                if (info.domain >= 0) {
                    if (first_domain < 0) {
                        first_domain = info.domain;
                    } else if (info.domain != first_domain) {
                        multi_domain = true;
                    }
                }
            }

            const int columns = static_cast<int>(group.size());
            if (multi_domain) {
                ++summary.multi_domain_groups;
                summary.multi_domain_columns += columns;
            } else if (has_var_domain) {
                ++summary.var_domain_groups;
                summary.var_domain_columns += columns;
            } else if (has_var_double) {
                ++summary.var_double_groups;
                summary.var_double_columns += columns;
            } else {
                ++summary.eligible_groups;
                summary.eligible_columns += columns;
            }
        }
        return summary;
    }

    void print_eligibility_summary(std::ostream& os, const EligibilitySummary& summary)
    {
        os << "  Eligible multi-column groups: " << summary.eligible_groups << " groups / "
           << summary.eligible_columns << " columns" << std::endl;
        os << "  Single-column groups: " << summary.single_groups << " groups / " << summary.single_columns
           << " columns" << std::endl;
        os << "  Fallback multi-domain: " << summary.multi_domain_groups << " groups / "
           << summary.multi_domain_columns << " columns" << std::endl;
        os << "  Fallback variable-domain: " << summary.var_domain_groups << " groups / "
           << summary.var_domain_columns << " columns" << std::endl;
        os << "  Fallback var_double: " << summary.var_double_groups << " groups / "
           << summary.var_double_columns << " columns" << std::endl;
    }

    MetadataSupportSummary summarize_metadata_group_actual_support(const std::vector<std::vector<int>>& groups,
                                                                   const std::vector<std::set<int>>& rows_per_column,
                                                                   const std::vector<char>& selected_columns)
    {
        MetadataSupportSummary summary;
        for (const auto& group : groups) {
            std::vector<int> selected;
            bool truncated = false;
            for (int cc : group) {
                if (cc >= 0 && cc < static_cast<int>(selected_columns.size()) &&
                    selected_columns[static_cast<std::size_t>(cc)]) {
                    selected.push_back(cc);
                } else {
                    truncated = true;
                }
            }
            if (selected.empty()) {
                continue;
            }

            ++summary.groups;
            summary.columns += static_cast<int>(selected.size());
            if (truncated) {
                ++summary.truncated_groups;
            }
            if (selected.size() == 1) {
                ++summary.single_groups;
                ++summary.single_columns;
                continue;
            }

            std::set<int> seen_rows;
            int collisions = 0;
            for (int cc : selected) {
                for (int row : rows_per_column[static_cast<std::size_t>(cc)]) {
                    if (!seen_rows.insert(row).second) {
                        ++collisions;
                    }
                }
            }

            summary.total_row_collisions += collisions;
            summary.max_row_collisions_in_group =
                std::max(summary.max_row_collisions_in_group, collisions);
            if (collisions == 0) {
                ++summary.safe_multi_groups;
                summary.safe_multi_columns += static_cast<int>(selected.size());
            } else {
                ++summary.unsafe_multi_groups;
                summary.unsafe_multi_columns += static_cast<int>(selected.size());
            }
        }
        return summary;
    }

    void print_metadata_support_summary(std::ostream& os, const MetadataSupportSummary& summary)
    {
        os << "  Metadata groups checked against actual field rows:" << std::endl;
        os << "    Groups: " << summary.groups << " / columns: " << summary.columns << std::endl;
        os << "    Single-column groups: " << summary.single_groups << " groups / "
           << summary.single_columns << " columns" << std::endl;
        os << "    Seed-safe multi-column groups: " << summary.safe_multi_groups << " groups / "
           << summary.safe_multi_columns << " columns" << std::endl;
        os << "    Row-colliding multi-column groups: " << summary.unsafe_multi_groups << " groups / "
           << summary.unsafe_multi_columns << " columns" << std::endl;
        os << "    Row collision count: total=" << summary.total_row_collisions
           << " max_per_group=" << summary.max_row_collisions_in_group << std::endl;
        if (summary.truncated_groups > 0) {
            os << "    Truncated sampled groups: " << summary.truncated_groups << std::endl;
        }

        const int evals_if_safe_only =
            summary.single_columns + summary.safe_multi_groups + summary.unsafe_multi_columns;
        if (evals_if_safe_only > 0) {
            os << "    Metadata-safe-only speedup ceiling: "
               << static_cast<double>(summary.columns) / static_cast<double>(evals_if_safe_only)
               << "x" << std::endl;
        }
    }

    SparseColumnMap sparse_map_from_array(const Array<double>& column, double drop_tol)
    {
        SparseColumnMap out;
        const int n = static_cast<int>(column.get_nbr());
        for (int row = 0; row < n; ++row) {
            const double value = column(row);
            if (std::fabs(value) > drop_tol) {
                out.emplace(row, value);
            }
        }
        return out;
    }

    SparseColumnComparison compare_sparse_columns(const SparseColumnMap& reference,
                                                  const SparseColumnMap& decoded,
                                                  double rtol,
                                                  double atol)
    {
        SparseColumnComparison cmp;
        cmp.reference_nnz = static_cast<int>(reference.size());
        cmp.decoded_nnz = static_cast<int>(decoded.size());

        auto ref_it = reference.begin();
        auto dec_it = decoded.begin();
        while (ref_it != reference.end() || dec_it != decoded.end()) {
            int row = -1;
            double ref_value = 0.0;
            double dec_value = 0.0;
            bool has_ref = false;
            bool has_dec = false;

            if (dec_it == decoded.end() || (ref_it != reference.end() && ref_it->first < dec_it->first)) {
                row = ref_it->first;
                ref_value = ref_it->second;
                has_ref = true;
                ++ref_it;
            } else if (ref_it == reference.end() || dec_it->first < ref_it->first) {
                row = dec_it->first;
                dec_value = dec_it->second;
                has_dec = true;
                ++dec_it;
            } else {
                row = ref_it->first;
                ref_value = ref_it->second;
                dec_value = dec_it->second;
                has_ref = true;
                has_dec = true;
                ++ref_it;
                ++dec_it;
            }

            if (has_ref && !has_dec) {
                ++cmp.missing_rows;
            } else if (!has_ref && has_dec) {
                ++cmp.extra_rows;
            }

            const double abs_error = std::fabs(dec_value - ref_value);
            const double scale = std::max(std::fabs(ref_value), std::fabs(dec_value));
            const double rel_error = (scale > 0.0) ? abs_error / scale : 0.0;
            cmp.max_abs_error = std::max(cmp.max_abs_error, abs_error);
            cmp.max_relative_error = std::max(cmp.max_relative_error, rel_error);

            const double tolerance = atol + rtol * scale;
            if (abs_error > tolerance) {
                cmp.passed = false;
                if (cmp.first_mismatch_row < 0) {
                    cmp.first_mismatch_row = row;
                    cmp.first_reference_value = ref_value;
                    cmp.first_decoded_value = dec_value;
                }
            }
        }

        return cmp;
    }

    SparseCooComparison compare_sparse_coo(const SparseCooMap& reference,
                                           const SparseCooMap& seeded,
                                           double rtol,
                                           double atol)
    {
        SparseCooComparison cmp;
        cmp.reference_nnz = static_cast<int>(reference.size());
        cmp.seeded_nnz = static_cast<int>(seeded.size());

        auto ref_it = reference.begin();
        auto seed_it = seeded.begin();
        while (ref_it != reference.end() || seed_it != seeded.end()) {
            SparseCooKey key{-1, -1};
            double ref_value = 0.0;
            double seed_value = 0.0;
            bool has_ref = false;
            bool has_seed = false;

            if (seed_it == seeded.end() || (ref_it != reference.end() && ref_it->first < seed_it->first)) {
                key = ref_it->first;
                ref_value = ref_it->second;
                has_ref = true;
                ++ref_it;
            } else if (ref_it == reference.end() || seed_it->first < ref_it->first) {
                key = seed_it->first;
                seed_value = seed_it->second;
                has_seed = true;
                ++seed_it;
            } else {
                key = ref_it->first;
                ref_value = ref_it->second;
                seed_value = seed_it->second;
                has_ref = true;
                has_seed = true;
                ++ref_it;
                ++seed_it;
            }

            if (has_ref && !has_seed) {
                ++cmp.missing_entries;
            } else if (!has_ref && has_seed) {
                ++cmp.extra_entries;
            }

            const double abs_error = std::fabs(seed_value - ref_value);
            const double scale = std::max(std::fabs(ref_value), std::fabs(seed_value));
            const double rel_error = (scale > 0.0) ? abs_error / scale : 0.0;
            cmp.max_abs_error = std::max(cmp.max_abs_error, abs_error);
            cmp.max_relative_error = std::max(cmp.max_relative_error, rel_error);

            const double tolerance = atol + rtol * scale;
            if (abs_error > tolerance) {
                cmp.passed = false;
                if (cmp.first_mismatch_row < 0) {
                    cmp.first_mismatch_row = key.first;
                    cmp.first_mismatch_col = key.second;
                    cmp.first_reference_value = ref_value;
                    cmp.first_seeded_value = seed_value;
                }
            }
        }

        return cmp;
    }
} // namespace Kadath::coloring_internals
