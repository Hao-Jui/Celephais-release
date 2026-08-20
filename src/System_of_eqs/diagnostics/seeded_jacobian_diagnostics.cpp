// Diagnostic / verification utilities for the seeded-Jacobian path.
//
// Split out of seeded_jacobian.cpp: these routines are exercised only by the
// unit tests (tests/unit/test_seeded_jacobian.cpp) and by the diagnostic
// branches in src_par/jacobian_assembler.cpp. They cross-check the production
// do_cols_J_seeded() assembly against the per-column do_col_J_sparse() oracle.
// None of them is on the Newton hot path.
#include "coloring_internals.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>

namespace Kadath
{
    using namespace coloring_internals;

    namespace
    {
        struct SelectedFallbackBucket {
            int color = -1;
            std::vector<int> columns;
            std::vector<int> domains;
        };

        std::string join_ints(const std::vector<int>& values)
        {
            std::ostringstream os;
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (i > 0) {
                    os << ",";
                }
                os << values[i];
            }
            return os.str();
        }

        std::uint64_t sparse_double_bits(double value)
        {
            std::uint64_t bits = 0;
            static_assert(sizeof(bits) == sizeof(value));
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        }

        struct SparseByteEntry {
            int row = -1;
            std::uint64_t value_bits = 0;
        };

        bool sparse_entries_match(const std::vector<SparseByteEntry>& reference,
                                  const std::vector<SparseByteEntry>& grouped,
                                  std::size_t& mismatch_index)
        {
            const std::size_t common = std::min(reference.size(), grouped.size());
            for (std::size_t index = 0; index < common; ++index) {
                if (reference[index].row != grouped[index].row ||
                    reference[index].value_bits != grouped[index].value_bits) {
                    mismatch_index = index;
                    return false;
                }
            }
            if (reference.size() != grouped.size()) {
                mismatch_index = common;
                return false;
            }
            return true;
        }
    } // namespace

    bool System_of_eqs::validate_fallback_coloring_bucket(std::ostream& os,
                                                          int max_columns,
                                                          int max_group_size,
                                                          int max_buckets,
                                                          double drop_tol,
                                                          double rtol,
                                                          double atol)
    {
        if (nbr_conditions < 0) {
            os << kColoringNeedsSecMember << std::endl;
            return false;
        }
        if (nbr_unknowns == 0) {
            os << "Fallback Coloring Bucket Validation:" << std::endl;
            os << "  pass: false" << std::endl;
            os << "  reason: system has no unknowns" << std::endl;
            return false;
        }

        const auto& metadata = get_column_coloring();
        const int columns_to_analyze =
            (max_columns > 0) ? std::min(nbr_unknowns, max_columns) : nbr_unknowns;
        const int group_cap = std::max(2, max_group_size);
        const int bucket_cap = std::max(1, max_buckets);

        std::vector<std::set<int>> field_rows(static_cast<std::size_t>(columns_to_analyze));
        reset_do_col_J_cache();
        try {
            for (int cc = 0; cc < columns_to_analyze; ++cc) {
                do_col_J_sparse(cc, drop_tol, [&](int row, double) {
                    if (row >= neq_int) {
                        field_rows[static_cast<std::size_t>(cc)].insert(row);
                    }
                });
            }
        } catch (...) {
            reset_do_col_J_cache();
            throw;
        }
        reset_do_col_J_cache();

        std::vector<SelectedFallbackBucket> selected_buckets;
        for (std::size_t color = 0; color < metadata.groups.size(); ++color) {
            if (static_cast<int>(selected_buckets.size()) >= bucket_cap) {
                break;
            }
            const auto& group = metadata.groups[color];
            if (group.size() < 2) {
                continue;
            }

            std::vector<int> candidates;
            for (int cc : group) {
                if (cc < 0 || cc >= columns_to_analyze) {
                    continue;
                }
                const auto& info = metadata.column_map[static_cast<std::size_t>(cc)];
                if (info.is_var_domain || info.var_double_idx >= 0 || info.domain < 0) {
                    continue;
                }
                candidates.push_back(cc);
            }
            if (candidates.size() < 2) {
                continue;
            }
            std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
                const auto rows_a = field_rows[static_cast<std::size_t>(a)].size();
                const auto rows_b = field_rows[static_cast<std::size_t>(b)].size();
                if (rows_a != rows_b) {
                    return rows_a > rows_b;
                }
                return a < b;
            });

            const int first_col = candidates.front();
            const int first_domain = metadata.column_map[static_cast<std::size_t>(first_col)].domain;
            int second_col = -1;
            for (int cc : candidates) {
                if (metadata.column_map[static_cast<std::size_t>(cc)].domain != first_domain) {
                    second_col = cc;
                    break;
                }
            }
            if (second_col < 0) {
                continue;
            }

            SelectedFallbackBucket bucket;
            bucket.color = static_cast<int>(color);
            bucket.columns.push_back(first_col);
            bucket.columns.push_back(second_col);
            bucket.domains.push_back(first_domain);
            bucket.domains.push_back(metadata.column_map[static_cast<std::size_t>(second_col)].domain);

            for (int cc : candidates) {
                if (static_cast<int>(bucket.columns.size()) >= group_cap) {
                    break;
                }
                if (cc == first_col || cc == second_col) {
                    continue;
                }
                const auto& info = metadata.column_map[static_cast<std::size_t>(cc)];
                bucket.columns.push_back(cc);
                bucket.domains.push_back(info.domain);
            }
            selected_buckets.push_back(std::move(bucket));
        }

        os << "Fallback Coloring Bucket Validation:" << std::endl;
        os << "  selection_source: production_metadata_groups" << std::endl;
        os << "  requested_fallback_reason: multi-domain" << std::endl;
        os << "  total_columns: " << nbr_unknowns << std::endl;
        os << "  analyzed_columns: " << columns_to_analyze;
        if (columns_to_analyze < nbr_unknowns) {
            os << " (sampled prefix)";
        }
        os << std::endl;
        os << "  production_metadata_groups: " << metadata.groups.size() << std::endl;
        os << "  requested_buckets: " << bucket_cap << std::endl;
        os << "  drop_tolerance: " << drop_tol << std::endl;
        os << "  rtol: " << rtol << std::endl;
        os << "  atol: " << atol << std::endl;
        os << "  seeded_mumps_input: retired" << std::endl;

        if (selected_buckets.empty()) {
            os << "  selected_color: none" << std::endl;
            os << "  selected_columns: none" << std::endl;
            os << "  fallback_reason: multi-domain" << std::endl;
            os << "  forced_seeded_attempts: 0" << std::endl;
            os << "  diagnostic_fallback_count: 0" << std::endl;
            os << "  guard_bypassed_for_diagnostic: false" << std::endl;
            os << "  pass: false" << std::endl;
            os << "  reason: no production metadata multi-domain fallback bucket found in analyzed columns" << std::endl;
            return false;
        }

        bool ok = true;
        int total_forced_seeded_attempts = 0;
        int total_diagnostic_fallback_count = 0;
        int passed_buckets = 0;
        for (std::size_t bucket_idx = 0; bucket_idx < selected_buckets.size(); ++bucket_idx) {
            const auto& bucket = selected_buckets[bucket_idx];
            os << "  bucket " << bucket_idx << ":" << std::endl;
            os << "    selected_color: " << bucket.color << std::endl;
            os << "    selected_columns: " << join_ints(bucket.columns) << std::endl;
            os << "    selected_domains: " << join_ints(bucket.domains) << std::endl;
            os << "    fallback_reason: multi-domain" << std::endl;
            os << "    production_metadata_group_size: " << metadata.groups[static_cast<std::size_t>(bucket.color)].size()
               << std::endl;

            std::vector<SparseColumnMap> reference_maps;
            reference_maps.reserve(bucket.columns.size());
            reset_do_col_J_cache();
            try {
                for (int cc : bucket.columns) {
                    SparseColumnMap rows;
                    do_col_J_sparse(cc, drop_tol, [&](int row, double value) {
                        rows[row] = value;
                    });
                    reference_maps.push_back(std::move(rows));
                }
            } catch (...) {
                reset_do_col_J_cache();
                throw;
            }
            reset_do_col_J_cache();

            bool guard_bypassed_for_diagnostic = false;
            bool diagnostic_completed = false;
            std::string failure_reason;
            std::vector<Array<double>> decoded_columns;
            std::vector<std::set<int>> selected_field_rows;
            selected_field_rows.reserve(bucket.columns.size());
            for (int cc : bucket.columns) {
                selected_field_rows.push_back(field_rows[static_cast<std::size_t>(cc)]);
            }

            ++total_forced_seeded_attempts;
            try {
                diagnostic_completed = do_cols_J_seeded_diagnostic_force(bucket.columns,
                                                                         selected_field_rows,
                                                                         decoded_columns,
                                                                         guard_bypassed_for_diagnostic,
                                                                         failure_reason);
            } catch (const std::exception& e) {
                failure_reason = e.what();
                diagnostic_completed = false;
            } catch (...) {
                failure_reason = "unknown exception during diagnostic seeded attempt";
                diagnostic_completed = false;
            }
            reset_do_col_J_cache();

            os << "    forced_seeded_attempts: 1" << std::endl;
            os << "    diagnostic_fallback_count: 0" << std::endl;
            os << "    guard_bypassed_for_diagnostic: "
               << (guard_bypassed_for_diagnostic ? "true" : "false") << std::endl;
            os << "    diagnostic_state_isolation: reset_do_col_J_cache before/after support and reference collection;"
               << " diagnostic derivative seeds zeroed after forced attempt" << std::endl;

            bool bucket_ok = diagnostic_completed && guard_bypassed_for_diagnostic &&
                             decoded_columns.size() == bucket.columns.size();
            if (!diagnostic_completed) {
                os << "    reason: " << (failure_reason.empty() ? "diagnostic seeded attempt failed" : failure_reason)
                   << std::endl;
            } else if (!guard_bypassed_for_diagnostic) {
                os << "    reason: multi-domain guard was not bypassed in the isolated diagnostic path" << std::endl;
            } else if (decoded_columns.size() != bucket.columns.size()) {
                os << "    reason: decoded column count does not match selected bucket size" << std::endl;
            }

            if (bucket_ok) {
                for (std::size_t k = 0; k < bucket.columns.size(); ++k) {
                    const SparseColumnMap decoded = sparse_map_from_array(decoded_columns[k], drop_tol);
                    const SparseColumnComparison cmp = compare_sparse_columns(reference_maps[k], decoded, rtol, atol);
                    bucket_ok = bucket_ok && cmp.passed;

                    os << "    column " << bucket.columns[k] << ":"
                       << " reference_nnz=" << cmp.reference_nnz
                       << " decoded_nnz=" << cmp.decoded_nnz
                       << " missing_rows=" << cmp.missing_rows
                       << " extra_rows=" << cmp.extra_rows
                       << " max_abs_error=" << cmp.max_abs_error
                       << " max_relative_error=" << cmp.max_relative_error
                       << " pass=" << (cmp.passed ? "true" : "false");
                    if (!cmp.passed) {
                        os << " first_mismatch_row=" << cmp.first_mismatch_row
                           << " reference_value=" << cmp.first_reference_value
                           << " decoded_value=" << cmp.first_decoded_value;
                    }
                    os << std::endl;
                }
            }

            if (bucket_ok) {
                ++passed_buckets;
            }
            os << "    pass: " << (bucket_ok ? "true" : "false") << std::endl;
            ok = ok && bucket_ok;
        }

        os << "  selected_buckets: " << selected_buckets.size() << std::endl;
        os << "  passed_buckets: " << passed_buckets << std::endl;
        os << "  forced_seeded_attempts: " << total_forced_seeded_attempts << std::endl;
        os << "  diagnostic_fallback_count: " << total_diagnostic_fallback_count << std::endl;
        os << "  pass: " << (ok ? "true" : "false") << std::endl;
        if (ok) {
            os << "  conclusion: selected production metadata multi-domain fallback buckets passed; this does not enable production seeded assembly."
               << std::endl;
        } else {
            os << "  conclusion: at least one selected production metadata bucket did not match individual sparse-column references."
               << std::endl;
        }
        return ok;
    }

    namespace
    {
        struct SeededCooSelectionSkipCounters {
            int singleton_or_small_group = 0;
            int no_in_analysis_columns = 0;
            int no_eligible_fixed_domain_candidates = 0;
            int fewer_than_two_candidates = 0;
            int no_second_domain = 0;
            int validation_cap = 0;
            int groups_with_out_of_analysis_exclusions = 0;
            int out_of_analysis_columns = 0;

            int primary_skip_total() const
            {
                return singleton_or_small_group + no_in_analysis_columns +
                       no_eligible_fixed_domain_candidates + fewer_than_two_candidates +
                       no_second_domain + validation_cap;
            }
        };
    } // namespace

    bool System_of_eqs::validate_seeded_coo_equivalence(std::ostream& os,
                                                        int max_columns,
                                                        int max_group_size,
                                                        int max_groups,
                                                        double drop_tol,
                                                        double rtol,
                                                        double atol)
    {
        if (nbr_conditions < 0) {
            os << kColoringNeedsSecMember << std::endl;
            return false;
        }
        if (nbr_unknowns == 0) {
            os << "Seeded COO Equivalence Validation:" << std::endl;
            os << "  pass: false" << std::endl;
            os << "  reason: system has no unknowns" << std::endl;
            return false;
        }

        const auto& metadata = get_column_coloring();
        const int columns_to_analyze =
            (max_columns > 0) ? std::min(nbr_unknowns, max_columns) : nbr_unknowns;
        const int group_cap = std::max(2, max_group_size);
        const int validation_cap = std::max(1, max_groups);

        std::vector<SelectedFallbackBucket> selected_groups;
        SeededCooSelectionSkipCounters skip_counters;
        for (std::size_t color = 0; color < metadata.groups.size(); ++color) {
            if (static_cast<int>(selected_groups.size()) >= validation_cap) {
                skip_counters.validation_cap =
                    static_cast<int>(metadata.groups.size() - color);
                break;
            }
            const auto& group = metadata.groups[color];
            if (group.size() < 2) {
                ++skip_counters.singleton_or_small_group;
                continue;
            }

            std::vector<int> candidates;
            int in_analysis_columns = 0;
            bool has_out_of_analysis_column = false;
            for (int cc : group) {
                if (cc < 0 || cc >= columns_to_analyze) {
                    has_out_of_analysis_column = true;
                    ++skip_counters.out_of_analysis_columns;
                    continue;
                }
                ++in_analysis_columns;
                const auto& info = metadata.column_map[static_cast<std::size_t>(cc)];
                if (info.is_var_domain || info.var_double_idx >= 0 || info.domain < 0) {
                    continue;
                }
                candidates.push_back(cc);
            }
            if (has_out_of_analysis_column) {
                ++skip_counters.groups_with_out_of_analysis_exclusions;
            }
            if (in_analysis_columns == 0) {
                ++skip_counters.no_in_analysis_columns;
                continue;
            }
            if (candidates.empty()) {
                ++skip_counters.no_eligible_fixed_domain_candidates;
                continue;
            }
            if (candidates.size() < 2) {
                ++skip_counters.fewer_than_two_candidates;
                continue;
            }

            const int first_col = candidates.front();
            const int first_domain = metadata.column_map[static_cast<std::size_t>(first_col)].domain;
            int second_col = -1;
            for (int cc : candidates) {
                if (metadata.column_map[static_cast<std::size_t>(cc)].domain != first_domain) {
                    second_col = cc;
                    break;
                }
            }
            if (second_col < 0) {
                ++skip_counters.no_second_domain;
                continue;
            }

            SelectedFallbackBucket bucket;
            bucket.color = static_cast<int>(color);
            bucket.columns.push_back(first_col);
            bucket.columns.push_back(second_col);
            bucket.domains.push_back(first_domain);
            bucket.domains.push_back(metadata.column_map[static_cast<std::size_t>(second_col)].domain);

            for (int cc : candidates) {
                if (static_cast<int>(bucket.columns.size()) >= group_cap) {
                    break;
                }
                if (cc == first_col || cc == second_col) {
                    continue;
                }
                const auto& info = metadata.column_map[static_cast<std::size_t>(cc)];
                bucket.columns.push_back(cc);
                bucket.domains.push_back(info.domain);
            }
            selected_groups.push_back(std::move(bucket));
        }

        os << "Seeded COO Equivalence Validation:" << std::endl;
        os << "  selection_source: production_metadata_groups" << std::endl;
        os << "  total_columns: " << nbr_unknowns << std::endl;
        os << "  analyzed_columns: " << columns_to_analyze;
        if (columns_to_analyze < nbr_unknowns) {
            os << " (sampled prefix)";
        }
        os << std::endl;
        os << "  production_metadata_groups: " << metadata.groups.size() << std::endl;
        os << "  requested_groups: " << validation_cap << std::endl;
        os << "  max_group_size: " << group_cap << std::endl;
        os << "  selection_skip_groups_total: " << skip_counters.primary_skip_total() << std::endl;
        os << "  selection_skip_singleton_or_small_group: "
           << skip_counters.singleton_or_small_group << std::endl;
        os << "  selection_skip_no_in_analysis_columns: "
           << skip_counters.no_in_analysis_columns << std::endl;
        os << "  selection_skip_no_eligible_fixed_domain_candidates: "
           << skip_counters.no_eligible_fixed_domain_candidates << std::endl;
        os << "  selection_skip_fewer_than_two_candidates: "
           << skip_counters.fewer_than_two_candidates << std::endl;
        os << "  selection_skip_no_second_domain: "
           << skip_counters.no_second_domain << std::endl;
        os << "  selection_skip_validation_cap: " << skip_counters.validation_cap << std::endl;
        os << "  selection_groups_with_out_of_analysis_exclusions: "
           << skip_counters.groups_with_out_of_analysis_exclusions << std::endl;
        os << "  selection_out_of_analysis_columns: "
           << skip_counters.out_of_analysis_columns << std::endl;
        os << "  drop_tolerance: " << drop_tol << std::endl;
        os << "  rtol: " << rtol << std::endl;
        os << "  atol: " << atol << std::endl;
        os << "  seeded_mumps_input: retired" << std::endl;
        os << "  support_collection: selected_groups_only" << std::endl;

        if (selected_groups.empty()) {
            os << "  selected_groups: 0" << std::endl;
            os << "  pass: false" << std::endl;
            os << "  reason: no production metadata multi-domain group selected for COO equivalence" << std::endl;
            return false;
        }

        SparseCooMap reference_coo;
        SparseCooMap seeded_coo;
        int passed_groups = 0;
        int forced_seeded_attempts = 0;
        bool ok = true;

        for (std::size_t group_idx = 0; group_idx < selected_groups.size(); ++group_idx) {
            const auto& group = selected_groups[group_idx];

            reset_do_col_J_cache();
            std::vector<std::set<int>> selected_field_rows(group.columns.size());
            try {
                for (std::size_t k = 0; k < group.columns.size(); ++k) {
                    const int cc = group.columns[k];
                    do_col_J_sparse(cc, drop_tol, [&](int row, double value) {
                        if (row >= neq_int) {
                            selected_field_rows[k].insert(row);
                        }
                        reference_coo[{row, cc}] = value;
                    });
                }
            } catch (...) {
                reset_do_col_J_cache();
                throw;
            }
            reset_do_col_J_cache();

            bool guard_bypassed_for_diagnostic = false;
            std::string failure_reason;
            std::vector<Array<double>> decoded_columns;

            ++forced_seeded_attempts;
            bool group_ok = false;
            try {
                group_ok = do_cols_J_seeded_diagnostic_force(group.columns,
                                                             selected_field_rows,
                                                             decoded_columns,
                                                             guard_bypassed_for_diagnostic,
                                                             failure_reason);
            } catch (const std::exception& e) {
                failure_reason = e.what();
                group_ok = false;
            } catch (...) {
                failure_reason = "unknown exception during diagnostic seeded COO attempt";
                group_ok = false;
            }
            reset_do_col_J_cache();

            group_ok = group_ok && guard_bypassed_for_diagnostic &&
                       decoded_columns.size() == group.columns.size();
            if (group_ok) {
                for (std::size_t k = 0; k < group.columns.size(); ++k) {
                    const int cc = group.columns[k];
                    for (const auto& [row, value] : sparse_map_from_array(decoded_columns[k], drop_tol)) {
                        seeded_coo[{row, cc}] = value;
                    }
                }
                ++passed_groups;
            } else if (ok) {
                os << "  first_failed_group: " << group_idx << std::endl;
                os << "  first_failed_color: " << group.color << std::endl;
                os << "  first_failed_columns: " << join_ints(group.columns) << std::endl;
                os << "  first_failed_reason: "
                   << (failure_reason.empty() ? "diagnostic seeded COO attempt failed" : failure_reason)
                   << std::endl;
            }
            ok = ok && group_ok;
        }

        const SparseCooComparison cmp = compare_sparse_coo(reference_coo, seeded_coo, rtol, atol);
        ok = ok && cmp.passed;

        os << "  selected_groups: " << selected_groups.size() << std::endl;
        os << "  passed_groups: " << passed_groups << std::endl;
        os << "  forced_seeded_attempts: " << forced_seeded_attempts << std::endl;
        os << "  reference_nnz: " << cmp.reference_nnz << std::endl;
        os << "  seeded_nnz: " << cmp.seeded_nnz << std::endl;
        os << "  missing_entries: " << cmp.missing_entries << std::endl;
        os << "  extra_entries: " << cmp.extra_entries << std::endl;
        os << "  max_abs_error: " << cmp.max_abs_error << std::endl;
        os << "  max_relative_error: " << cmp.max_relative_error << std::endl;
        if (!cmp.passed) {
            os << "  first_mismatch_row: " << cmp.first_mismatch_row << std::endl;
            os << "  first_mismatch_col: " << cmp.first_mismatch_col << std::endl;
            os << "  first_reference_value: " << cmp.first_reference_value << std::endl;
            os << "  first_seeded_value: " << cmp.first_seeded_value << std::endl;
        }
        os << "  pass: " << (ok ? "true" : "false") << std::endl;
        if (ok) {
            os << "  conclusion: selected metadata seeded COO entries match standard sparse COO; seeded MUMPS input is retired."
               << std::endl;
        } else {
            os << "  conclusion: selected metadata seeded COO entries differ from standard sparse COO."
               << std::endl;
        }
        return ok;
    }

    bool System_of_eqs::do_cols_J_seeded_diagnostic_force(const std::vector<int>& cols_to_compute,
                                                          const std::vector<std::set<int>>& rows_per_selected_column,
                                                          std::vector<Array<double>>& result_columns,
                                                          bool& guard_bypassed_for_diagnostic,
                                                          std::string& failure_reason)
    {
        guard_bypassed_for_diagnostic = false;
        failure_reason.clear();
        result_columns.clear();
        if (cols_to_compute.size() < 2) {
            failure_reason = "diagnostic bucket must contain at least two columns";
            return false;
        }
        if (rows_per_selected_column.size() != cols_to_compute.size()) {
            failure_reason = "diagnostic row-owner support does not match selected column count";
            return false;
        }

        auto zero_seed_derivatives = [&]() {
            for (int i = 0; i < nterm_cst; ++i)
                cst[i]->set_der_zero();
            for (int i = 0; i < nterm; ++i)
                term[i]->set_der_zero();
            for (int i = 0; i < nvar_double; ++i) {
                for (int dd = dom_min; dd <= dom_max; ++dd)
                    term_double[i * ndom + (dd - dom_min)]->set_der_d(0.0);
            }
        };

        try {
            const std::size_t num_cols = cols_to_compute.size();
            const auto& coloring = get_column_coloring();
            const auto& col_to_domain = coloring.col_to_domain;
            const auto& colmap = coloring.column_map;

            std::vector<int> row_owner(static_cast<std::size_t>(nbr_conditions), -1);
            std::set<int> seeded_domains;
            for (std::size_t k = 0; k < num_cols; ++k) {
                const int col = cols_to_compute[k];
                if (col < 0 || col >= nbr_unknowns) {
                    failure_reason = "selected column is outside the system unknown range";
                    return false;
                }
                const int dom = col_to_domain[static_cast<std::size_t>(col)];
                if (dom >= 0) {
                    seeded_domains.insert(dom);
                }
                for (int row : rows_per_selected_column[k]) {
                    if (row >= neq_int && row < nbr_conditions) {
                        if (row_owner[static_cast<std::size_t>(row)] >= 0) {
                            failure_reason = "selected actual-support bucket has overlapping field rows";
                            return false;
                        }
                        row_owner[static_cast<std::size_t>(row)] = static_cast<int>(k);
                    }
                }
            }

            bool has_var_domain = false;
            bool has_var_double = false;
            bool multi_domain = false;
            int first_domain = -1;
            for (int cc : cols_to_compute) {
                const ColumnInfo& info = colmap[static_cast<std::size_t>(cc)];
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

            if (has_var_domain || has_var_double) {
                failure_reason = "selected bucket mixes multi-domain with variable-domain or var_double columns";
                return false;
            }
            guard_bypassed_for_diagnostic = multi_domain;

            for (std::size_t k = 0; k < num_cols; ++k) {
                result_columns.push_back(Array<double>(nbr_conditions));
                result_columns.back() = 0.0;
            }

            zero_seed_derivatives();

            for (int cc : cols_to_compute) {
                int conte = 0;
                Array<int> zedoms(2);
                zedoms = -1;

                espace.affecte_coef_to_variable_domains(conte, cc, zedoms);
                if (zedoms(0) != -1) {
                    zero_seed_derivatives();
                    failure_reason = "diagnostic selected a variable-domain column after metadata filtering";
                    return false;
                }

                for (int i = 0; i < nvar_double; ++i) {
                    if (conte == cc) {
                        zero_seed_derivatives();
                        failure_reason = "diagnostic selected a var_double column after metadata filtering";
                        return false;
                    }
                    ++conte;
                }

                for (int i = 0; i < nterm; ++i) {
                    const int dom = term[i]->get_dom();
                    Tensor auxi(term[i]->get_val_t(), false);
                    espace.get_domain(dom)->affecte_tau_one_coef(auxi, dom, cc, conte);

                    for (int j = 0; j < auxi.get_n_comp(); ++j) {
                        auto idx = auxi.indices(j);
                        if (!auxi(idx)(dom).check_if_zero()) {
                            if (!auxi(idx)(dom).get_base().is_def())
                                auxi.set(idx).set_domain(dom).set_base() = term[i]->get_val_t()(idx)(dom).get_base();
                        }
                    }
                    term[i]->set_der_t(auxi + term[i]->get_der_t());
                }
            }

            if (met != nullptr) {
                for (int d = dom_min; d <= dom_max; ++d)
                    met->update(d);
            }
            for (int i = 0; i < ndef; ++i)
                def[i]->compute_res();

            std::vector<unsigned char> eq_affects(static_cast<std::size_t>(neq), 0);
            int conte = 0;
            for (int i = 0; i < neq; ++i) {
                bool affects_seeded = seeded_domains.empty();
                if (!affects_seeded) {
                    for (int dom : seeded_domains) {
                        if (eq[i]->take_into_account(dom)) {
                            affects_seeded = true;
                            break;
                        }
                    }
                }

                if (affects_seeded) {
                    eq_affects[static_cast<std::size_t>(i)] = 1;
                    Term_eq** results_raw = results_shadow_view();
                    eq[i]->apply(conte, results_raw);
                    results_shadow_sync();
                } else {
                    conte += eq[i]->n_ope;
                }
            }

            Array<double> full_result(nbr_conditions);
            full_result = 0.0;
            conte = 0;
            int pos_res = neq_int;

            Term_eq** results_raw = results_shadow_view();
            for (int i = 0; i < neq; ++i) {
                if (eq_affects[static_cast<std::size_t>(i)]) {
                    eq[i]->export_der(conte, results_raw, full_result, pos_res);
                } else {
                    pos_res += eq[i]->n_cond_tot;
                    conte += eq[i]->n_ope;
                }
            }

            for (int row = neq_int; row < nbr_conditions; ++row) {
                const int k = row_owner[static_cast<std::size_t>(row)];
                if (k >= 0 && k < static_cast<int>(num_cols)) {
                    result_columns[static_cast<std::size_t>(k)].set(row) = full_result(row);
                }
            }

            if (neq_int > 0) {
                for (std::size_t k = 0; k < num_cols; ++k) {
                    const int cc = cols_to_compute[k];
                    zero_seed_derivatives();

                    int conte_seed = 0;
                    Array<int> zedoms(2);
                    zedoms = -1;

                    espace.affecte_coef_to_variable_domains(conte_seed, cc, zedoms);
                    if (zedoms(0) != -1) {
                        update_terms_from_variable_domains(zedoms);
                    }

                    for (int i = 0; i < nvar_double; ++i) {
                        if (conte_seed == cc) {
                            for (int dd = dom_min; dd <= dom_max; ++dd)
                                term_double[i * ndom + (dd - dom_min)]->set_der_d(1.0);
                        }
                        ++conte_seed;
                    }

                    for (int i = 0; i < nterm; ++i) {
                        const int dom = term[i]->get_dom();
                        Tensor auxi(term[i]->get_val_t(), false);
                        espace.get_domain(dom)->affecte_tau_one_coef(auxi, dom, cc, conte_seed);

                        for (int j = 0; j < auxi.get_n_comp(); ++j) {
                            auto idx = auxi.indices(j);
                            if (!auxi(idx)(dom).check_if_zero()) {
                                if (!auxi(idx)(dom).get_base().is_def())
                                    auxi.set(idx).set_domain(dom).set_base() =
                                        term[i]->get_val_t()(idx)(dom).get_base();
                            }
                        }
                        term[i]->set_der_t(auxi);
                    }

                    if (met != nullptr) {
                        for (int d = dom_min; d <= dom_max; ++d) {
                            met->update(d);
                        }
                    }
                    for (int i = 0; i < ndef; ++i) {
                        def[i]->compute_res();
                    }
                    int conte_apply = 0;
                    Term_eq** integral_results = results_shadow_view();
                    for (int i = 0; i < neq; ++i) {
                        eq[i]->apply(conte_apply, integral_results);
                    }
                    results_shadow_sync();

                    for (int i = 0; i < neq_int; ++i) {
                        result_columns[k].set(i) = eq_int[i]->get_der();
                    }
                    for (int i = 0; i < ndef; ++i) {
                        def[i]->get_res()->set_der_zero();
                    }
                }
            }

            zero_seed_derivatives();
            return true;
        } catch (...) {
            zero_seed_derivatives();
            throw;
        }
    }

    bool System_of_eqs::validate_row_disjoint_seeded_group(
        const std::vector<int>& cols_to_compute,
        const std::vector<std::set<int>>& rows_per_selected_column,
        double drop_tol,
        std::ostream& report)
    {
        report << "RowDisjointSeededOracle:" << std::endl;
        report << "  columns=" << join_ints(cols_to_compute) << std::endl;
        report << "  drop_tol=" << std::setprecision(17) << drop_tol << std::endl;

        bool guard_bypassed_for_diagnostic = false;
        std::string failure_reason;
        std::vector<Array<double>> grouped_columns;
        if (!do_cols_J_seeded_diagnostic_force(cols_to_compute,
                                               rows_per_selected_column,
                                               grouped_columns,
                                               guard_bypassed_for_diagnostic,
                                               failure_reason)) {
            report << "  pass: false" << std::endl;
            report << "  reason: grouped seeded diagnostic setup failed: "
                   << failure_reason << std::endl;
            return false;
        }
        report << "  multi_domain_guard_bypassed: "
               << (guard_bypassed_for_diagnostic ? "true" : "false") << std::endl;

        if (grouped_columns.size() != cols_to_compute.size()) {
            report << "  pass: false" << std::endl;
            report << "  reason: grouped result column count mismatch" << std::endl;
            return false;
        }

        std::vector<SparseByteEntry> reference_entries;
        std::vector<SparseByteEntry> grouped_entries;
        for (std::size_t column_index = 0; column_index < cols_to_compute.size(); ++column_index) {
            reference_entries.clear();
            grouped_entries.clear();
            const int column = cols_to_compute[column_index];
            do_col_J_sparse(column, drop_tol, [&](int row, double value) {
                reference_entries.push_back(SparseByteEntry{row, sparse_double_bits(value)});
            });

            const Array<double>& grouped_column = grouped_columns[column_index];
            for (int row = 0; row < nbr_conditions; ++row) {
                const double value = grouped_column(row);
                if (std::abs(value) > drop_tol) {
                    grouped_entries.push_back(SparseByteEntry{row, sparse_double_bits(value)});
                }
            }

            std::size_t mismatch_index = 0;
            if (!sparse_entries_match(reference_entries, grouped_entries, mismatch_index)) {
                report << "  pass: false" << std::endl;
                report << "  failed_column=" << column << std::endl;
                report << "  ref_nnz=" << reference_entries.size() << std::endl;
                report << "  grouped_nnz=" << grouped_entries.size() << std::endl;
                report << "  mismatch_index=" << mismatch_index << std::endl;
                if (mismatch_index < reference_entries.size()) {
                    report << "  ref_row=" << reference_entries[mismatch_index].row
                           << " ref_bits=" << reference_entries[mismatch_index].value_bits
                           << std::endl;
                }
                if (mismatch_index < grouped_entries.size()) {
                    report << "  grouped_row=" << grouped_entries[mismatch_index].row
                           << " grouped_bits=" << grouped_entries[mismatch_index].value_bits
                           << std::endl;
                }
                return false;
            }
        }

        report << "  pass: true" << std::endl;
        return true;
    }
} // namespace Kadath
