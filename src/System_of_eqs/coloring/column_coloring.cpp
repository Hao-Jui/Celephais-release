#include "coloring_internals.hpp"

namespace Kadath
{
    using namespace coloring_internals;

    namespace
    {
        void color_from_row_support(int ncols, const std::vector<std::set<int>>& rows_per_column,
                                    std::vector<int>& color, std::vector<std::vector<int>>& groups)
        {
            if (ncols == 0) {
                color.clear();
                groups.clear();
                return;
            }

            std::vector<int> order(static_cast<std::size_t>(ncols));
            std::iota(order.begin(), order.end(), 0);

            std::map<int, int> row_index;
            std::vector<std::vector<int>> row_indices_per_column(static_cast<std::size_t>(ncols));
            for (int cc = 0; cc < ncols; ++cc) {
                auto& row_indices = row_indices_per_column[static_cast<std::size_t>(cc)];
                row_indices.reserve(rows_per_column[static_cast<std::size_t>(cc)].size());
                for (int row : rows_per_column[static_cast<std::size_t>(cc)]) {
                    auto [it, inserted] = row_index.emplace(row, static_cast<int>(row_index.size()));
                    row_indices.push_back(it->second);
                }
            }

            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return row_indices_per_column[static_cast<std::size_t>(a)].size() >
                       row_indices_per_column[static_cast<std::size_t>(b)].size();
            });

            color.assign(static_cast<std::size_t>(ncols), -1);
            std::vector<std::vector<int>> colors_by_row(row_index.size());
            std::vector<int> color_stamp(1, 0);
            int stamp = 0;
            int max_color = -1;
            for (int v : order) {
                if (static_cast<int>(color_stamp.size()) < max_color + 2) {
                    color_stamp.resize(static_cast<std::size_t>(max_color + 2), 0);
                }
                ++stamp;
                for (int row : row_indices_per_column[static_cast<std::size_t>(v)]) {
                    for (int c : colors_by_row[static_cast<std::size_t>(row)]) {
                        color_stamp[static_cast<std::size_t>(c)] = stamp;
                    }
                }
                int c = 0;
                while (c <= max_color && color_stamp[static_cast<std::size_t>(c)] == stamp)
                    ++c;
                color[static_cast<std::size_t>(v)] = c;
                if (c > max_color)
                    max_color = c;
                for (int row : row_indices_per_column[static_cast<std::size_t>(v)]) {
                    auto& row_colors = colors_by_row[static_cast<std::size_t>(row)];
                    if (std::find(row_colors.begin(), row_colors.end(), c) == row_colors.end()) {
                        row_colors.push_back(c);
                    }
                }
            }

            groups.clear();
            groups.resize(static_cast<std::size_t>(max_color + 1));
            for (int cc = 0; cc < ncols; ++cc) {
                groups[static_cast<std::size_t>(color[static_cast<std::size_t>(cc)])].push_back(cc);
            }
        }

    } // namespace

    void System_of_eqs::compute_column_coloring(ColumnColoring& coloring) const
    {
        if (nbr_conditions < 0) {
            KADATH_THROW(kColoringNeedsSecMember);
        }

        // Set system_ptr early so dump_column_coloring_stats can verify
        coloring.system_ptr = static_cast<const void*>(this);

        build_column_map(coloring.column_map);

        if (nbr_unknowns == 0) {
            coloring.num_colors = 0;
            coloring.initialized = true;
            return;
        }

        // Build row incidence (cached in coloring structure - Optimization 1)
        build_column_row_incidence(coloring.column_map, coloring.rows_per_column);

        // Build col_to_domain mapping (Optimization 4 prep)
        coloring.col_to_domain.resize(static_cast<size_t>(nbr_unknowns));
        for (int cc = 0; cc < nbr_unknowns; ++cc) {
            coloring.col_to_domain[cc] = coloring.column_map[cc].domain;
        }

        color_from_row_support(nbr_unknowns, coloring.rows_per_column, coloring.color, coloring.groups);
        coloring.num_colors = static_cast<int>(coloring.groups.size());

        coloring.initialized = true;
    }

    /*
     * Get the current column coloring (computes if not cached)
     */
    const ColumnColoring& System_of_eqs::get_column_coloring()
    {
        if (nbr_conditions < 0) {
            KADATH_THROW(kColoringNeedsSecMember);
        }

        auto& coloring_cache = seeded_jacobian_state.coloring_cache;
        if (!coloring_cache.initialized || coloring_cache.system_ptr != static_cast<const void*>(this) ||
            coloring_cache.column_map.size() != static_cast<size_t>(nbr_unknowns)) {
            compute_column_coloring(coloring_cache);
        }
        return coloring_cache;
    }

    /*
     * Reset column coloring cache
     */
    void System_of_eqs::reset_column_coloring()
    {
        auto& coloring_cache = seeded_jacobian_state.coloring_cache;
        coloring_cache.initialized = false;
        coloring_cache.system_ptr = nullptr;
        coloring_cache.column_map.clear();
        coloring_cache.color.clear();
        coloring_cache.groups.clear();
        coloring_cache.rows_per_column.clear();
        coloring_cache.col_to_domain.clear();
        coloring_cache.num_colors = 0;
    }

    /*
     * Print coloring statistics
     */
    void System_of_eqs::dump_column_coloring_stats(std::ostream& os) const
    {
        if (nbr_conditions < 0) {
            os << kColoringNeedsSecMember << std::endl;
            return;
        }

        const auto& coloring_cache = seeded_jacobian_state.coloring_cache;
        if (!coloring_cache.initialized || coloring_cache.system_ptr != static_cast<const void*>(this)) {
            os << "Column coloring not yet computed for this system." << std::endl;
            return;
        }

        os << "Column Coloring Statistics (excluding " << neq_int << " integral equations):" << std::endl;
        os << "  Total columns: " << nbr_unknowns << std::endl;
        os << "  Number of colors: " << coloring_cache.num_colors << std::endl;

        if (coloring_cache.num_colors > 0) {
            os << "  Potential speedup: " << static_cast<double>(nbr_unknowns) / coloring_cache.num_colors
               << "x (evaluations reduction)" << std::endl;

            // Show group size distribution
            std::vector<size_t> sizes;
            for (const auto& g : coloring_cache.groups) {
                sizes.push_back(g.size());
            }
            std::sort(sizes.begin(), sizes.end(), std::greater<size_t>());

            os << "  Largest groups: ";
            for (size_t i = 0; i < std::min(size_t(5), sizes.size()); ++i) {
                if (i > 0)
                    os << ", ";
                os << sizes[i];
            }
            if (sizes.size() > 5)
                os << ", ...";
            os << std::endl;
        }
    }

    void System_of_eqs::dump_column_coloring_analysis(std::ostream& os, int max_columns, double drop_tol)
    {
        if (nbr_conditions < 0) {
            os << kColoringNeedsSecMember << std::endl;
            return;
        }

        const auto& metadata = get_column_coloring();
        std::vector<int> selected_columns;
        selected_columns.reserve(static_cast<std::size_t>(nbr_unknowns));
        if (max_columns > 0 && max_columns < nbr_unknowns) {
            for (const auto& group : metadata.groups) {
                const int group_size = static_cast<int>(group.size());
                if (group_size == 0 || group_size > max_columns) {
                    continue;
                }
                if (static_cast<int>(selected_columns.size()) + group_size > max_columns) {
                    continue;
                }
                selected_columns.insert(selected_columns.end(), group.begin(), group.end());
                if (static_cast<int>(selected_columns.size()) == max_columns) {
                    break;
                }
            }
            if (selected_columns.empty()) {
                for (int cc = 0; cc < std::min(nbr_unknowns, max_columns); ++cc) {
                    selected_columns.push_back(cc);
                }
            }
            std::sort(selected_columns.begin(), selected_columns.end());
        } else {
            for (int cc = 0; cc < nbr_unknowns; ++cc) {
                selected_columns.push_back(cc);
            }
        }
        const int columns_to_analyze = static_cast<int>(selected_columns.size());
        const bool sampled = (columns_to_analyze < nbr_unknowns);

        std::vector<char> selected_column_mask(static_cast<std::size_t>(nbr_unknowns), 0);
        for (int cc : selected_columns) {
            selected_column_mask[static_cast<std::size_t>(cc)] = 1;
        }

        std::vector<std::set<int>> full_rows_by_column(static_cast<std::size_t>(nbr_unknowns));
        std::vector<std::set<int>> field_rows_by_column(static_cast<std::size_t>(nbr_unknowns));

        reset_do_col_J_cache();
        try {
            for (int cc : selected_columns) {
                do_col_J_sparse(cc, drop_tol, [&](int row, double) {
                    full_rows_by_column[static_cast<std::size_t>(cc)].insert(row);
                    if (row >= neq_int) {
                        field_rows_by_column[static_cast<std::size_t>(cc)].insert(row);
                    }
                });
            }
        } catch (...) {
            reset_do_col_J_cache();
            throw;
        }
        reset_do_col_J_cache();

        std::vector<std::set<int>> full_rows;
        std::vector<std::set<int>> field_rows;
        full_rows.reserve(static_cast<std::size_t>(columns_to_analyze));
        field_rows.reserve(static_cast<std::size_t>(columns_to_analyze));
        for (int cc : selected_columns) {
            full_rows.push_back(full_rows_by_column[static_cast<std::size_t>(cc)]);
            field_rows.push_back(field_rows_by_column[static_cast<std::size_t>(cc)]);
        }

        std::vector<int> actual_color;
        std::vector<std::vector<int>> actual_groups;
        color_from_row_support(columns_to_analyze, field_rows, actual_color, actual_groups);
        std::vector<std::vector<int>> actual_global_groups;
        actual_global_groups.reserve(actual_groups.size());
        for (const auto& group : actual_groups) {
            std::vector<int> global_group;
            global_group.reserve(group.size());
            for (int local_col : group) {
                global_group.push_back(selected_columns[static_cast<std::size_t>(local_col)]);
            }
            actual_global_groups.push_back(global_group);
        }

        os << "Column Coloring Actual-Support Analysis:" << std::endl;
        os << "  Total columns: " << nbr_unknowns << std::endl;
        os << "  Analyzed columns: " << columns_to_analyze;
        if (sampled) {
            os << " (sampled complete metadata groups; full-system color counts are not definitive)";
        }
        os << std::endl;
        os << "  Drop tolerance: " << drop_tol << std::endl;
        os << "  Row universe: full sparse rows include " << neq_int
           << " integral rows; seeded-eligible field rows exclude them." << std::endl;
        os << "  Metadata field-row colors: " << metadata.num_colors << std::endl;
        os << "  Actual field-support colors: " << actual_groups.size();
        if (sampled) {
            os << " (sampled)";
        }
        os << std::endl;
        if (!actual_groups.empty()) {
            os << "  Actual field-support potential speedup: "
               << static_cast<double>(columns_to_analyze) / static_cast<double>(actual_groups.size()) << "x";
            if (sampled) {
                os << " (sampled, not a full-system speedup)";
            }
            os << std::endl;
        }

        print_support_summary(os, "Full sparse row support", summarize_support(full_rows));
        print_support_summary(os, "Seeded-eligible field row support", summarize_support(field_rows));

        print_metadata_support_summary(os,
                                       summarize_metadata_group_actual_support(metadata.groups,
                                                                               field_rows_by_column,
                                                                               selected_column_mask));

        os << "  Seeded eligibility from actual field-support groups:" << std::endl;
        print_eligibility_summary(os, summarize_seeded_eligibility(actual_global_groups, metadata.column_map));
        os << "  Cache isolation: reset_do_col_J_cache before and after actual-support collection." << std::endl;
        os << "  Validation: not run by this diagnostic; use validate_column_coloring(...) separately." << std::endl;
    }


    bool System_of_eqs::validate_column_coloring(int max_checks, double rtol, double atol, std::ostream* os)
    {
        if (nbr_conditions < 0) {
            if (os) {
                *os << kColoringNeedsSecMember << std::endl;
            }
            return false;
        }
        if (nbr_unknowns == 0)
            return true;

        // DEBUG: First test if do_col_J() is deterministic
        if (os) {
            *os << "DEBUG: Testing do_col_J() determinism..." << std::endl;
            const int probe_col = std::min(100, nbr_unknowns - 1);
            Array<double> test1 = do_col_J(0);
            Array<double> test2 = do_col_J(0);
            double max_diff = 0.0;
            int max_idx = -1;
            for (std::size_t i = 0; i < test1.get_nbr(); ++i) {
                double d = std::fabs(test1(static_cast<int>(i)) - test2(static_cast<int>(i)));
                if (d > max_diff) {
                    max_diff = d;
                    max_idx = static_cast<int>(i);
                }
            }
            *os << "DEBUG: do_col_J(0) called twice: max_diff=" << max_diff << " at row " << max_idx << std::endl;

            // Also test with an intervening different column
            Array<double> test3 = do_col_J(0);
            Array<double> test4 = do_col_J(probe_col);
            Array<double> test5 = do_col_J(0);
            (void)test4;
            max_diff = 0.0;
            max_idx = -1;
            for (std::size_t i = 0; i < test3.get_nbr(); ++i) {
                double d = std::fabs(test3(static_cast<int>(i)) - test5(static_cast<int>(i)));
                if (d > max_diff) {
                    max_diff = d;
                    max_idx = static_cast<int>(i);
                }
            }
            *os << "DEBUG: do_col_J(0), do_col_J(100), do_col_J(0): max_diff=" << max_diff << " at row " << max_idx
                << std::endl;
        }

        const auto& coloring = get_column_coloring();
        int checked = 0;
        bool ok = true;

        for (const auto& group : coloring.groups) {
            if (group.empty())
                continue;
            if (max_checks > 0 && checked >= max_checks)
                break;

            // Compute using seeded method
            std::vector<Array<double>> seeded_cols;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
            do_cols_J_seeded(group, seeded_cols);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

            // Compare against single-column method
            for (size_t k = 0; k < group.size() && (max_checks < 0 || checked < max_checks); ++k) {
                Array<double> ref = do_col_J(group[k]);
                const double* a = seeded_cols[k].get_data();
                const double* b = ref.get_data();
                int n = static_cast<int>(ref.get_nbr());

                double max_abs = 0.0;
                double max_ref = 0.0;
                int max_idx = -1;
                for (int i = 0; i < n; ++i) {
                    double diff = std::fabs(a[i] - b[i]);
                    if (diff > max_abs) {
                        max_abs = diff;
                        max_idx = i;
                    }
                    if (std::fabs(b[i]) > max_ref)
                        max_ref = std::fabs(b[i]);
                }

                double tol = atol + rtol * max_ref;
                if (max_abs > tol) {
                    ok = false;
                    if (os) {
                        *os << "Validation FAILED for col " << group[k] << ": max_diff=" << max_abs << " at row "
                            << max_idx << " (tol=" << tol << ")" << std::endl;
                    }
                }
                ++checked;
            }
        }

        if (os) {
            if (ok)
                *os << "Validation PASSED for " << checked << " columns." << std::endl;
            else
                *os << "Validation FAILED." << std::endl;
        }
        return ok;
    }
} // namespace Kadath
