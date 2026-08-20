// Seeded multi-column Jacobian assembly (production Newton hot path).
//
// Diagnostic / validation helpers that used to live here now reside in
// seeded_jacobian_diagnostics.cpp.
#include "coloring_internals.hpp"

#include <algorithm>
#include <cmath>

namespace Kadath
{
    using namespace coloring_internals;

    // =============================================================================
    // Seeded Jacobian computation (main optimization target)
    // =============================================================================

    /*
     * Compute multiple Jacobian columns simultaneously using seeded finite differences.
     *
     * All columns in cols_to_compute must have the same color (disjoint row sets for
     * field equations). This allows seeding all their derivatives simultaneously,
     * computing definitions ONCE, and extracting each column's rows separately.
     *
     * IMPORTANT: Integral equations are computed separately for each column since
     * they affect all columns (global constraints).
     *
     * Optimizations applied:
     *   1. Uses cached row incidence from coloring structure
     *   2. Pre-computed row ownership for fast distribution
     *   3. Skips unaffected equations based on seeded domains
     *   4. Static memory pools to avoid allocations
     */
    void System_of_eqs::do_cols_J_seeded(const std::vector<int>& cols_to_compute,
                                         std::vector<Array<double>>& result_columns)
    {
        if (cols_to_compute.empty())
            return;

        const size_t num_cols = cols_to_compute.size();
        result_columns.clear();
        result_columns.reserve(num_cols);

        const bool do_timing = solver_runtime_config.diagnostics.timing;
        auto& profile = seeded_jacobian_state.profile;
        auto& row_owner_pool = seeded_jacobian_state.row_owner_pool;
        auto t_total0 = std::chrono::steady_clock::now();

        // For single column, just use do_col_J directly
        if (num_cols == 1) {
            result_columns.push_back(do_col_J(cols_to_compute[0]));
            if (do_timing) {
                profile.single_calls += 1;
                profile.calls += 1;
                profile.cols += 1;
                profile.total_s +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_total0).count();
            }
            return;
        }

        // Get cached coloring data (Optimization 1)
        const auto& coloring = get_column_coloring();
        const auto& rows_per_column = coloring.rows_per_column;
        const auto& col_to_domain = coloring.col_to_domain;
        const auto& colmap = coloring.column_map;

        // Build row-to-column-index mapping for this group (Optimization 2 - fast path)
        // Use pooled memory to avoid allocation
        row_owner_pool.assign(static_cast<size_t>(nbr_conditions), -1);
        int* row_owner = row_owner_pool.data();

        // Collect seeded domains for equation skipping (Optimization 3)
        std::set<int> seeded_domains;

        for (size_t k = 0; k < num_cols; ++k) {
            int col = cols_to_compute[k];
            int dom = col_to_domain[static_cast<size_t>(col)];
            if (dom >= 0)
                seeded_domains.insert(dom);

            for (int row : rows_per_column[static_cast<size_t>(col)]) {
                // Only field equation rows (>= neq_int) are disjoint
                if (row >= neq_int && row < nbr_conditions) {
                    row_owner[row] = static_cast<int>(k);
                }
            }
        }

        // Initialize result arrays
        for (size_t k = 0; k < num_cols; ++k) {
            result_columns.push_back(Array<double>(nbr_conditions));
            result_columns.back() = 0.0;
        }

        // === Phase 1: Reset all derivatives ===
        for (int i = 0; i < nterm_cst; ++i)
            cst[i]->set_der_zero();
        for (int i = 0; i < nterm; ++i)
            term[i]->set_der_zero();
        for (int i = 0; i < nvar_double; ++i) {
            for (int dd = dom_min; dd <= dom_max; ++dd)
                term_double[i * ndom + (dd - dom_min)]->set_der_d(0.0);
        }

        // === Phase 2: Seed all columns in the group ===
        // Track if any column involves variable domains or scalar variables
        bool has_var_domain = false;
        bool has_var_double = false;
        bool multi_domain = false;

        // Check if columns span multiple domains (would cause Term_eq domain mismatch)
        int first_domain = -1;
        for (size_t k = 0; k < num_cols; ++k) {
            int cc = cols_to_compute[k];
            const ColumnInfo& info = colmap[static_cast<size_t>(cc)];

            if (info.is_var_domain) {
                has_var_domain = true;
            } else if (info.var_double_idx >= 0) {
                has_var_double = true;
            } else if (info.domain >= 0) {
                if (first_domain < 0) {
                    first_domain = info.domain;
                } else if (info.domain != first_domain) {
                    multi_domain = true;
                }
            }
        }

        // If columns span multiple domains, fall back to single-column computation
        // because Term_eq requires domain consistency when accumulating derivatives
        if (multi_domain) {
            for (size_t k = 0; k < num_cols; ++k) {
                result_columns[k] = do_col_J(cols_to_compute[k]);
            }
            if (do_timing) {
                profile.fallback_calls += 1;
                profile.calls += 1;
                profile.cols += num_cols;
                profile.total_s +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_total0).count();
            }
            return;
        }

        // Seed variable domains and var_doubles (these force fallback)
        for (size_t k = 0; k < num_cols; ++k) {
            int cc = cols_to_compute[k];
            int conte = 0;
            Array<int> zedoms(2);
            zedoms = -1;

            // Variable Domains - these modify domain boundaries
            espace.affecte_coef_to_variable_domains(conte, cc, zedoms);
            if (zedoms(0) != -1) {
                update_terms_from_variable_domains(zedoms);
                has_var_domain = true;
            }

            // Scalar variables (var_double)
            for (int i = 0; i < nvar_double; ++i) {
                if (conte == cc) {
                    for (int dd = dom_min; dd <= dom_max; ++dd)
                        term_double[i * ndom + (dd - dom_min)]->set_der_d(1.0);
                    has_var_double = true;
                }
                conte++;
            }

            // Field variables - accumulate derivatives from all columns
            for (int i = 0; i < nterm; ++i) {
                int dom = term[i]->get_dom();
                Tensor auxi(term[i]->get_val_t(), false);
                espace.get_domain(dom)->affecte_tau_one_coef(auxi, dom, cc, conte);

                for (int j = 0; j < auxi.get_n_comp(); ++j) {
                    auto idx = auxi.indices(j);
                    if (!auxi(idx)(dom).check_if_zero()) {
                        if (!auxi(idx)(dom).get_base().is_def())
                            auxi.set(idx).set_domain(dom).set_base() = term[i]->get_val_t()(idx)(dom).get_base();
                    }
                }
                // Accumulate this column's derivative seed
                term[i]->set_der_t(auxi + term[i]->get_der_t());
            }
        }

        auto t_seed_done = std::chrono::steady_clock::now();

        // If there are variable domains or var_double, the seeded approach may
        // give incorrect results due to global coupling. Fall back to single-column.
        if (has_var_domain || has_var_double) {
            // Reset and compute each column separately
            for (size_t k = 0; k < num_cols; ++k) {
                result_columns[k] = do_col_J(cols_to_compute[k]);
            }
            if (do_timing) {
                profile.fallback_calls += 1;
                profile.calls += 1;
                profile.cols += num_cols;
                profile.total_s +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_total0).count();
            }
            return;
        }

        // === Phase 3: Compute metric and definitions ONCE ===
        if (met != nullptr) {
            for (int d = dom_min; d <= dom_max; ++d)
                met->update(d);
        }
        for (int i = 0; i < ndef; ++i)
            def[i]->compute_res();
        auto t_def_done = std::chrono::steady_clock::now();

        // === Phase 4: Apply equations (Optimization 3 - skip unaffected) ===
        int conte = 0;
        for (int i = 0; i < neq; ++i) {
            // Check if this equation is affected by any seeded domain
            bool affects_seeded = seeded_domains.empty(); // If no domains, apply all
            if (!affects_seeded) {
                for (int dom : seeded_domains) {
                    if (eq[i]->take_into_account(dom)) {
                        affects_seeded = true;
                        break;
                    }
                }
            }

            if (affects_seeded) {
                Term_eq** results_raw = results_shadow_view();
                eq[i]->apply(conte, results_raw);
                results_shadow_sync();
            } else {
                conte += eq[i]->get_n_cond_tot();
            }
        }

        // === Phase 5: Export all field equation rows ===
        Array<double> full_result(nbr_conditions);
        full_result = 0.0;
        conte = 0;
        int pos_res = neq_int; // Start after integral equations

        Term_eq** results_raw = results_shadow_view();
        for (int i = 0; i < neq; ++i)
            eq[i]->export_der(conte, results_raw, full_result, pos_res);

        // === Phase 6: Distribute rows to respective column arrays ===
        // Since columns have disjoint row sets, each row was only affected by one column
        for (int row = neq_int; row < nbr_conditions; ++row) {
            int k = row_owner[row];
            if (k >= 0 && k < static_cast<int>(num_cols)) {
                result_columns[static_cast<size_t>(k)].set(row) = full_result(row);
            }
        }

        // === Phase 7: Integral equation rows (per-column computation) ===
        // FIXED: Actually compute integral equations for each column
        if (neq_int > 0) {
            for (size_t k = 0; k < num_cols; ++k) {
                const int cc = cols_to_compute[k];

                // Reset derivatives for this single column
                for (int i = 0; i < nterm_cst; ++i)
                    cst[i]->set_der_zero();
                for (int i = 0; i < nterm; ++i)
                    term[i]->set_der_zero();
                for (int i = 0; i < nvar_double; ++i) {
                    for (int dd = dom_min; dd <= dom_max; ++dd)
                        term_double[i * ndom + (dd - dom_min)]->set_der_d(0.0);
                }

                // Seed only this column
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
                    conte_seed++;
                }

                for (int i = 0; i < nterm; ++i) {
                    int dom = term[i]->get_dom();
                    Tensor auxi(term[i]->get_val_t(), false);
                    espace.get_domain(dom)->affecte_tau_one_coef(auxi, dom, cc, conte_seed);

                    for (int j = 0; j < auxi.get_n_comp(); ++j) {
                        auto idx = auxi.indices(j);
                        if (!auxi(idx)(dom).check_if_zero()) {
                            if (!auxi(idx)(dom).get_base().is_def())
                                auxi.set(idx).set_domain(dom).set_base() = term[i]->get_val_t()(idx)(dom).get_base();
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

                // Compute integral equation derivatives for this column
                for (int i = 0; i < neq_int; ++i) {
                    result_columns[k].set(i) = eq_int[i]->get_der();
                }
                for (int i = 0; i < ndef; ++i) {
                    def[i]->get_res()->set_der_zero();
                }
            }
        }

        if (do_timing) {
            auto t_done = std::chrono::steady_clock::now();
            profile.calls += 1;
            profile.cols += num_cols;
            profile.seeded_groups += 1;
            profile.total_s += std::chrono::duration<double>(t_done - t_total0).count();
            profile.seed_s += std::chrono::duration<double>(t_seed_done - t_total0).count();
            profile.def_s += std::chrono::duration<double>(t_def_done - t_seed_done).count();
            profile.apply_s += std::chrono::duration<double>(t_done - t_def_done).count();
        }
    }

    void System_of_eqs::dump_do_cols_J_seeded_profile() const
    {
        if (!solver_runtime_config.diagnostics.timing)
            return;
        const auto& profile = seeded_jacobian_state.profile;
        if (profile.calls == 0) {
            std::cout << "Seeded Jacobian profile: no calls recorded." << std::endl;
            return;
        }

        const double avg_group = profile.calls ? static_cast<double>(profile.cols) / profile.calls : 0.0;
        const double avg_col = profile.cols ? profile.total_s / profile.cols : 0.0;

        std::cout << "Seeded Jacobian profile:" << std::endl;
        std::cout << "  calls:          " << profile.calls << std::endl;
        std::cout << "  columns:        " << profile.cols << " (avg/group=" << avg_group << ")" << std::endl;
        std::cout << "  seeded groups:  " << profile.seeded_groups << std::endl;
        std::cout << "  single calls:   " << profile.single_calls << std::endl;
        std::cout << "  fallback calls: " << profile.fallback_calls << std::endl;
        std::cout << "  total time:     " << profile.total_s << " s" << std::endl;
        std::cout << "  avg/column:     " << avg_col << " s" << std::endl;
        std::cout << "  phase seed:     " << profile.seed_s << " s" << std::endl;
        std::cout << "  phase defs:     " << profile.def_s << " s" << std::endl;
        std::cout << "  phase apply:    " << profile.apply_s << " s" << std::endl;
    }
} // namespace Kadath
