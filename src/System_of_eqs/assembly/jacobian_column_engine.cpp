// JacobianColumnEngine owns the scalar and packed sparse-Jacobian column paths.
// Keep experimental assembly schemes out of this file until they pass byte-level
// COO parity against the scalar/W-lane baselines.

#include "For_Kadath/System_of_eqs/jacobian_column_engine.hpp"

#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"
#include "mpi.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <set>
#include <unordered_map>

namespace Kadath {


namespace {

bool active_equation_range_verify_enabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("JACOBIAN_ACTIVE_EQ_VERIFY");
        if (value == nullptr)
            return false;
        return std::strcmp(value, "1") == 0 ||
               std::strcmp(value, "true") == 0 ||
               std::strcmp(value, "TRUE") == 0 ||
               std::strcmp(value, "on") == 0 ||
               std::strcmp(value, "ON") == 0;
    }();
    return enabled;
}

bool def_compute_per_def_enabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("DEF_COMPUTE_PER_DEF");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

// Diagnostic-only: classify an Equation pointer to its concrete subclass name.
// Used by the W8 export-by-kind census print. Subclass list is closed (see
// system_of_eqs.hpp) so dynamic_cast probing is bounded.
const char* equation_kind_name(const Equation* eq)
{
    if (eq == nullptr)                                              return "<null>";
    if (dynamic_cast<const Eq_matching_non_std*>(eq) != nullptr)    return "Eq_matching_non_std";
    if (dynamic_cast<const Eq_matching_one_side*>(eq) != nullptr)   return "Eq_matching_one_side";
    if (dynamic_cast<const Eq_matching_import*>(eq) != nullptr)     return "Eq_matching_import";
    if (dynamic_cast<const Eq_matching_exception*>(eq) != nullptr)  return "Eq_matching_exception";
    if (dynamic_cast<const Eq_matching_order_array*>(eq) != nullptr) return "Eq_matching_order_array";
    if (dynamic_cast<const Eq_matching*>(eq) != nullptr)            return "Eq_matching";
    if (dynamic_cast<const Eq_bc_exception*>(eq) != nullptr)        return "Eq_bc_exception";
    if (dynamic_cast<const Eq_bc_order_array*>(eq) != nullptr)      return "Eq_bc_order_array";
    if (dynamic_cast<const Eq_bc*>(eq) != nullptr)                  return "Eq_bc";
    if (dynamic_cast<const Eq_order_array*>(eq) != nullptr)         return "Eq_order_array";
    if (dynamic_cast<const Eq_order*>(eq) != nullptr)               return "Eq_order";
    if (dynamic_cast<const Eq_first_integral*>(eq) != nullptr)      return "Eq_first_integral";
    if (dynamic_cast<const Eq_one_side*>(eq) != nullptr)            return "Eq_one_side";
    if (dynamic_cast<const Eq_vel_pot*>(eq) != nullptr)             return "Eq_vel_pot";
    if (dynamic_cast<const Eq_inside*>(eq) != nullptr)              return "Eq_inside";
    if (dynamic_cast<const Eq_full*>(eq) != nullptr)                return "Eq_full";
    return "<unknown Equation subclass>";
}

void set_primary_zero_and_retain_extra_derivative_lanes(Term_eq* term,
                                                        int lane_count)
{
    if (term == nullptr)
        return;
    term->reset_derivative_tile_retain_storage(lane_count);
}

void set_primary_zero_and_retain_extra_derivative_lanes(const std::unique_ptr<Term_eq>& term,
                                                        int lane_count)
{
    set_primary_zero_and_retain_extra_derivative_lanes(term.get(), lane_count);
}

void set_primary_zero_and_release_extra_derivative_lanes(Term_eq* term)
{
    if (term == nullptr)
        return;
    term->reset_derivative_tile(1);
}

void set_primary_zero_and_release_extra_derivative_lanes(const std::unique_ptr<Term_eq>& term)
{
    set_primary_zero_and_release_extra_derivative_lanes(term.get());
}

template <typename ActiveRanges>
void clear_active_output_rows(Array<double>& output,
                              int integral_row_count,
                              int total_row_count,
                              const ActiveRanges& active_ranges)
{
#ifndef NDEBUG
    assert(integral_row_count >= 0 && integral_row_count <= total_row_count);
    int previous_row_end = integral_row_count;
    for (const auto& active_range : active_ranges) {
        assert(active_range.row_begin >= integral_row_count);
        assert(active_range.row_begin >= previous_row_end);
        assert(active_range.row_begin <= active_range.row_end);
        assert(active_range.row_end <= total_row_count);
        previous_row_end = active_range.row_end;
    }
#endif

    double* const data = output.get_data();
    int clear_begin = 0;
    int clear_end = integral_row_count;
    for (const auto& active_range : active_ranges) {
        if (clear_end == active_range.row_begin) {
            clear_end = active_range.row_end;
            continue;
        }
        if (clear_begin < clear_end)
            std::fill(data + clear_begin, data + clear_end, 0.0);
        clear_begin = active_range.row_begin;
        clear_end = active_range.row_end;
    }
    if (clear_begin < clear_end)
        std::fill(data + clear_begin, data + clear_end, 0.0);
}

} // namespace

void JacobianColumnEngine::DoColJTimingState::reset_totals()
{
    update_terms = 0.0;
    var_double = 0.0;
    fields = 0.0;
    eq_affects = 0.0;
    apply = 0.0;
    export_der = 0.0;
    calls = 0;
    var_domain_calls = 0;
    var_domain_total = 0.0;
    metric_update = 0.0;
    def_update = 0.0;
    def_clear = 0.0;
    def_compute = 0.0;
    sparse_scan_rows = 0;
    sparse_full_scan_rows = 0;
    equation_domain_active = 0;
    equation_variable_skipped = 0;
    equation_active = 0;
    packed_wlane2_calls = 0;
    packed_wlane2_total = 0.0;
    packed_wlane2_prepare = 0.0;
    packed_wlane2_seed = 0.0;
    packed_wlane2_metric_update = 0.0;
    packed_wlane2_def_update = 0.0;
    packed_wlane2_def_clear = 0.0;
    packed_wlane2_def_compute = 0.0;
    packed_wlane2_equation_filter = 0.0;
    packed_wlane2_apply = 0.0;
    packed_wlane2_export = 0.0;
    packed_wlane2_scan = 0.0;
    packed_wlane2_restore = 0.0;
    packed_wlane4_calls = 0;
    packed_wlane4_total = 0.0;
    packed_wlane4_prepare = 0.0;
    packed_wlane4_seed = 0.0;
    packed_wlane4_metric_update = 0.0;
    packed_wlane4_def_update = 0.0;
    packed_wlane4_def_clear = 0.0;
    packed_wlane4_def_compute = 0.0;
    packed_wlane4_equation_filter = 0.0;
    packed_wlane4_apply = 0.0;
    packed_wlane4_export = 0.0;
    packed_wlane4_scan = 0.0;
    packed_wlane4_restore = 0.0;
    packed_wlane8_calls = 0;
    packed_wlane8_total = 0.0;
    packed_wlane8_prepare = 0.0;
    packed_wlane8_seed = 0.0;
    packed_wlane8_metric_update = 0.0;
    packed_wlane8_def_update = 0.0;
    packed_wlane8_def_clear = 0.0;
    packed_wlane8_def_compute = 0.0;
    packed_wlane8_equation_filter = 0.0;
    packed_wlane8_apply = 0.0;
    packed_wlane8_export = 0.0;
    packed_wlane8_export_clear = 0.0;
    packed_wlane8_export_integral = 0.0;
    packed_wlane8_export_equations = 0.0;
    packed_wlane8_scan = 0.0;
    packed_wlane8_restore = 0.0;
    std::fill(def_eq.begin(), def_eq.end(), 0.0);
    std::fill(def_calls.begin(), def_calls.end(), 0);
    std::fill(apply_eq.begin(), apply_eq.end(), 0.0);
    std::fill(apply_calls.begin(), apply_calls.end(), 0);
    std::fill(packed_wlane8_export_eq.begin(), packed_wlane8_export_eq.end(), 0.0);
    std::fill(packed_wlane8_export_calls.begin(), packed_wlane8_export_calls.end(), 0);
    packed_wlane16_calls = 0;
    packed_wlane16_total = 0.0;
    packed_wlane16_prepare = 0.0;
    packed_wlane16_seed = 0.0;
    packed_wlane16_metric_update = 0.0;
    packed_wlane16_def_update = 0.0;
    packed_wlane16_def_clear = 0.0;
    packed_wlane16_def_compute = 0.0;
    packed_wlane16_equation_filter = 0.0;
    packed_wlane16_apply = 0.0;
    packed_wlane16_export = 0.0;
    packed_wlane16_export_clear = 0.0;
    packed_wlane16_export_integral = 0.0;
    packed_wlane16_export_equations = 0.0;
    packed_wlane16_scan = 0.0;
    packed_wlane16_restore = 0.0;
    std::fill(packed_wlane16_export_eq.begin(), packed_wlane16_export_eq.end(), 0.0);
    std::fill(packed_wlane16_export_calls.begin(), packed_wlane16_export_calls.end(), 0);
    packed_wlane32_calls = 0;
    packed_wlane32_total = 0.0;
    packed_wlane32_prepare = 0.0;
    packed_wlane32_seed = 0.0;
    packed_wlane32_metric_update = 0.0;
    packed_wlane32_def_update = 0.0;
    packed_wlane32_def_clear = 0.0;
    packed_wlane32_def_compute = 0.0;
    packed_wlane32_equation_filter = 0.0;
    packed_wlane32_apply = 0.0;
    packed_wlane32_export = 0.0;
    packed_wlane32_export_clear = 0.0;
    packed_wlane32_export_integral = 0.0;
    packed_wlane32_export_equations = 0.0;
    packed_wlane32_scan = 0.0;
    packed_wlane32_restore = 0.0;
    std::fill(packed_wlane32_export_eq.begin(), packed_wlane32_export_eq.end(), 0.0);
    std::fill(packed_wlane32_export_calls.begin(), packed_wlane32_export_calls.end(), 0);
}

JacobianColumnEngine::JacobianColumnEngine(System_of_eqs& sys) : sys_(sys) {}

void JacobianColumnEngine::reset_cache()
{
    reset_cache(default_workspace_);
}

void JacobianColumnEngine::reset_cache(Workspace& workspace)
{
    workspace.state_.def_dep_cache.nvar_domain = -1;
    workspace.state_.def_dep_cache.prev_was_var_domain = false;
    workspace.state_.def_dep_cache.tau_seed_descriptors_ready = false;
    workspace.state_.def_dep_cache.col_tau_seed.clear();
    ope_der_cache_probe_reset();
    reset_val_domain_der_abs_cache();
}

void JacobianColumnEngine::release_assembly_scratch()
{
    release_assembly_scratch(default_workspace_);
}

void JacobianColumnEngine::release_assembly_scratch(Workspace& workspace)
{
    // Tile storage is retained only between assembly groups. Drop it at this
    // phase boundary so dormant lane Tensors do not overlap MUMPS' peak factor
    // workspace. Keep this traversal in lockstep with packed-lane preparation.
    for (int i = 0; i < sys_.nterm_cst; ++i)
        set_primary_zero_and_release_extra_derivative_lanes(sys_.cst[i]);
    for (int i = 0; i < sys_.nterm; ++i)
        set_primary_zero_and_release_extra_derivative_lanes(sys_.term[i]);
    for (int i = 0; i < sys_.nterm_double; ++i)
        set_primary_zero_and_release_extra_derivative_lanes(sys_.term_double[i]);
    for (auto& result : sys_.results)
        set_primary_zero_and_release_extra_derivative_lanes(result);
    for (int i = 0; i < sys_.ndef; ++i)
        set_primary_zero_and_release_extra_derivative_lanes(sys_.def[i]->get_res());

    auto& state = workspace.state_;
    state.column_buffer.reset();
    state.column_buffer_size = 0;
    state.packed_first_buffer.reset();
    state.packed_second_buffer.reset();
    state.packed_buffer_size = 0;
    for (auto& buffer : state.packed_extra_buffers)
        buffer.reset();
    state.packed_extra_buffer_size = 0;
}

void JacobianColumnEngine::refresh_definition_filter_caches(Workspace& workspace)
{
    auto& def_domain_cache = workspace.state_.def_domain_cache;
    auto& def_dep_cache = workspace.state_.def_dep_cache;

    if (def_domain_cache.dom_min != sys_.dom_min || def_domain_cache.dom_max != sys_.dom_max ||
        def_domain_cache.ndef != sys_.ndef) {
        def_domain_cache.dom_min = sys_.dom_min;
        def_domain_cache.dom_max = sys_.dom_max;
        def_domain_cache.ndef = sys_.ndef;
        const int ndoms = (sys_.dom_max >= sys_.dom_min) ? (sys_.dom_max - sys_.dom_min + 1) : 0;
        def_domain_cache.defs_by_dom.assign(static_cast<std::size_t>(ndoms), {});
        def_domain_cache.defs_global.clear();
        for (int i = 0; i < sys_.ndef; ++i) {
            const int d = sys_.def[i]->get_dom();
            if (d < sys_.dom_min || d > sys_.dom_max) {
                def_domain_cache.defs_global.push_back(i);
            } else {
                def_domain_cache.defs_by_dom[static_cast<std::size_t>(d - sys_.dom_min)].push_back(i);
            }
        }
    }

    if (def_dep_cache.ndef != sys_.ndef ||
        def_dep_cache.nbr_unknowns != sys_.nbr_unknowns ||
        def_dep_cache.nterm != sys_.nterm ||
        def_dep_cache.nvar != sys_.nvar ||
        def_dep_cache.nvar_double != sys_.nvar_double ||
        def_dep_cache.neq != sys_.neq) {
        def_dep_cache.ndef = sys_.ndef;
        def_dep_cache.nbr_unknowns = sys_.nbr_unknowns;
        def_dep_cache.nterm = sys_.nterm;
        def_dep_cache.nvar = sys_.nvar;
        def_dep_cache.nvar_double = sys_.nvar_double;
        def_dep_cache.neq = sys_.neq;
        def_dep_cache.base_defs_ready = false;
        def_dep_cache.tau_seed_descriptors_ready = false;
        def_dep_cache.nvar_domain = -1;
        def_dep_cache.prev_was_var_domain = false;
        def_dep_cache.def_dep_ids.assign(static_cast<std::size_t>(sys_.ndef), {});
        def_dep_cache.equation_dep_ids.assign(static_cast<std::size_t>(sys_.neq), {});
        def_dep_cache.col_var_id.assign(static_cast<std::size_t>(sys_.nbr_unknowns), -1);
        def_dep_cache.col_term_idx.assign(static_cast<std::size_t>(sys_.nbr_unknowns), -1);
        def_dep_cache.term_start_col.assign(static_cast<std::size_t>(sys_.nterm), -1);
        def_dep_cache.col_is_var_domain.assign(static_cast<std::size_t>(sys_.nbr_unknowns), 0);
        def_dep_cache.col_is_var_double.assign(static_cast<std::size_t>(sys_.nbr_unknowns), 0);

        std::unordered_map<std::string, int> variable_ids;
        variable_ids.reserve(static_cast<std::size_t>(sys_.nvar + sys_.nvar_double + 2));
        int next_id = 0;
        for (int i = 0; i < sys_.nvar; ++i) {
            if (!sys_.names_var[i].empty())
                variable_ids.emplace(sys_.names_var[i].c_str(), next_id++);
        }
        for (int i = 0; i < sys_.nvar_double; ++i) {
            if (!sys_.names_var_double[i].empty())
                variable_ids.emplace(sys_.names_var_double[i].c_str(), next_id++);
        }
        variable_ids.emplace("__var_domain__", next_id++);

        std::vector<ColumnInfo> column_map;
        sys_.build_column_map(column_map);
        for (int c = 0; c < sys_.nbr_unknowns; ++c) {
            const auto& info = column_map[static_cast<std::size_t>(c)];
            auto it = variable_ids.find(info.var_name);
            if (it != variable_ids.end())
                def_dep_cache.col_var_id[static_cast<std::size_t>(c)] = it->second;
            def_dep_cache.col_term_idx[static_cast<std::size_t>(c)] = info.term_idx;
            if (info.term_idx >= 0 &&
                info.term_idx < sys_.nterm &&
                def_dep_cache.term_start_col[static_cast<std::size_t>(info.term_idx)] < 0) {
                def_dep_cache.term_start_col[static_cast<std::size_t>(info.term_idx)] = c;
            }
            def_dep_cache.col_is_var_domain[static_cast<std::size_t>(c)] = info.is_var_domain ? 1 : 0;
            def_dep_cache.col_is_var_double[static_cast<std::size_t>(c)] =
                (info.var_double_idx >= 0) ? 1 : 0;
        }

        std::set<std::string> variables;
        for (int i = 0; i < sys_.ndef; ++i) {
            variables.clear();
            sys_.def[i]->collect_vars(variables);
            auto& deps = def_dep_cache.def_dep_ids[static_cast<std::size_t>(i)];
            deps.clear();
            deps.reserve(variables.size());
            for (const auto& name : variables) {
                auto it = variable_ids.find(name);
                if (it != variable_ids.end())
                    deps.push_back(it->second);
            }
            std::sort(deps.begin(), deps.end());
            deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
        }

        for (int equation_index = 0; equation_index < sys_.neq; ++equation_index) {
            variables.clear();
            const Equation* equation = sys_.eq[static_cast<std::size_t>(equation_index)].get();
            if (equation != nullptr) {
                for (int part_index = 0; part_index < equation->n_ope; ++part_index) {
                    if (equation->parts[part_index] != nullptr)
                        equation->parts[part_index]->collect_vars(variables);
                }
            }

            auto& deps = def_dep_cache.equation_dep_ids[static_cast<std::size_t>(equation_index)];
            deps.clear();
            deps.reserve(variables.size());
            for (const auto& name : variables) {
                auto it = variable_ids.find(name);
                if (it != variable_ids.end())
                    deps.push_back(it->second);
            }
            std::sort(deps.begin(), deps.end());
            deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
        }
    }

    if (!def_dep_cache.tau_seed_descriptors_ready) {
        using CachedTauSeed = DoColJDefDepCache::CachedTauSeed;
        def_dep_cache.col_tau_seed.assign(
            static_cast<std::size_t>(sys_.nbr_unknowns), CachedTauSeed{});
        std::vector<ColumnInfo> column_map;
        sys_.build_column_map(column_map);
        if (column_map.size() == static_cast<std::size_t>(sys_.nbr_unknowns)) {
            std::vector<TauSeedDescriptor> descriptors;
            for (int term_index = 0; term_index < sys_.nterm; ++term_index) {
                const int term_start =
                    def_dep_cache.term_start_col[static_cast<std::size_t>(term_index)];
                if (term_start < 0)
                    continue;
                const Term_eq* const term = sys_.term[term_index].get();
                const int domain = term->get_dom();
                const Tensor& value = term->get_val_t();
                const Domain* const domain_ptr = sys_.espace.get_domain(domain);
                if (!domain_ptr->describe_tau_seed_block(
                        value, domain, descriptors)) {
                    continue;
                }
                const int expected = domain_ptr->nbr_unknowns(value, domain);
                if (expected < 0 ||
                    descriptors.size() != static_cast<std::size_t>(expected) ||
                    term_start + expected > sys_.nbr_unknowns) {
                    continue;
                }
                for (int mode = 0; mode < expected; ++mode) {
                    const int column = term_start + mode;
                    const ColumnInfo& info =
                        column_map[static_cast<std::size_t>(column)];
                    const TauSeedDescriptor& descriptor =
                        descriptors[static_cast<std::size_t>(mode)];
                    if (info.term_idx != term_index || info.domain != domain ||
                        info.basis_mode != mode || descriptor.component < 0 ||
                        descriptor.component >= value.get_n_comp() ||
                        descriptor.write_count <= 0 ||
                        descriptor.write_count > TauSeedDescriptor::max_writes) {
                        continue;
                    }
                    CachedTauSeed& cached =
                        def_dep_cache.col_tau_seed[static_cast<std::size_t>(column)];
                    cached.term_index = term_index;
                    cached.domain = domain;
                    cached.basis_mode = mode;
                    cached.supported = true;
                    cached.descriptor = descriptor;
                }
            }
        }
        def_dep_cache.tau_seed_descriptors_ready = true;
    }
}

void JacobianColumnEngine::compute_column(
    int cc, Array<double>& res, JacobianSelectedRows selected_rows)
{
    compute_column(default_workspace_, cc, res, selected_rows);
}

void JacobianColumnEngine::compute_column(
    Workspace& workspace, int cc, Array<double>& res,
    JacobianSelectedRows selected_rows)
{

    assert((cc >= 0) && (cc < sys_.nbr_unknowns));

    const bool timing = sys_.solver_runtime_config.diagnostics.timing;
    auto& state = workspace.state_;
    auto& timing_state = state.timing;
    auto& def_domain_cache = state.def_domain_cache;
    auto& def_dep_cache = state.def_dep_cache;
    std::chrono::steady_clock::time_point t0;
    std::chrono::steady_clock::time_point column_start;
    if (timing) {
        ++timing_state.calls;
        column_start = std::chrono::steady_clock::now();
        t0 = column_start;
    }

    // Affecte nterms derivatives :
    int conte = 0;
    int zedom = -1;
    bool is_var_double = false;
    Array<int> zedoms(2);
    zedoms = -1;
    // Variable Domains :
    // affecte_coef_to_variable_domains is expensive (calls update() on adapted domains).
    // Cache nvar_domain and skip the call for non-variable-domain columns, which only
    // need the derivative zeroed once after processing a variable-domain column.
    if (def_dep_cache.nvar_domain < 0) {
        // First call: discover nvar_domain
        sys_.espace.affecte_coef_to_variable_domains(conte, cc, zedoms);
        def_dep_cache.nvar_domain = conte;
        def_dep_cache.prev_was_var_domain = (zedoms(0) != -1);
    } else if (cc < def_dep_cache.nvar_domain) {
        // Variable-domain column: call normally
        sys_.espace.affecte_coef_to_variable_domains(conte, cc, zedoms);
        def_dep_cache.prev_was_var_domain = true;
    } else if (def_dep_cache.prev_was_var_domain) {
        // First non-var-domain column after a var-domain one: zero radius derivatives
        int tmp_conte = 0;
        sys_.espace.affecte_coef_to_variable_domains(tmp_conte, cc, zedoms);
        conte = def_dep_cache.nvar_domain;
        def_dep_cache.prev_was_var_domain = false;
    } else {
        // Non-var-domain column, radius derivatives already zero: skip entirely
        conte = def_dep_cache.nvar_domain;
    }
    for (int i = 0; i < sys_.nterm_cst; i++)
        sys_.cst[i]->set_der_zero();
    for (int i = 0; i < sys_.nterm; i++)
        sys_.term[i]->set_der_zero();
    if (zedoms(0) != -1)
        sys_.update_terms_from_variable_domains(zedoms);
    if (timing) {
        auto t1 = std::chrono::steady_clock::now();
        timing_state.update_terms += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    }

    // Double
    for (int i = 0; i < sys_.nvar_double; i++) {
        if (conte == cc) {
            for (int dd = sys_.dom_min; dd <= sys_.dom_max; dd++)
                sys_.term_double[i * sys_.ndom + (dd - sys_.dom_min)]->set_der_d(1.);
            is_var_double = true;
        } else
            for (int dd = sys_.dom_min; dd <= sys_.dom_max; dd++)
                sys_.term_double[i * sys_.ndom + (dd - sys_.dom_min)]->set_der_d(0.);
        conte++;
    }
    if (timing) {
        auto t1 = std::chrono::steady_clock::now();
        timing_state.var_double += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    }

    refresh_definition_filter_caches(workspace);

    const int ndoms = (sys_.dom_max >= sys_.dom_min) ? (sys_.dom_max - sys_.dom_min + 1) : 0;
    state.dom_affected.assign(static_cast<std::size_t>(ndoms), 0);

    // Fields
    const bool no_var_domain = (zedoms(0) == -1);
    auto seed_field_term_derivative = [&](int term_index, int& column_counter) -> bool {
        int dom = sys_.term[term_index]->get_dom();
        const Tensor& term_value = *sys_.term[term_index]->get_p_val_t();
        const Domain* const domain = sys_.espace.get_domain(dom);
        std::unique_ptr<Tensor> direct_seed;
        const bool cached_column =
            cc >= 0 && cc < sys_.nbr_unknowns &&
            def_dep_cache.col_tau_seed.size() ==
                static_cast<std::size_t>(sys_.nbr_unknowns) &&
            def_dep_cache.col_tau_seed[static_cast<std::size_t>(cc)].supported;
        if (cached_column) {
            const auto& cached =
                def_dep_cache.col_tau_seed[static_cast<std::size_t>(cc)];
            if (cached.term_index == term_index && cached.domain == dom &&
                cached.basis_mode == cc - column_counter) {
                auto candidate = std::make_unique<Tensor>(
                    one_domain_storage, dom, term_value, false);
                if (domain->materialize_tau_seed(
                        *candidate, term_value, dom, cached.descriptor)) {
                    direct_seed = std::move(candidate);
                    column_counter += domain->nbr_unknowns(term_value, dom);
                }
            }
        }
        std::unique_ptr<Tensor> legacy_seed;
        if (direct_seed == nullptr) {
            legacy_seed = std::make_unique<Tensor>(term_value, false);
            domain->affecte_tau_one_coef(
                *legacy_seed, dom, cc, column_counter);
        }
        Tensor& auxi = direct_seed != nullptr ? *direct_seed : *legacy_seed;
        bool term_is_affected = false;
        const int ncomp_auxi = auxi.get_n_comp();
        for (int j = 0; j < ncomp_auxi; j++) {
            // Si la base n'est pas affectee on la met
            const Array<int> idx = auxi.indices(j);
            if (!auxi(idx)(dom).check_if_zero()) {
                term_is_affected = true;
                if (!auxi(idx)(dom).get_base().is_def())
                    auxi.set(idx).set_domain(dom).set_base() = term_value(idx)(dom).get_base();
                if (zedom == -1)
                    zedom = dom;
            }
        }
        // When no variable domains, der_t is zero after set_der_zero() above — skip
        // the redundant get_der_t() copy (returns zero Tensor by value).
        if (no_var_domain) {
            sys_.term[term_index]->set_der_t(auxi);
        } else {
            const Tensor* term_derivative = sys_.term[term_index]->get_p_der_t();
            if (term_derivative != nullptr)
                sys_.term[term_index]->set_der_t(auxi + *term_derivative);
            else
                sys_.term[term_index]->set_der_t(auxi);
        }
        return term_is_affected;
    };

    const int mapped_term =
        (no_var_domain && !is_var_double && cc >= 0 && cc < sys_.nbr_unknowns &&
         def_dep_cache.col_term_idx.size() == static_cast<std::size_t>(sys_.nbr_unknowns))
            ? def_dep_cache.col_term_idx[static_cast<std::size_t>(cc)]
            : -1;
    const bool can_seed_mapped_field =
        mapped_term >= 0 &&
        mapped_term < sys_.nterm &&
        def_dep_cache.term_start_col.size() == static_cast<std::size_t>(sys_.nterm) &&
        def_dep_cache.term_start_col[static_cast<std::size_t>(mapped_term)] >= 0;
    if (can_seed_mapped_field) {
        int column_counter =
            def_dep_cache.term_start_col[static_cast<std::size_t>(mapped_term)];
        if (!seed_field_term_derivative(mapped_term, column_counter)) {
            for (int i = 0; i < sys_.nterm; i++)
                seed_field_term_derivative(i, conte);
        }
    } else {
        for (int i = 0; i < sys_.nterm; i++)
            seed_field_term_derivative(i, conte);
    }
    if (timing) {
        auto t1 = std::chrono::steady_clock::now();
        timing_state.fields += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    }

    if (sys_.do_col_J_def_filter_enabled()) {
        if (zedom != -1 && zedom >= sys_.dom_min && zedom <= sys_.dom_max) {
            state.dom_affected[static_cast<std::size_t>(zedom - sys_.dom_min)] = 1;
        }
        if (zedoms(0) != -1 && zedoms(0) >= sys_.dom_min && zedoms(0) <= sys_.dom_max) {
            state.dom_affected[static_cast<std::size_t>(zedoms(0) - sys_.dom_min)] = 1;
        }
        if (zedoms(1) != -1 && zedoms(1) >= sys_.dom_min && zedoms(1) <= sys_.dom_max) {
            state.dom_affected[static_cast<std::size_t>(zedoms(1) - sys_.dom_min)] = 1;
        }
    }
    // Delete the metric derivative terms :
    if (sys_.met != nullptr) {
        for (int d = sys_.dom_min; d <= sys_.dom_max; d++)
            sys_.met->update(d);
    }
    if (timing) {
        auto t1 = std::chrono::steady_clock::now();
        timing_state.metric_update += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    }

    // Delete the definitions
    if (timing && timing_state.def_eq.size() != static_cast<std::size_t>(sys_.ndef)) {
        timing_state.def_eq.assign(static_cast<std::size_t>(sys_.ndef), 0.0);
        timing_state.def_calls.assign(static_cast<std::size_t>(sys_.ndef), 0);
    }
    bool compute_all_defs = is_var_double || !sys_.do_col_J_def_filter_enabled();
    const bool cur_is_var_domain = (cc >= 0 && cc < sys_.nbr_unknowns)
                                       ? (def_dep_cache.col_is_var_domain[static_cast<std::size_t>(cc)] != 0)
                                       : false;
    const bool cur_is_var_double = (cc >= 0 && cc < sys_.nbr_unknowns)
                                       ? (def_dep_cache.col_is_var_double[static_cast<std::size_t>(cc)] != 0)
                                       : false;
    if (cur_is_var_domain || cur_is_var_double) {
        compute_all_defs = true;
    }
    const bool establishes_base_defs = !def_dep_cache.base_defs_ready;
    if (establishes_base_defs) {
        compute_all_defs = true;
    }
    const int cur_var_id =
        (cc >= 0 && cc < sys_.nbr_unknowns) ? def_dep_cache.col_var_id[static_cast<std::size_t>(cc)] : -1;
    auto def_depends_on_cur_var = [&](int idx) -> bool {
        if (cur_var_id < 0)
            return false;
        const auto& deps = def_dep_cache.def_dep_ids[static_cast<std::size_t>(idx)];
        return std::binary_search(deps.begin(), deps.end(), cur_var_id);
    };
    if (compute_all_defs) {
        if (timing) {
            auto t_def_compute_start = std::chrono::steady_clock::now();
            for (int i = 0; i < sys_.ndef; i++) {
                auto t_def_start = std::chrono::steady_clock::now();
                sys_.def[i]->compute_res();
                auto t_def_end = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double>(t_def_end - t_def_start).count();
                timing_state.def_eq[static_cast<std::size_t>(i)] += dt;
                timing_state.def_calls[static_cast<std::size_t>(i)] += 1;
            }
            auto t_def_compute_end = std::chrono::steady_clock::now();
            timing_state.def_compute +=
                std::chrono::duration<double>(t_def_compute_end - t_def_compute_start).count();
        } else {
            for (int i = 0; i < sys_.ndef; i++) {
                sys_.def[i]->compute_res();
            }
        }
        if (establishes_base_defs)
            def_dep_cache.base_defs_ready = true;
    } else {
        if (timing) {
            auto t_def_clear_start = std::chrono::steady_clock::now();
            for (int i = 0; i < sys_.ndef; ++i) {
                sys_.def[i]->get_res()->set_der_zero();
            }
            auto t_def_clear_end = std::chrono::steady_clock::now();
            timing_state.def_clear += std::chrono::duration<double>(t_def_clear_end - t_def_clear_start).count();
        } else {
            for (int i = 0; i < sys_.ndef; ++i) {
                sys_.def[i]->get_res()->set_der_zero();
            }
        }
        if (timing) {
            auto t_def_compute_start = std::chrono::steady_clock::now();
            for (int idx : def_domain_cache.defs_global) {
                if (!def_depends_on_cur_var(idx))
                    continue;
                auto t_def_start = std::chrono::steady_clock::now();
                sys_.def[idx]->compute_res();
                auto t_def_end = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double>(t_def_end - t_def_start).count();
                timing_state.def_eq[static_cast<std::size_t>(idx)] += dt;
                timing_state.def_calls[static_cast<std::size_t>(idx)] += 1;
            }
            for (int d = 0; d < ndoms; ++d) {
                if (!state.dom_affected[static_cast<std::size_t>(d)])
                    continue;
                for (int idx : def_domain_cache.defs_by_dom[static_cast<std::size_t>(d)]) {
                    if (!def_depends_on_cur_var(idx))
                        continue;
                    auto t_def_start = std::chrono::steady_clock::now();
                    sys_.def[idx]->compute_res();
                    auto t_def_end = std::chrono::steady_clock::now();
                    double dt = std::chrono::duration<double>(t_def_end - t_def_start).count();
                    timing_state.def_eq[static_cast<std::size_t>(idx)] += dt;
                    timing_state.def_calls[static_cast<std::size_t>(idx)] += 1;
                }
            }
            auto t_def_compute_end = std::chrono::steady_clock::now();
            timing_state.def_compute +=
                std::chrono::duration<double>(t_def_compute_end - t_def_compute_start).count();
        } else {
            for (int idx : def_domain_cache.defs_global) {
                if (!def_depends_on_cur_var(idx))
                    continue;
                sys_.def[idx]->compute_res();
            }
            for (int d = 0; d < ndoms; ++d) {
                if (!state.dom_affected[static_cast<std::size_t>(d)])
                    continue;
                for (int idx : def_domain_cache.defs_by_dom[static_cast<std::size_t>(d)]) {
                    if (!def_depends_on_cur_var(idx))
                        continue;
                    sys_.def[idx]->compute_res();
                }
            }
        }
    }
    if (timing) {
        auto t1 = std::chrono::steady_clock::now();
        timing_state.def_update += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    }

    if (timing && timing_state.apply_eq.size() != static_cast<std::size_t>(sys_.neq)) {
        timing_state.apply_eq.assign(static_cast<std::size_t>(sys_.neq), 0.0);
        timing_state.apply_calls.assign(static_cast<std::size_t>(sys_.neq), 0);
    }

    // Precompute per-equation filter once; reused by apply, export_der, and sparse export.
    // Instance-owned scratch avoids cross-System_of_eqs state leakage.
    state.eq_affects.resize(static_cast<std::size_t>(sys_.neq));
    state.active_equation_ranges.clear();
    state.active_equation_ranges.reserve(static_cast<std::size_t>(sys_.neq));
    std::vector<ActiveEquationRange> dependency_skipped_ranges_to_verify;
    if (active_equation_range_verify_enabled())
        dependency_skipped_ranges_to_verify.reserve(static_cast<std::size_t>(sys_.neq));
    int operator_offset = 0;
    int row_offset = sys_.neq_int;
    for (int i = 0; i < sys_.neq; i++) {
        const bool domain_marks_equation = is_var_double || sys_.eq[i]->take_into_account(zedom) ||
                                           sys_.eq[i]->take_into_account(zedoms(0)) ||
                                           sys_.eq[i]->take_into_account(zedoms(1));
        bool variable_marks_equation = true;
        if (!is_var_double && !cur_is_var_domain && !cur_is_var_double && cur_var_id >= 0 &&
            def_dep_cache.equation_dep_ids.size() == static_cast<std::size_t>(sys_.neq)) {
            const auto& equation_deps =
                def_dep_cache.equation_dep_ids[static_cast<std::size_t>(i)];
            variable_marks_equation =
                std::binary_search(equation_deps.begin(), equation_deps.end(), cur_var_id);
        }
        const int row_count = sys_.eq[i]->get_n_cond_tot();
        const bool selection_marks_equation =
            jacobian_column_engine_detail::range_contains_selected_index(
                row_offset, row_offset + row_count, selected_rows);
        const bool equation_is_active = domain_marks_equation &&
                                        variable_marks_equation &&
                                        selection_marks_equation;
        if (timing && domain_marks_equation) {
            ++timing_state.equation_domain_active;
            if (equation_is_active)
                ++timing_state.equation_active;
            else if (!variable_marks_equation)
                ++timing_state.equation_variable_skipped;
        }
        state.eq_affects[static_cast<std::size_t>(i)] = equation_is_active ? 1 : 0;
        if (equation_is_active) {
            ActiveEquationRange active_range;
            active_range.equation_index = i;
            active_range.operator_offset = operator_offset;
            active_range.row_begin = row_offset;
            active_range.row_end = row_offset + row_count;
            state.active_equation_ranges.push_back(active_range);
        } else if (active_equation_range_verify_enabled() &&
                   domain_marks_equation &&
                   !variable_marks_equation) {
            ActiveEquationRange skipped_range;
            skipped_range.equation_index = i;
            skipped_range.operator_offset = operator_offset;
            skipped_range.row_begin = row_offset;
            skipped_range.row_end = row_offset + row_count;
            dependency_skipped_ranges_to_verify.push_back(skipped_range);
        }
        operator_offset += sys_.eq[i]->n_ope;
        row_offset += row_count;
    }
    if (active_equation_range_verify_enabled()) {
        std::size_t next_active_range = 0;
        int expected_operator_offset = 0;
        int expected_row_offset = sys_.neq_int;
        for (int i = 0; i < sys_.neq; ++i) {
            const bool range_marks_equation =
                next_active_range < state.active_equation_ranges.size() &&
                state.active_equation_ranges[next_active_range].equation_index == i;
            const bool mask_marks_equation = state.eq_affects[static_cast<std::size_t>(i)] != 0;
            if (range_marks_equation != mask_marks_equation) {
                KADATH_THROW("JacobianColumnEngine active-equation range verifier failed: mask/range mismatch");
            }
            const int row_count = sys_.eq[i]->get_n_cond_tot();
            if (range_marks_equation) {
                const ActiveEquationRange& active_range =
                    state.active_equation_ranges[next_active_range];
                if (active_range.operator_offset != expected_operator_offset ||
                    active_range.row_begin != expected_row_offset ||
                    active_range.row_end != expected_row_offset + row_count) {
                    KADATH_THROW("JacobianColumnEngine active-equation range verifier failed: offset mismatch");
                }
                ++next_active_range;
            }
            expected_operator_offset += sys_.eq[i]->n_ope;
            expected_row_offset += row_count;
        }
        if (next_active_range != state.active_equation_ranges.size() ||
            expected_row_offset != sys_.nbr_conditions) {
            KADATH_THROW("JacobianColumnEngine active-equation range verifier failed: range count mismatch");
        }
    }
    if (timing) {
        auto t1 = std::chrono::steady_clock::now();
        timing_state.eq_affects += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    }

    Term_eq** results_raw = sys_.results_shadow_view();
    if (active_equation_range_verify_enabled()) {
        for (const ActiveEquationRange& skipped_range : dependency_skipped_ranges_to_verify) {
            const int equation_index = skipped_range.equation_index;
            const int row_count = skipped_range.row_end - skipped_range.row_begin;
            if (row_count <= 0)
                continue;

            int equation_operator_offset = skipped_range.operator_offset;
            sys_.eq[equation_index]->apply(equation_operator_offset, results_raw);
            if (equation_operator_offset != skipped_range.operator_offset + sys_.eq[equation_index]->n_ope) {
                KADATH_THROW("JacobianColumnEngine active-equation range verifier failed: skipped apply offset mismatch");
            }

            Array<double> skipped_export(row_count);
            skipped_export = 0.0;
            equation_operator_offset = skipped_range.operator_offset;
            int local_pos_res = 0;
            sys_.eq[equation_index]->export_der(equation_operator_offset, results_raw,
                                                skipped_export, local_pos_res);
            if (equation_operator_offset != skipped_range.operator_offset + sys_.eq[equation_index]->n_ope ||
                local_pos_res != row_count) {
                KADATH_THROW("JacobianColumnEngine active-equation range verifier failed: skipped export offset mismatch");
            }

            for (int local_row = 0; local_row < row_count; ++local_row) {
                if (std::abs(skipped_export(local_row)) > 1e-13) {
                    KADATH_THROW("JacobianColumnEngine variable-dependency equation filter skipped a nonzero derivative");
                }
            }
        }
    }
    for (const ActiveEquationRange& active_range : state.active_equation_ranges) {
        const int equation_index = active_range.equation_index;
        int equation_operator_offset = active_range.operator_offset;
        if (timing) {
            auto t_apply_start = std::chrono::steady_clock::now();
            sys_.eq[equation_index]->apply(equation_operator_offset, results_raw);
            auto t_apply_end = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(t_apply_end - t_apply_start).count();
            timing_state.apply_eq[static_cast<std::size_t>(equation_index)] += dt;
            timing_state.apply_calls[static_cast<std::size_t>(equation_index)] += 1;
        } else {
            sys_.eq[equation_index]->apply(equation_operator_offset, results_raw);
        }
        if (active_equation_range_verify_enabled() &&
            equation_operator_offset != active_range.operator_offset + sys_.eq[equation_index]->n_ope) {
            KADATH_THROW("JacobianColumnEngine active-equation range verifier failed: apply offset mismatch");
        }
    }
    sys_.results_shadow_sync();
    if (timing) {
        auto t1 = std::chrono::steady_clock::now();
        timing_state.apply += std::chrono::duration<double>(t1 - t0).count();
        t0 = t1;
    }

    // Dense callers own freshly zeroed output. The sparse path reuses its
    // column buffer, so clear only the rows this column can export and scan.
    // This must precede export: equation exporters may omit a missing
    // derivative and rely on the destination row already containing zero.
    if (state.column_buffer && &res == state.column_buffer.get()) {
        clear_active_output_rows(
            res, sys_.neq_int, sys_.nbr_conditions, state.active_equation_ranges);
    }

    int pos_res = 0;
    jacobian_column_engine_detail::for_each_selected_index_in_range(
        0, sys_.neq_int, selected_rows,
        [&](int row) { res.set(row) = sys_.eq_int[row]->get_der(); });

    for (const ActiveEquationRange& active_range : state.active_equation_ranges) {
        const int equation_index = active_range.equation_index;
        int equation_operator_offset = active_range.operator_offset;
        pos_res = active_range.row_begin;
        sys_.eq[equation_index]->export_der(equation_operator_offset, results_raw, res, pos_res);
        if (active_equation_range_verify_enabled()) {
            if (equation_operator_offset != active_range.operator_offset + sys_.eq[equation_index]->n_ope ||
                pos_res != active_range.row_end) {
                KADATH_THROW("JacobianColumnEngine active-equation range verifier failed: export offset mismatch");
            }
        }
    }
    // Zero def derivatives after use; keep der_t allocated to avoid per-column realloc.
    for (int i = 0; i < sys_.ndef; ++i) {
        sys_.def[i]->get_res()->set_der_zero();
    }
    if (timing) {
        auto t1 = std::chrono::steady_clock::now();
        timing_state.export_der += std::chrono::duration<double>(t1 - t0).count();
        if (cur_is_var_domain) {
            ++timing_state.var_domain_calls;
            timing_state.var_domain_total +=
                std::chrono::duration<double>(t1 - column_start).count();
        }
    }
}

Array<double>& JacobianColumnEngine::do_col_J_buffer()
{
    return do_col_J_buffer(default_workspace_);
}

Array<double>& JacobianColumnEngine::do_col_J_buffer(Workspace& workspace)
{
    if (workspace.state_.column_buffer_size != sys_.nbr_conditions) {
        workspace.state_.column_buffer =
            std::make_unique<Array<double>>(sys_.nbr_conditions);
        workspace.state_.column_buffer_size = sys_.nbr_conditions;
    }
    return *workspace.state_.column_buffer;
}

void JacobianColumnEngine::compute_column_sparse(int cc, double drop_tol,
                                                 SparseColumnEmitter emit,
                                                 JacobianSelectedRows selected_rows)
{
    compute_column_sparse(default_workspace_, cc, drop_tol, emit,
                          selected_rows);
}

void JacobianColumnEngine::compute_column_sparse(
    Workspace& workspace, int cc, double drop_tol, SparseColumnEmitter emit,
    JacobianSelectedRows selected_rows)
{
    if (sys_.nbr_conditions == -1) {
        KADATH_THROW("Number of conditions unknown ; call sec_member first");
    }
    Array<double>& buf = do_col_J_buffer(workspace);
    compute_column(workspace, cc, buf, selected_rows);
    const bool timing = sys_.solver_runtime_config.diagnostics.timing;
    long long scanned_rows = 0;
    const bool use_equation_ranges =
        workspace.state_.eq_affects.size() == static_cast<std::size_t>(sys_.neq);
    if (use_equation_ranges) {
        auto scan_range = [&](int begin, int end) {
            begin = std::max(0, begin);
            end = std::min(sys_.nbr_conditions, end);
            if (end <= begin) {
                return;
            }
            scanned_rows +=
                jacobian_column_engine_detail::for_each_selected_index_in_range(
                    begin, end, selected_rows, [&](int row) {
                        const double val = buf(row);
                        if (std::abs(val) > drop_tol)
                            emit(row, val);
                    });
        };

        scan_range(0, sys_.neq_int);
        for (const ActiveEquationRange& active_range :
             workspace.state_.active_equation_ranges) {
            scan_range(active_range.row_begin, active_range.row_end);
        }
    } else {
        scanned_rows =
            jacobian_column_engine_detail::for_each_selected_index_in_range(
                0, sys_.nbr_conditions, selected_rows, [&](int row) {
                    const double val = buf(row);
                    if (std::abs(val) > drop_tol)
                        emit(row, val);
                });
    }
    if (timing) {
        auto& timing_state = workspace.state_.timing;
        timing_state.sparse_scan_rows += scanned_rows;
        timing_state.sparse_full_scan_rows += static_cast<long long>(sys_.nbr_conditions);
    }
}

template <int W>
bool JacobianColumnEngine::compute_packed_wlaneN_columns_sparse(
    Workspace& workspace,
    PackedJacobianColumns<W> columns,
    double drop_tol,
    PackedSparseColumnEmitters<W> emitters,
    std::string& failure_reason,
    JacobianSelectedRows selected_rows)
{
    static_assert(W == 2 || W == 4 || W == 8 || W == 16 || W == 32,
                  "packed W-lane sparse path supports only W in {2,4,8,16,32}");
    constexpr const char* distinct_columns_message =
        (W == 2)  ? "packed W=2 sparse path requires two distinct columns"
        : (W == 4)  ? "packed W=4 sparse path requires four distinct columns"
        : (W == 8)  ? "packed W=8 sparse path requires eight distinct columns"
        : (W == 16) ? "packed W=16 sparse path requires sixteen distinct columns"
                    : "packed W=32 sparse path requires thirty-two distinct columns";

    failure_reason.clear();
    if (sys_.nbr_conditions == -1) {
        KADATH_THROW("Number of conditions unknown ; call sec_member first");
    }
    for (int k = 0; k < W; ++k) {
        if (columns[k] < 0 || columns[k] >= sys_.nbr_unknowns) {
            failure_reason = "selected column is outside the system unknown range";
            return false;
        }
    }
    for (int a = 0; a < W; ++a) {
        for (int b = a + 1; b < W; ++b) {
            if (columns[a] == columns[b]) {
                failure_reason = distinct_columns_message;
                return false;
            }
        }
    }

    auto& state = workspace.state_;
    auto& def_domain_cache = state.def_domain_cache;
    auto& def_dep_cache = state.def_dep_cache;
    refresh_definition_filter_caches(workspace);
    bool any_variable_domain = false;
    bool all_variable_domain = true;
    for (int k = 0; k < W; ++k) {
        const bool is_variable_domain =
            def_dep_cache.col_is_var_domain[static_cast<std::size_t>(columns[k])] != 0;
        any_variable_domain = any_variable_domain || is_variable_domain;
        all_variable_domain = all_variable_domain && is_variable_domain;
    }
    if (any_variable_domain) {
        if (!all_variable_domain) {
            failure_reason = "packed variable-domain groups require only variable-domain columns";
            return false;
        }
        if (!env_flag_enabled("JACOBIAN_VARDOM_WLANE2", true)) {
            failure_reason = "packed variable-domain lanes are disabled";
            return false;
        }
        if (!sys_.espace.supports_packed_variable_domain_jacobian()) {
            failure_reason = "adapted space does not support packed variable-domain lanes";
            return false;
        }
    }

    if (state.packed_buffer_size != sys_.nbr_conditions) {
        state.packed_first_buffer = std::make_unique<Array<double>>(sys_.nbr_conditions);
        state.packed_second_buffer = std::make_unique<Array<double>>(sys_.nbr_conditions);
        state.packed_buffer_size = sys_.nbr_conditions;
    }
    if constexpr (W > 2) {
        if (state.packed_extra_buffer_size != sys_.nbr_conditions) {
            for (auto& buffer : state.packed_extra_buffers)
                buffer.reset();
            state.packed_extra_buffer_size = sys_.nbr_conditions;
        }
        for (int lane = 2; lane < W; ++lane) {
            auto& buffer = state.packed_extra_buffers[static_cast<std::size_t>(lane - 2)];
            if (!buffer)
                buffer = std::make_unique<Array<double>>(sys_.nbr_conditions);
        }
    }

    std::array<Array<double>*, W> outputs{};
    outputs[0] = state.packed_first_buffer.get();
    outputs[1] = state.packed_second_buffer.get();
    if constexpr (W > 2) {
        for (int lane = 2; lane < W; ++lane)
            outputs[static_cast<std::size_t>(lane)] =
                state.packed_extra_buffers[static_cast<std::size_t>(lane - 2)].get();
    }

    const bool timing = sys_.solver_runtime_config.diagnostics.timing;
    using Clock = std::chrono::steady_clock;
    const auto elapsed_seconds = [](Clock::time_point start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    // Per-def def_compute attribution (DEF_COMPUTE_PER_DEF=1). When off,
    // each call routes straight to compute_res() with a single bool compare;
    // when on, the per-def wall is accumulated into the shared def_eq/def_calls
    // arrays (same slots the scalar fallback writes). Sizing matches line ~460.
    const bool per_def_timing = timing && def_compute_per_def_enabled();
    if (per_def_timing &&
        state.timing.def_eq.size() != static_cast<std::size_t>(sys_.ndef)) {
        state.timing.def_eq.assign(static_cast<std::size_t>(sys_.ndef), 0.0);
        state.timing.def_calls.assign(static_cast<std::size_t>(sys_.ndef), 0);
    }
    const auto compute_def_timed = [&](int idx) {
        if (per_def_timing) {
            const auto def_start = Clock::now();
            sys_.def[idx]->compute_res();
            state.timing.def_eq[static_cast<std::size_t>(idx)] += elapsed_seconds(def_start);
            state.timing.def_calls[static_cast<std::size_t>(idx)] += 1;
        } else {
            sys_.def[idx]->compute_res();
        }
    };
    Clock::time_point packed_total_start;
    if (timing)
        packed_total_start = Clock::now();
    double packed_prepare_seconds = 0.0;
    double packed_seed_seconds = 0.0;
    double packed_metric_seconds = 0.0;
    double packed_definition_seconds = 0.0;
    double packed_definition_clear_seconds = 0.0;
    double packed_definition_compute_seconds = 0.0;
    double packed_equation_filter_seconds = 0.0;
    double packed_apply_seconds = 0.0;
    double packed_export_seconds = 0.0;
    double packed_export_clear_seconds = 0.0;
    double packed_export_integral_seconds = 0.0;
    double packed_export_equation_seconds = 0.0;
    double packed_scan_seconds = 0.0;
    double packed_restore_seconds = 0.0;
    if constexpr (W == 8) {
        if (timing &&
            state.timing.packed_wlane8_export_eq.size() != static_cast<std::size_t>(sys_.neq)) {
            state.timing.packed_wlane8_export_eq.assign(static_cast<std::size_t>(sys_.neq), 0.0);
            state.timing.packed_wlane8_export_calls.assign(static_cast<std::size_t>(sys_.neq), 0);
        }
    }
    if constexpr (W == 16) {
        if (timing &&
            state.timing.packed_wlane16_export_eq.size() != static_cast<std::size_t>(sys_.neq)) {
            state.timing.packed_wlane16_export_eq.assign(static_cast<std::size_t>(sys_.neq), 0.0);
            state.timing.packed_wlane16_export_calls.assign(static_cast<std::size_t>(sys_.neq), 0);
        }
    }
    if constexpr (W == 32) {
        if (timing &&
            state.timing.packed_wlane32_export_eq.size() != static_cast<std::size_t>(sys_.neq)) {
            state.timing.packed_wlane32_export_eq.assign(static_cast<std::size_t>(sys_.neq), 0.0);
            state.timing.packed_wlane32_export_calls.assign(static_cast<std::size_t>(sys_.neq), 0);
        }
    }

    // Widen every derivative-bearing category to W lanes. All five categories
    // (cst / term / term_double / results / def) must be widened for every W:
    // propagate_*_lanes consumes results / def via logical has_der_t presence.
    // All five categories must therefore have their active-lane masks cleared;
    // merely retaining their Tensor owners without clearing occupancy recreates
    // the +52172-nnz W=4 stale-lane regression. See
    // project-wlane-n-prepare-must-zero-results-and-def.
    auto prepare_packed_lane_derivatives = [&]() {
        for (int i = 0; i < sys_.nterm_cst; ++i) {
            set_primary_zero_and_retain_extra_derivative_lanes(sys_.cst[i], W);
        }
        for (int i = 0; i < sys_.nterm; ++i) {
            set_primary_zero_and_retain_extra_derivative_lanes(sys_.term[i], W);
        }
        for (int i = 0; i < sys_.nterm_double; ++i) {
            set_primary_zero_and_retain_extra_derivative_lanes(sys_.term_double[i], W);
        }
        for (auto& result : sys_.results) {
            if (result) {
                set_primary_zero_and_retain_extra_derivative_lanes(result.get(), W);
            }
        }
        for (int i = 0; i < sys_.ndef; ++i) {
            set_primary_zero_and_retain_extra_derivative_lanes(sys_.def[i]->get_res(), W);
        }
    };

    auto restore_scalar_derivative_lanes = [&]() {
        for (int i = 0; i < sys_.nterm_cst; ++i) {
            set_primary_zero_and_retain_extra_derivative_lanes(sys_.cst[i], 1);
        }
        for (int i = 0; i < sys_.nterm; ++i) {
            set_primary_zero_and_retain_extra_derivative_lanes(sys_.term[i], 1);
        }
        for (int i = 0; i < sys_.nterm_double; ++i) {
            set_primary_zero_and_retain_extra_derivative_lanes(sys_.term_double[i], 1);
        }
        for (auto& result : sys_.results) {
            if (result) {
                set_primary_zero_and_retain_extra_derivative_lanes(result.get(), 1);
            }
        }
        for (int i = 0; i < sys_.ndef; ++i) {
            set_primary_zero_and_retain_extra_derivative_lanes(sys_.def[i]->get_res(), 1);
        }
    };

    if (def_dep_cache.nvar_domain < 0 && any_variable_domain) {
        def_dep_cache.nvar_domain = sys_.espace.nbr_unknowns_from_variable_domains();
        def_dep_cache.prev_was_var_domain = false;
    } else if (def_dep_cache.nvar_domain < 0) {
        int variable_domain_count = 0;
        Array<int> affected_variable_domains(2);
        affected_variable_domains = -1;
        sys_.espace.affecte_coef_to_variable_domains(
            variable_domain_count, columns[0], affected_variable_domains);
        if (affected_variable_domains(0) != -1 || affected_variable_domains(1) != -1) {
            failure_reason = "variable-domain derivative appeared during W-lane setup";
            return false;
        }
        def_dep_cache.nvar_domain = variable_domain_count;
        def_dep_cache.prev_was_var_domain = false;
    }

    const int ndoms = (sys_.dom_max >= sys_.dom_min) ? (sys_.dom_max - sys_.dom_min + 1) : 0;
    state.dom_affected.assign(static_cast<std::size_t>(ndoms), 0);

    struct PackedColumnSeed {
        int zedom = -1;
        int second_zedom = -1;
        bool is_var_domain = false;
        bool is_var_double = false;
        int variable_id = -1;
    };

    auto mark_affected_domain = [&](int domain) {
        if (domain >= sys_.dom_min && domain <= sys_.dom_max) {
            state.dom_affected[static_cast<std::size_t>(domain - sys_.dom_min)] = 1;
        }
    };

    auto seed_field_term_derivative =
        [&](int column, int lane, int term_index, int& column_counter, PackedColumnSeed& seed) -> bool {
        const int domain = sys_.term[term_index]->get_dom();
        const Tensor& term_value = *sys_.term[term_index]->get_p_val_t();
        const Domain* const domain_ptr = sys_.espace.get_domain(domain);
        std::unique_ptr<Tensor> direct_seed;
        const bool cached_column =
            column >= 0 && column < sys_.nbr_unknowns &&
            def_dep_cache.col_tau_seed.size() ==
                static_cast<std::size_t>(sys_.nbr_unknowns) &&
            def_dep_cache.col_tau_seed[static_cast<std::size_t>(column)].supported;
        if (cached_column) {
            const auto& cached =
                def_dep_cache.col_tau_seed[static_cast<std::size_t>(column)];
            if (cached.term_index == term_index && cached.domain == domain &&
                cached.basis_mode == column - column_counter) {
                auto candidate = std::make_unique<Tensor>(
                    one_domain_storage, domain, term_value, false);
                if (domain_ptr->materialize_tau_seed(
                        *candidate, term_value, domain, cached.descriptor)) {
                    direct_seed = std::move(candidate);
                    column_counter += domain_ptr->nbr_unknowns(term_value, domain);
                }
            }
        }
        std::unique_ptr<Tensor> legacy_seed;
        if (direct_seed == nullptr) {
            legacy_seed = std::make_unique<Tensor>(term_value, false);
            domain_ptr->affecte_tau_one_coef(
                *legacy_seed, domain, column, column_counter);
        }
        Tensor& derivative_seed =
            direct_seed != nullptr ? *direct_seed : *legacy_seed;

        bool term_is_affected = false;
        const int ncomp_auxi = derivative_seed.get_n_comp();
        for (int component = 0; component < ncomp_auxi; ++component) {
            const Array<int> index = derivative_seed.indices(component);
            if (!derivative_seed(index)(domain).check_if_zero()) {
                term_is_affected = true;
                if (!derivative_seed(index)(domain).get_base().is_def()) {
                    derivative_seed.set(index).set_domain(domain).set_base() =
                        term_value(index)(domain).get_base();
                }
                if (seed.zedom == -1)
                    seed.zedom = domain;
            }
        }
        if (term_is_affected) {
            sys_.term[term_index]->set_der_t(lane, derivative_seed);
        }
        return term_is_affected;
    };

    auto seed_column_into_lane = [&](int column, int lane, PackedColumnSeed& seed) -> bool {
        seed.variable_id = def_dep_cache.col_var_id[static_cast<std::size_t>(column)];
        int column_counter = def_dep_cache.nvar_domain;
        for (int variable_index = 0; variable_index < sys_.nvar_double; ++variable_index) {
            if (column_counter == column) {
                for (int domain = sys_.dom_min; domain <= sys_.dom_max; ++domain) {
                    sys_.term_double[static_cast<std::size_t>(
                        variable_index * sys_.ndom + (domain - sys_.dom_min))]->set_der_d(lane, 1.0);
                }
                seed.is_var_double = true;
            }
            ++column_counter;
        }
        if (seed.is_var_double)
            return true;

        const int mapped_term =
            def_dep_cache.col_term_idx[static_cast<std::size_t>(column)];
        const bool can_seed_mapped_field =
            mapped_term >= 0 &&
            mapped_term < sys_.nterm &&
            def_dep_cache.term_start_col[static_cast<std::size_t>(mapped_term)] >= 0;
        if (can_seed_mapped_field) {
            int mapped_counter =
                def_dep_cache.term_start_col[static_cast<std::size_t>(mapped_term)];
            if (seed_field_term_derivative(column, lane, mapped_term, mapped_counter, seed))
                return true;
        }

        int field_column_counter = def_dep_cache.nvar_domain + sys_.nvar_double;
        bool any_field_seeded = false;
        for (int term_index = 0; term_index < sys_.nterm; ++term_index) {
            any_field_seeded =
                seed_field_term_derivative(column, lane, term_index, field_column_counter, seed) ||
                any_field_seeded;
        }
        return any_field_seeded;
    };

    std::array<PackedColumnSeed, W> seeds{};

    auto definition_depends_on_any_seed = [&](int definition_index) -> bool {
        const auto& deps = def_dep_cache.def_dep_ids[static_cast<std::size_t>(definition_index)];
        for (const PackedColumnSeed& seed : seeds) {
            if (seed.variable_id >= 0 &&
                std::binary_search(deps.begin(), deps.end(), seed.variable_id))
                return true;
        }
        return false;
    };

    // Packed-export volume path: zero only rows that this tile can export/scan,
    // write per-lane Eq_int
    // integral derivatives (via the packed get_der_lanes accessor, which sums
    // the same per-operator terms in the same order as the scalar get_der(lane)
    // and is therefore bit-identical), then dispatch each active equation
    // through export_der_lanes once. Eq_full / Eq_inside read get_p_der_t(lane)
    // directly via their override; other subclasses fall back to a local
    // n_ope-scoped primary-slot swap inside Equation::export_der_lanes
    // (byte-identical to the retired global swap because operator-slot ranges
    // do not overlap across equations).
    auto clear_export_outputs = [&]() {
        Clock::time_point export_part_start;
        if (timing)
            export_part_start = Clock::now();
        for (Array<double>* output : outputs)
            clear_active_output_rows(
                *output, sys_.neq_int, sys_.nbr_conditions, state.active_equation_ranges);
        if (timing) {
            packed_export_clear_seconds += elapsed_seconds(export_part_start);
        }
    };

    auto export_integral_lanes = [&]() {
        Clock::time_point export_part_start;
        if (timing)
            export_part_start = Clock::now();
        jacobian_column_engine_detail::for_each_selected_index_in_range(
            0, sys_.neq_int, selected_rows, [&](int row) {
                std::array<double, W> integral_derivatives{};
                sys_.eq_int[row]->get_der_lanes(W, integral_derivatives.data());
                for (int lane = 0; lane < W; ++lane)
                    outputs[static_cast<std::size_t>(lane)]->set(row) =
                        integral_derivatives[static_cast<std::size_t>(lane)];
            });
        if (timing)
            packed_export_integral_seconds += elapsed_seconds(export_part_start);
    };

    auto export_active_equations_packed = [&]() {
        Term_eq** results_raw = sys_.results_shadow_view();
        std::array<Array<double>*, W> secs_arr;
        for (int lane = 0; lane < W; ++lane)
            secs_arr[static_cast<std::size_t>(lane)] = outputs[static_cast<std::size_t>(lane)];
        for (const ActiveEquationRange& active_range : state.active_equation_ranges) {
            const int equation_index = active_range.equation_index;
            int conte = active_range.operator_offset;
            std::array<int, W> pos_res_arr;
            for (int lane = 0; lane < W; ++lane)
                pos_res_arr[static_cast<std::size_t>(lane)] = active_range.row_begin;
            Clock::time_point export_part_start;
            if (timing)
                export_part_start = Clock::now();
            sys_.eq[equation_index]->export_der_lanes(
                conte, results_raw, W, secs_arr.data(), pos_res_arr.data());
            if (timing) {
                const double seconds = elapsed_seconds(export_part_start);
                packed_export_equation_seconds += seconds;
                if constexpr (W == 8) {
                    auto& timing_state = state.timing;
                    timing_state.packed_wlane8_export_eq[static_cast<std::size_t>(equation_index)] += seconds;
                    timing_state.packed_wlane8_export_calls[static_cast<std::size_t>(equation_index)] += 1;
                }
                if constexpr (W == 16) {
                    auto& timing_state = state.timing;
                    timing_state.packed_wlane16_export_eq[static_cast<std::size_t>(equation_index)] += seconds;
                    timing_state.packed_wlane16_export_calls[static_cast<std::size_t>(equation_index)] += 1;
                }
                if constexpr (W == 32) {
                    auto& timing_state = state.timing;
                    timing_state.packed_wlane32_export_eq[static_cast<std::size_t>(equation_index)] += seconds;
                    timing_state.packed_wlane32_export_calls[static_cast<std::size_t>(equation_index)] += 1;
                }
            }
        }
    };

    auto scan_sparse_output = [&](const Array<double>& output, SparseColumnEmitter& emit) {
        auto scan_range = [&](int begin, int end) {
            begin = std::max(0, begin);
            end = std::min(sys_.nbr_conditions, end);
            jacobian_column_engine_detail::for_each_selected_index_in_range(
                begin, end, selected_rows, [&](int row) {
                    const double value = output(row);
                    if (std::abs(value) > drop_tol)
                        emit(row, value);
                });
        };

        scan_range(0, sys_.neq_int);
        for (const ActiveEquationRange& active_range : state.active_equation_ranges) {
            scan_range(active_range.row_begin, active_range.row_end);
        }
    };

    try {
        Clock::time_point phase_start;
        if (timing)
            phase_start = Clock::now();
        prepare_packed_lane_derivatives();
        if (timing)
            packed_prepare_seconds += elapsed_seconds(phase_start);

        if (timing)
            phase_start = Clock::now();
        if (any_variable_domain) {
            Array<int> affected_domains(2 * W);
            affected_domains = -1;
            for (int k = 0; k < W; ++k) {
                int variable_domain_counter = 0;
                Array<int> lane_domains(2);
                lane_domains = -1;
                if (!sys_.espace.affecte_coef_to_variable_domains_lane(
                        variable_domain_counter, columns[k], k, W, lane_domains)) {
                    failure_reason = "packed variable-domain seeding found no adapted coefficient";
                    sys_.espace.restore_scalar_variable_domain_derivatives();
                    def_dep_cache.prev_was_var_domain = false;
                    restore_scalar_derivative_lanes();
                    return false;
                }
                seeds[k].is_var_domain = true;
                seeds[k].zedom = lane_domains(0);
                seeds[k].second_zedom = lane_domains(1);
                affected_domains.set(2 * k) = lane_domains(0);
                affected_domains.set(2 * k + 1) = lane_domains(1);
            }
            // Geometry/radius tangents must be materialized before any field
            // seeding or metric/definition update, matching scalar do_col_J.
            sys_.update_terms_from_variable_domains(affected_domains);
        } else {
            for (int k = 0; k < W; ++k) {
                if (!seed_column_into_lane(columns[k], k, seeds[k])) {
                    failure_reason = "filtered W-lane seeding produced no field or scalar derivative";
                    restore_scalar_derivative_lanes();
                    return false;
                }
            }
        }
        for (int k = 0; k < W; ++k) {
            mark_affected_domain(seeds[k].zedom);
            mark_affected_domain(seeds[k].second_zedom);
        }
        if (timing)
            packed_seed_seconds += elapsed_seconds(phase_start);

        if (timing)
            phase_start = Clock::now();
        if (sys_.met != nullptr) {
            for (int domain = sys_.dom_min; domain <= sys_.dom_max; ++domain)
                sys_.met->update(domain);
        }
        if (timing)
            packed_metric_seconds += elapsed_seconds(phase_start);

        bool compute_all_defs = !sys_.do_col_J_def_filter_enabled();
        for (int k = 0; k < W; ++k)
            compute_all_defs = compute_all_defs || seeds[k].is_var_double || seeds[k].is_var_domain;
        const bool establishes_base_defs = !def_dep_cache.base_defs_ready;
        if (establishes_base_defs) {
            compute_all_defs = true;
        }
        Clock::time_point definition_start;
        if (timing)
            definition_start = Clock::now();
        if (compute_all_defs) {
            if (timing)
                phase_start = Clock::now();
            for (int i = 0; i < sys_.ndef; ++i)
                compute_def_timed(i);
            if (timing)
                packed_definition_compute_seconds += elapsed_seconds(phase_start);
        } else {
            if (timing)
                phase_start = Clock::now();
            for (int i = 0; i < sys_.ndef; ++i)
                set_primary_zero_and_retain_extra_derivative_lanes(sys_.def[i]->get_res(), W);
            if (timing)
                packed_definition_clear_seconds += elapsed_seconds(phase_start);

            if (timing)
                phase_start = Clock::now();
            for (int idx : def_domain_cache.defs_global) {
                if (definition_depends_on_any_seed(idx))
                    compute_def_timed(idx);
            }
            for (int d = 0; d < ndoms; ++d) {
                if (!state.dom_affected[static_cast<std::size_t>(d)])
                    continue;
                for (int idx : def_domain_cache.defs_by_dom[static_cast<std::size_t>(d)]) {
                    if (definition_depends_on_any_seed(idx))
                        compute_def_timed(idx);
                }
            }
            if (timing)
                packed_definition_compute_seconds += elapsed_seconds(phase_start);
        }
        if (establishes_base_defs)
            def_dep_cache.base_defs_ready = true;
        if (timing)
            packed_definition_seconds += elapsed_seconds(definition_start);

        if (timing)
            phase_start = Clock::now();
        state.eq_affects.resize(static_cast<std::size_t>(sys_.neq));
        state.active_equation_ranges.clear();
        state.active_equation_ranges.reserve(static_cast<std::size_t>(sys_.neq));
        bool has_scalar_column = false;
        for (int k = 0; k < W; ++k)
            has_scalar_column = has_scalar_column || seeds[k].is_var_double;
        int operator_offset = 0;
        int row_offset = sys_.neq_int;
        for (int equation_index = 0; equation_index < sys_.neq; ++equation_index) {
            bool domain_marks_equation = has_scalar_column;
            for (int k = 0; k < W && !domain_marks_equation; ++k) {
                domain_marks_equation =
                    sys_.eq[equation_index]->take_into_account(seeds[k].zedom) ||
                    sys_.eq[equation_index]->take_into_account(seeds[k].second_zedom);
            }
            // Variable-dependency prune. Only narrow when EVERY seed carries a
            // resolved variable_id; if any packed column lacks a name->id
            // mapping (variable_id < 0) the dependency of equations on that
            // field cannot be answered by equation_dep_ids, so we must not
            // prune (stay active). This generalises the W=2 "both seeds
            // mapped" guard; for W>2 it is bit-identical to the per-seed loop
            // whenever every packed column is mapped (the only case reached on
            // the tested fixtures, enforced by the COO byte-hash gate).
            bool variable_marks_equation = true;
            if (!has_scalar_column &&
                def_dep_cache.equation_dep_ids.size() == static_cast<std::size_t>(sys_.neq)) {
                bool all_seeds_mapped = true;
                for (int k = 0; k < W; ++k)
                    all_seeds_mapped = all_seeds_mapped && (seeds[k].variable_id >= 0);
                if (all_seeds_mapped) {
                    const auto& equation_deps =
                        def_dep_cache.equation_dep_ids[static_cast<std::size_t>(equation_index)];
                    variable_marks_equation = false;
                    for (int k = 0; k < W && !variable_marks_equation; ++k)
                        variable_marks_equation =
                            std::binary_search(equation_deps.begin(), equation_deps.end(),
                                               seeds[k].variable_id);
                }
            }
            const int row_count = sys_.eq[equation_index]->get_n_cond_tot();
            const bool selection_marks_equation =
                jacobian_column_engine_detail::range_contains_selected_index(
                    row_offset, row_offset + row_count, selected_rows);
            const bool equation_is_active = domain_marks_equation &&
                                            variable_marks_equation &&
                                            selection_marks_equation;
            if (sys_.solver_runtime_config.diagnostics.timing && domain_marks_equation) {
                auto& timing_state = state.timing;
                ++timing_state.equation_domain_active;
                if (equation_is_active)
                    ++timing_state.equation_active;
                else if (!variable_marks_equation)
                    ++timing_state.equation_variable_skipped;
            }
            state.eq_affects[static_cast<std::size_t>(equation_index)] =
                equation_is_active ? 1 : 0;
            if (equation_is_active) {
                ActiveEquationRange active_range;
                active_range.equation_index = equation_index;
                active_range.operator_offset = operator_offset;
                active_range.row_begin = row_offset;
                active_range.row_end = row_offset + row_count;
                state.active_equation_ranges.push_back(active_range);
            }
            operator_offset += sys_.eq[equation_index]->n_ope;
            row_offset += row_count;
        }
        if (timing)
            packed_equation_filter_seconds += elapsed_seconds(phase_start);

        if (timing)
            phase_start = Clock::now();
        Term_eq** results_raw = sys_.results_shadow_view();
        for (const ActiveEquationRange& active_range : state.active_equation_ranges) {
            int equation_operator_offset = active_range.operator_offset;
            sys_.eq[active_range.equation_index]->apply(equation_operator_offset, results_raw);
        }
        sys_.results_shadow_sync();
        if (timing)
            packed_apply_seconds += elapsed_seconds(phase_start);

        if (timing)
            phase_start = Clock::now();
        // Pre-zero every row this tile can export/scan BEFORE any export
        // dispatch. Integral rows and active-equation ranges are exactly the
        // ranges consumed by scan_sparse_output; inactive rows may retain old
        // values because they are neither exported nor scanned for this tile.
        // This ordering is an invariant the export overrides rely on: the
        // skip-on-missing-lane fast paths in Eq_full::export_der_lanes
        // (eq_full.cpp:54), Eq_inside, and Equation::export_der_lanes_single_tau
        // write only the lanes they own and assume every other lane is already
        // zero. clear_export_outputs() must stay sequenced first.
        clear_export_outputs();
        export_integral_lanes();
        export_active_equations_packed();
        if (timing)
            packed_export_seconds += elapsed_seconds(phase_start);

        if (timing)
            phase_start = Clock::now();
        for (int k = 0; k < W; ++k)
            scan_sparse_output(*outputs[static_cast<std::size_t>(k)], emitters[static_cast<std::size_t>(k)]);
        if (timing)
            packed_scan_seconds += elapsed_seconds(phase_start);

        if (sys_.solver_runtime_config.diagnostics.timing) {
            auto& timing_state = state.timing;
            long long selected_scan_rows = 0;
            selected_scan_rows +=
                jacobian_column_engine_detail::for_each_selected_index_in_range(
                    0, sys_.neq_int, selected_rows, [](int) {});
            for (const ActiveEquationRange& active_range : state.active_equation_ranges)
                selected_scan_rows +=
                    jacobian_column_engine_detail::for_each_selected_index_in_range(
                        active_range.row_begin, active_range.row_end,
                        selected_rows, [](int) {});
            const long long scanned_rows =
                static_cast<long long>(W) * selected_scan_rows;
            timing_state.sparse_scan_rows += scanned_rows;
            timing_state.sparse_full_scan_rows +=
                static_cast<long long>(W) * static_cast<long long>(sys_.nbr_conditions);
        }

        if (timing)
            phase_start = Clock::now();
        if (any_variable_domain) {
            sys_.espace.restore_scalar_variable_domain_derivatives();
            def_dep_cache.prev_was_var_domain = false;
        }
        restore_scalar_derivative_lanes();
        if (timing) {
            packed_restore_seconds += elapsed_seconds(phase_start);
            auto& timing_state = state.timing;
            const double packed_total_seconds = elapsed_seconds(packed_total_start);
            if constexpr (W == 2) {
                ++timing_state.packed_wlane2_calls;
                timing_state.packed_wlane2_total += packed_total_seconds;
                timing_state.packed_wlane2_prepare += packed_prepare_seconds;
                timing_state.packed_wlane2_seed += packed_seed_seconds;
                timing_state.packed_wlane2_metric_update += packed_metric_seconds;
                timing_state.packed_wlane2_def_update += packed_definition_seconds;
                timing_state.packed_wlane2_def_clear += packed_definition_clear_seconds;
                timing_state.packed_wlane2_def_compute += packed_definition_compute_seconds;
                timing_state.packed_wlane2_equation_filter += packed_equation_filter_seconds;
                timing_state.packed_wlane2_apply += packed_apply_seconds;
                timing_state.packed_wlane2_export += packed_export_seconds;
                timing_state.packed_wlane2_scan += packed_scan_seconds;
                timing_state.packed_wlane2_restore += packed_restore_seconds;
            } else if constexpr (W == 4) {
                ++timing_state.packed_wlane4_calls;
                timing_state.packed_wlane4_total += packed_total_seconds;
                timing_state.packed_wlane4_prepare += packed_prepare_seconds;
                timing_state.packed_wlane4_seed += packed_seed_seconds;
                timing_state.packed_wlane4_metric_update += packed_metric_seconds;
                timing_state.packed_wlane4_def_update += packed_definition_seconds;
                timing_state.packed_wlane4_def_clear += packed_definition_clear_seconds;
                timing_state.packed_wlane4_def_compute += packed_definition_compute_seconds;
                timing_state.packed_wlane4_equation_filter += packed_equation_filter_seconds;
                timing_state.packed_wlane4_apply += packed_apply_seconds;
                timing_state.packed_wlane4_export += packed_export_seconds;
                timing_state.packed_wlane4_scan += packed_scan_seconds;
                timing_state.packed_wlane4_restore += packed_restore_seconds;
            } else if constexpr (W == 8) {
                ++timing_state.packed_wlane8_calls;
                timing_state.packed_wlane8_total += packed_total_seconds;
                timing_state.packed_wlane8_prepare += packed_prepare_seconds;
                timing_state.packed_wlane8_seed += packed_seed_seconds;
                timing_state.packed_wlane8_metric_update += packed_metric_seconds;
                timing_state.packed_wlane8_def_update += packed_definition_seconds;
                timing_state.packed_wlane8_def_clear += packed_definition_clear_seconds;
                timing_state.packed_wlane8_def_compute += packed_definition_compute_seconds;
                timing_state.packed_wlane8_equation_filter += packed_equation_filter_seconds;
                timing_state.packed_wlane8_apply += packed_apply_seconds;
                timing_state.packed_wlane8_export += packed_export_seconds;
                timing_state.packed_wlane8_export_clear += packed_export_clear_seconds;
                timing_state.packed_wlane8_export_integral += packed_export_integral_seconds;
                timing_state.packed_wlane8_export_equations += packed_export_equation_seconds;
                timing_state.packed_wlane8_scan += packed_scan_seconds;
                timing_state.packed_wlane8_restore += packed_restore_seconds;
            } else if constexpr (W == 16) {
                ++timing_state.packed_wlane16_calls;
                timing_state.packed_wlane16_total += packed_total_seconds;
                timing_state.packed_wlane16_prepare += packed_prepare_seconds;
                timing_state.packed_wlane16_seed += packed_seed_seconds;
                timing_state.packed_wlane16_metric_update += packed_metric_seconds;
                timing_state.packed_wlane16_def_update += packed_definition_seconds;
                timing_state.packed_wlane16_def_clear += packed_definition_clear_seconds;
                timing_state.packed_wlane16_def_compute += packed_definition_compute_seconds;
                timing_state.packed_wlane16_equation_filter += packed_equation_filter_seconds;
                timing_state.packed_wlane16_apply += packed_apply_seconds;
                timing_state.packed_wlane16_export += packed_export_seconds;
                timing_state.packed_wlane16_export_clear += packed_export_clear_seconds;
                timing_state.packed_wlane16_export_integral += packed_export_integral_seconds;
                timing_state.packed_wlane16_export_equations += packed_export_equation_seconds;
                timing_state.packed_wlane16_scan += packed_scan_seconds;
                timing_state.packed_wlane16_restore += packed_restore_seconds;
            } else if constexpr (W == 32) {
                ++timing_state.packed_wlane32_calls;
                timing_state.packed_wlane32_total += packed_total_seconds;
                timing_state.packed_wlane32_prepare += packed_prepare_seconds;
                timing_state.packed_wlane32_seed += packed_seed_seconds;
                timing_state.packed_wlane32_metric_update += packed_metric_seconds;
                timing_state.packed_wlane32_def_update += packed_definition_seconds;
                timing_state.packed_wlane32_def_clear += packed_definition_clear_seconds;
                timing_state.packed_wlane32_def_compute += packed_definition_compute_seconds;
                timing_state.packed_wlane32_equation_filter += packed_equation_filter_seconds;
                timing_state.packed_wlane32_apply += packed_apply_seconds;
                timing_state.packed_wlane32_export += packed_export_seconds;
                timing_state.packed_wlane32_export_clear += packed_export_clear_seconds;
                timing_state.packed_wlane32_export_integral += packed_export_integral_seconds;
                timing_state.packed_wlane32_export_equations += packed_export_equation_seconds;
                timing_state.packed_wlane32_scan += packed_scan_seconds;
                timing_state.packed_wlane32_restore += packed_restore_seconds;
            }
        }
        return true;
    } catch (...) {
        const std::exception_ptr original_exception = std::current_exception();
        if (any_variable_domain) {
            def_dep_cache.prev_was_var_domain = false;
            try {
                sys_.espace.restore_scalar_variable_domain_derivatives();
            } catch (...) {
            }
        }
        try {
            restore_scalar_derivative_lanes();
        } catch (...) {
        }
        std::rethrow_exception(original_exception);
    }
}

bool JacobianColumnEngine::compute_packed_wlane2_columns_sparse(int first_column,
                                                                int second_column,
                                                                double drop_tol,
                                                                SparseColumnEmitter emit_first,
                                                                SparseColumnEmitter emit_second,
                                                                std::string& failure_reason,
                                                                JacobianSelectedRows selected_rows)
{
    const std::array<int, 2> columns{ first_column, second_column };
    std::array<SparseColumnEmitter, 2> emitters{ std::move(emit_first), std::move(emit_second) };
    return compute_packed_wlane2_columns_sparse(
        default_workspace_, columns, drop_tol, emitters, failure_reason,
        selected_rows);
}

bool JacobianColumnEngine::compute_packed_wlane2_columns_sparse(
    Workspace& workspace, PackedJacobianColumns<2> columns, double drop_tol,
    PackedSparseColumnEmitters<2> emitters, std::string& failure_reason,
    JacobianSelectedRows selected_rows)
{
    return compute_packed_wlaneN_columns_sparse<2>(
        workspace, columns, drop_tol, emitters, failure_reason, selected_rows);
}

bool JacobianColumnEngine::compute_packed_wlane4_columns_sparse(
    const std::array<int, 4>& columns,
    double drop_tol,
    std::array<SparseColumnEmitter, 4>& emitters,
    std::string& failure_reason,
    JacobianSelectedRows selected_rows)
{
    return compute_packed_wlane4_columns_sparse(
        default_workspace_, columns, drop_tol, emitters, failure_reason,
        selected_rows);
}

bool JacobianColumnEngine::compute_packed_wlane4_columns_sparse(
    Workspace& workspace, PackedJacobianColumns<4> columns, double drop_tol,
    PackedSparseColumnEmitters<4> emitters, std::string& failure_reason,
    JacobianSelectedRows selected_rows)
{
    return compute_packed_wlaneN_columns_sparse<4>(
        workspace, columns, drop_tol, emitters, failure_reason, selected_rows);
}

bool JacobianColumnEngine::compute_packed_wlane8_columns_sparse(
    const std::array<int, 8>& columns,
    double drop_tol,
    std::array<SparseColumnEmitter, 8>& emitters,
    std::string& failure_reason,
    JacobianSelectedRows selected_rows)
{
    return compute_packed_wlane8_columns_sparse(
        default_workspace_, columns, drop_tol, emitters, failure_reason,
        selected_rows);
}

bool JacobianColumnEngine::compute_packed_wlane8_columns_sparse(
    Workspace& workspace, PackedJacobianColumns<8> columns, double drop_tol,
    PackedSparseColumnEmitters<8> emitters, std::string& failure_reason,
    JacobianSelectedRows selected_rows)
{
    return compute_packed_wlaneN_columns_sparse<8>(
        workspace, columns, drop_tol, emitters, failure_reason, selected_rows);
}

bool JacobianColumnEngine::compute_packed_wlane16_columns_sparse(
    const std::array<int, 16>& columns,
    double drop_tol,
    std::array<SparseColumnEmitter, 16>& emitters,
    std::string& failure_reason,
    JacobianSelectedRows selected_rows)
{
    return compute_packed_wlane16_columns_sparse(
        default_workspace_, columns, drop_tol, emitters, failure_reason,
        selected_rows);
}

bool JacobianColumnEngine::compute_packed_wlane16_columns_sparse(
    Workspace& workspace, PackedJacobianColumns<16> columns, double drop_tol,
    PackedSparseColumnEmitters<16> emitters, std::string& failure_reason,
    JacobianSelectedRows selected_rows)
{
    return compute_packed_wlaneN_columns_sparse<16>(
        workspace, columns, drop_tol, emitters, failure_reason, selected_rows);
}

bool JacobianColumnEngine::compute_packed_wlane32_columns_sparse(
    const std::array<int, 32>& columns,
    double drop_tol,
    std::array<SparseColumnEmitter, 32>& emitters,
    std::string& failure_reason,
    JacobianSelectedRows selected_rows)
{
    return compute_packed_wlane32_columns_sparse(
        default_workspace_, columns, drop_tol, emitters, failure_reason,
        selected_rows);
}

bool JacobianColumnEngine::compute_packed_wlane32_columns_sparse(
    Workspace& workspace, PackedJacobianColumns<32> columns, double drop_tol,
    PackedSparseColumnEmitters<32> emitters, std::string& failure_reason,
    JacobianSelectedRows selected_rows)
{
    return compute_packed_wlaneN_columns_sparse<32>(
        workspace, columns, drop_tol, emitters, failure_reason, selected_rows);
}

void JacobianColumnEngine::dump_profile() const
{
    dump_profile(default_workspace_);
}

void JacobianColumnEngine::dump_profile(Workspace& workspace) const
{
    if (!sys_.solver_runtime_config.diagnostics.timing) {
        return;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
        return;
    }
    auto& timing_state = workspace.state_.timing;
    double total = timing_state.update_terms + timing_state.var_double + timing_state.fields +
                   timing_state.eq_affects + timing_state.apply + timing_state.export_der +
                   timing_state.def_update + timing_state.metric_update;
    if (total <= 0.0) {
        total = 1.0;
    }
    std::cout << "do_col_J totals (" << timing_state.calls << " calls, " << total << "s total):" << std::endl;
    if (timing_state.var_domain_calls > 0) {
        std::cout << "- variable_domain: calls=" << timing_state.var_domain_calls
                  << " total=" << timing_state.var_domain_total << "s"
                  << " avg/column="
                  << (timing_state.var_domain_total /
                      static_cast<double>(timing_state.var_domain_calls))
                  << "s (" << (timing_state.var_domain_total / total * 100.0)
                  << "% of scalar do_col_J wall)" << std::endl;
    }
    std::cout << "- apply:         " << timing_state.apply << "s  (" << (timing_state.apply / total * 100.0)
              << "%)" << std::endl;
    std::cout << "- eq_affects:    " << timing_state.eq_affects << "s  ("
              << (timing_state.eq_affects / total * 100.0) << "%)" << std::endl;
    std::cout << "- update_terms:  " << timing_state.update_terms << "s  ("
              << (timing_state.update_terms / total * 100.0) << "%)" << std::endl;
    std::cout << "- def_update:    " << timing_state.def_update << "s  ("
              << (timing_state.def_update / total * 100.0) << "%)" << std::endl;
    std::cout << "  * def_clear:   " << timing_state.def_clear << "s" << std::endl;
    std::cout << "  * def_compute: " << timing_state.def_compute << "s" << std::endl;
    std::cout << "- metric_update: " << timing_state.metric_update << "s  ("
              << (timing_state.metric_update / total * 100.0) << "%)" << std::endl;
    std::cout << "- export:        " << timing_state.export_der << "s  ("
              << (timing_state.export_der / total * 100.0) << "%)" << std::endl;
    if (timing_state.sparse_full_scan_rows > 0) {
        const double scan_frac = static_cast<double>(timing_state.sparse_scan_rows) /
                                 static_cast<double>(timing_state.sparse_full_scan_rows);
        std::cout << "- sparse_scan:   rows=" << timing_state.sparse_scan_rows
                  << " full_rows=" << timing_state.sparse_full_scan_rows
                  << " (" << (100.0 * scan_frac) << "%)" << std::endl;
    }
    if (timing_state.equation_domain_active > 0) {
        const double skipped_frac =
            static_cast<double>(timing_state.equation_variable_skipped) /
            static_cast<double>(timing_state.equation_domain_active);
        std::cout << "- eq_filter:     domain_active=" << timing_state.equation_domain_active
                  << " active=" << timing_state.equation_active
                  << " variable_skipped=" << timing_state.equation_variable_skipped
                  << " (" << (100.0 * skipped_frac) << "%)" << std::endl;
    }
    if (timing_state.packed_wlane2_calls > 0) {
        const double packed_total =
            (timing_state.packed_wlane2_total > 0.0)
                ? timing_state.packed_wlane2_total
                : 1.0;
        const auto print_packed_stage = [&](const char* label, double seconds) {
            std::cout << "  * " << label << ": " << seconds << "s  ("
                      << (seconds / packed_total * 100.0) << "%)" << std::endl;
        };
        std::cout << "- packed_wlane2: pairs=" << timing_state.packed_wlane2_calls
                  << " total=" << timing_state.packed_wlane2_total << "s"
                  << " avg/pair="
                  << (timing_state.packed_wlane2_total /
                      static_cast<double>(timing_state.packed_wlane2_calls))
                  << "s" << std::endl;
        print_packed_stage("prepare", timing_state.packed_wlane2_prepare);
        print_packed_stage("seed", timing_state.packed_wlane2_seed);
        print_packed_stage("metric_update", timing_state.packed_wlane2_metric_update);
        print_packed_stage("def_update", timing_state.packed_wlane2_def_update);
        print_packed_stage("def_clear", timing_state.packed_wlane2_def_clear);
        print_packed_stage("def_compute", timing_state.packed_wlane2_def_compute);
        print_packed_stage("eq_filter", timing_state.packed_wlane2_equation_filter);
        print_packed_stage("apply", timing_state.packed_wlane2_apply);
        print_packed_stage("export", timing_state.packed_wlane2_export);
        print_packed_stage("scan", timing_state.packed_wlane2_scan);
        print_packed_stage("restore", timing_state.packed_wlane2_restore);
    }
    if (timing_state.packed_wlane4_calls > 0) {
        const double packed_total =
            (timing_state.packed_wlane4_total > 0.0)
                ? timing_state.packed_wlane4_total
                : 1.0;
        const auto print_packed_stage = [&](const char* label, double seconds) {
            std::cout << "  * " << label << ": " << seconds << "s  ("
                      << (seconds / packed_total * 100.0) << "%)" << std::endl;
        };
        std::cout << "- packed_wlane4: quartets=" << timing_state.packed_wlane4_calls
                  << " total=" << timing_state.packed_wlane4_total << "s"
                  << " avg/quartet="
                  << (timing_state.packed_wlane4_total /
                      static_cast<double>(timing_state.packed_wlane4_calls))
                  << "s" << std::endl;
        print_packed_stage("prepare", timing_state.packed_wlane4_prepare);
        print_packed_stage("seed", timing_state.packed_wlane4_seed);
        print_packed_stage("metric_update", timing_state.packed_wlane4_metric_update);
        print_packed_stage("def_update", timing_state.packed_wlane4_def_update);
        print_packed_stage("def_clear", timing_state.packed_wlane4_def_clear);
        print_packed_stage("def_compute", timing_state.packed_wlane4_def_compute);
        print_packed_stage("eq_filter", timing_state.packed_wlane4_equation_filter);
        print_packed_stage("apply", timing_state.packed_wlane4_apply);
        print_packed_stage("export", timing_state.packed_wlane4_export);
        print_packed_stage("scan", timing_state.packed_wlane4_scan);
        print_packed_stage("restore", timing_state.packed_wlane4_restore);
    }
    if (timing_state.packed_wlane8_calls > 0) {
        const double packed_total =
            (timing_state.packed_wlane8_total > 0.0)
                ? timing_state.packed_wlane8_total
                : 1.0;
        const auto print_packed_stage = [&](const char* label, double seconds) {
            std::cout << "  * " << label << ": " << seconds << "s  ("
                      << (seconds / packed_total * 100.0) << "%)" << std::endl;
        };
        std::cout << "- packed_wlane8: octets=" << timing_state.packed_wlane8_calls
                  << " total=" << timing_state.packed_wlane8_total << "s"
                  << " avg/octet="
                  << (timing_state.packed_wlane8_total /
                      static_cast<double>(timing_state.packed_wlane8_calls))
                  << "s" << std::endl;
        print_packed_stage("prepare", timing_state.packed_wlane8_prepare);
        print_packed_stage("seed", timing_state.packed_wlane8_seed);
        print_packed_stage("metric_update", timing_state.packed_wlane8_metric_update);
        print_packed_stage("def_update", timing_state.packed_wlane8_def_update);
        print_packed_stage("def_clear", timing_state.packed_wlane8_def_clear);
        print_packed_stage("def_compute", timing_state.packed_wlane8_def_compute);
        print_packed_stage("eq_filter", timing_state.packed_wlane8_equation_filter);
        print_packed_stage("apply", timing_state.packed_wlane8_apply);
        print_packed_stage("export", timing_state.packed_wlane8_export);
        const double export_total =
            (timing_state.packed_wlane8_export > 0.0)
                ? timing_state.packed_wlane8_export
                : 1.0;
        const auto print_export_stage = [&](const char* label, double seconds) {
            std::cout << "    - " << label << ": " << seconds << "s  ("
                      << (seconds / export_total * 100.0) << "% of W8 export, "
                      << (seconds / packed_total * 100.0) << "% of W8 total)" << std::endl;
        };
        print_export_stage("dense_clear", timing_state.packed_wlane8_export_clear);
        print_export_stage("integral_rows", timing_state.packed_wlane8_export_integral);
        print_export_stage("equation_export", timing_state.packed_wlane8_export_equations);
        print_packed_stage("scan", timing_state.packed_wlane8_scan);
        print_packed_stage("restore", timing_state.packed_wlane8_restore);
        if (!timing_state.packed_wlane8_export_eq.empty()) {
            std::vector<int> export_indices(timing_state.packed_wlane8_export_eq.size());
            for (std::size_t i = 0; i < export_indices.size(); ++i)
                export_indices[i] = static_cast<int>(i);
            std::sort(export_indices.begin(), export_indices.end(), [&](int a, int b) {
                return timing_state.packed_wlane8_export_eq[static_cast<std::size_t>(a)] >
                       timing_state.packed_wlane8_export_eq[static_cast<std::size_t>(b)];
            });
            const int top_export = std::min(10, static_cast<int>(export_indices.size()));
            std::cout << "Top W8 export_der() costs:" << std::endl;
            for (int k = 0; k < top_export; ++k) {
                const int i = export_indices[static_cast<std::size_t>(k)];
                const double seconds = timing_state.packed_wlane8_export_eq[static_cast<std::size_t>(i)];
                if (seconds <= 0.0)
                    break;
                const int calls = timing_state.packed_wlane8_export_calls[static_cast<std::size_t>(i)];
                const double avg = (calls > 0) ? seconds / static_cast<double>(calls) : 0.0;
                const auto& [name, dom, bc] = sys_.eq_list[static_cast<std::size_t>(i)];
                std::cout << "- [" << i << "] " << name << " dom=" << dom << " bc=" << bc;
                if (i < static_cast<int>(sys_.eq_column_attachments.size())) {
                    const auto& attachment = sys_.eq_column_attachments[static_cast<std::size_t>(i)];
                    if (!attachment.owner_var_name.empty())
                        std::cout << " owner=" << attachment.owner_var_name;
                }
                std::cout << " rows=" << sys_.eq[i]->get_n_cond_tot()
                          << " time=" << seconds << "s"
                          << " calls=" << calls
                          << " avg=" << avg << "s" << std::endl;
            }
        }
        if (!timing_state.packed_wlane8_export_eq.empty()) {
            std::unordered_map<std::string, std::pair<double, long long>> by_kind;
            for (int i = 0; i < sys_.neq; ++i) {
                const double secs = timing_state.packed_wlane8_export_eq[static_cast<std::size_t>(i)];
                const int    calls = timing_state.packed_wlane8_export_calls[static_cast<std::size_t>(i)];
                auto& slot = by_kind[equation_kind_name(sys_.eq[i].get())];
                slot.first  += secs;
                slot.second += calls;
            }
            std::vector<std::pair<std::string, std::pair<double, long long>>> sorted_kinds(
                by_kind.begin(), by_kind.end());
            std::sort(sorted_kinds.begin(), sorted_kinds.end(),
                [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
            const double w8_export = (timing_state.packed_wlane8_export > 0.0)
                ? timing_state.packed_wlane8_export : 1.0;
            std::cout << "W8 export by equation kind (calls = export_der invocations, 8 per octet x active eq):"
                      << std::endl;
            for (const auto& [kind, agg] : sorted_kinds) {
                const double  secs  = agg.first;
                const long long calls = agg.second;
                if (secs <= 0.0 && calls == 0) continue;
                const double avg = (calls > 0) ? secs / static_cast<double>(calls) : 0.0;
                std::cout << "  - " << kind
                          << ": " << secs << "s  ("
                          << (secs / w8_export * 100.0) << "% of W8 export)"
                          << "  calls=" << calls
                          << "  avg=" << avg << "s" << std::endl;
            }
        }
    }
    if (timing_state.packed_wlane16_calls > 0) {
        const double packed_total =
            (timing_state.packed_wlane16_total > 0.0)
                ? timing_state.packed_wlane16_total
                : 1.0;
        const auto print_packed_stage = [&](const char* label, double seconds) {
            std::cout << "  * " << label << ": " << seconds << "s  ("
                      << (seconds / packed_total * 100.0) << "%)" << std::endl;
        };
        std::cout << "- packed_wlane16: hexadectets=" << timing_state.packed_wlane16_calls
                  << " total=" << timing_state.packed_wlane16_total << "s"
                  << " avg/hexadectet="
                  << (timing_state.packed_wlane16_total /
                      static_cast<double>(timing_state.packed_wlane16_calls))
                  << "s" << std::endl;
        print_packed_stage("prepare", timing_state.packed_wlane16_prepare);
        print_packed_stage("seed", timing_state.packed_wlane16_seed);
        print_packed_stage("metric_update", timing_state.packed_wlane16_metric_update);
        print_packed_stage("def_update", timing_state.packed_wlane16_def_update);
        print_packed_stage("def_clear", timing_state.packed_wlane16_def_clear);
        print_packed_stage("def_compute", timing_state.packed_wlane16_def_compute);
        print_packed_stage("eq_filter", timing_state.packed_wlane16_equation_filter);
        print_packed_stage("apply", timing_state.packed_wlane16_apply);
        print_packed_stage("export", timing_state.packed_wlane16_export);
        print_packed_stage("scan", timing_state.packed_wlane16_scan);
        print_packed_stage("restore", timing_state.packed_wlane16_restore);
    }
    if (timing_state.packed_wlane32_calls > 0) {
        const double ref =
            (timing_state.packed_wlane32_total > 0.0)
                ? timing_state.packed_wlane32_total
                : 1.0;
        const auto print_packed_stage = [&](const char* name, double secs) {
            std::cout << "  " << name << ": " << secs << "s  ("
                      << (secs / ref * 100.0) << "%)\n";
        };
        std::cout << "- packed_wlane32: triacontadyads=" << timing_state.packed_wlane32_calls
                  << " total=" << timing_state.packed_wlane32_total << "s"
                  << " avg="
                  << (timing_state.packed_wlane32_total /
                      static_cast<double>(timing_state.packed_wlane32_calls))
                  << "s/group\n";
        print_packed_stage("prepare", timing_state.packed_wlane32_prepare);
        print_packed_stage("seed", timing_state.packed_wlane32_seed);
        print_packed_stage("metric_update", timing_state.packed_wlane32_metric_update);
        print_packed_stage("def_update", timing_state.packed_wlane32_def_update);
        print_packed_stage("def_clear", timing_state.packed_wlane32_def_clear);
        print_packed_stage("def_compute", timing_state.packed_wlane32_def_compute);
        print_packed_stage("eq_filter", timing_state.packed_wlane32_equation_filter);
        print_packed_stage("apply", timing_state.packed_wlane32_apply);
        print_packed_stage("export", timing_state.packed_wlane32_export);
        print_packed_stage("scan", timing_state.packed_wlane32_scan);
        print_packed_stage("restore", timing_state.packed_wlane32_restore);
    }
    std::cout << "- fields:        " << timing_state.fields << "s  (" << (timing_state.fields / total * 100.0)
              << "%)" << std::endl;
    std::cout << "- var_double:    " << timing_state.var_double << "s  ("
              << (timing_state.var_double / total * 100.0) << "%)" << std::endl;
    long long total_def_calls = 0;
    for (std::size_t i = 0; i < timing_state.def_calls.size(); ++i) {
        total_def_calls += static_cast<long long>(timing_state.def_calls[i]);
    }
    if (timing_state.calls > 0 && sys_.ndef > 0) {
        const double avg_defs_per_col =
            static_cast<double>(total_def_calls) / static_cast<double>(timing_state.calls);
        const double pct_defs = 100.0 * avg_defs_per_col / static_cast<double>(sys_.ndef);
        std::cout << "- def_calls:     total=" << total_def_calls << " avg/do_col_J=" << avg_defs_per_col << " ("
                  << pct_defs << "% of ndef=" << sys_.ndef << ")" << std::endl;
    }
    // DEF_COMPUTE_PER_DEF=1: detailed per-def def_compute table covering
    // the packed W-lane paths (which the always-on "Top def() costs" block below
    // never reaches in production). Percentages are taken against the summed
    // def_compute wall across the scalar fallback and all packed lanes, so the
    // ranking is directly attributable to the J-build def_compute floor.
    if (def_compute_per_def_enabled() && !timing_state.def_eq.empty()) {
        const double summed_def_compute =
            timing_state.def_compute +
            timing_state.packed_wlane2_def_compute +
            timing_state.packed_wlane4_def_compute +
            timing_state.packed_wlane8_def_compute;
        const double def_compute_denom = (summed_def_compute > 0.0) ? summed_def_compute : 1.0;
        double accounted_def_seconds = 0.0;
        for (double secs : timing_state.def_eq)
            accounted_def_seconds += secs;
        std::vector<int> per_def_idx(timing_state.def_eq.size());
        for (std::size_t i = 0; i < per_def_idx.size(); ++i)
            per_def_idx[i] = static_cast<int>(i);
        std::sort(per_def_idx.begin(), per_def_idx.end(), [&](int a, int b) {
            return timing_state.def_eq[static_cast<std::size_t>(a)] >
                   timing_state.def_eq[static_cast<std::size_t>(b)];
        });
        const int per_def_top = std::min(15, static_cast<int>(per_def_idx.size()));
        std::cout << "Per-def def_compute (DEF_COMPUTE_PER_DEF): summed_def_compute="
                  << summed_def_compute << "s"
                  << " (scalar=" << timing_state.def_compute
                  << " w2=" << timing_state.packed_wlane2_def_compute
                  << " w4=" << timing_state.packed_wlane4_def_compute
                  << " w8=" << timing_state.packed_wlane8_def_compute << ")"
                  << " accounted=" << accounted_def_seconds << "s"
                  << " (" << (accounted_def_seconds / def_compute_denom * 100.0) << "% of def_compute)"
                  << std::endl;
        for (int k = 0; k < per_def_top; ++k) {
            const int i = per_def_idx[static_cast<std::size_t>(k)];
            const double secs = timing_state.def_eq[static_cast<std::size_t>(i)];
            const int calls = timing_state.def_calls[static_cast<std::size_t>(i)];
            const double avg = (calls > 0) ? (secs / static_cast<double>(calls)) : 0.0;
            const char* name = (i < static_cast<int>(sys_.names_def.size()) &&
                                !sys_.names_def[static_cast<std::size_t>(i)].empty())
                                   ? sys_.names_def[static_cast<std::size_t>(i)].c_str()
                                   : "<unnamed_def>";
            std::cout << "- [" << i << "] " << name
                      << " dom=" << sys_.def[i]->get_dom()
                      << " calls=" << calls
                      << " time=" << secs << "s"
                      << " (" << (secs / def_compute_denom * 100.0) << "% of def_compute)"
                      << " avg=" << avg << "s" << std::endl;
        }
    }
    if (!timing_state.def_eq.empty()) {
        std::vector<int> didx(timing_state.def_eq.size());
        for (std::size_t i = 0; i < didx.size(); ++i) {
            didx[i] = static_cast<int>(i);
        }
        std::sort(didx.begin(), didx.end(), [&](int a, int b) {
            return timing_state.def_eq[static_cast<std::size_t>(a)] >
                   timing_state.def_eq[static_cast<std::size_t>(b)];
        });
        int topd = std::min(10, static_cast<int>(didx.size()));
        std::cout << "Top def() costs:" << std::endl;
        for (int k = 0; k < topd; ++k) {
            int i = didx[static_cast<std::size_t>(k)];
            double t = timing_state.def_eq[static_cast<std::size_t>(i)];
            int calls = timing_state.def_calls[static_cast<std::size_t>(i)];
            double avg = (calls > 0) ? (t / static_cast<double>(calls)) : 0.0;
            const char* name = !sys_.names_def[i].empty() ? sys_.names_def[i].c_str() : "<unnamed_def>";
            std::cout << "- [" << i << "] " << name << " dom=" << sys_.def[i]->get_dom() << " time=" << t << "s"
                      << " calls=" << calls << " avg=" << avg << "s" << std::endl;
        }
    }
    if (!timing_state.apply_eq.empty()) {
        std::vector<int> idx(timing_state.apply_eq.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            idx[i] = static_cast<int>(i);
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return timing_state.apply_eq[static_cast<std::size_t>(a)] >
                   timing_state.apply_eq[static_cast<std::size_t>(b)];
        });
        int top = std::min(10, static_cast<int>(idx.size()));
        std::cout << "Top apply() costs:" << std::endl;
        for (int k = 0; k < top; ++k) {
            int i = idx[static_cast<std::size_t>(k)];
            const auto& [name, dom, bc] = sys_.eq_list[static_cast<std::size_t>(i)];
            double t = timing_state.apply_eq[static_cast<std::size_t>(i)];
            int calls = timing_state.apply_calls[static_cast<std::size_t>(i)];
            std::cout << "- [" << i << "] " << name << " dom=" << dom << " bc=" << bc << " time=" << t << "s"
                      << " calls=" << calls << std::endl;
        }
        // apply-by-kind census: attribute the apply() wall across concrete
        // Equation subclasses so the dominant apply kernel can be confirmed
        // before any block-batching work targets a specific operator family.
        if (sys_.neq > 0 &&
            static_cast<int>(timing_state.apply_eq.size()) >= sys_.neq &&
            static_cast<int>(timing_state.apply_calls.size()) >= sys_.neq) {
            std::unordered_map<std::string, std::pair<double, long long>> by_kind;
            double apply_total = 0.0;
            for (int i = 0; i < sys_.neq; ++i) {
                const double secs = timing_state.apply_eq[static_cast<std::size_t>(i)];
                const int calls = timing_state.apply_calls[static_cast<std::size_t>(i)];
                auto& slot = by_kind[equation_kind_name(sys_.eq[i].get())];
                slot.first += secs;
                slot.second += calls;
                apply_total += secs;
            }
            std::vector<std::pair<std::string, std::pair<double, long long>>> sorted_kinds(
                by_kind.begin(), by_kind.end());
            std::sort(sorted_kinds.begin(), sorted_kinds.end(),
                [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
            const double denom = (apply_total > 0.0) ? apply_total : 1.0;
            std::cout << "apply() by equation kind (share of summed per-equation apply wall):" << std::endl;
            for (const auto& [kind, agg] : sorted_kinds) {
                const double secs = agg.first;
                const long long calls = agg.second;
                if (secs <= 0.0 && calls == 0) continue;
                const double avg = (calls > 0) ? secs / static_cast<double>(calls) : 0.0;
                std::cout << "  - " << kind << ": " << secs << "s  ("
                          << (secs / denom * 100.0) << "% of apply)"
                          << "  calls=" << calls << "  avg=" << avg << "s" << std::endl;
            }
        }
    }

    timing_state.reset_totals();
    ope_der_dispatch_census_dump();
    ope_der_dispatch_census_reset();
}

} // namespace Kadath
