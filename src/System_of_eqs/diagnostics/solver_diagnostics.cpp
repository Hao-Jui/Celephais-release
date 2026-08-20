// Diagnostic / verification utilities for the Jacobian column path.
//
// Split out of solver.cpp: row/column classification, packed W-lane parity
// oracles, and primal/tangent + cache-RSS measurement helpers. These are used
// only by the unit tests (tests/unit/test_system_of_eqs_row_classification,
// test_jacobian_column_engine) and by the diagnostic branches in
// src_par/jacobian_assembler.cpp. None of them is on the Newton hot path; the
// production assembly entry points (do_col_J, do_JX, sec_member, the
// do_cols_J_wlane*_sparse shims) remain in solver.cpp.
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include "For_Kadath/Metric/metric.hpp"
#include "mpi.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <set>
#include <string_view>
#include <sys/resource.h>
#include <vector>
#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace Kadath
{
    namespace
    {
        System_of_eqs::RowTaxonomy row_taxonomy_for_equation(const Equation* equation)
        {
            if (dynamic_cast<const Eq_inside*>(equation) != nullptr ||
                dynamic_cast<const Eq_order*>(equation) != nullptr ||
                dynamic_cast<const Eq_order_array*>(equation) != nullptr ||
                dynamic_cast<const Eq_full*>(equation) != nullptr ||
                dynamic_cast<const Eq_one_side*>(equation) != nullptr ||
                dynamic_cast<const Eq_vel_pot*>(equation) != nullptr) {
                return System_of_eqs::RowTaxonomy::Vol;
            }
            if (dynamic_cast<const Eq_bc*>(equation) != nullptr ||
                dynamic_cast<const Eq_bc_exception*>(equation) != nullptr ||
                dynamic_cast<const Eq_bc_order_array*>(equation) != nullptr) {
                return System_of_eqs::RowTaxonomy::TauBc;
            }
            if (dynamic_cast<const Eq_matching*>(equation) != nullptr ||
                dynamic_cast<const Eq_matching_one_side*>(equation) != nullptr ||
                dynamic_cast<const Eq_matching_non_std*>(equation) != nullptr ||
                dynamic_cast<const Eq_matching_import*>(equation) != nullptr ||
                dynamic_cast<const Eq_matching_order_array*>(equation) != nullptr ||
                dynamic_cast<const Eq_matching_exception*>(equation) != nullptr) {
                return System_of_eqs::RowTaxonomy::TauMatch;
            }
            if (dynamic_cast<const Eq_first_integral*>(equation) != nullptr) {
                return System_of_eqs::RowTaxonomy::GlobalInt;
            }
            return System_of_eqs::RowTaxonomy::Unknown;
        }

        System_of_eqs::RowClass legacy_row_class(System_of_eqs::RowTaxonomy taxonomy)
        {
            switch (taxonomy) {
                case System_of_eqs::RowTaxonomy::Vol:
                    return System_of_eqs::RowClass::Galerkin;
                case System_of_eqs::RowTaxonomy::GlobalInt:
                    return System_of_eqs::RowClass::FirstIntegral;
                case System_of_eqs::RowTaxonomy::TauBc:
                case System_of_eqs::RowTaxonomy::TauMatch:
                case System_of_eqs::RowTaxonomy::Unknown:
                    return System_of_eqs::RowClass::Constraint;
            }
            return System_of_eqs::RowClass::Constraint;
        }

        std::string equation_type_name(const Equation* equation)
        {
            if (dynamic_cast<const Eq_inside*>(equation) != nullptr) return "Eq_inside";
            if (dynamic_cast<const Eq_order*>(equation) != nullptr) return "Eq_order";
            if (dynamic_cast<const Eq_order_array*>(equation) != nullptr) return "Eq_order_array";
            if (dynamic_cast<const Eq_full*>(equation) != nullptr) return "Eq_full";
            if (dynamic_cast<const Eq_one_side*>(equation) != nullptr) return "Eq_one_side";
            if (dynamic_cast<const Eq_vel_pot*>(equation) != nullptr) return "Eq_vel_pot";
            if (dynamic_cast<const Eq_bc*>(equation) != nullptr) return "Eq_bc";
            if (dynamic_cast<const Eq_bc_exception*>(equation) != nullptr) return "Eq_bc_exception";
            if (dynamic_cast<const Eq_bc_order_array*>(equation) != nullptr) return "Eq_bc_order_array";
            if (dynamic_cast<const Eq_matching*>(equation) != nullptr) return "Eq_matching";
            if (dynamic_cast<const Eq_matching_one_side*>(equation) != nullptr) return "Eq_matching_one_side";
            if (dynamic_cast<const Eq_matching_non_std*>(equation) != nullptr) return "Eq_matching_non_std";
            if (dynamic_cast<const Eq_matching_import*>(equation) != nullptr) return "Eq_matching_import";
            if (dynamic_cast<const Eq_matching_order_array*>(equation) != nullptr) return "Eq_matching_order_array";
            if (dynamic_cast<const Eq_matching_exception*>(equation) != nullptr) return "Eq_matching_exception";
            if (dynamic_cast<const Eq_first_integral*>(equation) != nullptr) return "Eq_first_integral";
            return "unknown";
        }

        int matching_other_domain(const Equation* equation)
        {
            if (const auto* eqm = dynamic_cast<const Eq_matching*>(equation)) return eqm->other_dom;
            if (const auto* eqm = dynamic_cast<const Eq_matching_one_side*>(equation)) return eqm->other_dom;
            if (const auto* eqm = dynamic_cast<const Eq_matching_order_array*>(equation)) return eqm->other_dom;
            if (const auto* eqm = dynamic_cast<const Eq_matching_exception*>(equation)) return eqm->other_dom;
            return -1;
        }

        const char* column_class_label(ColumnClass column_class)
        {
            switch (column_class) {
                case ColumnClass::FieldUnknown: return "FieldUnknown";
                case ColumnClass::FieldInterior: return "FieldInterior";
                case ColumnClass::FieldBoundary: return "FieldBoundary";
                case ColumnClass::FieldInteriorVol: return "FieldInteriorVol";
                case ColumnClass::FieldBoundaryTau: return "FieldBoundaryTau";
                case ColumnClass::FieldOuterShellTau: return "FieldOuterShellTau";
                case ColumnClass::FieldMatching: return "FieldMatching";
                case ColumnClass::FieldGauge: return "FieldGauge";
                case ColumnClass::VarDomain: return "VarDomain";
                case ColumnClass::ScalarGlobal: return "ScalarGlobal";
                case ColumnClass::Unknown: return "Unknown";
            }
            return "Unknown";
        }

        const char* row_taxonomy_label(System_of_eqs::RowTaxonomy taxonomy)
        {
            switch (taxonomy) {
                case System_of_eqs::RowTaxonomy::Vol: return "Vol";
                case System_of_eqs::RowTaxonomy::TauBc: return "TauBc";
                case System_of_eqs::RowTaxonomy::TauMatch: return "TauMatch";
                case System_of_eqs::RowTaxonomy::GlobalInt: return "GlobalInt";
                case System_of_eqs::RowTaxonomy::Unknown: return "Unknown";
            }
            return "Unknown";
        }

        bool same_representative_column_bucket(const ColumnMetadata& left,
                                               const ColumnMetadata& right)
        {
            return left.column_class == right.column_class &&
                   left.domain == right.domain &&
                   left.var_name == right.var_name;
        }
    }

    void System_of_eqs::classify_equation_rows(std::vector<RowClass>& out) const
    {
        if (nbr_conditions == -1) {
            KADATH_THROW("Number of conditions unknown ; call sec_member first");
        }

        out.clear();
        out.reserve(static_cast<std::size_t>(nbr_conditions));
        for (int i = 0; i < neq_int; ++i) {
            out.push_back(RowClass::Constraint);
        }
        for (int i = 0; i < neq; ++i) {
            const Equation* equation = eq[static_cast<std::size_t>(i)].get();
            const bool galerkin =
                dynamic_cast<const Eq_inside*>(equation) != nullptr ||
                dynamic_cast<const Eq_order*>(equation) != nullptr;
            const bool first_integral =
                dynamic_cast<const Eq_first_integral*>(equation) != nullptr;
            const bool vel_pot =
                dynamic_cast<const Eq_vel_pot*>(equation) != nullptr;
            const int n_rows = equation != nullptr ? equation->get_n_cond_tot() : 0;
            for (int row = 0; row < n_rows; ++row) {
                const RowClass row_class =
                    (galerkin || (vel_pot && row > 0))
                        ? RowClass::Galerkin
                        : (first_integral ? RowClass::FirstIntegral : RowClass::Constraint);
                out.push_back(row_class);
            }
        }
        if (static_cast<int>(out.size()) != nbr_conditions) {
            KADATH_THROW("Equation row classification size mismatch");
        }
    }

    void System_of_eqs::classify_equation_row_metadata(std::vector<RowMetadata>& out) const
    {
        if (nbr_conditions == -1) {
            KADATH_THROW("Number of conditions unknown ; call sec_member first");
        }

        out.clear();
        out.reserve(static_cast<std::size_t>(nbr_conditions));
        for (int i = 0; i < neq_int; ++i) {
            RowMetadata row;
            row.row = static_cast<int>(out.size());
            row.legacy_class = RowClass::FirstIntegral;
            row.taxonomy = RowTaxonomy::GlobalInt;
            row.eq_index = i;
            row.eq_local_row = 0;
            row.equation_type = "Eq_int";
            out.push_back(row);
        }
        for (int i = 0; i < neq; ++i) {
            const Equation* equation = eq[static_cast<std::size_t>(i)].get();
            const RowTaxonomy equation_taxonomy = row_taxonomy_for_equation(equation);
            const bool vel_pot =
                dynamic_cast<const Eq_vel_pot*>(equation) != nullptr;
            bool bispheric_inside = false;
            if (dynamic_cast<const Eq_inside*>(equation) != nullptr &&
                equation != nullptr && equation->ndom >= 0) {
                const Domain* domain = espace.get_domain(equation->ndom);
                bispheric_inside =
                    dynamic_cast<const Domain_bispheric_chi_first*>(domain) != nullptr ||
                    dynamic_cast<const Domain_bispheric_rect*>(domain) != nullptr ||
                    dynamic_cast<const Domain_bispheric_eta_first*>(domain) != nullptr ||
                    dynamic_cast<const Domain_bispheric_chi_first_nosym*>(domain) != nullptr ||
                    dynamic_cast<const Domain_bispheric_rect_nosym*>(domain) != nullptr ||
                    dynamic_cast<const Domain_bispheric_eta_first_nosym*>(domain) != nullptr;
            }
            const std::string equation_owner_var_name =
                i < static_cast<int>(eq_column_attachments.size())
                    ? eq_column_attachments[static_cast<std::size_t>(i)].owner_var_name
                    : std::string();
            const bool bet_interior_owner =
                equation_owner_var_name.find("bet") != std::string::npos;
            const int n_rows = equation != nullptr ? equation->get_n_cond_tot() : 0;
            for (int row = 0; row < n_rows; ++row) {
                const RowTaxonomy taxonomy =
                    (vel_pot && row == 0)
                        ? RowTaxonomy::TauBc
                        : (bispheric_inside && bet_interior_owner &&
                           (row == 0 || row == n_rows - 1))
                        ? RowTaxonomy::TauBc
                        : equation_taxonomy;
                const RowClass row_class = legacy_row_class(taxonomy);
                RowMetadata metadata;
                metadata.row = static_cast<int>(out.size());
                metadata.legacy_class = row_class;
                metadata.taxonomy = taxonomy;
                metadata.dom = equation != nullptr ? equation->ndom : -1;
                metadata.dom_pair = matching_other_domain(equation);
                metadata.eq_index = i;
                metadata.eq_local_row = row;
                metadata.equation_type = equation_type_name(equation);
                metadata.owner_var_name = equation_owner_var_name;
                out.push_back(metadata);
            }
        }
        if (static_cast<int>(out.size()) != nbr_conditions) {
            KADATH_THROW("Equation row classification does not match number of conditions");
        }
    }

    void System_of_eqs::write_def_var_census_csv(const std::string& path) const
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0)
            return;
        std::ofstream out(path);
        if (!out) {
            std::cerr << "write_def_var_census_csv: could not open " << path << '\n';
            return;
        }
        out << "def_idx,vars\n";
        std::set<std::string> vars;
        for (int i = 0; i < ndef; ++i) {
            vars.clear();
            if (def[i] != nullptr)
                def[i]->collect_vars(vars);
            out << i << ',';
            bool first = true;
            for (const std::string& name : vars) {
                if (!first)
                    out << ';';
                out << name;
                first = false;
            }
            out << '\n';
        }
    }

    void System_of_eqs::measure_primal_tangent_split(int reps,
                                                      double& out_primal,
                                                      double& out_full_w1,
                                                      double& out_full_w8)
    {
        using Clock = std::chrono::steady_clock;
        const int safe_reps = (reps > 0) ? reps : 1;

        auto clear_input_derivatives = [&]() {
            for (int i = 0; i < nterm; ++i)
                term[i]->clear_der();
            for (int i = 0; i < nterm_double; ++i)
                term_double[i]->clear_der();
        };
        // set_der_zero allocates a (zeroed) der tensor per requested lane, so
        // der_t != nullptr afterwards; the AD then runs the full product-rule
        // tangent arithmetic regardless of the zero values (it gates on
        // der_t != nullptr, not on magnitude). Seeding every field term makes
        // every dependent definition fully tangent-active at the chosen width.
        auto seed_input_derivatives = [&](int lane_count) {
            for (int i = 0; i < nterm; ++i)
                for (int lane = 0; lane < lane_count; ++lane)
                    term[i]->set_der_zero(lane);
            for (int i = 0; i < nterm_double; ++i)
                for (int lane = 0; lane < lane_count; ++lane)
                    term_double[i]->set_der_zero(lane);
        };
        auto mean_def_sweep_seconds = [&]() {
            const auto start = Clock::now();
            for (int r = 0; r < safe_reps; ++r)
                for (int i = 0; i < ndef; ++i)
                    def[i]->compute_res();
            const double total =
                std::chrono::duration<double>(Clock::now() - start).count();
            return total / static_cast<double>(safe_reps);
        };

        // Definitions may read the metric; refresh once so the sweep matches an
        // in-assembly compute_res.
        if (met != nullptr)
            for (int d = dom_min; d <= dom_max; ++d)
                met->update(d);

        // Primal only: inputs carry no derivative (der_t == nullptr) -> val_t
        // only, all tangent arithmetic skipped.
        clear_input_derivatives();
        for (int i = 0; i < ndef; ++i)
            def[i]->compute_res(); // warm: allocate def-result storage
        out_primal = mean_def_sweep_seconds();

        // Full W=1: one tangent lane on every field term -> primal + 1x tangent.
        clear_input_derivatives();
        seed_input_derivatives(1);
        for (int i = 0; i < ndef; ++i)
            def[i]->compute_res(); // warm
        out_full_w1 = mean_def_sweep_seconds();

        // Full W=8: eight tangent lanes -> primal + 8x tangent (production width).
        clear_input_derivatives();
        seed_input_derivatives(8);
        for (int i = 0; i < ndef; ++i)
            def[i]->compute_res(); // warm
        out_full_w8 = mean_def_sweep_seconds();

        clear_input_derivatives(); // restore null-derivative state
    }

    namespace
    {
        // Current resident-set bytes of this process. Keep this distinct from
        // ru_maxrss: phase attribution needs current RSS and the process
        // high-water as separate channels.
        long long current_resident_bytes()
        {
#if defined(__APPLE__)
            mach_task_basic_info_data_t info;
            mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
            if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                          reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
                return static_cast<long long>(info.resident_size);
#elif defined(__linux__)
            std::ifstream statm("/proc/self/statm");
            long long total_pages = 0;
            long long resident_pages = 0;
            if (statm >> total_pages >> resident_pages) {
                const long page_size = sysconf(_SC_PAGESIZE);
                if (page_size > 0)
                    return resident_pages * static_cast<long long>(page_size);
            }
#endif
            return -1;
        }

        long long peak_resident_bytes()
        {
            struct rusage ru;
            if (getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
                return static_cast<long long>(ru.ru_maxrss); // bytes on macOS
#else
                return static_cast<long long>(ru.ru_maxrss) * 1024LL; // KB on Linux
#endif
            }
            return -1;
        }
    } // namespace

    void report_memory_mapper_phase(const char* phase, MPI_Comm communicator)
    {
        if (!MemoryMapper::phase_profile_enabled())
            return;

        const MemoryMapperPhaseSnapshot snapshot =
            MemoryMapper::phase_snapshot();
        const long long current_rss = current_resident_bytes();
        const long long peak_rss = peak_resident_bytes();

        constexpr std::size_t field_count = 19;
        constexpr std::array<const char*, field_count> labels = {
            "requested_live_bytes",
            "capacity_live_bytes",
            "capacity_free_bytes",
            "capacity_reserved_bytes",
            "header_reserved_bytes",
            "peak_requested_live_bytes",
            "peak_capacity_live_bytes",
            "peak_capacity_reserved_bytes",
            "live_blocks",
            "free_blocks",
            "total_blocks",
            "get_calls",
            "release_calls",
            "pool_hits",
            "pool_misses",
            "system_malloc_calls",
            "system_malloc_payload_bytes",
            "current_rss_bytes",
            "peak_rss_bytes",
        };
        const std::array<long long, field_count> local = {
            static_cast<long long>(snapshot.requested_live_bytes),
            static_cast<long long>(snapshot.capacity_live_bytes),
            static_cast<long long>(snapshot.capacity_free_bytes),
            static_cast<long long>(snapshot.capacity_reserved_bytes),
            static_cast<long long>(snapshot.header_reserved_bytes),
            static_cast<long long>(snapshot.peak_requested_live_bytes),
            static_cast<long long>(snapshot.peak_capacity_live_bytes),
            static_cast<long long>(snapshot.peak_capacity_reserved_bytes),
            static_cast<long long>(snapshot.live_blocks),
            static_cast<long long>(snapshot.free_blocks),
            static_cast<long long>(snapshot.total_blocks),
            static_cast<long long>(snapshot.get_calls),
            static_cast<long long>(snapshot.release_calls),
            static_cast<long long>(snapshot.pool_hits),
            static_cast<long long>(snapshot.pool_misses),
            static_cast<long long>(snapshot.system_malloc_calls),
            static_cast<long long>(snapshot.system_malloc_payload_bytes),
            current_rss,
            peak_rss,
        };

        int rank = 0;
        int nproc = 1;
        MPI_Comm_rank(communicator, &rank);
        MPI_Comm_size(communicator, &nproc);
        std::vector<long long> gathered(
            rank == 0 ? field_count * static_cast<std::size_t>(nproc) : 0U);
        MPI_Gather(local.data(), static_cast<int>(field_count), MPI_LONG_LONG,
                   rank == 0 ? gathered.data() : nullptr,
                   static_cast<int>(field_count), MPI_LONG_LONG, 0,
                   communicator); // NOLINT(bugprone-casting-through-void)

        if (rank != 0)
            return;

        std::cout << "MemoryMapperPhase phase=" << phase
                  << " ranks=" << nproc;
        for (std::size_t field = 0; field < field_count; ++field) {
            long long sum = 0;
            long long minimum = std::numeric_limits<long long>::max();
            long long maximum = std::numeric_limits<long long>::min();
            int available = 0;
            for (int source_rank = 0; source_rank < nproc; ++source_rank) {
                const long long value = gathered[
                    static_cast<std::size_t>(source_rank) * field_count + field];
                if (value < 0)
                    continue;
                sum += value;
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
                ++available;
            }
            std::cout << ' ' << labels[field] << "_sum=";
            if (available == 0) {
                std::cout << "unavailable";
            } else {
                std::cout << sum
                          << ' ' << labels[field] << "_min=" << minimum
                          << ' ' << labels[field] << "_max=" << maximum;
                if (available != nproc)
                    std::cout << ' ' << labels[field]
                              << "_available_ranks=" << available;
            }
        }
        std::cout << std::endl;
    }

    void System_of_eqs::measure_val_cache_rss(long long& out_cache_bytes,
                                              long long& out_rss_before,
                                              long long& out_rss_after,
                                              long long& out_heavy_nodes,
                                              long long& out_total_nodes)
    {
        // Definitions read the metric; refresh once so action() on heavy nodes is
        // valid and representative of an in-assembly evaluation.
        if (met != nullptr)
            for (int d = dom_min; d <= dom_max; ++d)
                met->update(d);

        out_heavy_nodes = 0;
        out_total_nodes = 0;
        out_rss_before = current_resident_bytes();
        {
            // Hold a copy of every transform-heavy node's primal val across all
            // definition trees -- the per-node cache the hoisting lever would add.
            // unique_ptr so vector growth never deep-copies the held Term_eqs.
            std::vector<std::unique_ptr<Term_eq>> holder;
            for (int i = 0; i < ndef; i++)
                def[i]->collect_heavy_node_vals(holder, out_heavy_nodes, out_total_nodes);
            out_rss_after = current_resident_bytes();
            // holder released here
        }
        out_cache_bytes = (out_rss_after >= 0 && out_rss_before >= 0)
                              ? (out_rss_after - out_rss_before)
                              : -1;
    }

    bool System_of_eqs::validate_packed_wlane2_columns(int first_column, int second_column,
                                                       double abs_tol, double rel_tol,
                                                       std::ostream& report)
    {
        if (nbr_conditions == -1) {
            KADATH_THROW("Number of conditions unknown ; call sec_member first");
        }

        Array<double> scalar_first(do_col_J(first_column));
        Array<double> scalar_second(do_col_J(second_column));
        Array<double> packed_first(nbr_conditions);
        Array<double> packed_second(nbr_conditions);
        packed_first = 0.0;
        packed_second = 0.0;
        std::vector<RowMetadata> row_metadata;
        classify_equation_row_metadata(row_metadata);

        std::string failure_reason;
        if (!jac_col_engine_.compute_packed_wlane2_columns_for_verification(
                first_column, second_column, packed_first, packed_second, failure_reason)) {
            report << "packed W=2 verification setup failed: " << failure_reason << '\n';
            return false;
        }

        auto compare_column = [&](const char* label,
                                  const Array<double>& scalar_column,
                                  const Array<double>& packed_column) -> bool {
            int mismatch_count = 0;
            double max_abs_error = 0.0;
            double max_rel_error = 0.0;
            int max_error_row = -1;
            for (int row = 0; row < nbr_conditions; ++row) {
                const double scalar_value = scalar_column(row);
                const double packed_value = packed_column(row);
                const double abs_error = std::abs(scalar_value - packed_value);
                const double scale = std::max({1.0, std::abs(scalar_value), std::abs(packed_value)});
                const double rel_error = abs_error / scale;
                if (abs_error > abs_tol + rel_tol * scale) {
                    ++mismatch_count;
                    if (abs_error > max_abs_error) {
                        max_abs_error = abs_error;
                        max_rel_error = rel_error;
                        max_error_row = row;
                    }
                }
            }
            if (mismatch_count > 0) {
                report << label << " mismatch_count=" << mismatch_count
                       << " max_abs_error=" << max_abs_error
                       << " max_rel_error=" << max_rel_error
                       << " row=" << max_error_row
                       << " scalar=" << scalar_column(max_error_row)
                       << " packed=" << packed_column(max_error_row);
                if (max_error_row >= 0 &&
                    max_error_row < static_cast<int>(row_metadata.size())) {
                    const RowMetadata& row = row_metadata[static_cast<std::size_t>(max_error_row)];
                    report << " taxonomy=" << row_taxonomy_label(row.taxonomy)
                           << " equation=" << row.equation_type
                           << " eq_index=" << row.eq_index
                           << " domain=" << row.dom
                           << " owner=" << row.owner_var_name;
                    if (row.dom_pair >= 0)
                        report << " other_domain=" << row.dom_pair;
                }
                report << '\n';
                return false;
            }
            report << label << " ok rows=" << nbr_conditions << '\n';
            return true;
        };

        const bool first_ok = compare_column("first_column", scalar_first, packed_first);
        const bool second_ok = compare_column("second_column", scalar_second, packed_second);

        auto compare_sparse_column =
            [&](const char* label,
                const Array<double>& scalar_column,
                const std::vector<std::pair<int, double>>& sparse_entries) -> bool {
            std::vector<unsigned char> emitted(static_cast<std::size_t>(nbr_conditions), 0);
            int mismatch_count = 0;
            double max_abs_error = 0.0;
            double max_rel_error = 0.0;
            int max_error_row = -1;
            for (const auto& [row, value] : sparse_entries) {
                if (row < 0 || row >= nbr_conditions) {
                    report << label << " sparse row outside valid range: " << row << '\n';
                    return false;
                }
                emitted[static_cast<std::size_t>(row)] = 1;
                const double scalar_value = scalar_column(row);
                const double abs_error = std::abs(scalar_value - value);
                const double scale = std::max({1.0, std::abs(scalar_value), std::abs(value)});
                const double rel_error = abs_error / scale;
                if (abs_error > abs_tol + rel_tol * scale) {
                    ++mismatch_count;
                    if (abs_error > max_abs_error) {
                        max_abs_error = abs_error;
                        max_rel_error = rel_error;
                        max_error_row = row;
                    }
                }
            }
            for (int row = 0; row < nbr_conditions; ++row) {
                if (std::abs(scalar_column(row)) > abs_tol &&
                    emitted[static_cast<std::size_t>(row)] == 0) {
                    ++mismatch_count;
                    const double abs_error = std::abs(scalar_column(row));
                    const double scale = std::max(1.0, abs_error);
                    const double rel_error = abs_error / scale;
                    if (abs_error > max_abs_error) {
                        max_abs_error = abs_error;
                        max_rel_error = rel_error;
                        max_error_row = row;
                    }
                }
            }
            if (mismatch_count > 0) {
                report << label << " filtered_sparse mismatch_count=" << mismatch_count
                       << " max_abs_error=" << max_abs_error
                       << " max_rel_error=" << max_rel_error
                       << " row=" << max_error_row
                       << " scalar=" << scalar_column(max_error_row);
                if (max_error_row >= 0 &&
                    max_error_row < static_cast<int>(row_metadata.size())) {
                    const RowMetadata& row = row_metadata[static_cast<std::size_t>(max_error_row)];
                    report << " taxonomy=" << row_taxonomy_label(row.taxonomy)
                           << " equation=" << row.equation_type
                           << " eq_index=" << row.eq_index
                           << " domain=" << row.dom
                           << " owner=" << row.owner_var_name;
                    if (row.dom_pair >= 0)
                        report << " other_domain=" << row.dom_pair;
                }
                report << '\n';
                return false;
            }
            report << label << " filtered_sparse ok entries=" << sparse_entries.size() << '\n';
            return true;
        };

        std::vector<std::pair<int, double>> sparse_first;
        std::vector<std::pair<int, double>> sparse_second;
        std::string sparse_failure_reason;
        const bool sparse_setup_ok = jac_col_engine_.compute_packed_wlane2_columns_sparse(
            first_column, second_column, 0.0,
            [&](int row, double value) { sparse_first.emplace_back(row, value); },
            [&](int row, double value) { sparse_second.emplace_back(row, value); },
            sparse_failure_reason);
        if (!sparse_setup_ok) {
            report << "packed W=2 filtered sparse setup failed: "
                   << sparse_failure_reason << '\n';
            return false;
        }
        const bool first_sparse_ok =
            compare_sparse_column("first_column", scalar_first, sparse_first);
        const bool second_sparse_ok =
            compare_sparse_column("second_column", scalar_second, sparse_second);

        return first_ok && second_ok && first_sparse_ok && second_sparse_ok;
    }

    bool System_of_eqs::validate_packed_wlane4_columns(const std::array<int, 4>& columns,
                                                       double abs_tol, double rel_tol,
                                                       std::ostream& report)
    {
        if (nbr_conditions == -1) {
            KADATH_THROW("Number of conditions unknown ; call sec_member first");
        }

        std::array<Array<double>, 4> scalar_columns = {
            Array<double>(do_col_J(columns[0])),
            Array<double>(do_col_J(columns[1])),
            Array<double>(do_col_J(columns[2])),
            Array<double>(do_col_J(columns[3])),
        };
        std::array<Array<double>, 4> packed_columns = {
            Array<double>(nbr_conditions), Array<double>(nbr_conditions),
            Array<double>(nbr_conditions), Array<double>(nbr_conditions),
        };
        for (Array<double>& column : packed_columns)
            column = 0.0;
        std::array<Array<double>*, 4> packed_ptrs = {
            &packed_columns[0], &packed_columns[1], &packed_columns[2], &packed_columns[3],
        };

        std::vector<RowMetadata> row_metadata;
        classify_equation_row_metadata(row_metadata);

        std::string failure_reason;
        if (!jac_col_engine_.compute_packed_wlane4_columns_for_verification(
                columns, packed_ptrs, failure_reason)) {
            report << "packed W=4 verification setup failed: " << failure_reason << '\n';
            return false;
        }

        auto compare_column = [&](const char* label,
                                  const Array<double>& scalar_column,
                                  const Array<double>& packed_column) -> bool {
            int mismatch_count = 0;
            double max_abs_error = 0.0;
            double max_rel_error = 0.0;
            int max_error_row = -1;
            for (int row = 0; row < nbr_conditions; ++row) {
                const double scalar_value = scalar_column(row);
                const double packed_value = packed_column(row);
                const double abs_error = std::abs(scalar_value - packed_value);
                const double scale = std::max({1.0, std::abs(scalar_value), std::abs(packed_value)});
                const double rel_error = abs_error / scale;
                if (abs_error > abs_tol + rel_tol * scale) {
                    ++mismatch_count;
                    if (abs_error > max_abs_error) {
                        max_abs_error = abs_error;
                        max_rel_error = rel_error;
                        max_error_row = row;
                    }
                }
            }
            if (mismatch_count > 0) {
                report << label << " mismatch_count=" << mismatch_count
                       << " max_abs_error=" << max_abs_error
                       << " max_rel_error=" << max_rel_error
                       << " row=" << max_error_row
                       << " scalar=" << scalar_column(max_error_row)
                       << " packed=" << packed_column(max_error_row);
                if (max_error_row >= 0 &&
                    max_error_row < static_cast<int>(row_metadata.size())) {
                    const RowMetadata& row = row_metadata[static_cast<std::size_t>(max_error_row)];
                    report << " taxonomy=" << row_taxonomy_label(row.taxonomy)
                           << " equation=" << row.equation_type
                           << " eq_index=" << row.eq_index
                           << " domain=" << row.dom
                           << " owner=" << row.owner_var_name;
                    if (row.dom_pair >= 0)
                        report << " other_domain=" << row.dom_pair;
                }
                report << '\n';
                return false;
            }
            report << label << " ok rows=" << nbr_conditions << '\n';
            return true;
        };

        bool all_full_ok = true;
        for (int k = 0; k < 4; ++k) {
            const std::string label = "lane_" + std::to_string(k) + "_column_" +
                                      std::to_string(columns[k]);
            all_full_ok = compare_column(label.c_str(), scalar_columns[k], packed_columns[k])
                          && all_full_ok;
        }

        auto compare_sparse_column =
            [&](const char* label,
                const Array<double>& scalar_column,
                const std::vector<std::pair<int, double>>& sparse_entries) -> bool {
            std::vector<unsigned char> emitted(static_cast<std::size_t>(nbr_conditions), 0);
            int mismatch_count = 0;
            double max_abs_error = 0.0;
            double max_rel_error = 0.0;
            int max_error_row = -1;
            for (const auto& [row, value] : sparse_entries) {
                if (row < 0 || row >= nbr_conditions) {
                    report << label << " sparse row outside valid range: " << row << '\n';
                    return false;
                }
                emitted[static_cast<std::size_t>(row)] = 1;
                const double scalar_value = scalar_column(row);
                const double abs_error = std::abs(scalar_value - value);
                const double scale = std::max({1.0, std::abs(scalar_value), std::abs(value)});
                const double rel_error = abs_error / scale;
                if (abs_error > abs_tol + rel_tol * scale) {
                    ++mismatch_count;
                    if (abs_error > max_abs_error) {
                        max_abs_error = abs_error;
                        max_rel_error = rel_error;
                        max_error_row = row;
                    }
                }
            }
            for (int row = 0; row < nbr_conditions; ++row) {
                if (std::abs(scalar_column(row)) > abs_tol &&
                    emitted[static_cast<std::size_t>(row)] == 0) {
                    ++mismatch_count;
                    const double abs_error = std::abs(scalar_column(row));
                    const double scale = std::max(1.0, abs_error);
                    const double rel_error = abs_error / scale;
                    if (abs_error > max_abs_error) {
                        max_abs_error = abs_error;
                        max_rel_error = rel_error;
                        max_error_row = row;
                    }
                }
            }
            if (mismatch_count > 0) {
                report << label << " filtered_sparse mismatch_count=" << mismatch_count
                       << " max_abs_error=" << max_abs_error
                       << " max_rel_error=" << max_rel_error
                       << " row=" << max_error_row
                       << " scalar=" << scalar_column(max_error_row);
                report << '\n';
                return false;
            }
            report << label << " filtered_sparse ok entries=" << sparse_entries.size() << '\n';
            return true;
        };

        std::array<std::vector<std::pair<int, double>>, 4> sparse_columns;
        // SparseColumnEmitter stores a borrowed pointer to the callable; name
        // the lambdas as locals so they outlive the sparse call.
        auto sparse_collect_0 = [&](int row, double value) { sparse_columns[0].emplace_back(row, value); };
        auto sparse_collect_1 = [&](int row, double value) { sparse_columns[1].emplace_back(row, value); };
        auto sparse_collect_2 = [&](int row, double value) { sparse_columns[2].emplace_back(row, value); };
        auto sparse_collect_3 = [&](int row, double value) { sparse_columns[3].emplace_back(row, value); };
        std::array<SparseColumnEmitter, 4> sparse_emitters = {
            SparseColumnEmitter{sparse_collect_0},
            SparseColumnEmitter{sparse_collect_1},
            SparseColumnEmitter{sparse_collect_2},
            SparseColumnEmitter{sparse_collect_3},
        };
        std::string sparse_failure_reason;
        const bool sparse_setup_ok = jac_col_engine_.compute_packed_wlane4_columns_sparse(
            columns, 0.0, sparse_emitters, sparse_failure_reason);
        if (!sparse_setup_ok) {
            report << "packed W=4 filtered sparse setup failed: "
                   << sparse_failure_reason << '\n';
            return false;
        }
        bool all_sparse_ok = true;
        for (int k = 0; k < 4; ++k) {
            const std::string label = "lane_" + std::to_string(k) + "_column_" +
                                      std::to_string(columns[k]);
            all_sparse_ok = compare_sparse_column(label.c_str(), scalar_columns[k], sparse_columns[k])
                            && all_sparse_ok;
        }

        return all_full_ok && all_sparse_ok;
    }

    bool System_of_eqs::validate_packed_wlane4_representative_columns(int max_quartets_per_class,
                                                                      double abs_tol, double rel_tol,
                                                                      std::ostream& report)
    {
        if (max_quartets_per_class <= 0) {
            KADATH_THROW("W=4 representative oracle requires max_quartets_per_class > 0");
        }
        if (nbr_conditions == -1) {
            (void)sec_member();
        }

        std::vector<ColumnMetadata> columns;
        classify_columns(columns);

        constexpr std::array<ColumnClass, 7> sampled_classes{
            ColumnClass::FieldInteriorVol,
            ColumnClass::FieldBoundaryTau,
            ColumnClass::FieldOuterShellTau,
            ColumnClass::FieldMatching,
            ColumnClass::FieldGauge,
            ColumnClass::FieldUnknown,
            ColumnClass::ScalarGlobal,
        };

        bool all_ok = true;
        int checked_quartets = 0;
        for (ColumnClass sampled_class : sampled_classes) {
            int class_quartets = 0;
            for (std::size_t i = 0;
                 i + 3 < columns.size() && class_quartets < max_quartets_per_class;
                 ++i) {
                const ColumnMetadata& c0 = columns[i];
                if (c0.column_class != sampled_class)
                    continue;
                const ColumnMetadata& c1 = columns[i + 1];
                const ColumnMetadata& c2 = columns[i + 2];
                const ColumnMetadata& c3 = columns[i + 3];
                if (!same_representative_column_bucket(c0, c1) ||
                    !same_representative_column_bucket(c0, c2) ||
                    !same_representative_column_bucket(c0, c3)) {
                    continue;
                }

                report << "packed W=4 representative quartet class="
                       << column_class_label(sampled_class)
                       << " domain=" << c0.domain
                       << " var=" << c0.var_name
                       << " columns=" << c0.column << "," << c1.column << ","
                       << c2.column << "," << c3.column << '\n';
                const std::array<int, 4> quartet{c0.column, c1.column, c2.column, c3.column};
                const bool quartet_ok = validate_packed_wlane4_columns(
                    quartet, abs_tol, rel_tol, report);
                all_ok = all_ok && quartet_ok;
                ++checked_quartets;
                ++class_quartets;
            }
            if (class_quartets == 0) {
                report << "packed W=4 representative class="
                       << column_class_label(sampled_class)
                       << " skipped: no adjacent same-bucket quartet\n";
            }
        }

        report << "packed W=4 representative checked_quartets=" << checked_quartets
               << " verdict=" << (all_ok && checked_quartets > 0 ? "PASS" : "FAIL") << '\n';
        return all_ok && checked_quartets > 0;
    }

    bool System_of_eqs::validate_packed_wlane8_columns(const std::array<int, 8>& columns,
                                                       double abs_tol, double rel_tol,
                                                       std::ostream& report)
    {
        if (nbr_conditions == -1) {
            (void)sec_member();
        }

        std::array<Array<double>, 8> scalar_columns = {
            Array<double>(do_col_J(columns[0])),
            Array<double>(do_col_J(columns[1])),
            Array<double>(do_col_J(columns[2])),
            Array<double>(do_col_J(columns[3])),
            Array<double>(do_col_J(columns[4])),
            Array<double>(do_col_J(columns[5])),
            Array<double>(do_col_J(columns[6])),
            Array<double>(do_col_J(columns[7])),
        };

        std::vector<RowMetadata> row_metadata;
        classify_equation_row_metadata(row_metadata);

        std::array<std::vector<std::pair<int, double>>, 8> sparse_columns;
        auto sparse_collect_0 = [&](int row, double value) { sparse_columns[0].emplace_back(row, value); };
        auto sparse_collect_1 = [&](int row, double value) { sparse_columns[1].emplace_back(row, value); };
        auto sparse_collect_2 = [&](int row, double value) { sparse_columns[2].emplace_back(row, value); };
        auto sparse_collect_3 = [&](int row, double value) { sparse_columns[3].emplace_back(row, value); };
        auto sparse_collect_4 = [&](int row, double value) { sparse_columns[4].emplace_back(row, value); };
        auto sparse_collect_5 = [&](int row, double value) { sparse_columns[5].emplace_back(row, value); };
        auto sparse_collect_6 = [&](int row, double value) { sparse_columns[6].emplace_back(row, value); };
        auto sparse_collect_7 = [&](int row, double value) { sparse_columns[7].emplace_back(row, value); };
        std::array<SparseColumnEmitter, 8> sparse_emitters = {
            SparseColumnEmitter{sparse_collect_0},
            SparseColumnEmitter{sparse_collect_1},
            SparseColumnEmitter{sparse_collect_2},
            SparseColumnEmitter{sparse_collect_3},
            SparseColumnEmitter{sparse_collect_4},
            SparseColumnEmitter{sparse_collect_5},
            SparseColumnEmitter{sparse_collect_6},
            SparseColumnEmitter{sparse_collect_7},
        };
        std::string sparse_failure_reason;
        const bool sparse_setup_ok = jac_col_engine_.compute_packed_wlane8_columns_sparse(
            columns, 0.0, sparse_emitters, sparse_failure_reason);
        if (!sparse_setup_ok) {
            report << "packed W=8 filtered sparse setup failed: "
                   << sparse_failure_reason << '\n';
            return false;
        }

        auto compare_sparse_column =
            [&](const char* label,
                const Array<double>& scalar_column,
                const std::vector<std::pair<int, double>>& sparse_entries) -> bool {
            std::vector<unsigned char> emitted(static_cast<std::size_t>(nbr_conditions), 0);
            int mismatch_count = 0;
            int exact_bit_mismatch_count = 0;
            double max_abs_error = 0.0;
            double max_rel_error = 0.0;
            int max_error_row = -1;
            double max_error_scalar = 0.0;
            double max_error_packed = 0.0;
            for (const auto& [row, value] : sparse_entries) {
                if (row < 0 || row >= nbr_conditions) {
                    report << label << " sparse row outside valid range: " << row << '\n';
                    return false;
                }
                emitted[static_cast<std::size_t>(row)] = 1;
                const double scalar_value = scalar_column(row);
                const double abs_error = std::abs(scalar_value - value);
                const double scale = std::max({1.0, std::abs(scalar_value), std::abs(value)});
                const double rel_error = abs_error / scale;
                if (std::memcmp(&scalar_value, &value, sizeof(double)) != 0)
                    ++exact_bit_mismatch_count;
                if (abs_error > abs_tol + rel_tol * scale) {
                    ++mismatch_count;
                    if (abs_error > max_abs_error) {
                        max_abs_error = abs_error;
                        max_rel_error = rel_error;
                        max_error_row = row;
                        max_error_scalar = scalar_value;
                        max_error_packed = value;
                    }
                }
            }
            for (int row = 0; row < nbr_conditions; ++row) {
                if (std::abs(scalar_column(row)) > abs_tol &&
                    emitted[static_cast<std::size_t>(row)] == 0) {
                    ++mismatch_count;
                    const double abs_error = std::abs(scalar_column(row));
                    const double scale = std::max(1.0, abs_error);
                    const double rel_error = abs_error / scale;
                    if (abs_error > max_abs_error) {
                        max_abs_error = abs_error;
                        max_rel_error = rel_error;
                        max_error_row = row;
                        max_error_scalar = scalar_column(row);
                        max_error_packed = 0.0;
                    }
                }
            }
            if (mismatch_count > 0) {
                report << label << " filtered_sparse mismatch_count=" << mismatch_count
                       << " exact_bit_mismatches=" << exact_bit_mismatch_count
                       << " entries=" << sparse_entries.size()
                       << " max_abs_error=" << max_abs_error
                       << " max_rel_error=" << max_rel_error
                       << " row=" << max_error_row
                       << " scalar=" << max_error_scalar
                       << " packed=" << max_error_packed;
                if (max_error_row >= 0 &&
                    max_error_row < static_cast<int>(row_metadata.size())) {
                    const RowMetadata& row = row_metadata[static_cast<std::size_t>(max_error_row)];
                    report << " taxonomy=" << row_taxonomy_label(row.taxonomy)
                           << " equation=" << row.equation_type
                           << " eq_index=" << row.eq_index
                           << " domain=" << row.dom
                           << " owner=" << row.owner_var_name;
                    if (row.dom_pair >= 0)
                        report << " other_domain=" << row.dom_pair;
                }
                report << '\n';
                return false;
            }
            report << label << " filtered_sparse ok entries=" << sparse_entries.size()
                   << " exact_bit_mismatches=" << exact_bit_mismatch_count << '\n';
            return true;
        };

        bool all_sparse_ok = true;
        for (int k = 0; k < 8; ++k) {
            const std::string label = "lane_" + std::to_string(k) + "_column_" +
                                      std::to_string(columns[k]);
            all_sparse_ok = compare_sparse_column(label.c_str(), scalar_columns[k], sparse_columns[k])
                            && all_sparse_ok;
        }
        return all_sparse_ok;
    }

    bool System_of_eqs::validate_packed_wlane8_representative_columns(int max_octets_per_class,
                                                                      double abs_tol, double rel_tol,
                                                                      std::ostream& report)
    {
        if (max_octets_per_class <= 0) {
            KADATH_THROW("W=8 representative oracle requires max_octets_per_class > 0");
        }
        if (nbr_conditions == -1) {
            (void)sec_member();
        }

        std::vector<ColumnMetadata> columns;
        classify_columns(columns);

        constexpr std::array<ColumnClass, 7> sampled_classes{
            ColumnClass::FieldInteriorVol,
            ColumnClass::FieldBoundaryTau,
            ColumnClass::FieldOuterShellTau,
            ColumnClass::FieldMatching,
            ColumnClass::FieldGauge,
            ColumnClass::FieldUnknown,
            ColumnClass::ScalarGlobal,
        };

        bool all_ok = true;
        int checked_octets = 0;
        for (ColumnClass sampled_class : sampled_classes) {
            int class_octets = 0;
            for (std::size_t i = 0;
                 i + 7 < columns.size() && class_octets < max_octets_per_class;
                 ++i) {
                const ColumnMetadata& c0 = columns[i];
                if (c0.column_class != sampled_class)
                    continue;
                bool same_bucket = true;
                for (int lane = 1; lane < 8; ++lane) {
                    same_bucket = same_bucket &&
                        same_representative_column_bucket(
                            c0, columns[i + static_cast<std::size_t>(lane)]);
                }
                if (!same_bucket)
                    continue;

                std::array<int, 8> octet{};
                report << "packed W=8 representative octet class="
                       << column_class_label(sampled_class)
                       << " domain=" << c0.domain
                       << " var=" << c0.var_name
                       << " columns=";
                for (int lane = 0; lane < 8; ++lane) {
                    octet[static_cast<std::size_t>(lane)] =
                        columns[i + static_cast<std::size_t>(lane)].column;
                    report << (lane == 0 ? "" : ",")
                           << octet[static_cast<std::size_t>(lane)];
                }
                report << '\n';
                const bool octet_ok = validate_packed_wlane8_columns(
                    octet, abs_tol, rel_tol, report);
                all_ok = all_ok && octet_ok;
                ++checked_octets;
                ++class_octets;
            }
            if (class_octets == 0) {
                report << "packed W=8 representative class="
                       << column_class_label(sampled_class)
                       << " skipped: no adjacent same-bucket octet\n";
            }
        }

        report << "packed W=8 representative checked_octets=" << checked_octets
               << " verdict=" << (all_ok && checked_octets > 0 ? "PASS" : "FAIL") << '\n';
        return all_ok && checked_octets > 0;
    }

    bool System_of_eqs::validate_packed_wlane2_representative_columns(int max_pairs_per_class,
                                                                      double abs_tol, double rel_tol,
                                                                      std::ostream& report)
    {
        if (max_pairs_per_class <= 0) {
            KADATH_THROW("W=2 representative oracle requires max_pairs_per_class > 0");
        }
        if (nbr_conditions == -1) {
            (void)sec_member();
        }

        std::vector<ColumnMetadata> columns;
        classify_columns(columns);

        constexpr std::array<ColumnClass, 6> sampled_classes{
            ColumnClass::FieldInteriorVol,
            ColumnClass::FieldBoundaryTau,
            ColumnClass::FieldOuterShellTau,
            ColumnClass::FieldMatching,
            ColumnClass::FieldGauge,
            ColumnClass::ScalarGlobal,
        };

        bool all_ok = true;
        int checked_pairs = 0;
        for (ColumnClass sampled_class : sampled_classes) {
            int class_pairs = 0;
            for (std::size_t i = 0; i + 1 < columns.size() && class_pairs < max_pairs_per_class; ++i) {
                const ColumnMetadata& first = columns[i];
                const ColumnMetadata& second = columns[i + 1];
                if (first.column_class != sampled_class ||
                    !same_representative_column_bucket(first, second)) {
                    continue;
                }

                report << "packed W=2 representative pair class="
                       << column_class_label(sampled_class)
                       << " domain=" << first.domain
                       << " var=" << first.var_name
                       << " columns=" << first.column << "," << second.column << '\n';
                const bool pair_ok = validate_packed_wlane2_columns(first.column, second.column,
                                                                    abs_tol, rel_tol, report);
                all_ok = all_ok && pair_ok;
                ++checked_pairs;
                ++class_pairs;
            }
            if (class_pairs == 0) {
                report << "packed W=2 representative class="
                       << column_class_label(sampled_class)
                       << " skipped: no adjacent same-bucket pair\n";
            }
        }

        report << "packed W=2 representative checked_pairs=" << checked_pairs
               << " verdict=" << (all_ok && checked_pairs > 0 ? "PASS" : "FAIL") << '\n';
        return all_ok && checked_pairs > 0;
    }
} // namespace Kadath
