/*
    Copyright 2017 Philippe Grandclement & Gregoire Martinon
    Kadath - GNU GPL v3+
*/

#include "mpi.h"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"
#include "For_Kadath/System_of_eqs/system_dof_record.hpp"
#include "For_Kadath/Space/bin_ns_nosym.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace Kadath
{

    extern "C"
    {
    void sl_init_(int*, int*, int*);
    void blacs_gridexit_(int*);
    void blacs_gridinfo_(int*, int*, int*, int*, int*);
    int numroc_(int*, int*, int*, int*, int*);
    void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
    void pdgesv_(int*, int*, double*, int*, int*, int*, int*, double*, int*, int*, int*, int*);
    void pdgemr2d_(int*, int*, double*, int*, int*, int*, double*, int*, int*, int*, int*);
    // Ope_mult profile hooks (defined in src/Ope_eq/ope_mult.cpp). Env-gated
    // by OPE_MULT_PROFILE=1; otherwise both are no-ops.
    void dump_ope_mult_profile();
    void reset_ope_mult_profile();
    // Per-operator action census (defined in src/Ope_eq/ope_eq.cpp). Env-gated
    // by OPE_ACTION_PROFILE=1; otherwise both are no-ops.
    void dump_ope_action_profile();
    void reset_ope_action_profile();
    }

    namespace
    {
        template <typename T>
        class Array2d
        {
        public:
            Array2d(std::size_t rows, std::size_t cols)
                : rows_(rows), values_(rows * cols)
            {
            }

            T& operator()(std::size_t row, std::size_t col)
            {
                return values_[row + rows_ * col];
            }

            T* data()
            {
                return values_.data();
            }

        private:
            std::size_t rows_;
            std::vector<T> values_;
        };

        double elapsed_seconds(std::chrono::time_point<std::chrono::system_clock> const& begin)
        {
            return std::chrono::duration<double>(std::chrono::system_clock::now() - begin).count();
        }

        void split_process_grid(int target, int& low, int& up)
        {
            up = static_cast<int>(std::ceil(std::sqrt(target)));
            while (target % up)
                ++up;
            low = target / up;
        }

        std::vector<int> dense_env_int_list(const char* name)
        {
            std::vector<int> result;
            const char* cursor = std::getenv(name);
            if (cursor == nullptr || cursor[0] == '\0')
                return result;

            while (*cursor != '\0') {
                char* end = nullptr;
                const long parsed = std::strtol(cursor, &end, 10);
                if (end != cursor)
                    result.push_back(static_cast<int>(parsed));
                cursor = end;
                while (*cursor == ',' || *cursor == ':' || *cursor == ' ' || *cursor == '\t')
                    ++cursor;
                if (end == cursor && *cursor != '\0')
                    ++cursor;
            }
            return result;
        }

        void zero_selected_dense_corrections(Array<double>& solution, int column_count)
        {
            const std::vector<int> selected = dense_env_int_list("DENSE_ZERO_COLUMN_LIST");
            if (selected.empty())
                return;

            std::cout << "Zeroing dense Newton correction columns:";
            for (int col : selected) {
                if (col < 0 || col >= column_count)
                    continue;
                solution.set(col) = 0.0;
                std::cout << " " << col;
            }
            std::cout << std::endl;
        }

        int surface_phi_quantum(int k)
        {
            return (k % 2 == 0) ? k / 2 : (k - 1) / 2;
        }

        int surface_shape_jmin(int k, bool cos_path)
        {
            const int regularity_min = surface_phi_quantum(k);
            const int galerkin_min = cos_path ? ((k >= 2) ? 1 : 0) : ((k >= 4) ? 2 : 0);
            return std::max(regularity_min, galerkin_min);
        }

        int surface_shape_jmax(int ntheta, bool cos_path)
        {
            return cos_path ? ntheta : ntheta - 1;
        }

        bool surface_mode_is_z_even(const Domain& domain, int local_mode)
        {
            const Dim_array coefficients = domain.get_nbr_coefs();
            int mode = 0;
            for (int k = 0; k < coefficients(2); ++k) {
                if (k == 1 || k == coefficients(2) - 1)
                    continue;
                const int m = surface_phi_quantum(k);
                const bool cos_path = (m % 2 == 0);
                const int jmin = surface_shape_jmin(k, cos_path);
                const int jmax = surface_shape_jmax(coefficients(1), cos_path);
                for (int j = jmin; j < jmax; ++j) {
                    if (mode == local_mode)
                        return cos_path ? (j % 2 == 0) : (j % 2 == 1);
                    ++mode;
                }
            }
            return true;
        }

        bool component_flips_z_parity(const Array<int>& component)
        {
            int z_indices = 0;
            for (int i = 0; i < component.get_size(0); ++i) {
                if (component(i) == 3)
                    ++z_indices;
            }
            return (z_indices % 2) != 0;
        }

        bool coefficient_scalar_is_z_even(const Base_spectral& base, const Index& position)
        {
            if (!base.is_def())
                return true;

            const Array<int>* theta_bases = base.get_base_1d(1);
            const Array<int>* phi_bases = base.get_base_1d(2);

            if (theta_bases != nullptr && theta_bases->get_nbr() > 0) {
                const int theta_slot = (theta_bases->get_nbr() == 1) ? 0 : position(2);
                if (theta_slot >= 0 && theta_slot < static_cast<int>(theta_bases->get_nbr())) {
                    const int theta_basis = (*theta_bases)(theta_slot);
                    if (theta_basis == COS)
                        return (position(1) % 2) == 0;
                    if (theta_basis == SIN)
                        return (position(1) % 2) == 1;
                }
            }

            if (phi_bases != nullptr && phi_bases->get_nbr() > 0 && (*phi_bases)(0) == COSSIN)
                return (position(2) % 2) == 0;

            return true;
        }

        bool val_domain_column_has_z_symmetric_content(const Val_domain& value, const Array<int>& component)
        {
            if (value.check_if_zero())
                return false;

            const bool flip = component_flips_z_parity(component);
            const Array<double> coefficients = value.get_coef();
            const Base_spectral& base = value.get_base();
            bool saw = false;
            bool saw_kept = false;
            bool saw_removed = false;
            Index position(coefficients.get_dimensions());

            do {
                if (std::fabs(coefficients(position)) <= 0.0)
                    continue;
                saw = true;
                const bool scalar_even = coefficient_scalar_is_z_even(base, position);
                const bool keep = flip ? !scalar_even : scalar_even;
                saw_kept = saw_kept || keep;
                saw_removed = saw_removed || !keep;
            } while (position.inc());

            if (!saw)
                return false;

            // Coupled Galerkin columns should stay grouped unless every
            // coefficient in the seed is outside the aligned z-symmetric subspace.
            return saw_kept || (saw_kept && saw_removed);
        }

        void assign_probe_bases(const Tensor& reference, Tensor& probe, int domain)
        {
            for (int comp = 0; comp < probe.get_n_comp(); ++comp) {
                const Array<int> index = probe.indices(comp);
                if (!probe(index)(domain).check_if_zero() &&
                    !probe(index)(domain).get_base().is_def()) {
                    probe.set(index).set_domain(domain).set_base() =
                        reference(index)(domain).get_base();
                }
            }
        }

        const char* spectral_basis_label(int basis)
        {
            switch (basis) {
                case CHEB: return "CHEB";
                case CHEB_EVEN: return "CHEB_EVEN";
                case CHEB_ODD: return "CHEB_ODD";
                case COSSIN: return "COSSIN";
                case COS: return "COS";
                case COS_EVEN: return "COS_EVEN";
                case COS_ODD: return "COS_ODD";
                case SIN: return "SIN";
                case SIN_EVEN: return "SIN_EVEN";
                case SIN_ODD: return "SIN_ODD";
                case LEG: return "LEG";
                case LEG_EVEN: return "LEG_EVEN";
                case LEG_ODD: return "LEG_ODD";
                case COSSIN_EVEN: return "COSSIN_EVEN";
                case COSSIN_ODD: return "COSSIN_ODD";
                default: return "UNDEFINED";
            }
        }

        int basis_value_at(const Array<int>& bases, int transformed_dimension, const Index& coefficient)
        {
            if (bases.get_nbr() == 0)
                return -1;
            if (bases.get_nbr() == 1)
                return bases(0);

            switch (bases.get_ndim()) {
                case 1: {
                    int slot = 0;
                    if (transformed_dimension == 0 || transformed_dimension == 1)
                        slot = coefficient(2);
                    else
                        slot = coefficient(0);
                    slot = std::max(0, std::min(slot, bases.get_size(0) - 1));
                    return bases(slot);
                }
                case 2: {
                    const int first = std::max(0, std::min(coefficient(1), bases.get_size(0) - 1));
                    const int second = std::max(0, std::min(coefficient(2), bases.get_size(1) - 1));
                    return bases(first, second);
                }
                case 3: {
                    const int first = std::max(0, std::min(coefficient(0), bases.get_size(0) - 1));
                    const int second = std::max(0, std::min(coefficient(1), bases.get_size(1) - 1));
                    const int third = std::max(0, std::min(coefficient(2), bases.get_size(2) - 1));
                    return bases(first, second, third);
                }
                default:
                    return -1;
            }
        }

        void print_component_indices(std::ostream& os, const Array<int>& component)
        {
            os << "(";
            for (int i = 0; i < component.get_size(0); ++i) {
                if (i > 0)
                    os << ",";
                os << component(i);
            }
            os << ")";
        }

        void print_coefficient_bases(std::ostream& os, const Base_spectral& base, const Index& coefficient)
        {
            os << " bases=(";
            for (int dim = 0; dim < 3; ++dim) {
                if (dim > 0)
                    os << ",";
                const Array<int>* bases = base.get_base_1d(dim);
                if (bases == nullptr) {
                    os << "UNDEFINED";
                    continue;
                }
                os << spectral_basis_label(basis_value_at(*bases, dim, coefficient));
            }
            os << ")";
        }

    }

    int System_of_eqs::project_z_symmetric_diagnostic_correction(Array<double>& correction,
                                                                 std::ostream* os) const
    {
        const Space_bin_ns_nosym* binary_space = dynamic_cast<const Space_bin_ns_nosym*>(&espace);
        std::vector<ColumnMetadata> columns;
        classify_columns(columns);

        std::vector<int> term_start(static_cast<std::size_t>(nterm), -1);
        for (const ColumnMetadata& metadata : columns) {
            if (metadata.term_idx >= 0 &&
                metadata.term_idx < nterm &&
                term_start[static_cast<std::size_t>(metadata.term_idx)] < 0) {
                term_start[static_cast<std::size_t>(metadata.term_idx)] = metadata.column;
            }
        }

        int zeroed = 0;
        int considered = 0;
        int mixed_or_unknown = 0;
        const int vardom_count = espace.nbr_unknowns_from_variable_domains();
        const int vardom_per_star = (binary_space != nullptr && vardom_count > 0) ? vardom_count / 2 : 0;

        for (const ColumnMetadata& metadata : columns) {
            bool keep = true;

            if (metadata.column_class == ColumnClass::VarDomain &&
                binary_space != nullptr &&
                vardom_per_star > 0) {
                const bool second_star = metadata.vardom_param >= vardom_per_star;
                const int local_mode = metadata.vardom_param - (second_star ? vardom_per_star : 0);
                const int adapted_domain = second_star ? binary_space->ADAPTED2 : binary_space->ADAPTED1;
                keep = surface_mode_is_z_even(*espace.get_domain(adapted_domain), local_mode);
                ++considered;
            } else if (metadata.term_idx >= 0 &&
                       metadata.term_idx < nterm &&
                       metadata.domain >= 0 &&
                       metadata.column >= 0) {
                const int start = term_start[static_cast<std::size_t>(metadata.term_idx)];
                if (start >= 0) {
                    Tensor probe(term[static_cast<std::size_t>(metadata.term_idx)]->get_val_t(), false);
                    probe.annule_hard();
                    int column_counter = start;
                    espace.get_domain(metadata.domain)->affecte_tau_one_coef(
                        probe, metadata.domain, metadata.column, column_counter);
                    assign_probe_bases(term[static_cast<std::size_t>(metadata.term_idx)]->get_val_t(),
                                       probe, metadata.domain);

                    bool saw_component = false;
                    bool has_kept_component = false;
                    bool has_removed_component = false;
                    for (int comp = 0; comp < probe.get_n_comp(); ++comp) {
                        const Array<int> index = probe.indices(comp);
                        const Val_domain& value = probe(index)(metadata.domain);
                        if (value.check_if_zero())
                            continue;
                        saw_component = true;
                        const bool component_kept =
                            val_domain_column_has_z_symmetric_content(value, index);
                        has_kept_component = has_kept_component || component_kept;
                        has_removed_component = has_removed_component || !component_kept;
                    }

                    if (saw_component) {
                        keep = has_kept_component;
                        if (has_kept_component && has_removed_component)
                            ++mixed_or_unknown;
                        ++considered;
                    }
                }
            }

            if (!keep && metadata.column >= 0 && metadata.column < static_cast<int>(correction.get_nbr())) {
                correction.set(metadata.column) = 0.0;
                ++zeroed;
            }
        }

        if (os != nullptr) {
            *os << "Dense z-symmetric diagnostic projection: zeroed " << zeroed
                << " of " << columns.size()
                << " correction columns; considered " << considered
                << ", mixed kept " << mixed_or_unknown << std::endl;
        }
        return zeroed;
    }

    void System_of_eqs::describe_column_seed(int column, std::ostream& os, int max_coefficients) const
    {
        std::vector<ColumnMetadata> columns;
        classify_columns(columns);
        if (column < 0 || column >= static_cast<int>(columns.size())) {
            os << "    seed: column out of range" << std::endl;
            return;
        }

        const ColumnMetadata& metadata = columns[static_cast<std::size_t>(column)];
        os << "    seed: class=" << static_cast<int>(metadata.column_class)
           << " term=" << metadata.term_idx
           << " var_idx=" << metadata.var_idx
           << " var_double_idx=" << metadata.var_double_idx
           << " vardom_param=" << metadata.vardom_param << std::endl;

        if (metadata.column_class == ColumnClass::ScalarGlobal) {
            os << "      scalar global column has no field seed" << std::endl;
            return;
        }
        if (metadata.column_class == ColumnClass::VarDomain) {
            os << "      variable-domain surface coefficient, local mode="
               << metadata.vardom_param << std::endl;
            return;
        }
        if (metadata.term_idx < 0 || metadata.term_idx >= nterm || metadata.domain < 0) {
            os << "      unmapped field column" << std::endl;
            return;
        }

        std::vector<int> term_start(static_cast<std::size_t>(nterm), -1);
        for (const ColumnMetadata& item : columns) {
            if (item.term_idx >= 0 &&
                item.term_idx < nterm &&
                term_start[static_cast<std::size_t>(item.term_idx)] < 0) {
                term_start[static_cast<std::size_t>(item.term_idx)] = item.column;
            }
        }

        const int start = term_start[static_cast<std::size_t>(metadata.term_idx)];
        if (start < 0) {
            os << "      term start not found" << std::endl;
            return;
        }

        Tensor probe(term[static_cast<std::size_t>(metadata.term_idx)]->get_val_t(), false);
        probe.annule_hard();
        int column_counter = start;
        espace.get_domain(metadata.domain)->affecte_tau_one_coef(
            probe, metadata.domain, metadata.column, column_counter);
        assign_probe_bases(term[static_cast<std::size_t>(metadata.term_idx)]->get_val_t(),
                           probe, metadata.domain);

        int printed = 0;
        for (int comp = 0; comp < probe.get_n_comp(); ++comp) {
            const Array<int> component = probe.indices(comp);
            const Val_domain& value = probe(component)(metadata.domain);
            if (value.check_if_zero())
                continue;
            const Array<double> coefficients = value.get_coef();
            const Base_spectral& base = value.get_base();
            Index position(coefficients.get_dimensions());
            do {
                const double coefficient_value = coefficients(position);
                if (std::fabs(coefficient_value) <= 0.0)
                    continue;
                os << "      component=";
                print_component_indices(os, component);
                os << " coeff=(" << position(0) << "," << position(1) << "," << position(2) << ")"
                   << " value=" << coefficient_value;
                if (base.is_def())
                    print_coefficient_bases(os, base, position);
                else
                    os << " bases=UNDEFINED";
                os << std::endl;
                ++printed;
                if (printed >= max_coefficients)
                    return;
            } while (position.inc());
        }

        if (printed == 0)
            os << "      no nonzero seed coefficients found" << std::endl;
    }

    bool System_of_eqs::do_newton_configured(double precision, double& error, const SolverRuntimeConfig& config)
    {
        set_solver_runtime_config(config);
        // Stash the DOF so save_to_file can stamp it into datasets saved after this
        // solve, for every backend below -- including a warm-started stage that
        // returns converged without ever assembling. nbr_unknowns is final once the
        // variables are registered.
        last_solved_system_dof() = nbr_unknowns;
        switch (config.backend) {
        case NewtonBackend::JfnkMumps:
            return do_newton_jfnk_mumps(precision, error, config);
        case NewtonBackend::JfnkDense:
            return do_newton_jfnk_dense(precision, error, config);
        case NewtonBackend::JfnkSchur:
            return do_newton_jfnk_schur(precision, error, config);
        case NewtonBackend::Mumps:
            return do_newton_sparse(precision, error, config);
        case NewtonBackend::Dense:
        default:
            return do_newton(precision, error, System_of_eqs::SOLVER::NEWTON_RAPHSON);
        }
    }

    void System_of_eqs::print_error_init_diagnostic(const Array<double>& residual, double error) const
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0)
            return;

        // Per-equation absolute errors, to locate the equation owning the
        // largest residual (matches the dense-solver report).
        Array<double> abs_errs(neq_int + neq);
        int pos = 0;
        for (int i = 0; i < neq_int; i++)
            abs_errs.set(i) = std::fabs(residual(pos++));
        for (int i = 0; i < neq; i++) {
            double max_val = 0.0;
            for (int j = 0; j < eq[i]->get_n_cond_tot(); j++)
                max_val = std::max(max_val, std::fabs(residual(pos++)));
            abs_errs.set(neq_int + i) = max_val;
        }

        const int i_max =
            static_cast<int>(std::max_element(abs_errs.data, abs_errs.data + abs_errs.nbr) - abs_errs.data);
        auto [eq_expr, dom, boundary] =
            (i_max < neq_int && neq_int > 0) ? eq_int_list[i_max] : eq_list[i_max - neq_int];
        const char* boundary_str = "INTERIOR";
        if (boundary == 0)
            boundary_str = "INNER";
        else if (boundary == 1)
            boundary_str = "OUTER";
        std::cout << "Error init = " << error << ", Eq: " << eq_expr << " Dom: " << dom << " Boundary: " << boundary_str
                  << std::endl;

        // Full per-integral-equation residual dump (index -> expression -> |r|_inf),
        // to read the eq_int layout / attribute the residual to a specific
        // constraint. Diagnostic only; env-gated, zero overhead when off.
        if (env_flag_enabled("EQ_INIT_DUMP", false)) {
            std::cout << "--- eq_int residual breakdown (index : expr : dom : |r|_inf) ---" << std::endl;
            for (int i = 0; i < neq_int; i++) {
                auto [e_expr, e_dom, e_bnd] = eq_int_list[static_cast<std::size_t>(i)];
                (void)e_bnd;
                std::cout << "  eq_int[" << i << "]  " << e_expr << "  dom=" << e_dom
                          << "  |r|=" << abs_errs(i) << std::endl;
            }
        }

        abs_errs.delete_data();
    }

    void System_of_eqs::print_dof_rank_diagnostic(int n, int m, int nproc) const
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0)
            return;

        int bsize = 256;
        while (nproc * bsize > m)
            bsize /= 2;
        std::cout << "DOF / rank = " << n << " x " << m << " / " << nproc << " = " << 1.0 * n * m / nproc
                  << "  Block size: " << bsize << std::endl;
        if (bsize < 64)
            std::cout << "Block size is " << bsize << " (< 64); consider reducing cores." << std::endl;
    }

    void System_of_eqs::print_sparse_dof_rank_diagnostic(int n, int m, int nproc, long long nnz) const
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank != 0)
            return;

        std::cout << "DOF / rank = " << n << " x " << m << " / " << nproc << " = "
                  << 1.0 * n * m / nproc << "  nnz: " << nnz << std::endl;
    }

    bool System_of_eqs::do_newton(double precision, double& error, System_of_eqs::SOLVER solver)
    {
        last_solved_system_dof() = nbr_unknowns;  // stamped into datasets saved after this solve
        int rank, nproc;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);

        Array<double> second(sec_member());
        Array<double> abs_errs(neq_int + neq);

        // Calculate absolute errors
        int pos = 0;
        for (int i = 0; i < neq_int; i++)
            abs_errs.set(i) = std::fabs(second(pos++));

        for (int i = 0; i < neq; i++) {
            double max_val = 0.0;
            for (int j = 0; j < eq[i]->get_n_cond_tot(); j++)
                max_val = std::max(max_val, std::fabs(second(pos++)));
            abs_errs.set(neq_int + i) = max_val;
        }

        error = max(abs_errs);

        // Check for NaN and report
        for (std::size_t i = 0; i < abs_errs.get_nbr(); ++i) {
            if (std::isnan(abs_errs(static_cast<int>(i)))) {
                std::cout << "NaN detected in residuals at equation row: " << i << std::endl;
                break;
            }
        }

        print_error_init_diagnostic(second, error);

        if (error < precision)
            return true;

        abs_errs.delete_data();
        int m = second.get_size(0), n = nbr_unknowns;

        if (n != m) {
            if (rank == 0) {
                auto cpd = conditions_per_domain();
                auto upd = unknowns_per_domain();
                for (int d = dom_min; d <= dom_max; ++d)
                    std::cerr << "[DOFDUMP] dom " << d << " cond=" << cpd[d] << " unk=" << upd[d]
                              << " (c-u)=" << (cpd[d] - upd[d]) << std::endl;
                std::cerr << "[DOFDUMP] neq_int=" << neq_int << " nvar_double=" << nvar_double
                          << " m=" << m << " n=" << n << std::endl;
            }
            std::ostringstream oss;
            oss << "Equations (" << m << ") != unknowns (" << n << ")" << std::endl;
            KADATH_THROW(oss.str());
        }

        print_dof_rank_diagnostic(n, m, nproc);

        // Setup block size
        int zero_i = 0, one_i = 1;
        int bsize = 256;
        while (nproc * bsize > m)
            bsize /= 2;

        // Initialize column cost cache (indices only, costs populated during first build)
        auto& cost_cache = newton_column_cost_state;
        cost_cache.reset(n);

        const bool first_iteration = !cost_cache.initialized;
        if (first_iteration) {
            cost_cache.costs.resize(n, 0.0);
            cost_cache.indices.resize(n);
            std::iota(cost_cache.indices.begin(), cost_cache.indices.end(), 0); // Natural order
            if (rank == 0)
                std::cout << "First iteration: using natural column order, will profile costs" << std::endl;
        }

        // Create grids
        int ictxt_in, nprow_in = 1, npcol_in = nproc, prow_in, pcol_in;
        sl_init_(&ictxt_in, &nprow_in, &npcol_in);
        blacs_gridinfo_(&ictxt_in, &nprow_in, &npcol_in, &prow_in, &pcol_in);

        int ictxt, nprow, npcol, prow, pcol;
        split_process_grid(nproc, nprow, npcol);
        sl_init_(&ictxt, &nprow, &npcol);
        blacs_gridinfo_(&ictxt, &nprow, &npcol, &prow, &pcol);

        // Setup B vector
        int nrow_B = numroc_(&m, &bsize, &prow, &zero_i, &nprow);
        int ncol_B_loc = numroc_(&one_i, &bsize, &pcol, &zero_i, &npcol);
        Array<double> B(nrow_B * ncol_B_loc);

        if (pcol == 0) {
            int idx = 0;
            for (int gb = prow; gb < (m + bsize - 1) / bsize; gb += nprow) {
                int start = gb * bsize;
                int limit = std::min(bsize, m - start);
                for (int k = 0; k < limit; ++k)
                    B.set(idx++) = second(start + k);
            }
        }
        second.delete_data();

        // Generate Jacobian columns - use cost-balanced round-robin distribution
        int bsize_in = 1;
        int nrow_in = numroc_(&m, &bsize_in, &prow_in, &zero_i, &nprow_in);

        // Distribute sorted columns in round-robin: rank i gets indices [i, i+nproc, i+2*nproc, ...]
        // This balances load since sorted array alternates high-cost columns across ranks
        int ncol_in = numroc_(&n, &bsize_in, &pcol_in, &zero_i, &npcol_in);

        std::vector<int> globcol_in;
        globcol_in.reserve(ncol_in);
        for (int i = pcol_in; i < n; i += npcol_in)
            globcol_in.push_back(cost_cache.indices[i]);

        Array2d<double> J_in(static_cast<std::size_t>(nrow_in), static_cast<std::size_t>(ncol_in));
        std::vector<double> local_costs(first_iteration ? n : 0, 0.0);
        reset_do_col_J_cache();

        auto t0 = std::chrono::system_clock::now();
        for (int j = 0; j < ncol_in; ++j) {
            int global_col = globcol_in[j];
            auto t_col_start = std::chrono::steady_clock::now();
            Array<double> col(do_col_J(global_col));
            if (first_iteration) {
                auto t_col_end = std::chrono::steady_clock::now();
                local_costs[global_col] = std::chrono::duration<double>(t_col_end - t_col_start).count();
            }
            std::memcpy(&J_in(0, static_cast<std::size_t>(j)), col.get_data(),
                        static_cast<std::size_t>(nrow_in) * sizeof(double));
        }
        double t_col = elapsed_seconds(t0);

        if (solver_runtime_config.diagnostics.timing) {
            dump_do_col_J_profile();
        }
        // Env-gated Ope_mult profile (OPE_MULT_PROFILE=1); no-op otherwise.
        dump_ope_mult_profile();
        reset_ope_mult_profile();
        // Env-gated Ope_eq action census (OPE_ACTION_PROFILE=1); no-op otherwise.
        dump_ope_action_profile();
        reset_ope_action_profile();

        // After first iteration build: gather costs (sorting deferred until after un-permutation)
        if (first_iteration) {
            MPI_Allreduce(local_costs.data(), cost_cache.costs.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }

        // Setup descriptors and redistribute
        int info;
        Array<int> J_desc_in(9);
        descinit_(J_desc_in.set_data(), &m, &n, &bsize_in, &bsize_in, &zero_i, &zero_i, &ictxt_in, &nrow_in, &info);

        int nrow = numroc_(&m, &bsize, &prow, &zero_i, &nprow);
        int ncol = numroc_(&n, &bsize, &pcol, &zero_i, &npcol);
        Array<double> J(ncol, nrow);
        Array<int> J_desc(9);
        descinit_(J_desc.set_data(), &m, &n, &bsize, &bsize, &zero_i, &zero_i, &ictxt, &nrow, &info);

        t0 = std::chrono::system_clock::now();
        pdgemr2d_(&m, &n, J_in.data(), &one_i, &one_i, J_desc_in.set_data(), J.set_data(), &one_i, &one_i,
                  J_desc.set_data(), &ictxt);
        double t_redist = elapsed_seconds(t0);

        Array<int> B_desc(9);
        descinit_(B_desc.set_data(), &m, &one_i, &bsize, &one_i, &zero_i, &zero_i, &ictxt, &nrow_B, &info);

        // Report timing helper
        auto report_timing = [&](const char* label, double t) {
            double stats[3];
            MPI_Reduce(&t, &stats[0], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&t, &stats[1], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&t, &stats[2], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0)
                std::cout << label << ": " << stats[2] / nproc << " / " << stats[1] << " / " << stats[0]
                          << " seconds (avg/min/max)" << std::endl;
        };

        report_timing("Load Jacobian", t_col);
        report_timing("Redistribute", t_redist);

        // Solve
        Array<double>& X_sol = B;
        double t_solve = 0;
        if (solver == System_of_eqs::SOLVER::NEWTON_RAPHSON) {
            Array<int> ipiv(n);
            t0 = std::chrono::system_clock::now();
            pdgesv_(&n, &one_i, J.set_data(), &one_i, &one_i, J_desc.set_data(), ipiv.set_data(), B.set_data(), &one_i,
                    &one_i, B_desc.set_data(), &info);
            t_solve = elapsed_seconds(t0);
        }

        if (info != 0) {
            if (rank == 0)
                std::cout << (info > 0 ? "\nJacobian is singular at column " + std::to_string(info - 1)
                                       : "pdgesv error: illegal argument " + std::to_string(-info))
                          << std::endl;
            KADATH_THROW("Aborted");
        }

        // Gather solution
        std::vector<int> n_elements(nproc), disp(nproc, 0);
        int offset = 0;
        for (int i = 0; i < nprow; ++i) {
            for (int j = 0; j < npcol; ++j) {
                int r = i * npcol + j;
                n_elements[r] = (j == 0) ? numroc_(&n, &bsize, &i, &zero_i, &nprow) : 0;
                disp[r] = offset;
                offset += n_elements[r];
            }
        }

        std::vector<double> recv(rank == 0 ? n : 0);

        t0 = std::chrono::system_clock::now();
        Array<double> X(n);
        MPI_Gatherv(X_sol.set_data(), n_elements[rank], MPI_DOUBLE, rank == 0 ? recv.data() : nullptr,
                    n_elements.data(), disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Reconstruct block-cyclic -> global permuted vector
            std::vector<double> x_perm(static_cast<std::size_t>(n), 0.0);
            int nblocks = (n + bsize - 1) / bsize;
            for (int kb = 0; kb < nblocks; ++kb) {
                int proc_row = kb % nprow;
                int r = proc_row * npcol;
                int src_off = disp[r] + (kb / nprow) * bsize;
                int dst_off = kb * bsize;
                int copy_size = std::min(bsize, n - dst_off);
                std::memcpy(x_perm.data() + dst_off, recv.data() + src_off,
                            static_cast<std::size_t>(copy_size) * sizeof(double));
            }

            // Un-permute solution (uses indices as they were when Jacobian was built)
            for (int i = 0; i < n; ++i)
                X.set(cost_cache.indices[i]) = x_perm[static_cast<std::size_t>(i)];
        }

        MPI_Bcast(X.set_data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        const double step_scale = env_double_value("DENSE_STEP_SCALE", 1.0);
        if (step_scale != 1.0) {
            for (int i = 0; i < n; ++i)
                X.set(i) *= step_scale;
            if (rank == 0)
                std::cout << "Dense Newton step scaled by " << step_scale << std::endl;
        }
        if (env_flag_enabled("DENSE_PROJECT_ZSYM_ALIGNED")) {
            project_z_symmetric_diagnostic_correction(X, rank == 0 ? &std::cout : nullptr);
        }
        zero_selected_dense_corrections(X, n);
        double t_gather = elapsed_seconds(t0);

        // Now sort indices for future iterations (after un-permutation is done)
        if (first_iteration) {
            std::sort(cost_cache.indices.begin(), cost_cache.indices.end(),
                      [&](int a, int b) { return cost_cache.costs[a] > cost_cache.costs[b]; });
            cost_cache.initialized = true;
            if (rank == 0)
                std::cout << "Cost profiling complete, sorted for future iterations" << std::endl;
        }

        blacs_gridexit_(&ictxt_in);
        blacs_gridexit_(&ictxt);

        int conte = 0;
        espace.xx_to_vars_variable_domains(this, X, conte);
        xx_to_vars_delta(X, conte);

        report_timing("Inverting the matrix", t_solve);
        report_timing("Gather+reorder+bcast", t_gather);

        return false;
    }

}
