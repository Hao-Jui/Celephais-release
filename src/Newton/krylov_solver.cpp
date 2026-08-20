#include "Linear_algebra/krylov_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
    // Diagnostic-only per-iteration residual trace (GMRES_ITER_TRACE=1).
    // Read once; zero overhead when unset. Used by the p-coarse PC probe A/B to
    // plot geometric-vs-plateau convergence curves.
    bool gmres_iter_trace_enabled()
    {
        static const bool enabled = []() {
            const char* raw = std::getenv("GMRES_ITER_TRACE");
            return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
        }();
        return enabled;
    }
} // namespace

namespace
{
    using SteadyClock = std::chrono::steady_clock;

    inline SteadyClock::time_point timing_now()
    {
        return SteadyClock::now();
    }

    inline double timing_elapsed(SteadyClock::time_point start)
    {
        const auto delta = SteadyClock::now() - start;
        return std::chrono::duration<double>(delta).count();
    }
} // namespace

namespace Kadath
{
    namespace
    {
        double dot(const std::vector<double>& lhs, const std::vector<double>& rhs)
        {
            double value = 0.0;
            for (std::size_t i = 0; i < lhs.size(); ++i) {
                value += lhs[i] * rhs[i];
            }
            return value;
        }

        double vector_norm(const std::vector<double>& values)
        {
            return std::sqrt(dot(values, values));
        }

        template <typename Action>
        void run_timed(double* accumulator, Action&& action)
        {
            if (accumulator == nullptr) {
                action();
                return;
            }
            const auto start = timing_now();
            action();
            *accumulator += timing_elapsed(start);
        }

        // Row-packed storage for an upper-Hessenberg matrix. Row `r` stores only
        // columns max(0, r - 1)..m-1, so the live rotated matrix is contiguous and
        // uses m(m+3)/2 doubles rather than an (m+1) x m rectangle.
        class PackedHessenberg
        {
        public:
            explicit PackedHessenberg(int columns)
                : columns_(static_cast<std::size_t>(columns)),
                  values_(columns_ * (columns_ + 3) / 2, 0.0)
            {
            }

            double& operator()(int row, int column)
            {
                return values_[offset(static_cast<std::size_t>(row),
                                      static_cast<std::size_t>(column))];
            }

            double operator()(int row, int column) const
            {
                return values_[offset(static_cast<std::size_t>(row),
                                      static_cast<std::size_t>(column))];
            }

        private:
            std::size_t offset(std::size_t row, std::size_t column) const
            {
                if (row == 0)
                    return column;
                const std::size_t row_offset =
                    columns_ + (row - 1) * (columns_ + 1) - (row - 1) * row / 2;
                return row_offset + column - (row - 1);
            }

            std::size_t columns_;
            std::vector<double> values_;
        };

        // Modified Gram-Schmidt. The last subtraction also accumulates ||w||^2:
        // each updated entry and the norm reduction keep the same arithmetic and
        // entry order as the former separate passes, with or without timing.
        double modified_gram_schmidt(std::vector<double>& operator_vector,
                                     const std::vector<std::vector<double>>& arnoldi_basis,
                                     PackedHessenberg& hessenberg,
                                     int iteration)
        {
            double norm_squared = 0.0;
            for (int basis_index = 0; basis_index <= iteration; ++basis_index) {
                const double projection =
                    dot(operator_vector, arnoldi_basis[static_cast<std::size_t>(basis_index)]);
                hessenberg(basis_index, iteration) = projection;
                if (basis_index == iteration) {
                    for (std::size_t entry = 0; entry < operator_vector.size(); ++entry) {
                        operator_vector[entry] -=
                            projection *
                            arnoldi_basis[static_cast<std::size_t>(basis_index)][entry];
                        norm_squared += operator_vector[entry] * operator_vector[entry];
                    }
                } else {
                    for (std::size_t entry = 0; entry < operator_vector.size(); ++entry) {
                        operator_vector[entry] -=
                            projection *
                            arnoldi_basis[static_cast<std::size_t>(basis_index)][entry];
                    }
                }
            }
            return std::sqrt(norm_squared);
        }

        // The exact protective test used before triangular back-substitution. A
        // prefix is well posed only when every diagonal is strictly above
        // sqrt(eps) times the largest diagonal magnitude in that prefix.
        bool rotated_hessenberg_prefix_is_well_posed(const PackedHessenberg& hessenberg,
                                                     int columns)
        {
            constexpr double kRelativeRankTolerance = 1.4901161193847656e-08; // sqrt(eps)
            double max_abs_diagonal = 0.0;
            for (int row = 0; row < columns; ++row) {
                max_abs_diagonal = std::max(
                    max_abs_diagonal,
                    std::abs(hessenberg(row, row)));
            }
            const double rank_floor = kRelativeRankTolerance * max_abs_diagonal;
            for (int row = columns; row-- > 0;) {
                if (std::abs(hessenberg(row, row)) <= rank_floor)
                    return false;
            }
            return true;
        }

        // Back-substitution is deferred until an iterate is emitted. The caller
        // has already accepted this prefix with the exact rank guard above.
        void back_substitute_rotated_hessenberg(const PackedHessenberg& hessenberg,
                                                int columns,
                                                const std::vector<double>& rotated_rhs,
                                                std::vector<double>& coefficients)
        {
            coefficients.resize(static_cast<std::size_t>(columns));
            for (int row = columns; row-- > 0;) {
                double value = rotated_rhs[static_cast<std::size_t>(row)];
                for (int column = row + 1; column < columns; ++column) {
                    value -= hessenberg(row, column) *
                             coefficients[static_cast<std::size_t>(column)];
                }
                coefficients[static_cast<std::size_t>(row)] = value / hessenberg(row, row);
            }
        }

        void apply_solution_update(std::vector<double>& solution,
                                   const std::vector<std::vector<double>>& solution_basis,
                                   const std::vector<double>& coefficients)
        {
            constexpr std::size_t kTileEntries = 2048;
            for (std::size_t tile_begin = 0; tile_begin < solution.size();
                 tile_begin += kTileEntries) {
                const std::size_t tile_end =
                    std::min(solution.size(), tile_begin + kTileEntries);
                std::fill(solution.begin() + static_cast<std::ptrdiff_t>(tile_begin),
                          solution.begin() + static_cast<std::ptrdiff_t>(tile_end),
                          0.0);
                for (std::size_t basis_index = 0; basis_index < coefficients.size();
                     ++basis_index) {
                    for (std::size_t entry = tile_begin; entry < tile_end; ++entry) {
                        solution[entry] +=
                            coefficients[basis_index] * solution_basis[basis_index][entry];
                    }
                }
            }
        }
    } // namespace

    GmresStatus right_preconditioned_gmres(const std::vector<double>& b,
                                           std::vector<double>& x,
                                           const KrylovOperator& matvec,
                                           const KrylovOperator& right_preconditioner,
                                           const GmresConfig& config)
    {
        if (b.empty() || x.size() != b.size() || !matvec || config.max_iters <= 0 ||
            config.tolerance <= 0.0) {
            return {GmresStatus::Code::InvalidInput, false, 0, 0.0};
        }

        GmresTiming* const timing = config.timing;

        const std::size_t dimension = b.size();
        const int max_iters = std::min<int>(config.max_iters, static_cast<int>(dimension));
        double residual_norm_0;
        run_timed(timing != nullptr ? &timing->vector_norm_seconds : nullptr,
                  [&]() { residual_norm_0 = vector_norm(b); });
        if (residual_norm_0 <= config.tolerance) {
            std::fill(x.begin(), x.end(), 0.0);
            return {GmresStatus::Code::Converged, true, 0, residual_norm_0};
        }
        if (config.residual_history != nullptr) {
            config.residual_history->reserve(
                config.residual_history->size() + static_cast<std::size_t>(max_iters));
        }

        std::vector<std::vector<double>> arnoldi_basis;
        arnoldi_basis.reserve(static_cast<std::size_t>(max_iters) + 1);
        arnoldi_basis.emplace_back(dimension, 0.0);
        std::vector<std::vector<double>> preconditioned_basis;
        if (right_preconditioner)
            preconditioned_basis.reserve(static_cast<std::size_t>(max_iters));
        PackedHessenberg hessenberg(max_iters);

        // Diagnostic spectral capture (both pointers set): a separate copy of the
        // RAW Hessenberg entries is retained because the Givens QR below overwrites
        // `hessenberg` in place with the rotated upper-triangular R. Allocated only
        // when requested -> zero overhead otherwise.
        const bool capture_spectral =
            config.arnoldi_basis_out != nullptr && config.hessenberg_raw_out != nullptr;
        std::vector<std::vector<double>> hessenberg_raw;
        if (capture_spectral) {
            hessenberg_raw.assign(
                static_cast<std::size_t>(max_iters) + 1,
                std::vector<double>(static_cast<std::size_t>(max_iters), 0.0));
        }
        // Move the retained Krylov data out to the caller. Called only just before a
        // Converged / MaxIterations return, after any apply_solution_update, so the
        // move is safe even when the null-preconditioner path uses this basis directly.
        const auto emit_spectral = [&](int columns) {
            if (!capture_spectral)
                return;
            if (columns >= 0 && columns < static_cast<int>(arnoldi_basis.size()))
                arnoldi_basis.resize(static_cast<std::size_t>(columns));
            *config.arnoldi_basis_out = std::move(arnoldi_basis);
            *config.hessenberg_raw_out = std::move(hessenberg_raw);
        };

        for (std::size_t entry = 0; entry < dimension; ++entry) {
            arnoldi_basis[0][entry] = b[entry] / residual_norm_0;
        }

        // Incremental Givens QR of the Hessenberg matrix: rotations are stored
        // per column, the rotated Hessenberg (upper-triangular R) overwrites
        // `hessenberg` in place, and `rotated_rhs` carries Q^T (residual_norm_0 e_1).
        // |rotated_rhs[k+1]| is the exact GMRES residual after iteration k.
        std::vector<double> givens_cos(static_cast<std::size_t>(max_iters), 0.0);
        std::vector<double> givens_sin(static_cast<std::size_t>(max_iters), 0.0);
        std::vector<double> rotated_rhs(static_cast<std::size_t>(max_iters) + 1, 0.0);
        rotated_rhs[0] = residual_norm_0;

        double last_residual = residual_norm_0;
        bool have_iterate = false;
        int last_good_columns = 0;

        // Materialize and apply only the last prefix accepted by the per-iteration
        // rank guard. Earlier R and rotated-rhs entries are immutable, so this is
        // the same triangular solve formerly repeated after every Arnoldi step.
        const auto apply_best_iterate = [&]() {
            std::vector<double> coefficients;
            run_timed(timing != nullptr ? &timing->least_squares_seconds : nullptr,
                      [&]() {
                          back_substitute_rotated_hessenberg(
                              hessenberg, last_good_columns, rotated_rhs, coefficients);
                      });
            const auto& solution_basis =
                right_preconditioner ? preconditioned_basis : arnoldi_basis;
            run_timed(timing != nullptr ? &timing->update_seconds : nullptr,
                      [&]() { apply_solution_update(x, solution_basis, coefficients); });
        };

        for (int iteration = 0; iteration < max_iters; ++iteration) {
            std::vector<double> preconditioned_vector;
            const std::vector<double>* operator_input =
                &arnoldi_basis[static_cast<std::size_t>(iteration)];
            if (right_preconditioner) {
                run_timed(timing != nullptr ? &timing->precondition_seconds : nullptr,
                          [&]() {
                              right_preconditioner(
                                  arnoldi_basis[static_cast<std::size_t>(iteration)],
                                  preconditioned_vector);
                          });
                if (timing != nullptr)
                    ++timing->preconditions;
                if (preconditioned_vector.size() != dimension) {
                    if (have_iterate)
                        apply_best_iterate();
                    return {GmresStatus::Code::InvalidInput,
                            false,
                            iteration,
                            last_residual};
                }
                operator_input = &preconditioned_vector;
            }

            std::vector<double> operator_vector;
            run_timed(timing != nullptr ? &timing->matvec_seconds : nullptr,
                      [&]() { matvec(*operator_input, operator_vector); });
            if (timing != nullptr)
                ++timing->matvecs;
            if (operator_vector.size() != dimension) {
                if (have_iterate)
                    apply_best_iterate();
                return {GmresStatus::Code::InvalidInput, false, iteration, last_residual};
            }
            if (right_preconditioner)
                preconditioned_basis.push_back(std::move(preconditioned_vector));

            double subdiagonal_norm;
            run_timed(timing != nullptr ? &timing->orthog_seconds : nullptr,
                      [&]() {
                          subdiagonal_norm = modified_gram_schmidt(
                              operator_vector, arnoldi_basis, hessenberg, iteration);
                      });
            hessenberg(iteration + 1, iteration) = subdiagonal_norm;
            // Snapshot the raw Hessenberg column now, before the Givens rotations
            // below overwrite `hessenberg` in place with the rotated R.
            if (capture_spectral) {
                for (int row = 0; row <= iteration + 1; ++row)
                    hessenberg_raw[static_cast<std::size_t>(row)]
                                  [static_cast<std::size_t>(iteration)] =
                        hessenberg(row, iteration);
            }

            // Incremental QR update: rotate the new Hessenberg column by the
            // previous Givens rotations, then build the rotation zeroing the
            // subdiagonal entry. |rotated_rhs[iteration + 1]| is the exact
            // GMRES residual norm after this iteration. residual_estimate holds
            // it locally and is committed to last_residual only together with
            // last_good_columns (when the prefix rank guard succeeds), so every
            // exit returns a residual consistent with the applied iterate; it carries
            // the previous value when the rotation is degenerate.
            double residual_estimate = last_residual;
            bool rotation_ok;
            run_timed(timing != nullptr ? &timing->least_squares_seconds : nullptr,
                      [&]() {
                          for (int previous = 0; previous < iteration; ++previous) {
                              double& upper_entry = hessenberg(previous, iteration);
                              double& lower_entry = hessenberg(previous + 1, iteration);
                              const double rotated_upper =
                                  givens_cos[static_cast<std::size_t>(previous)] * upper_entry +
                                  givens_sin[static_cast<std::size_t>(previous)] * lower_entry;
                              lower_entry =
                                  -givens_sin[static_cast<std::size_t>(previous)] * upper_entry +
                                  givens_cos[static_cast<std::size_t>(previous)] * lower_entry;
                              upper_entry = rotated_upper;
                          }
                          const double diagonal_value = hessenberg(iteration, iteration);
                          const double rotation_magnitude =
                              std::hypot(diagonal_value, subdiagonal_norm);
                          rotation_ok =
                              rotation_magnitude > std::numeric_limits<double>::epsilon();
                          if (rotation_ok) {
                              givens_cos[static_cast<std::size_t>(iteration)] =
                                  diagonal_value / rotation_magnitude;
                              givens_sin[static_cast<std::size_t>(iteration)] =
                                  subdiagonal_norm / rotation_magnitude;
                              hessenberg(iteration, iteration) = rotation_magnitude;
                              hessenberg(iteration + 1, iteration) = 0.0;
                              rotated_rhs[static_cast<std::size_t>(iteration) + 1] =
                                  -givens_sin[static_cast<std::size_t>(iteration)] *
                                  rotated_rhs[static_cast<std::size_t>(iteration)];
                              rotated_rhs[static_cast<std::size_t>(iteration)] =
                                  givens_cos[static_cast<std::size_t>(iteration)] *
                                  rotated_rhs[static_cast<std::size_t>(iteration)];
                              residual_estimate = std::abs(
                                  rotated_rhs[static_cast<std::size_t>(iteration) + 1]);
                          }
                      });

            if (gmres_iter_trace_enabled()) {
                // |rotated_rhs[iteration + 1]| is the exact GMRES residual after
                // this iteration (residual_estimate carries it).
                std::cout << "gmres iter=" << iteration
                          << " resid=" << residual_estimate << '\n';
            }
            // Optional diagnostic capture: one residual per executed iteration.
            if (config.residual_history != nullptr)
                config.residual_history->push_back(residual_estimate);

            // Validate the current prefix without materializing its coefficients.
            // If a later column is rank deficient, every exit path solves and
            // applies this recorded last-good prefix.
            bool ls_ok = rotation_ok;
            if (rotation_ok) {
                run_timed(timing != nullptr ? &timing->least_squares_seconds : nullptr,
                          [&]() {
                              ls_ok = rotated_hessenberg_prefix_is_well_posed(
                                  hessenberg, iteration + 1);
                          });
            }
            if (ls_ok) {
                // Commit the residual together with the coefficients it belongs
                // to, so every exit path reports a residual consistent with the
                // iterate actually applied to x.
                last_residual = residual_estimate;
                have_iterate = true;
                last_good_columns = iteration + 1;
                if (residual_estimate <= config.tolerance) {
                    apply_best_iterate();
                    emit_spectral(iteration + 1);
                    return {GmresStatus::Code::Converged, true, iteration + 1, last_residual};
                }
                if (subdiagonal_norm <= std::numeric_limits<double>::epsilon()) {
                    apply_best_iterate();
                    return {GmresStatus::Code::Breakdown, false, iteration + 1, last_residual};
                }
            } else if (subdiagonal_norm <= std::numeric_limits<double>::epsilon()) {
                if (have_iterate) {
                    apply_best_iterate();
                } else {
                    // No well-posed iterate yet: a defined zero correction
                    // beats leaving the caller's input vector untouched.
                    std::fill(x.begin(), x.end(), 0.0);
                }
                return {GmresStatus::Code::Breakdown, false, iteration + 1, last_residual};
            }

            // No exit consumes v_{iteration+1}; only now normalize the existing
            // operator buffer in place and retain it for the next iteration.
            if (iteration + 1 < max_iters) {
                for (double& entry : operator_vector)
                    entry /= subdiagonal_norm;
                arnoldi_basis.push_back(std::move(operator_vector));
            }
        }

        if (have_iterate) {
            apply_best_iterate();
        } else {
            std::fill(x.begin(), x.end(), 0.0);
        }
        emit_spectral(max_iters);
        return {GmresStatus::Code::MaxIterations, false, max_iters, last_residual};
    }
} // namespace Kadath
