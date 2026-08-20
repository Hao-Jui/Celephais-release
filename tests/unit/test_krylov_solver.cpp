#include "Linear_algebra/krylov_solver.hpp"
#include "Linear_algebra/jacobian_parity_mask.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <vector>

using Catch::Approx;
using Kadath::GmresConfig;
using Kadath::GmresStatus;
using Kadath::GmresTiming;
using Kadath::right_preconditioned_gmres;

TEST_CASE("Reduced GMRES keeps distinct full row and column coordinates",
          "[krylov][selection_plan][jfnk-mumps]")
{
    const std::array<std::array<double, 4>, 4> full_matrix{{
        {{0.0, 2.0, 0.0, 0.0}},
        {{4.0, 0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 5.0}},
        {{0.0, 1.0, 3.0, 0.0}},
    }};
    const std::array<double, 4> full_rhs{{4.0, 0.0, 0.0, 7.0}};
    const std::vector<int> selected_rows{0, 3};
    const std::vector<int> selected_columns{1, 2};

    const Kadath::JacobianSelectedValues gathered_rhs =
        Kadath::gather_jacobian_selected_values(full_rhs, selected_rows);
    REQUIRE(gathered_rhs);
    std::vector<double> solution(2, 0.0);
    const auto reduced_matvec =
        [&](const std::vector<double>& reduced_input,
            std::vector<double>& reduced_output) {
            const Kadath::JacobianSelectedValues scattered =
                Kadath::scatter_jacobian_selected_values(
                    reduced_input, 4, selected_columns);
            REQUIRE(scattered);
            std::array<double, 4> full_output{};
            for (std::size_t row = 0; row < full_matrix.size(); ++row) {
                for (std::size_t column = 0; column < full_matrix.size();
                     ++column) {
                    full_output[row] += full_matrix[row][column] *
                        scattered.values[column];
                }
            }
            const Kadath::JacobianSelectedValues gathered =
                Kadath::gather_jacobian_selected_values(
                    full_output, selected_rows);
            REQUIRE(gathered);
            reduced_output = gathered.values;
        };

    GmresConfig config;
    config.max_iters = 4;
    config.tolerance = 1e-12;
    const GmresStatus status = right_preconditioned_gmres(
        gathered_rhs.values, solution, reduced_matvec, nullptr, config);

    REQUIRE(status.converged);
    REQUIRE(solution[0] == Approx(2.0));
    REQUIRE(solution[1] == Approx(5.0 / 3.0));
    const Kadath::JacobianSelectedValues full_solution =
        Kadath::scatter_jacobian_selected_values(
            solution, 4, selected_columns);
    REQUIRE(full_solution);
    REQUIRE(full_solution.values[0] == Approx(0.0));
    REQUIRE(full_solution.values[1] == Approx(2.0));
    REQUIRE(full_solution.values[2] == Approx(5.0 / 3.0));
    REQUIRE(full_solution.values[3] == Approx(0.0));
}

TEST_CASE("GMRES solves a diagonal system without preconditioner", "[krylov]")
{
    const std::vector<double> rhs{2.0, 6.0};
    std::vector<double> solution(2, 0.0);

    const auto matvec = [](const std::vector<double>& in, std::vector<double>& out) {
        out = {2.0 * in[0], 3.0 * in[1]};
    };

    GmresConfig config;
    config.max_iters = 4;
    config.tolerance = 1e-12;

    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE(status.converged);
    REQUIRE(status.code == GmresStatus::Code::Converged);
    REQUIRE(solution[0] == Approx(1.0));
    REQUIRE(solution[1] == Approx(2.0));
}

TEST_CASE("GMRES treats solution as output-only", "[krylov]")
{
    const std::vector<double> rhs{2.0, 6.0};
    std::vector<double> solution{99.0, -99.0};

    const auto matvec = [](const std::vector<double>& in, std::vector<double>& out) {
        out = {2.0 * in[0], 3.0 * in[1]};
    };

    GmresConfig config;
    config.max_iters = 4;
    config.tolerance = 1e-12;

    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE(status.converged);
    REQUIRE(solution[0] == Approx(1.0));
    REQUIRE(solution[1] == Approx(2.0));
}

TEST_CASE("GMRES applies right preconditioner in solution space", "[krylov]")
{
    const std::vector<double> rhs{2.0, 6.0};
    std::vector<double> solution(2, 0.0);
    int preconditioner_calls = 0;

    const auto matvec = [](const std::vector<double>& in, std::vector<double>& out) {
        out = {2.0 * in[0], 3.0 * in[1]};
    };
    const auto preconditioner = [&preconditioner_calls](const std::vector<double>& in,
                                                        std::vector<double>& out) {
        ++preconditioner_calls;
        out = {0.5 * in[0], in[1] / 3.0};
    };

    GmresConfig config;
    config.max_iters = 2;
    config.tolerance = 1e-12;

    const auto status = right_preconditioned_gmres(rhs, solution, matvec, preconditioner, config);

    REQUIRE(status.converged);
    REQUIRE(preconditioner_calls == 1);
    REQUIRE(solution[0] == Approx(1.0));
    REQUIRE(solution[1] == Approx(2.0));
}

TEST_CASE("GMRES timing attributes callbacks and inner phases without changing the solve",
          "[krylov][timing]")
{
    const std::vector<double> rhs{2.0, 6.0};
    std::vector<double> solution(2, 0.0);
    int matvec_calls = 0;
    int preconditioner_calls = 0;

    const auto matvec = [&matvec_calls](const std::vector<double>& in,
                                        std::vector<double>& out) {
        ++matvec_calls;
        out = {2.0 * in[0], 3.0 * in[1]};
    };
    const auto preconditioner = [&preconditioner_calls](
                                    const std::vector<double>& in,
                                    std::vector<double>& out) {
        ++preconditioner_calls;
        out = {0.5 * in[0], in[1] / 3.0};
    };

    GmresTiming timing;
    GmresConfig config;
    config.max_iters = 2;
    config.tolerance = 1e-12;
    config.timing = &timing;

    const auto status =
        right_preconditioned_gmres(rhs, solution, matvec, preconditioner, config);

    REQUIRE(status.converged);
    REQUIRE(solution[0] == Approx(1.0));
    REQUIRE(solution[1] == Approx(2.0));
    REQUIRE(timing.matvecs == matvec_calls);
    REQUIRE(timing.preconditions == preconditioner_calls);
    REQUIRE(timing.matvecs == 1);
    REQUIRE(timing.preconditions == 1);

    const std::array<double, 6> phase_seconds = {
        timing.precondition_seconds,
        timing.matvec_seconds,
        timing.orthog_seconds,
        timing.vector_norm_seconds,
        timing.least_squares_seconds,
        timing.update_seconds};
    for (double seconds : phase_seconds) {
        REQUIRE(std::isfinite(seconds));
        REQUIRE(seconds >= 0.0);
    }

    const int timed_matvec_calls = matvec_calls;
    const int timed_preconditioner_calls = preconditioner_calls;
    std::vector<double> untimed_solution(2, 0.0);
    config.timing = nullptr;
    const auto untimed_status =
        right_preconditioned_gmres(rhs, untimed_solution, matvec, preconditioner, config);
    REQUIRE(untimed_status.code == status.code);
    REQUIRE(untimed_status.iterations == status.iterations);
    REQUIRE(untimed_status.residual_norm == status.residual_norm);
    REQUIRE(untimed_solution == solution);
    REQUIRE(matvec_calls - timed_matvec_calls == timed_matvec_calls);
    REQUIRE(preconditioner_calls - timed_preconditioner_calls ==
            timed_preconditioner_calls);
}

TEST_CASE("GMRES rejects bad preconditioner output size", "[krylov]")
{
    const std::vector<double> rhs{1.0, 0.0};
    std::vector<double> solution(2, 0.0);

    const auto matvec = [](const std::vector<double>& in, std::vector<double>& out) {
        out = in;
    };
    const auto preconditioner = [](const std::vector<double>&, std::vector<double>& out) {
        out = {1.0};
    };

    GmresConfig config;
    config.max_iters = 1;
    config.tolerance = 1e-14;

    const auto status = right_preconditioned_gmres(rhs, solution, matvec, preconditioner, config);

    REQUIRE_FALSE(status.converged);
    REQUIRE(status.code == GmresStatus::Code::InvalidInput);
}

TEST_CASE("GMRES reports Arnoldi breakdown for a rank-deficient operator", "[krylov]")
{
    const std::vector<double> rhs{1.0, 0.0};
    std::vector<double> solution(2, 0.0);

    const auto matvec = [](const std::vector<double>&, std::vector<double>& out) {
        out = {0.0, 0.0};
    };

    GmresConfig config;
    config.max_iters = 2;
    config.tolerance = 1e-14;

    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE_FALSE(status.converged);
    REQUIRE(status.code == GmresStatus::Code::Breakdown);
}

TEST_CASE("GMRES reports max iteration failure with the best current iterate", "[krylov]")
{
    const std::vector<double> rhs{1.0, 1.0};
    std::vector<double> solution(2, 0.0);

    const auto matvec = [](const std::vector<double>& in, std::vector<double>& out) {
        out = {2.0 * in[0], 3.0 * in[1]};
    };

    GmresConfig config;
    config.max_iters = 1;
    config.tolerance = 1e-14;

    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE_FALSE(status.converged);
    REQUIRE(status.code == GmresStatus::Code::MaxIterations);
    REQUIRE(solution[0] != Approx(0.0));
    REQUIRE(solution[1] != Approx(0.0));
}

TEST_CASE("GMRES applies the last-good iterate when a later matvec has invalid size", "[krylov]")
{
    const std::vector<double> diagonal{2.0, 3.0};
    const std::vector<double> rhs{1.0, 1.0};
    std::vector<double> solution(rhs.size(), 0.0);
    std::vector<double> residual_history;
    int matvec_calls = 0;

    const auto matvec = [&diagonal, &matvec_calls](const std::vector<double>& in,
                                                   std::vector<double>& out) {
        ++matvec_calls;
        if (matvec_calls == 2) {
            out.clear();
            return;
        }
        out.resize(in.size());
        for (std::size_t entry = 0; entry < in.size(); ++entry)
            out[entry] = diagonal[entry] * in[entry];
    };

    GmresConfig config;
    config.max_iters = 2;
    config.tolerance = 1e-300;
    config.residual_history = &residual_history;

    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE_FALSE(status.converged);
    REQUIRE(status.code == GmresStatus::Code::InvalidInput);
    REQUIRE(status.iterations == 1);
    REQUIRE(matvec_calls == 2);
    REQUIRE(residual_history.size() == 1);
    REQUIRE(status.residual_norm == residual_history.back());
    REQUIRE(solution[0] != Approx(0.0));
    REQUIRE(solution[1] != Approx(0.0));

    const double residual_0 = rhs[0] - diagonal[0] * solution[0];
    const double residual_1 = rhs[1] - diagonal[1] * solution[1];
    const double true_residual = std::sqrt(residual_0 * residual_0 + residual_1 * residual_1);
    REQUIRE(true_residual == Approx(status.residual_norm).margin(1e-12));
}
