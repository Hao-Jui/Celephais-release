#include "Linear_algebra/krylov_solver.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

using Catch::Approx;
using Kadath::GmresConfig;
using Kadath::GmresStatus;
using Kadath::GmresTiming;
using Kadath::right_preconditioned_gmres;

namespace
{
    double euclidean_norm(const std::vector<double>& values)
    {
        double sum_of_squares = 0.0;
        for (const double value : values) {
            sum_of_squares += value * value;
        }
        return std::sqrt(sum_of_squares);
    }
} // namespace

TEST_CASE("GMRES with identity operator and identity preconditioner converges in one iteration",
          "[krylov-gmres]")
{
    const std::vector<double> rhs{1.0, -2.0, 3.0, 0.5};
    std::vector<double> solution(rhs.size(), 0.0);

    const auto identity = [](const std::vector<double>& in, std::vector<double>& out) {
        out = in;
    };

    GmresConfig config;
    config.max_iters = 10;
    config.tolerance = 1e-12;

    const auto status = right_preconditioned_gmres(rhs, solution, identity, identity, config);

    REQUIRE(status.converged);
    REQUIRE(status.code == GmresStatus::Code::Converged);
    REQUIRE(status.iterations == 1);
    for (std::size_t entry = 0; entry < rhs.size(); ++entry) {
        REQUIRE(solution[entry] == Approx(rhs[entry]));
    }
}

TEST_CASE("GMRES converges on a well-conditioned SPD diagonal system", "[krylov-gmres]")
{
    constexpr std::size_t dimension = 20;
    std::vector<double> diagonal(dimension, 0.0);
    std::vector<double> expected_solution(dimension, 0.0);
    std::vector<double> rhs(dimension, 0.0);
    for (std::size_t entry = 0; entry < dimension; ++entry) {
        diagonal[entry] = 1.0 + 0.1 * static_cast<double>(entry);
        expected_solution[entry] = 1.0 + 0.05 * static_cast<double>(entry);
        rhs[entry] = diagonal[entry] * expected_solution[entry];
    }
    std::vector<double> solution(dimension, 0.0);

    const auto matvec = [&diagonal](const std::vector<double>& in, std::vector<double>& out) {
        out.resize(in.size());
        for (std::size_t entry = 0; entry < in.size(); ++entry) {
            out[entry] = diagonal[entry] * in[entry];
        }
    };

    GmresConfig config;
    config.max_iters = 40;
    config.tolerance = 1e-10;

    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE(status.converged);
    REQUIRE(status.code == GmresStatus::Code::Converged);
    for (std::size_t entry = 0; entry < dimension; ++entry) {
        REQUIRE(solution[entry] == Approx(expected_solution[entry]).margin(1e-8));
    }
}

TEST_CASE("GMRES returns Converged with zero iterations when the tolerance is already met",
          "[krylov-gmres]")
{
    const std::vector<double> rhs{1e-10, -1e-10, 1e-10};
    std::vector<double> solution{7.0, 7.0, 7.0};

    const auto matvec = [](const std::vector<double>& in, std::vector<double>& out) {
        out = in;
    };

    GmresConfig config;
    config.max_iters = 5;
    config.tolerance = 1e-8;

    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE(status.converged);
    REQUIRE(status.code == GmresStatus::Code::Converged);
    REQUIRE(status.iterations == 0);
    REQUIRE(status.residual_norm == Approx(euclidean_norm(rhs)));
    for (const double entry : solution) {
        REQUIRE(entry == 0.0);
    }
}

TEST_CASE("GMRES returns MaxIterations with the best iterate when the tolerance is unreachable",
          "[krylov-gmres]")
{
    const std::vector<double> diagonal{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> rhs{1.0, 1.0, 1.0, 1.0};
    std::vector<double> solution(rhs.size(), 0.0);

    const auto matvec = [&diagonal](const std::vector<double>& in, std::vector<double>& out) {
        out.resize(in.size());
        for (std::size_t entry = 0; entry < in.size(); ++entry) {
            out[entry] = diagonal[entry] * in[entry];
        }
    };

    GmresConfig config;
    config.max_iters = 2;
    config.tolerance = 1e-300;

    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE_FALSE(status.converged);
    REQUIRE(status.code == GmresStatus::Code::MaxIterations);
    REQUIRE(status.iterations == 2);

    std::vector<double> residual(rhs.size(), 0.0);
    for (std::size_t entry = 0; entry < rhs.size(); ++entry) {
        residual[entry] = rhs[entry] - diagonal[entry] * solution[entry];
    }
    const double residual_norm = euclidean_norm(residual);
    REQUIRE(residual_norm < euclidean_norm(rhs));
    REQUIRE(residual_norm == Approx(status.residual_norm).margin(1e-12));
}

namespace
{
    // Shared check for the rank-deficient fixtures: the solver must never
    // commit an unbounded Krylov combination when the operator is singular —
    // the rank guard in the rotated-Hessenberg back-substitution falls back to
    // the last well-posed iterate instead of dividing by a noise diagonal.
    void require_bounded_singular_solve(const std::vector<double>& diagonal,
                                        const std::vector<double>& rhs,
                                        int max_iters)
    {
        std::vector<double> solution(rhs.size(), 0.0);
        const auto matvec = [&diagonal](const std::vector<double>& in, std::vector<double>& out) {
            out.resize(in.size());
            for (std::size_t entry = 0; entry < in.size(); ++entry) {
                out[entry] = diagonal[entry] * in[entry];
            }
        };

        GmresConfig config;
        config.max_iters = max_iters;
        config.tolerance = 1e-300;

        const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

        REQUIRE_FALSE(status.code == GmresStatus::Code::Converged);
        std::vector<double> residual(rhs.size(), 0.0);
        for (std::size_t entry = 0; entry < rhs.size(); ++entry) {
            REQUIRE(std::isfinite(solution[entry]));
            REQUIRE(std::abs(solution[entry]) < 1e2);
            residual[entry] = rhs[entry] - diagonal[entry] * solution[entry];
        }
        // The committed iterate must be no worse than the zero step.
        REQUIRE(euclidean_norm(residual) <= euclidean_norm(rhs) * (1.0 + 1e-12));
    }
} // namespace

TEST_CASE("GMRES on a singular operator commits a bounded last-good iterate", "[krylov-gmres]")
{
    require_bounded_singular_solve({1.0, 2.0, 0.0}, {1.0, 1.0, 1.0}, 3);
}

TEST_CASE("GMRES on a larger rank-deficient operator stays bounded at the iteration cap",
          "[krylov-gmres]")
{
    std::vector<double> diagonal(12, 0.0);
    std::vector<double> rhs(12, 1.0);
    for (std::size_t entry = 0; entry + 1 < diagonal.size(); ++entry) {
        diagonal[entry] = 1.0 + static_cast<double>(entry);
    }
    diagonal.back() = 0.0;
    require_bounded_singular_solve(diagonal, rhs, 12);
}

TEST_CASE("GMRES timing instrumentation preserves the arithmetic path", "[krylov-gmres]")
{
    const std::vector<double> diagonal{1.0, 1.5, 2.0, 2.5, 3.0, 4.0};
    const std::vector<double> rhs{1.0, -0.5, 2.0, 1.5, -1.0, 0.25};
    const auto matvec = [&diagonal](const std::vector<double>& in, std::vector<double>& out) {
        out.resize(in.size());
        for (std::size_t entry = 0; entry < in.size(); ++entry)
            out[entry] = diagonal[entry] * in[entry];
    };

    std::vector<double> untimed_solution(rhs.size(), 0.0);
    std::vector<double> untimed_history;
    GmresConfig untimed_config;
    untimed_config.max_iters = 4;
    untimed_config.tolerance = 1e-300;
    untimed_config.residual_history = &untimed_history;
    const auto untimed_status =
        right_preconditioned_gmres(rhs, untimed_solution, matvec, nullptr, untimed_config);

    std::vector<double> timed_solution(rhs.size(), 0.0);
    std::vector<double> timed_history;
    GmresTiming timing;
    GmresConfig timed_config = untimed_config;
    timed_config.timing = &timing;
    timed_config.residual_history = &timed_history;
    const auto timed_status =
        right_preconditioned_gmres(rhs, timed_solution, matvec, nullptr, timed_config);

    REQUIRE(timed_status.code == untimed_status.code);
    REQUIRE(timed_status.converged == untimed_status.converged);
    REQUIRE(timed_status.iterations == untimed_status.iterations);
    REQUIRE(timed_status.residual_norm == untimed_status.residual_norm);
    REQUIRE(timed_solution == untimed_solution);
    REQUIRE(timed_history == untimed_history);
    REQUIRE(timing.matvecs == timed_status.iterations);
    REQUIRE(timing.preconditions == 0);
    REQUIRE(timing.vector_norm_seconds >= 0.0);
    REQUIRE(timing.orthog_seconds >= 0.0);
    REQUIRE(timing.least_squares_seconds >= 0.0);
    REQUIRE(timing.update_seconds >= 0.0);
}

TEST_CASE("GMRES appends residual history in iteration order and reserves the cap",
          "[krylov-gmres]")
{
    const std::vector<double> diagonal{1.0, 2.0, 3.0, 4.0};
    const std::vector<double> rhs{1.0, 1.0, 1.0, 1.0};
    std::vector<double> solution(rhs.size(), 0.0);
    std::vector<double> history{-1.0};
    const auto matvec = [&diagonal](const std::vector<double>& in, std::vector<double>& out) {
        out.resize(in.size());
        for (std::size_t entry = 0; entry < in.size(); ++entry)
            out[entry] = diagonal[entry] * in[entry];
    };

    GmresConfig config;
    config.max_iters = 3;
    config.tolerance = 1e-300;
    config.residual_history = &history;
    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE(status.code == GmresStatus::Code::MaxIterations);
    REQUIRE(history.size() == 4);
    REQUIRE(history.capacity() >= 4);
    REQUIRE(history[0] == -1.0);
    REQUIRE(history[2] <= history[1]);
    REQUIRE(history[3] <= history[2]);
    REQUIRE(status.residual_norm == history.back());
}

TEST_CASE("GMRES spectral capture preserves the public raw Hessenberg layout",
          "[krylov-gmres]")
{
    const std::vector<double> diagonal{1.0, 2.0, 4.0};
    const std::vector<double> rhs{1.0, 1.0, 1.0};
    std::vector<double> solution(rhs.size(), 0.0);
    std::vector<std::vector<double>> basis;
    std::vector<std::vector<double>> raw_hessenberg;
    const auto matvec = [&diagonal](const std::vector<double>& in, std::vector<double>& out) {
        out.resize(in.size());
        for (std::size_t entry = 0; entry < in.size(); ++entry)
            out[entry] = diagonal[entry] * in[entry];
    };

    GmresConfig config;
    config.max_iters = 2;
    config.tolerance = 1e-300;
    config.arnoldi_basis_out = &basis;
    config.hessenberg_raw_out = &raw_hessenberg;
    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE(status.code == GmresStatus::Code::MaxIterations);
    REQUIRE(basis.size() == 2);
    REQUIRE(basis[0].size() == rhs.size());
    REQUIRE(basis[1].size() == rhs.size());
    REQUIRE(raw_hessenberg.size() == 3);
    for (const auto& row : raw_hessenberg)
        REQUIRE(row.size() == 2);
    REQUIRE(raw_hessenberg[2][0] == 0.0);
    REQUIRE(raw_hessenberg[1][0] > 0.0);
    REQUIRE(raw_hessenberg[2][1] > 0.0);

    for (std::size_t entry = 0; entry < rhs.size(); ++entry) {
        const double reconstructed = raw_hessenberg[0][0] * basis[0][entry] +
                                     raw_hessenberg[1][0] * basis[1][entry];
        REQUIRE(reconstructed ==
                Approx(diagonal[entry] * basis[0][entry]).margin(1e-12));
    }
}

TEST_CASE("GMRES spectral convergence omits the unused next basis vector",
          "[krylov-gmres]")
{
    const std::vector<double> rhs{1.0, -2.0, 3.0, -4.0};
    std::vector<double> solution(rhs.size(), 0.0);
    std::vector<std::vector<double>> basis;
    std::vector<std::vector<double>> raw_hessenberg;
    const auto matvec = [](const std::vector<double>& in, std::vector<double>& out) {
        out = in;
    };

    GmresConfig config;
    config.max_iters = 10;
    config.tolerance = 1e-14;
    config.arnoldi_basis_out = &basis;
    config.hessenberg_raw_out = &raw_hessenberg;
    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE(status.code == GmresStatus::Code::Converged);
    REQUIRE(status.iterations == 1);
    for (std::size_t entry = 0; entry < rhs.size(); ++entry)
        REQUIRE(solution[entry] == Approx(rhs[entry]).epsilon(1e-14));
    REQUIRE(basis.size() == 1);
    REQUIRE(basis.front().size() == rhs.size());
    REQUIRE(raw_hessenberg.size() == rhs.size() + 1);
    for (const auto& row : raw_hessenberg)
        REQUIRE(row.size() == rhs.size());
    REQUIRE(raw_hessenberg[0][0] == Approx(1.0).epsilon(1e-14));
    REQUIRE(raw_hessenberg[1][0] == Approx(0.0).margin(1e-14));
}

TEST_CASE("GMRES tiled null-preconditioner update crosses tile boundaries",
          "[krylov-gmres]")
{
    constexpr std::size_t dimension = 4097;
    std::vector<double> diagonal(dimension, 0.0);
    std::vector<double> rhs(dimension, 0.0);
    std::vector<double> solution(dimension, 0.0);
    for (std::size_t entry = 0; entry < dimension; ++entry) {
        diagonal[entry] = (entry % 2 == 0) ? 2.0 : 3.0;
        rhs[entry] = 1.0 + 0.01 * static_cast<double>(entry % 7);
    }
    const auto matvec = [&diagonal](const std::vector<double>& in, std::vector<double>& out) {
        out.resize(in.size());
        for (std::size_t entry = 0; entry < in.size(); ++entry)
            out[entry] = diagonal[entry] * in[entry];
    };

    GmresConfig config;
    config.max_iters = 2;
    config.tolerance = 1e-10;
    const auto status = right_preconditioned_gmres(rhs, solution, matvec, nullptr, config);

    REQUIRE(status.converged);
    REQUIRE(status.iterations == 2);
    for (std::size_t entry = 0; entry < dimension; ++entry)
        REQUIRE(solution[entry] == Approx(rhs[entry] / diagonal[entry]).margin(1e-10));
}
