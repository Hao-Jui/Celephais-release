#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/System_of_eqs/solver_runtime_config.hpp"
#include "For_Kadath/Array/array.hpp"
#include "Linear_algebra/jacobian_assembler.hpp"
#include "Linear_algebra/jacobian_parity_mask.hpp"
#include "Linear_algebra/linear_solver.hpp"
#include "Linear_algebra/mumps_linear_solver.hpp"
#include "Linear_algebra/sparse_direct_mumps_solve.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Kadath
{
    int resolve_sparse_direct_mumps_ranks_per_node(
        const MumpsRuntimeConfig& config, MPI_Comm communicator);

    void sparse_direct_collective_throw_if_failed(
        MPI_Comm communicator, const char* phase,
        const std::string& local_error);

    void zero_selected_sparse_corrections(
        Array<double>& correction, int column_count,
        std::ostream* diagnostic);
}

using namespace Kadath;
using Catch::Matchers::ContainsSubstring;

namespace {

class MockLinearSolver : public LinearSolver
{
  public:
    void set_pattern(int n, long long nnz, const int* /*irn*/, const int* /*jcn*/) override
    {
        n_ = n;
        nnz_ = nnz;
        pattern_set = true;
    }

    void factor(const double* /*values*/) override
    {
        if (!pattern_set) {
            throw LinearSolverError(__FILE__, __LINE__, "factor called before set_pattern");
        }
        factored = true;
    }

    void solve(double* /*rhs_inout*/) override
    {
        if (!factored) {
            throw LinearSolverError(__FILE__, __LINE__, "solve called before factor");
        }
        // identity solver: leave rhs unchanged
    }

    void reset() override
    {
        pattern_set = false;
        factored = false;
        reset_called = true;
        n_ = 0;
        nnz_ = 0;
    }

    int n() const { return n_; }
    long long nnz() const { return nnz_; }

    bool pattern_set = false;
    bool factored = false;
    bool reset_called = false;

  private:
    int n_ = 0;
    long long nnz_ = 0;
};

class ScopedEnvironmentVariable
{
  public:
    ScopedEnvironmentVariable(const char* name, const char* value) : name_(name)
    {
        if (const char* current = std::getenv(name)) {
            had_original_ = true;
            original_ = current;
        }
        if (setenv(name, value, 1) != 0)
            throw std::runtime_error("setenv failed for " + name_);
    }

    ~ScopedEnvironmentVariable()
    {
        if (had_original_)
            setenv(name_.c_str(), original_.c_str(), 1);
        else
            unsetenv(name_.c_str());
    }

    ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
    ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

  private:
    std::string name_;
    std::string original_;
    bool had_original_ = false;
};

class ScopedMumpsMemoryRatchetReset
{
  public:
    ScopedMumpsMemoryRatchetReset()
    {
        mumps_memory_detail::reset_node_available_memory_ratchet_for_tests();
    }

    ~ScopedMumpsMemoryRatchetReset()
    {
        mumps_memory_detail::reset_node_available_memory_ratchet_for_tests();
    }

    ScopedMumpsMemoryRatchetReset(const ScopedMumpsMemoryRatchetReset&) = delete;
    ScopedMumpsMemoryRatchetReset& operator=(
        const ScopedMumpsMemoryRatchetReset&) = delete;
};

void ensure_mpi_initialized()
{
    static const bool initialized_for_tests = []() {
        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized != 0) {
            throw std::runtime_error("MPI is already finalized");
        }

        int initialized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Initialized failed");
        }
        if (initialized == 0) {
            int argc = 0;
            char** argv = nullptr;
            if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
                throw std::runtime_error("MPI_Init failed");
            }
            std::atexit([]() {
                int mpi_finalized = 0;
                if (MPI_Finalized(&mpi_finalized) == MPI_SUCCESS && mpi_finalized == 0)
                    MPI_Finalize();
            });
        }
        return true;
    }();
    (void)initialized_for_tests;
}

std::filesystem::path find_sparse_newton_source()
{
    const std::array<std::filesystem::path, 2> seeds{{
        std::filesystem::absolute(std::filesystem::path(__FILE__)).parent_path(),
        std::filesystem::current_path(),
    }};

    for (const auto& seed : seeds) {
        auto directory = seed;
        while (!directory.empty()) {
            const auto candidate =
                directory / "src" / "Newton" / "do_newton_sparse.cpp";
            if (std::filesystem::is_regular_file(candidate))
                return candidate;

            const auto parent = directory.parent_path();
            if (parent == directory)
                break;
            directory = parent;
        }
    }

    throw std::runtime_error(
        "could not locate src/Newton/do_newton_sparse.cpp from __FILE__");
}

std::filesystem::path find_jfnk_mumps_newton_source()
{
    const std::array<std::filesystem::path, 2> seeds{{
        std::filesystem::absolute(std::filesystem::path(__FILE__)).parent_path(),
        std::filesystem::current_path(),
    }};

    for (const auto& seed : seeds) {
        auto directory = seed;
        while (!directory.empty()) {
            const auto candidate =
                directory / "src" / "Newton" /
                "do_newton_jfnk_mumps.cpp";
            if (std::filesystem::is_regular_file(candidate))
                return candidate;

            const auto parent = directory.parent_path();
            if (parent == directory)
                break;
            directory = parent;
        }
    }

    throw std::runtime_error(
        "could not locate src/Newton/do_newton_jfnk_mumps.cpp from __FILE__");
}

std::string read_source_file(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input.is_open())
        throw std::runtime_error("could not open source file: " + path.string());
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

bool every_log_occurrence_is_guarded_by(
    const std::string& source, const std::string& message,
    const std::string& guard_fragment)
{
    bool found = false;
    for (std::size_t position = source.find(message);
         position != std::string::npos;
         position = source.find(message, position + message.size())) {
        found = true;
        const std::size_t guard_begin = source.rfind("if (", position);
        if (guard_begin == std::string::npos)
            return false;
        const std::size_t guard_end = source.find('{', guard_begin);
        if (guard_end == std::string::npos || guard_end > position)
            return false;
        if (source.substr(guard_begin, guard_end - guard_begin)
                .find(guard_fragment) == std::string::npos) {
            return false;
        }
    }
    return found;
}

std::vector<std::string> sparse_direct_solver_constructions(
    const std::string& source)
{
    constexpr const char* marker =
        "std::make_unique<MumpsLinearSolver>(";
    std::vector<std::string> constructions;
    std::size_t search_from = 0;

    while (true) {
        const auto begin = source.find(marker, search_from);
        if (begin == std::string::npos)
            break;

        const auto arguments_begin = begin + std::string(marker).size();
        int depth = 1;
        auto end = arguments_begin;
        for (; end < source.size() && depth > 0; ++end) {
            if (source[end] == '(')
                ++depth;
            else if (source[end] == ')')
                --depth;
        }
        if (depth != 0) {
            throw std::runtime_error(
                "unterminated MumpsLinearSolver construction");
        }

        constructions.emplace_back(source.substr(begin, end - begin));
        search_from = end;
    }
    return constructions;
}

AssembledJacobianCoo make_two_sector_matrix()
{
    AssembledJacobianCoo matrix;
    matrix.n = 4;
    matrix.nnz = 8;
    matrix.parity_sector_block_diagonal = true;
    matrix.irn = {1, 1, 3, 3, 2, 2, 4, 4};
    matrix.jcn = {1, 3, 1, 3, 2, 4, 2, 4};
    matrix.a = {4.0, 1.0, 1.0, 3.0, 5.0, 2.0, 2.0, 6.0};
    return matrix;
}

JacobianParityMaskState make_two_sector_state()
{
    JacobianParityMaskState state;
    state.n = 4;
    state.decision = JacobianParityMaskState::Decision::Engaged;
    state.row_sector = {1, -1, 1, -1};
    state.column_sector = {1, -1, 1, -1};
    return state;
}

std::shared_ptr<const JacobianSelectionPlan> make_two_sector_plan(int sector)
{
    const JacobianSelectionPlanBuild built = make_jacobian_selection_plan(
        {1, -1, 1, -1}, {1, -1, 1, -1}, sector, -sector);
    REQUIRE(built.plan);
    REQUIRE(built.fallback_reason.empty());
    return built.plan;
}

AssembledJacobianCooBlock make_physical_sector_block(int sector)
{
    AssembledJacobianCooBlock block;
    block.parity_label = sector;
    block.n = 2;
    block.nnz = 4;
    block.selection_plan = make_two_sector_plan(sector);
    int rank = 0;
    REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
    if (rank == 0) {
        block.irn = {1, 1, 2, 2};
        block.jcn = {1, 2, 1, 2};
        block.a = sector > 0
            ? AssembledJacobianCooBlock::ValueVector{4.0, 1.0, 1.0, 3.0}
            : AssembledJacobianCooBlock::ValueVector{5.0, 2.0, 2.0, 6.0};
    }
    return block;
}

AssembledJacobianCoo make_physical_parity_matrix(bool plus_only)
{
    AssembledJacobianCoo matrix;
    matrix.n = plus_only ? 2 : 4;
    matrix.nnz = plus_only ? 4 : 8;
    matrix.parity_sector_block_diagonal = true;
    matrix.parity_blocks.push_back(make_physical_sector_block(1));
    if (!plus_only)
        matrix.parity_blocks.push_back(make_physical_sector_block(-1));
    return matrix;
}

SparseDirectMumpsSolveOptions make_direct_mumps_solve_options()
{
    SparseDirectMumpsSolveOptions options;
    options.ordering = 7;
    options.out_of_core_mode = MumpsOutOfCoreMode::Off;
    options.communicator = MPI_COMM_WORLD;
    options.ranks_per_node = 0;
    return options;
}

void check_two_sector_solution(const std::array<double, 4>& solution)
{
    constexpr std::array<double, 4> expected{{1.0, 2.0, 3.0, 4.0}};
    for (std::size_t i = 0; i < solution.size(); ++i) {
        CAPTURE(i);
        CHECK(solution[i] == Catch::Approx(expected[i]).margin(1e-12));
    }
}

} // namespace

TEST_CASE("LinearSolver: set_pattern records n + nnz", "[linear_solver]") {
    MockLinearSolver solver;
    int irn[3] = {1, 2, 3};
    int jcn[3] = {1, 2, 3};
    solver.set_pattern(/*n=*/3, /*nnz=*/3, irn, jcn);
    REQUIRE(solver.n() == 3);
    REQUIRE(solver.nnz() == 3);
    REQUIRE(solver.pattern_set);
}

TEST_CASE("LinearSolver: factor + solve on identity preserves rhs",
          "[linear_solver]") {
    MockLinearSolver solver;
    int irn[3] = {1, 2, 3};
    int jcn[3] = {1, 2, 3};
    double values[3] = {1.0, 1.0, 1.0};
    double rhs[3] = {7.0, 11.0, 13.0};
    solver.set_pattern(3, 3, irn, jcn);
    solver.factor(values);
    solver.solve(rhs);
    REQUIRE(rhs[0] == 7.0);
    REQUIRE(rhs[1] == 11.0);
    REQUIRE(rhs[2] == 13.0);
}

TEST_CASE("LinearSolver: reset clears pattern + factor state",
          "[linear_solver]") {
    MockLinearSolver solver;
    int irn[1] = {1};
    int jcn[1] = {1};
    double values[1] = {1.0};
    solver.set_pattern(1, 1, irn, jcn);
    solver.factor(values);
    solver.reset();
    REQUIRE_FALSE(solver.pattern_set);
    REQUIRE_FALSE(solver.factored);
    REQUIRE(solver.reset_called);
}

TEST_CASE("LinearSolver: solve before factor throws LinearSolverError",
          "[linear_solver]") {
    MockLinearSolver solver;
    double rhs[1] = {1.0};
    REQUIRE_THROWS_AS(solver.solve(rhs), LinearSolverError);
}

TEST_CASE("MUMPS memory probe parses Linux MemAvailable fixtures",
          "[linear_solver][mumps-memory-probe]")
{
    using mumps_memory_detail::parse_mem_available_mb;

    CHECK(parse_mem_available_mb(
              "MemTotal:       33554432 kB\n"
              "MemFree:         524288 kB\n"
              "MemAvailable:   12582912 kB\n") == 12288);
    CHECK(parse_mem_available_mb("MemAvailable: 1023 kB\n") == 0);
    CHECK(parse_mem_available_mb("MemFree: 4096 kB\n") == -1);
    CHECK(parse_mem_available_mb("MemAvailable: -1 kB\n") == -1);
    CHECK(parse_mem_available_mb("MemAvailable: 4096 MB\n") == -1);
    CHECK(parse_mem_available_mb("MemAvailable: 4096 kB trailing\n") == -1);
}

TEST_CASE("MUMPS memory probe parses cgroup headroom fixtures",
          "[linear_solver][mumps-memory-probe]")
{
    using mumps_memory_detail::CgroupMemoryStatus;
    using mumps_memory_detail::parse_cgroup_memory_headroom;

    const auto finite = parse_cgroup_memory_headroom(
        "1073741824\n", "268435456\n");
    CHECK(finite.status == CgroupMemoryStatus::Limited);
    CHECK(finite.available_mb == 768);

    const auto exhausted = parse_cgroup_memory_headroom("4096", "8192");
    CHECK(exhausted.status == CgroupMemoryStatus::Limited);
    CHECK(exhausted.available_mb == 0);

    const auto v2_unlimited = parse_cgroup_memory_headroom("max\n", "1234\n");
    CHECK(v2_unlimited.status == CgroupMemoryStatus::Unlimited);
    CHECK(v2_unlimited.available_mb == -1);

    const auto v1_unlimited = parse_cgroup_memory_headroom(
        "9223372036854771712\n", "1234\n");
    CHECK(v1_unlimited.status == CgroupMemoryStatus::Unlimited);
    CHECK(v1_unlimited.available_mb == -1);

    const auto malformed = parse_cgroup_memory_headroom("1GiB", "0");
    CHECK(malformed.status == CgroupMemoryStatus::Unreadable);
    CHECK(malformed.available_mb == -1);

    const auto malformed_usage = parse_cgroup_memory_headroom("1048576", "-1");
    CHECK(malformed_usage.status == CgroupMemoryStatus::Unreadable);
    CHECK(malformed_usage.available_mb == -1);

    const auto overflow = parse_cgroup_memory_headroom(
        "18446744073709551616", "0");
    CHECK(overflow.status == CgroupMemoryStatus::Unreadable);
    CHECK(overflow.available_mb == -1);
}

TEST_CASE("MUMPS live memory probe has the documented sentinel contract",
          "[linear_solver][mumps-memory-probe]")
{
    const long long available_mb =
        mumps_memory_detail::node_available_memory_mb();
    CAPTURE(available_mb);
    CHECK((available_mb == -1 || available_mb >= 0));
}

TEST_CASE("MUMPS node memory ratchet retains the lowest valid probe",
          "[linear_solver][mumps-memory-ratchet]")
{
    const ScopedMumpsMemoryRatchetReset reset;
    using mumps_memory_detail::ratcheted_node_available_memory_mb;

    CHECK(ratcheted_node_available_memory_mb(9000) == 9000);
    CHECK(ratcheted_node_available_memory_mb(7000) == 7000);
    CHECK(ratcheted_node_available_memory_mb(8000) == 7000);
    CHECK(ratcheted_node_available_memory_mb(0) == 0);
    CHECK(ratcheted_node_available_memory_mb(500) == 0);
}

TEST_CASE("MUMPS node memory ratchet keeps AUTO OOC on after a low probe",
          "[linear_solver][mumps-memory-ratchet][mumps-ooc]")
{
    const ScopedMumpsMemoryRatchetReset reset;
    using mumps_memory_detail::ratcheted_node_available_memory_mb;

    // The probe helper accepts whole MiB. These values consistently floor the
    // measured decimal-MiB sequence, matching the probe parsers' truncation.
    constexpr std::array<long long, 7> probe_mb{{
        6766, 8850, 7221, 10329, 7952, 12591, 8198,
    }};

    long long ratcheted_mb = -1;
    for (const long long current_probe_mb : probe_mb) {
        ratcheted_mb = ratcheted_node_available_memory_mb(current_probe_mb);
        REQUIRE(ratcheted_mb == probe_mb.front());
    }

    constexpr double estimated_mb = 18465.0;
    constexpr int icntl14 = 400;
    constexpr double touch = 1.3;
    constexpr double safety = 0.7;
    constexpr int factor_ranks_on_host_node = 1;
    const double expected_mb_per_rank =
        estimated_mb / (1.0 + static_cast<double>(icntl14) / 100.0) * touch;
    CHECK(expected_mb_per_rank == Catch::Approx(4800.9));

    const MumpsOutOfCoreDecision ratcheted_decision = decide_mumps_out_of_core(
        estimated_mb, icntl14, touch, safety, factor_ranks_on_host_node,
        static_cast<double>(ratcheted_mb));
    REQUIRE(ratcheted_decision.valid);
    CHECK(ratcheted_decision.expected_mb_per_rank ==
          Catch::Approx(expected_mb_per_rank));
    CHECK(ratcheted_decision.node_expected_mb == Catch::Approx(4800.9));
    CHECK(ratcheted_decision.budget_mb == Catch::Approx(4736.2));
    CHECK(ratcheted_decision.use_out_of_core);

    const MumpsOutOfCoreDecision unratcheted_final_decision =
        decide_mumps_out_of_core(
            estimated_mb, icntl14, touch, safety,
            factor_ranks_on_host_node,
            static_cast<double>(probe_mb.back()));
    REQUIRE(unratcheted_final_decision.valid);
    CHECK(unratcheted_final_decision.expected_mb_per_rank ==
          Catch::Approx(expected_mb_per_rank));
    CHECK(unratcheted_final_decision.budget_mb == Catch::Approx(5738.6));
    CHECK_FALSE(unratcheted_final_decision.use_out_of_core);
}

TEST_CASE("MUMPS node memory ratchet ignores an unreadable negative probe",
          "[linear_solver][mumps-memory-ratchet]")
{
    const ScopedMumpsMemoryRatchetReset reset;
    using mumps_memory_detail::ratcheted_node_available_memory_mb;

    REQUIRE(ratcheted_node_available_memory_mb(6000) == 6000);
    CHECK(ratcheted_node_available_memory_mb(-1) == -1);
    CHECK(ratcheted_node_available_memory_mb(7000) == 6000);
}

TEST_CASE("MUMPS rank-capped solve overwrites and broadcasts the caller RHS buffer",
          "[linear_solver][mumps-inout]")
{
    static_assert(std::is_constructible_v<
                  MumpsLinearSolver, int, int, bool, int, int, MPI_Comm, int, bool>);
    static_assert(std::is_constructible_v<
                  MumpsLinearSolver, int, int, MumpsOutOfCoreMode, int, int,
                  MPI_Comm, int, bool>);

    ensure_mpi_initialized();
    int rank = 0;
    int size = 0;
    REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
    REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, &size) == MPI_SUCCESS);

    const ScopedEnvironmentVariable factor_rank_cap(
        "MUMPS_RANKS_PER_NODE", "1");
    const SolverRuntimeConfig config = SolverRuntimeConfig::from_environment();
    REQUIRE(config.mumps.ranks_per_node == 1);

    const std::array<int, 4> irn{{1, 1, 2, 2}};
    const std::array<int, 4> jcn{{1, 2, 1, 2}};
    const std::array<double, 4> values{{4.0, 1.0, 2.0, 3.0}};
    std::array<double, 2> rhs = rank == 0
                                   ? std::array<double, 2>{{1.0, 2.0}}
                                   : std::array<double, 2>{{-101.0, -103.0}};

    {
        MumpsLinearSolver solver(
            2, config.mumps.ordering, config.mumps.out_of_core, config.mumps.blr,
            100, MPI_COMM_WORLD, config.mumps.ranks_per_node, false,
            config.mumps.out_of_core_touch,
            config.mumps.out_of_core_safety,
            config.mumps.out_of_core_budget_mb);
        solver.set_pattern(2, 4, irn.data(), jcn.data());
        solver.factor(values.data());
        if (rank == 0) {
            CHECK(solver.factor_nnz() > 0);
            CHECK(solver.factor_real_slots() > 0);
            CHECK(solver.max_front_order() > 0);
            CHECK(solver.estimated_max_front_order() > 0);
            CHECK(solver.factor_allocated_memory_mb() > 0);
            CHECK(solver.factor_allocated_memory_mb() >=
                  solver.factor_memory_mb());
            CHECK(solver.factor_memory_total_mb() >= solver.factor_memory_mb());
            CHECK(solver.estimated_factor_memory_total_mb() >=
                  solver.estimated_factor_memory_mb());
        }
        solver.solve(rhs.data());
    }

    CAPTURE(rank, size);
    CHECK(rhs[0] == Catch::Approx(0.1).margin(1e-13));
    CHECK(rhs[1] == Catch::Approx(0.6).margin(1e-13));
}

TEST_CASE("MUMPS OOC constructors retain the requested tri-state policy",
          "[linear_solver][mumps-ooc]")
{
    ensure_mpi_initialized();

    SECTION("Auto remains requested and initializes ICNTL(22) off") {
        MumpsLinearSolver solver(
            1, 7, MumpsOutOfCoreMode::Auto, 0, 100, MPI_COMM_WORLD);
        CHECK(solver.requested_out_of_core_mode() == MumpsOutOfCoreMode::Auto);
        CHECK_FALSE(solver.out_of_core_enabled());
    }
    SECTION("bool false maps exactly to Off") {
        MumpsLinearSolver solver(1, 7, false, 0, 100, MPI_COMM_WORLD);
        CHECK(solver.requested_out_of_core_mode() == MumpsOutOfCoreMode::Off);
        CHECK_FALSE(solver.out_of_core_enabled());
    }
    SECTION("bool true maps exactly to On") {
        MumpsLinearSolver solver(1, 7, true, 0, 100, MPI_COMM_WORLD);
        CHECK(solver.requested_out_of_core_mode() == MumpsOutOfCoreMode::On);
        CHECK(solver.out_of_core_enabled());
    }
}

TEST_CASE("MUMPS automatic OOC decision uses de-inflated node memory",
          "[linear_solver][mumps-ooc]")
{
    SECTION("node expectation above the safety-adjusted budget selects OOC") {
        const MumpsOutOfCoreDecision decision =
            decide_mumps_out_of_core(1000.0, 100, 1.3, 0.7, 4, 3000.0);
        REQUIRE(decision.valid);
        CHECK(decision.expected_mb_per_rank == Catch::Approx(650.0));
        CHECK(decision.node_expected_mb == Catch::Approx(2600.0));
        CHECK(decision.budget_mb == Catch::Approx(2100.0));
        CHECK(decision.use_out_of_core);
    }
    SECTION("node expectation below the budget stays in-core") {
        const MumpsOutOfCoreDecision decision =
            decide_mumps_out_of_core(1000.0, 100, 1.3, 0.7, 2, 3000.0);
        REQUIRE(decision.valid);
        CHECK(decision.node_expected_mb == Catch::Approx(1300.0));
        CHECK_FALSE(decision.use_out_of_core);
    }
    SECTION("equality stays in-core because the comparison is strict") {
        const MumpsOutOfCoreDecision decision =
            decide_mumps_out_of_core(1000.0, 100, 1.0, 0.5, 2, 2000.0);
        REQUIRE(decision.valid);
        CHECK(decision.node_expected_mb == Catch::Approx(decision.budget_mb));
        CHECK_FALSE(decision.use_out_of_core);
    }
    SECTION("corrected res13 reservation arithmetic predicts in-core") {
        const MumpsOutOfCoreDecision decision =
            decide_mumps_out_of_core(34662.0, 400, 1.3, 0.7, 1, 36000.0);
        REQUIRE(decision.valid);
        CHECK(decision.expected_mb_per_rank == Catch::Approx(9012.12));
        CHECK(decision.budget_mb == Catch::Approx(25200.0));
        CHECK_FALSE(decision.use_out_of_core);
    }
}

TEST_CASE("MUMPS automatic OOC decision fails closed on invalid inputs",
          "[linear_solver][mumps-ooc]")
{
    const auto check_refused = [](const MumpsOutOfCoreDecision& decision) {
        CHECK_FALSE(decision.valid);
        CHECK_FALSE(decision.use_out_of_core);
    };
    check_refused(decide_mumps_out_of_core(
        -1.0, 100, 1.3, 0.7, 1, 1000.0));
    check_refused(decide_mumps_out_of_core(
        100.0, -1, 1.3, 0.7, 1, 1000.0));
    check_refused(decide_mumps_out_of_core(
        100.0, 100, 0.0, 0.7, 1, 1000.0));
    check_refused(decide_mumps_out_of_core(
        100.0, 100, 1.3, 0.0, 1, 1000.0));
    check_refused(decide_mumps_out_of_core(
        100.0, 100, 1.3, 0.7, 0, 1000.0));
    check_refused(decide_mumps_out_of_core(
        100.0, 100, 1.3, 0.7, 1, -1.0));
    check_refused(decide_mumps_out_of_core(
        std::numeric_limits<double>::infinity(),
        100, 1.3, 0.7, 1, 1000.0));
}

TEST_CASE("MUMPS column permutation accessor honors the analysis contract",
          "[linear_solver][mumps-inout]")
{
    ensure_mpi_initialized();
    int rank = 0;
    REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);

    // Zero diagonal: the maximum transversal must permute columns to build a
    // zero-free diagonal, so UNS_PERM is exercised, not merely defaulted.
    const std::array<int, 6> irn{{1, 2, 3, 4, 1, 4}};
    const std::array<int, 6> jcn{{4, 3, 2, 1, 2, 3}};

    MumpsLinearSolver solver(4, 7, false, 0, 100, MPI_COMM_WORLD);
    solver.set_pattern(4, 6, irn.data(), jcn.data());

    std::vector<int> uns_perm;
    bool matching_applied = false;
    CHECK_THROWS_AS(solver.copy_column_permutation_1based(uns_perm, matching_applied),
                    LinearSolverError);

    solver.analyze_pattern();
    solver.copy_column_permutation_1based(uns_perm, matching_applied);
    if (rank == 0) {
        REQUIRE(uns_perm.size() == 4);
        std::array<int, 4> seen{{0, 0, 0, 0}};
        for (const int position : uns_perm) {
            REQUIRE(position >= 1);
            REQUIRE(position <= 4);
            ++seen[static_cast<std::size_t>(position - 1)];
        }
        CHECK(seen == std::array<int, 4>{{1, 1, 1, 1}});
        if (!matching_applied) {
            CHECK(uns_perm == std::vector<int>{1, 2, 3, 4});
        }
    } else {
        CHECK(uns_perm.empty());
    }
}

TEST_CASE("MUMPS composed solve back-maps its column permutation",
          "[linear_solver][mumps-inout]")
{
    ensure_mpi_initialized();
    int rank = 0;
    REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);

    // Original dense A has solution x={1,2,3,4} for this RHS. The COO below
    // represents A*P with columns mapped by inverse({3,1,4,2}), so JOB=3
    // returns y={3,1,4,2}; the configured solve contract must return x.
    const std::array<int, 16> irn{{
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
    }};
    const std::array<int, 16> composed_jcn{{
        2, 2, 2, 2,
        4, 4, 4, 4,
        1, 1, 1, 1,
        3, 3, 3, 3,
    }};
    const std::array<double, 16> values{{
        4.0, 1.0, 1.0, 1.0,
        1.0, 5.0, 1.0, 1.0,
        1.0, 1.0, 6.0, 1.0,
        1.0, 1.0, 1.0, 7.0,
    }};
    std::array<double, 4> rhs = rank == 0
                                    ? std::array<double, 4>{{13.0, 18.0, 25.0, 34.0}}
                                    : std::array<double, 4>{{-101.0, -103.0, -107.0, -109.0}};

    MumpsLinearSolver solver(4, 5, false, 0, 100, MPI_COMM_WORLD, 1, false);
    solver.set_pattern(4, 16, irn.data(), composed_jcn.data());
    solver.set_solution_column_permutation_1based({3, 1, 4, 2});
    solver.factor(values.data());
    solver.solve(rhs.data());

    CAPTURE(rank);
    for (std::size_t i = 0; i < rhs.size(); ++i)
        CHECK(rhs[i] == Catch::Approx(static_cast<double>(i + 1)).margin(1e-12));
}

TEST_CASE("MUMPS solution column permutation validates its lifecycle contract",
          "[linear_solver][mumps-inout]")
{
    ensure_mpi_initialized();
    MumpsLinearSolver solver(4, 5, false, 0, 100, MPI_COMM_WORLD);

    REQUIRE_THROWS_WITH(
        solver.set_solution_column_permutation_1based({1, 2, 3}),
        ContainsSubstring("exactly n=4"));
    REQUIRE_THROWS_WITH(
        solver.set_solution_column_permutation_1based({0, 2, 3, 4}),
        ContainsSubstring("outside [1,4]"));
    REQUIRE_THROWS_WITH(
        solver.set_solution_column_permutation_1based({1, 2, 3, 5}),
        ContainsSubstring("outside [1,4]"));
    REQUIRE_THROWS_WITH(
        solver.set_solution_column_permutation_1based({1, 2, 2, 4}),
        ContainsSubstring("repeats original column 2"));

    CHECK_NOTHROW(solver.set_solution_column_permutation_1based({3, 1, 4, 2}));
    REQUIRE_THROWS_WITH(
        solver.set_solution_column_permutation_1based({1, 2, 3, 4}),
        ContainsSubstring("already configured"));

    const std::array<int, 3> irn{{1, 2, 3}};
    const std::array<int, 3> jcn{{1, 2, 3}};
    REQUIRE_THROWS_WITH(
        solver.set_pattern(3, 3, irn.data(), jcn.data()),
        ContainsSubstring("set_pattern received n=3"));
    CHECK_NOTHROW(solver.clear_solution_column_permutation());
    CHECK_NOTHROW(solver.set_solution_column_permutation_1based({1, 2, 3, 4}));
    solver.reset();
    CHECK_NOTHROW(solver.set_solution_column_permutation_1based({4, 3, 2, 1}));

    const std::array<int, 4> identity_irn{{1, 2, 3, 4}};
    const std::array<int, 4> identity_jcn{{1, 2, 3, 4}};
    MumpsLinearSolver analyzed_solver(4, 5, false, 0, 100, MPI_COMM_WORLD);
    analyzed_solver.set_pattern(4, 4, identity_irn.data(), identity_jcn.data());
    analyzed_solver.analyze_pattern();
    REQUIRE_THROWS_WITH(
        analyzed_solver.set_solution_column_permutation_1based({1, 2, 3, 4}),
        ContainsSubstring("before analyze_pattern"));
    REQUIRE_THROWS_WITH(
        analyzed_solver.clear_solution_column_permutation(),
        ContainsSubstring("cannot be cleared after analyze_pattern"));
}

TEST_CASE("MUMPS block analysis validates partitions and matching opt-out",
          "[linear_solver][mumps-block-analysis]")
{
    ensure_mpi_initialized();
    using Matching = MumpsLinearSolver::BlockAnalysisMatching;

    SECTION("default policy refuses silent matching loss") {
        MumpsLinearSolver solver(4, 7, false, 0, 100, MPI_COMM_WORLD);
        REQUIRE_THROWS_WITH(
            solver.enable_block_analysis({1, 3, 5}),
            ContainsSubstring("cannot preserve the configured matching"));
        CHECK_FALSE(solver.block_analysis_enabled());
    }

    SECTION("block pointers must be a non-empty partition of every variable") {
        MumpsLinearSolver solver(4, 7, false, 0, 100, MPI_COMM_WORLD);
        REQUIRE_THROWS_WITH(
            solver.enable_block_analysis(
                {1, 3, 3, 5}, {},
                Matching::ExplicitlyDisable),
            ContainsSubstring("BLKPTR must be strictly increasing"));
        REQUIRE_THROWS_WITH(
            solver.enable_block_analysis(
                {1, 3, 4}, {},
                Matching::ExplicitlyDisable),
            ContainsSubstring("BLKPTR must end at n+1=5"));
    }

    SECTION("noncontiguous variables must be a permutation") {
        MumpsLinearSolver solver(4, 7, false, 0, 100, MPI_COMM_WORLD);
        REQUIRE_THROWS_WITH(
            solver.enable_block_analysis(
                {1, 3, 5}, {1, 3, 3, 4},
                Matching::ExplicitlyDisable),
            ContainsSubstring("BLKVAR repeats variable 3"));
    }

    SECTION("user ordering must be a bijection compatible with every block") {
        MumpsLinearSolver solver(4, 7, false, 0, 100, MPI_COMM_WORLD);
        solver.enable_block_analysis(
            {1, 3, 5}, {},
            Matching::ExplicitlyDisable);
        REQUIRE_THROWS_WITH(
            solver.set_user_permutation_1based({1, 3, 2, 4}),
            ContainsSubstring("incompatible with block 1"));
        REQUIRE_THROWS_WITH(
            solver.set_user_permutation_1based({1, 2, 2, 4}),
            ContainsSubstring("repeats pivot position 2"));
    }
}

TEST_CASE("MUMPS analysis rejects rank-divergent experimental metadata collectively",
          "[linear_solver][mumps-analysis-metadata]")
{
    ensure_mpi_initialized();
    using Matching = MumpsLinearSolver::BlockAnalysisMatching;

    int rank = 0;
    int size = 0;
    REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
    REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, &size) == MPI_SUCCESS);
    if (size < 2)
        SKIP("requires at least two MPI ranks");

    const std::array<int, 4> irn{{1, 2, 3, 4}};
    const std::array<int, 4> jcn{{1, 2, 3, 4}};
    const std::array<double, 4> values{{1.0, 1.0, 1.0, 1.0}};
    const std::array<const char*, 12> divergence_cases{{
        "block enabled state",
        "BLKPTR same-size content",
        "BLKVAR empty versus explicit identity",
        "BLKVAR same-size content",
        "user permutation enabled state",
        "PERM_IN same-size content",
        "symmetric-permutation JOB=1 entry",
        "Schur JOB=1 entry before local block refusal",
        "OOC mode",
        "OOC touch",
        "OOC safety",
        "OOC budget override",
    }};

    for (std::size_t case_index = 0;
         case_index < divergence_cases.size(); ++case_index) {
        CAPTURE(case_index, divergence_cases[case_index], rank);
        std::string refusal_message;
        int locally_refused = 0;
        {
            const MumpsOutOfCoreMode ooc_mode =
                case_index == 8 && rank != 0
                    ? MumpsOutOfCoreMode::Auto
                    : MumpsOutOfCoreMode::Off;
            const double ooc_touch =
                case_index == 9 && rank != 0 ? 1.4 : 1.3;
            const double ooc_safety =
                case_index == 10 && rank != 0 ? 0.6 : 0.7;
            const double ooc_budget_mb =
                case_index == 11 && rank != 0 ? 1024.0 : -1.0;
            // With two ranks on one test node, rank 1 is deliberately excluded
            // from MUMPS. It must still join the world-communicator metadata
            // check and receive the same refusal as the factor rank.
            MumpsLinearSolver solver(
                4, 7, ooc_mode, 0, 100, MPI_COMM_WORLD, 1, false,
                ooc_touch, ooc_safety, ooc_budget_mb);
            CHECK(solver.analysis_rank_count() < size);
            solver.set_pattern(4, 4, irn.data(), jcn.data());

            switch (case_index) {
            case 0:
            case 6:
            case 7:
                if (rank != 0) {
                    solver.enable_block_analysis(
                        {1, 3, 5}, {}, Matching::ExplicitlyDisable);
                }
                break;
            case 1:
                solver.enable_block_analysis(
                    rank == 0 ? std::vector<int>{1, 3, 5}
                              : std::vector<int>{1, 2, 5},
                    {}, Matching::ExplicitlyDisable);
                break;
            case 2:
                solver.enable_block_analysis(
                    {1, 3, 5},
                    rank == 0 ? std::vector<int>{}
                              : std::vector<int>{1, 2, 3, 4},
                    Matching::ExplicitlyDisable);
                break;
            case 3:
                solver.enable_block_analysis(
                    {1, 3, 5},
                    rank == 0 ? std::vector<int>{1, 2, 3, 4}
                              : std::vector<int>{2, 1, 3, 4},
                    Matching::ExplicitlyDisable);
                break;
            case 4:
                if (rank != 0)
                    solver.set_user_permutation_1based({1, 2, 3, 4});
                break;
            case 5:
                solver.set_user_permutation_1based(
                    rank == 0 ? std::vector<int>{1, 2, 3, 4}
                              : std::vector<int>{2, 1, 3, 4});
                break;
            case 8:
            case 9:
            case 10:
            case 11:
                break;
            default:
                FAIL("unhandled metadata-divergence case");
            }

            try {
                if (case_index == 6) {
                    std::vector<int> symmetric_permutation;
                    solver.analyze_symmetric_permutation(
                        values.data(), symmetric_permutation);
                } else if (case_index == 7) {
                    std::vector<double> schur;
                    solver.extract_schur(values.data(), {1}, schur);
                } else {
                    solver.analyze_pattern();
                }
            } catch (const LinearSolverError& error) {
                locally_refused = 1;
                refusal_message = error.what();
            }
        }

        int every_rank_refused = 0;
        REQUIRE(MPI_Allreduce(&locally_refused, &every_rank_refused, 1,
                              MPI_INT, MPI_MIN,
                              MPI_COMM_WORLD) == MPI_SUCCESS);
        CHECK(every_rank_refused == 1);
        if (locally_refused) {
            CHECK_THAT(
                refusal_message,
                ContainsSubstring(
                    "MUMPS analysis metadata differs across world_comm_ ranks"));
        }
    }
}

TEST_CASE("MUMPS block analysis solves with contiguous and noncontiguous blocks",
          "[linear_solver][mumps-block-analysis]")
{
    ensure_mpi_initialized();
    using Matching = MumpsLinearSolver::BlockAnalysisMatching;
    int rank = 0;
    REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);

    const std::array<int, 16> irn{{
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
    }};
    const std::array<int, 16> jcn{{
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
    }};
    const std::array<double, 16> values{{
        4.0, 1.0, 1.0, 1.0,
        1.0, 5.0, 1.0, 1.0,
        1.0, 1.0, 6.0, 1.0,
        1.0, 1.0, 1.0, 7.0,
    }};

    for (const bool noncontiguous : {false, true}) {
        CAPTURE(noncontiguous);
        MumpsLinearSolver solver(4, 7, false, 0, 100, MPI_COMM_WORLD);
        solver.set_pattern(4, 16, irn.data(), jcn.data());
        if (noncontiguous) {
            solver.set_user_permutation_1based({1, 3, 2, 4});
            solver.enable_block_analysis(
                {1, 3, 5}, {1, 3, 2, 4},
                Matching::ExplicitlyDisable);
        } else {
            solver.enable_block_analysis(
                {1, 3, 5}, {},
                Matching::ExplicitlyDisable);
        }
        CHECK(solver.configured_block_count() == 2);
        CHECK(solver.estimated_factor_nnz() == 0);
        std::vector<int> symmetric_permutation;
        std::array<std::int32_t, 60> captured_icntl{};
        std::array<double, 15> captured_cntl{};
        REQUIRE_THROWS_WITH(
            solver.copy_symmetric_permutation_1based(symmetric_permutation),
            ContainsSubstring("before successful analysis"));
        REQUIRE_THROWS_WITH(
            solver.copy_analysis_controls(captured_icntl, captured_cntl),
            ContainsSubstring("before successful analysis"));
        solver.analyze_pattern();
        CHECK(solver.estimated_factor_nnz() > 0);
        CHECK(solver.estimated_factor_real_slots() > 0);
        CHECK(solver.estimated_factor_integer_slots() > 0);
        CHECK(solver.estimated_tree_node_count() > 0);
        CHECK(solver.estimated_factor_flops_gflop() > 0.0);
        CHECK(solver.analysis_rank_count() >= 1);
        CHECK(solver.factor_ranks_per_node() == 0);
        solver.copy_analysis_controls(captured_icntl, captured_cntl);
        if (rank == 0) {
            CHECK(captured_icntl[5] == 0);  // matching explicitly disabled
            CHECK(captured_icntl[14] == 1); // block analysis active
            CHECK(captured_icntl[7] != 0);  // scaling remains configured
        }
        solver.copy_symmetric_permutation_1based(symmetric_permutation);
        if (rank == 0) {
            REQUIRE(symmetric_permutation.size() == 4);
            std::sort(symmetric_permutation.begin(), symmetric_permutation.end());
            CHECK(symmetric_permutation == std::vector<int>{1, 2, 3, 4});
        } else {
            CHECK(symmetric_permutation.empty());
        }
        REQUIRE_THROWS_WITH(
            solver.set_user_permutation_1based({1, 2, 3, 4}),
            ContainsSubstring("before analyze_pattern"));
        REQUIRE_THROWS_WITH(
            solver.disable_block_analysis(),
            ContainsSubstring("cannot be disabled after analyze_pattern"));
        solver.factor_analyzed(values.data());
        CHECK(solver.factor_retry_count() == 0);
        CHECK(solver.successful_factor_icntl14() == solver.last_icntl14());

        std::array<double, 4> rhs{{13.0, 18.0, 25.0, 34.0}};
        solver.solve(rhs.data());
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            CHECK(rhs[i] == Catch::Approx(static_cast<double>(i + 1)).margin(1e-12));
        }
        CHECK(solver.delayed_pivot_count() == 0);
    }
}

TEST_CASE("Sparse direct MUMPS rank cap resolves explicit and default values",
          "[linear_solver][sparse-direct][mumps-rank-cap]")
{
    ensure_mpi_initialized();

    MumpsRuntimeConfig config;
    config.ranks_per_node = 0;
    CHECK(resolve_sparse_direct_mumps_ranks_per_node(
              config, MPI_COMM_WORLD) == 0);

    config.ranks_per_node = 1;
    CHECK(resolve_sparse_direct_mumps_ranks_per_node(
              config, MPI_COMM_WORLD) == 1);

    MPI_Comm shared_communicator = MPI_COMM_NULL;
    REQUIRE(MPI_Comm_split_type(
                MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                &shared_communicator) == MPI_SUCCESS);
    int local_size = 0;
    REQUIRE(MPI_Comm_size(shared_communicator, &local_size) == MPI_SUCCESS);

    config.ranks_per_node = -1;
    CHECK(resolve_sparse_direct_mumps_ranks_per_node(
              config, MPI_COMM_WORLD) == std::max(local_size / 4, 1));
    CHECK(MPI_Comm_free(&shared_communicator) == MPI_SUCCESS);
}

TEST_CASE("Sparse MUMPS system summary covers every reachable layout",
          "[linear_solver][sparse-direct][runtime-log]")
{
    const std::array<std::pair<SparseDirectMumpsSystemMode, const char*>, 3>
        cases{{
            {SparseDirectMumpsSystemMode::FullUnmasked,
             "Sector: full, Mask: off | System: 41257 (68832973 nnz)\n"},
            {SparseDirectMumpsSystemMode::FullMasked,
             "Sector: full, Mask: on | System: 41257 (68832973 nnz)\n"},
            {SparseDirectMumpsSystemMode::ReducedMasked,
             "Sector: reduced, Mask: on | System: 41257 (68832973 nnz)\n"},
        }};
    for (const auto& [mode, expected] : cases) {
        std::ostringstream output;
        print_sparse_direct_mumps_system_summary(
            output, mode, 41257, 68832973);
        CHECK(output.str() == expected);
    }

    JacobianParityMaskState parity_state;
    CHECK(sparse_direct_mumps_system_mode(false, nullptr) ==
          SparseDirectMumpsSystemMode::FullUnmasked);
    CHECK(sparse_direct_mumps_system_mode(false, &parity_state) ==
          SparseDirectMumpsSystemMode::FullUnmasked);
    parity_state.decision = JacobianParityMaskState::Decision::Engaged;
    CHECK(sparse_direct_mumps_system_mode(false, &parity_state) ==
          SparseDirectMumpsSystemMode::FullMasked);
    // Reduction itself implies masking; the API has no reduced/off state.
    CHECK(sparse_direct_mumps_system_mode(true, nullptr) ==
          SparseDirectMumpsSystemMode::ReducedMasked);
}

TEST_CASE("Sparse MUMPS compact timings and memory use shared renderers",
          "[linear_solver][sparse-direct][runtime-log]")
{
    ensure_mpi_initialized();
    int rank = 0;
    REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);

    std::ostringstream output;
    report_sparse_direct_mumps_jacobian_timing(
        MPI_COMM_WORLD, rank == 0 ? &output : nullptr,
        18.6239, 651504256);
    if (rank == 0) {
        CHECK(output.str() ==
              "Jacobian build+gather: 18.6239 s (COO: 651.504 MB)\n");
    } else {
        CHECK(output.str().empty());
    }

    output.str("");
    output.clear();
    report_sparse_direct_mumps_apply_timing(
        MPI_COMM_WORLD, rank == 0 ? &output : nullptr, 6.64068);
    if (rank == 0) {
        CHECK(output.str() == "MUMPS apply: 6.64068 s\n");
    } else {
        CHECK(output.str().empty());
    }

    output.str("");
    output.clear();
    report_sparse_direct_mumps_factorization(
        MPI_COMM_WORLD, rank == 0 ? &output : nullptr,
        "MUMPS analyze+factorize", 6.81782, 4.58729,
        "METIS", false, 1788, 2837);
    if (rank == 0) {
        CHECK(output.str() ==
              "MUMPS analyze+factorize: 6.81782 + 4.58729 s "
              "(METIS, OOC off, 1788 MB used, 2837 MB allocated)\n");
    } else {
        CHECK(output.str().empty());
    }
}

TEST_CASE("Sparse MUMPS COO allocation includes physical parity blocks",
          "[linear_solver][sparse-direct][runtime-log]")
{
    AssembledJacobianCoo matrix;
    matrix.irn.reserve(3);
    matrix.jcn.reserve(5);
    matrix.a.reserve(7);
    matrix.parity_blocks.resize(2);
    matrix.parity_blocks[0].irn.reserve(11);
    matrix.parity_blocks[0].jcn.reserve(13);
    matrix.parity_blocks[0].a.reserve(17);
    matrix.parity_blocks[1].irn.reserve(19);
    matrix.parity_blocks[1].jcn.reserve(23);
    matrix.parity_blocks[1].a.reserve(29);

    const std::size_t expected =
        (matrix.irn.capacity() + matrix.jcn.capacity() +
         matrix.parity_blocks[0].irn.capacity() +
         matrix.parity_blocks[0].jcn.capacity() +
         matrix.parity_blocks[1].irn.capacity() +
         matrix.parity_blocks[1].jcn.capacity()) * sizeof(int) +
        (matrix.a.capacity() + matrix.parity_blocks[0].a.capacity() +
         matrix.parity_blocks[1].a.capacity()) * sizeof(double);
    CHECK(sparse_direct_mumps_coo_allocated_bytes(matrix) == expected);
}

TEST_CASE("Sparse correction masking is identical on every rank",
          "[linear_solver][sparse-direct][zero-column]")
{
    ensure_mpi_initialized();
    const ScopedEnvironmentVariable zero_columns(
        "SPARSE_ZERO_COLUMN_LIST", "1,3,-1,9");

    Array<double> correction(4);
    for (int i = 0; i < 4; ++i)
        correction.set(i) = static_cast<double>(i + 1);
    zero_selected_sparse_corrections(correction, 4, nullptr);

    const std::array<double, 4> expected{{1.0, 0.0, 3.0, 0.0}};
    for (int i = 0; i < 4; ++i) {
        CAPTURE(i);
        CHECK(correction(i) == expected[static_cast<std::size_t>(i)]);

        double minimum = 0.0;
        double maximum = 0.0;
        const double local = correction(i);
        REQUIRE(MPI_Allreduce(&local, &minimum, 1, MPI_DOUBLE, MPI_MIN,
                              MPI_COMM_WORLD) == MPI_SUCCESS);
        REQUIRE(MPI_Allreduce(&local, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                              MPI_COMM_WORLD) == MPI_SUCCESS);
        CHECK(minimum == maximum);
    }
}

TEST_CASE("Sparse direct collective failure helper reaches all ranks",
          "[linear_solver][sparse-direct][mumps-collective-failure]")
{
    ensure_mpi_initialized();

    int rank = 0;
    int size = 0;
    REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
    REQUIRE(MPI_Comm_size(MPI_COMM_WORLD, &size) == MPI_SUCCESS);
    if (size < 2)
        SKIP("requires at least two MPI ranks");

    const std::string local_error =
        rank == 0 ? "forced rank-capped MUMPS failure" : std::string{};
    bool caught = false;
    std::string message;
    try {
        sparse_direct_collective_throw_if_failed(
            MPI_COMM_WORLD, "forced factorization", local_error);
    } catch (const LinearSolverError& error) {
        caught = true;
        message = error.what();
    }

    CHECK(caught);
    CHECK(message.find("forced factorization") != std::string::npos);
    CHECK(message.find("forced rank-capped MUMPS failure") !=
          std::string::npos);

    const int local_caught = caught ? 1 : 0;
    int caught_on_all_ranks = 0;
    REQUIRE(MPI_Allreduce(&local_caught, &caught_on_all_ranks, 1, MPI_INT,
                          MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS);
    CHECK(caught_on_all_ranks == 1);

    const int local_value = rank + 1;
    int world_sum = 0;
    REQUIRE(MPI_Allreduce(&local_value, &world_sum, 1, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD) == MPI_SUCCESS);
    CHECK(world_sum == size * (size + 1) / 2);
}

TEST_CASE("Sparse direct MUMPS primitive preserves ordinary and split solves",
          "[linear_solver][sparse-direct][mumps-orchestration]")
{
    ensure_mpi_initialized();
    JacobianParityMaskState parity_state = make_two_sector_state();

    SECTION("ordinary transient solve") {
        AssembledJacobianCoo matrix = make_two_sector_matrix();
        std::array<double, 4> rhs{{7.0, 18.0, 10.0, 28.0}};
        int icntl14 = 100;
        std::ostringstream diagnostic;
        SparseDirectMumpsSolveOptions options =
            make_direct_mumps_solve_options();
        options.measure_phases = true;
        options.report_apply_timing = true;
        options.diagnostic = &diagnostic;

        const SparseDirectMumpsSolveResult result =
            run_sparse_direct_mumps_solve(
                matrix, rhs.data(), &parity_state, icntl14, options);

        CHECK_FALSE(result.parity_split);
        CHECK(result.parity_layout ==
              SparseDirectMumpsParityLayout::None);
        CHECK_FALSE(result.retained_factor);
        CHECK(result.solve_seconds > 0.0);
        CHECK(matrix.irn.empty());
        CHECK(matrix.jcn.empty());
        CHECK(matrix.a.empty());
        check_two_sector_solution(rhs);

        int rank = 0;
        REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
        if (rank == 0) {
            const std::string output = diagnostic.str();
            const std::size_t summary =
                output.find("MUMPS analyze+factorize: ");
            const std::size_t apply = output.find("MUMPS apply: ", summary);
            REQUIRE(summary != std::string::npos);
            REQUIRE(apply != std::string::npos);
            REQUIRE(summary < apply);
            CHECK(output.find(" MB used, ", summary) != std::string::npos);
            CHECK(output.find(" MB allocated)", summary) !=
                  std::string::npos);
            CHECK(output.find("MUMPS ordering: ") == std::string::npos);
            CHECK(output.find("MUMPS OOC auto:") == std::string::npos);
            CHECK(output.find("(max; avg", apply) == std::string::npos);
            CHECK(output.find("[MUMPS: Parity ") == std::string::npos);
        }
    }

    SECTION("physical plus-only COO drives one parity block") {
        AssembledJacobianCoo matrix = make_physical_parity_matrix(true);
        std::array<double, 2> rhs{{7.0, 10.0}};
        int icntl14 = 100;
        std::ostringstream diagnostic;
        SparseDirectMumpsSolveOptions options =
            make_direct_mumps_solve_options();
        options.out_of_core_mode = MumpsOutOfCoreMode::Auto;
        options.out_of_core_budget_mb = 1024.0;
        options.diagnostic = &diagnostic;

        const SparseDirectMumpsSolveResult result =
            run_sparse_direct_mumps_solve(
                matrix, rhs.data(), &parity_state, icntl14, options);

        REQUIRE_FALSE(result.parity_split);
        CHECK(result.parity_layout ==
              SparseDirectMumpsParityLayout::Plus);
        CHECK(result.analyze_seconds > 0.0);
        CHECK(result.factorize_seconds > 0.0);
        CHECK(result.solve_seconds > 0.0);
        CHECK(rhs[0] == Catch::Approx(1.0).margin(1e-12));
        CHECK(rhs[1] == Catch::Approx(3.0).margin(1e-12));
        CHECK(matrix.parity_blocks.empty());

        int rank = 0;
        REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
        if (rank == 0) {
            const std::string output = diagnostic.str();
            const std::size_t header =
                output.find("[MUMPS: Parity +]\n");
            const std::size_t summary =
                output.find("analyze+factorize: ", header);
            const std::size_t apply = output.find("MUMPS apply: ", summary);

            REQUIRE(header != std::string::npos);
            REQUIRE(header < summary);
            REQUIRE(summary < apply);
            CHECK(output.find("OOC auto:") == std::string::npos);
            CHECK(output.find("MUMPS ordering:") == std::string::npos);
            CHECK(output.find("MUMPS analyze:") == std::string::npos);
            CHECK(output.find("MUMPS factorize:") == std::string::npos);
            CHECK(output.find(" MB used, ") != std::string::npos);
            CHECK(output.find(" MB allocated)") != std::string::npos);
            CHECK(output.find("(max; avg", apply) == std::string::npos);
        }
    }

    SECTION("classic approximate masked emission takes the transient split path") {
        AssembledJacobianCoo matrix = make_two_sector_matrix();
        CHECK(jacobian_parity_mask_emission_is_block_diagonal(
            parity_state, matrix.n, true, false));
        parity_state.approximate_engagement = true;
        CHECK_FALSE(jacobian_fused_parity_mask_ready(
            parity_state, matrix.n));
        CHECK(jacobian_parity_split_ready_for_next_emission(
            parity_state, matrix.n, true, true, true));
        matrix.parity_sector_block_diagonal =
            jacobian_parity_mask_emission_is_block_diagonal(
                parity_state, matrix.n, true, false);
        REQUIRE(matrix.parity_sector_block_diagonal);
        std::array<double, 4> rhs{{7.0, 18.0, 10.0, 28.0}};
        int icntl14 = 100;
        SparseDirectMumpsSolveOptions options =
            make_direct_mumps_solve_options();
        options.parity_split_requested = true;
        options.measure_phases = true;
        options.ordinary_factor_lifecycle =
            SparseDirectMumpsFactorLifecycle::Retained;

        const SparseDirectMumpsSolveResult result =
            run_sparse_direct_mumps_solve(
                matrix, rhs.data(), &parity_state, icntl14, options);

        CHECK(result.parity_split);
        CHECK_FALSE(result.retained_factor);
        CHECK(result.solve_seconds > 0.0);
        CHECK(matrix.irn.empty());
        CHECK(matrix.jcn.empty());
        CHECK(matrix.a.empty());
        check_two_sector_solution(rhs);
    }

    SECTION("physical plus-minus COO drives two ordered parity blocks") {
        AssembledJacobianCoo matrix = make_physical_parity_matrix(false);
        std::array<double, 4> rhs{{7.0, 18.0, 10.0, 28.0}};
        int icntl14 = 100;
        std::ostringstream diagnostic;
        SparseDirectMumpsSolveOptions options =
            make_direct_mumps_solve_options();
        options.out_of_core_mode = MumpsOutOfCoreMode::Auto;
        options.out_of_core_budget_mb = 1024.0;
        options.diagnostic = &diagnostic;

        const SparseDirectMumpsSolveResult result =
            run_sparse_direct_mumps_solve(
                matrix, rhs.data(), &parity_state, icntl14, options);

        REQUIRE(result.parity_split);
        CHECK(result.parity_layout ==
              SparseDirectMumpsParityLayout::PlusMinus);
        CHECK(matrix.parity_blocks.empty());
        check_two_sector_solution(rhs);

        int rank = 0;
        REQUIRE(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
        if (rank == 0) {
            const std::string output = diagnostic.str();
            const auto count_occurrences = [&](const std::string& needle) {
                std::size_t count = 0;
                std::size_t position = 0;
                while ((position = output.find(needle, position)) !=
                       std::string::npos) {
                    ++count;
                    position += needle.size();
                }
                return count;
            };

            const std::size_t plus_header =
                output.find("[MUMPS: Parity +]\n");
            const std::size_t plus_summary =
                output.find("analyze+factorize: ", plus_header);
            const std::size_t minus_header =
                output.find("[MUMPS: Parity -]\n", plus_summary);
            const std::size_t minus_summary =
                output.find("analyze+factorize: ", minus_header);
            const std::size_t apply =
                output.find("MUMPS apply: ", minus_summary);

            REQUIRE(plus_header != std::string::npos);
            REQUIRE(plus_header < plus_summary);
            REQUIRE(plus_summary < minus_header);
            REQUIRE(minus_header < minus_summary);
            REQUIRE(minus_summary < apply);
            CHECK(count_occurrences("[MUMPS: Parity ") == 2);
            CHECK(count_occurrences("analyze+factorize: ") == 2);
            CHECK(count_occurrences(" MB used, ") == 2);
            CHECK(count_occurrences(" MB allocated)") == 2);
            CHECK(count_occurrences("MUMPS apply: ") == 1);
            CHECK(output.find("OOC auto:") == std::string::npos);
            CHECK(output.find("MUMPS ordering:") == std::string::npos);
            CHECK(output.find("Jacobian parity split solve: sector") ==
                  std::string::npos);
        }
    }

    SECTION("verify emission keeps the single both-block solve") {
        AssembledJacobianCoo matrix = make_two_sector_matrix();
        parity_state.approximate_engagement = true;
        matrix.parity_sector_block_diagonal =
            jacobian_parity_mask_emission_is_block_diagonal(
                parity_state, matrix.n, true, true);
        REQUIRE_FALSE(matrix.parity_sector_block_diagonal);
        CHECK_FALSE(jacobian_parity_mask_emission_is_block_diagonal(
            parity_state, matrix.n, false, false));
        JacobianParityMaskState malformed_state = parity_state;
        malformed_state.row_sector.pop_back();
        CHECK_FALSE(jacobian_parity_mask_emission_is_block_diagonal(
            malformed_state, matrix.n, true, false));
        malformed_state = parity_state;
        malformed_state.column_sector[0] = 0;
        CHECK_FALSE(jacobian_parity_mask_emission_is_block_diagonal(
            malformed_state, matrix.n, true, false));

        JacobianParityMaskState fused_ready_state = parity_state;
        fused_ready_state.structural_labels_available = true;
        fused_ready_state.unmasked_full_j_emitted = true;
        REQUIRE(jacobian_fused_parity_mask_ready(
            fused_ready_state, matrix.n));
        CHECK_FALSE(jacobian_parity_split_ready_for_next_emission(
            fused_ready_state, matrix.n, true, true, true));
        CHECK(jacobian_parity_split_ready_for_next_emission(
            fused_ready_state, matrix.n, true, true, false));

        std::array<double, 4> rhs{{7.0, 18.0, 10.0, 28.0}};
        int icntl14 = 100;
        SparseDirectMumpsSolveOptions options =
            make_direct_mumps_solve_options();
        options.parity_split_requested = true;

        const SparseDirectMumpsSolveResult result =
            run_sparse_direct_mumps_solve(
                matrix, rhs.data(), &parity_state, icntl14, options);

        CHECK_FALSE(result.parity_split);
        check_two_sector_solution(rhs);
    }

    SECTION("retained ordinary factor supports chord reuse") {
        AssembledJacobianCoo matrix = make_two_sector_matrix();
        std::array<double, 4> rhs{{7.0, 18.0, 10.0, 28.0}};
        int icntl14 = 100;
        SparseDirectMumpsSolveOptions options =
            make_direct_mumps_solve_options();
        options.ordinary_factor_lifecycle =
            SparseDirectMumpsFactorLifecycle::Retained;

        SparseDirectMumpsSolveResult result =
            run_sparse_direct_mumps_solve(
                matrix, rhs.data(), &parity_state, icntl14, options);

        REQUIRE_FALSE(result.parity_split);
        REQUIRE(result.retained_factor);
        check_two_sector_solution(rhs);
        rhs = {{14.0, 36.0, 20.0, 56.0}};
        result.retained_factor->solve(rhs.data());
        constexpr std::array<double, 4> doubled{{2.0, 4.0, 6.0, 8.0}};
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            CAPTURE(i);
            CHECK(rhs[i] == Catch::Approx(doubled[i]).margin(1e-12));
        }
    }

    SECTION("retained ordinary replay is owned by the primitive") {
        AssembledJacobianCoo matrix = make_two_sector_matrix();
        const std::vector<int> replay_columns(matrix.jcn.begin(),
                                              matrix.jcn.end());
        const std::vector<int> identity_permutation{1, 2, 3, 4};
        SparseDirectMumpsReplayOptions replay;
        replay.column_indices_1based = &replay_columns;
        replay.symmetric_permutation_1based = &identity_permutation;
        replay.solution_column_permutation_1based = &identity_permutation;
        int analysis_observations = 0;
        int replay_successes = 0;
        int replay_failures = 0;
        int icntl14 = 100;
        SparseDirectMumpsSolveOptions options =
            make_direct_mumps_solve_options();
        options.ordinary_factor_lifecycle =
            SparseDirectMumpsFactorLifecycle::Retained;
        options.ordinary_replay = &replay;
        options.ordinary_analysis_observer =
            [&](const SparseDirectMumpsAnalysisSnapshot&) {
                ++analysis_observations;
            };
        options.ordinary_replay_success_observer =
            [&]() { ++replay_successes; };
        options.ordinary_replay_failure_observer =
            [&](const std::string&) { ++replay_failures; };

        SparseDirectMumpsSolveResult result =
            run_sparse_direct_mumps_solve(
                matrix, nullptr, &parity_state, icntl14, options);

        REQUIRE(result.retained_factor);
        CHECK(result.replay_attempted);
        CHECK(result.replay_succeeded);
        CHECK(result.replay_failure_reason.empty());
        CHECK(analysis_observations == 0);
        CHECK(replay_successes == 1);
        CHECK(replay_failures == 0);
        std::array<double, 4> rhs{{7.0, 18.0, 10.0, 28.0}};
        result.retained_factor->solve(rhs.data());
        check_two_sector_solution(rhs);
    }

    SECTION("invalid replay falls back to the configured ordinary solve") {
        AssembledJacobianCoo matrix = make_two_sector_matrix();
        const std::vector<int> replay_columns(matrix.jcn.begin(),
                                              matrix.jcn.end());
        const std::vector<int> invalid_symmetric_permutation{1, 2, 2, 4};
        const std::vector<int> identity_permutation{1, 2, 3, 4};
        SparseDirectMumpsReplayOptions replay;
        replay.column_indices_1based = &replay_columns;
        replay.symmetric_permutation_1based =
            &invalid_symmetric_permutation;
        replay.solution_column_permutation_1based = &identity_permutation;
        int analysis_observations = 0;
        int replay_successes = 0;
        int replay_failures = 0;
        int icntl14 = 100;
        SparseDirectMumpsSolveOptions options =
            make_direct_mumps_solve_options();
        options.ordinary_factor_lifecycle =
            SparseDirectMumpsFactorLifecycle::Retained;
        options.ordinary_replay = &replay;
        options.ordinary_analysis_observer =
            [&](const SparseDirectMumpsAnalysisSnapshot&) {
                ++analysis_observations;
            };
        options.ordinary_replay_success_observer =
            [&]() { ++replay_successes; };
        options.ordinary_replay_failure_observer =
            [&](const std::string&) { ++replay_failures; };

        SparseDirectMumpsSolveResult result =
            run_sparse_direct_mumps_solve(
                matrix, nullptr, &parity_state, icntl14, options);

        REQUIRE(result.retained_factor);
        CHECK(result.replay_attempted);
        CHECK_FALSE(result.replay_succeeded);
        CHECK_FALSE(result.replay_failure_reason.empty());
        CHECK(analysis_observations == 1);
        CHECK(replay_successes == 0);
        CHECK(replay_failures == 1);
        std::array<double, 4> rhs{{7.0, 18.0, 10.0, 28.0}};
        result.retained_factor->solve(rhs.data());
        check_two_sector_solution(rhs);
    }

    SECTION("retained split solves repeatedly") {
        AssembledJacobianCoo matrix = make_physical_parity_matrix(false);
        const std::vector<int> replay_columns(matrix.jcn.begin(),
                                              matrix.jcn.end());
        const std::vector<int> invalid_permutation{1, 2, 2, 4};
        SparseDirectMumpsReplayOptions replay;
        replay.column_indices_1based = &replay_columns;
        replay.symmetric_permutation_1based = &invalid_permutation;
        replay.solution_column_permutation_1based = &invalid_permutation;
        int replay_callbacks = 0;
        int icntl14 = 100;
        SparseDirectMumpsSolveOptions options =
            make_direct_mumps_solve_options();
        options.split_factor_lifecycle =
            SparseDirectMumpsFactorLifecycle::Retained;
        options.ordinary_replay = &replay;
        options.ordinary_replay_success_observer =
            [&]() { ++replay_callbacks; };
        options.ordinary_replay_failure_observer =
            [&](const std::string&) { ++replay_callbacks; };

        SparseDirectMumpsSolveResult result =
            run_sparse_direct_mumps_solve(
                matrix, nullptr, &parity_state, icntl14, options);

        REQUIRE(result.parity_split);
        CHECK(result.parity_layout ==
              SparseDirectMumpsParityLayout::PlusMinus);
        REQUIRE(result.retained_factor);
        CHECK_FALSE(result.replay_attempted);
        CHECK(replay_callbacks == 0);
        std::array<double, 4> rhs{{7.0, 18.0, 10.0, 28.0}};
        result.retained_factor->solve(rhs.data());
        check_two_sector_solution(rhs);
        rhs = {{14.0, 36.0, 20.0, 56.0}};
        result.retained_factor->solve(rhs.data());
        constexpr std::array<double, 4> doubled{{2.0, 4.0, 6.0, 8.0}};
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            CAPTURE(i);
            CHECK(rhs[i] == Catch::Approx(doubled[i]).margin(1e-12));
        }
    }
}

TEST_CASE("Sparse direct primitive and special paths retain MUMPS policy",
          "[linear_solver][sparse-direct][mumps-rank-cap][source-wiring]")
{
    const auto source_path = find_sparse_newton_source();
    std::ifstream input(source_path);
    REQUIRE(input.is_open());
    const std::string source((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());

    const std::array<const char*, 5> primitive_policy_forwarding{{
        "solve_options.out_of_core_mode = out_of_core_mode;",
        "solve_options.ranks_per_node = mumps_ranks_per_node;",
        "solve_options.out_of_core_touch = out_of_core_touch;",
        "solve_options.out_of_core_safety = out_of_core_safety;",
        "solve_options.out_of_core_budget_mb = out_of_core_budget_mb;",
    }};
    for (const char* assignment : primitive_policy_forwarding) {
        CAPTURE(assignment);
        CHECK(source.find(assignment) != std::string::npos);
    }

    // Two primitive-owned constructors cover ordinary and split solves. The
    // two concrete special-path constructors cover analyze reuse and replay
    // capture, whose mid-lifecycle MUMPS access cannot use the primitive.
    const auto constructions = sparse_direct_solver_constructions(source);
    REQUIRE(constructions.size() == 4);
    std::size_t primitive_constructions = 0;
    std::size_t special_constructions = 0;
    for (const auto& construction : constructions) {
        const bool primitive_owned =
            construction.find("options.ranks_per_node") != std::string::npos;
        const bool special_path =
            construction.find("mumps_ranks_per_node") != std::string::npos;
        CHECK(primitive_owned != special_path);
        if (primitive_owned) {
            ++primitive_constructions;
            CHECK(construction.find("options.out_of_core_mode") !=
                  std::string::npos);
            CHECK(construction.find("options.out_of_core_touch") !=
                  std::string::npos);
            CHECK(construction.find("options.out_of_core_safety") !=
                  std::string::npos);
            CHECK(construction.find("options.out_of_core_budget_mb") !=
                  std::string::npos);
        }
        if (special_path) {
            ++special_constructions;
            CHECK(construction.find("out_of_core_mode") != std::string::npos);
            CHECK(construction.find("out_of_core_touch") != std::string::npos);
            CHECK(construction.find("out_of_core_safety") != std::string::npos);
            CHECK(construction.find("out_of_core_budget_mb") !=
                  std::string::npos);
        }
    }
    CHECK(primitive_constructions == 2);
    CHECK(special_constructions == 2);

    CHECK(source.find(
              "sparse_direct_ranks_per_node ==\n"
              "                mumps_ranks_per_node") != std::string::npos);
    CHECK(source.find(
              "sparse_direct_ranks_per_node = mumps_ranks_per_node") !=
          std::string::npos);
    CHECK(source.find("sparse_direct_out_of_core_mode ==") !=
          std::string::npos);
    CHECK(source.find("sparse_direct_out_of_core_touch ==") !=
          std::string::npos);
    CHECK(source.find("sparse_direct_out_of_core_safety ==") !=
          std::string::npos);
    CHECK(source.find("sparse_direct_out_of_core_budget_mb ==") !=
          std::string::npos);
    CHECK(source.find(
              "solve_options.report_apply_timing = true;") !=
          std::string::npos);
    CHECK(source.find(
              "elapsed_time(chord_solve_start)") !=
          std::string::npos);
    CHECK(source.find("mumps_solve_seconds") == std::string::npos);
    const std::array<const char*, 7> emission_plan_wiring{{
        "plan_jacobian_emission(",
        "emission_caps.physical_payload_supported = true;",
        "emission_caps.analyze_reuse_requested = analyze_reuse_requested;",
        "emission_caps.replay_capture_requested =\n"
        "            direct_replay_capture_requested;",
        "emission_caps.parity_mass_probe_requested =\n"
        "            parity_mass_probe_requested;",
        "require_collective_jacobian_emission_plan_agreement(",
        "assembler.assemble(assembly_drop_tol, emission_plan);",
    }};
    for (const char* wiring : emission_plan_wiring) {
        CAPTURE(wiring);
        CHECK(source.find(wiring) != std::string::npos);
    }
    const std::array<const char*, 4> removed_duplicate_guards{{
        "const bool selected_parity_block_requested =",
        "const bool fused_parity_blocks_requested =",
        "const bool physical_parity_blocks_requested =",
        "assembler.assemble(assembly_drop_tol,\n"
        "                               !parity_mass_probe_requested,",
    }};
    for (const char* removed : removed_duplicate_guards) {
        CAPTURE(removed);
        CHECK(source.find(removed) == std::string::npos);
    }
    CHECK(source.find("solve_options.correction_timing_start") ==
          std::string::npos);
}

TEST_CASE("Reduced sparse guard failures restore the rejected step",
          "[linear_solver][sparse-direct][reduced-solve][source-wiring]")
{
    const auto source_path = find_sparse_newton_source();
    std::ifstream input(source_path);
    REQUIRE(input.is_open());
    const std::string source((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());

    const auto restore_helper =
        source.find("auto restore_rejected_reduced_step =");
    REQUIRE(restore_helper != std::string::npos);
    const auto restore_state =
        source.find("restore_state(snapshot);", restore_helper);
    const auto refresh_residual = source.find(
        "restored_residual = sec_member_partitioned();", restore_state);
    const auto refresh_error = source.find("error = 0.0;", refresh_residual);
    REQUIRE(restore_state != std::string::npos);
    REQUIRE(refresh_residual != std::string::npos);
    REQUIRE(refresh_error != std::string::npos);
    CHECK(restore_state < refresh_residual);
    CHECK(refresh_residual < refresh_error);

    std::size_t restore_calls = 0;
    constexpr const char* restore_call =
        "restore_rejected_reduced_step(";
    for (std::size_t position = source.find(restore_call);
         position != std::string::npos;
         position = source.find(restore_call, position + 1)) {
        ++restore_calls;
    }
    CHECK(restore_calls == 2); // chord and ordinary post-apply guard exits

    const auto chord_snapshot =
        source.find("State_snapshot chord_snapshot = snapshot_state();");
    const auto chord_apply = source.find(
        "espace.xx_to_vars_variable_domains(\n"
        "                        this, chord_delta, chord_offset);",
        chord_snapshot);
    const auto chord_restore = source.find(
        "restore_rejected_reduced_step(\n"
        "                                chord_snapshot, chord_residual);",
        chord_apply);
    const auto chord_forward = source.find(
        "store_forwarded_residual(\n"
        "                                    std::move(chord_residual));",
        chord_restore);
    REQUIRE(chord_snapshot != std::string::npos);
    REQUIRE(chord_apply != std::string::npos);
    REQUIRE(chord_restore != std::string::npos);
    REQUIRE(chord_forward != std::string::npos);
    CHECK(chord_snapshot < chord_apply);
    CHECK(chord_apply < chord_restore);
    CHECK(chord_restore < chord_forward);

    const auto ordinary_snapshot = source.find(
        "std::unique_ptr<State_snapshot> reduced_step_snapshot;");
    const auto ordinary_apply = source.find(
        "espace.xx_to_vars_variable_domains(this, full_delta, offset);",
        ordinary_snapshot);
    const auto ordinary_restore = source.find(
        "restore_rejected_reduced_step(\n"
        "                        *reduced_step_snapshot, trial_residual);",
        ordinary_apply);
    const auto ordinary_forward = source.find(
        "store_forwarded_residual(std::move(trial_residual));",
        ordinary_restore);
    REQUIRE(ordinary_snapshot != std::string::npos);
    REQUIRE(ordinary_apply != std::string::npos);
    REQUIRE(ordinary_restore != std::string::npos);
    REQUIRE(ordinary_forward != std::string::npos);
    CHECK(ordinary_snapshot < ordinary_apply);
    CHECK(ordinary_apply < ordinary_restore);
    CHECK(ordinary_restore < ordinary_forward);
}

TEST_CASE("JFNK routes retained ordinary and split PCs through direct MUMPS",
          "[linear_solver][sparse-direct][jfnk-mumps][source-wiring]")
{
    const auto source_path = find_jfnk_mumps_newton_source();
    std::ifstream input(source_path);
    REQUIRE(input.is_open());
    const std::string source((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());

    constexpr const char* primitive_call =
        "run_sparse_direct_mumps_solve(";
    const auto first_primitive_call = source.find(primitive_call);
    REQUIRE(first_primitive_call != std::string::npos);
    CHECK(source.find(primitive_call, first_primitive_call + 1) ==
          std::string::npos);
    CHECK(source.find("std::make_unique<MumpsLinearSolver>(") ==
          std::string::npos);
    CHECK(source.find("MumpsLinearSolver") == std::string::npos);

    const std::array<const char*, 5> policy_forwarding{{
        "solve_options.out_of_core_mode = out_of_core_mode;",
        "solve_options.ranks_per_node = factor_ranks_per_node;",
        "solve_options.out_of_core_touch = out_of_core_touch;",
        "solve_options.out_of_core_safety = out_of_core_safety;",
        "solve_options.out_of_core_budget_mb = out_of_core_budget_mb;",
    }};
    for (const char* assignment : policy_forwarding) {
        CAPTURE(assignment);
        CHECK(source.find(assignment) != std::string::npos);
    }
    CHECK(source.find(
              "solve_options.ordinary_factor_lifecycle =\n"
              "                SparseDirectMumpsFactorLifecycle::Retained;") !=
          std::string::npos);
    CHECK(source.find(
              "solve_options.split_factor_lifecycle =\n"
              "                SparseDirectMumpsFactorLifecycle::Retained;") !=
          std::string::npos);

    CHECK(source.find("SparseDirectMumpsReplayOptions replay_options;") !=
          std::string::npos);
    CHECK(source.find("replay_would_be_requested && !split_candidate") !=
          std::string::npos);
    CHECK(source.find("solve_options.ordinary_replay = replay_policy;") !=
          std::string::npos);

    const std::array<const char*, 8> identity_wiring{{
        "jfnk_preconditioner_ranks_per_node !=",
        "jfnk_preconditioner_ranks_per_node =",
        "!mumps_state.jfnk_preconditioner_emission_fingerprint ||",
        "*mumps_state.jfnk_preconditioner_emission_fingerprint !=\n"
        "                emission_plan.fingerprint",
        "mumps_state.jfnk_preconditioner_emission_fingerprint.reset();",
        "JacobianEmissionFingerprint retained_plan_fingerprint =",
        "retained_plan_fingerprint.parity_split_ready =",
        "mumps_state.jfnk_preconditioner_emission_fingerprint =",
    }};
    for (const char* wiring : identity_wiring) {
        CAPTURE(wiring);
        CHECK(source.find(wiring) != std::string::npos);
    }

    const std::array<const char*, 5> emission_plan_wiring{{
        "plan_jacobian_emission(",
        "emission_caps.physical_payload_supported = true;",
        "emission_caps.parity_mass_probe_requested =\n"
        "            parity_mass_probe_requested;",
        "require_collective_jacobian_emission_plan_agreement(",
        "emission_plan.fingerprint.parity_split_requested",
    }};
    for (const char* wiring : emission_plan_wiring) {
        CAPTURE(wiring);
        CHECK(source.find(wiring) != std::string::npos);
    }
    const std::array<const char*, 7> removed_duplicate_wiring{{
        "jacobian_parity_split_ready_for_next_emission(",
        "const bool selected_parity_block_requested =",
        "const bool fused_parity_blocks_requested =",
        "const bool physical_parity_blocks_requested =",
        "jfnk_preconditioner_row_sector",
        "jfnk_preconditioner_column_sector",
        "assembler.assemble(\n"
        "                drop_tol, !parity_mass_probe_requested,",
    }};
    for (const char* removed : removed_duplicate_wiring) {
        CAPTURE(removed);
        CHECK(source.find(removed) == std::string::npos);
    }

    const std::array<const char*, 7> reduced_system_wiring{{
        "make_jacobian_pre_j1_selection_plan(",
        "assembler.assemble(\n"
        "                drop_tol, emission_plan);",
        "step_selection_plan->selected_rows()",
        "step_selection_plan->selected_columns()",
        "gather_jacobian_selected_values(",
        "scatter_jacobian_selected_values(",
        "sparse_direct_mumps_system_mode(",
    }};
    for (const char* wiring : reduced_system_wiring) {
        CAPTURE(wiring);
        CHECK(source.find(wiring) != std::string::npos);
    }
    CHECK(source.find(
              "replay_would_be_requested && !split_candidate &&\n"
              "                !step_selection_plan") !=
          std::string::npos);
    CHECK(source.find(
              "replay_would_be_requested && split_candidate &&\n"
              "                !step_selection_plan && rank == 0") !=
          std::string::npos);

    CHECK(source.find(
              "report_sparse_direct_mumps_apply_timing(") !=
          std::string::npos);
    CHECK(source.find("\"MUMPS analyze\"") == std::string::npos);
    CHECK(source.find("\"MUMPS factorize\"") == std::string::npos);
}

TEST_CASE("Parity progress logs require CELEPHAIS_TIMING",
          "[linear_solver][parity-log][source-wiring]")
{
    const auto sparse_path = find_sparse_newton_source();
    const std::string sparse_source = read_source_file(sparse_path);
    const std::string jfnk_source =
        read_source_file(find_jfnk_mumps_newton_source());
    const std::string assembler_source =
        read_source_file(sparse_path.parent_path() / "jacobian_assembler.cpp");

    const std::array<const char*, 2> sparse_messages{{
        "Jacobian sector reduction residual: active_Linf=",
        "Jacobian sector reduction: inactive state drift Linf=",
    }};
    for (const char* message : sparse_messages) {
        CAPTURE(message);
        CHECK(every_log_occurrence_is_guarded_by(
            sparse_source, message, "rank == 0 && timing_enabled"));
    }

    const std::array<const char*, 2> jfnk_messages{{
        "JFNK sector reduction residual: active_Linf=",
        "JFNK sector reduction: inactive state drift Linf=",
    }};
    for (const char* message : jfnk_messages) {
        CAPTURE(message);
        CHECK(every_log_occurrence_is_guarded_by(
            jfnk_source, message, "rank == 0 && timing_enabled"));
    }

    CHECK(assembler_source.find(
              "Jacobian sector reduction: assembling selected block, n=") ==
          std::string::npos);
    CHECK(sparse_source.find(
              "sparse direct Newton (MUMPS solve+bcast+apply)") ==
          std::string::npos);
    CHECK(sparse_source.find("sparse direct Newton step total") ==
          std::string::npos);
    CHECK(sparse_source.find("MUMPS ordering:") == std::string::npos);
    CHECK(sparse_source.find("MUMPS OOC auto:") == std::string::npos);
    CHECK(jfnk_source.find("JFNK-MUMPS Newton step total") ==
          std::string::npos);
    CHECK(sparse_source.find("Jacobian sector reduction: pre-J1 REFUSED") ==
          std::string::npos);
    CHECK(jfnk_source.find("JFNK sector reduction: pre-J1 REFUSED") ==
          std::string::npos);
}

TEST_CASE("Sparse direct split refuses cross-sector COO entries collectively",
          "[linear_solver][sparse-direct][mumps-orchestration][failure]")
{
    ensure_mpi_initialized();
    AssembledJacobianCoo matrix = make_two_sector_matrix();
    matrix.irn.push_back(1);
    matrix.jcn.push_back(2);
    matrix.a.push_back(0.5);
    ++matrix.nnz;
    JacobianParityMaskState parity_state = make_two_sector_state();
    SparseDirectMumpsSolveOptions options = make_direct_mumps_solve_options();
    options.parity_split_requested = true;
    std::array<double, 4> rhs{{7.0, 18.0, 10.0, 28.0}};
    int icntl14 = 100;

    CHECK_THROWS_AS(
        run_sparse_direct_mumps_solve(
            matrix, rhs.data(), &parity_state, icntl14, options),
        LinearSolverError);
}

TEST_CASE("Sparse direct MUMPS rejects malformed physical parity order collectively",
          "[linear_solver][sparse-direct][mumps-orchestration][failure]")
{
    ensure_mpi_initialized();
    AssembledJacobianCoo matrix = make_physical_parity_matrix(false);
    matrix.parity_blocks[1].parity_label = 1;
    JacobianParityMaskState parity_state = make_two_sector_state();
    SparseDirectMumpsSolveOptions options = make_direct_mumps_solve_options();
    std::array<double, 4> rhs{{7.0, 18.0, 10.0, 28.0}};
    int icntl14 = 100;

    CHECK_THROWS_AS(
        run_sparse_direct_mumps_solve(
            matrix, rhs.data(), &parity_state, icntl14, options),
        LinearSolverError);
}

TEST_CASE("MUMPS ordering IDs have stable diagnostic names", "[mumps-ordering]") {
    constexpr std::array<std::pair<int, const char*>, 8> expected{{
        {0, "AMD"},
        {1, "user-provided"},
        {2, "AMF"},
        {3, "SCOTCH"},
        {4, "PORD"},
        {5, "METIS"},
        {6, "QAMD"},
        {7, "automatic"},
    }};

    for (const auto& [ordering, name] : expected) {
        CHECK(mumps_ordering_name(ordering) == name);
    }
    CHECK(mumps_ordering_name(42) == "unknown(42)");
}

TEST_CASE("MUMPS pattern superset preserves numerical values as explicit zeros",
          "[mumps-pattern-superset]")
{
    std::vector<int> pattern_irn;
    std::vector<int> pattern_jcn;
    std::vector<long long> offsets;
    std::vector<int> next_pattern_irn;
    std::vector<int> next_pattern_jcn;
    std::vector<long long> next_offsets;
    std::vector<double> aligned_values;

    // Column 2 deliberately arrives before column 1, and its rows are
    // deliberately unsorted. This matches rank-major gathered COO while
    // exercising the bounded per-column sort.
    const std::array<int, 4> irn{{3, 1, 2, 1}};
    const std::array<int, 4> jcn{{2, 2, 1, 3}};
    const std::array<double, 4> values{{
        std::numeric_limits<double>::quiet_NaN(), 4.0, 2.0, 1e-8}};
    const auto seed = update_mumps_pattern_superset(
        3, 4, irn.data(), jcn.data(), values.data(), 0.5,
        pattern_irn, pattern_jcn, offsets,
        next_pattern_irn, next_pattern_jcn, next_offsets, aligned_values);

    CHECK(seed.pattern_changed);
    CHECK(seed.candidate_nnz == 4);
    CHECK(seed.numerical_nnz == 2);
    CHECK(seed.superset_nnz == 4);
    CHECK(seed.new_pattern_entries == 4);
    CHECK(seed.explicit_zero_entries == 2);
    CHECK(pattern_irn.empty());
    CHECK(pattern_jcn.empty());
    CHECK(offsets.empty());
    CHECK(next_pattern_irn == std::vector<int>{2, 1, 3, 1});
    CHECK(next_pattern_jcn == std::vector<int>{1, 2, 2, 3});
    CHECK(next_offsets == std::vector<long long>{0, 1, 3, 4});
    CHECK(aligned_values == std::vector<double>{2.0, 4.0, 0.0, 0.0});
    pattern_irn.swap(next_pattern_irn);
    pattern_jcn.swap(next_pattern_jcn);
    offsets.swap(next_offsets);

    const std::array<int, 3> next_irn{{1, 2, 3}};
    const std::array<int, 3> next_jcn{{2, 1, 1}};
    // A coordinate exactly at the numerical threshold remains an explicit
    // zero and does not force support growth.
    const std::array<double, 3> next_values{{6.0, 5.0, 0.5}};
    const auto reuse = update_mumps_pattern_superset(
        3, 3, next_irn.data(), next_jcn.data(), next_values.data(), 0.5,
        pattern_irn, pattern_jcn, offsets,
        next_pattern_irn, next_pattern_jcn, next_offsets, aligned_values);

    CHECK_FALSE(reuse.pattern_changed);
    CHECK(reuse.new_pattern_entries == 0);
    CHECK(reuse.explicit_zero_entries == 2);
    CHECK(aligned_values == std::vector<double>{5.0, 6.0, 0.0, 0.0});
}

TEST_CASE("MUMPS fixed-threshold pattern reuses the seed analysis",
          "[mumps-pattern-superset][sparse-direct]")
{
    std::vector<int> pattern_irn;
    std::vector<int> pattern_jcn;
    std::vector<long long> offsets;
    std::vector<int> next_pattern_irn;
    std::vector<int> next_pattern_jcn;
    std::vector<long long> next_offsets;
    std::vector<double> aligned_values;

    constexpr double drop_tol = 0.5;
    const std::array<int, 3> seed_irn{{1, 2, 2}};
    const std::array<int, 3> seed_jcn{{1, 1, 2}};
    const std::array<double, 3> seed_values{{2.0, 3.0, 4.0}};
    const auto seed = update_mumps_pattern_superset(
        2, 3, seed_irn.data(), seed_jcn.data(), seed_values.data(), drop_tol,
        pattern_irn, pattern_jcn, offsets,
        next_pattern_irn, next_pattern_jcn, next_offsets, aligned_values);

    REQUIRE(seed.pattern_changed);
    CHECK(seed.numerical_nnz == seed.superset_nnz);
    CHECK(seed.explicit_zero_entries == 0);
    pattern_irn.swap(next_pattern_irn);
    pattern_jcn.swap(next_pattern_jcn);
    offsets.swap(next_offsets);

    // A retained coordinate may fall below the same fixed threshold. It is
    // represented by an explicit zero and does not invalidate JOB=1.
    const std::array<double, 3> next_values{{5.0, 0.25, 6.0}};
    const auto reuse = update_mumps_pattern_superset(
        2, 3, seed_irn.data(), seed_jcn.data(), next_values.data(), drop_tol,
        pattern_irn, pattern_jcn, offsets,
        next_pattern_irn, next_pattern_jcn, next_offsets, aligned_values);

    CHECK_FALSE(reuse.pattern_changed);
    CHECK(reuse.new_pattern_entries == 0);
    CHECK(reuse.explicit_zero_entries == 1);
    CHECK(aligned_values == std::vector<double>{5.0, 0.0, 6.0});
}

TEST_CASE("MUMPS pattern superset grows on a support miss and retains duplicates",
          "[mumps-pattern-superset][failure]")
{
    std::vector<int> pattern_irn;
    std::vector<int> pattern_jcn;
    std::vector<long long> offsets;
    std::vector<int> next_pattern_irn;
    std::vector<int> next_pattern_jcn;
    std::vector<long long> next_offsets;
    std::vector<double> aligned_values;

    const std::array<int, 2> seed_irn{{1, 1}};
    const std::array<int, 2> seed_jcn{{1, 1}};
    const std::array<double, 2> seed_values{{2.0, 3.0}};
    update_mumps_pattern_superset(
        2, 2, seed_irn.data(), seed_jcn.data(), seed_values.data(), 0.0,
        pattern_irn, pattern_jcn, offsets,
        next_pattern_irn, next_pattern_jcn, next_offsets, aligned_values);
    pattern_irn.swap(next_pattern_irn);
    pattern_jcn.swap(next_pattern_jcn);
    offsets.swap(next_offsets);

    const std::array<int, 3> next_irn{{1, 1, 2}};
    const std::array<int, 3> next_jcn{{1, 1, 1}};
    const std::array<double, 3> next_values{{5.0, 7.0, 11.0}};
    const auto growth = update_mumps_pattern_superset(
        2, 3, next_irn.data(), next_jcn.data(), next_values.data(), 0.0,
        pattern_irn, pattern_jcn, offsets,
        next_pattern_irn, next_pattern_jcn, next_offsets, aligned_values);

    CHECK(growth.pattern_changed);
    CHECK(growth.new_pattern_entries == 1);
    CHECK(pattern_irn == std::vector<int>{1, 1});
    CHECK(pattern_jcn == std::vector<int>{1, 1});
    CHECK(next_pattern_irn == std::vector<int>{1, 1, 2});
    CHECK(next_pattern_jcn == std::vector<int>{1, 1, 1});
    CHECK(aligned_values == std::vector<double>{5.0, 7.0, 11.0});
}

TEST_CASE("MUMPS pattern superset refuses non-contiguous column groups",
          "[mumps-pattern-superset][failure]")
{
    std::vector<int> pattern_irn;
    std::vector<int> pattern_jcn;
    std::vector<long long> offsets;
    std::vector<int> next_pattern_irn;
    std::vector<int> next_pattern_jcn;
    std::vector<long long> next_offsets;
    std::vector<double> aligned_values;
    const std::array<int, 3> irn{{1, 1, 2}};
    const std::array<int, 3> jcn{{1, 2, 1}};
    const std::array<double, 3> values{{1.0, 1.0, 1.0}};

    CHECK_THROWS(update_mumps_pattern_superset(
        2, 3, irn.data(), jcn.data(), values.data(), 0.0,
        pattern_irn, pattern_jcn, offsets,
        next_pattern_irn, next_pattern_jcn, next_offsets, aligned_values));
}
