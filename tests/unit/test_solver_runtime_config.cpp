#include "For_Kadath/System_of_eqs/solver_runtime_config.hpp"
#include "For_Kadath/Utilities/runtime_env.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{
    class EnvGuard
    {
      public:
        explicit EnvGuard(std::vector<std::string> names) : names_(std::move(names))
        {
            for (const auto& name : names_) {
                const char* value = std::getenv(name.c_str());
                old_values_.push_back(value ? std::optional<std::string>(value) : std::nullopt);
                unsetenv(name.c_str());
            }
        }

        ~EnvGuard()
        {
            for (std::size_t i = 0; i < names_.size(); ++i) {
                if (old_values_[i]) {
                    setenv(names_[i].c_str(), old_values_[i]->c_str(), 1);
                } else {
                    unsetenv(names_[i].c_str());
                }
            }
        }

        void set(const char* name, const char* value) { setenv(name, value, 1); }
        void unset(const char* name) { unsetenv(name); }

      private:
        std::vector<std::string> names_;
        std::vector<std::optional<std::string>> old_values_;
    };

    EnvGuard solver_env_guard()
    {
        return EnvGuard({
            "CELEPHAIS_SOLVER",
            "DROP_TOL",
            "MUMPS_ORDERING",
            "MUMPS_OOC",
            "MUMPS_OOC_TOUCH",
            "MUMPS_OOC_SAFETY",
            "MUMPS_OOC_BUDGET_MB",
            "MUMPS_BLR",
            "MUMPS_TREE_CACHE",
            "MUMPS_RANKS_PER_NODE",
            "SPARSE_MUMPS_ANALYZE_REUSE",
            "SPARSE_CHORD_REUSE",
            "SPARSE_PARITY",
            "SPARSE_PARITY_MASK",
            "SPARSE_SECTOR_REDUCE",
            "SPARSE_PARITY_SPLIT_SOLVE",
            "SPARSE_MUMPS_PATTERN_DROP_TOL",
            "SPARSE_MUMPS_SUPERSET_MAX_NNZ_RATIO",
            "JFNK_MAX_ITERS",
            "JFNK_MUMPS_PC_REFRESH",
            "JFNK_MUMPS_PC_ADAPTIVE",
            "JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS",
            "JFNK_RTOL",
            "JFNK_LINESEARCH",
            "CELEPHAIS_TIMING",
            "DEF_FILTER",
            "DIRECT_REPLAY_CAPTURE",
            "DIRECT_REPLAY_CAPTURE_ORDINAL",
        });
    }
} // namespace

using Catch::Matchers::ContainsSubstring;

TEST_CASE("SolverRuntimeConfig defaults are production-only", "[solver-runtime-config]")
{
    auto env = solver_env_guard();

    const Kadath::SolverRuntimeConfig config = Kadath::SolverRuntimeConfig::from_environment();

    CHECK(config.backend == Kadath::NewtonBackend::JfnkMumps);
    CHECK_FALSE(config.backend_explicitly_selected);
    CHECK(config.sparse_parity_mask);
    CHECK(config.sparse_sector_reduce);
    CHECK_FALSE(config.sparse_parity_split_solve);
    CHECK(config.mumps.drop_tol == Catch::Approx(1e-14));
    CHECK(config.mumps.ordering == 7);
    CHECK(config.mumps.out_of_core == Kadath::MumpsOutOfCoreMode::Auto);
    CHECK(config.mumps.out_of_core_touch == Catch::Approx(1.3));
    CHECK(config.mumps.out_of_core_safety == Catch::Approx(0.7));
    CHECK(config.mumps.out_of_core_budget_mb == Catch::Approx(-1.0));
    CHECK(config.mumps.blr == 0);
    CHECK_FALSE(config.mumps.tree_cache_enabled);
    CHECK(config.mumps.tree_cache_path.empty());
    CHECK(config.mumps.ranks_per_node == -1);
    CHECK_FALSE(config.mumps.sparse_analyze_reuse);
    CHECK(config.mumps.sparse_chord_reuse);
    CHECK(config.mumps.sparse_pattern_drop_tol == Catch::Approx(-1.0));
    CHECK(config.mumps.sparse_superset_max_nnz_ratio == Catch::Approx(2.0));
    CHECK(config.jfnk_mumps.max_linear_iterations == 64);
    CHECK(config.jfnk_mumps.preconditioner_refresh_steps == 10);
    CHECK_FALSE(config.jfnk_mumps.adaptive_preconditioner_refresh);
    CHECK(config.jfnk_mumps.adaptive_preconditioner_max_steps == 20);
    CHECK(config.jfnk_mumps.linear_relative_tolerance == Catch::Approx(1e-8));
    CHECK(config.jfnk_mumps.line_search); // global default on (bit-exact on every space)
    CHECK_FALSE(config.diagnostics.timing);
    CHECK(config.diagnostics.def_filter);
    CHECK(config.diagnostics.direct_replay_capture_path.empty());
    CHECK(config.diagnostics.direct_replay_capture_ordinal == 1);
}

TEST_CASE("SolverRuntimeConfig maps supported environment variables", "[solver-runtime-config]")
{
    auto env = solver_env_guard();
    env.set("CELEPHAIS_SOLVER", "mumps");
    env.set("DROP_TOL", "1e-6");
    env.set("MUMPS_ORDERING", "7");
    env.set("MUMPS_OOC", "1");
    env.set("MUMPS_OOC_TOUCH", "1.5");
    env.set("MUMPS_OOC_SAFETY", "0.6");
    env.set("MUMPS_OOC_BUDGET_MB", "2048");
    env.set("MUMPS_BLR", "2");
    env.set("SPARSE_MUMPS_ANALYZE_REUSE", "yes");
    env.set("SPARSE_CHORD_REUSE", "1");
    env.set("SPARSE_SECTOR_REDUCE", "yes");
    env.set("SPARSE_MUMPS_PATTERN_DROP_TOL", "1e-18");
    env.set("SPARSE_MUMPS_SUPERSET_MAX_NNZ_RATIO", "1.5");
    env.set("CELEPHAIS_TIMING", "1");
    env.set("DEF_FILTER", "0");
    env.set("DIRECT_REPLAY_CAPTURE", "/tmp/first.kcsr");
    env.set("DIRECT_REPLAY_CAPTURE_ORDINAL", "3");

    const Kadath::SolverRuntimeConfig config = Kadath::SolverRuntimeConfig::from_environment();

    CHECK(config.backend == Kadath::NewtonBackend::Mumps);
    CHECK(config.backend_explicitly_selected);
    CHECK(config.mumps.drop_tol == Catch::Approx(1e-6));
    CHECK(config.mumps.ordering == 7);
    CHECK(config.mumps.out_of_core == Kadath::MumpsOutOfCoreMode::On);
    CHECK(config.mumps.out_of_core_touch == Catch::Approx(1.5));
    CHECK(config.mumps.out_of_core_safety == Catch::Approx(0.6));
    CHECK(config.mumps.out_of_core_budget_mb == Catch::Approx(2048.0));
    CHECK(config.mumps.blr == 2);
    CHECK(config.mumps.sparse_analyze_reuse);
    CHECK(config.mumps.sparse_chord_reuse);
    CHECK(config.sparse_sector_reduce);
    CHECK(config.mumps.sparse_pattern_drop_tol == Catch::Approx(1e-18));
    CHECK(config.mumps.sparse_superset_max_nnz_ratio == Catch::Approx(1.5));
    CHECK(config.diagnostics.timing);
    CHECK_FALSE(config.diagnostics.def_filter);
    CHECK(config.diagnostics.direct_replay_capture_path == "/tmp/first.kcsr");
    CHECK(config.diagnostics.direct_replay_capture_ordinal == 3);
}

TEST_CASE("SolverRuntimeConfig maps dense backend override", "[solver-runtime-config]")
{
    auto env = solver_env_guard();
    env.set("CELEPHAIS_SOLVER", "dense");

    const Kadath::SolverRuntimeConfig config = Kadath::SolverRuntimeConfig::from_environment();

    CHECK(config.backend == Kadath::NewtonBackend::Dense);
    CHECK(config.backend_explicitly_selected);
}

TEST_CASE("SolverRuntimeConfig maps JFNK-MUMPS backend and controls", "[solver-runtime-config]")
{
    auto env = solver_env_guard();
    env.set("CELEPHAIS_SOLVER", "jfnk-mumps");
    env.set("JFNK_MAX_ITERS", "48");
    env.set("JFNK_MUMPS_PC_REFRESH", "6");
    env.set("JFNK_MUMPS_PC_ADAPTIVE", "1");
    env.set("JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS", "24");
    env.set("JFNK_RTOL", "1e-10");
    env.set("JFNK_LINESEARCH", "1");

    const Kadath::SolverRuntimeConfig config = Kadath::SolverRuntimeConfig::from_environment();

    CHECK(config.backend == Kadath::NewtonBackend::JfnkMumps);
    CHECK(config.backend_explicitly_selected);
    CHECK(config.jfnk_mumps.max_linear_iterations == 48);
    CHECK(config.jfnk_mumps.preconditioner_refresh_steps == 6);
    CHECK(config.jfnk_mumps.adaptive_preconditioner_refresh);
    CHECK(config.jfnk_mumps.adaptive_preconditioner_max_steps == 24);
    CHECK(config.jfnk_mumps.linear_relative_tolerance == Catch::Approx(1e-10));
    CHECK(config.jfnk_mumps.line_search);
}

TEST_CASE("SolverRuntimeConfig ignores retired or invalid values", "[solver-runtime-config]")
{
    auto env = solver_env_guard();
    env.set("CELEPHAIS_SOLVER", "unsupported");
    env.set("DROP_TOL", "-1.0");
    env.set("MUMPS_ORDERING", "99");
    env.set("MUMPS_OOC", "yes");
    env.set("MUMPS_OOC_TOUCH", "1.3junk");
    env.set("MUMPS_OOC_SAFETY", "inf");
    env.set("MUMPS_OOC_BUDGET_MB", "-2");
    env.set("MUMPS_BLR", "true");
    env.set("SPARSE_MUMPS_ANALYZE_REUSE", "off");
    env.set("SPARSE_CHORD_REUSE", "off");
    env.set("SPARSE_MUMPS_PATTERN_DROP_TOL", "-1");
    env.set("SPARSE_MUMPS_SUPERSET_MAX_NNZ_RATIO", "0");
    env.set("JFNK_MAX_ITERS", "48oops");
    env.set("JFNK_MUMPS_PC_REFRESH", "6oops");
    env.set("JFNK_MUMPS_PC_ADAPTIVE", "off");
    env.set("JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS", "24oops");
    env.set("JFNK_RTOL", "-1e-6");
    env.set("JFNK_LINESEARCH", "0");
    env.set("DIRECT_REPLAY_CAPTURE_ORDINAL", "0");

    const Kadath::SolverRuntimeConfig config = Kadath::SolverRuntimeConfig::from_environment();

    CHECK(config.backend == Kadath::NewtonBackend::JfnkMumps);
    CHECK_FALSE(config.backend_explicitly_selected);
    CHECK(config.mumps.drop_tol == Catch::Approx(1e-14));
    CHECK(config.mumps.ordering == 7);
    CHECK(config.mumps.out_of_core == Kadath::MumpsOutOfCoreMode::Auto);
    CHECK(config.mumps.out_of_core_touch == Catch::Approx(1.3));
    CHECK(config.mumps.out_of_core_safety == Catch::Approx(0.7));
    CHECK(config.mumps.out_of_core_budget_mb == Catch::Approx(-1.0));
    CHECK(config.mumps.blr == 1);
    CHECK_FALSE(config.mumps.sparse_analyze_reuse);
    CHECK(config.mumps.sparse_chord_reuse);
    CHECK(config.mumps.sparse_pattern_drop_tol == Catch::Approx(-1.0));
    CHECK(config.mumps.sparse_superset_max_nnz_ratio == Catch::Approx(2.0));
    CHECK(config.jfnk_mumps.max_linear_iterations == 64);
    CHECK(config.jfnk_mumps.preconditioner_refresh_steps == 10);
    CHECK_FALSE(config.jfnk_mumps.adaptive_preconditioner_refresh);
    CHECK(config.jfnk_mumps.adaptive_preconditioner_max_steps == 20);
    CHECK(config.jfnk_mumps.linear_relative_tolerance == Catch::Approx(1e-8));
    CHECK_FALSE(config.jfnk_mumps.line_search);
    CHECK(config.diagnostics.direct_replay_capture_ordinal == 1);
}

TEST_CASE("SolverRuntimeConfig parses the exact MUMPS OOC tri-state",
          "[solver-runtime-config][mumps-ooc]")
{
    auto env = solver_env_guard();

    SECTION("unset defaults to Auto") {
        CHECK(Kadath::SolverRuntimeConfig::from_environment().mumps.out_of_core ==
              Kadath::MumpsOutOfCoreMode::Auto);
    }
    SECTION("exact zero forces Off") {
        env.set("MUMPS_OOC", "0");
        CHECK(Kadath::SolverRuntimeConfig::from_environment().mumps.out_of_core ==
              Kadath::MumpsOutOfCoreMode::Off);
    }
    SECTION("exact one forces On") {
        env.set("MUMPS_OOC", "1");
        CHECK(Kadath::SolverRuntimeConfig::from_environment().mumps.out_of_core ==
              Kadath::MumpsOutOfCoreMode::On);
    }
    SECTION("exact auto retains Auto") {
        env.set("MUMPS_OOC", "auto");
        CHECK(Kadath::SolverRuntimeConfig::from_environment().mumps.out_of_core ==
              Kadath::MumpsOutOfCoreMode::Auto);
    }
    SECTION("invalid values fall back to Auto") {
        env.set("MUMPS_OOC", "AUTO");
        CHECK(Kadath::SolverRuntimeConfig::from_environment().mumps.out_of_core ==
              Kadath::MumpsOutOfCoreMode::Auto);
    }
    SECTION("zero test budget is accepted without becoming a production default") {
        env.set("MUMPS_OOC_BUDGET_MB", "0");
        CHECK(Kadath::SolverRuntimeConfig::from_environment()
                  .mumps.out_of_core_budget_mb == Catch::Approx(0.0));
    }
}

TEST_CASE("BNS binary stages own their default backend and preserve explicit choices",
          "[solver-runtime-config][bns-backend-routing]")
{
    auto env = solver_env_guard();

    SECTION("unset uses the stage default instead of ambient backend state") {
        auto ambient_config = Kadath::SolverRuntimeConfig::from_environment();
        ambient_config.backend = Kadath::NewtonBackend::Dense;
        const auto config = ambient_config.with_stage_default_backend(
            Kadath::NewtonBackend::JfnkMumps);
        CHECK(config.backend == Kadath::NewtonBackend::JfnkMumps);
        CHECK_FALSE(config.backend_explicitly_selected);
    }

    SECTION("explicit MUMPS wins") {
        env.set("CELEPHAIS_SOLVER", "mumps");
        const auto config = Kadath::SolverRuntimeConfig::from_environment()
                                .with_stage_default_backend(Kadath::NewtonBackend::JfnkMumps);
        CHECK(config.backend == Kadath::NewtonBackend::Mumps);
        CHECK(config.backend_explicitly_selected);
    }

    SECTION("explicit JFNK-MUMPS remains explicit") {
        env.set("CELEPHAIS_SOLVER", "jfnk-mumps");
        const auto config = Kadath::SolverRuntimeConfig::from_environment()
                                .with_stage_default_backend(Kadath::NewtonBackend::JfnkMumps);
        CHECK(config.backend == Kadath::NewtonBackend::JfnkMumps);
        CHECK(config.backend_explicitly_selected);
    }

    SECTION("invalid input does not become an explicit choice") {
        env.set("CELEPHAIS_SOLVER", "unsupported");
        const auto config = Kadath::SolverRuntimeConfig::from_environment()
                                .with_stage_default_backend(Kadath::NewtonBackend::JfnkMumps);
        CHECK(config.backend == Kadath::NewtonBackend::JfnkMumps);
        CHECK_FALSE(config.backend_explicitly_selected);
    }
}

TEST_CASE("SolverRuntimeConfig rejects oversized integer controls",
          "[solver-runtime-config][failure]")
{
    auto env = solver_env_guard();
    env.set("MUMPS_BLR", "2147483648");
    env.set("JFNK_MAX_ITERS", "2147483648");

    const Kadath::SolverRuntimeConfig config =
        Kadath::SolverRuntimeConfig::from_environment();

    CHECK(config.mumps.blr == 0);
    CHECK(config.jfnk_mumps.max_linear_iterations == 64);
}

TEST_CASE("Default-on sparse controls opt out only with exact zero",
          "[solver-runtime-config][sparse-chord-reuse][parity_mask][sector-reduce][mumps-tree-cache]")
{
    auto env = solver_env_guard();

    CHECK(Kadath::SolverRuntimeConfig::from_environment()
              .mumps.sparse_chord_reuse);
    CHECK(Kadath::SolverRuntimeConfig::from_environment().sparse_parity_mask);
    CHECK(Kadath::SolverRuntimeConfig::from_environment().sparse_sector_reduce);

    const char* enabled_values[] = {
        "", "1", "false", "off", "no", "00", "invalid"};
    for (const char* value : enabled_values) {
        env.set("SPARSE_CHORD_REUSE", value);
        env.set("SPARSE_PARITY_MASK", value);
        env.set("SPARSE_SECTOR_REDUCE", value);
        CAPTURE(value);
        CHECK(Kadath::SolverRuntimeConfig::from_environment()
                  .mumps.sparse_chord_reuse);
        CHECK(Kadath::SolverRuntimeConfig::from_environment().sparse_parity_mask);
        CHECK(Kadath::SolverRuntimeConfig::from_environment()
                  .sparse_sector_reduce);
    }

    // The tree cache is default-OFF opt-in (env_flag_enabled semantics).
    env.unset("MUMPS_TREE_CACHE");
    CHECK_FALSE(Kadath::SolverRuntimeConfig::from_environment()
                    .mumps.tree_cache_enabled);
    const char* tree_cache_disabled_values[] = {"", "0", "false", "off", "no"};
    for (const char* value : tree_cache_disabled_values) {
        env.set("MUMPS_TREE_CACHE", value);
        CAPTURE(value);
        CHECK_FALSE(Kadath::SolverRuntimeConfig::from_environment()
                        .mumps.tree_cache_enabled);
    }
    const char* tree_cache_enabled_values[] = {"1", "true", "on", "00", "invalid"};
    for (const char* value : tree_cache_enabled_values) {
        env.set("MUMPS_TREE_CACHE", value);
        CAPTURE(value);
        CHECK(Kadath::SolverRuntimeConfig::from_environment()
                  .mumps.tree_cache_enabled);
    }

    env.set("SPARSE_CHORD_REUSE", "0");
    env.set("SPARSE_PARITY_MASK", "0");
    env.set("SPARSE_SECTOR_REDUCE", "0");
    env.set("MUMPS_TREE_CACHE", "0");
    CHECK_FALSE(Kadath::SolverRuntimeConfig::from_environment()
                    .mumps.sparse_chord_reuse);
    CHECK_FALSE(
        Kadath::SolverRuntimeConfig::from_environment().sparse_parity_mask);
    CHECK_FALSE(
        Kadath::SolverRuntimeConfig::from_environment().sparse_sector_reduce);
    CHECK_FALSE(Kadath::SolverRuntimeConfig::from_environment()
                    .mumps.tree_cache_enabled);
}

TEST_CASE("SPARSE_PARITY maps each permission-ladder rung",
          "[solver-runtime-config][parity_mask][sparse-parity]")
{
    auto env = solver_env_guard();
    struct LadderCase
    {
        const char* rung;
        bool mask;
        bool split;
        bool reduce;
    };
    const LadderCase cases[] = {
        {"off", false, false, false},
        {"mask", true, false, false},
        {"split", true, true, false},
        {"reduce", true, true, true},
    };
    for (const LadderCase& test : cases) {
        env.set("SPARSE_PARITY", test.rung);
        const Kadath::SolverRuntimeConfig config =
            Kadath::SolverRuntimeConfig::from_environment();
        CAPTURE(test.rung);
        CHECK(config.sparse_parity_mask == test.mask);
        CHECK(config.sparse_parity_split_solve == test.split);
        CHECK(config.sparse_sector_reduce == test.reduce);
    }
}

TEST_CASE("Unset SPARSE_PARITY preserves independent legacy flags",
          "[solver-runtime-config][parity_mask][sparse-parity]")
{
    auto env = solver_env_guard();
    env.set("SPARSE_PARITY_MASK", "0");
    env.set("SPARSE_PARITY_SPLIT_SOLVE", "1");
    env.set("SPARSE_SECTOR_REDUCE", "0");

    const Kadath::SolverRuntimeConfig config =
        Kadath::SolverRuntimeConfig::from_environment();
    CHECK_FALSE(config.sparse_parity_mask);
    CHECK(config.sparse_parity_split_solve);
    CHECK_FALSE(config.sparse_sector_reduce);
}

TEST_CASE("SPARSE_PARITY rejects invalid exact values",
          "[solver-runtime-config][parity_mask][sparse-parity][failure]")
{
    auto env = solver_env_guard();
    const char* invalid_values[] = {"", "typo", "REDUCE"};
    for (const char* value : invalid_values) {
        env.set("SPARSE_PARITY", value);
        CAPTURE(value);
        CHECK_THROWS_WITH(
            Kadath::SolverRuntimeConfig::from_environment(),
            ContainsSubstring("SPARSE_PARITY has invalid value '") &&
                ContainsSubstring(std::string("'") + value + "'"));
    }
}

TEST_CASE("SPARSE_PARITY rejects contradictory legacy aliases",
          "[solver-runtime-config][parity_mask][sparse-parity][failure]")
{
    auto env = solver_env_guard();
    struct ConflictCase
    {
        const char* rung;
        const char* alias;
        const char* alias_value;
    };
    const ConflictCase cases[] = {
        {"off", "SPARSE_PARITY_MASK", "1"},
        {"off", "SPARSE_PARITY_SPLIT_SOLVE", "1"},
        {"off", "SPARSE_SECTOR_REDUCE", "1"},
        {"reduce", "SPARSE_PARITY_MASK", "0"},
        {"reduce", "SPARSE_PARITY_SPLIT_SOLVE", "0"},
        {"reduce", "SPARSE_SECTOR_REDUCE", "0"},
    };
    for (const ConflictCase& test : cases) {
        env.set("SPARSE_PARITY", test.rung);
        env.set(test.alias, test.alias_value);
        CAPTURE(test.rung, test.alias, test.alias_value);
        CHECK_THROWS_WITH(
            Kadath::SolverRuntimeConfig::from_environment(),
            ContainsSubstring(std::string("SPARSE_PARITY=") + test.rung) &&
                ContainsSubstring(std::string(test.alias) + "=" +
                                  test.alias_value));
        env.unset(test.alias);
    }
}

TEST_CASE("SPARSE_PARITY accepts agreeing legacy aliases",
          "[solver-runtime-config][parity_mask][sparse-parity]")
{
    auto env = solver_env_guard();
    struct RedundancyCase
    {
        const char* rung;
        const char* mask;
        const char* split;
        const char* reduce;
    };
    const RedundancyCase cases[] = {
        {"off", "0", "0", "0"},
        {"mask", "1", "0", "0"},
        {"split", "1", "1", "0"},
        {"reduce", "1", "1", "1"},
    };
    for (const RedundancyCase& test : cases) {
        env.set("SPARSE_PARITY", test.rung);
        env.set("SPARSE_PARITY_MASK", test.mask);
        env.set("SPARSE_PARITY_SPLIT_SOLVE", test.split);
        env.set("SPARSE_SECTOR_REDUCE", test.reduce);
        const Kadath::SolverRuntimeConfig config =
            Kadath::SolverRuntimeConfig::from_environment();
        CAPTURE(test.rung);
        CHECK(config.sparse_parity_mask == (std::string(test.mask) == "1"));
        CHECK(config.sparse_parity_split_solve ==
              (std::string(test.split) == "1"));
        CHECK(config.sparse_sector_reduce ==
              (std::string(test.reduce) == "1"));
    }
}

TEST_CASE("Parity split solve is opt-in", "[solver-runtime-config][parity_mask]")
{
    auto env = solver_env_guard();

    CHECK_FALSE(Kadath::SolverRuntimeConfig::from_environment()
                    .sparse_parity_split_solve);
    const char* disabled_values[] = {"", "0", "false", "off", "no"};
    for (const char* value : disabled_values) {
        env.set("SPARSE_PARITY_SPLIT_SOLVE", value);
        CAPTURE(value);
        CHECK_FALSE(Kadath::SolverRuntimeConfig::from_environment()
                        .sparse_parity_split_solve);
    }
    const char* enabled_values[] = {"1", "true", "on", "00", "invalid"};
    for (const char* value : enabled_values) {
        env.set("SPARSE_PARITY_SPLIT_SOLVE", value);
        CAPTURE(value);
        CHECK(Kadath::SolverRuntimeConfig::from_environment()
                  .sparse_parity_split_solve);
    }
}

TEST_CASE("MUMPS tree cache stage path is an immutable config copy",
          "[solver-runtime-config][mumps-tree-cache]")
{
    const Kadath::SolverRuntimeConfig base;
    const auto stage = base.with_mumps_tree_cache_path(
        "/tmp/converged_UNIROT.mumpstree");

    CHECK(base.mumps.tree_cache_path.empty());
    CHECK(stage.mumps.tree_cache_path ==
          "/tmp/converged_UNIROT.mumpstree");
    CHECK_FALSE(stage.mumps.tree_cache_enabled);
}

TEST_CASE("MUMPS tree cache replay requires every eligibility condition",
          "[solver-runtime-config][mumps-tree-cache]")
{
    CHECK(Kadath::should_replay_mumps_tree_cache(
        true, false, true, true, true));

    CHECK_FALSE(Kadath::should_replay_mumps_tree_cache(
        false, false, true, true, true));
    CHECK_FALSE(Kadath::should_replay_mumps_tree_cache(
        true, true, true, true, true));
    CHECK_FALSE(Kadath::should_replay_mumps_tree_cache(
        true, false, false, true, true));
    CHECK_FALSE(Kadath::should_replay_mumps_tree_cache(
        true, false, true, false, true));
    CHECK_FALSE(Kadath::should_replay_mumps_tree_cache(
        true, false, true, true, false));
}

TEST_CASE("Sparse-direct drop policy preserves the initial adaptive threshold then freezes it",
          "[solver-runtime-config][sparse-direct]")
{
    Kadath::MumpsRuntimeConfig config;
    config.drop_tol = 1e-14;
    Kadath::SparseDirectDropState state;

    const auto first = Kadath::resolve_sparse_direct_drop_policy(
        config, 4.5e-4, 36489, state);
    const double historical_first_threshold =
        std::max(1e-16, config.drop_tol * std::sqrt(std::sqrt(4.5e-4)));
    CHECK(first.numerical_drop_tol == historical_first_threshold);
    CHECK(first.pattern_drop_tol == historical_first_threshold);

    const auto later = Kadath::resolve_sparse_direct_drop_policy(
        config, 1e-12, 36489, state);
    CHECK(later.numerical_drop_tol == first.numerical_drop_tol);
    CHECK(later.pattern_drop_tol == first.pattern_drop_tol);

    state.reset();
    const auto restarted = Kadath::resolve_sparse_direct_drop_policy(
        config, 1e-12, 36489, state);
    CHECK(restarted.numerical_drop_tol == Catch::Approx(1e-16));
    CHECK(restarted.numerical_drop_tol != first.numerical_drop_tol);
    CHECK(restarted.pattern_drop_tol == restarted.numerical_drop_tol);
}

TEST_CASE("Sparse-direct drop policy clamps explicit pattern thresholds",
          "[solver-runtime-config][sparse-direct]")
{
    Kadath::MumpsRuntimeConfig config;
    config.drop_tol = 2e-12;
    Kadath::SparseDirectDropState state;

    config.sparse_pattern_drop_tol = 1e-14;
    auto policy = Kadath::resolve_sparse_direct_drop_policy(
        config, 1.0, 17, state);
    CHECK(policy.numerical_drop_tol == Catch::Approx(2e-12));
    CHECK(policy.pattern_drop_tol == Catch::Approx(1e-14));

    config.sparse_pattern_drop_tol = 1e-10;
    policy = Kadath::resolve_sparse_direct_drop_policy(
        config, 1e-20, 17, state);
    CHECK(policy.pattern_drop_tol == Catch::Approx(2e-12));

    config.drop_tol = 4e-12;
    policy = Kadath::resolve_sparse_direct_drop_policy(
        config, 1.0, 17, state);
    CHECK(policy.numerical_drop_tol == Catch::Approx(4e-12));

    policy = Kadath::resolve_sparse_direct_drop_policy(
        config, 1e-8, 18, state);
    CHECK(policy.numerical_drop_tol == Catch::Approx(4e-14));
}

TEST_CASE("Sparse-direct drop policy rejects non-finite or non-positive controls",
          "[solver-runtime-config][sparse-direct][failure]")
{
    Kadath::MumpsRuntimeConfig config;
    config.drop_tol = std::numeric_limits<double>::infinity();
    config.sparse_pattern_drop_tol =
        std::numeric_limits<double>::quiet_NaN();
    Kadath::SparseDirectDropState state;

    auto policy = Kadath::resolve_sparse_direct_drop_policy(
        config, std::numeric_limits<double>::infinity(), 9, state);
    CHECK(policy.numerical_drop_tol == Catch::Approx(1e-16));
    CHECK(policy.pattern_drop_tol == Catch::Approx(1e-16));

    config.drop_tol = 0.0;
    config.sparse_pattern_drop_tol = -2.0;
    state.reset();
    policy = Kadath::resolve_sparse_direct_drop_policy(
        config, -1.0, 9, state);
    CHECK(policy.numerical_drop_tol == Catch::Approx(1e-16));
    CHECK(policy.pattern_drop_tol == Catch::Approx(1e-16));

    config.sparse_analyze_reuse = false;
    state.reset();
    policy = Kadath::resolve_sparse_direct_drop_policy(
        config, -1.0, 9, state);
    CHECK(policy.pattern_drop_tol == policy.numerical_drop_tol);
}

TEST_CASE("canonical integer environment parser rejects overflow and suffixes",
          "[solver-runtime-config][failure]")
{
    EnvGuard env({"TEST_INTEGER"});

    env.set("TEST_INTEGER", "17oops");
    CHECK(Kadath::env_int_value("TEST_INTEGER", 9) == 9);

    env.set("TEST_INTEGER", "2147483648");
    CHECK(Kadath::env_int_value("TEST_INTEGER", 9) == 9);

    env.set("TEST_INTEGER", "-17");
    CHECK(Kadath::env_int_value("TEST_INTEGER", 9) == -17);
}

TEST_CASE("SolverRuntimeConfig normalizes adaptive maximum to fixed cadence",
          "[solver-runtime-config][adaptive-pc-refresh]")
{
    auto env = solver_env_guard();
    env.set("JFNK_MUMPS_PC_REFRESH", "8");
    env.set("JFNK_MUMPS_PC_ADAPTIVE", "1");
    env.set("JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS", "3");

    const Kadath::SolverRuntimeConfig config =
        Kadath::SolverRuntimeConfig::from_environment();

    CHECK(config.jfnk_mumps.preconditioner_refresh_steps == 8);
    CHECK(config.jfnk_mumps.adaptive_preconditioner_max_steps == 8);
}

TEST_CASE("Adaptive PC refresh preserves fixed cadence without evidence",
          "[solver-runtime-config][adaptive-pc-refresh]")
{
    Kadath::JfnkPreconditionerRefreshInput input;
    input.has_preconditioner = true;
    input.adaptive_enabled = true;
    input.preconditioner_age = 5;
    input.fixed_refresh_steps = 5;
    input.adaptive_max_steps = 20;

    const auto decision = Kadath::decide_jfnk_preconditioner_refresh(input);

    CHECK(decision.refresh);
    CHECK(decision.reason ==
          Kadath::JfnkPreconditionerRefreshReason::AdaptiveKrylovFailure);
}

TEST_CASE("Fixed PC refresh behavior is unchanged when adaptation is disabled",
          "[solver-runtime-config][adaptive-pc-refresh]")
{
    Kadath::JfnkPreconditionerRefreshInput input;
    input.has_preconditioner = true;
    input.fixed_refresh_steps = 5;
    input.preconditioner_age = 4;

    auto decision = Kadath::decide_jfnk_preconditioner_refresh(input);
    CHECK_FALSE(decision.refresh);
    CHECK_FALSE(decision.adaptively_deferred);
    CHECK(decision.reason ==
          Kadath::JfnkPreconditionerRefreshReason::ReuseBeforeCadence);

    input.preconditioner_age = 5;
    decision = Kadath::decide_jfnk_preconditioner_refresh(input);
    CHECK(decision.refresh);
    CHECK(decision.reason == Kadath::JfnkPreconditionerRefreshReason::FixedCadence);
}

TEST_CASE("Adaptive PC refresh defers only a healthy economic rebuild",
          "[solver-runtime-config][adaptive-pc-refresh]")
{
    Kadath::JfnkPreconditionerRefreshInput input;
    input.has_preconditioner = true;
    input.adaptive_enabled = true;
    input.previous_krylov_usable = true;
    input.preconditioner_age = 5;
    input.fixed_refresh_steps = 5;
    input.adaptive_max_steps = 20;
    input.previous_krylov_iterations = 12;
    input.max_krylov_iterations = 48;
    input.nonlinear_error_ratio = 0.25;
    input.previous_krylov_seconds = 1.2;
    input.previous_refresh_seconds = 12.0;

    const auto decision = Kadath::decide_jfnk_preconditioner_refresh(input);

    CHECK_FALSE(decision.refresh);
    CHECK(decision.adaptively_deferred);
    CHECK(decision.reason == Kadath::JfnkPreconditionerRefreshReason::AdaptiveReuse);
    CHECK(decision.projected_stale_krylov_seconds == Catch::Approx(4.8));
}

TEST_CASE("Adaptive PC refresh rejects stagnation and uneconomic stale solves",
          "[solver-runtime-config][adaptive-pc-refresh]")
{
    Kadath::JfnkPreconditionerRefreshInput input;
    input.has_preconditioner = true;
    input.adaptive_enabled = true;
    input.previous_krylov_usable = true;
    input.preconditioner_age = 5;
    input.fixed_refresh_steps = 5;
    input.adaptive_max_steps = 20;
    input.previous_krylov_iterations = 40;
    input.max_krylov_iterations = 48;
    input.nonlinear_error_ratio = 0.95;
    input.previous_krylov_seconds = 8.0;
    input.previous_refresh_seconds = 12.0;

    auto decision = Kadath::decide_jfnk_preconditioner_refresh(input);
    CHECK(decision.refresh);
    CHECK(decision.reason ==
          Kadath::JfnkPreconditionerRefreshReason::AdaptivePoorProgress);

    input.nonlinear_error_ratio = 0.5;
    input.previous_refresh_seconds = 8.0;
    decision = Kadath::decide_jfnk_preconditioner_refresh(input);
    CHECK(decision.refresh);
    CHECK(decision.reason ==
          Kadath::JfnkPreconditionerRefreshReason::AdaptiveKrylovCost);
    CHECK(decision.projected_stale_krylov_seconds == Catch::Approx(11.52));
}

TEST_CASE("Adaptive PC refresh recovery and hard limit are unconditional",
          "[solver-runtime-config][adaptive-pc-refresh]")
{
    Kadath::JfnkPreconditionerRefreshInput input;
    input.has_preconditioner = true;
    input.adaptive_enabled = true;
    input.previous_krylov_usable = true;
    input.preconditioner_age = 20;
    input.fixed_refresh_steps = 5;
    input.adaptive_max_steps = 20;
    input.previous_krylov_iterations = 2;
    input.max_krylov_iterations = 48;
    input.nonlinear_error_ratio = 0.1;
    input.previous_krylov_seconds = 0.1;
    input.previous_refresh_seconds = 12.0;

    auto decision = Kadath::decide_jfnk_preconditioner_refresh(input);
    CHECK(decision.refresh);
    CHECK(decision.reason ==
          Kadath::JfnkPreconditionerRefreshReason::AdaptiveHardLimit);

    input.preconditioner_age = 5;
    input.recovery_retry = true;
    decision = Kadath::decide_jfnk_preconditioner_refresh(input);
    CHECK(decision.refresh);
    CHECK(decision.reason == Kadath::JfnkPreconditionerRefreshReason::RecoveryRetry);
}

TEST_CASE("Adaptive PC cost projection is safe at integer limits",
          "[solver-runtime-config][adaptive-pc-refresh][failure]")
{
    Kadath::JfnkPreconditionerRefreshInput input;
    input.has_preconditioner = true;
    input.adaptive_enabled = true;
    input.previous_krylov_usable = true;
    input.preconditioner_age = 5;
    input.fixed_refresh_steps = 5;
    input.adaptive_max_steps = std::numeric_limits<int>::max();
    input.previous_krylov_iterations = std::numeric_limits<int>::max();
    input.max_krylov_iterations = std::numeric_limits<int>::max();
    input.nonlinear_error_ratio = 0.5;
    input.previous_krylov_seconds = 1.0;
    input.previous_refresh_seconds = 2.0;

    const auto decision = Kadath::decide_jfnk_preconditioner_refresh(input);

    CHECK_FALSE(decision.refresh);
    CHECK(decision.projected_stale_krylov_seconds == Catch::Approx(1.0));
}

TEST_CASE("Sparse chord reuse accepts only strict contraction",
          "[solver-runtime-config][sparse-chord-reuse]")
{
    using Action = Kadath::SparseChordReuseAction;
    using Reason = Kadath::SparseChordReuseReason;

    Kadath::SparseChordReuseInput input;
    input.candidate_evaluated = true;
    input.previous_error = 10.0;
    const double threshold =
        Kadath::kSparseChordAcceptanceTheta * input.previous_error;

    input.candidate_error = std::nextafter(threshold, 0.0);
    auto decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::AcceptChord);
    CHECK(decision.reason == Reason::SufficientContraction);

    input.candidate_error = std::nextafter(
        threshold, std::numeric_limits<double>::infinity());
    decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::RefreshJacobian);
    CHECK(decision.reason == Reason::InsufficientContraction);

    input.candidate_error = threshold;
    decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::RefreshJacobian);
    CHECK(decision.reason == Reason::InsufficientContraction);

    input.previous_error = 0.0;
    input.candidate_error = 0.0;
    decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::RefreshJacobian);
    CHECK(decision.reason == Reason::InsufficientContraction);
}

TEST_CASE("Sparse chord reuse engages only for a finite warm solve entry",
          "[solver-runtime-config][sparse-chord-reuse]")
{
    constexpr double tolerance = Kadath::kSparseChordEntryTolerance;

    CHECK(Kadath::sparse_chord_entry_allowed(4.5e-4));
    CHECK_FALSE(Kadath::sparse_chord_entry_allowed(0.1266));
    CHECK(Kadath::sparse_chord_entry_allowed(
        std::nextafter(tolerance, 0.0)));
    CHECK_FALSE(Kadath::sparse_chord_entry_allowed(tolerance));
    CHECK_FALSE(Kadath::sparse_chord_entry_allowed(
        std::nextafter(tolerance, std::numeric_limits<double>::infinity())));
    CHECK_FALSE(Kadath::sparse_chord_entry_allowed(
        std::numeric_limits<double>::quiet_NaN()));
    CHECK_FALSE(Kadath::sparse_chord_entry_allowed(-1.0));
}

TEST_CASE("Sparse chord reuse refreshes at its consecutive-step limit",
          "[solver-runtime-config][sparse-chord-reuse]")
{
    using Action = Kadath::SparseChordReuseAction;
    using Reason = Kadath::SparseChordReuseReason;

    Kadath::SparseChordReuseInput input;
    input.consecutive_chord_steps =
        Kadath::kSparseChordConsecutiveStepLimit - 1;

    auto decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::AttemptChord);
    CHECK(decision.reason == Reason::WithinConsecutiveLimit);

    input.consecutive_chord_steps =
        Kadath::kSparseChordConsecutiveStepLimit;
    decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::RefreshJacobian);
    CHECK(decision.reason == Reason::ConsecutiveLimit);

    input.candidate_evaluated = true;
    input.previous_error = 1.0;
    input.candidate_error = 0.0;
    decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::RefreshJacobian);
    CHECK(decision.reason == Reason::ConsecutiveLimit);
}

TEST_CASE("Sparse chord reuse fails closed on invalid inputs",
          "[solver-runtime-config][sparse-chord-reuse][failure]")
{
    using Action = Kadath::SparseChordReuseAction;
    using Reason = Kadath::SparseChordReuseReason;

    Kadath::SparseChordReuseInput input;
    input.candidate_evaluated = true;
    input.previous_error = 1.0;

    input.candidate_error = std::numeric_limits<double>::quiet_NaN();
    auto decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::RefreshJacobian);
    CHECK(decision.reason == Reason::InvalidInput);

    input.candidate_error = 0.0;
    input.previous_error = std::numeric_limits<double>::infinity();
    input.candidate_evaluated = false;
    decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::RefreshJacobian);
    CHECK(decision.reason == Reason::InvalidInput);

    input.candidate_evaluated = true;
    input.previous_error = -1.0;
    decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::RefreshJacobian);
    CHECK(decision.reason == Reason::InvalidInput);

    input.previous_error = 1.0;
    input.consecutive_chord_steps = -1;
    decision = Kadath::decide_sparse_chord_reuse(input);
    CHECK(decision.action == Action::RefreshJacobian);
    CHECK(decision.reason == Reason::InvalidInput);
}
