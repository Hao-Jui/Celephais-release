#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Apps/Formalism/Shared/PreBinary/bco_binary_setup.hpp"
#include "Apps/Workflow/bns_app_workflow.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

using namespace Kadath;
using Catch::Matchers::ContainsSubstring;

namespace {

using BnsWorkflowConfig = kadath_config<BIN_INFO>;

BnsWorkflowConfig make_bns_workflow_config()
{
    BnsWorkflowConfig config;
    config.initialize_binary({"ns", "ns"});
    config.set_defaults();
    config.set(BIN_RES) = 9;
    config.seq_setting(INIT_RES) = 9;
    config.control(REGRID) = false;
    return config;
}

struct BnsWorkflowPolicyWithoutAmr {
    using config_t = BnsWorkflowConfig;
    using space_t = Kadath::Space_bin_ns;

    inline static int solve_count = 0;

    static void reset()
    {
        solve_count = 0;
    }

    static STAGE low_res_stage() { return QUASI_EQUIL; }
    static void setup_bin_config(config_t&) {}
    static void setup_space(config_t&) {}
    static bool stop_after_setup() { return false; }
    static std::string regrid_filename() { return "unused-regrid"; }
    static void regrid(config_t&, const std::string&) {}
    static bool stop_after_regrid() { return false; }

    static void solve(config_t&, const std::string&)
    {
        ++solve_count;
    }
};

struct BnsWorkflowPolicyWithAmr : BnsWorkflowPolicyWithoutAmr {
    inline static std::vector<int> amr_cycles;
    inline static std::vector<int> amr_resolutions;

    static void reset()
    {
        BnsWorkflowPolicyWithoutAmr::reset();
        amr_cycles.clear();
        amr_resolutions.clear();
    }

    static bool amr_refine(config_t& config, const std::string&, int, int cycle)
    {
        amr_cycles.push_back(cycle);
        amr_resolutions.push_back(static_cast<int>(config(BIN_RES)));
        return true;
    }
};

struct BnsWorkflowPolicyWithStatefulAmr : BnsWorkflowPolicyWithoutAmr {
    inline static std::vector<int> amr_cycles;
    inline static std::vector<double> previous_radial_demands;

    static void reset()
    {
        BnsWorkflowPolicyWithoutAmr::reset();
        amr_cycles.clear();
        previous_radial_demands.clear();
    }

    static bool amr_refine(config_t&,
                           const std::string&,
                           int,
                           int cycle,
                           KadathApps::bns_amr::RefinementPolicyState& policy_state)
    {
        amr_cycles.push_back(cycle);
        previous_radial_demands.push_back(policy_state.previous_radial_demand);
        policy_state.record_h_baseline(100.0 + cycle, 1, {});
        return cycle == 0;
    }
};

struct BnsWorkflowPolicyReloadOnce : BnsWorkflowPolicyWithoutAmr {
    inline static int solve_calls = 0;
    inline static std::vector<int> warmup_flags;

    static void reset()
    {
        solve_calls = 0;
        warmup_flags.clear();
    }

    static int solve(config_t&, const std::string&, bool want_warmup)
    {
        ++solve_calls;
        warmup_flags.push_back(want_warmup ? 1 : 0);
        return solve_calls == 1 ? Kadath::RELOAD_FILE : EXIT_SUCCESS;
    }
};

struct BnsWorkflowColdSetupPolicy : BnsWorkflowPolicyWithoutAmr {
    inline static int setup_binary_res = 0;
    inline static int setup_ns1_res = 0;
    inline static int setup_ns2_res = 0;

    static void reset()
    {
        BnsWorkflowPolicyWithoutAmr::reset();
        setup_binary_res = 0;
        setup_ns1_res = 0;
        setup_ns2_res = 0;
    }

    static void setup_space(config_t& config)
    {
        setup_binary_res = static_cast<int>(config(BIN_RES));
        setup_ns1_res = static_cast<int>(config(BCO_RES, BCO1));
        setup_ns2_res = static_cast<int>(config(BCO_RES, BCO2));
    }
};

static_assert(!KadathApps::bns_workflow_detail::AmrRefineHook<
              BnsWorkflowPolicyWithoutAmr>);
static_assert(KadathApps::bns_workflow_detail::AmrRefineHook<
              BnsWorkflowPolicyWithAmr>);
static_assert(!KadathApps::bns_workflow_detail::StatefulAmrRefineHook<
              BnsWorkflowPolicyWithAmr>);
static_assert(KadathApps::bns_workflow_detail::StatefulAmrRefineHook<
              BnsWorkflowPolicyWithStatefulAmr>);

struct TempKadathHome {
    std::filesystem::path root;
    std::optional<std::string> old_home;

    TempKadathHome()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() /
               ("kadath_bns_seed_resolution_" + std::to_string(stamp));
        if (const char* existing = std::getenv("HOME_CELEPHAIS"))
            old_home = existing;
        setenv("HOME_CELEPHAIS", root.string().c_str(), 1);
    }

    ~TempKadathHome()
    {
        if (old_home)
            setenv("HOME_CELEPHAIS", old_home->c_str(), 1);
        else
            unsetenv("HOME_CELEPHAIS");
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    TempKadathHome(const TempKadathHome&) = delete;
    TempKadathHome& operator=(const TempKadathHome&) = delete;
};

class ScopedEnvVar
{
  public:
    explicit ScopedEnvVar(const char* name)
        : name_(name)
    {
        if (const char* existing = std::getenv(name))
            old_value_ = existing;
    }

    ~ScopedEnvVar()
    {
        if (old_value_)
            setenv(name_.c_str(), old_value_->c_str(), 1);
        else
            unsetenv(name_.c_str());
    }

    void set(const char* value) { setenv(name_.c_str(), value, 1); }
    void unset() { unsetenv(name_.c_str()); }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

  private:
    std::string name_;
    std::optional<std::string> old_value_;
};

} // namespace

TEST_CASE("BNS workflow leaves AMR-disabled configs on the existing solve path",
          "[bns-workflow][amr]")
{
    auto config = make_bns_workflow_config();
    config.amr_setting(AMR_ENABLED) = false;

    BnsWorkflowPolicyWithoutAmr::reset();
    KadathApps::run_bns_solver_sequence<BnsWorkflowPolicyWithoutAmr>(
        config, 0, false, "./");

    CHECK(BnsWorkflowPolicyWithoutAmr::solve_count == 1);
}

TEST_CASE("BNS workflow starts AMR from a loaded final-resolution solution",
          "[bns-workflow][amr]")
{
    auto config = make_bns_workflow_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MAX_CYCLES) = 2;

    BnsWorkflowPolicyWithAmr::reset();
    KadathApps::run_bns_solver_sequence<BnsWorkflowPolicyWithAmr>(
        config, 0, false, "./");

    CHECK(BnsWorkflowPolicyWithAmr::solve_count == 2);
    CHECK(BnsWorkflowPolicyWithAmr::amr_cycles == std::vector<int>{0, 1});
    CHECK(BnsWorkflowPolicyWithAmr::amr_resolutions == std::vector<int>{9, 9});
}

TEST_CASE("BNS workflow preserves AMR policy state across cycles",
          "[bns-workflow][amr]")
{
    auto config = make_bns_workflow_config();
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MAX_CYCLES) = 2;

    BnsWorkflowPolicyWithStatefulAmr::reset();
    KadathApps::run_bns_solver_sequence<BnsWorkflowPolicyWithStatefulAmr>(
        config, 0, false, "./");

    CHECK(BnsWorkflowPolicyWithStatefulAmr::solve_count == 1);
    CHECK(BnsWorkflowPolicyWithStatefulAmr::amr_cycles == std::vector<int>{0, 1});
    CHECK(BnsWorkflowPolicyWithStatefulAmr::previous_radial_demands ==
          std::vector<double>{0.0, 100.0});
}

TEST_CASE("BNS workflow reloads fields when a solver resumes a different dataset",
          "[bns-workflow][reload]")
{
    auto config = make_bns_workflow_config();
    config.control(FIXED_GOMEGA) = true;
    config.amr_setting(AMR_ENABLED) = false;

    BnsWorkflowPolicyReloadOnce::reset();
    KadathApps::run_bns_solver_sequence<BnsWorkflowPolicyReloadOnce>(
        config, 0, false, "./");

    CHECK(BnsWorkflowPolicyReloadOnce::solve_calls == 2);
    CHECK(BnsWorkflowPolicyReloadOnce::warmup_flags == std::vector<int>{1, 0});
}

TEST_CASE("BNS workflow delays AMR until the base resolution ladder finishes",
          "[bns-workflow][amr]")
{
    auto config = make_bns_workflow_config();
    config.set(BIN_RES) = 13;
    config.seq_setting(INIT_RES) = 9;
    config.amr_setting(AMR_ENABLED) = true;
    config.amr_setting(AMR_MAX_CYCLES) = 1;

    BnsWorkflowPolicyWithAmr::reset();
    KadathApps::run_bns_solver_sequence<BnsWorkflowPolicyWithAmr>(
        config, 0, false, "./");

    CHECK(BnsWorkflowPolicyWithAmr::solve_count == 4);
    CHECK(BnsWorkflowPolicyWithAmr::amr_cycles == std::vector<int>{0});
    CHECK(BnsWorkflowPolicyWithAmr::amr_resolutions == std::vector<int>{13});
}

TEST_CASE("BNS cold setup keeps component seed resolutions independent from the binary ladder",
          "[bns-workflow][resolution]")
{
    auto config = make_bns_workflow_config();
    config.set(BIN_RES) = 13;
    config.set(BCO_RES, BCO1) = 9;
    config.set(BCO_RES, BCO2) = 11;
    config.seq_setting(INIT_RES) = 9;

    BnsWorkflowColdSetupPolicy::reset();
    KadathApps::run_bns_solver_sequence<BnsWorkflowColdSetupPolicy>(
        config, 0, true, "./");

    CHECK(BnsWorkflowColdSetupPolicy::setup_binary_res == 9);
    CHECK(BnsWorkflowColdSetupPolicy::setup_ns1_res == 9);
    CHECK(BnsWorkflowColdSetupPolicy::setup_ns2_res == 11);
    CHECK(static_cast<int>(config(BIN_RES)) == 13);
    CHECK(static_cast<int>(config(BCO_RES, BCO1)) == 9);
    CHECK(static_cast<int>(config(BCO_RES, BCO2)) == 11);
}

TEST_CASE("BNS workflow rejects enabled AMR when the policy has no AMR hook",
          "[bns-workflow][amr]")
{
    auto config = make_bns_workflow_config();
    config.amr_setting(AMR_ENABLED) = true;

    BnsWorkflowPolicyWithoutAmr::reset();
    REQUIRE_THROWS_WITH(
        KadathApps::run_bns_solver_sequence<BnsWorkflowPolicyWithoutAmr>(
            config, 0, false, "./"),
        ContainsSubstring("requires BNS Policy::amr_refine"));
    CHECK(BnsWorkflowPolicyWithoutAmr::solve_count == 0);
}
