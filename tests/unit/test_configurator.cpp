#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "Apps/Startup/app_startup.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_three_body.hpp"
#include "For_Kadath/Config/configurator_toml.hpp"
#include "For_Kadath/Config/config_bco.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>

using namespace Kadath;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;

TEST_CASE("BCO_BH config set/get round-trip", "[configurator]") {
    kadath_config<BCO_BH_INFO> config;
    config.set(RIN) = 0.25;
    config.set(RMID) = 0.75;
    config.set(BCO_RES) = 11;
    REQUIRE_THAT(config(RIN), WithinAbs(0.25, 1e-14));
    REQUIRE_THAT(config(RMID), WithinAbs(0.75, 1e-14));
    REQUIRE_THAT(config(BCO_RES), WithinAbs(11.0, 1e-14));
}

TEST_CASE("BCO_BH defaults populate all params", "[configurator]") {
    kadath_config<BCO_BH_INFO> config;
    config.set_defaults();
    REQUIRE(config(RIN) > 0.0);
    REQUIRE(config(RMID) > config(RIN));
    REQUIRE_THAT(config(MIRR), WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(config(MCH), WithinAbs(1.0, 1e-14));
}

TEST_CASE("BCO_BH control toggle round-trip", "[configurator]") {
    kadath_config<BCO_BH_INFO> config;
    REQUIRE_FALSE(config.control(SCALAR_CONTROL));
    config.control(CHECKPOINT) = true;
    REQUIRE(config.control(CHECKPOINT) == true);
    config.control(CHECKPOINT) = false;
    REQUIRE(config.control(CHECKPOINT) == false);
}

TEST_CASE("BCO_BH stage toggle round-trip", "[configurator]") {
    kadath_config<BCO_BH_INFO> config;
    config.set_stage(VON_NEUMANN) = true;
    REQUIRE(config.set_stage(VON_NEUMANN) == true);
}

TEST_CASE("Configurator appends TOML suffix to dotted Kadath solution basenames", "[configurator]") {
    kadath_config<BCO_BH_INFO> config;

    config.set_filename("converged_NS_UNIROT.sly4.1.6.0.1.0.09");

    REQUIRE(config.config_filename() == "converged_NS_UNIROT.sly4.1.6.0.1.0.09.toml");
    REQUIRE(config.space_filename() == "./converged_NS_UNIROT.sly4.1.6.0.1.0.09.dat");
}

TEST_CASE("Configurator rejects explicit non-TOML sidecar filenames", "[configurator]") {
    kadath_config<BCO_BH_INFO> config;

    REQUIRE_THROWS_WITH(config.set_filename("converged_NS_UNIROT.sly4.1.6.0.1.0.09.info"),
        ContainsSubstring("Configurator accepts only .toml files"));
}

TEST_CASE("BCO_BH config parses and writes TOML", "[configurator]") {
    const auto config_path = std::filesystem::temp_directory_path() / "kadath_config_bh_input.toml";
    const auto output_path = std::filesystem::temp_directory_path() / "kadath_config_bh_output.toml";

    {
        std::ofstream config_file{config_path};
        config_file << R"(
[bh]
dim = 2
fixed_lapse = 0.2
mch = 1.2
mirr = 1.2
nshells = 1
qpig = 12.566370614359172
res = 11
rin = 0.4
rmid = 0.8
rout = 1.6
trumpet_bh_seed = true

[fields]
conf = true
lapse = true
shift = true

[sequence_controls]
initial_regrid = true
scalar_control = true

[sequence_settings]
initial_resolution = 11
solver_max_iterations = 15
solver_precision = 1e-8

[adaptive_mesh_refinement]
enabled = true
max_cycles = 2
tail_width = 2
l2_tail_threshold = 1e-8
linf_tail_threshold = 1e-7
mode = "p"
max_shells = 1
h_gain_min = 3.0
h_stall_cycles = 2
h_saturation_linf = 0.9
h_angular_dominance = 1.25
p_dof_growth_limit = 1.5
p_fallback_dof_growth_limit = 1.2
p_fallback_max_candidates = 3

[stages]
von_neumann = true
)";
    }

    kadath_config<BCO_BH_INFO> config{config_path.string()};
    REQUIRE_THAT(config(RIN), WithinAbs(0.4, 1e-14));
    REQUIRE_THAT(config(RMID), WithinAbs(0.8, 1e-14));
    REQUIRE(config.field(CONF));
    REQUIRE(config.set_stage(VON_NEUMANN));
    REQUIRE(config.control(REGRID));
    REQUIRE(config.control(SCALAR_CONTROL));
    REQUIRE(config.template value_as<int>(TRUMPET_BH_SEED) == 1);
    REQUIRE_THAT(config.template seq_setting_as<double>(PREC), WithinAbs(1e-8, 1e-16));
    REQUIRE(config.template amr_setting_as<bool>(AMR_ENABLED));
    REQUIRE(config.template amr_setting_as<int>(AMR_MAX_CYCLES) == 2);
    REQUIRE(config.template amr_setting_as<int>(AMR_TAIL_WIDTH) == 2);
    REQUIRE_THAT(config.template amr_setting_as<double>(AMR_L2_TAIL_THRESHOLD), WithinAbs(1e-8, 1e-16));
    REQUIRE_THAT(config.template amr_setting_as<double>(AMR_LINF_TAIL_THRESHOLD), WithinAbs(1e-7, 1e-15));
    REQUIRE(config.template amr_setting_as<std::string>(AMR_MODE) == "p");
    REQUIRE(config.template amr_setting_as<int>(AMR_MAX_SHELLS) == 1);
    REQUIRE_THAT(config.template amr_setting_as<double>(AMR_NORM_FLOOR), WithinAbs(0.0, 1e-16));
    REQUIRE_THAT(config.template amr_setting_as<double>(AMR_H_GAIN_MIN), WithinAbs(3.0, 1e-14));
    REQUIRE(config.template amr_setting_as<int>(AMR_H_STALL_CYCLES) == 2);
    REQUIRE_THAT(config.template amr_setting_as<double>(AMR_H_SATURATION_LINF), WithinAbs(0.9, 1e-14));
    REQUIRE_THAT(config.template amr_setting_as<double>(AMR_H_ANGULAR_DOMINANCE), WithinAbs(1.25, 1e-14));
    REQUIRE_THAT(config.template amr_setting_as<double>(AMR_P_DOF_GROWTH_LIMIT), WithinAbs(1.5, 1e-14));
    REQUIRE_THAT(config.template amr_setting_as<double>(AMR_P_FALLBACK_DOF_GROWTH_LIMIT), WithinAbs(1.2, 1e-14));
    REQUIRE(config.template amr_setting_as<int>(AMR_P_FALLBACK_MAX_CANDIDATES) == 3);

    config.write_config(output_path.string());
    std::ifstream written_config{output_path};
    const std::string written_text{
        std::istreambuf_iterator<char>{written_config},
        std::istreambuf_iterator<char>{}};
    REQUIRE_THAT(written_text, ContainsSubstring("trumpet_bh_seed = true"));
    REQUIRE_THAT(written_text, ContainsSubstring("von_neumann = true"));
    REQUIRE_THAT(written_text, ContainsSubstring("[adaptive_mesh_refinement]"));
    REQUIRE_THAT(written_text, ContainsSubstring("enabled = true"));
    REQUIRE_THAT(written_text, ContainsSubstring("mode = \"p\""));
    REQUIRE_THAT(written_text, ContainsSubstring("max_shells = 1"));
    REQUIRE_THAT(written_text, ContainsSubstring("h_gain_min = 3"));
    REQUIRE_THAT(written_text, ContainsSubstring("p_dof_growth_limit = 1.5"));
    REQUIRE_THAT(written_text, ContainsSubstring("p_fallback_dof_growth_limit = 1.2"));
    REQUIRE_THAT(written_text, ContainsSubstring("p_fallback_max_candidates = 3"));
    REQUIRE_THAT(written_text, !ContainsSubstring("total_bc"));

    kadath_config<BCO_BH_INFO> reread{output_path.string()};
    REQUIRE_THAT(reread(RIN), WithinAbs(0.4, 1e-14));
    REQUIRE(reread.field(CONF));
    REQUIRE(reread.set_stage(VON_NEUMANN));
    REQUIRE(reread.control(SCALAR_CONTROL));
    REQUIRE(reread.template value_as<int>(TRUMPET_BH_SEED) == 1);
    REQUIRE(reread.template amr_setting_as<bool>(AMR_ENABLED));
    REQUIRE(reread.template amr_setting_as<std::string>(AMR_MODE) == "p");
    REQUIRE_THAT(reread.template amr_setting_as<double>(AMR_H_GAIN_MIN), WithinAbs(3.0, 1e-14));
    REQUIRE_THAT(reread.template amr_setting_as<double>(AMR_P_DOF_GROWTH_LIMIT), WithinAbs(1.5, 1e-14));
    REQUIRE_THAT(reread.template amr_setting_as<double>(AMR_P_FALLBACK_DOF_GROWTH_LIMIT), WithinAbs(1.2, 1e-14));
    REQUIRE(reread.template amr_setting_as<int>(AMR_P_FALLBACK_MAX_CANDIDATES) == 3);
}

TEST_CASE("Configurator rejects legacy v1 stage keys", "[configurator]") {
    const auto config_path = std::filesystem::temp_directory_path() / "kadath_config_bh_legacy_stage.toml";

    {
        std::ofstream config_file{config_path};
        config_file << R"(
[bh]
res = 11
rin = 0.4
rmid = 0.8
rout = 1.6

[stages]
total_bc = true
)";
    }

    REQUIRE_THROWS_WITH((kadath_config<BCO_BH_INFO>{config_path.string()}),
        ContainsSubstring("Legacy v1 stage key"));
}

TEST_CASE("Runtime example writer uses kadath_config TOML serialization", "[configurator]") {
    const auto output_path = std::filesystem::temp_directory_path() / "kadath_config_runtime_example.toml";

    KadathApps::write_example_toml<kadath_config<BCO_NS_INFO>>(0, output_path.string(), [](auto& config) {
        config.set_defaults();
        config.set(DIM) = 2;
        config.set(BCO_RES) = 13;
        config.seq_setting(INIT_RES) = 13;
    });

    kadath_config<BCO_NS_INFO> reread{output_path.string()};
    REQUIRE_THAT(reread(BCO_RES), WithinAbs(13.0, 1e-14));
    REQUIRE_THAT(reread(DIM), WithinAbs(2.0, 1e-14));
    REQUIRE_THAT(reread.template seq_setting_as<double>(INIT_RES), WithinAbs(13.0, 1e-14));
    REQUIRE(reread.field(CONF));
    REQUIRE(reread.set_stage(UNIROT));
}

TEST_CASE("ConfigSlot writes Newton-updated integer literals as doubles", "[configurator]") {
    const auto output_path = std::filesystem::temp_directory_path() / "kadath_config_ns_updated_omega.toml";

    kadath_config<BCO_NS_INFO> config;
    config.set_defaults();

    config.set(OMEGA) = 0;
    double& omega_unknown = config(OMEGA);
    omega_unknown = 0.018385900096569072;

    REQUIRE_FALSE(config.set(OMEGA).holds_int());

    config.write_config(output_path.string());

    kadath_config<BCO_NS_INFO> reread{output_path.string()};
    REQUIRE_THAT(reread(OMEGA), WithinAbs(0.018385900096569072, 1e-16));

    std::ifstream written{output_path};
    const std::string contents{std::istreambuf_iterator<char>{written}, std::istreambuf_iterator<char>{}};
    REQUIRE_THAT(contents, ContainsSubstring("omega = 0.018385900096569"));
}

TEST_CASE("Binary runtime writer preserves compact-object branch identity", "[configurator]") {
    const auto output_path = std::filesystem::temp_directory_path() / "kadath_config_binary_example.toml";

    KadathApps::write_example_toml<kadath_config<BIN_INFO>>(0, output_path.string(), [](auto& config) {
        config.initialize_binary({"ns", "bh"});
        config.set_defaults();
    });

    kadath_config<BIN_INFO> reread{output_path.string()};
    REQUIRE_THAT(reread(MADM, BCO1), WithinAbs(1.4, 1e-14));
    REQUIRE_THAT(reread(MIRR, BCO2), WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(reread(MCH, BCO2), WithinAbs(1.0, 1e-14));
    REQUIRE(reread.eos<std::string>(EOSTYPE, BCO1) == "Cold_Table");
    REQUIRE(reread.field(PHI));
    REQUIRE_FALSE(reread.control(FIXED_GOMEGA));
}

TEST_CASE("Three-body runtime writer round-trips globals and per-body branches", "[configurator]") {
    const auto output_path = std::filesystem::temp_directory_path() / "kadath_config_three_body_example.toml";

    KadathApps::write_example_toml<kadath_config<TRI_INFO>>(0, output_path.string(), [](auto& config) {
        config.initialize_three_body({"ns", "ns", "bh"});
        config.set_defaults();
        config.set(PARENT_DIST) = 47.5;
        config.set(MADM, BCO2) = 1.35;
    });

    kadath_config<TRI_INFO> reread{output_path.string()};
    REQUIRE_THAT(reread(PARENT_DIST), WithinAbs(47.5, 1e-14));
    REQUIRE_THAT(reread(CHILD_DIST), WithinAbs(34.0, 1e-14));
    REQUIRE_THAT(reread(CHILD_REXT), WithinAbs(36.0, 1e-14));
    REQUIRE_THAT(reread(MADM, BCO1), WithinAbs(1.4, 1e-14));
    REQUIRE_THAT(reread(MADM, BCO2), WithinAbs(1.35, 1e-14));
    REQUIRE_THAT(reread(MIRR, BCO3), WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(reread(MCH, BCO3), WithinAbs(1.0, 1e-14));
    REQUIRE(reread.eos<std::string>(EOSTYPE, BCO1) == "Cold_Table");
    REQUIRE(reread.field(LOGH));
    REQUIRE_FALSE(reread.field(PHI)); // static three-body: no velocity potential

    // Branch identity: body slots must survive the write/read cycle by suffix.
    std::ifstream written{output_path};
    const std::string contents{std::istreambuf_iterator<char>{written}, std::istreambuf_iterator<char>{}};
    REQUIRE_THAT(contents, ContainsSubstring("[three_body.ns1]"));
    REQUIRE_THAT(contents, ContainsSubstring("[three_body.ns2]"));
    REQUIRE_THAT(contents, ContainsSubstring("[three_body.bh3]"));
}
