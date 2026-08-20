#include <catch2/catch_test_macros.hpp>

#include "Apps/Startup/app_startup.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace
{
    struct StubConfig
    {
        std::string filename;
        bool opened = false;

        void set_filename(const std::string& name) { filename = name; }
        void open_config() { opened = true; }
        std::string space_filename() const
        {
            std::filesystem::path data_path{filename};
            data_path.replace_extension(".dat");
            return data_path.string();
        }
    };

    struct TmpFile
    {
        std::filesystem::path path;
        TmpFile(const std::filesystem::path& p, const std::string& contents = "") : path(p)
        {
            std::ofstream ofs(path);
            ofs << contents;
        }
        ~TmpFile() { std::error_code ec; std::filesystem::remove(path, ec); }
        TmpFile(const TmpFile&) = delete;
        TmpFile& operator=(const TmpFile&) = delete;
    };

    std::filesystem::path unique_tmp(const std::string& stem, const std::string& ext)
    {
        static std::mt19937_64 rng{std::random_device{}()};
        const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto rnd = rng();
        return std::filesystem::temp_directory_path() /
               (stem + "_" + std::to_string(ts) + "_" + std::to_string(rnd) + ext);
    }

    char* cstr(std::string& s) { return s.data(); }
}

TEST_CASE("reader input path normalization accepts solution sidecars", "[solver_startup]") {
    REQUIRE(KadathApps::toml_config_path_from_reader_input("solution.toml") == "solution.toml");
    REQUIRE(KadathApps::toml_config_path_from_reader_input("solution.dat") == "solution.toml");
    REQUIRE(KadathApps::toml_config_path_from_reader_input("solution.info") == "solution.info");
    REQUIRE(KadathApps::toml_config_path_from_reader_input("converged_NS_UNIROT.sly4.1.6.0.1.0.09.dat") ==
            "converged_NS_UNIROT.sly4.1.6.0.1.0.09.toml");
}

TEST_CASE("TOML startup with config and data continues prior solution", "[solver_startup]") {
    auto toml_path = unique_tmp("kadath_startup", ".toml");
    auto dat_path = toml_path; dat_path.replace_extension(".dat");
    TmpFile toml(toml_path, "stub");
    TmpFile dat(dat_path, "stub");

    std::string prog = "solve";
    std::string a1 = toml_path.string();
    std::vector<char*> argv{cstr(prog), cstr(a1)};
    auto result = KadathApps::parse_kadath_config_toml_startup<StubConfig>(2, argv.data(), 1);
    REQUIRE(result.example_setup == false);
    REQUIRE(result.setup_first == false);
    REQUIRE(result.outputdir == "./");
    REQUIRE(result.bconfig.filename == a1);
    REQUIRE(result.bconfig.opened);
}

TEST_CASE("TOML startup with config only creates setup first", "[solver_startup]") {
    auto toml_path = unique_tmp("kadath_startup", ".toml");
    TmpFile toml(toml_path, "stub");

    std::string prog = "solve";
    std::string a1 = toml_path.string();
    std::vector<char*> argv{cstr(prog), cstr(a1)};
    auto result = KadathApps::parse_kadath_config_toml_startup<StubConfig>(2, argv.data(), 1);
    REQUIRE(result.example_setup == false);
    REQUIRE(result.setup_first == true);
    REQUIRE(result.outputdir == "./");
    REQUIRE(result.bconfig.filename == a1);
    REQUIRE(result.bconfig.opened);
}

TEST_CASE("TOML startup argc==3 sets outputdir from argv[2]", "[solver_startup]") {
    auto toml_path = unique_tmp("kadath_startup", ".toml");
    auto dat_path = toml_path; dat_path.replace_extension(".dat");
    TmpFile toml(toml_path, "stub");
    TmpFile dat(dat_path, "stub");

    std::string prog = "solve";
    std::string a1 = toml_path.string();
    std::string a2 = "/tmp/custom_outdir/";
    std::vector<char*> argv{cstr(prog), cstr(a1), cstr(a2)};
    auto result = KadathApps::parse_kadath_config_toml_startup<StubConfig>(3, argv.data(), 1);
    REQUIRE(result.outputdir == "/tmp/custom_outdir/");
    REQUIRE(result.example_setup == false);
}
