#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/be_file_source.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>

using namespace Kadath;

namespace {
std::string temp_path() {
    namespace fs = std::filesystem;
    fs::path p = fs::temp_directory_path()
               / ("kadath_be_file_sink_test_" + std::to_string(::getpid()) + ".bin");
    return p.string();
}
} // namespace

TEST_CASE("BeFileSink writes big-endian bytes for 4-byte int",
          "[be_file_sink]") {
    const std::string path = temp_path();
    {
        BeFileSink sink(path);
        const int value = 0x01020304;
        sink.write<int>(value);
    }
    // Read raw bytes; expect big-endian byte order.
    FILE* f = std::fopen(path.c_str(), "rb");
    REQUIRE(f != nullptr);
    unsigned char bytes[4] = {0, 0, 0, 0};
    REQUIRE(std::fread(bytes, 1, 4, f) == 4);
    std::fclose(f);
    std::remove(path.c_str());
    REQUIRE(bytes[0] == 0x01);
    REQUIRE(bytes[1] == 0x02);
    REQUIRE(bytes[2] == 0x03);
    REQUIRE(bytes[3] == 0x04);
}

TEST_CASE("BeFileSink + BeFileSource round-trip int + double",
          "[be_file_sink]") {
    const std::string path = temp_path();
    {
        BeFileSink sink(path);
        sink.write<int>(42);
        sink.write<double>(3.14);
    }
    {
        BeFileSource source(path);
        REQUIRE(source.read<int>() == 42);
        REQUIRE(source.read<double>() == 3.14);
    }
    std::remove(path.c_str());
}
