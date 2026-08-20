#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Domain/polar.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace Kadath;

namespace {
std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<unsigned char>{
        std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

Dim_array nbr2d() {
    Dim_array nbr(2);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    return nbr;
}

Point center2d() {
    Point p(2);
    p.set(1) = 0.5;
    p.set(2) = -0.25;
    return p;
}
} // namespace

TEST_CASE("Domain_polar_nucleus round-trip via MemorySink", "[domain_polar_binary]") {
    Dim_array nbr = nbr2d();
    Point c = center2d();
    Domain_polar_nucleus original(0, CHEB_TYPE, /*radius=*/1.0, c, nbr);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Domain_polar_nucleus restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Domain_polar_shell round-trip via MemorySink", "[domain_polar_binary]") {
    Dim_array nbr = nbr2d();
    Point c = center2d();
    Domain_polar_shell original(0, CHEB_TYPE, /*r_int=*/1.0, /*r_ext=*/2.0, c, nbr);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Domain_polar_shell restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Domain_polar_compact round-trip via MemorySink", "[domain_polar_binary]") {
    Dim_array nbr = nbr2d();
    Point c = center2d();
    Domain_polar_compact original(0, CHEB_TYPE, /*r_int=*/1.0, c, nbr);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Domain_polar_compact restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}
