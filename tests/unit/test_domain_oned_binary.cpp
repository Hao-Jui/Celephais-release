#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Domain/oned.hpp"

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

Dim_array nbr1d() {
    Dim_array nbr(1);
    nbr.set(0) = 9;
    return nbr;
}
} // namespace

TEST_CASE("Domain_oned_ori round-trip via MemorySink", "[domain_oned_binary]") {
    Dim_array nbr = nbr1d();
    Domain_oned_ori original(0, CHEB_TYPE, /*radius=*/1.0, nbr);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Domain_oned_ori restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Domain_oned_qcq round-trip via MemorySink", "[domain_oned_binary]") {
    Dim_array nbr = nbr1d();
    Domain_oned_qcq original(0, CHEB_TYPE, /*x_int=*/1.0, /*x_ext=*/2.0, nbr);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Domain_oned_qcq restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Domain_oned_inf round-trip via MemorySink", "[domain_oned_binary]") {
    Dim_array nbr = nbr1d();
    Domain_oned_inf original(0, CHEB_TYPE, /*radius=*/1.0, nbr);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Domain_oned_inf restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}
