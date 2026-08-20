#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Domain/spheric_periodic.hpp"

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
} // namespace

TEST_CASE("Domain_spheric_periodic_nucleus round-trip via MemorySink",
          "[domain_spheric_periodic_binary]") {
    Dim_array nbr = nbr2d();
    Domain_spheric_periodic_nucleus original(0, CHEB_TYPE, TO_PI, /*radius=*/1.0,
                                             /*ome=*/0.5, nbr);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Domain_spheric_periodic_nucleus restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Domain_spheric_periodic_shell round-trip via MemorySink",
          "[domain_spheric_periodic_binary]") {
    Dim_array nbr = nbr2d();
    Domain_spheric_periodic_shell original(0, CHEB_TYPE, TO_PI_OVER_2, /*r_int=*/1.0,
                                           /*r_ext=*/2.0, /*ome=*/0.5, nbr);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Domain_spheric_periodic_shell restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}

TEST_CASE("Domain_spheric_periodic_compact round-trip via MemorySink",
          "[domain_spheric_periodic_binary]") {
    Dim_array nbr = nbr2d();
    Domain_spheric_periodic_compact original(0, CHEB_TYPE, TO_PI, /*r_int=*/1.0,
                                             /*ome=*/0.5, nbr);
    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Domain_spheric_periodic_compact restored(0, source);
    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}
