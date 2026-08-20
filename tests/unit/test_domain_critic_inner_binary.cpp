#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Domain/critic.hpp"

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
} // namespace

TEST_CASE("Domain_critic_inner round-trip via MemorySink",
          "[domain_critic_inner_binary]") {
    Dim_array nbr(2);
    nbr.set(0) = 9;
    nbr.set(1) = 5;
    Domain_critic_inner original(0, CHEB_TYPE, nbr, /*xl=*/1.0);
    MemorySink sink1;
    original.save(sink1);
    REQUIRE(!sink1.buffer().empty());

    MemorySource source(sink1.buffer());
    Domain_critic_inner restored(0, source);
    REQUIRE(restored.get_nbr_points()(0) == 9);

    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}
