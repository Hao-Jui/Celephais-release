#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Array/point.hpp"

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

Point make_point(int rank) {
    Point p(rank);
    for (int i = 1; i <= rank; ++i)
        p.set(i) = 1.25 * i - 2.0;
    return p;
}
} // namespace

TEST_CASE("Point round-trip via MemorySink", "[point_binary]") {
    auto check_round_trip = [](int rank) {
        Point original = make_point(rank);
        MemorySink sink1;
        original.save(sink1);

        MemorySource source(sink1.buffer());
        Point restored(source);

        REQUIRE(restored.get_ndim() == original.get_ndim());
        for (int i = 1; i <= original.get_ndim(); ++i) {
            REQUIRE(restored(i) == original(i));
        }

        MemorySink sink2;
        restored.save(sink2);
        REQUIRE(sink1.buffer() == sink2.buffer());
    };

    check_round_trip(3);
    check_round_trip(4);
}
