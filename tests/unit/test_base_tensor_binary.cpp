#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Base_tensor/base_tensor.hpp"
#include "For_Kadath/Domain/spheric.hpp"
#include "For_Kadath/Array/point.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace Kadath;

namespace {
Space_spheric make_space() {
    Point center(3);
    center.set(1) = 0;
    center.set(2) = 0;
    center.set(3) = 0;
    Dim_array res(3);
    res.set(0) = 5;
    res.set(1) = 5;
    res.set(2) = 4;
    Dim_array bounds_dim(1);
    bounds_dim.set(0) = 2;
    Array<double> bounds(bounds_dim);
    bounds.set(0) = 1.0;
    bounds.set(1) = 10.0;
    return Space_spheric(CHEB_TYPE, center, res, bounds);
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<unsigned char>{
        std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
} // namespace

TEST_CASE("Base_tensor round-trip via MemorySink", "[base_tensor_binary]") {
    Space_spheric space = make_space();
    Base_tensor original(space, CARTESIAN_BASIS);

    MemorySink sink1;
    original.save(sink1);
    MemorySource source(sink1.buffer());
    Base_tensor restored(space, source);

    MemorySink sink2;
    restored.save(sink2);
    REQUIRE(sink1.buffer() == sink2.buffer());
}
