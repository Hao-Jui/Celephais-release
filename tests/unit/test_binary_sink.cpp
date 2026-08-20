#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/IO/file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"

#include <vector>

using namespace Kadath;

TEST_CASE("MemorySink + MemorySource round-trip int + double + array",
          "[binary_sink]") {
    MemorySink sink;
    sink.write<int>(42);
    sink.write<double>(3.14);
    const double arr[3] = {1.0, 2.0, 3.0};
    sink.write_array(arr, 3);

    MemorySource source(sink.buffer());
    REQUIRE(source.read<int>() == 42);
    REQUIRE(source.read<double>() == 3.14);
    double out[3] = {0.0, 0.0, 0.0};
    source.read_array(out, 3);
    REQUIRE(out[0] == 1.0);
    REQUIRE(out[1] == 2.0);
    REQUIRE(out[2] == 3.0);
}

TEST_CASE("MemorySource throws KadathError on read past end",
          "[binary_sink]") {
    std::vector<unsigned char> buf(2, 0);
    MemorySource source(std::move(buf));
    int dummy = 0;
    REQUIRE_THROWS_AS(source.read_bytes(&dummy, sizeof(int), 1), Kadath::KadathError);
}

TEST_CASE("FileSink throws KadathError on bad path", "[binary_sink]") {
    REQUIRE_THROWS_AS(FileSink("/no/such/dir/kadath_test.bin"), Kadath::KadathError);
}
