#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Array/array.hpp"

using namespace Kadath;

TEST_CASE("Array<int> inline and fallback payloads preserve the binary format",
          "[array_binary][inline-storage]") {
    for (const int count : {0, 1, 2, 3, 4}) {
        Array<int> original(count);
        for (int i = 0; i < count; ++i)
            original.set(i) = 100 + i;

        MemorySink original_sink;
        original.save(original_sink);
        MemorySource source(original_sink.buffer());
        Array<int> restored(source);

        REQUIRE(restored.get_nbr() == static_cast<std::size_t>(count));
        if (count == 0)
            REQUIRE(restored.get_data() == nullptr);
        else
            REQUIRE(restored.get_data() != nullptr);
        for (int i = 0; i < count; ++i)
            REQUIRE(restored(i) == 100 + i);

        MemorySink restored_sink;
        restored.save(restored_sink);
        REQUIRE(restored_sink.buffer() == original_sink.buffer());
    }
}

TEST_CASE("Array<int> 2x3 round-trip via MemorySink", "[array_binary]") {
    Dim_array dims(2);
    dims.set(0) = 2;
    dims.set(1) = 3;
    Array<int> original(dims);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            original.set(i, j) = 10 + i * 3 + j;

    MemorySink sink;
    original.save(sink);

    MemorySource source(sink.buffer());
    Array<int> restored(source);
    REQUIRE(restored.get_dimensions().get_ndim() == 2);
    REQUIRE(restored.get_dimensions()(0) == 2);
    REQUIRE(restored.get_dimensions()(1) == 3);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(restored(i, j) == 10 + i * 3 + j);
}

TEST_CASE("Array<double> 4-element round-trip via MemorySink",
          "[array_binary]") {
    Dim_array dims(1);
    dims.set(0) = 4;
    Array<double> original(dims);
    original.set(0) = 1.5;
    original.set(1) = -2.25;
    original.set(2) = 3.75;
    original.set(3) = 0.0;

    MemorySink sink;
    original.save(sink);

    MemorySource source(sink.buffer());
    Array<double> restored(source);
    REQUIRE(restored.get_dimensions().get_ndim() == 1);
    REQUIRE(restored.get_dimensions()(0) == 4);
    REQUIRE(restored(0) == 1.5);
    REQUIRE(restored(1) == -2.25);
    REQUIRE(restored(2) == 3.75);
    REQUIRE(restored(3) == 0.0);
}
