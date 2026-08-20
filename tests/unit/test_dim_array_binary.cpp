#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Array/dim_array.hpp"

#include <span>
#include <type_traits>

using namespace Kadath;

TEST_CASE("Dim_array round-trip via MemorySink/Source", "[dim_array_binary]") {
    auto check_round_trip = [](int rank) {
        Dim_array original(rank);
        for (int i = 0; i < rank; ++i)
            original.set(i) = 2 * i + 3;

        MemorySink sink1;
        original.save(sink1);

        MemorySource source(sink1.buffer());
        Dim_array restored(source);
        REQUIRE(restored.get_ndim() == rank);
        for (int i = 0; i < rank; ++i)
            REQUIRE(restored(i) == original(i));

        MemorySink sink2;
        restored.save(sink2);
        REQUIRE(sink1.buffer() == sink2.buffer());
    };

    check_round_trip(3);
    check_round_trip(4);
}

TEST_CASE("MemorySink exposes a non-owning byte span", "[dim_array_binary]")
{
    Dim_array original(2);
    original.set(0) = 7;
    original.set(1) = 3;

    MemorySink sink;
    original.save(sink);

    const auto bytes = sink.bytes();
    static_assert(std::is_same_v<decltype(bytes), const std::span<const unsigned char>>);
    REQUIRE(bytes.size() == sink.buffer().size());
    REQUIRE(bytes.data() == sink.buffer().data());
}
