#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/IO/be_file_sink.hpp"
#include "For_Kadath/IO/memory_sink.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Kadath;

namespace {
class BinarySpectralBase : public Base_spectral {
  public:
    explicit BinarySpectralBase(int dimensions) : Base_spectral(dimensions) {}

    void define_slots(int first_basis) {
        for (int axis = 0; axis < ndim; ++axis) {
            auto value = std::make_unique<Array<int>>(1);
            value->set(0) = first_basis + axis;
            bases_1d[static_cast<std::size_t>(axis)] = std::move(value);
        }
        def = true;
    }
};

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<unsigned char>{
        std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
} // namespace

TEST_CASE("Base_spectral non-defined round-trip via MemorySink",
          "[base_spectral_binary]") {
    for (int dimensions : {1, 2, 3, 4, 5}) {
        Base_spectral original(dimensions);
        MemorySink sink1;
        original.save(sink1);
        MemorySource source(sink1.buffer());
        Base_spectral restored(source);
        REQUIRE_FALSE(restored.is_def());

        // Byte identity proves both the indicator and arbitrary dimension are preserved.
        MemorySink sink2;
        restored.save(sink2);
        REQUIRE(sink1.buffer() == sink2.buffer());
    }
}

TEST_CASE("Base_spectral defined bounded and fallback round trips preserve values",
          "[base_spectral_binary]") {
    for (int dimensions : {1, 2, 3, 4, 5}) {
        BinarySpectralBase original(dimensions);
        original.define_slots(CHEB);

        MemorySink sink1;
        original.save(sink1);
        MemorySource source(sink1.buffer());
        Base_spectral restored(source);
        REQUIRE(restored.is_def());
        REQUIRE(original == restored);

        MemorySink sink2;
        restored.save(sink2);
        REQUIRE(sink1.buffer() == sink2.buffer());
    }
}

TEST_CASE("Base_spectral binary constructor refuses negative dimensions",
          "[base_spectral_binary]") {
    MemorySink sink;
    sink.write<int>(1);
    sink.write<int>(-1);
    MemorySource source(sink.buffer());

    REQUIRE_THROWS_AS(Base_spectral(source), std::length_error);
}
