#include <catch2/catch_test_macros.hpp>
#include "For_Kadath/Array/index.hpp"

#include <vector>

using namespace Kadath;

TEST_CASE("Index iteration order", "[index]") {
    Dim_array dims(2);
    dims.set(0) = 3; dims.set(1) = 2;
    Index idx(dims);

    REQUIRE(idx(0) == 0);
    REQUIRE(idx(1) == 0);

    idx.inc(); idx.inc(); idx.inc();
    REQUIRE(idx(0) == 0);
    REQUIRE(idx(1) == 1);
}

TEST_CASE("Index full traversal count", "[index]") {
    Dim_array dims(2);
    dims.set(0) = 4; dims.set(1) = 3;
    Index idx(dims);
    int count = 0;
    do { count++; } while (idx.inc());
    REQUIRE(count == 12);
}

TEST_CASE("Index 3D traversal", "[index]") {
    Dim_array dims(3);
    dims.set(0) = 2; dims.set(1) = 3; dims.set(2) = 2;
    Index idx(dims);
    int count = 0;
    do { count++; } while (idx.inc());
    REQUIRE(count == 12);
}

TEST_CASE("Index heap fallback traverses 4D and 5D extents", "[index]") {
    SECTION("rank 4") {
        Dim_array dims(4);
        for (int i = 0; i < dims.get_ndim(); ++i)
            dims.set(i) = 2;
        Index idx(dims);
        int count = 0;
        do { ++count; } while (idx.inc());
        REQUIRE(count == 16);
    }

    SECTION("rank 5") {
        Dim_array dims(5);
        for (int i = 0; i < dims.get_ndim(); ++i)
            dims.set(i) = 2;
        Index idx(dims);
        int count = 0;
        do { ++count; } while (idx.inc());
        REQUIRE(count == 32);
    }
}

TEST_CASE("Index copy and assignment preserve independent coordinates", "[index]") {
    for (const int rank : {3, 4}) {
        Dim_array dims(rank);
        for (int i = 0; i < dims.get_ndim(); ++i)
            dims.set(i) = 6;

        Index source(dims);
        for (int i = 0; i < rank; ++i)
            source.set(i) = i + 1;

        Index copy(source);
        copy.set(rank - 1) = 5;
        REQUIRE(source(rank - 1) == rank);
        REQUIRE(copy(rank - 1) == 5);

        Index assigned(dims);
        assigned = source;
        const Index& same = assigned;
        assigned = same;
        REQUIRE(assigned == source);
    }
}

TEST_CASE("Index copies survive vector relocation", "[index]") {
    Dim_array rank3(3);
    Dim_array rank5(5);
    for (int i = 0; i < rank3.get_ndim(); ++i)
        rank3.set(i) = 4;
    for (int i = 0; i < rank5.get_ndim(); ++i)
        rank5.set(i) = 4;

    std::vector<Index> indices;
    indices.reserve(1);
    indices.emplace_back(rank3);
    indices[0].set(0) = 1;
    indices[0].set(1) = 2;
    indices[0].set(2) = 3;
    indices.emplace_back(rank5);
    indices[1].set(0) = 3;
    indices[1].set(4) = 2;
    indices.emplace_back(rank3);

    REQUIRE(indices[0].get_ndim() == 3);
    REQUIRE(indices[0](0) == 1);
    REQUIRE(indices[0](2) == 3);
    REQUIRE(indices[1].get_ndim() == 5);
    REQUIRE(indices[1](0) == 3);
    REQUIRE(indices[1](4) == 2);
}

#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
TEST_CASE("Index coordinate access checks dimension bounds", "[index][bounds]") {
    Dim_array dims(2);
    dims.set(0) = 3; dims.set(1) = 2;
    Index idx(dims);

    REQUIRE_THROWS_AS(idx.set(-1), std::out_of_range);
    REQUIRE_THROWS_AS(idx.set(2), std::out_of_range);

    const Index& const_idx = idx;
    REQUIRE_THROWS_AS(const_idx(-1), std::out_of_range);
    REQUIRE_THROWS_AS(const_idx(2), std::out_of_range);
}
#endif
