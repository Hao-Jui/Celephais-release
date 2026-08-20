#include <catch2/catch_test_macros.hpp>
#include "For_Kadath/Array/dim_array.hpp"

#include <vector>

using namespace Kadath;

TEST_CASE("Dim_array construction", "[dim_array]") {
    SECTION("rank zero") {
        Dim_array d(0);
        REQUIRE(d.get_ndim() == 0);
    }

    SECTION("inline rank") {
        Dim_array d(3);
        d.set(0) = 5; d.set(1) = 7; d.set(2) = 9;
        REQUIRE(d(0) == 5);
        REQUIRE(d(1) == 7);
        REQUIRE(d(2) == 9);
        REQUIRE(d.get_ndim() == 3);
    }

    SECTION("heap fallback ranks") {
        Dim_array rank4(4);
        Dim_array rank5(5);
        for (int i = 0; i < rank4.get_ndim(); ++i)
            rank4.set(i) = i + 1;
        for (int i = 0; i < rank5.get_ndim(); ++i)
            rank5.set(i) = i + 6;

        REQUIRE(rank4(3) == 4);
        REQUIRE(rank5(4) == 10);
    }
}

TEST_CASE("Dim_array copy and assignment preserve value semantics", "[dim_array]") {
    SECTION("inline storage") {
        Dim_array source(3);
        source.set(0) = 3; source.set(1) = 4; source.set(2) = 5;
        Dim_array copy(source);
        copy.set(0) = 30;

        REQUIRE(source(0) == 3);
        REQUIRE(copy(0) == 30);

        Dim_array assigned(3);
        assigned = source;
        const Dim_array& same = assigned;
        assigned = same;
        REQUIRE(assigned == source);
    }

    SECTION("heap fallback") {
        Dim_array source(5);
        for (int i = 0; i < source.get_ndim(); ++i)
            source.set(i) = 10 + i;
        Dim_array copy(source);
        copy.set(4) = 99;

        REQUIRE(source(4) == 14);
        REQUIRE(copy(4) == 99);

        Dim_array assigned(5);
        assigned = source;
        const Dim_array& same = assigned;
        assigned = same;
        REQUIRE(assigned == source);
    }
}

TEST_CASE("Dim_array swap handles all storage modes", "[dim_array][swap]") {
    SECTION("self") {
        Dim_array value(3);
        value.set(0) = 2; value.set(1) = 3; value.set(2) = 5;
        value.swap(value);
        REQUIRE(value(0) == 2);
        REQUIRE(value(1) == 3);
        REQUIRE(value(2) == 5);
    }

    SECTION("inline with inline") {
        Dim_array left(2);
        left.set(0) = 2; left.set(1) = 3;
        Dim_array right(3);
        right.set(0) = 5; right.set(1) = 7; right.set(2) = 11;

        left.swap(right);

        REQUIRE(left.get_ndim() == 3);
        REQUIRE(left(0) == 5);
        REQUIRE(left(2) == 11);
        REQUIRE(right.get_ndim() == 2);
        REQUIRE(right(0) == 2);
        REQUIRE(right(1) == 3);
    }

    SECTION("rank zero with rank one") {
        Dim_array empty(0);
        Dim_array scalar(1);
        scalar.set(0) = 17;

        empty.swap(scalar);

        REQUIRE(empty.get_ndim() == 1);
        REQUIRE(empty(0) == 17);
        REQUIRE(scalar.get_ndim() == 0);
    }

    SECTION("heap with heap") {
        Dim_array left(4);
        Dim_array right(5);
        for (int i = 0; i < left.get_ndim(); ++i)
            left.set(i) = 10 + i;
        for (int i = 0; i < right.get_ndim(); ++i)
            right.set(i) = 20 + i;

        left.swap(right);

        REQUIRE(left.get_ndim() == 5);
        REQUIRE(left(4) == 24);
        REQUIRE(right.get_ndim() == 4);
        REQUIRE(right(3) == 13);
    }

    SECTION("inline with heap in both directions") {
        Dim_array inline_value(3);
        inline_value.set(0) = 1;
        inline_value.set(1) = 2;
        inline_value.set(2) = 3;
        Dim_array heap_value(4);
        for (int i = 0; i < heap_value.get_ndim(); ++i)
            heap_value.set(i) = 10 + i;

        inline_value.swap(heap_value);
        REQUIRE(inline_value.get_ndim() == 4);
        REQUIRE(inline_value(3) == 13);
        REQUIRE(heap_value.get_ndim() == 3);
        REQUIRE(heap_value(2) == 3);

        inline_value.swap(heap_value);
        REQUIRE(inline_value.get_ndim() == 3);
        REQUIRE(inline_value(2) == 3);
        REQUIRE(heap_value.get_ndim() == 4);
        REQUIRE(heap_value(3) == 13);
    }
}

TEST_CASE("Dim_array copies survive vector relocation", "[dim_array]") {
    std::vector<Dim_array> dimensions;
    dimensions.reserve(1);
    dimensions.emplace_back(3);
    dimensions[0].set(0) = 2;
    dimensions[0].set(1) = 3;
    dimensions[0].set(2) = 5;
    dimensions.emplace_back(4);
    dimensions[1].set(0) = 7;
    dimensions[1].set(1) = 11;
    dimensions[1].set(2) = 13;
    dimensions[1].set(3) = 17;
    dimensions.emplace_back(2);

    REQUIRE(dimensions[0].get_ndim() == 3);
    REQUIRE(dimensions[0](0) == 2);
    REQUIRE(dimensions[0](2) == 5);
    REQUIRE(dimensions[1].get_ndim() == 4);
    REQUIRE(dimensions[1](3) == 17);
}
