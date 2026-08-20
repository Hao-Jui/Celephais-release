#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/index.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

using namespace Kadath;
using Catch::Matchers::WithinAbs;

TEST_CASE("Array arithmetic", "[array]") {
    Dim_array dims(1); dims.set(0) = 4;
    Array<double> a(dims); Array<double> b(dims);
    a.set(0)=1; a.set(1)=2; a.set(2)=3; a.set(3)=4;
    b.set(0)=10; b.set(1)=20; b.set(2)=30; b.set(3)=40;
    Array<double> c = a + b;
    REQUIRE_THAT(c(0), WithinAbs(11.0, 1e-14));
    REQUIRE_THAT(c(3), WithinAbs(44.0, 1e-14));
}

TEST_CASE("Array compound arithmetic updates storage directly and handles aliasing", "[array][compound]") {
    Array<double> a(4);
    Array<double> b(4);
    a.set(0) = -2.0; a.set(1) = 4.0; a.set(2) = 0.0; a.set(3) = 8.0;
    b.set(0) = 3.0; b.set(1) = -2.0; b.set(2) = 5.0; b.set(3) = 0.5;

    const double* const storage = a.get_data();
    a += b;
    REQUIRE(a.get_data() == storage);
    REQUIRE(a(0) == 1.0);
    REQUIRE(a(1) == 2.0);
    REQUIRE(a(2) == 5.0);
    REQUIRE(a(3) == 8.5);

    a -= b;
    a *= b;
    a /= b;
    REQUIRE(a(0) == -2.0);
    REQUIRE(a(1) == 4.0);
    REQUIRE(a(2) == 0.0);
    REQUIRE(a(3) == 8.0);

    Array<double> self_add(a);
    self_add += self_add;
    REQUIRE(self_add(0) == -4.0);
    REQUIRE(self_add(3) == 16.0);

    Array<double> self_sub(a);
    self_sub -= self_sub;
    for (std::size_t i = 0; i < self_sub.get_nbr(); ++i)
        REQUIRE(self_sub.get_data()[i] == 0.0);

    Array<double> self_mul(a);
    self_mul *= self_mul;
    REQUIRE(self_mul(0) == 4.0);
    REQUIRE(self_mul(1) == 16.0);
    REQUIRE(self_mul(2) == 0.0);
    REQUIRE(self_mul(3) == 64.0);

    Array<double> self_div(a);
    self_div /= self_div;
    REQUIRE(self_div(0) == 1.0);
    REQUIRE(self_div(1) == 1.0);
    REQUIRE(std::isnan(self_div(2)));
    REQUIRE(self_div(3) == 1.0);
}

TEST_CASE("Array scalar compound arithmetic snapshots aliased scalars", "[array][compound]") {
    Array<double> values(3);
    values.set(0) = 1.0;
    values.set(1) = 2.0;
    values.set(2) = 3.0;

    const double& aliased_scalar = values.set(1);
    values += aliased_scalar;
    REQUIRE(values(0) == 3.0);
    REQUIRE(values(1) == 4.0);
    REQUIRE(values(2) == 5.0);

    values -= 1.0;
    values *= -2.0;
    values /= 2.0;
    REQUIRE(values(0) == -2.0);
    REQUIRE(values(1) == -3.0);
    REQUIRE(values(2) == -4.0);
}

TEST_CASE("Array compound arithmetic covers empty, scalar, and non-finite inputs", "[array][compound]") {
    Array<double> empty(0);
    Array<double> other_empty(0);
    empty += other_empty;
    empty -= 1.0;
    REQUIRE(empty.get_nbr() == 0);

    Array<double> scalar(1);
    scalar.set(0) = -3.0;
    scalar *= scalar;
    REQUIRE(scalar(0) == 9.0);

    Array<double> non_finite(2);
    non_finite.set(0) = std::numeric_limits<double>::infinity();
    non_finite.set(1) = std::numeric_limits<double>::quiet_NaN();
    non_finite += 1.0;
    REQUIRE(std::isinf(non_finite(0)));
    REQUIRE(std::isnan(non_finite(1)));

}

TEST_CASE("Array transcendentals", "[array]") {
    Dim_array dims(1); dims.set(0) = 3;
    Array<double> a(dims);
    a.set(0) = 0.0; a.set(1) = M_PI/4; a.set(2) = M_PI/2;
    Array<double> s = sin(a);
    REQUIRE_THAT(s(0), WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(s(2), WithinAbs(1.0, 1e-14));
}

TEST_CASE("Array sum reduction", "[array]") {
    Dim_array dims(1); dims.set(0) = 4;
    Array<double> a(dims);
    a.set(0)=1; a.set(1)=2; a.set(2)=3; a.set(3)=4;
    // scal(a,a) = dot product = 1+4+9+16 = 30
    double dot = scal(a, a);
    REQUIRE_THAT(dot, WithinAbs(30.0, 1e-14));
}

TEST_CASE("Array 2D indexing", "[array]") {
    Dim_array dims(2); dims.set(0) = 3; dims.set(1) = 2;
    Array<double> a(dims);
    // Fill with pattern: a(i,j) = i*10 + j
    Index idx(dims);
    do {
        a.set(idx) = idx(0)*10 + idx(1);
    } while(idx.inc());

    Index check(dims);
    check.set(0) = 2; check.set(1) = 1;
    REQUIRE_THAT(a(check), WithinAbs(21.0, 1e-14));
}

TEST_CASE("Array<int> capacity-two storage preserves value semantics",
          "[array][inline-storage]") {
    for (int count = 0; count <= 4; ++count) {
        Array<int> original(count);
        REQUIRE(original.get_nbr() == static_cast<std::size_t>(count));
        REQUIRE(original.data == original.get_data());
        if (count == 0)
            REQUIRE(original.get_data() == nullptr);
        else
            REQUIRE(original.get_data() != nullptr);

        for (int i = 0; i < count; ++i)
            original.set(i) = 10 + i;

        Array<int> copied(original);
        REQUIRE(copied.get_nbr() == original.get_nbr());
        if (count == 0) {
            REQUIRE(copied.get_data() == nullptr);
        } else {
            REQUIRE(copied.get_data() != original.get_data());
            for (int i = 0; i < count; ++i)
                REQUIRE(copied(i) == original(i));
            copied.set(0) = -1;
            REQUIRE(original(0) == 10);

            Array<int> assigned(count);
            int* const assigned_storage = assigned.get_data();
            assigned = original;
            REQUIRE(assigned.get_data() == assigned_storage);
            for (int i = 0; i < count; ++i)
                REQUIRE(assigned(i) == original(i));

        }
    }

    Array<int> zero_product(0, 4);
    REQUIRE(zero_product.get_nbr() == 0U);
    REQUIRE(zero_product.get_data() == nullptr);

    Dim_array shaped_dimensions(2);
    shaped_dimensions.set(0) = 1;
    shaped_dimensions.set(1) = 1;
    Array<int> shaped_scalar(shaped_dimensions);
    Index shaped_index(shaped_dimensions);
    shaped_scalar.set(shaped_index) = 17;
    REQUIRE(shaped_scalar.get_nbr() == 1U);
    REQUIRE(shaped_scalar.get_data() != nullptr);
    REQUIRE(shaped_scalar(shaped_index) == 17);

    Array<int> matrix(1, 2);
    matrix.set(0, 0) = 19;
    matrix.set(0, 1) = 23;
    REQUIRE(matrix.get_nbr() == 2U);
    REQUIRE(matrix.get_data() != nullptr);
    REQUIRE(matrix(0, 0) == 19);
    REQUIRE(matrix(0, 1) == 23);

    Array<int> cube(1, 1, 2);
    cube.set(0, 0, 0) = 29;
    cube.set(0, 0, 1) = 31;
    REQUIRE(cube.get_nbr() == 2U);
    REQUIRE(cube.get_data() != nullptr);
    REQUIRE(cube(0, 0, 0) == 29);
    REQUIRE(cube(0, 0, 1) == 31);

    Array<int> left(2);
    Array<int> right(2);
    left.set(0) = 2;
    left.set(1) = -3;
    right.set(0) = 5;
    right.set(1) = 7;
    const Array<int> sum_values = left + right;
    REQUIRE(sum_values(0) == 7);
    REQUIRE(sum_values(1) == 4);
}

TEST_CASE("Array public layout retains the declared LP64 ABI offsets",
          "[array][inline-storage][layout]") {
    REQUIRE(sizeof(Array<int>) == 48U);
    REQUIRE(sizeof(Array<double>) == 48U);

    Array<int> integer_array(1);
    const auto* const integer_base =
        reinterpret_cast<const unsigned char*>(&integer_array);
    REQUIRE(reinterpret_cast<const unsigned char*>(&integer_array.dimensions) -
                integer_base ==
            8);
    REQUIRE(reinterpret_cast<const unsigned char*>(&integer_array.nbr) -
                integer_base ==
            32);
    REQUIRE(reinterpret_cast<const unsigned char*>(&integer_array.data) -
                integer_base ==
            40);

    Array<double> double_array(1);
    const auto* const double_base =
        reinterpret_cast<const unsigned char*>(&double_array);
    REQUIRE(reinterpret_cast<const unsigned char*>(&double_array.dimensions) -
                double_base ==
            8);
    REQUIRE(reinterpret_cast<const unsigned char*>(&double_array.nbr) -
                double_base ==
            32);
    REQUIRE(reinterpret_cast<const unsigned char*>(&double_array.data) -
                double_base ==
            40);
}

TEST_CASE("Array<int> move rebases inline storage and transfers heap storage",
          "[array][inline-storage][move]") {
    SECTION("move construction") {
        Array<int> inline_source(2);
        inline_source.set(0) = 3;
        inline_source.set(1) = 5;
        int* const source_inline_pointer = inline_source.get_data();

        Array<int> inline_moved(std::move(inline_source));
        REQUIRE(inline_moved.get_data() != source_inline_pointer);
        REQUIRE(inline_moved(0) == 3);
        REQUIRE(inline_moved(1) == 5);
        REQUIRE(inline_source.get_data() == nullptr);
        REQUIRE(inline_source.get_nbr() == 0U);

        Array<int> heap_source(4);
        for (int i = 0; i < 4; ++i)
            heap_source.set(i) = 20 + i;
        int* const source_heap_pointer = heap_source.get_data();

        Array<int> heap_moved(std::move(heap_source));
        REQUIRE(heap_moved.get_data() == source_heap_pointer);
        REQUIRE(heap_moved(3) == 23);
        REQUIRE(heap_source.get_data() == nullptr);
        REQUIRE(heap_source.get_nbr() == 0U);

        Array<int> empty_source(0);
        Array<int> empty_moved(std::move(empty_source));
        REQUIRE(empty_moved.get_data() == nullptr);
        REQUIRE(empty_moved.get_nbr() == 0U);
        REQUIRE(empty_source.get_data() == nullptr);
        REQUIRE(empty_source.get_nbr() == 0U);
    }

    SECTION("move assignment covers inline, heap, and mixed ownership") {
        Array<int> destination(2);
        destination.set(0) = -1;
        destination.set(1) = -2;
        int* const destination_inline_pointer = destination.get_data();

        Array<int> inline_source(2);
        inline_source.set(0) = 7;
        inline_source.set(1) = 11;
        int* const source_inline_pointer = inline_source.get_data();
        destination = std::move(inline_source);
        REQUIRE(destination.get_data() == destination_inline_pointer);
        REQUIRE(destination.get_data() != source_inline_pointer);
        REQUIRE(destination(0) == 7);
        REQUIRE(destination(1) == 11);
        REQUIRE(inline_source.get_data() == nullptr);

        Array<int> heap_source(4);
        for (int i = 0; i < 4; ++i)
            heap_source.set(i) = 30 + i;
        int* const source_heap_pointer = heap_source.get_data();
        destination = std::move(heap_source);
        REQUIRE(destination.get_data() == source_heap_pointer);
        REQUIRE(destination(3) == 33);
        REQUIRE(heap_source.get_data() == nullptr);

        Array<int> second_inline_source(1);
        second_inline_source.set(0) = 41;
        destination = std::move(second_inline_source);
        REQUIRE(destination.get_data() == destination_inline_pointer);
        REQUIRE(destination(0) == 41);
        REQUIRE(second_inline_source.get_data() == nullptr);

        Array<int> heap_destination(3);
        Array<int> second_heap_source(4);
        second_heap_source.set(3) = 59;
        int* const second_heap_pointer = second_heap_source.get_data();
        heap_destination = std::move(second_heap_source);
        REQUIRE(heap_destination.get_data() == second_heap_pointer);
        REQUIRE(heap_destination(3) == 59);
    }
}

TEST_CASE("Array<int> swap handles every ownership pairing",
          "[array][inline-storage][swap]") {
    SECTION("inline-inline and self") {
        Array<int> left(1);
        left.set(0) = 3;
        Array<int> right(2);
        right.set(0) = 5;
        right.set(1) = 7;
        int* const left_pointer = left.get_data();
        int* const right_pointer = right.get_data();

        left.swap(right);
        REQUIRE(left.get_data() == left_pointer);
        REQUIRE(right.get_data() == right_pointer);
        REQUIRE(left.get_nbr() == 2U);
        REQUIRE(left(0) == 5);
        REQUIRE(left(1) == 7);
        REQUIRE(right.get_nbr() == 1U);
        REQUIRE(right(0) == 3);

        int* const self_pointer = left.get_data();
        left.swap(left);
        REQUIRE(left.get_data() == self_pointer);
        REQUIRE(left(1) == 7);

        Array<int>* const self = &left;
        left = std::move(*self);
        REQUIRE(left.get_data() == self_pointer);
        REQUIRE(left(1) == 7);
    }

    SECTION("heap-heap") {
        Array<int> left(3);
        Array<int> right(4);
        left.set(2) = 13;
        right.set(3) = 17;
        int* const left_pointer = left.get_data();
        int* const right_pointer = right.get_data();

        left.swap(right);
        REQUIRE(left.get_data() == right_pointer);
        REQUIRE(right.get_data() == left_pointer);
        REQUIRE(left.get_nbr() == 4U);
        REQUIRE(left(3) == 17);
        REQUIRE(right.get_nbr() == 3U);
        REQUIRE(right(2) == 13);
    }

    SECTION("mixed in both call directions") {
        Array<int> small(2);
        small.set(0) = 19;
        small.set(1) = 23;
        Array<int> large(4);
        large.set(3) = 29;
        int* const small_inline_pointer = small.get_data();
        int* const large_heap_pointer = large.get_data();

        small.swap(large);
        REQUIRE(small.get_data() == large_heap_pointer);
        REQUIRE(small.get_nbr() == 4U);
        REQUIRE(small(3) == 29);
        REQUIRE(large.get_data() != small_inline_pointer);
        REQUIRE(large.get_nbr() == 2U);
        REQUIRE(large(0) == 19);
        REQUIRE(large(1) == 23);

        small.swap(large);
        REQUIRE(small.get_data() == small_inline_pointer);
        REQUIRE(small.get_nbr() == 2U);
        REQUIRE(small(1) == 23);
        REQUIRE(large.get_data() == large_heap_pointer);
        REQUIRE(large.get_nbr() == 4U);
        REQUIRE(large(3) == 29);
    }

    SECTION("zero-sized and inline") {
        Array<int> empty(0);
        Array<int> scalar(1);
        scalar.set(0) = 31;
        int* const scalar_inline_pointer = scalar.get_data();

        empty.swap(scalar);
        REQUIRE(empty.get_data() != nullptr);
        REQUIRE(empty.get_data() != scalar_inline_pointer);
        REQUIRE(empty.get_nbr() == 1U);
        REQUIRE(empty(0) == 31);
        REQUIRE(scalar.get_data() == nullptr);
        REQUIRE(scalar.get_nbr() == 0U);

        scalar.swap(empty);
        REQUIRE(scalar.get_data() == scalar_inline_pointer);
        REQUIRE(scalar(0) == 31);
        REQUIRE(empty.get_data() == nullptr);
        REQUIRE(empty.get_nbr() == 0U);
    }

    SECTION("deleted inline and fallback states") {
        Array<int> deleted_inline(2);
        deleted_inline.delete_data();
        Array<int> live_heap(4);
        live_heap.set(3) = 37;
        int* const heap_pointer = live_heap.get_data();

        deleted_inline.swap(live_heap);
        REQUIRE(deleted_inline.get_data() == heap_pointer);
        REQUIRE(deleted_inline.get_nbr() == 4U);
        REQUIRE(deleted_inline(3) == 37);
        REQUIRE(live_heap.get_data() == nullptr);
        REQUIRE(live_heap.get_nbr() == 2U);

        Array<int> deleted_fallback(4);
        deleted_fallback.delete_data();
        Array<int> live_inline(2);
        live_inline.set(0) = 41;
        live_inline.set(1) = 43;
        int* const source_inline_pointer = live_inline.get_data();

        deleted_fallback.swap(live_inline);
        REQUIRE(deleted_fallback.get_data() != nullptr);
        REQUIRE(deleted_fallback.get_data() != source_inline_pointer);
        REQUIRE(deleted_fallback.get_nbr() == 2U);
        REQUIRE(deleted_fallback(0) == 41);
        REQUIRE(deleted_fallback(1) == 43);
        REQUIRE(live_inline.get_data() == nullptr);
        REQUIRE(live_inline.get_nbr() == 4U);
    }
}

TEST_CASE("Array<int> vector relocation preserves inline and heap ownership",
          "[array][inline-storage][move]") {
    std::vector<Array<int>> values;
    values.reserve(2);
    values.emplace_back(2);
    values[0].set(0) = 37;
    values[0].set(1) = 41;
    values.emplace_back(4);
    values[1].set(3) = 43;

    int* const old_inline_pointer = values[0].get_data();
    int* const heap_pointer = values[1].get_data();
    values.reserve(values.capacity() + 1U);

    REQUIRE(values[0].get_data() != old_inline_pointer);
    REQUIRE(values[0](0) == 37);
    REQUIRE(values[0](1) == 41);
    REQUIRE(values[1].get_data() == heap_pointer);
    REQUIRE(values[1](3) == 43);
}

TEST_CASE("Array<int> refuses overflowing element counts before allocation",
          "[array][inline-storage][failure]") {
    Dim_array dimensions(3);
    dimensions.set(0) = std::numeric_limits<int>::max();
    dimensions.set(1) = std::numeric_limits<int>::max();
    dimensions.set(2) = std::numeric_limits<int>::max();

    REQUIRE_THROWS_AS(Array<int>(dimensions), std::overflow_error);
}

TEST_CASE("Array<int> delete_data preserves existing public count semantics",
          "[array][inline-storage]") {
    Array<int> inline_array(2);
    inline_array.delete_data();
    REQUIRE(inline_array.get_data() == nullptr);
    REQUIRE(inline_array.get_nbr() == 2U);
    inline_array.delete_data();
    REQUIRE(inline_array.get_data() == nullptr);
    REQUIRE(inline_array.get_nbr() == 2U);

    Array<int> fallback_array(4);
    fallback_array.delete_data();
    REQUIRE(fallback_array.get_data() == nullptr);
    REQUIRE(fallback_array.get_nbr() == 4U);
    fallback_array.delete_data();
    REQUIRE(fallback_array.get_data() == nullptr);
    REQUIRE(fallback_array.get_nbr() == 4U);
}

TEST_CASE("Array swap preserves inline and fallback shapes with their data",
          "[array][swap]") {
    Dim_array inline_dims(3);
    inline_dims.set(0) = 1;
    inline_dims.set(1) = 1;
    inline_dims.set(2) = 2;
    Dim_array fallback_dims(4);
    fallback_dims.set(0) = 1;
    fallback_dims.set(1) = 1;
    fallback_dims.set(2) = 1;
    fallback_dims.set(3) = 3;

    Array<double> inline_array(inline_dims);
    inline_array.set(0) = 5.0;
    inline_array.set(1) = 7.0;
    Array<double> fallback_array(fallback_dims);
    fallback_array.set(0) = 11.0;
    fallback_array.set(1) = 13.0;
    fallback_array.set(2) = 17.0;

    inline_array.swap(fallback_array);

    REQUIRE(inline_array.get_dimensions() == fallback_dims);
    REQUIRE(inline_array.get_nbr() == 3);
    REQUIRE(inline_array(2) == 17.0);
    REQUIRE(fallback_array.get_dimensions() == inline_dims);
    REQUIRE(fallback_array.get_nbr() == 2);
    REQUIRE(fallback_array(1) == 7.0);
}

TEST_CASE("Array move ctor takes ownership", "[array][move]") {
    Dim_array dims(1); dims.set(0) = 5;
    Array<double> src(dims);
    src.set(0)=1; src.set(1)=2; src.set(2)=3; src.set(3)=4; src.set(4)=5;

    Array<double> moved(std::move(src));

    REQUIRE_THAT(moved(0), WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(moved(4), WithinAbs(5.0, 1e-14));
    // src must be safe to destroy (data == nullptr; dtor is null-safe).
}

TEST_CASE("Array move assignment takes ownership", "[array][move]") {
    Dim_array dims_a(1); dims_a.set(0) = 3;
    Dim_array dims_b(1); dims_b.set(0) = 5;
    Array<double> dst(dims_a);
    dst.set(0)=10; dst.set(1)=20; dst.set(2)=30;
    Array<double> src(dims_b);
    src.set(0)=1; src.set(1)=2; src.set(2)=3; src.set(3)=4; src.set(4)=5;

    dst = std::move(src);

    // dst adopts src's shape (5 elements) and values.
    REQUIRE_THAT(dst(0), WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(dst(4), WithinAbs(5.0, 1e-14));
    // src is in a valid moved-from state; dtor is safe.
}

TEST_CASE("Array move self-assignment is a no-op", "[array][move]") {
    Dim_array dims(1); dims.set(0) = 3;
    Array<double> a(dims);
    a.set(0)=7; a.set(1)=8; a.set(2)=9;

    Array<double>* self = &a;
    a = std::move(*self);

    REQUIRE_THAT(a(0), WithinAbs(7.0, 1e-14));
    REQUIRE_THAT(a(2), WithinAbs(9.0, 1e-14));
}

TEST_CASE("Array moved-from object can be reused via assignment", "[array][move]") {
    Dim_array dims(1); dims.set(0) = 3;
    Array<double> src(dims);
    src.set(0)=1; src.set(1)=2; src.set(2)=3;
    Array<double> moved(std::move(src));

    // src now has data == nullptr. Reassign from a fresh Array (move).
    Array<double> fresh(dims);
    fresh.set(0)=100; fresh.set(1)=200; fresh.set(2)=300;
    src = std::move(fresh);

    REQUIRE_THAT(src(0), WithinAbs(100.0, 1e-14));
    REQUIRE_THAT(src(2), WithinAbs(300.0, 1e-14));
}

TEST_CASE("Array chained moves preserve values", "[array][move]") {
    Dim_array dims(1); dims.set(0) = 4;
    Array<double> a(dims);
    a.set(0)=11; a.set(1)=22; a.set(2)=33; a.set(3)=44;

    Array<double> b(std::move(a));
    Array<double> c(std::move(b));
    Array<double> d(std::move(c));

    REQUIRE_THAT(d(0), WithinAbs(11.0, 1e-14));
    REQUIRE_THAT(d(3), WithinAbs(44.0, 1e-14));
}
