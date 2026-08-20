#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <barrier>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace Kadath;
using Catch::Matchers::WithinAbs;

namespace
{
    class MutableSpectralBase : public Base_spectral
    {
      public:
        MutableSpectralBase() : Base_spectral(1)
        {
            define_slots(CHEB);
        }

        explicit MutableSpectralBase(int dimensions) : Base_spectral(dimensions) {}

        MutableSpectralBase(const MutableSpectralBase&) = default;
        MutableSpectralBase(MutableSpectralBase&&) noexcept = default;
        MutableSpectralBase& operator=(const MutableSpectralBase&) = default;
        MutableSpectralBase& operator=(MutableSpectralBase&&) noexcept = default;

        void define_slots(int first_basis)
        {
            for (int axis = 0; axis < ndim; ++axis)
                set_slot(axis, first_basis + axis);
            def = true;
        }

        void assign_from(const MutableSpectralBase& source)
        {
            Base_spectral::operator=(source);
        }

        void set_slot(int axis, int basis)
        {
            auto value = std::make_unique<Array<int>>(1);
            value->set(0) = basis;
            bases_1d[static_cast<std::size_t>(axis)] = std::move(value);
        }

        void assign_slot(int destination, int source)
        {
            bases_1d[static_cast<std::size_t>(destination)] =
                bases_1d[static_cast<std::size_t>(source)];
        }

        void set_slot_value(int axis, int basis)
        {
            bases_1d[static_cast<std::size_t>(axis)]->set(0) = basis;
        }

        void set_basis(int basis)
        {
            bases_1d[0]->set(0) = basis;
        }

        int basis() const
        {
            return (*bases_1d[0])(0);
        }

        int slot_value(int axis) const
        {
            return (*bases_1d[static_cast<std::size_t>(axis)])(0);
        }

        const Array<int>* slot_address(int axis) const
        {
            return bases_1d[static_cast<std::size_t>(axis)].get();
        }

        bool slot_is_null(int axis) const
        {
            return bases_1d[static_cast<std::size_t>(axis)] == nullptr;
        }

        bool all_slots_null() const
        {
            for (int axis = 0; axis < ndim; ++axis) {
                if (bases_1d[static_cast<std::size_t>(axis)] != nullptr)
                    return false;
            }
            return true;
        }

        int dimensions() const noexcept { return ndim; }

        void swap_with(MutableSpectralBase& source) noexcept
        {
            swap(source);
        }
    };
}

// --- Point (1-BASED indexing) ---

TEST_CASE("Point construction and access", "[point]") {
    Point p(3);
    p.set(1) = 1.0; p.set(2) = 2.0; p.set(3) = 3.0;
    REQUIRE_THAT(p(1), WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(p(3), WithinAbs(3.0, 1e-14));
    REQUIRE(p.get_ndim() == 3);
}

TEST_CASE("Point copy and assignment preserve value semantics", "[point]") {
    SECTION("inline rank") {
        Point source(3);
        source.set(1) = 5.0;
        source.set(2) = 7.0;
        source.set(3) = 11.0;
        Point copy(source);
        copy.set(1) = 13.0;

        REQUIRE_THAT(source(1), WithinAbs(5.0, 1e-14));
        REQUIRE_THAT(copy(1), WithinAbs(13.0, 1e-14));

        Point assigned(3);
        assigned = source;
        const Point& same = assigned;
        assigned = same;
        REQUIRE_THAT(assigned(3), WithinAbs(11.0, 1e-14));
    }

    SECTION("heap fallback") {
        Point source(4);
        for (int i = 1; i <= source.get_ndim(); ++i)
            source.set(i) = 2.0 * i;
        Point copy(source);
        copy.set(4) = 17.0;

        REQUIRE_THAT(source(4), WithinAbs(8.0, 1e-14));
        REQUIRE_THAT(copy(4), WithinAbs(17.0, 1e-14));

        Point assigned(4);
        assigned = source;
        const Point& same = assigned;
        assigned = same;
        REQUIRE_THAT(assigned(4), WithinAbs(8.0, 1e-14));
    }
}

TEST_CASE("Point copies survive vector relocation", "[point]") {
    std::vector<Point> points;
    points.reserve(1);
    points.emplace_back(3);
    points[0].set(1) = 1.5;
    points[0].set(2) = 2.5;
    points[0].set(3) = 3.5;
    points.emplace_back(4);
    points[1].set(1) = 4.5;
    points[1].set(4) = 7.5;
    points.emplace_back(2);

    REQUIRE(points[0].get_ndim() == 3);
    REQUIRE_THAT(points[0](1), WithinAbs(1.5, 1e-14));
    REQUIRE_THAT(points[0](3), WithinAbs(3.5, 1e-14));
    REQUIRE(points[1].get_ndim() == 4);
    REQUIRE_THAT(points[1](1), WithinAbs(4.5, 1e-14));
    REQUIRE_THAT(points[1](4), WithinAbs(7.5, 1e-14));
}

// --- Base_spectral ---

TEST_CASE("Base_spectral starts undefined", "[base_spectral]") {
    Base_spectral base(3);
    REQUIRE(base.is_def() == false);
}

TEST_CASE("Base_spectral allocate and set", "[base_spectral]") {
    Dim_array nbr_coefs(3);
    nbr_coefs.set(0) = 9; nbr_coefs.set(1) = 5; nbr_coefs.set(2) = 4;

    Base_spectral base(3);
    base.allocate(nbr_coefs);
    base.set(nbr_coefs, COSSIN, COS_EVEN, CHEB);
    REQUIRE(base.is_def() == true);
}

TEST_CASE("Base_spectral copy preserves def", "[base_spectral]") {
    Dim_array nbr_coefs(3);
    nbr_coefs.set(0) = 9; nbr_coefs.set(1) = 5; nbr_coefs.set(2) = 4;

    Base_spectral base(3);
    base.allocate(nbr_coefs);
    base.set(nbr_coefs, COSSIN, COS_EVEN, CHEB);

    Base_spectral copy(base);
    REQUIRE(copy.is_def() == true);
}

TEST_CASE("Base_spectral copy-on-write keeps copied bases independent", "[base_spectral]") {
    MutableSpectralBase original;
    original.set_basis(CHEB);

    SECTION("copy construction") {
        MutableSpectralBase copy(original);
        REQUIRE(copy.get_base_1d(0) == original.get_base_1d(0));

        copy.set_basis(LEG);

        REQUIRE(copy.basis() == LEG);
        REQUIRE(original.basis() == CHEB);
        REQUIRE(copy.get_base_1d(0) != original.get_base_1d(0));
    }

    SECTION("copy assignment") {
        MutableSpectralBase copy;
        copy.set_basis(COS);
        copy.assign_from(original);
        REQUIRE(copy.get_base_1d(0) == original.get_base_1d(0));

        copy.set_basis(LEG);

        REQUIRE(copy.basis() == LEG);
        REQUIRE(original.basis() == CHEB);
        REQUIRE(copy.get_base_1d(0) != original.get_base_1d(0));
    }
}

TEST_CASE("Base_spectral preserves bounded and fallback dimensions",
          "[base_spectral]") {
    for (int dimensions : {1, 2, 3, 4, 5}) {
        MutableSpectralBase base(dimensions);
        REQUIRE(base.dimensions() == dimensions);
        REQUIRE_FALSE(base.is_def());
        REQUIRE(base.all_slots_null());

        base.define_slots(CHEB);
        REQUIRE(base.is_def());
        for (int axis = 0; axis < dimensions; ++axis)
            REQUIRE(base.slot_value(axis) == CHEB + axis);

        base.set_non_def();
        REQUIRE_FALSE(base.is_def());
        REQUIRE(base.all_slots_null());
    }
}

TEST_CASE("Base_spectral refuses negative dimensions", "[base_spectral]") {
    REQUIRE_THROWS_AS(Base_spectral(-1), std::length_error);
}

TEST_CASE("Base_spectral outer and inner snapshots detach on first mutation",
          "[base_spectral]") {
    for (int dimensions : {1, 2, 3, 4, 5}) {
        MutableSpectralBase original(dimensions);
        original.define_slots(CHEB);
        MutableSpectralBase copy(original);

        for (int axis = 0; axis < dimensions; ++axis)
            REQUIRE(copy.slot_address(axis) == original.slot_address(axis));

        copy.set_slot(dimensions - 1, LEG);
        REQUIRE(copy.slot_value(dimensions - 1) == LEG);
        REQUIRE(original.slot_value(dimensions - 1) == CHEB + dimensions - 1);
        for (int axis = 0; axis + 1 < dimensions; ++axis)
            REQUIRE(copy.slot_address(axis) == original.slot_address(axis));

        copy.set_non_def();
        REQUIRE(copy.all_slots_null());
        REQUIRE(original.is_def());
        REQUIRE(original.slot_value(0) == CHEB);
    }
}

TEST_CASE("Base_spectral copy and self-assignment share value snapshots",
          "[base_spectral]") {
    for (int dimensions : {1, 2, 3, 4, 5}) {
        MutableSpectralBase source(dimensions);
        source.define_slots(CHEB);
        MutableSpectralBase assigned(dimensions);
        assigned.assign_from(source);

        for (int axis = 0; axis < dimensions; ++axis)
            REQUIRE(assigned.slot_address(axis) == source.slot_address(axis));

        const MutableSpectralBase& same = assigned;
        assigned.assign_from(same);
        assigned.set_slot(0, LEG);
        REQUIRE(assigned.slot_value(0) == LEG);
        REQUIRE(source.slot_value(0) == CHEB);
    }
}

TEST_CASE("Base_spectral slot proxy preserves same-slot and cross-slot values",
          "[base_spectral]") {
    for (int dimensions : {3, 5}) {
        MutableSpectralBase original(dimensions);
        original.define_slots(CHEB);
        const int last_axis = dimensions - 1;
        const int last_value = CHEB + last_axis;

        MutableSpectralBase same_slot(original);
        same_slot.assign_slot(last_axis, last_axis);
        REQUIRE(same_slot.slot_address(last_axis) == original.slot_address(last_axis));
        REQUIRE(same_slot.slot_value(last_axis) == last_value);

        same_slot.set_slot_value(last_axis, LEG);
        REQUIRE(same_slot.slot_value(last_axis) == LEG);
        REQUIRE(original.slot_value(last_axis) == last_value);

        MutableSpectralBase cross_slot(original);
        cross_slot.assign_slot(0, last_axis);
        REQUIRE(cross_slot.slot_address(0) == cross_slot.slot_address(last_axis));
        REQUIRE(cross_slot.slot_address(0) == original.slot_address(last_axis));
        REQUIRE(cross_slot.slot_value(0) == last_value);
        REQUIRE(original.slot_value(0) == CHEB);

        cross_slot.set_slot_value(0, LEG);
        REQUIRE(cross_slot.slot_value(0) == LEG);
        REQUIRE(cross_slot.slot_value(last_axis) == last_value);
        REQUIRE(original.slot_value(last_axis) == last_value);
        REQUIRE(cross_slot.slot_address(0) != cross_slot.slot_address(last_axis));
    }
}

TEST_CASE("Base_spectral const snapshot supports concurrent reads",
          "[base_spectral]") {
    constexpr int worker_count = 4;
    constexpr int repetitions = 25'000;

    MutableSpectralBase prepared(3);
    prepared.define_slots(CHEB);
    const MutableSpectralBase& base = prepared;

    std::vector<const Array<int>*> original_addresses;
    for (int axis = 0; axis < base.dimensions(); ++axis)
        original_addresses.push_back(base.slot_address(axis));

    std::barrier start_line(worker_count);
    std::vector<std::uint64_t> checksums(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            start_line.arrive_and_wait();
            std::uint64_t checksum = 0;
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                for (int axis = 0; axis < base.dimensions(); ++axis)
                    checksum += static_cast<std::uint64_t>(base.slot_value(axis));
            }
            checksums[worker] = checksum;
        });
    }
    for (std::thread& worker : workers)
        worker.join();

    const std::uint64_t expected =
        static_cast<std::uint64_t>(repetitions) * (CHEB + CHEB + 1 + CHEB + 2);
    for (std::uint64_t checksum : checksums)
        REQUIRE(checksum == expected);
    for (int axis = 0; axis < base.dimensions(); ++axis)
        REQUIRE(base.slot_address(axis) == original_addresses[static_cast<std::size_t>(axis)]);
}

TEST_CASE("Base_spectral move construction leaves a valid source",
          "[base_spectral]") {
    SECTION("bounded dimensions release the source to null slots") {
        MutableSpectralBase source(3);
        source.define_slots(CHEB);
        const Array<int>* const first = source.slot_address(0);

        MutableSpectralBase moved(std::move(source));
        REQUIRE(moved.dimensions() == 3);
        REQUIRE(moved.is_def());
        REQUIRE(moved.slot_address(0) == first);
        REQUIRE_FALSE(source.is_def());
        REQUIRE(source.dimensions() == 3);
        REQUIRE(source.all_slots_null());

        source.define_slots(COS);
        REQUIRE(source.slot_value(0) == COS);
        REQUIRE(moved.slot_value(0) == CHEB);
    }

    SECTION("fallback dimensions retain a usable moved-from snapshot") {
        MutableSpectralBase source(5);
        source.define_slots(CHEB);
        MutableSpectralBase moved(std::move(source));

        REQUIRE(moved.dimensions() == 5);
        REQUIRE(moved.is_def());
        REQUIRE_FALSE(source.is_def());
        REQUIRE(source.dimensions() == 5);

        MutableSpectralBase assigned(5);
        assigned.define_slots(LEG);
        assigned.assign_from(source);
        REQUIRE_FALSE(assigned.is_def());
        REQUIRE(assigned.all_slots_null());

        source.set_non_def();
        REQUIRE(source.all_slots_null());
        source.define_slots(COS);
        REQUIRE(source.slot_value(4) == COS + 4);
        REQUIRE(moved.slot_value(4) == CHEB + 4);
    }
}

TEST_CASE("Base_spectral move assignment and swap preserve general storage",
          "[base_spectral]") {
    MutableSpectralBase bounded(2);
    bounded.define_slots(COS);
    MutableSpectralBase fallback(5);
    fallback.define_slots(CHEB);

    SECTION("move assignment swaps with the source without allocation") {
        bounded = std::move(fallback);
        REQUIRE(bounded.dimensions() == 5);
        REQUIRE(bounded.slot_value(4) == CHEB + 4);
        REQUIRE(fallback.dimensions() == 2);
        REQUIRE(fallback.slot_value(1) == COS + 1);
    }

    SECTION("member swap handles bounded and fallback snapshots") {
        bounded.swap_with(fallback);
        REQUIRE(bounded.dimensions() == 5);
        REQUIRE(bounded.slot_value(4) == CHEB + 4);
        REQUIRE(fallback.dimensions() == 2);
        REQUIRE(fallback.slot_value(1) == COS + 1);
    }
}

TEST_CASE("Base_spectral layout and move operations remain bounded",
          "[base_spectral]") {
    static_assert(sizeof(Base_spectral) == 32);
    static_assert(sizeof(Val_domain) == 152);
    static_assert(std::is_nothrow_move_constructible_v<Base_spectral>);
    static_assert(std::is_nothrow_move_assignable_v<Base_spectral>);

    REQUIRE(sizeof(Base_spectral) == 32);
    REQUIRE(sizeof(Val_domain) == 152);
}
