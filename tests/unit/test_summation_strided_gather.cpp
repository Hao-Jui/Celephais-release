#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/index.hpp"
#include "For_Kadath/Array/memory.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"

#include <array>
#include <barrier>
#include <bit>
#include <cstdint>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

namespace Kadath {
double summation_1d(int base, double xx, const Array<double>& tab);
}

namespace {

/**
 * Reference collapse of every dimension but the last, using the per-coefficient
 * \c Index walk that \c Base_spectral::summation_but_last performed before it read the
 * coefficient line straight out of the buffer at constant stride. Only the addressing
 * changed, so the two must agree to the bit for every basis.
 */
Kadath::Array<double> reference_but_last(const Kadath::Base_spectral& base, int ndim,
                                        const Kadath::Point& num, const Kadath::Array<double>& cf)
{
    auto* courant = new Kadath::Array<double>(cf);
    Kadath::Dim_array nbr_coefs(cf.get_dimensions());

    for (int d = 0; d < ndim - 1; d++) {
        const int dim_output = ndim - 1 - d;
        Kadath::Dim_array nbr_output(dim_output);
        for (int k = 0; k < dim_output; k++)
            nbr_output.set(k) = nbr_coefs(k + d + 1);
        Kadath::Array<double> output(nbr_output);

        Kadath::Index inout(nbr_output);
        Kadath::Array<double> tab_1d(cf.get_size(d));
        Kadath::Index incourant(courant->get_dimensions());

        bool loop = true;
        while (loop) {
            const int base_1d = (*base.get_base_1d(d))(inout);
            for (int k = 0; k < dim_output; k++)
                incourant.set(k + 1) = inout(k);
            for (int k = 0; k < cf.get_size(d); k++) {
                incourant.set(0) = k;
                tab_1d.set(k) = (*courant)(incourant);
            }
            output.set(inout) = Kadath::summation_1d(base_1d, num(d + 1), tab_1d);
            loop = inout.inc();
        }
        delete courant;
        courant = new Kadath::Array<double>(std::move(output));
    }

    Kadath::Array<double> partial(std::move(*courant));
    delete courant;
    return partial;
}

bool same_bits(double a, double b)
{
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

} // namespace

TEST_CASE("summation gathers coefficient lines bit-identically", "[summation]")
{
    // Distinct sizes per axis so a stride or flat-offset mix-up cannot pass, and one
    // triple repeating a basis on every axis so a per-axis carry-over would show up.
    const std::vector<std::array<int, 3>> base_triples{
        {COSSIN, CHEB_EVEN, CHEB},   {COSSIN, CHEB_ODD, CHEB},
        {COS, COS_EVEN, CHEB},       {SIN, SIN_ODD, CHEB_EVEN},
        {COSSIN_EVEN, COS_ODD, LEG}, {COSSIN_ODD, SIN_EVEN, LEG_EVEN},
        {COS, COS, COS},             {CHEB, CHEB, CHEB},
        {LEG_ODD, COS_EVEN, CHEB_ODD},
    };

    for (const auto& triple : base_triples)
        for (int nr : {1, 7}) {
            Kadath::Dim_array nbr_coefs(3);
            nbr_coefs.set(0) = nr;
            nbr_coefs.set(1) = 5;
            nbr_coefs.set(2) = 4;

            Kadath::Base_spectral base(3);
            // set(nbr_coefs, basephi, basetheta, baser): axis 0 is radial.
            base.set(nbr_coefs, triple[0], triple[1], triple[2]);

            Kadath::Array<double> cf(nbr_coefs);
            for (std::size_t i = 0; i < cf.get_nbr(); ++i)
                cf.get_data()[i] =
                    0.125 * static_cast<double>(i % 11) - 0.03125 * static_cast<double>(i % 7);

            Kadath::Point num(3);
            num.set(1) = 0.3125;
            num.set(2) = 0.75;
            num.set(3) = -0.5;

            const Kadath::Array<double> got = base.summation_but_last(num, cf);
            const Kadath::Array<double> want = reference_but_last(base, 3, num, cf);
            REQUIRE(got.get_nbr() == want.get_nbr());
            for (std::size_t i = 0; i < got.get_nbr(); ++i)
                REQUIRE(same_bits(got.get_data()[i], want.get_data()[i]));

            REQUIRE(same_bits(base.summation(num, cf), base.summation_last_dim(num(3), want)));
        }
}

TEST_CASE("direct-strided Chebyshev collapse preserves scalar reduction bits", "[summation]")
{
    const std::array<int, 3> chebyshev_bases{CHEB, CHEB_EVEN, CHEB_ODD};
    const std::array<double, 5> coordinates{
        -1.0, -0.0, 0.21875, 0.9375, 1.0,
    };

    for (const int radial : chebyshev_bases)
        for (const int polar : chebyshev_bases)
            for (const int azimuthal : chebyshev_bases)
                for (const int order : {10, 12, 16}) {
                    Kadath::Dim_array dimensions(3);
                    dimensions.set(0) = order;
                    dimensions.set(1) = order - 1;
                    dimensions.set(2) = order - 2;

                    Kadath::Base_spectral base(3);
                    base.set(dimensions, radial, polar, azimuthal);

                    Kadath::Array<double> coefficients(dimensions);
                    for (std::size_t i = 0; i < coefficients.get_nbr(); ++i) {
                        const double alternating = (i % 2 == 0) ? 1.0 : -1.0;
                        coefficients.get_data()[i] =
                            alternating * (0.0009765625 * static_cast<double>((i * 37U) % 257U)) +
                            0.000030517578125 * static_cast<double>((i * 19U) % 31U);
                    }

                    for (std::size_t coordinate = 0; coordinate < coordinates.size(); ++coordinate) {
                        Kadath::Point numerical(3);
                        numerical.set(1) = coordinates[coordinate];
                        numerical.set(2) = coordinates[(coordinate + 2) % coordinates.size()];
                        numerical.set(3) = coordinates[(coordinate + 4) % coordinates.size()];

                        const Kadath::Array<double> expected =
                            reference_but_last(base, 3, numerical, coefficients);
                        const Kadath::Array<double> observed =
                            base.summation_but_last(numerical, coefficients);
                        REQUIRE(observed.get_nbr() == expected.get_nbr());
                        for (std::size_t i = 0; i < observed.get_nbr(); ++i)
                            REQUIRE(same_bits(observed.get_data()[i], expected.get_data()[i]));
                        REQUIRE(same_bits(
                            base.summation(numerical, coefficients),
                            base.summation_last_dim(numerical(3), expected)));
                    }
                }
}

TEST_CASE("four-point spectral lanes preserve independent scalar summation bits", "[summation]")
{
    const std::array<std::array<int, 3>, 5> base_triples{
        std::array<int, 3>{CHEB, CHEB, COSSIN},
        std::array<int, 3>{CHEB_EVEN, CHEB_ODD, COSSIN},
        std::array<int, 3>{CHEB_ODD, CHEB_EVEN, COSSIN},
        std::array<int, 3>{CHEB, CHEB_ODD, CHEB_EVEN},
        std::array<int, 3>{LEG, COS_EVEN, SIN_ODD},
    };
    const std::array<std::array<double, 3>, 4> coordinates{
        std::array<double, 3>{-1.0, -0.0, 0.21875},
        std::array<double, 3>{-0.625, 0.3125, -0.875},
        std::array<double, 3>{0.0, -0.9375, 0.5},
        std::array<double, 3>{1.0, 0.75, -0.25},
    };

    for (const auto& bases : base_triples)
        for (const int order : {10, 12, 16}) {
            Kadath::Dim_array dimensions(3);
            dimensions.set(0) = order;
            dimensions.set(1) = order - 1;
            dimensions.set(2) = order - 2;

            Kadath::Base_spectral base(3);
            base.set(dimensions, bases[0], bases[1], bases[2]);

            Kadath::Array<double> coefficients(dimensions);
            for (std::size_t i = 0; i < coefficients.get_nbr(); ++i) {
                const double alternating = (i % 2 == 0) ? 1.0 : -1.0;
                coefficients.get_data()[i] =
                    alternating * 0.0009765625 * static_cast<double>((i * 41U) % 263U) +
                    0.000030517578125 * static_cast<double>((i * 23U) % 29U);
            }

            std::array<Kadath::Point, 4> points{
                Kadath::Point(3), Kadath::Point(3), Kadath::Point(3), Kadath::Point(3)};
            std::array<const Kadath::Point*, 4> point_ptrs{};
            std::array<double, 4> observed{};
            for (std::size_t point = 0; point < points.size(); ++point) {
                point_ptrs[point] = &points[point];
                for (int dimension = 0; dimension < 3; ++dimension)
                    points[point].set(dimension + 1) = coordinates[point][dimension];
            }

            base.summation_points4(point_ptrs, coefficients, observed);
            for (std::size_t point = 0; point < points.size(); ++point)
                REQUIRE(same_bits(observed[point], base.summation(points[point], coefficients)));

            // Reuse the retained worker-local workspace with different coordinates.
            for (std::size_t point = 0; point < points.size(); ++point)
                points[point].set(1) *= 0.5;
            base.summation_points4(point_ptrs, coefficients, observed);
            for (std::size_t point = 0; point < points.size(); ++point)
                REQUIRE(same_bits(observed[point], base.summation(points[point], coefficients)));
        }
}

TEST_CASE("four-point spectral lanes reject null points and isolate worker scratch", "[summation]")
{
    Kadath::Dim_array dimensions(3);
    dimensions.set(0) = 12;
    dimensions.set(1) = 11;
    dimensions.set(2) = 10;
    Kadath::Base_spectral base(3);
    base.set(dimensions, CHEB_ODD, CHEB_EVEN, COSSIN);

    Kadath::Array<double> coefficients(dimensions);
    for (std::size_t i = 0; i < coefficients.get_nbr(); ++i)
        coefficients.get_data()[i] =
            (i % 3 == 0 ? -1.0 : 1.0) * 0.00390625 * static_cast<double>((i * 17U) % 127U);

    std::array<Kadath::Point, 4> points{
        Kadath::Point(3), Kadath::Point(3), Kadath::Point(3), Kadath::Point(3)};
    std::array<const Kadath::Point*, 4> point_ptrs{};
    for (std::size_t point = 0; point < points.size(); ++point) {
        point_ptrs[point] = &points[point];
        points[point].set(1) = -0.75 + 0.5 * static_cast<double>(point);
        points[point].set(2) = 0.625 - 0.25 * static_cast<double>(point);
        points[point].set(3) = -0.5 + 0.375 * static_cast<double>(point);
    }

    std::array<double, 4> ignored{};
    auto invalid_ptrs = point_ptrs;
    invalid_ptrs[2] = nullptr;
    REQUIRE_THROWS(base.summation_points4(invalid_ptrs, coefficients, ignored));

    constexpr std::size_t worker_count = 6;
    std::array<std::array<std::uint64_t, 4>, worker_count> expected{};
    std::array<std::array<std::uint64_t, 4>, worker_count> observed{};
    for (std::size_t worker = 0; worker < worker_count; ++worker)
        for (std::size_t point = 0; point < points.size(); ++point)
            expected[worker][point] =
                std::bit_cast<std::uint64_t>(base.summation(points[point], coefficients));

    std::barrier ready(static_cast<std::ptrdiff_t>(worker_count));
    std::array<std::thread, worker_count> workers;
    for (std::size_t worker = 0; worker < worker_count; ++worker)
        workers[worker] = std::thread([&, worker] {
            ready.arrive_and_wait();
            std::array<double, 4> values{};
            for (int repeat = 0; repeat < 16; ++repeat)
                base.summation_points4(point_ptrs, coefficients, values);
            for (std::size_t point = 0; point < values.size(); ++point)
                observed[worker][point] = std::bit_cast<std::uint64_t>(values[point]);
        });
    for (std::thread& worker : workers)
        worker.join();
    REQUIRE(observed == expected);
}

TEST_CASE("summation still rejects a basis with no 1D kernel", "[summation]")
{
    Kadath::Dim_array nbr_coefs(3);
    nbr_coefs.set(0) = 4;
    nbr_coefs.set(1) = 3;
    nbr_coefs.set(2) = 2;

    Kadath::Base_spectral base(3);
    base.set(nbr_coefs, COSSIN, CHEB_EVEN, NBR_MAX_BASE - 1);

    Kadath::Array<double> cf(nbr_coefs);
    for (std::size_t i = 0; i < cf.get_nbr(); ++i)
        cf.get_data()[i] = 1.0;

    Kadath::Point num(3);
    num.set(1) = 0.5;
    num.set(2) = 0.5;
    num.set(3) = 0.5;

    REQUIRE_THROWS(base.summation_but_last(num, cf));
    std::array<Kadath::Point, 4> points{num, num, num, num};
    std::array<const Kadath::Point*, 4> point_ptrs{
        &points[0], &points[1], &points[2], &points[3]};
    std::array<double, 4> point_values{};
    REQUIRE_THROWS(base.summation_points4(point_ptrs, cf, point_values));

    // The throwing call must release its scratch lease so a subsequent valid
    // call on the same worker can reuse the slot normally.
    Kadath::Base_spectral valid_base(3);
    valid_base.set(nbr_coefs, COSSIN, CHEB_EVEN, CHEB);
    const Kadath::Array<double> got = valid_base.summation_but_last(num, cf);
    const Kadath::Array<double> want = reference_but_last(valid_base, 3, num, cf);
    REQUIRE(got.get_nbr() == want.get_nbr());
    for (std::size_t i = 0; i < got.get_nbr(); ++i)
        REQUIRE(same_bits(got.get_data()[i], want.get_data()[i]));
    valid_base.summation_points4(point_ptrs, cf, point_values);
    for (const double value : point_values)
        REQUIRE(same_bits(value, valid_base.summation(num, cf)));
}

TEST_CASE("summation carries non-finite coefficients unchanged", "[summation]")
{
    Kadath::Dim_array nbr_coefs(3);
    nbr_coefs.set(0) = 6;
    nbr_coefs.set(1) = 3;
    nbr_coefs.set(2) = 2;

    Kadath::Base_spectral base(3);
    base.set(nbr_coefs, COSSIN, CHEB_EVEN, CHEB);

    Kadath::Array<double> cf(nbr_coefs);
    for (std::size_t i = 0; i < cf.get_nbr(); ++i)
        cf.get_data()[i] = 0.0625 * static_cast<double>(i % 5);
    cf.get_data()[0] = std::numeric_limits<double>::quiet_NaN();
    cf.get_data()[7] = std::numeric_limits<double>::infinity();
    cf.get_data()[11] = -std::numeric_limits<double>::infinity();
    cf.get_data()[13] = -0.0;

    Kadath::Point num(3);
    num.set(1) = -0.875;
    num.set(2) = 0.0;
    num.set(3) = 1.0;

    const Kadath::Array<double> got = base.summation_but_last(num, cf);
    const Kadath::Array<double> want = reference_but_last(base, 3, num, cf);
    for (std::size_t i = 0; i < got.get_nbr(); ++i)
        REQUIRE(same_bits(got.get_data()[i], want.get_data()[i]));

    std::array<Kadath::Point, 4> points{num, num, num, num};
    points[1].set(1) = -0.0;
    points[2].set(2) = -0.5;
    points[3].set(3) = -1.0;
    const std::array<const Kadath::Point*, 4> point_ptrs{
        &points[0], &points[1], &points[2], &points[3]};
    std::array<double, 4> point_values{};
    base.summation_points4(point_ptrs, cf, point_values);
    for (std::size_t point = 0; point < points.size(); ++point)
        REQUIRE(same_bits(point_values[point], base.summation(points[point], cf)));
}

TEST_CASE("summation reuses bounded worker scratch and returns detached partials", "[summation]")
{
    Kadath::Dim_array nbr_coefs(3);
    nbr_coefs.set(0) = 7;
    nbr_coefs.set(1) = 5;
    nbr_coefs.set(2) = 4;

    Kadath::Base_spectral base(3);
    base.set(nbr_coefs, COSSIN_ODD, LEG_EVEN, CHEB_ODD);

    Kadath::Array<double> cf(nbr_coefs);
    for (std::size_t i = 0; i < cf.get_nbr(); ++i)
        cf.get_data()[i] = 0.015625 * static_cast<double>((i * 17U) % 29U) - 0.1875;

    Kadath::Point num(3);
    num.set(1) = -0.4375;
    num.set(2) = 0.625;
    num.set(3) = -0.75;

    const Kadath::Array<double> want = reference_but_last(base, 3, num, cf);
    const double want_full = base.summation_last_dim(num(3), want);

    // Warm this worker's retained shapes before observing allocator traffic.
    REQUIRE(same_bits(base.summation(num, cf), want_full));
    {
        const Kadath::Array<double> warm = base.summation_but_last(num, cf);
        REQUIRE(warm.get_nbr() == want.get_nbr());
    }

    constexpr std::size_t calls = 8;
    Kadath::MemoryMapperTrafficSnapshot full_traffic;
    {
        Kadath::MemoryMapperJacobianTrafficScope scope(true);
        for (std::size_t call = 0; call < calls; ++call)
            REQUIRE(same_bits(base.summation(num, cf), want_full));
        full_traffic = scope.finish();
    }
    REQUIRE(full_traffic.get_calls == 0);
    REQUIRE(full_traffic.release_calls == 0);

    Kadath::MemoryMapperTrafficSnapshot partial_traffic;
    {
        Kadath::MemoryMapperJacobianTrafficScope scope(true);
        for (std::size_t call = 0; call < calls; ++call) {
            const Kadath::Array<double> got = base.summation_but_last(num, cf);
            for (std::size_t i = 0; i < got.get_nbr(); ++i)
                REQUIRE(same_bits(got.get_data()[i], want.get_data()[i]));
        }
        partial_traffic = scope.finish();
    }
    // Only the detached, caller-owned 1-D payload allocates after warm-up.
    REQUIRE(partial_traffic.get_calls == calls);
    REQUIRE(partial_traffic.release_calls == calls);
    REQUIRE(partial_traffic.requested_get_bytes == calls * want.get_nbr() * sizeof(double));
    REQUIRE(partial_traffic.requested_release_bytes == calls * want.get_nbr() * sizeof(double));

    Kadath::Array<double> escaped = base.summation_but_last(num, cf);
    const std::uint64_t saved = std::bit_cast<std::uint64_t>(escaped.get_data()[0]);
    escaped.get_data()[0] = 1234.5;
    const Kadath::Array<double> next = base.summation_but_last(num, cf);
    REQUIRE(std::bit_cast<std::uint64_t>(next.get_data()[0]) == saved);
    REQUIRE(escaped.get_data()[0] == 1234.5);
}

TEST_CASE("summation scratch tolerates size changes and overlapping result lifetimes", "[summation]")
{
    Kadath::Dim_array first_dims(3);
    first_dims.set(0) = 3;
    first_dims.set(1) = 4;
    first_dims.set(2) = 2;
    Kadath::Base_spectral first_base(3);
    first_base.set(first_dims, COSSIN, CHEB_EVEN, LEG);

    Kadath::Dim_array second_dims(3);
    second_dims.set(0) = 9;
    second_dims.set(1) = 3;
    second_dims.set(2) = 5;
    Kadath::Base_spectral second_base(3);
    second_base.set(second_dims, COS, LEG_ODD, CHEB);

    Kadath::Array<double> first_cf(first_dims);
    Kadath::Array<double> second_cf(second_dims);
    for (std::size_t i = 0; i < first_cf.get_nbr(); ++i)
        first_cf.get_data()[i] = 0.25 - 0.03125 * static_cast<double>(i);
    for (std::size_t i = 0; i < second_cf.get_nbr(); ++i)
        second_cf.get_data()[i] = 0.0078125 * static_cast<double>((i * 13U) % 37U) - 0.0625;

    Kadath::Point first_num(3);
    first_num.set(1) = 0.125;
    first_num.set(2) = -0.25;
    first_num.set(3) = 0.75;
    Kadath::Point second_num(3);
    second_num.set(1) = -0.625;
    second_num.set(2) = 0.375;
    second_num.set(3) = -0.5;

    const Kadath::Array<double> first_want = reference_but_last(first_base, 3, first_num, first_cf);
    const Kadath::Array<double> second_want = reference_but_last(second_base, 3, second_num, second_cf);

    std::function<bool(int)> reenter = [&](int depth) {
        const bool first = depth % 2 == 0;
        const Kadath::Base_spectral& selected_base = first ? first_base : second_base;
        const Kadath::Point& selected_num = first ? first_num : second_num;
        const Kadath::Array<double>& selected_cf = first ? first_cf : second_cf;
        const Kadath::Array<double>& selected_want = first ? first_want : second_want;

        const Kadath::Array<double> outer = selected_base.summation_but_last(selected_num, selected_cf);
        std::vector<std::uint64_t> before(outer.get_nbr());
        for (std::size_t i = 0; i < outer.get_nbr(); ++i)
            before[i] = std::bit_cast<std::uint64_t>(outer.get_data()[i]);

        if (depth > 0 && !reenter(depth - 1))
            return false;

        if (outer.get_nbr() != selected_want.get_nbr())
            return false;
        for (std::size_t i = 0; i < outer.get_nbr(); ++i) {
            if (std::bit_cast<std::uint64_t>(outer.get_data()[i]) != before[i] ||
                !same_bits(outer.get_data()[i], selected_want.get_data()[i]))
                return false;
        }
        return true;
    };

    REQUIRE(reenter(8));

    // The first reduced stage alone exceeds the 64 KiB retained-scratch ceiling.
    // Warming cannot suppress allocator traffic because this shape must use the
    // call-local fallback, whose complete allocation set is reclaimed on return.
    Kadath::Dim_array oversized_dims(3);
    oversized_dims.set(0) = 2;
    oversized_dims.set(1) = 92;
    oversized_dims.set(2) = 90;
    Kadath::Base_spectral oversized_base(3);
    oversized_base.set(oversized_dims, CHEB, CHEB, CHEB);
    Kadath::Array<double> oversized_cf(oversized_dims);
    for (std::size_t i = 0; i < oversized_cf.get_nbr(); ++i)
        oversized_cf.get_data()[i] = 0.001953125 * static_cast<double>((i * 5U) % 31U) - 0.03125;
    Kadath::Point oversized_num(3);
    oversized_num.set(1) = 0.25;
    oversized_num.set(2) = -0.125;
    oversized_num.set(3) = 0.5;
    const Kadath::Array<double> oversized_want = reference_but_last(oversized_base, 3, oversized_num, oversized_cf);
    const double oversized_full = oversized_base.summation_last_dim(oversized_num(3), oversized_want);
    REQUIRE(same_bits(oversized_base.summation(oversized_num, oversized_cf), oversized_full));

    Kadath::MemoryMapperTrafficSnapshot fallback_traffic;
    {
        Kadath::MemoryMapperJacobianTrafficScope scope(true);
        REQUIRE(same_bits(oversized_base.summation(oversized_num, oversized_cf), oversized_full));
        fallback_traffic = scope.finish();
    }
    REQUIRE(fallback_traffic.get_calls > 0);
    REQUIRE(fallback_traffic.release_calls == fallback_traffic.get_calls);
    REQUIRE(fallback_traffic.requested_get_bytes >= 92U * 90U * sizeof(double));
    REQUIRE(fallback_traffic.requested_release_bytes == fallback_traffic.requested_get_bytes);
}

TEST_CASE("summation scratch is isolated across worker threads", "[summation]")
{
    Kadath::Dim_array nbr_coefs(3);
    nbr_coefs.set(0) = 7;
    nbr_coefs.set(1) = 5;
    nbr_coefs.set(2) = 4;
    Kadath::Base_spectral base(3);
    base.set(nbr_coefs, COSSIN_EVEN, LEG_ODD, CHEB);

    Kadath::Point num(3);
    num.set(1) = 0.3125;
    num.set(2) = -0.6875;
    num.set(3) = 0.875;

    constexpr std::size_t workers = 6;
    constexpr std::size_t repetitions = 64;
    std::array<std::uint64_t, workers> observed{};
    std::barrier start(static_cast<std::ptrdiff_t>(workers));
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            Kadath::Array<double> cf(nbr_coefs);
            for (std::size_t i = 0; i < cf.get_nbr(); ++i)
                cf.get_data()[i] = 0.00390625 * static_cast<double>((i * 11U + worker * 7U) % 41U) - 0.078125;
            start.arrive_and_wait();

            std::uint64_t value = 0;
            for (std::size_t repeat = 0; repeat < repetitions; ++repeat)
                value = std::bit_cast<std::uint64_t>(base.summation(num, cf));
            observed[worker] = value;
        });
    }
    for (auto& thread : threads)
        thread.join();

    // Compute serial references after the synchronized worker calls. When this
    // case is selected alone, workers exercise cold dispatch initialization.
    std::array<std::uint64_t, workers> expected{};
    for (std::size_t worker = 0; worker < workers; ++worker) {
        Kadath::Array<double> cf(nbr_coefs);
        for (std::size_t i = 0; i < cf.get_nbr(); ++i)
            cf.get_data()[i] =
                0.00390625 * static_cast<double>((i * 11U + worker * 7U) % 41U) -
                0.078125;
        expected[worker] = std::bit_cast<std::uint64_t>(base.summation(num, cf));
    }
    REQUIRE(observed == expected);
}
