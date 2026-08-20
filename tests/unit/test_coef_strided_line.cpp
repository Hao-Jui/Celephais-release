#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"

#include <cstring>
#include <vector>

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    bool coef_1d(int, const double*, double*, int, int, int);
    void coef_i_1d(int, Array<double>&);
    bool coef_i_1d(int, const double*, double*, int, int, int);
} // namespace Kadath

using namespace Kadath;

namespace
{
    // Every basis the two dispatchers route, forward and inverse alike.
    const std::vector<int>& every_basis()
    {
        static const std::vector<int> bases{CHEB,     CHEB_EVEN, CHEB_ODD, COSSIN,      COS_EVEN,
                                           COS_ODD,  SIN_EVEN,  SIN_ODD,  COS,         SIN,
                                           LEG,      LEG_EVEN,  LEG_ODD,  COSSIN_EVEN, COSSIN_ODD};
        return bases;
    }

    // Materialised once and shared by every arm: comparing two arms that
    // recompute an input formula compares their rounding, not the code under
    // test (Lane F's false alarm).
    std::vector<double> line_fixture(int nbr, int salt)
    {
        std::vector<double> values(static_cast<std::size_t>(nbr));
        for (int k = 0; k < nbr; k++) {
            const int centered = (k * 13 + 5 * salt) % 31 - 15;
            values[static_cast<std::size_t>(k)] = static_cast<double>(centered) / 8. + 0.125;
        }
        return values;
    }

    const double kPoison = -7.5;

    bool same_bytes(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
} // namespace

// The driver hands the kernels a line of a larger array, so the same transform
// has to come out of any stride, and nothing outside the line may be touched -
// the slots between two line entries belong to other lines.
TEST_CASE("strided coefficient kernels are stride invariant", "[coef-strided]")
{
    int checked = 0;
    for (int basis : every_basis()) {
        // Exercise every supported transform length through N=32. Folding
        // families require odd nbr (N=nbr-1), COSSIN requires even nbr
        // (N=nbr-2), and the doubled COSSIN parity wrappers reach N=32 at
        // nbr=18. Legendre keeps its historical limit and does not route
        // through the native real FFT under test.
        const bool legendre = basis == LEG || basis == LEG_EVEN || basis == LEG_ODD;
        const bool cossin = basis == COSSIN;
        const bool doubled_cossin = basis == COSSIN_EVEN || basis == COSSIN_ODD;
        const int maximum_nbr = legendre ? 21 : (doubled_cossin ? 18 : (cossin ? 34 : 33));
        for (int nbr = 3; nbr <= maximum_nbr; nbr++) {
            if ((cossin && nbr % 2 != 0)
                || (!legendre && !cossin && !doubled_cossin && nbr % 2 == 0))
                continue;
            const std::vector<double> in = line_fixture(nbr, nbr * 3 + basis);

            // Contiguous, in place: the shape every legacy call site uses.
            Array<double> contiguous(nbr);
            for (int k = 0; k < nbr; k++)
                contiguous.set(k) = in[static_cast<std::size_t>(k)];
            coef_1d(basis, contiguous);

            // The inverse COSSIN_EVEN/COSSIN_ODD terminal-pair stores are
            // guarded at odd nbr since 2026-08-09 (they were out-of-bounds
            // dead writes; the 4-slot packing loop already fills the doubled
            // line there). Odd nbr stays out of contract - the symmetric phi
            // axis is always even - so the inverse arm still skips it rather
            // than asserting an undefined transform.
            const bool inverse_defined =
                !((basis == COSSIN_EVEN || basis == COSSIN_ODD) && nbr % 2 != 0);

            Array<double> contiguous_i(nbr);
            if (inverse_defined) {
                for (int k = 0; k < nbr; k++)
                    contiguous_i.set(k) = in[static_cast<std::size_t>(k)];
                coef_i_1d(basis, contiguous_i);
            }

            for (int stride : {1, 2, 3, 5}) {
                const std::size_t span = static_cast<std::size_t>(nbr) * static_cast<std::size_t>(stride);
                std::vector<double> src(span, 0.);
                for (int k = 0; k < nbr; k++)
                    src[static_cast<std::size_t>(k) * stride] = in[static_cast<std::size_t>(k)];

                INFO("basis=" << basis << " nbr=" << nbr << " stride=" << stride);

                for (int direction = 0; direction < 2; direction++) {
                    if (direction == 1 && !inverse_defined)
                        continue;
                    INFO("direction=" << direction);
                    std::vector<double> dst(span, kPoison);
                    const bool ran = direction == 0
                                         ? coef_1d(basis, src.data(), dst.data(), nbr, nbr, stride)
                                         : coef_i_1d(basis, src.data(), dst.data(), nbr, nbr, stride);
                    REQUIRE(ran); // equal extents are always accepted
                    const Array<double>& expected = direction == 0 ? contiguous : contiguous_i;
                    for (int k = 0; k < nbr; k++) {
                        INFO("slot=" << k);
                        REQUIRE(same_bytes(dst[static_cast<std::size_t>(k) * stride], expected(k)));
                    }
                    for (std::size_t i = 0; i < span; i++)
                        if (i % static_cast<std::size_t>(stride) != 0) {
                            INFO("off-line slot=" << i);
                            REQUIRE(same_bytes(dst[i], kPoison));
                        }
                    checked++;
                }
            }
        }
    }
    REQUIRE(checked > 0);
}

// The COSSIN phi axis is the one axis whose two extents differ: nbr_coefs =
// nbr_points + 2 forward, and the reverse inverse. Its kernels take that
// geometry directly, and must land on what the gathering driver produced - the
// gathered samples with a zero tail, transformed at equal extents.
TEST_CASE("unequal-extent COSSIN matches the zero-tailed gathered line", "[coef-strided]")
{
    int checked = 0;
    // The live phi axis is even, so N=nbr-2 remains even through the N=32
    // upper boundary.
    for (int nbr = 6; nbr <= 34; nbr += 2) {
        const std::vector<double> in = line_fixture(nbr, nbr + 4);

        INFO("nbr=" << nbr);
        for (const int stride : {1, 3, 5}) {
            INFO("stride=" << stride);
            const auto span = static_cast<std::size_t>((nbr - 1) * stride + 1);
            std::vector<double> src(span, kPoison);
            for (int k = 0; k < nbr; ++k)
                src[static_cast<std::size_t>(k * stride)] = in[static_cast<std::size_t>(k)];

            // nbr-2 gathered samples, nbr coefficient slots.  At nbr=34 this
            // reaches the selected N=32 raw transform.
            {
                Array<double> reference(nbr);
                for (int k = 0; k < nbr - 2; k++)
                    reference.set(k) = in[static_cast<std::size_t>(k)];
                for (int k = nbr - 2; k < nbr; k++)
                    reference.set(k) = 0.;
                coef_1d(COSSIN, reference);

                std::vector<double> dst(span, kPoison);
                REQUIRE(coef_1d(COSSIN, src.data(), dst.data(), nbr - 2, nbr, stride));
                for (int k = 0; k < nbr; k++) {
                    INFO("forward slot=" << k);
                    REQUIRE(same_bytes(dst[static_cast<std::size_t>(k * stride)], reference(k)));
                }
                for (std::size_t k = 0; k < span; ++k)
                    if (k % static_cast<std::size_t>(stride) != 0)
                        REQUIRE(same_bytes(dst[k], kPoison));
            }

            // Inverse: nbr coefficients in, nbr-2 collocation samples out.
            {
                Array<double> reference(nbr);
                for (int k = 0; k < nbr; k++)
                    reference.set(k) = in[static_cast<std::size_t>(k)];
                coef_i_1d(COSSIN, reference);

                std::vector<double> dst(span, kPoison);
                REQUIRE(coef_i_1d(COSSIN, src.data(), dst.data(), nbr, nbr - 2, stride));
                for (int k = 0; k < nbr - 2; k++) {
                    INFO("inverse slot=" << k);
                    REQUIRE(same_bytes(dst[static_cast<std::size_t>(k * stride)], reference(k)));
                }
                // Beyond the collocation line the destination belongs to no output.
                REQUIRE(same_bytes(dst[static_cast<std::size_t>((nbr - 2) * stride)], kPoison));
                REQUIRE(same_bytes(dst[static_cast<std::size_t>((nbr - 1) * stride)], kPoison));
                for (std::size_t k = 0; k < span; ++k)
                    if (k % static_cast<std::size_t>(stride) != 0)
                        REQUIRE(same_bytes(dst[k], kPoison));
            }
            checked++;
        }
    }
    REQUIRE(checked > 0);
}

// Everything but COSSIN is written for a line whose two extents match. The
// dispatchers must decline the rest untouched, because that is what keeps the
// driver's gathering path - and with it Lane G's zero-fill - alive.
TEST_CASE("unequal extents are declined for every basis but COSSIN", "[coef-strided]")
{
    const int nbr = 10; // COSSIN control arm uses supported raw N=nbr-2=8.
    for (int basis : every_basis()) {
        const std::vector<double> in = line_fixture(nbr, basis);
        std::vector<double> dst(static_cast<std::size_t>(nbr), kPoison);

        INFO("basis=" << basis);
        const bool grow = coef_1d(basis, in.data(), dst.data(), nbr - 2, nbr, 1);
        REQUIRE(grow == (basis == COSSIN));
        const bool shrink = coef_i_1d(basis, in.data(), dst.data(), nbr, nbr - 2, 1);
        REQUIRE(shrink == (basis == COSSIN));
        // A forward transform never shrinks and an inverse one never grows, so
        // those combinations are declined even for COSSIN.
        REQUIRE_FALSE(coef_1d(basis, in.data(), dst.data(), nbr, nbr - 2, 1));
        REQUIRE_FALSE(coef_i_1d(basis, in.data(), dst.data(), nbr - 2, nbr, 1));

        if (basis != COSSIN)
            for (int k = 0; k < nbr; k++) {
                INFO("slot=" << k);
                REQUIRE(same_bytes(dst[static_cast<std::size_t>(k)], kPoison));
            }
    }
}
