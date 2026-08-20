#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    void coef_i_1d(int, Array<double>&);
} // namespace Kadath

using namespace Kadath;

namespace
{
    // ------------------------------------------------------------------
    // Dense modal reference.
    //
    // Written straight from the definition of each basis with std::cos and
    // std::sin, sharing no code with the transforms under test, so it is an
    // independent oracle rather than a second copy of the kernel.
    // ------------------------------------------------------------------

    /// Collocation samples carried by a line of \c nbr slots.
    int sample_count(int base, int nbr) { return (base == COSSIN) ? nbr - 2 : nbr; }

    /// Abscissa of collocation point \c j: an angle, or x for the Chebyshev
    /// parity bases, matching each domain's do_coloc().
    double collocation_point(int base, int j, int nbr)
    {
        switch (base) {
            case CHEB: // x = -cos(pi j / (nbr-1)), passed as the angle acos(x)
                return M_PI - M_PI * j / (nbr - 1);
            case CHEB_EVEN:
            case CHEB_ODD:
                return std::sin(M_PI / 2. * j / (nbr - 1));
            case COS:
            case SIN:
                return M_PI * j / (nbr - 1);
            case COSSIN:
                return M_PI * 2. * j / (nbr - 2);
            default: // COS_EVEN, COS_ODD, SIN_EVEN, SIN_ODD
                return M_PI / 2. * j / (nbr - 1);
        }
    }

    double dense_modal_sum(int base, double t, const Array<double>& coefficients)
    {
        const int nbr = coefficients.get_size(0);
        double total = 0.;
        for (int i = 0; i < nbr; i++) {
            const double c = coefficients(i);
            switch (base) {
                case CHEB: total += c * std::cos(i * t); break;
                case CHEB_EVEN: total += c * std::cos(2 * i * std::acos(t)); break;
                case CHEB_ODD: total += c * std::cos((2 * i + 1) * std::acos(t)); break;
                case COS: total += c * std::cos(i * t); break;
                case SIN: total += (i == 0) ? 0. : c * std::sin(i * t); break;
                case COS_EVEN: total += c * std::cos(2 * i * t); break;
                case COS_ODD: total += c * std::cos((2 * i + 1) * t); break;
                case SIN_EVEN: total += (i == 0) ? 0. : c * std::sin(2 * i * t); break;
                case SIN_ODD: total += c * std::sin((2 * i + 1) * t); break;
                case COSSIN:
                    total += (i % 2 == 0) ? c * std::cos((i / 2) * t) : c * std::sin(((i - 1) / 2) * t);
                    break;
                default: break;
            }
        }
        return total;
    }

    double sample_value(int seed, int i) { return std::sin(0.71 * i + 0.13 * seed) + 0.25 * std::cos(1.9 * i); }

    /// A line of samples that the basis represents exactly: pick coefficients,
    /// then evaluate them densely at the collocation points.
    Array<double> representable_line(int base, int nbr, int seed, Array<double>& coefficients_out)
    {
        Array<double> coefficients(nbr);
        for (int i = 0; i < nbr; i++)
            coefficients.set(i) = sample_value(seed, i);
        coefficients_out = coefficients;

        Array<double> samples(nbr);
        samples = 0.;
        for (int j = 0; j < sample_count(base, nbr); j++)
            samples.set(j) = dense_modal_sum(base, collocation_point(base, j, nbr), coefficients);
        return samples;
    }

    /// Runs an unrelated transform of the same length, so the shared transform
    /// workspace for that size holds something different on the next call.
    void perturb_call_history(void (*transform)(int, Array<double>&), int base, int nbr, int seed)
    {
        Array<double> junk(nbr);
        for (int i = 0; i < nbr; i++)
            junk.set(i) = 1e3 * sample_value(seed, i) + 7.;
        transform(base, junk);
    }

    const std::vector<int>& folding_bases()
    {
        static const std::vector<int> bases{CHEB, CHEB_EVEN, CHEB_ODD, COS,      SIN,
                                            COS_EVEN, COS_ODD, SIN_EVEN, SIN_ODD};
        return bases;
    }

    /// Line lengths each basis is defined for under the even-transform contract.
    const std::vector<int>& supported_lengths(int base)
    {
        static const std::vector<int> odd{5, 7, 9, 11};
        static const std::vector<int> even{6, 8, 10, 12};
        return (base == COSSIN) ? even : odd;
    }

    const std::vector<int>& rejected_lengths(int base)
    {
        // 4 is the smallest folding line with an unpaired slot: the pair loop
        // is empty there and slot 2 is the one that used to carry over.
        static const std::vector<int> even{4, 6, 8, 10, 12};
        static const std::vector<int> odd{5, 7, 9, 11};
        return (base == COSSIN) ? odd : even;
    }

    /// Bases whose transforms are exact at the supported line-length parity.
    std::vector<int> exact_bases()
    {
        std::vector<int> bases = folding_bases();
        bases.push_back(COSSIN);
        return bases;
    }

    /// Exposes the protected basis arrays so a test can choose the basis of an axis.
    class OpenBase : public Base_spectral
    {
      public:
        explicit OpenBase(const Dim_array& dimensions) : Base_spectral(dimensions.get_ndim())
        {
            allocate(dimensions);
            def = true;
        }

        void set_axis_uniform(int axis, int basis)
        {
            Array<int>& axis_bases = *bases_1d[static_cast<std::size_t>(axis)];
            for (std::size_t i = 0; i < axis_bases.get_nbr(); i++)
                axis_bases.get_data()[i] = basis;
        }
    };
} // namespace

TEST_CASE("one-dimensional transforms do not depend on call history", "[coef-line-length]")
{
    // The buffered-route scratch workspace is shared by every transform of a given size.
    // A transform that writes every slot it reads cannot see the previous
    // call, so the same input must give byte-identical output whatever ran
    // before it.
    for (int base : exact_bases()) {
        INFO("base=" << base);
        for (int nbr : supported_lengths(base)) {
            INFO("nbr=" << nbr);
            Array<double> coefficients(nbr);
            const Array<double> samples = representable_line(base, nbr, 3, coefficients);

            perturb_call_history(coef_1d, base, nbr, 11);
            Array<double> first(samples);
            coef_1d(base, first);

            perturb_call_history(coef_1d, base, nbr, 29);
            Array<double> second(samples);
            coef_1d(base, second);

            REQUIRE(std::memcmp(first.get_data(), second.get_data(), nbr * sizeof(double)) == 0);

            perturb_call_history(coef_i_1d, base, nbr, 17);
            Array<double> back_first(first);
            coef_i_1d(base, back_first);

            perturb_call_history(coef_i_1d, base, nbr, 41);
            Array<double> back_second(first);
            coef_i_1d(base, back_second);

            REQUIRE(std::memcmp(back_first.get_data(), back_second.get_data(),
                                nbr * sizeof(double)) == 0);
        }
    }
}

TEST_CASE("one-dimensional transforms reject unsupported line-length parity",
          "[coef-line-length]")
{
    for (int base : exact_bases()) {
        INFO("base=" << base);
        for (int nbr : rejected_lengths(base)) {
            INFO("nbr=" << nbr);
            Array<double> line(nbr);
            line = 0.;
            REQUIRE_THROWS_AS(coef_1d(base, line), KadathError);
            REQUIRE_THROWS_AS(coef_i_1d(base, line), KadathError);
        }
    }
}

TEST_CASE("forward transform interpolates the dense modal reference", "[coef-line-length]")
{
    for (int base : exact_bases()) {
        INFO("base=" << base);
        for (int nbr : supported_lengths(base)) {
            INFO("nbr=" << nbr);
            for (int seed : {1, 2, 5}) {
                INFO("seed=" << seed);
                Array<double> coefficients(nbr);
                const Array<double> samples = representable_line(base, nbr, seed, coefficients);

                Array<double> produced(samples);
                coef_1d(base, produced);

                for (int j = 0; j < sample_count(base, nbr); j++) {
                    const double reference =
                        dense_modal_sum(base, collocation_point(base, j, nbr), produced);
                    REQUIRE(std::fabs(reference - samples(j)) < 1e-10);
                }
            }
        }
    }
}

TEST_CASE("inverse transform undoes the forward transform", "[coef-line-length]")
{
    for (int base : exact_bases()) {
        INFO("base=" << base);
        for (int nbr : supported_lengths(base)) {
            INFO("nbr=" << nbr);
            for (int seed : {1, 2, 5}) {
                INFO("seed=" << seed);
                Array<double> coefficients(nbr);
                const Array<double> samples = representable_line(base, nbr, seed, coefficients);

                Array<double> round_trip(samples);
                coef_1d(base, round_trip);
                coef_i_1d(base, round_trip);

                for (int j = 0; j < sample_count(base, nbr); j++)
                    REQUIRE(std::fabs(round_trip(j) - samples(j)) < 1e-10);
            }
        }
    }
}

TEST_CASE("a growing line transform does not consume the previous line", "[coef-line-length]")
{
    // coef_dim_into sizes the line scratch to max(nbr_coef, nbr_conf) but
    // gathers only nbr_conf values, and reuses that scratch across the lines
    // of one call. Two inputs that agree on a line must agree on its output,
    // whatever the other lines hold.
    const int conf_len = 7;
    const int coef_len = 9; // grows by 2, both odd, so CHEB is defined at both
    Dim_array coef_dims(3);
    coef_dims.set(0) = 2;
    coef_dims.set(1) = 3;
    coef_dims.set(2) = coef_len;
    Dim_array conf_dims(coef_dims);
    conf_dims.set(2) = conf_len;

    OpenBase prepared(coef_dims);
    for (int axis = 0; axis < 3; axis++)
        prepared.set_axis_uniform(axis, CHEB);
    const Base_spectral base(prepared);

    Array<double> plain(conf_dims);
    Array<double> perturbed(conf_dims);
    for (std::size_t i = 0; i < plain.get_nbr(); i++) {
        const double v = sample_value(4, static_cast<int>(i));
        plain.get_data()[i] = v;
        perturbed.get_data()[i] = v;
    }
    // Lines along the transformed axis are contiguous here (before == 1);
    // change only the first one.
    for (int k = 0; k < conf_len; k++)
        perturbed.get_data()[k] += 1.5;

    const Array<double> from_plain = base.coef_dim(2, coef_len, plain);
    const Array<double> from_perturbed = base.coef_dim(2, coef_len, perturbed);

    const int line_count = coef_dims(0) * coef_dims(1);
    for (int line = 1; line < line_count; line++) {
        INFO("line=" << line);
        const std::size_t offset = static_cast<std::size_t>(line * coef_len);
        REQUIRE(std::memcmp(from_plain.get_data() + offset, from_perturbed.get_data() + offset,
                            coef_len * sizeof(double)) == 0);
    }
}
