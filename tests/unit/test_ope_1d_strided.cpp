#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/index.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace Kadath
{
    int mult_cos_1d(int, const double*, double*, int, int);
    int mult_sin_1d(int, const double*, double*, int, int);
} // namespace Kadath

using namespace Kadath;

namespace
{
    using Strided_kernel = int (*)(int, const double*, double*, int, int);

    /// Every basis mult_cos_1d and mult_sin_1d dispatch to.
    const std::array<int, 7> supported_bases = {COSSIN,   COS_EVEN, COS_ODD, SIN_EVEN,
                                                SIN_ODD,  COS,      SIN};

    /**
     * The cossin kernels index the line up to \c src[5*stride] unconditionally
     * and pair their coefficients two at a time, so they are defined only for
     * even lines of at least six entries.
     */
    bool basis_accepts(int basis, int line_length)
    {
        if (basis == COSSIN)
            return line_length % 2 == 0 && line_length >= 8;
        return line_length >= 4;
    }

    /// Exposes the protected basis arrays so a reference walk can write them.
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

        /// Cycles the supported table so consecutive slots of an axis differ.
        void set_axis_pattern(int axis, int first)
        {
            Array<int>& axis_bases = *bases_1d[static_cast<std::size_t>(axis)];
            for (std::size_t i = 0; i < axis_bases.get_nbr(); i++)
                axis_bases.get_data()[i] =
                    supported_bases[(static_cast<std::size_t>(first) + i) %
                                    supported_bases.size()];
        }

        Array<int>& axis(int a) { return *bases_1d[static_cast<std::size_t>(a)]; }
    };

    Dim_array make_dimensions(std::initializer_list<int> extents)
    {
        Dim_array dimensions(static_cast<int>(extents.size()));
        int axis = 0;
        for (int extent : extents)
            dimensions.set(axis++) = extent;
        return dimensions;
    }

    Array<double> fixture(const Dim_array& dimensions, int salt)
    {
        Array<double> values(dimensions);
        for (std::size_t i = 0; i < values.get_nbr(); i++) {
            const int centered = static_cast<int>((i * 13 + 5 * salt) % 31) - 15;
            values.get_data()[i] = static_cast<double>(centered) / 8. + 0.125;
        }
        return values;
    }

    void require_byte_equal(const Array<double>& actual, const Array<double>& expected)
    {
        REQUIRE(actual.get_nbr() == expected.get_nbr());
        for (std::size_t i = 0; i < actual.get_nbr(); i++) {
            if (std::memcmp(actual.get_data() + i, expected.get_data() + i, sizeof(double)) != 0) {
                std::uint64_t actual_bits = 0;
                std::uint64_t expected_bits = 0;
                std::memcpy(&actual_bits, actual.get_data() + i, sizeof(double));
                std::memcpy(&expected_bits, expected.get_data() + i, sizeof(double));
                UNSCOPED_INFO("first mismatch flat=" << i << " actual_bits=" << actual_bits
                                                     << " expected_bits=" << expected_bits);
                break;
            }
        }
        REQUIRE(std::memcmp(actual.get_data(), expected.get_data(),
                            actual.get_nbr() * sizeof(double)) == 0);
    }

    void require_bases_equal(const Array<int>& actual, const Array<int>& expected)
    {
        REQUIRE(actual.get_nbr() == expected.get_nbr());
        REQUIRE(std::memcmp(actual.get_data(), expected.get_data(),
                            actual.get_nbr() * sizeof(int)) == 0);
    }

    /**
     * The upstream per-element \c Index walk (references/Phillipe/src/Ope_1d/ope_1d.cpp):
     * every buffer address goes through \c Array::operator()(Index), and the line is
     * gathered into a contiguous buffer and scattered back. Only the kernel call is
     * adapted, at stride one, so any difference is the driver's addressing alone.
     */
    Array<double> reference_ope_1d(const Base_spectral& base, int ndim, Strided_kernel func,
                                   int var, const Array<double>& in, OpenBase& base_out)
    {
        Array<double> res(in.get_dimensions());

        int after = 1;
        for (int i = 0; i < var; i++)
            after *= in.get_size(i);
        int before = 1;
        for (int i = var + 1; i < ndim; i++)
            before *= in.get_size(i);

        const int nbr = in.get_size(var);

        Index index_base(base.get_base_1d(var)->get_dimensions());
        Index demarre(in.get_dimensions());
        Index loop_before(in.get_dimensions());
        Index lit(in.get_dimensions());
        Index put(in.get_dimensions());
        std::vector<double> gathered(static_cast<std::size_t>(nbr));
        std::vector<double> transformed(static_cast<std::size_t>(nbr));

        for (int i = 0; i < before; i++) {
            demarre = loop_before;
            const int base_1d = (*base.get_base_1d(var))(index_base);
            for (int j = 0; j < after; j++) {
                lit = demarre;
                for (int k = 0; k < nbr; k++) {
                    gathered[static_cast<std::size_t>(k)] = in(lit);
                    lit.inc(after);
                }
                base_out.axis(var).set(index_base) =
                    func(base_1d, gathered.data(), transformed.data(), nbr, 1);
                put = demarre;
                for (int k = 0; k < nbr; k++) {
                    res.set(put) = transformed[static_cast<std::size_t>(k)];
                    put.inc(after);
                }
                demarre.inc();
            }
            index_base.inc();
            loop_before.inc(1, var + 1);
        }
        return res;
    }

    const std::vector<std::vector<int>>& test_shapes()
    {
        static const std::vector<std::vector<int>> shapes{{5, 7, 9},  {8, 5, 4}, {4, 9, 8},
                                                          {5, 5, 5},  {8, 8, 10}, {7, 5},
                                                          {10, 8},    {9},        {8}};
        return shapes;
    }
} // namespace

TEST_CASE("strided 1-d operators reproduce the Index reference walk", "[ope-1d-strided]")
{
    const std::array<Strided_kernel, 2> kernels = {mult_cos_1d, mult_sin_1d};

    int covered = 0;
    for (const std::vector<int>& extents : test_shapes()) {
        Dim_array dimensions(static_cast<int>(extents.size()));
        for (std::size_t axis = 0; axis < extents.size(); axis++)
            dimensions.set(static_cast<int>(axis)) = extents[axis];
        const int ndim = dimensions.get_ndim();

        for (int var = 0; var < ndim; var++) {
            for (std::size_t which = 0; which < kernels.size(); which++) {
                // one pass per supported basis, then one with the axis carrying
                // a different basis in every slot
                for (std::size_t choice = 0; choice <= supported_bases.size(); choice++) {
                    const bool patterned = choice == supported_bases.size();
                    if (patterned) {
                        bool every_basis_fits = true;
                        for (int basis : supported_bases)
                            every_basis_fits =
                                every_basis_fits && basis_accepts(basis, dimensions(var));
                        if (!every_basis_fits)
                            continue;
                    } else if (!basis_accepts(supported_bases[choice], dimensions(var))) {
                        continue;
                    }

                    OpenBase prepared(dimensions);
                    for (int axis = 0; axis < ndim; axis++)
                        prepared.set_axis_uniform(axis, CHEB);
                    if (patterned)
                        prepared.set_axis_pattern(var, static_cast<int>(which));
                    else
                        prepared.set_axis_uniform(var, supported_bases[choice]);
                    const Base_spectral base(prepared);

                    const Array<double> in =
                        fixture(dimensions, var + 3 * static_cast<int>(choice));

                    OpenBase expected_base(prepared);
                    const Array<double> expected = reference_ope_1d(
                        base, ndim, kernels[which], var, in, expected_base);

                    OpenBase actual_base(prepared);
                    const Array<double> actual =
                        base.ope_1d(kernels[which], var, in, actual_base);

                    UNSCOPED_INFO("ndim=" << ndim << " var=" << var << " kernel=" << which
                                          << " choice=" << choice);
                    require_byte_equal(actual, expected);
                    require_bases_equal(actual_base.axis(var), expected_base.axis(var));
                    covered++;
                }
            }
        }
    }
    REQUIRE(covered > 0);
    UNSCOPED_INFO("configurations compared: " << covered);
}

TEST_CASE("strided 1-d kernels are addressing invariant", "[ope-1d-strided]")
{
    const std::array<Strided_kernel, 2> kernels = {mult_cos_1d, mult_sin_1d};
    // any slot the kernel fails to write keeps this and fails the comparison
    const double poison = -1234.5678e30;

    int slots = 0;
    for (Strided_kernel func : kernels) {
        for (int basis : supported_bases) {
            for (int nbr = 4; nbr <= 17; nbr++) {
                if (!basis_accepts(basis, nbr))
                    continue;

                std::vector<double> line(static_cast<std::size_t>(nbr));
                for (int k = 0; k < nbr; k++)
                    line[static_cast<std::size_t>(k)] =
                        static_cast<double>((k * 13 + 5 * nbr) % 31 - 15) / 8. + 0.125;

                std::vector<double> contiguous(static_cast<std::size_t>(nbr), poison);
                const int contiguous_base =
                    func(basis, line.data(), contiguous.data(), nbr, 1);

                for (int stride : {2, 3, 5}) {
                    const std::size_t span = static_cast<std::size_t>(nbr * stride);
                    std::vector<double> src(span, 0.);
                    std::vector<double> dst(span, poison);
                    for (int k = 0; k < nbr; k++)
                        src[static_cast<std::size_t>(k * stride)] =
                            line[static_cast<std::size_t>(k)];

                    const int strided_base =
                        func(basis, src.data(), dst.data(), nbr, stride);

                    UNSCOPED_INFO("basis=" << basis << " nbr=" << nbr << " stride=" << stride);
                    REQUIRE(strided_base == contiguous_base);
                    for (int k = 0; k < nbr; k++) {
                        const double got = dst[static_cast<std::size_t>(k * stride)];
                        const double want = contiguous[static_cast<std::size_t>(k)];
                        REQUIRE(got != poison);
                        REQUIRE(std::memcmp(&got, &want, sizeof(double)) == 0);
                        slots++;
                    }
                }
            }
        }
    }
    REQUIRE(slots > 0);
    UNSCOPED_INFO("line slots compared: " << slots);
}
