#include <catch2/catch_test_macros.hpp>

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Array/index.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"
#include "For_Kadath/Base_spectral/transform_line_offsets.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    void coef_i_1d(int, Array<double>&);
    int der_1d(int, Array<double>&);
} // namespace Kadath

using namespace Kadath;

namespace
{
    /// Exposes the protected basis arrays so a reference walk can write them.
    class OpenBase : public Base_spectral
    {
      public:
        explicit OpenBase(const Dim_array& dimensions)
            : Base_spectral(dimensions.get_ndim())
        {
            allocate(dimensions);
            def = true;
        }

        /// Cycles the full basis table so every slot of every axis differs.
        void set_axis_pattern(int axis, int first_basis)
        {
            static const std::array<int, 9> bases = {CHEB, CHEB_EVEN, CHEB_ODD, COS,   SIN,
                                                     COS_EVEN, COS_ODD, SIN_EVEN, SIN_ODD};
            Array<int>& axis_bases = *bases_1d[static_cast<std::size_t>(axis)];
            for (std::size_t i = 0; i < axis_bases.get_nbr(); i++)
                axis_bases.get_data()[i] =
                    bases[(static_cast<std::size_t>(first_basis) + i) % bases.size()];
        }

        /// Puts one basis on every slot of an axis.
        void set_axis_uniform(int axis, int basis)
        {
            Array<int>& axis_bases = *bases_1d[static_cast<std::size_t>(axis)];
            for (std::size_t i = 0; i < axis_bases.get_nbr(); i++)
                axis_bases.get_data()[i] = basis;
        }

        Array<int>& axis(int a) { return *bases_1d[static_cast<std::size_t>(a)]; }
        int ndim_of() const { return ndim; }
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

    // ---------------------------------------------------------------------
    // Reference drivers: the upstream per-element Index walk
    // (references/Phillipe/src/Coef/coef.cpp, coef_i.cpp, Ope_1d/ope_1d.cpp),
    // which computes every buffer address through Array::operator()(Index).
    // Independent of both the flat-stride line walk and the carried offsets.
    // ---------------------------------------------------------------------

    Array<double> reference_coef_dim(const Base_spectral& base, int ndim, int dim, int nbr_coef,
                                     const Array<double>& inout)
    {
        Dim_array res_out(inout.get_dimensions());
        res_out.set(dim) = nbr_coef;
        Array<double> res(res_out);

        int after = 1;
        for (int i = 0; i < dim; i++)
            after *= inout.get_size(i);
        int before = 1;
        for (int i = dim + 1; i < ndim; i++)
            before *= inout.get_size(i);

        const int nbr_conf = inout.get_size(dim);
        const int nbr = (nbr_coef > nbr_conf) ? nbr_coef : nbr_conf;

        Index index_base(base.get_base_1d(dim)->get_dimensions());
        Index demarre_conf(inout.get_dimensions());
        Index demarre_coef(res_out);
        Index loop_before_conf(inout.get_dimensions());
        Index loop_before_out(res_out);
        Index lit_in(inout.get_dimensions());
        Index put_out(res_out);
        Array<double> tab_1d(nbr);

        for (int i = 0; i < before; i++) {
            demarre_conf = loop_before_conf;
            demarre_coef = loop_before_out;
            const int base_1d = (*base.get_base_1d(dim))(index_base);
            for (int j = 0; j < after; j++) {
                lit_in = demarre_conf;
                for (int k = 0; k < nbr_conf; k++) {
                    tab_1d.set(k) = inout(lit_in);
                    lit_in.inc(after);
                }
                // Mirrors the driver: the line carries only nbr_conf values, so
                // a growing transform must not hand the kernel the tail the
                // previous line left in the scratch (see src/Coef/coef.cpp).
                for (int k = nbr_conf; k < nbr; k++)
                    tab_1d.set(k) = 0.;
                coef_1d(base_1d, tab_1d);
                put_out = demarre_coef;
                for (int k = 0; k < nbr_coef; k++) {
                    res.set(put_out) = tab_1d(k);
                    put_out.inc(after);
                }
                demarre_conf.inc();
                demarre_coef.inc();
            }
            index_base.inc();
            loop_before_conf.inc(1, dim + 1);
            loop_before_out.inc(1, dim + 1);
        }
        return res;
    }

    Array<double> reference_coef_i_dim(const Base_spectral& base, int ndim, int dim, int nbr_conf,
                                       const Array<double>& inout)
    {
        Dim_array res_out(inout.get_dimensions());
        res_out.set(dim) = nbr_conf;
        Array<double> res(res_out);

        int after = 1;
        for (int i = 0; i < dim; i++)
            after *= inout.get_size(i);
        int before = 1;
        for (int i = dim + 1; i < ndim; i++)
            before *= inout.get_size(i);

        const int nbr_coef = inout.get_size(dim);
        const int nbr = (nbr_coef > nbr_conf) ? nbr_coef : nbr_conf;

        Index index_base(base.get_base_1d(dim)->get_dimensions());
        Index demarre_coef(inout.get_dimensions());
        Index demarre_conf(res_out);
        Index loop_before_coef(inout.get_dimensions());
        Index loop_before_out(res_out);
        Index lit_in(inout.get_dimensions());
        Index put_out(res_out);
        Array<double> tab_1d(nbr);

        for (int i = 0; i < before; i++) {
            demarre_coef = loop_before_coef;
            demarre_conf = loop_before_out;
            const int base_1d = (*base.get_base_1d(dim))(index_base);
            for (int j = 0; j < after; j++) {
                lit_in = demarre_coef;
                for (int k = 0; k < nbr_coef; k++) {
                    tab_1d.set(k) = inout(lit_in);
                    lit_in.inc(after);
                }
                // Mirrors the driver; see src/Coef/coef_i.cpp.
                for (int k = nbr_coef; k < nbr; k++)
                    tab_1d.set(k) = 0.;
                coef_i_1d(base_1d, tab_1d);
                put_out = demarre_conf;
                for (int k = 0; k < nbr_conf; k++) {
                    res.set(put_out) = tab_1d(k);
                    put_out.inc(after);
                }
                demarre_conf.inc();
                demarre_coef.inc();
            }
            index_base.inc();
            loop_before_coef.inc(1, dim + 1);
            loop_before_out.inc(1, dim + 1);
        }
        return res;
    }

    Array<double> reference_ope_1d(const Base_spectral& base, int ndim,
                                   int (*func)(int, Array<double>&), int var,
                                   const Array<double>& in, OpenBase& base_out)
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
        Array<double> tab_1d(nbr);

        for (int i = 0; i < before; i++) {
            demarre = loop_before;
            const int base_1d = (*base.get_base_1d(var))(index_base);
            for (int j = 0; j < after; j++) {
                lit = demarre;
                for (int k = 0; k < nbr; k++) {
                    tab_1d.set(k) = in(lit);
                    lit.inc(after);
                }
                base_out.axis(var).set(index_base) = func(base_1d, tab_1d);
                put = demarre;
                for (int k = 0; k < nbr; k++) {
                    res.set(put) = tab_1d(k);
                    put.inc(after);
                }
                demarre.inc();
            }
            index_base.inc();
            loop_before.inc(1, var + 1);
        }
        return res;
    }

    /**
     * Odd extents throughout.
     *
     * \c coef_1d_cheb_even (src/Coef/coef_1d.cpp:115) leaves the middle slot of
     * its transform workspace unwritten when the line length is even -- for
     * nr = 4 the pair loop \c for(i=1;i<(nr-1)/2;i++) is empty, so \c tab_auxi[2]
     * still holds the previous transform's data when the buffered route runs. That
     * kernel's output then depends on call history, which no addressing test can
     * be held to. Even lengths are therefore out of scope here; the determinism
     * guard below fails loudly if any basis in the table develops the same flaw.
     */
    const std::vector<std::vector<int>>& test_shapes()
    {
        static const std::vector<std::vector<int>> shapes{
            {5, 7, 9}, {7, 5, 3}, {3, 9, 7}, {5, 5, 5}, {7, 5}, {5, 7}, {9}};
        return shapes;
    }

} // namespace

TEST_CASE("carried line offsets reproduce the Index odometer", "[transform-line-offsets]")
{
    // Same digit tuples in the same order, and the flat offsets of two buffers
    // that differ in the size of the transformed axis.
    for (const std::vector<int>& shape : test_shapes()) {
        const int ndim = static_cast<int>(shape.size());
        Dim_array in_dims(ndim);
        for (int axis = 0; axis < ndim; axis++)
            in_dims.set(axis) = shape[static_cast<std::size_t>(axis)];

        for (int dim = 0; dim < ndim; dim++) {
            for (int growth : {0, 2}) {
                Dim_array out_dims(in_dims);
                out_dims.set(dim) = in_dims(dim) + growth;

                int before = 1;
                for (int axis = dim + 1; axis < ndim; axis++)
                    before *= in_dims(axis);
                int after = 1;
                for (int axis = 0; axis < dim; axis++)
                    after *= in_dims(axis);

                // Outer walk: the axes above `dim`, weighted alike in both buffers.
                Transform_line_offsets outer(in_dims, dim + 1, ndim, 1, 1);
                Index loop_before_in(in_dims);
                Index loop_before_out(out_dims);
                for (int i = 0; i < before; i++) {
                    int expected_in = loop_before_in(0);
                    int expected_out = loop_before_out(0);
                    for (int axis = 1; axis < ndim; axis++) {
                        expected_in = expected_in * in_dims(axis) + loop_before_in(axis);
                        expected_out = expected_out * out_dims(axis) + loop_before_out(axis);
                    }
                    REQUIRE(outer.in_offset() == expected_in);
                    REQUIRE(outer.out_offset() == expected_out);
                    outer.advance();
                    loop_before_in.inc(1, dim + 1);
                    loop_before_out.inc(1, dim + 1);
                }
                // A full cycle returns the odometer to its starting tuple.
                REQUIRE(outer.in_offset() == 0);

                // Inner walk: the axes below `dim`, whose weight carries the
                // transformed axis and so differs between the two buffers.
                Transform_line_offsets inner(in_dims, 0, dim, in_dims(dim) * before,
                                             out_dims(dim) * before);
                Index demarre_in(in_dims);
                Index demarre_out(out_dims);
                for (int j = 0; j < after; j++) {
                    int expected_in = demarre_in(0);
                    int expected_out = demarre_out(0);
                    for (int axis = 1; axis < ndim; axis++) {
                        expected_in = expected_in * in_dims(axis) + demarre_in(axis);
                        expected_out = expected_out * out_dims(axis) + demarre_out(axis);
                    }
                    REQUIRE(inner.in_offset() == expected_in);
                    REQUIRE(inner.out_offset() == expected_out);
                    inner.advance();
                    demarre_in.inc();
                    demarre_out.inc();
                }
                REQUIRE(inner.in_offset() == 0);
                REQUIRE(inner.out_offset() == 0);
            }
        }
    }
}

TEST_CASE("transform drivers address lines bit-identically", "[transform-line-offsets]")
{
    int checked = 0;
    for (const std::vector<int>& shape : test_shapes()) {
        const int ndim = static_cast<int>(shape.size());

        for (int dim = 0; dim < ndim; dim++) {
            // The basis of an axis is stored over the axes above it, so the
            // basis allocation fixes the shape of those axes; the transformed
            // axis itself may grow or shrink, as Base_spectral::coef does.
            for (int growth : {0, 2, -2}) {
                Dim_array coef_dims(ndim);
                for (int axis = 0; axis < ndim; axis++)
                    coef_dims.set(axis) = shape[static_cast<std::size_t>(axis)];

                Dim_array conf_dims(coef_dims);
                conf_dims.set(dim) = coef_dims(dim) - growth;
                if (conf_dims(dim) < 2)
                    continue;

                OpenBase prepared(coef_dims);
                for (int axis = 0; axis < ndim; axis++)
                    prepared.set_axis_pattern(axis, axis + 1);
                const Base_spectral base(prepared);

                INFO("ndim=" << ndim << " dim=" << dim << " growth=" << growth
                             << " coef_dim=" << coef_dims(dim) << " conf_dim=" << conf_dims(dim));

                // Growing transforms are compared too. They used to be skipped
                // because the line scratch is sized to the larger count while
                // the gather fills only the smaller one, so the kernel read the
                // previous line's leftovers and its output was a function of
                // call history rather than of the input alone. The drivers now
                // zero that gap (coef.cpp, coef_i.cpp), which is what the
                // determinism check below pins down.

                // Forward: conf sizes in, coef sizes out.
                const Array<double> conf_values = fixture(conf_dims, dim + 1);
                {
                    INFO("driver=coef_dim");
                    const Array<double> expected =
                        reference_coef_dim(base, ndim, dim, coef_dims(dim), conf_values);
                    // Guard: a kernel that reads its own leftovers would make any
                    // comparison depend on which walk ran first.
                    INFO("check=kernel determinism");
                    require_byte_equal(
                        reference_coef_dim(base, ndim, dim, coef_dims(dim), conf_values), expected);
                    INFO("check=addressing");
                    require_byte_equal(base.coef_dim(dim, coef_dims(dim), conf_values), expected);
                }

                // Inverse: coef sizes in, conf sizes out.
                const Array<double> coef_values = fixture(coef_dims, dim + 5);
                {
                    INFO("driver=coef_i_dim");
                    const Array<double> expected =
                        reference_coef_i_dim(base, ndim, dim, conf_dims(dim), coef_values);
                    INFO("check=kernel determinism");
                    require_byte_equal(
                        reference_coef_i_dim(base, ndim, dim, conf_dims(dim), coef_values),
                        expected);
                    INFO("check=addressing");
                    require_byte_equal(base.coef_i_dim(dim, conf_dims(dim), coef_values), expected);
                }

                // Derivative operator: shape preserving, and it writes the
                // output basis of the transformed axis as it goes.
                if (growth == 0) {
                    INFO("driver=ope_1d");
                    OpenBase produced(coef_dims);
                    OpenBase expected(coef_dims);
                    for (int axis = 0; axis < ndim; axis++) {
                        produced.set_axis_pattern(axis, axis + 1);
                        expected.set_axis_pattern(axis, axis + 1);
                    }
                    require_byte_equal(
                        base.ope_1d(der_1d, dim, coef_values, produced),
                        reference_ope_1d(base, ndim, der_1d, dim, coef_values, expected));
                    REQUIRE(produced.axis(dim).get_nbr() == expected.axis(dim).get_nbr());
                    REQUIRE(std::memcmp(produced.axis(dim).get_data(),
                                        expected.axis(dim).get_data(),
                                        produced.axis(dim).get_nbr() * sizeof(int)) == 0);
                }
                checked++;
            }
        }
    }
    REQUIRE(checked > 0);
}

TEST_CASE("transform drivers address even-extent lines bit-identically",
          "[transform-line-offsets]")
{
    // Even line extents are defined only for the COSSIN family, which folds
    // nbr-2 samples rather than nbr-1; the folding bases above need an odd
    // extent and are rejected at an even one. This is the phi axis of every
    // spheric and bispheric domain, and the only axis that grows in practice
    // (nbr_coefs = nbr_points + 2), so it exercises the odometer at even
    // extents and on a growing transform at once.
    int checked = 0;
    for (const std::vector<int>& shape : {
             std::vector<int>{4, 6, 8}, std::vector<int>{6, 4},
             std::vector<int>{8}, std::vector<int>{3, 5, 6},
             // GCC/x86 cached-plan targets: output extent N+2 and line
             // strides 1, 2, 3, 5. Extents 10 and 20 pin both fallback edges.
             std::vector<int>{4, 10, 1}, std::vector<int>{4, 12, 1},
             std::vector<int>{4, 14, 2}, std::vector<int>{4, 16, 3},
             std::vector<int>{4, 18, 5}, std::vector<int>{4, 20, 1}}) {
        const int ndim = static_cast<int>(shape.size());

        for (int dim = 0; dim < ndim; dim++) {
            if (shape[static_cast<std::size_t>(dim)] % 2 != 0)
                continue; // COSSIN is not defined at an odd extent

            for (int growth : {0, 2, -2}) {
                Dim_array coef_dims(ndim);
                for (int axis = 0; axis < ndim; axis++)
                    coef_dims.set(axis) = shape[static_cast<std::size_t>(axis)];

                Dim_array conf_dims(coef_dims);
                conf_dims.set(dim) = coef_dims(dim) - growth;
                if (conf_dims(dim) < 4)
                    continue;

                OpenBase prepared(coef_dims);
                for (int axis = 0; axis < ndim; axis++)
                    prepared.set_axis_uniform(axis, COSSIN);
                const Base_spectral base(prepared);

                INFO("ndim=" << ndim << " dim=" << dim << " growth=" << growth
                             << " coef_dim=" << coef_dims(dim) << " conf_dim=" << conf_dims(dim));

                const Array<double> conf_values = fixture(conf_dims, dim + 1);
                {
                    INFO("driver=coef_dim");
                    const Array<double> expected =
                        reference_coef_dim(base, ndim, dim, coef_dims(dim), conf_values);
                    INFO("check=kernel determinism");
                    require_byte_equal(
                        reference_coef_dim(base, ndim, dim, coef_dims(dim), conf_values), expected);
                    INFO("check=addressing");
                    require_byte_equal(base.coef_dim(dim, coef_dims(dim), conf_values), expected);
                }

                const Array<double> coef_values = fixture(coef_dims, dim + 5);
                {
                    INFO("driver=coef_i_dim");
                    const Array<double> expected =
                        reference_coef_i_dim(base, ndim, dim, conf_dims(dim), coef_values);
                    INFO("check=kernel determinism");
                    require_byte_equal(
                        reference_coef_i_dim(base, ndim, dim, conf_dims(dim), coef_values),
                        expected);
                    INFO("check=addressing");
                    require_byte_equal(base.coef_i_dim(dim, conf_dims(dim), coef_values), expected);
                }
                checked++;
            }
        }
    }
    REQUIRE(checked > 0);
}
