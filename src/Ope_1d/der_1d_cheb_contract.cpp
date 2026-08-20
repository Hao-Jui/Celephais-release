/*
 * Copyright 2017 Philippe Grandclement
 *
 * This file is part of Kadath and is distributed under the terms of the GNU
 * General Public License, version 3 or (at your option) any later version.
 */

/*
 * Modifications (Celephais):
 *   2026-07-31  Split out an explicit Chebyshev derivative arithmetic
 *               contract; see LICENSE_SOURCE_AUDIT.tsv.
 */

#include "der_1d_cheb_contract.hpp"

#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Base_spectral/base_spectral.hpp"

#include <cassert>
#include <cmath>

namespace Kadath
{
    int der_1d_cheb(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        const int nr = tab.get_size(0);

        Array<double>& res = ope_1d_line_scratch(nr);
        res = 0;
        for (int coefficient = 0; coefficient < nr; ++coefficient) {
            for (int source = coefficient + 1; source < nr; source += 2) {
                res.set(coefficient) = std::fma(
                    static_cast<double>(source), tab(source),
                    res(coefficient));
            }
        }
        res *= 2;
        res.set(0) /= 2;
        tab = res;
        return CHEB;
    }

    int der_1d_cheb_even(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        const int nr = tab.get_size(0);

        Array<double>& res = ope_1d_line_scratch(nr);
        res = 0;
        for (int coefficient = 0; coefficient < nr; ++coefficient) {
            for (int source = coefficient + 1; source < nr; ++source) {
                res.set(coefficient) = std::fma(
                    static_cast<double>(2 * source), tab(source),
                    res(coefficient));
            }
        }
        res.set(nr - 1) = 0;
        res *= 2;
        tab = res;
        return CHEB_ODD;
    }

    int der_1d_cheb_odd(Array<double>& tab)
    {
        assert(tab.get_ndim() == 1);
        const int nr = tab.get_size(0);

        Array<double>& res = ope_1d_line_scratch(nr);
        res = 0;
        for (int coefficient = 0; coefficient < nr; ++coefficient) {
            int source = coefficient;
            const int rounded_end =
                derivative_arithmetic::cheb_odd_rounded_end(coefficient, nr);
            for (; source < rounded_end; ++source) {
                const double weighted_source =
                    (2 * source + 1) * tab(source);
                res.set(coefficient) += weighted_source;
            }
            for (; source < nr; ++source) {
                res.set(coefficient) = std::fma(
                    static_cast<double>(2 * source + 1), tab(source),
                    res(coefficient));
            }
        }
        res *= 2;
        res.set(0) /= 2;
        tab = res;
        return CHEB_EVEN;
    }
}
