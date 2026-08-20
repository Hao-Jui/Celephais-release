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

#ifndef KADATH_DER_1D_CHEB_CONTRACT_HPP
#define KADATH_DER_1D_CHEB_CONTRACT_HPP

namespace Kadath
{
    template <typename T>
    class Array;

    int der_1d_cheb(Array<double>& tab);
    int der_1d_cheb_even(Array<double>& tab);
    int der_1d_cheb_odd(Array<double>& tab);

    namespace derivative_arithmetic
    {
        inline int cheb_odd_rounded_end(int coefficient,
                                        int coefficient_count) noexcept
        {
            return coefficient +
                   ((coefficient_count - coefficient) / 8) * 8;
        }
    }
}

#endif
