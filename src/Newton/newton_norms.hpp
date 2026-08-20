/*
    Added 2026 Hao-Jui Kuan

    Shared residual-norm helpers for the src/Newton solver backends. The
    infinity norm of the Newton residual was previously copied verbatim into
    do_newton_jfnk_mumps.cpp, do_newton_jfnk_linesearch.cpp, and
    do_newton_jfnk_schur.cpp; this private header holds the single definition.
*/

#pragma once

#include "For_Kadath/Array/array.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace Kadath
{
    // Infinity norm (max |component|) of the Newton residual vector.
    inline double infinity_norm(const Array<double>& values)
    {
        double value = 0.0;
        for (std::size_t i = 0; i < values.get_nbr(); ++i) {
            value = std::max(value, std::abs(values(static_cast<int>(i))));
        }
        return value;
    }
} // namespace Kadath
