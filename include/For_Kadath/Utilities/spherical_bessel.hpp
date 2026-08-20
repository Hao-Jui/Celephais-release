/*
    Copyright 2026 Celephais contributors

    This file is part of Kadath.

    Kadath is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Kadath is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Kadath.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef KADATH_UTILITIES_SPHERICAL_BESSEL_HPP
#define KADATH_UTILITIES_SPHERICAL_BESSEL_HPP

#include <algorithm>
#include <cmath>
#include <limits>

namespace Kadath::special_functions
{
    namespace detail
    {
        inline constexpr double quiet_nan = std::numeric_limits<double>::quiet_NaN();

        inline long double spherical_j0(long double x) noexcept
        {
            if (x < 1.e-4L) {
                const long double x2 = x * x;
                return 1.L - x2 / 6.L + x2 * x2 / 120.L - x2 * x2 * x2 / 5040.L;
            }
            return std::sin(x) / x;
        }

        inline long double spherical_j1(long double x) noexcept
        {
            if (x < 1.e-3L) {
                const long double x2 = x * x;
                return x * (1.L / 3.L - x2 / 30.L + x2 * x2 / 840.L - x2 * x2 * x2 / 45360.L);
            }
            return std::sin(x) / (x * x) - std::cos(x) / x;
        }

        // The power series is inexpensive and avoids cancellation in the
        // trigonometric definitions when the argument is small:
        //
        //   j_l(x) = x^l/(2l+1)!! sum_k (-x^2)^k /
        //            [2^k k! (2l+3)(2l+5)...(2l+2k+1)].
        inline long double spherical_j_series(int order, long double x) noexcept
        {
            long double prefactor = 1.L;
            for (int n = 1; n <= order; ++n)
                prefactor *= x / static_cast<long double>(2 * n + 1);

            long double term = 1.L;
            long double sum = term;
            const long double x2 = x * x;
            const long double tolerance = 4.L * std::numeric_limits<long double>::epsilon();
            for (int k = 1; k <= 128; ++k) {
                term *= -x2 /
                        (2.L * static_cast<long double>(k) * static_cast<long double>(2 * order + 2 * k + 1));
                sum += term;
                if (std::abs(term) <= tolerance * std::abs(sum))
                    break;
            }
            return prefactor * sum;
        }

        inline long double spherical_j_upward(int order, long double x) noexcept
        {
            long double previous = spherical_j0(x);
            long double current = spherical_j1(x);
            for (int n = 1; n < order; ++n) {
                const long double next = static_cast<long double>(2 * n + 1) * current / x - previous;
                previous = current;
                current = next;
            }
            return current;
        }

        // Upward recurrence follows the dominant y_l solution when l exceeds
        // x and is consequently unstable for j_l.  Miller's downward
        // recurrence follows the minimal j_l solution; normalising with j_0 or
        // j_1 removes its arbitrary starting scale.
        inline long double spherical_j_downward(int order, long double x) noexcept
        {
            const int extra = 64 + static_cast<int>(std::sqrt(40.L * static_cast<long double>(order + 1)));
            const int start = order + extra;
            constexpr long double rescale_threshold = 1.e100L;
            constexpr long double rescale_factor = 1.e-100L;

            long double next = 0.L;
            long double current = 1.L;
            long double target = 0.L;
            bool target_captured = false;

            for (int n = start; n >= 1; --n) {
                if (n == order) {
                    target = current;
                    target_captured = true;
                }

                const long double previous = static_cast<long double>(2 * n + 1) * current / x - next;
                next = current;
                current = previous;

                if (std::abs(current) > rescale_threshold) {
                    current *= rescale_factor;
                    next *= rescale_factor;
                    if (target_captured)
                        target *= rescale_factor;
                }
            }

            const long double j0 = spherical_j0(x);
            const long double j1 = spherical_j1(x);
            const long double normalisation =
                (std::abs(j0) >= std::abs(j1)) ? j0 / current : j1 / next;
            return target * normalisation;
        }
    } // namespace detail

    // These functions intentionally match the domain accepted by GSL's
    // gsl_sf_bessel_jl/yl calls that they replace: order must be nonnegative,
    // j_l accepts x == 0, and y_l requires x > 0.  Invalid or non-finite input
    // returns NaN, matching the GSL result when its error handler is disabled.
    inline double spherical_bessel_j(int order, double x) noexcept
    {
        if (order < 0 || x < 0. || !std::isfinite(x))
            return detail::quiet_nan;
        if (x == 0.)
            return order == 0 ? 1. : 0.;

        const long double argument = x;
        if (argument <= 1.L)
            return static_cast<double>(detail::spherical_j_series(order, argument));
        if (order == 0)
            return static_cast<double>(detail::spherical_j0(argument));
        if (order == 1)
            return static_cast<double>(detail::spherical_j1(argument));
        if (argument > static_cast<long double>(order))
            return static_cast<double>(detail::spherical_j_upward(order, argument));
        return static_cast<double>(detail::spherical_j_downward(order, argument));
    }

    inline double spherical_bessel_y(int order, double x) noexcept
    {
        if (order < 0 || x <= 0. || !std::isfinite(x))
            return detail::quiet_nan;

        const long double argument = x;
        long double previous = -std::cos(argument) / argument;
        if (order == 0)
            return static_cast<double>(previous);

        long double current = -std::cos(argument) / (argument * argument) - std::sin(argument) / argument;
        for (int n = 1; n < order; ++n) {
            const long double next = static_cast<long double>(2 * n + 1) * current / argument - previous;
            previous = current;
            current = next;
        }
        return static_cast<double>(current);
    }
} // namespace Kadath::special_functions

#endif // KADATH_UTILITIES_SPHERICAL_BESSEL_HPP
