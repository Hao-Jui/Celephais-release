/*
Copyright 2017 Philippe Grandclement

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

#pragma once

#include "assert.h"
#include "array.hpp"
#include "headcpp.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "dim_array.hpp"

/**
 * @brief Template implementations for Array mathematical operations and transformations
 * @author Philippe Grandclement
 * @date 2017
 *
 * This file provides comprehensive mathematical operations for the Array<T> template class,
 * including element-wise arithmetic, transcendental functions, and utility operations.
 */

namespace Kadath
{
    namespace
    {
        /**
         * @brief Apply a unary function to each element of an array
         * @tparam T Element type of the array
         * @tparam F Function type (typically a lambda or function pointer)
         * @param so Source array
         * @param func Unary function to apply to each element
         * @return New array with function applied to each element
         */
        template <typename T, typename F> Array<T> map_unary(const Array<T>& so, F func)
        {
            Array<T> res(so.dimensions);
            for (std::size_t i = 0; i < so.nbr; i++)
                res.data[i] = func(so.data[i]);
            return res;
        }

        /**
         * @brief Apply a binary function element-wise to two arrays
         * @tparam T Element type of the arrays
         * @tparam F Function type (typically a lambda or function pointer)
         * @param a First array operand
         * @param b Second array operand
         * @param func Binary function to apply element-wise
         * @return New array with function applied element-wise
         * @pre Arrays a and b must have identical dimensions
         */
        template <typename T, typename F> Array<T> map_binary(const Array<T>& a, const Array<T>& b, F func)
        {
            assert(a.dimensions == b.dimensions);
            Array<T> res(a.dimensions);
            for (std::size_t i = 0; i < a.nbr; i++)
                res.data[i] = func(a.data[i], b.data[i]);
            return res;
        }

        /**
         * @brief Apply a binary function with array on left and scalar on right
         * @tparam T Element type
         * @tparam F Function type
         * @param so Array operand
         * @param xx Scalar operand
         * @param func Binary function to apply
         * @return New array with function applied to each element and scalar
         */
        template <typename T, typename F> Array<T> map_scalar_right(const Array<T>& so, const T& xx, F func)
        {
            Array<T> res(so.dimensions);
            for (std::size_t i = 0; i < so.nbr; i++)
                res.data[i] = func(so.data[i], xx);
            return res;
        }

        /**
         * @brief Apply a binary function with scalar on left and array on right
         * @tparam T Element type
         * @tparam F Function type
         * @param xx Scalar operand
         * @param so Array operand
         * @param func Binary function to apply
         * @return New array with function applied to scalar and each element
         */
        template <typename T, typename F> Array<T> map_scalar_left(const T& xx, const Array<T>& so, F func)
        {
            Array<T> res(so.dimensions);
            for (std::size_t i = 0; i < so.nbr; i++)
                res.data[i] = func(xx, so.data[i]);
            return res;
        }
    } // namespace

    // ============================================================================
    // Member Functions
    // ============================================================================

    /**
     * @brief Check if array elements are in strictly increasing order
     * @tparam T Element type (must support comparison operators)
     * @return true if all elements satisfy data[i] > data[i-1], false otherwise
     * @pre Array must be one-dimensional
     */
    template <typename T> bool Array<T>::is_increasing() const
    {
        assert(get_ndim() == 1); // Ensures the array is 1D
        for (std::size_t i = 1; i < nbr; i++) {
            if (data[i] - data[i - 1] <= 0) {
                return false;
            }
        }
        return true;
    }

    // ============================================================================
    // Compound Assignment Operators
    // ============================================================================

    /** @brief Element-wise addition assignment */
    template <typename T> void Array<T>::operator+=(const Array<T>& so)
    {
        assert(dimensions == so.dimensions);
        for (std::size_t i = 0; i < nbr; ++i)
            data[i] += so.data[i];
    }

    /** @brief Element-wise subtraction assignment */
    template <typename T> void Array<T>::operator-=(const Array<T>& so)
    {
        assert(dimensions == so.dimensions);
        for (std::size_t i = 0; i < nbr; ++i)
            data[i] -= so.data[i];
    }

    /** @brief Element-wise multiplication assignment */
    template <typename T> void Array<T>::operator*=(const Array<T>& so)
    {
        assert(dimensions == so.dimensions);
        for (std::size_t i = 0; i < nbr; ++i)
            data[i] *= so.data[i];
    }

    /** @brief Element-wise division assignment */
    template <typename T> void Array<T>::operator/=(const Array<T>& so)
    {
        assert(dimensions == so.dimensions);
        for (std::size_t i = 0; i < nbr; ++i)
            data[i] /= so.data[i];
    }

    /** @brief Scalar addition assignment */
    template <typename T> void Array<T>::operator+=(const T& xx)
    {
        // Copy first because xx may refer to an element of this Array.
        const T value = xx;
        for (std::size_t i = 0; i < nbr; ++i)
            data[i] += value;
    }

    /** @brief Scalar subtraction assignment */
    template <typename T> void Array<T>::operator-=(const T& xx)
    {
        const T value = xx;
        for (std::size_t i = 0; i < nbr; ++i)
            data[i] -= value;
    }

    /** @brief Scalar multiplication assignment */
    template <typename T> void Array<T>::operator*=(const T& xx)
    {
        const T value = xx;
        for (std::size_t i = 0; i < nbr; ++i)
            data[i] *= value;
    }

    /** @brief Scalar division assignment */
    template <typename T> void Array<T>::operator/=(const T& xx)
    {
        const T value = xx;
        for (std::size_t i = 0; i < nbr; ++i)
            data[i] /= value;
    }

    // ============================================================================
    // Transcendental Functions
    // ============================================================================

    /** @brief Element-wise sine function */
    template <typename T> Array<T> sin(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return sin(x); });
    }

    /** @brief Element-wise cosine function */
    template <typename T> Array<T> cos(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return cos(x); });
    }

    /** @brief Element-wise hyperbolic sine function */
    template <typename T> Array<T> sinh(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return sinh(x); });
    }

    /** @brief Element-wise hyperbolic cosine function */
    template <typename T> Array<T> cosh(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return cosh(x); });
    }

    /** @brief Element-wise inverse hyperbolic tangent function */
    template <typename T> Array<T> atanh(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return atanh(x); });
    }

    /** @brief Element-wise arctangent function */
    template <typename T> Array<T> atan(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return atan(x); });
    }

    // ============================================================================
    // Unary Arithmetic Operators
    // ============================================================================

    /** @brief Unary plus operator (identity) */
    template <typename T> Array<T> operator+(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return x; });
    }

    /** @brief Unary minus operator (negation) */
    template <typename T> Array<T> operator-(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return -x; });
    }

    // ============================================================================
    // Binary Arithmetic Operators - Array + Array
    // ============================================================================

    /** @brief Element-wise addition of two arrays */
    template <typename T> Array<T> operator+(const Array<T>& a, const Array<T>& b)
    {
        return map_binary(a, b, [](const T& x, const T& y) { return x + y; });
    }

    /** @brief Element-wise subtraction of two arrays */
    template <typename T> Array<T> operator-(const Array<T>& a, const Array<T>& b)
    {
        return map_binary(a, b, [](const T& x, const T& y) { return x - y; });
    }

    /** @brief Element-wise multiplication of two arrays */
    template <typename T> Array<T> operator*(const Array<T>& a, const Array<T>& b)
    {
        return map_binary(a, b, [](const T& x, const T& y) { return x * y; });
    }

    /** @brief Element-wise division of two arrays */
    template <typename T> Array<T> operator/(const Array<T>& a, const Array<T>& b)
    {
        return map_binary(a, b, [](const T& x, const T& y) { return x / y; });
    }

    // ============================================================================
    // Binary Arithmetic Operators - Array + Scalar
    // ============================================================================

    /** @brief Add scalar to each array element */
    template <typename T> Array<T> operator+(const Array<T>& so, T xx)
    {
        return map_scalar_right(so, xx, [](const T& x, const T& y) { return x + y; });
    }

    /** @brief Add array to scalar (commutative) */
    template <typename T> Array<T> operator+(T xx, const Array<T>& so)
    {
        return map_scalar_left(xx, so, [](const T& x, const T& y) { return x + y; });
    }

    /** @brief Subtract scalar from each array element */
    template <typename T> Array<T> operator-(const Array<T>& so, T xx)
    {
        return map_scalar_right(so, xx, [](const T& x, const T& y) { return x - y; });
    }

    /** @brief Subtract array from scalar */
    template <typename T> Array<T> operator-(T xx, const Array<T>& so)
    {
        return map_scalar_left(xx, so, [](const T& x, const T& y) { return x - y; });
    }

    /** @brief Multiply each array element by scalar */
    template <typename T> Array<T> operator*(const Array<T>& so, T xx)
    {
        return map_scalar_right(so, xx, [](const T& x, const T& y) { return x * y; });
    }

    /** @brief Multiply scalar by array (commutative) */
    template <typename T> Array<T> operator*(T xx, const Array<T>& so)
    {
        return map_scalar_left(xx, so, [](const T& x, const T& y) { return x * y; });
    }

    /** @brief Divide each array element by scalar */
    template <typename T> Array<T> operator/(const Array<T>& so, T xx)
    {
        return map_scalar_right(so, xx, [](const T& x, const T& y) { return x / y; });
    }

    /** @brief Divide scalar by each array element */
    template <typename T> Array<T> operator/(T xx, const Array<T>& so)
    {
        return map_scalar_left(xx, so, [](const T& x, const T& y) { return x / y; });
    }

    // ============================================================================
    // Mathematical Functions
    // ============================================================================

    /** @brief Element-wise power function with integer exponent */
    template <typename T> Array<T> pow(const Array<T>& so, int n)
    {
        return map_unary(so, [n](const T& x) { return pow(x, n); });
    }

    /** @brief Element-wise power function with floating-point exponent */
    template <typename T> Array<T> pow(const Array<T>& so, double nn)
    {
        return map_unary(so, [nn](const T& x) { return pow(x, nn); });
    }

    /** @brief Element-wise square root function */
    template <typename T> Array<T> sqrt(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return sqrt(x); });
    }

    /** @brief Element-wise exponential function */
    template <typename T> Array<T> exp(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return exp(x); });
    }

    /** @brief Element-wise natural logarithm function */
    template <typename T> Array<T> log(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return log(x); });
    }

    /** @brief Element-wise absolute value function */
    template <typename T> Array<T> fabs(const Array<T>& so)
    {
        return map_unary(so, [](const T& x) { return fabs(x); });
    }

    // ============================================================================
    // Reduction Operations
    // ============================================================================

    /**
     * @brief Compute scalar (dot) product of two arrays
     * @param a First array
     * @param b Second array
     * @return Sum of element-wise products
     * @pre Arrays must have compatible dimensions
     */
    template <typename T> T scal(const Array<T>& a, const Array<T>& b)
    {
        T res = a.data[0] * b.data[0];
        for (int i = 1; i < a.get_size(0); i++)
            res += a.data[i] * b.data[i];
        return res;
    }

    /**
     * @brief Compute maximum absolute difference between two arrays
     * @param a First array
     * @param b Second array
     * @return Maximum of |a[i] - b[i]| over all elements
     * @pre Arrays must have identical dimensions
     */
    template <typename T> T diffmax(const Array<T>& a, const Array<T>& b)
    {
        assert(a.dimensions == b.dimensions);
        T res = 0;
        T diff;
        for (std::size_t i = 0; i < a.nbr; i++) {
            diff = fabs(a.data[i] - b.data[i]);
            if (diff > res)
                res = diff;
        }
        return res;
    }

    /**
     * @brief Find maximum element in array
     * @param so Input array
     * @return Maximum value in the array
     */
    template <typename T> T max(const Array<T>& so)
    {
        T res = so.data[0];
        for (std::size_t i = 0; i < so.nbr; i++)
            if (so.data[i] > res)
                res = so.data[i];
        return res;
    }

    /**
     * @brief Find minimum element in array
     * @param so Input array
     * @return Minimum value in the array
     */
    template <typename T> T min(const Array<T>& so)
    {
        T res = so.data[0];
        for (std::size_t i = 0; i < so.nbr; i++)
            if (so.data[i] < res)
                res = so.data[i];
        return res;
    }

    /**
     * @brief Compute sum of all array elements
     * @param so Input array
     * @return Sum of all elements
     */
    template <typename T> T sum(const Array<T>& so)
    {
        T res = 0;
        for (std::size_t i = 0; i < so.nbr; i++)
            res += so.data[i];
        return res;
    }

} // namespace Kadath
