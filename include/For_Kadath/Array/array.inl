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

/*
 * Modifications (Celephais):
 *   2026-08-10  Added capacity-2 inline payload ownership for Array<int>.
 */

#pragma once

#include "assert.h"
#include <algorithm>
#include "array.hpp"
#include "headcpp.hpp"
#include "For_Kadath/Utilities/utilities.hpp"
#include "dim_array.hpp"

/**
 * @file array.cpp
 * @brief Template implementation of the Array<T> class
 * @author Philippe Grandclement
 * @date 2017
 *
 * This file contains the implementation of constructors, destructors,
 * assignment operators, element access methods, and I/O operations
 * for the Array<T> template class.
 */

namespace Kadath
{

    // ============================================================================
    // Constructors
    // ============================================================================

    /**
     * @brief Construct array from dimension specification
     * @tparam T Element type
     * @param res Dimension array specifying the size in each dimension
     * @post Array storage is available with uninitialized elements
     */
    template <typename T> Array<T>::Array(const Dim_array& res) : dimensions(res)
    {
        nbr = 1;
        for (int i = 0; i < res.ndim; i++)
            nbr = checked_mul(nbr, static_cast<std::size_t>(res(i)));

        data = allocate_data(nbr);
    }

    /**
     * @brief Construct 1-dimensional array
     * @tparam T Element type
     * @param i Size of the array
     * @post Array is allocated with uninitialized elements
     */
    template <typename T> Array<T>::Array(int i) : dimensions(1)
    {
        dimensions.set(0) = i;
        nbr = static_cast<std::size_t>(i);
        data = allocate_data(nbr);
    }

    /**
     * @brief Construct 2-dimensional array
     * @tparam T Element type
     * @param i Size in first dimension
     * @param j Size in second dimension
     * @post Array is allocated with uninitialized elements
     */
    template <typename T> Array<T>::Array(int i, int j) : dimensions(2)
    {
        dimensions.set(0) = i;
        dimensions.set(1) = j;
        nbr = checked_mul(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        data = allocate_data(nbr);
    }

    /**
     * @brief Construct 3-dimensional array
     * @tparam T Element type
     * @param i Size in first dimension
     * @param j Size in second dimension
     * @param k Size in third dimension
     * @post Array is allocated with uninitialized elements
     */
    template <typename T> Array<T>::Array(int i, int j, int k) : dimensions(3)
    {
        dimensions.set(0) = i;
        dimensions.set(1) = j;
        dimensions.set(2) = k;
        nbr = checked_mul(checked_mul(static_cast<std::size_t>(i), static_cast<std::size_t>(j)),
                          static_cast<std::size_t>(k));
        data = allocate_data(nbr);
    }

    /**
     * @brief Copy constructor - creates deep copy
     * @tparam T Element type
     * @param so Source array to copy from
     * @post New array has same dimensions and element values as source
     */
    template <typename T> Array<T>::Array(const Array<T>& so) : dimensions(so.dimensions)
    {
        nbr = so.nbr;
        if constexpr (std::is_same_v<T, int>) {
            data = so.data == nullptr ? nullptr : allocate_data(nbr);
            if (nbr != 0U)
                std::memcpy(data, so.data, nbr * sizeof(T));
        } else {
            data = MemoryMapper::get_memory<T>(nbr);
            std::copy(so.data, so.data + nbr, data);
        }
        // Gate at the call site so the default-off path cannot resolve the
        // KernelContext TLV while entering the recorder.
        if (array_copy_profile_enabled())
            array_copy_record_ctor(nbr * sizeof(T));
    }

    // Move constructor. Heap-backed arrays transfer their pointer; inline
    // Array<int> values are copied into the destination-owned buffer. The source
    // is left in a valid moved-from state (data == nullptr, nbr == 0).
    // Dimensions are deep-copied because Dim_array has no move support.
    template <typename T> Array<T>::Array(Array<T>&& so) noexcept
        : dimensions(so.dimensions)
        , nbr(so.nbr)
        , data(nullptr)
    {
        if constexpr (std::is_same_v<T, int>) {
            if (so.uses_inline_storage()) {
                data = inline_storage();
                if (nbr != 0U)
                    std::memcpy(data, so.data, nbr * sizeof(T));
            } else {
                data = so.data;
            }
        } else {
            data = so.data;
        }
        so.nbr = 0;
        so.data = nullptr;
    }

    // ============================================================================
    // Destructor
    // ============================================================================

    /**
     * @brief Destructor - releases allocated memory
     * @tparam T Element type
     */
    template <typename T> Array<T>::~Array()
    {
        delete_data();
    }

    // ============================================================================
    // I/O Operations
    // ============================================================================

    template <typename T> Array<T>::Array(BinarySource& source) : dimensions(source)
    {
        const int nbr_raw = source.read<int>();
        nbr = static_cast<std::size_t>(nbr_raw);
        data = allocate_data(nbr);
        source.read_array(data, nbr);
    }

    template <typename T> void Array<T>::save(BinarySink& sink) const
    {
        dimensions.save(sink);
        sink.write<int>(static_cast<int>(nbr));
        sink.write_array(data, nbr);
    }

    // ============================================================================
    // Assignment Operators
    // ============================================================================

    /**
     * @brief Assignment operator - deep copy from another array
     * @tparam T Element type
     * @param so Source array to copy from
     * @pre Arrays must have identical dimensions
     * @post All elements are copied from source array
     */
    template <typename T> void Array<T>::operator=(const Array<T>& so)
    {
        assert(dimensions == so.dimensions);
        std::copy(so.data, so.data + nbr, data);
        // Phase-1 copy-traffic probe (paired with copy-ctor hook). Keep the
        // disabled path outside the recorder's TLS-backed context lookup.
        if (array_copy_profile_enabled())
            array_copy_record_assign(nbr * sizeof(T));
    }

    // Move assignment. Releases *this's existing data, then transfers heap
    // ownership or copies source inline values. Source is left empty. Self-move
    // is a no-op.
    // Unlike the const-ref operator=, this does NOT require matching
    // dimensions because the LHS adopts the RHS's shape entirely.
    template <typename T> Array<T>& Array<T>::operator=(Array<T>&& so) noexcept
    {
        if (this == &so)
            return *this;
        delete_data();
        dimensions = so.dimensions;
        nbr = so.nbr;
        if constexpr (std::is_same_v<T, int>) {
            if (so.uses_inline_storage()) {
                data = inline_storage();
                if (nbr != 0U)
                    std::memcpy(data, so.data, nbr * sizeof(T));
            } else {
                data = so.data;
            }
        } else {
            data = so.data;
        }
        so.nbr = 0;
        so.data = nullptr;
        return *this;
    }

    /**
     * @brief Assign scalar value to all elements
     * @tparam T Element type
     * @param xx Value to assign to all elements
     * @post All array elements are set to xx
     */
    template <typename T> void Array<T>::operator=(T xx)
    {
        std::fill(data, data + nbr, xx);
    }

    // ============================================================================
    // Element Access - Read/Write (non-const)
    // ============================================================================

    /**
     * @brief Access element at multi-dimensional index (read/write)
     * @tparam T Element type
     * @param point Index object specifying position in all dimensions
     * @return Reference to element at specified position
     * @pre point.sizes must match array dimensions
     * @pre All indices must be within valid range
     */
    template <typename T> T& Array<T>::set(const Index& point)
    {
        // Dimension-match and per-axis bounds validation are debug-only: they
        // are the hottest leaf in the Jacobian assembly (operator==(Dim_array)
        // per element access, ~12% of the first-J build) yet only catch
        // programmer error, never fire on correct code, and are never caught
        // for control flow. Gated on CELEPHAIS_ARRAY_BOUNDS_CHECK (defined in the
        // Debug build only, decoupled from NDEBUG/asserts) so optimized builds
        // do the lean offset computation the header already advertises ("No
        // bounds checking in release builds"). asan/ubsan still flags the
        // downstream out-of-bounds data[index] access without this check.
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
        if (!(point.sizes == dimensions))
            throw std::invalid_argument("Array::set(Index): index dimensions do not match array dimensions");
#endif
        int index = point(0);
        for (int i = 1; i < dimensions.ndim; i++) {
            index *= dimensions(i);
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
            if (point(i) < 0 || point(i) >= dimensions(i))
                throw std::out_of_range("Array::set(Index): index out of range");
#endif
            index += point(i);
        }
        return data[index];
    }

    /**
     * @brief Access element in 1D array (read/write)
     * @tparam T Element type
     * @param i Index in first dimension
     * @return Reference to element at position i
     * @pre Array must be 1-dimensional
     * @pre 0 <= i < dimensions(0)
     */
    template <typename T> T& Array<T>::set(int i)
    {
        // Debug-only validation (CELEPHAIS_ARRAY_BOUNDS_CHECK); see set(const Index&).
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
        if (dimensions.ndim != 1)
            throw std::invalid_argument("Array::set(int): called on non-1D array");
        if (i < 0 || i >= dimensions(0))
            throw std::out_of_range("Array::set(int): index out of range");
#endif
        return data[i];
    }

    /**
     * @brief Access element in 2D array (read/write)
     * @tparam T Element type
     * @param i Index in first dimension
     * @param j Index in second dimension
     * @return Reference to element at position (i, j)
     * @pre Array must be 2-dimensional
     * @pre Indices must be within valid range
     */
    template <typename T> T& Array<T>::set(int i, int j)
    {
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
        if (dimensions.ndim != 2)
            throw std::invalid_argument("Array::set(int,int): called on non-2D array");
        if (i < 0 || i >= dimensions(0))
            throw std::out_of_range("Array::set(int,int): first index out of range");
        if (j < 0 || j >= dimensions(1))
            throw std::out_of_range("Array::set(int,int): second index out of range");
#endif
        return data[i * dimensions(1) + j];
    }

    /**
     * @brief Access element in 3D array (read/write)
     * @tparam T Element type
     * @param i Index in first dimension
     * @param j Index in second dimension
     * @param k Index in third dimension
     * @return Reference to element at position (i, j, k)
     * @pre Array must be 3-dimensional
     * @pre All indices must be within valid range
     */
    template <typename T> T& Array<T>::set(int i, int j, int k)
    {
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
        if (dimensions.ndim != 3)
            throw std::invalid_argument("Array::set(int,int,int): called on non-3D array");
        if (i < 0 || i >= dimensions(0))
            throw std::out_of_range("Array::set(int,int,int): first index out of range");
        if (j < 0 || j >= dimensions(1))
            throw std::out_of_range("Array::set(int,int,int): second index out of range");
        if (k < 0 || k >= dimensions(2))
            throw std::out_of_range("Array::set(int,int,int): third index out of range");
#endif
        return data[i * dimensions(1) * dimensions(2) + j * dimensions(2) + k];
    }

    // ============================================================================
    // Element Access - Read-Only (const)
    // ============================================================================

    /**
     * @brief Access element at multi-dimensional index (read-only)
     * @tparam T Element type
     * @param point Index object specifying position in all dimensions
     * @return Copy of element at specified position
     * @pre point.sizes must match array dimensions
     * @pre All indices must be within valid range
     */
    template <typename T> T Array<T>::operator()(const Index& point) const
    {
        // Debug-only validation; see Array<T>::set(const Index&) above.
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
        if (!(point.sizes == dimensions))
            throw std::invalid_argument("Array::operator()(Index): index dimensions do not match array dimensions");
#endif
        int index = point(0);
        for (int i = 1; i < dimensions.ndim; i++) {
            index *= dimensions(i);
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
            if (point(i) < 0 || point(i) >= dimensions(i))
                throw std::out_of_range("Array::operator()(Index): index out of range");
#endif
            index += point(i);
        }
        return data[index];
    }

    /**
     * @brief Access element in 1D array (read-only)
     * @tparam T Element type
     * @param i Index in first dimension
     * @return Copy of element at position i
     * @pre Array must be 1-dimensional
     * @pre 0 <= i < dimensions(0)
     */
    template <typename T> T Array<T>::operator()(int i) const
    {
        // Debug-only validation (CELEPHAIS_ARRAY_BOUNDS_CHECK); see set(const Index&).
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
        if (dimensions.ndim != 1)
            throw std::invalid_argument("Array::operator()(int): called on non-1D array");
        if (i < 0 || i >= dimensions(0))
            throw std::out_of_range("Array::operator()(int): index out of range");
#endif
        return data[i];
    }

    /**
     * @brief Access element in 2D array (read-only)
     * @tparam T Element type
     * @param i Index in first dimension
     * @param j Index in second dimension
     * @return Copy of element at position (i, j)
     * @pre Array must be 2-dimensional
     * @pre Indices must be within valid range
     */
    template <typename T> T Array<T>::operator()(int i, int j) const
    {
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
        if (dimensions.ndim != 2)
            throw std::invalid_argument("Array::operator()(int,int): called on non-2D array");
        if (i < 0 || i >= dimensions(0))
            throw std::out_of_range("Array::operator()(int,int): first index out of range");
        if (j < 0 || j >= dimensions(1))
            throw std::out_of_range("Array::operator()(int,int): second index out of range");
#endif
        return data[i * dimensions(1) + j];
    }

    /**
     * @brief Access element in 3D array (read-only)
     * @tparam T Element type
     * @param i Index in first dimension
     * @param j Index in second dimension
     * @param k Index in third dimension
     * @return Copy of element at position (i, j, k)
     * @pre Array must be 3-dimensional
     * @pre All indices must be within valid range
     */
    template <typename T> T Array<T>::operator()(int i, int j, int k) const
    {
#ifdef CELEPHAIS_ARRAY_BOUNDS_CHECK
        if (dimensions.ndim != 3)
            throw std::invalid_argument("Array::operator()(int,int,int): called on non-3D array");
        if (i < 0 || i >= dimensions(0))
            throw std::out_of_range("Array::operator()(int,int,int): first index out of range");
        if (j < 0 || j >= dimensions(1))
            throw std::out_of_range("Array::operator()(int,int,int): second index out of range");
        if (k < 0 || k >= dimensions(2))
            throw std::out_of_range("Array::operator()(int,int,int): third index out of range");
#endif
        return data[i * dimensions(1) * dimensions(2) + j * dimensions(2) + k];
    }

    // ============================================================================
    // Memory Management
    // ============================================================================

    /**
     * @brief Release memory allocated for array data
     * @tparam T Element type
     * @post data pointer is set to nullptr
     * @note Safe to call multiple times
     */
    template <typename T> void Array<T>::delete_data()
    {
        if constexpr (std::is_same_v<T, int>) {
            if (data != nullptr && !uses_inline_storage())
                MemoryMapper::release_memory<T>(data, nbr);
        } else {
            if (data != nullptr)
                MemoryMapper::release_memory<T>(data, nbr);
        }
        data = nullptr;
    }

    // ============================================================================
    // Stream Operators
    // ============================================================================

    /**
     * @brief Stream insertion operator for formatted array output
     * @tparam T Element type (must support operator<<)
     * @param o Output stream
     * @param so Array to output
     * @return Reference to output stream for chaining
     *
     * Output format varies by dimensionality:
     * - 1D: Elements printed in a single line
     * - 2D: Matrix format with row labels
     * - 3D: Layered matrix format with depth labels
     * - Higher dimensions: Raw linear data output
     */
    template <typename T> ostream& operator<<(ostream& o, const Array<T>& so)
    {
        int ndim = so.dimensions.get_ndim();
        o << "Array of " << ndim << " dimension(s)" << endl;
        Index xx(so.get_dimensions());

        switch (ndim) {
            case 1:
                // 1D array: print all elements in a line
                for (int i = 0; i < so.dimensions(0); i++) {
                    xx.set(0) = i;
                    o << so(xx) << " ";
                }
                break;

            case 2:
                // 2D array: print as matrix with row labels
                for (int i = 0; i < so.dimensions(1); i++) {
                    o << i << " : ";
                    xx.set(1) = i;
                    for (int j = 0; j < so.dimensions(0); j++) {
                        xx.set(0) = j;
                        o << so(xx) << " ";
                    }
                    o << endl;
                }
                break;

            case 3:
                // 3D array: print as layers of matrices
                for (int i = 0; i < so.dimensions(2); i++) {
                    o << "      " << i << endl; // Layer label
                    xx.set(2) = i;
                    for (int j = 0; j < so.dimensions(1); j++) {
                        o << j << " : "; // Row label
                        xx.set(1) = j;
                        for (int k = 0; k < so.dimensions(0); k++) {
                            xx.set(0) = k;
                            o << so(xx) << " ";
                        }
                        o << endl;
                    }
                    o << endl;
                }
                break;

            default:
                // Higher dimensions: print raw linear data
                for (std::size_t i = 0; i < so.nbr; i++)
                    o << so.data[i] << " ";
                o << endl;
                break;
        }
        return o;
    }

} // namespace Kadath
