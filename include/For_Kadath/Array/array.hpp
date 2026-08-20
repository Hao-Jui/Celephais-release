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
 *   2026-08-10  Added capacity-2 inline payload storage for Array<int> with
 *               pointer-preserving heap fallback.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "headcpp.hpp"
#include "index.hpp"
#include "dim_array.hpp"
#include "memory.hpp"
#include "array_iterator.hpp"

namespace Kadath
{

    namespace detail
    {
        template <typename T> struct ArrayInlineStorage {
        };

        template <> struct ArrayInlineStorage<int> {
            int values[2];
        };
    }

    // =============================================================================
    // Forward Declarations and Friend Function Templates
    // =============================================================================

    template <typename T> class Array;

    // Stream output
    template <typename T> ostream& operator<<(ostream&, const Array<T>&);

    // Trigonometric and hyperbolic functions
    template <typename T> Array<T> sin(const Array<T>&);
    template <typename T> Array<T> cos(const Array<T>&);
    template <typename T> Array<T> sinh(const Array<T>&);
    template <typename T> Array<T> cosh(const Array<T>&);

    // Unary operators
    template <typename T> Array<T> operator+(const Array<T>&);
    template <typename T> Array<T> operator-(const Array<T>&);

    // Binary arithmetic operators (Array-Array)
    template <typename T> Array<T> operator+(const Array<T>&, const Array<T>&);
    template <typename T> Array<T> operator-(const Array<T>&, const Array<T>&);
    template <typename T> Array<T> operator*(const Array<T>&, const Array<T>&);
    template <typename T> Array<T> operator/(const Array<T>&, const Array<T>&);

    // Binary arithmetic operators (Array-Scalar)
    template <typename T> Array<T> operator+(const Array<T>&, T);
    template <typename T> Array<T> operator-(const Array<T>&, T);
    template <typename T> Array<T> operator*(const Array<T>&, T);
    template <typename T> Array<T> operator/(const Array<T>&, T);

    // Binary arithmetic operators (Scalar-Array)
    template <typename T> Array<T> operator+(T, const Array<T>&);
    template <typename T> Array<T> operator-(T, const Array<T>&);
    template <typename T> Array<T> operator*(T, const Array<T>&);
    template <typename T> Array<T> operator/(T, const Array<T>&);

    // Mathematical functions
    template <typename T> Array<T> pow(const Array<T>&, int);
    template <typename T> Array<T> pow(const Array<T>&, double);
    template <typename T> Array<T> sqrt(const Array<T>&);
    template <typename T> Array<T> exp(const Array<T>&);
    template <typename T> Array<T> log(const Array<T>&);
    template <typename T> Array<T> atanh(const Array<T>&);
    template <typename T> Array<T> atan(const Array<T>&);
    template <typename T> Array<T> fabs(const Array<T>&);

    // Reductions and norms
    template <typename T> T scal(const Array<T>&, const Array<T>&);
    template <typename T> T diffmax(const Array<T>&, const Array<T>&);
    template <typename T> T max(const Array<T>&);
    template <typename T> T min(const Array<T>&);
    template <typename T> T sum(const Array<T>&);

    // =============================================================================
    // Array Template Class
    // =============================================================================

    /**
     * @class Array
     * @brief Multi-dimensional array template with element-wise operations
     *
     * @tparam T Element type (typically int or double)
     *
     * @details
     * Array is a general-purpose multi-dimensional array container designed for
     * numerical computations in spectral methods. Key features include:
     *
     * - **Multi-dimensional support**: Handles 1D, 2D, 3D, and higher-dimensional arrays
     * - **Element-wise operations**: Full suite of mathematical operations applied element-wise
     * - **Type flexibility**: Templated for any numeric type (int, double, complex, etc.)
     * - **Efficient indexing**: Direct memory access with convenient multi-index notation
     * - **Persistence**: Serialization/deserialization for checkpoint-restart
     *
     * ## Memory Layout
     *
     * Elements are stored in row-major (C-style) contiguous memory layout:
     * - For 2D array A(i,j): memory address = i * n_j + j
     * - For 3D array A(i,j,k): memory address = (i * n_j + j) * n_k + k
     *
     * This layout ensures:
     * - Cache-friendly access patterns for innermost loops
     * - Compatibility with external libraries (BLAS, LAPACK)
     * - Efficient slicing and striding operations
     *
     * ## Mathematical Operations
     *
     * All mathematical functions operate element-wise:
     * - Arithmetic: +, -, *, / (both Array-Array and Array-Scalar)
     * - Transcendental: sin, cos, exp, log, sqrt, pow
     * - Hyperbolic: sinh, cosh, atanh
     * - Reductions: sum, max, min, scalar product
     *
     * Example usage:
     * @code
     * Array<double> a(10, 10);     // 10×10 matrix
     * Array<double> b(10, 10);
     * a = 1.0;                     // Set all elements to 1.0
     * b = sin(a);                  // Element-wise sine
     * Array<double> c = a + 2*b;   // Linear combination
     * double norm = sqrt(scal(c, c));  // L2 norm
     * @endcode
     *
     * ## Type Requirements
     *
     * Template parameter T must support:
     * - Default construction
     * - Copy assignment
     * - Arithmetic operators (+, -, *, /)
     * - Standard mathematical functions (for mathematical operations)
     * - File I/O via operator<< (for save/load)
     *
     * @note This class does NOT perform bounds checking in release builds for performance.
     *       Use debug builds during development to catch index errors.
     *
     * @see Dim_array for dimension specification
     * @see Index for multi-dimensional indexing
     *
     * @ingroup util
     */
    template <typename T>
    class Array : public MemoryMappable, private detail::ArrayInlineStorage<T>
    {
        static_assert(std::is_arithmetic<T>::value, "Array<T>: T must be an arithmetic type (int, double, etc.)");

      private:
        static constexpr std::size_t inline_capacity = 2U;

        static std::size_t checked_mul(std::size_t a, std::size_t b)
        {
            if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
                throw std::overflow_error("Array size overflow");
            return a * b;
        }

        [[nodiscard]] static constexpr bool can_use_inline_storage(
            std::size_t count) noexcept
        {
            if constexpr (std::is_same_v<T, int>)
                return count != 0U && count <= inline_capacity;
            return false;
        }

        [[nodiscard]] T* inline_storage() noexcept
        {
            if constexpr (std::is_same_v<T, int>)
                return this->values;
            return nullptr;
        }

        [[nodiscard]] const T* inline_storage() const noexcept
        {
            if constexpr (std::is_same_v<T, int>)
                return this->values;
            return nullptr;
        }

        [[nodiscard]] bool uses_inline_storage() const noexcept
        {
            if constexpr (std::is_same_v<T, int>)
                return data == inline_storage();
            return false;
        }

        [[nodiscard]] T* allocate_data(std::size_t count)
        {
            if constexpr (std::is_same_v<T, int>) {
                if (can_use_inline_storage(count))
                    return inline_storage();
            }
            return MemoryMapper::get_memory<T>(count);
        }

      public:
        // =========================================================================
        // Public Data Members
        // =========================================================================

        /**
         * @brief Dimensional structure of the array
         * @details Stores number of dimensions and size along each axis
         */
        Dim_array dimensions;

        /**
         * @brief Total number of elements
         * @details Product of all dimension sizes: n_0 × n_1 × ... × n_d
         */
        std::size_t nbr;

        /**
         * @brief Pointer to contiguous data storage
         * @details Elements stored in row-major order for cache efficiency
         */
        T* data;

      public:
        // =========================================================================
        // Constructors and Destructor
        // =========================================================================

        /**
         * @brief Constructs array from dimension specification
         *
         * @param[in] dims  Dim_array specifying array shape
         *
         * @post Array storage is available but elements are uninitialized for performance
         * @note User must initialize elements before use to avoid undefined behavior
         *
         * @warning Elements contain garbage values after construction
         *
         * Example:
         * @code
         * Dim_array dims(3);
         * dims.set(0) = 10; dims.set(1) = 20; dims.set(2) = 5;
         * Array<double> arr(dims);  // 10×20×5 array (uninitialized)
         * arr = 0.0;                // Initialize to zero
         * @endcode
         */
        explicit Array(const Dim_array& dims);

        /**
         * @brief Constructs 1D array
         *
         * @param[in] i  Number of elements
         *
         * @post 1D array with i elements (uninitialized)
         * @note Equivalent to std::vector<T>(i) but without initialization
         */
        explicit Array(int i);

        /**
         * @brief Constructs 2D array (matrix)
         *
         * @param[in] i  Number of rows
         * @param[in] j  Number of columns
         *
         * @post 2D array with i×j elements in row-major order (uninitialized)
         * @note Element (i,j) located at memory offset: i * n_cols + j
         */
        explicit Array(int i, int j);

        /**
         * @brief Constructs 3D array
         *
         * @param[in] i  Size of first dimension
         * @param[in] j  Size of second dimension
         * @param[in] k  Size of third dimension
         *
         * @post 3D array with i×j×k elements (uninitialized)
         * @note Element (i,j,k) at offset: (i * n_j + j) * n_k + k
         */
        explicit Array(int i, int j, int k);

        /**
         * @brief Construct array from a BinarySource (modern API).
         */
        Array(BinarySource& source);

        /**
         * @brief Copy constructor - performs deep copy
         *
         * @param[in] source  Array to copy
         *
         * @post Independent copy with separately owned storage
         * @note All elements copied, not just memory pointer (true deep copy)
         */
        Array(const Array<T>& source);

        /**
         * @brief Move constructor - takes ownership of source's data
         *
         * @param[in,out] source  Array whose heap storage is transferred, or
         *                        whose inline elements are copied; left in a valid
         *                        moved-from state with data == nullptr and
         *                        nbr == 0. Dimensions are copied from source
         *                        because Dim_array has no move support
         *                        (the int* it owns is small, so this is
         *                        negligible).
         *
         * @post New array owns the values formerly held by source; source is empty.
         *       Heap-backed moves preserve the payload pointer, while inline
         *       moves rebase it into the destination object.
         * @note Eliminates the deep copy that would otherwise happen on
         *       `new Array<T>(rvalue_temp)`. Targeted by the
         *       array-copy-profile-bns evidence bundle: ~669M copy ctors
         *       per FirstJ assembly were measured; move-construction
         *       removes that traffic at sites that hand off ownership.
         */
        Array(Array<T>&& source) noexcept;

        /**
         * @brief Destructor - releases owned data storage
         *
         * @post All memory freed, pointers invalidated
         * @note Safe to call on default-constructed or moved-from objects
         *       (delete_data is null-safe on data == nullptr).
         */
        ~Array();

      public:
        // =========================================================================
        // Memory Management
        // =========================================================================

        /**
         * @brief Logical destructor - deallocates data without destroying object
         *
         * @post data pointer set to nullptr; the existing nbr metadata is retained
         * @note Object remains safe to destroy or replace through move assignment
         * @warning Do not access elements after calling this method
         *
         * Use case: Clearing large arrays to free memory while keeping object alive
         */
        void delete_data();

        /**
         * Swap contents with another Array of the same type (no-throw).
         * Outstanding element pointers and references are invalidated.
         */
        void swap(Array<T>& so) noexcept
        {
            if constexpr (!std::is_same_v<T, int>) {
                dimensions.swap(so.dimensions);
                std::swap(nbr, so.nbr);
                std::swap(data, so.data);
            } else {
                if (this == &so)
                    return;

                const bool this_inline = uses_inline_storage();
                const bool other_inline = so.uses_inline_storage();

                if (this_inline && other_inline) {
                    int temporary[inline_capacity];
                    const std::size_t this_nbr = nbr;
                    const std::size_t other_nbr = so.nbr;
                    std::memcpy(temporary, data, this_nbr * sizeof(int));
                    std::memcpy(inline_storage(), so.data,
                                other_nbr * sizeof(int));
                    std::memcpy(so.inline_storage(), temporary,
                                this_nbr * sizeof(int));
                    dimensions.swap(so.dimensions);
                    nbr = other_nbr;
                    so.nbr = this_nbr;
                    data = inline_storage();
                    so.data = so.inline_storage();
                    return;
                }

                if (!this_inline && !other_inline) {
                    dimensions.swap(so.dimensions);
                    std::swap(nbr, so.nbr);
                    std::swap(data, so.data);
                    return;
                }

                if (!this_inline) {
                    so.swap(*this);
                    return;
                }

                int* const other_data = so.data;
                const std::size_t this_nbr = nbr;
                const std::size_t other_nbr = so.nbr;
                std::memcpy(so.inline_storage(), data,
                            this_nbr * sizeof(int));
                dimensions.swap(so.dimensions);
                nbr = other_nbr;
                data = other_data;
                so.nbr = this_nbr;
                so.data = so.inline_storage();
            }
        }

        /**
         * @brief Save array to a BinarySink (modern API).
         */
        void save(BinarySink& sink) const;

      public:
        // =========================================================================
        // Assignment Operators
        // =========================================================================

        /**
         * @brief Assigns from another array (deep copy)
         *
         * @param[in] source  Array to copy from
         *
         * @pre Arrays must have identical dimensions
         * @post All elements copied from source
         *
         * @throws std::invalid_argument if dimensions don't match
         * @note This is assignment, not construction - object must exist
         */
        void operator=(const Array<T>& source);

        /**
         * @brief Move assignment - takes ownership of source's values
         *
         * @param[in,out] source  Array whose heap storage is transferred, or
         *                        whose inline elements are copied; left in a
         *                        valid moved-from state.
         *
         * @return Reference to *this for chaining.
         * @post *this owns source's data; source is empty (data == nullptr,
         *       nbr == 0). Self-move is a no-op.
         * @note Releases *this's existing data via delete_data() before
         *       taking source's pointer. Unlike the const-ref operator=,
         *       move assignment does NOT require matching dimensions; the
         *       LHS adopts the RHS's shape entirely.
         */
        Array<T>& operator=(Array<T>&& source) noexcept;

        /**
         * @brief Assigns same value to all elements (broadcast)
         *
         * @param[in] value  Scalar value to assign
         *
         * @post All elements set to value
         * @note Efficient way to initialize: arr = 0.0, arr = 1.0, etc.
         *
         * Example:
         * @code
         * Array<double> matrix(100, 100);
         * matrix = 0.0;  // Initialize all 10,000 elements to zero
         * @endcode
         */
        void operator=(T value);

      public:
        // =========================================================================
        // Element Access - Read/Write (set methods)
        // =========================================================================

        /**
         * @brief Accesses element by multi-index (read/write)
         *
         * @param[in] pos  Index object specifying position
         * @return Reference to element at position pos
         *
         * @warning No bounds checking in release builds
         * @note Use for modifying elements: arr.set(idx) = 3.14
         */
        T& set(const Index& pos);

        /// Direct flat-index access via Array_iterator (read/write) — O(1), no index recomputation.
        T& set(const Array_iterator& pos) { return data[pos.position]; }

        /**
         * @brief Accesses element in 1D array (read/write)
         *
         * @param[in] i  Position index (0 ≤ i < size)
         * @return Reference to i-th element
         *
         * @pre Array must be 1D
         * @warning Undefined behavior if i out of range or array not 1D
         */
        T& set(int i);

        /**
         * @brief Accesses element in 2D array (read/write)
         *
         * @param[in] i  Row index
         * @param[in] j  Column index
         * @return Reference to element at (i, j)
         *
         * @pre Array must be 2D
         * @note Memory offset: i * n_cols + j
         */
        T& set(int i, int j);

        /**
         * @brief Accesses element in 3D array (read/write)
         *
         * @param[in] i  First index
         * @param[in] j  Second index
         * @param[in] k  Third index
         * @return Reference to element at (i, j, k)
         *
         * @pre Array must be 3D
         * @note Memory offset: (i * n_j + j) * n_k + k
         */
        T& set(int i, int j, int k);

      public:
        // =========================================================================
        // Element Access - Read Only (operator() methods)
        // =========================================================================

        /**
         * @brief Accesses element by multi-index (read-only)
         *
         * @param[in] pos  Index object specifying position
         * @return Copy of element at position pos
         *
         * @note Returns by value, not reference - cannot be modified
         * @warning No bounds checking in release builds
         */
        T operator()(const Index& pos) const;

        /// Direct flat-index read via Array_iterator (read-only) — O(1), no index recomputation.
        T operator()(const Array_iterator& pos) const { return data[pos.position]; }

        /**
         * @brief Accesses element in 1D array (read-only)
         *
         * @param[in] i  Position index
         * @return Copy of i-th element
         *
         * @pre Array must be 1D
         */
        T operator()(int i) const;

        /**
         * @brief Accesses element in 2D array (read-only)
         *
         * @param[in] i  Row index
         * @param[in] j  Column index
         * @return Copy of element at (i, j)
         *
         * @pre Array must be 2D
         */
        T operator()(int i, int j) const;

        /**
         * @brief Accesses element in 3D array (read-only)
         *
         * @param[in] i  First index
         * @param[in] j  Second index
         * @param[in] k  Third index
         * @return Copy of element at (i, j, k)
         *
         * @pre Array must be 3D
         */
        T operator()(int i, int j, int k) const;

      public:
        // =========================================================================
        // Direct Data Access
        // =========================================================================

        /**
         * @brief Returns const pointer to raw data array
         *
         * @return Const pointer to first element
         *
         * @note Use for interfacing with C libraries or performance-critical loops
         * @warning Pointer invalidated if array is modified or destroyed
         *
         * Example:
         * @code
         * Array<double> arr(1000);
         * const double* ptr = arr.get_data();
         * // Pass to BLAS/LAPACK routine
         * cblas_daxpy(1000, 2.0, ptr, 1, result_ptr, 1);
         * @endcode
         */
        const T* get_data() const { return data; }

        /// Mutable raw data pointer — for hot loops that walk a known flat
        /// stride directly (e.g. the spectral transform drivers in src/Coef,
        /// src/Ope_1d) instead of recomputing a multi-dimensional offset per
        /// element. Same contiguous storage as the const overload.
        T* get_data() { return data; }

        /**
         * @brief Returns non-const pointer to raw data array
         *
         * @return Pointer to first element (modifiable)
         *
         * @note Allows direct memory manipulation for expert users
         * @warning Bypasses all safety checks - use with extreme caution
         */
        T* set_data() { return data; }

      public:
        // =========================================================================
        // Dimension Queries
        // =========================================================================

        /**
         * @brief Returns number of dimensions
         * @return Dimension count (1, 2, 3, or higher)
         */
        int get_ndim() const { return dimensions.get_ndim(); }

        /**
         * @brief Returns total number of elements
         * @return Product of all dimension sizes
         * @note Equivalent to size() in std::vector
         */
        std::size_t get_nbr() const { return nbr; }

        /**
         * @brief Returns size along specific dimension
         *
         * @param[in] i  Dimension index (0-based)
         * @return Number of elements along dimension i
         *
         * @pre 0 ≤ i < get_ndim()
         *
         * Example:
         * @code
         * Array<double> matrix(10, 20);
         * int nrows = matrix.get_size(0);  // Returns 10
         * int ncols = matrix.get_size(1);  // Returns 20
         * @endcode
         */
        int get_size(int i) const { return dimensions(i); }

        /**
         * @brief Returns dimensional structure
         * @return Const reference to Dim_array object
         * @note Useful for creating arrays with matching dimensions
         */
        const Dim_array& get_dimensions() const { return dimensions; }

      public:
        // =========================================================================
        // Array Properties
        // =========================================================================

        /**
         * @brief Checks if 1D array is monotonically increasing
         *
         * @return true if arr[i] < arr[i+1] for all valid i
         * @return false otherwise
         *
         * @pre Array must be 1D
         * @note Returns true for arrays with 0 or 1 elements
         *
         * Use case: Validating sorted coordinate arrays or grid points
         */
        bool is_increasing() const;

      public:
        // =========================================================================
        // Compound Assignment Operators
        // =========================================================================

        /**
         * @brief Element-wise addition assignment
         * @param[in] other  Array to add
         * @pre Arrays must have identical dimensions
         * @post this[i] = this[i] + other[i] for all i
         */
        void operator+=(const Array<T>& other);

        /**
         * @brief Element-wise subtraction assignment
         * @param[in] other  Array to subtract
         * @pre Arrays must have identical dimensions
         * @post this[i] = this[i] - other[i] for all i
         */
        void operator-=(const Array<T>& other);

        /**
         * @brief Element-wise multiplication assignment
         * @param[in] other  Array to multiply by
         * @pre Arrays must have identical dimensions
         * @post this[i] = this[i] * other[i] for all i
         * @note This is element-wise product, not matrix multiplication
         */
        void operator*=(const Array<T>& other);

        /**
         * @brief Element-wise division assignment
         * @param[in] other  Array to divide by
         * @pre Arrays must have identical dimensions
         * @post this[i] = this[i] / other[i] for all i
         * @warning No check for division by zero
         */
        void operator/=(const Array<T>& other);

        /**
         * @brief Scalar addition assignment
         * @param[in] scalar  Value to add to all elements
         * @post this[i] = this[i] + scalar for all i
         */
        void operator+=(const T& scalar);

        /**
         * @brief Scalar subtraction assignment
         * @param[in] scalar  Value to subtract from all elements
         * @post this[i] = this[i] - scalar for all i
         */
        void operator-=(const T& scalar);

        /**
         * @brief Scalar multiplication assignment
         * @param[in] scalar  Value to multiply all elements by
         * @post this[i] = this[i] * scalar for all i
         */
        void operator*=(const T& scalar);

        /**
         * @brief Scalar division assignment
         * @param[in] scalar  Value to divide all elements by
         * @post this[i] = this[i] / scalar for all i
         * @warning No check for division by zero
         */
        void operator/=(const T& scalar);

        // =========================================================================
        // Friend Declarations
        // =========================================================================

        friend class Matrice;
        friend class Array_iterator;

        // Stream I/O
        friend ostream& operator<< <>(ostream&, const Array<T>&);

        // Transcendental functions
        friend Array<T> sin<>(const Array<T>&);
        friend Array<T> cos<>(const Array<T>&);
        friend Array<T> sinh<>(const Array<T>&);
        friend Array<T> cosh<>(const Array<T>&);

        // Unary operators
        friend Array<T> operator+ <>(const Array<T>&);
        friend Array<T> operator- <>(const Array<T>&);

        // Binary operators (Array-Array)
        friend Array<T> operator+ <>(const Array<T>&, const Array<T>&);
        friend Array<T> operator- <>(const Array<T>&, const Array<T>&);
        friend Array<T> operator* <>(const Array<T>&, const Array<T>&);
        friend Array<T> operator/ <>(const Array<T>&, const Array<T>&);

        // Binary operators (Array-Scalar)
        friend Array<T> operator+ <>(const Array<T>&, T);
        friend Array<T> operator- <>(const Array<T>&, T);
        friend Array<T> operator* <>(const Array<T>&, T);
        friend Array<T> operator/ <>(const Array<T>&, T);

        // Binary operators (Scalar-Array)
        friend Array<T> operator+ <>(T, const Array<T>&);
        friend Array<T> operator- <>(T, const Array<T>&);
        friend Array<T> operator* <>(T, const Array<T>&);
        friend Array<T> operator/ <>(T, const Array<T>&);

        // Mathematical functions
        friend Array<T> pow<>(const Array<T>&, int);
        friend Array<T> pow<>(const Array<T>&, double);
        friend Array<T> sqrt<>(const Array<T>&);
        friend Array<T> exp<>(const Array<T>&);
        friend Array<T> log<>(const Array<T>&);
        friend Array<T> atanh<>(const Array<T>&);
        friend Array<T> atan<>(const Array<T>&);
        friend Array<T> fabs<>(const Array<T>&);

        // Reduction operations
        friend T scal<>(const Array<T>&, const Array<T>&);
        friend T diffmax<>(const Array<T>&, const Array<T>&);
        friend T max<>(const Array<T>&);
        friend T min<>(const Array<T>&);
        friend T sum<>(const Array<T>&);
    };

    static_assert(sizeof(Array<int>) == 48U,
                  "Array<int> public layout must remain 48 bytes");
    static_assert(sizeof(Array<double>) == 48U,
                  "Array<double> public layout must remain 48 bytes");

} // namespace Kadath

// =============================================================================
// Template Implementation Includes
// =============================================================================

#include "array.inl"
#include "array_math.inl"
