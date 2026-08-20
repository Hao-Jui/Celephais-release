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
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 *   2026-08-10  Added capacity-3 inline storage with heap fallback.
 */

#pragma once

#include "headcpp.hpp"
#include "memory.hpp"
#include "For_Kadath/IO/binary_sink.hpp"
#include "For_Kadath/IO/binary_source.hpp"

#include <cstring>

namespace Kadath
{
    /**
     * Stores the dimensional extents for a multi-dimensional Array.
     *
     * This is a lightweight container holding the number of dimensions and the
     * size of each dimension. It is used by Array to describe shapes and indexing.
     *
     * \ingroup util
     */

    class Dim_array : public MemoryMappable
    {
      protected:
        static constexpr int inline_capacity = 3;

        int inline_nbr[inline_capacity];
        int ndim; ///< Number of dimensions.
        int* nbr; ///< Size of the Array in each dimension.

        [[nodiscard]] static bool uses_inline_storage(int dim) noexcept
        {
            return dim >= 0 && dim <= inline_capacity;
        }

        [[nodiscard]] int* allocate_storage(int dim)
        {
            return uses_inline_storage(dim)
                       ? inline_nbr
                       : MemoryMapper::get_memory<int>(dim);
        }

      public:
        /**
         * Standard constructor.
         * @param nd [input] Number of dimensions; sizes are uninitialized.
         */
        explicit Dim_array(int);
        Dim_array(const Dim_array&); ///< Copy constructor.
        Dim_array(BinarySource&);    ///< Constructor from a BinarySource.
        ~Dim_array();                ///< Destructor.

        /**
         * Read/write access to the size of a given dimension.
         * @param i [input] Dimension index (0-based).
         */
        int& set(int i);
        /**
         * Read-only access to the size of a given dimension.
         * @param i [input] Dimension index (0-based).
         */
        [[nodiscard]] int operator()(int i) const;
        /**
         * Returns the number of dimensions.
         */
        [[nodiscard]] int get_ndim() const { return ndim; };
        void operator=(const Dim_array&); ///< Assignment to another Dim_array.

        void save(BinarySink&) const; ///< Save via BinarySink (modern API).

        /**
         * Swap contents with another Dim_array (no-throw).
         * Outstanding element references are invalidated.
         */
        void swap(Dim_array& so) noexcept
        {
            if (this == &so)
                return;

            const bool this_inline = uses_inline_storage(ndim);
            const bool other_inline = uses_inline_storage(so.ndim);

            if (this_inline && other_inline) {
                int temporary[inline_capacity];
                const int this_ndim = ndim;
                const int other_ndim = so.ndim;
                std::memcpy(temporary, inline_nbr,
                            static_cast<std::size_t>(this_ndim) * sizeof(int));
                std::memcpy(inline_nbr, so.inline_nbr,
                            static_cast<std::size_t>(other_ndim) * sizeof(int));
                std::memcpy(so.inline_nbr, temporary,
                            static_cast<std::size_t>(this_ndim) * sizeof(int));
                std::swap(ndim, so.ndim);
                nbr = inline_nbr;
                so.nbr = so.inline_nbr;
                return;
            }

            if (!this_inline && !other_inline) {
                std::swap(ndim, so.ndim);
                std::swap(nbr, so.nbr);
                return;
            }

            if (!this_inline) {
                so.swap(*this);
                return;
            }

            int* const other_heap = so.nbr;
            const int other_ndim = so.ndim;
            const int this_ndim = ndim;
            std::memcpy(so.inline_nbr, inline_nbr,
                        static_cast<std::size_t>(this_ndim) * sizeof(int));
            ndim = other_ndim;
            nbr = other_heap;
            so.ndim = this_ndim;
            so.nbr = so.inline_nbr;
        }

        template <class> friend class Array;
        friend ostream& operator<<(ostream&, const Dim_array&);
        friend bool operator==(const Dim_array&, const Dim_array&);
        friend bool operator!=(const Dim_array&, const Dim_array&);
    };

    ostream& operator<<(ostream&, const Dim_array&);
    bool operator==(const Dim_array&, const Dim_array&);

} // namespace Kadath

#include "dim_array.inl"
