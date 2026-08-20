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
 *   2026-08-10  Added capacity-3 inline coordinate storage with heap fallback.
 */

#pragma once

#include "headcpp.hpp"
#include "memory.hpp"
#include "For_Kadath/IO/binary_sink.hpp"
#include "For_Kadath/IO/binary_source.hpp"

namespace Kadath
{
    /**
     * The class \c Point is used to store the coordinates of a point.
     * \ingroup fields
     */

    class Point : public MemoryMappable
    {
      protected:
        static constexpr int inline_capacity = 3;

        double inline_coord[inline_capacity];
        int ndim;      ///< Number of dimensions.
        double* coord; ///< Array on the coordinates (mainly designed for absolute Cartesian coordinates).

        [[nodiscard]] static bool uses_inline_storage(int dim) noexcept
        {
            return dim >= 0 && dim <= inline_capacity;
        }

        [[nodiscard]] double* allocate_storage(int dim)
        {
            return uses_inline_storage(dim)
                       ? inline_coord
                       : MemoryMapper::get_memory<double>(dim);
        }

      public:
        /**
         * Standard constructor (the coordinates are not affected).
         * @param n [input] : number of dimensions.
         */
        explicit Point(int n);
        Point(BinarySource&); ///< Constructor from a BinarySource (modern API).
        Point(const Point&); ///< Constructor by copy.
        ~Point();            ///< Destuctor

        void save(BinarySink&) const; ///< Save via BinarySink (modern API).
        void operator=(const Point&); ///< Assignement to another \c Point
        double& set(int);             ///< Read/write of a coordinate
        double operator()(int) const; ///< Read only of a coordinate.
        /**
         * @returns the number of dimensions.
         */
        const int& get_ndim() const { return ndim; };

        friend ostream& operator<<(ostream&, const Point&); ///< Display
    };
} // namespace Kadath

#include "point.inl"
