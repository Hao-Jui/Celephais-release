/*
    Copyright 2018 Philippe Grandclement

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
 *   2026-08-06  RAII/span modernization.
 */

#pragma once

#include "For_Kadath/Array/headcpp.hpp"

#include <memory>
#include <vector>

namespace Kadath
{
    /**
     * Class for storing a list of tensorial components.
     *
     * It simply consists of a list of integer Arrays. It is intended to be used when passing equations to a \c
     * System_of_eqs
     * \ingroup util
     **/

    class List_comp
    {
      protected:
        int ncomp;                                             ///< Number of stored components.
        int valence;                                           ///< Number of indices for each component.
        std::vector<std::unique_ptr<Array<int>>> owned_pcomp;  ///< Owned components.
        mutable std::vector<Array<int>*> pcomp;                 ///< Raw handoff cache for legacy interfaces.

      public:
        /** Standard constructor
         * @param nc [input] number of components.
         * @param val [input] valence of the components.
         **/
        List_comp(int nc, int val);
        List_comp(const List_comp&); ///< Constructor by copy.

        ~List_comp(); ///< Destructor

        /**
         * Read/write of one particular component.
         * @param i [input] which component.
         */
        Array<int>* set(int i);

        /**
         * Read/write of one particular component.
         * @param i [input] which component.
         */
        Array<int>* operator()(int i) const;

        /**
         * Returns the number of components.
         */
        int get_ncomp() const { return ncomp; };

        /**
         * Returns a pointer of the liste
         */
        Array<int>** get_pcomp() const { return pcomp.data(); };
    };

} // namespace Kadath
