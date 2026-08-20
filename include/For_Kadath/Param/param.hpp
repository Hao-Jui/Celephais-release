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
 *   2026-08-06  RAII/span modernization.
 */

#pragma once

#include "For_Kadath/Array/memory.hpp"

#include <memory>

namespace Kadath
{
    /** Parameter storage.
     *
     *  This class is intended to store addresses of various Kadath objects to
     *  pass them as parameters in some subroutines.
     * \ingroup util
     */
    class Param : public MemoryMappable
    {

        // Data :
        // -----
      private:
        int n_int; ///< Number of \c int 's (integers).
        /// Array (size \c n_int ) of the \c int 's addresses.
        std::unique_ptr<int[]> p_int;

        int n_double; ///< Number of \c double 's (double precis. numbers).
        /// Array (size \c n_double ) of the \c double 's addresses.
        std::unique_ptr<double[]> p_double;

        // Constructors - Destructor
        // -------------------------

      public:
        Param(); ///< Default constructor is the only constructor

      private:
        /** Copy constructor (private and not implemented to make \c Param
         * a non-copyable class)
         */
        Param(const Param&);

      public:
        ~Param(); ///< Destructor

        // Assignment
        // -----------
      private:
        /** Assignment operator (private and not implemented to make
         *   \c Param  a non-copyable class)
         */
        void operator=(const Param&);

        // Addition/Extraction of one element
        // ----------------------------------
      public:
        /// Returns the number of stored \c int 's addresses.
        int get_n_int() const;

        /** Adds the address of a new \c int  to the list.
         *
         *  @param n [input] \c int  the address of which is
         *                             to be stored
         *  @param position [input] position of the \c int  in the list
         *			    of stored \c int  addresses (default
         *			    value = 0)
         *
         */
        void add_int(int n, int position = 0);

        /** Returns the reference of a \c int  stored in the list.
         *
         *  @param position [input] position of the \c int  in the list
         *			    of stored \c int  addresses (default
         *			    value = 0)
         *  @return Reference to the \c int  the address of
         *           which is stored at the location  \c position  in the
         *           list
         */
        const int& get_int(int position = 0) const;

        /// Returns the number of modifiable \c int 's addresses in the list.
        int get_n_int_mod() const;

        /// Returns the number of stored \c double 's addresses.
        int get_n_double() const;

        /** Adds the the address of a new \c double  to the list.
         *
         *  @param x [input] \c double  the address of which is
         *                             to be stored
         *  @param position [input] position of the \c double  in the list
         *			    of stored \c double  addresses (default
         *			    value = 0)
         *
         */
        void add_double(double x, int position = 0);

        /** Returns the reference of a \c double  stored in the list.
         *
         *  @param position [input] position of the \c double  in the list
         *			    of stored \c double  addresses (default
         *			    value = 0)
         *  @return Reference to the \c double  the address of
         *           which is stored at the location  \c position  in the
         *           list
         */
        const double& get_double(int position = 0) const;
    };
} // namespace Kadath
