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
 */

/*
 *  Definition of class Vector
 *
 */

#pragma once

#include "tensor.hpp"

namespace Kadath
{

    //-------------------------//
    //       class Vector      //
    //-------------------------//

    /**
     * A class derived from \c Tensor to deal specificaly with objects of valence 1 (and so also 1-forms).
     * \ingroup fields
     */
    class Vector : public Tensor
    {

      public:
        /** Standard constructor
         * @param sp : the \c Space.
         * @param tipe : the type tensor (COV vs CON).
         * @param ba : the tensorial basis used.
         */
        Vector(const Space& sp, int tipe, const Base_tensor& ba);

        Vector(const Vector& a);        ///< Copy constructor
        Vector(const Space& sp, BinarySource&); ///< Modern API: from BinarySource.

        /** Constructor from a \c Tensor .
         *  The \c Tensor  must be of valence one.
         */
        Vector(const Tensor& a);

        ~Vector() override; ///< Destructor

        // Mutators / assignment
        // ---------------------
      public:
        void operator=(const Vector&); ///< Assignment to another \c Vector.
        void operator=(const Tensor&) override;
        void operator=(double) override;
        void annule_hard() override;

        // Accessors
        // ---------
      public:
        using Tensor::set;
        Scalar& set(int);                    ///< Read/write access to a component
        const Scalar& operator()(int) const; ///< Readonly access to a component
        const Scalar& at(int) const;         ///< Readonly access to a component

        int position(const Array<int>& idx) const override
        {
            assert(idx.get_ndim() == 1);
            assert(idx.get_size(0) == 1);
            assert((idx(0) >= 1) && (idx(0) <= espace.get_ndim()));

            return (idx(0) - 1);
        };

        int position(const Index& idx) const override
        {
            assert((idx(0) >= 0) && (idx(0) < ndim));
            return (idx(0));
        }

        /**
         * Returns the type of the objects (CON or COV)
         */
        int get_index_type() const { return type_indice(0); };

        Array<int> indices(int place) const override
        {
            assert((place >= 0) && (place < espace.get_ndim()));

            Array<int> res(1);
            res = place + 1;
            return res;
        };
    };
} // namespace Kadath
