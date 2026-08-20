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

#include "For_Kadath/Tensor/vector.hpp"
#include "For_Kadath/Tensor/tensor.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    //--------------//
    // Constructors //
    //--------------//

    // Standard constructor
    // --------------------
    Vector::Vector(const Space& sp, int tipe, const Base_tensor& bb) : Tensor(sp, 1, tipe, bb) {}

    // Copy constructor
    // ----------------
    Vector::Vector(const Vector& source) : Tensor(source) {}

    // Constructor from a {\tt Tensor}.
    //--------------------------------
    Vector::Vector(const Tensor& uu) : Tensor(uu)
    {
        assert(valence == 1);
    }

    Vector::Vector(const Space& sp, BinarySource& source) : Tensor(sp, source)
    {
        assert(valence == 1);
    }

    //--------------//
    //  Destructor  //
    //--------------//
    Vector::~Vector() {}

    void Vector::operator=(const Vector& t)
    {

        assert(&espace == &t.espace);
        basis = t.basis;
        assert(t.type_indice(0) == type_indice(0));

        for (int i = 0; i < 3; i++) {
            *cmp[i] = *t.cmp[i];
        }
    }

    void Vector::operator=(const Tensor& t)
    {

        assert(t.valence == 1);

        assert(&espace == &t.espace);
        basis = t.basis;
        assert(t.type_indice(0) == type_indice(0));

        for (int i = 0; i < 3; i++) {
            *cmp[i] = *t.cmp[i];
        }
    }

    void Vector::operator=(double xx)
    {
        for (int i = 0; i < 3; i++) {
            *cmp[i] = xx;
        }
    }

    void Vector::annule_hard()
    {
        for (int i = 0; i < 3; i++)
            cmp[i]->annule_hard();
    }

    // Affectation d'une composante :
    Scalar& Vector::set(int index)
    {
        assert((index >= 1) && (index <= 3));
        return *cmp[index - 1];
    }

    const Scalar& Vector::operator()(int index) const
    {
        assert((index >= 1) && (index <= 3));
        return *cmp[index - 1];
    }

    const Scalar& Vector::at(int index) const
    {
        return operator()(index);
    }
} // namespace Kadath
