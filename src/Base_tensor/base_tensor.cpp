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

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Base_tensor/base_tensor.hpp"

namespace Kadath
{
    // Constructors

    Base_tensor::Base_tensor(const Space& sp) : space(sp), basis(sp.get_nbr_domains())
    {
        basis = -1;
    }

    Base_tensor::Base_tensor(const Space& sp, int bb) : space(sp), basis(sp.get_nbr_domains())
    {
        basis = bb;
    }

    Base_tensor::Base_tensor(const Base_tensor& so) : space(so.space), basis(so.basis) {}

    Base_tensor::Base_tensor(const Space& sp, BinarySource& source) : space(sp), basis(source) {}

    // Destructor

    Base_tensor::~Base_tensor() {}

    // Accessors :

    int& Base_tensor::set_basis(int nd)
    {
        return basis.set(nd);
    }

    int Base_tensor::get_basis(int nd) const
    {
        return basis(nd);
    }

    void Base_tensor::operator=(const Base_tensor& so)
    {
        assert(&space == &so.get_space());
        for (int i = 0; i < basis.get_size(0); i++)
            basis.set(i) = so.basis(i);
    }

    void Base_tensor::save(BinarySink& sink) const
    {
        basis.save(sink);
    }

    ostream& operator<<(ostream& flux, const Base_tensor& so)
    {
        flux << "Tensorial basis" << endl;
        for (int d = 0; d < so.space.get_nbr_domains(); d++) {
            flux << "Domain " << d << " : ";
            switch (so.get_basis(d)) {
                case -1:
                    flux << "Undefined" << endl;
                    break;
                case CARTESIAN_BASIS:
                    flux << "Cartesian" << endl;
                    break;
                case SPHERICAL_BASIS:
                    flux << "Spherical" << endl;
                    break;
                case MTZ_BASIS:
                    flux << "MTZ" << endl;
                    break;
                default:
                    KADATH_THROW("Unknown tensorial basis");
            }
        }
        return flux;
    }

    bool operator==(const Base_tensor& b1, const Base_tensor& b2)
    {
        assert(&b1.space == &b2.space);
        int res = true;
        for (int i = 0; i < b1.space.get_nbr_domains(); i++)
            if (b1.get_basis(i) != b2.get_basis(i))
                res = false;
        return res;
    }

    bool operator!=(const Base_tensor& b1, const Base_tensor& b2)
    {
        return !(b1 == b2);
    }
} // namespace Kadath
