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

#include <sstream>
#include "For_Kadath/Space/space.hpp"
#include "For_Kadath/Scalar/scalar.hpp"
#include <assert.h>
namespace Kadath
{
    double eta_lim_chi(double chi, double rext, double a, double eta_c);
    double chi_lim_eta(double chi, double rext, double a, double chi_c);

    Space::Space() : nbr_domains(0), ndim(0), type_base(0), domains(nullptr) {}

    Space::~Space()
    {
        if (domains != nullptr) {
            for (int i = 0; i < nbr_domains; i++)
                delete domains[i];
            delete[] domains;
        }
    }

    ostream& operator<<(ostream& o, const Space& so)
    {
        o << "Space of " << so.nbr_domains << " domains" << endl;
        for (int i = 0; i < so.nbr_domains; i++) {
            o << "Domain " << i + 1 << endl;
            o << *so.domains[i] << endl;
            o << "********************" << endl;
        }
        switch (so.type_base) {
            case CHEB_TYPE:
                o << "Chebyshev polynomials are used." << endl;
                break;
            case LEG_TYPE:
                o << "Legendre polynomials are used." << endl;
                break;
            default:
                KADATH_THROW("Unknown type of base");
        }
        return o;
    }

    void Space::save(BinarySink&) const
    {
        cerr << "save not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Array<int> Space::get_indices_matching_non_std(int, int) const
    {
        cerr << "get_indices_matching_non_std not implemented for" << endl;
        std::ostringstream oss;
        oss << *this << endl;
        KADATH_THROW(oss.str());
    }

    Scalar Space::get_cart_field(int const cart) const
    {
        assert(cart <= ndim);
        Scalar cart_field(*this);
        cart_field.annule_hard();
        if (domains != nullptr) {
            for (auto dom = 0; dom < nbr_domains; ++dom)
                cart_field.set_domain(dom) = domains[dom]->get_cart(cart);
        }
        cart_field.std_base();
        return cart_field;
    }

    Scalar Space::get_cart_field_bound(int const cart, int const bound) const
    {
        assert(cart <= ndim);
        Scalar cart_field(*this);
        cart_field = 1.;

        if (domains != nullptr) {
            for (auto dom = 0; dom < nbr_domains; ++dom) {
                auto npts = domains[dom]->get_nbr_points();
                auto npts_r = npts(0);
                int bound_idx{0};
                switch (bound) {
                    case INNER_BC:
                        break;
                    case OUTER_BC:
                        bound_idx = npts_r - 1;
                        break;
                    default:
                        std::ostringstream oss;
                        oss << "Undefined bound " << bound << "\n";
                        KADATH_THROW(oss.str());
                }

                Index cart_pos(npts);
                do {
                    Index bound_pos(npts);
                    bound_pos.set(0) = bound_idx;
                    bound_pos.set(1) = cart_pos(1);
                    bound_pos.set(2) = cart_pos(2);
                    cart_field.set_domain(dom).set(cart_pos) = domains[dom]->get_cart(cart)(bound_pos);
                } while (cart_pos.inc());
            }
        }
        return cart_field;
    }
} // namespace Kadath
