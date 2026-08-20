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
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Domain/critic.hpp"
#include "For_Kadath/Array/point.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"
namespace Kadath
{
    // Standard constructor
    Domain_critic_inner::Domain_critic_inner(int num, int ttype, const Dim_array& nbr, double xx)
        : Domain(num, ttype, nbr), p_X(nullptr), p_T(nullptr), xlim(xx)
    {
        do_coloc();
    }

    // Constructor by copy
    Domain_critic_inner::Domain_critic_inner(const Domain_critic_inner& so) : Domain(so)
    {
        p_X = (so.p_X != nullptr) ? new Val_domain(*so.p_X) : nullptr;
        p_T = (so.p_T != nullptr) ? new Val_domain(*so.p_T) : nullptr;
        xlim = so.xlim;
    }

    Domain_critic_inner::Domain_critic_inner(int num, BinarySource& source) : Domain(num, source)
    {
        p_X = nullptr;
        p_T = nullptr;
        xlim = source.read<double>();
        do_coloc();
    }

    // Destructor
    Domain_critic_inner::~Domain_critic_inner()
    {
        del_deriv();
    }

    void Domain_critic_inner::del_deriv() const
    {
        for (int l = 0; l < ndim; l++) {
            if (coloc[l] != nullptr)
                delete coloc[l];
            coloc[l] = nullptr;
        }
        if (p_X != nullptr)
            delete p_X;
        if (p_T != nullptr)
            delete p_T;
    }

    void Domain_critic_inner::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        sink.write<double>(xlim);
    }

    ostream& Domain_critic_inner::print(ostream& o) const
    {
        o << "Inner critic domain (cylindric)" << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << "X max   = " << xlim << endl;
        o << endl;
        return o;
    }

    double coloc_leg_parity(int, int);
    void Domain_critic_inner::do_coloc()
    {

        switch (type_base) {
            case CHEB_TYPE:
                nbr_coefs = nbr_points;
                nbr_coefs.set(1) += 2;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = xlim * sin(M_PI * i / (nbr_points(0) - 1) / 2.);
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI * j / nbr_points(1);
                break;
            case LEG_TYPE:
                nbr_coefs = nbr_points;
                nbr_coefs.set(1) += 2;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = xlim * coloc_leg_parity(i, nbr_points(0));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = M_PI * j / nbr_points(1);
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_critic_inner::do_coloc");
        }
    }

    void Domain_critic_inner::do_X() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        assert(p_X == nullptr);
        p_X = new Val_domain(this);
        p_X->allocate_conf();
        Index index(nbr_points);
        do
            p_X->set(index) = (*coloc[0])(index(0));
        while (index.inc());
    }

    void Domain_critic_inner::do_T() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        assert(p_T == nullptr);
        p_T = new Val_domain(this);
        p_T->allocate_conf();
        Index index(nbr_points);
        do
            p_T->set(index) = (*coloc[1])(index(1));
        while (index.inc());
    }

    Val_domain Domain_critic_inner::get_X() const
    {
        if (p_X == nullptr)
            do_X();
        return *p_X;
    }

    Val_domain Domain_critic_inner::get_T() const
    {
        if (p_T == nullptr)
            do_T();
        return *p_T;
    }

    // Computes the Cartesian coordinates
    void Domain_critic_inner::do_absol() const
    {
        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        for (int i = 0; i < 2; i++)
            assert(absol[i] == nullptr);
        for (int i = 0; i < 2; i++) {
            absol[i] = new Val_domain(this);
            absol[i]->allocate_conf();
        }
        Index index(nbr_points);

        do {
            absol[0]->set(index) = (*coloc[0])(index(0));
            absol[1]->set(index) = (*coloc[1])(index(1));
        } while (index.inc());
    }

    bool Domain_critic_inner::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 2);
        bool res = true;
        if ((xx(1) < -prec) || (xx(1) > xlim + prec))
            res = false;
        if ((xx(2) < -prec) || (xx(2) > M_PI + prec))
            res = false;
        return res;
    }

    const Point Domain_critic_inner::absol_to_num(const Point& so) const
    {

        Point res(2);
        res.set(1) = so(1) / xlim;
        res.set(2) = so(2);

        return res;
    }

    // standard base
    void Domain_critic_inner::set_cheb_base(Base_spectral& base) const
    {
        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);
        base.def = true;
        base.bases_1d[1]->set(0) = COSSIN_EVEN;
        for (int k = 0; k < nbr_coefs(1); k++)
            base.bases_1d[0]->set(k) = CHEB_EVEN;
    }

    // standard base
    void Domain_critic_inner::set_legendre_base(Base_spectral& base) const
    {
        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);
        base.def = true;
        base.bases_1d[1]->set(0) = COSSIN_EVEN;
        for (int k = 0; k < nbr_coefs(1); k++)
            base.bases_1d[0]->set(k) = LEG_EVEN;
    }

    // standard base
    void Domain_critic_inner::set_cheb_xodd_base(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);
        base.def = true;
        base.bases_1d[1]->set(0) = COSSIN_EVEN;
        for (int k = 0; k < nbr_coefs(1); k++)
            base.bases_1d[0]->set(k) = CHEB_ODD;
    }

    // standard base
    void Domain_critic_inner::set_legendre_xodd_base(Base_spectral& base) const
    {
        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);
        base.def = true;
        base.bases_1d[1]->set(0) = COSSIN_EVEN;
        for (int k = 0; k < nbr_coefs(1); k++)
            base.bases_1d[0]->set(k) = LEG_ODD;
    }

    // standard base
    void Domain_critic_inner::set_cheb_todd_base(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);
        base.def = true;
        base.bases_1d[1]->set(0) = COSSIN_ODD;
        for (int k = 0; k < nbr_coefs(1); k++)
            base.bases_1d[0]->set(k) = CHEB_EVEN;
    }

    // standard base
    void Domain_critic_inner::set_legendre_todd_base(Base_spectral& base) const
    {
        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);
        base.def = true;
        base.bases_1d[1]->set(0) = COSSIN_ODD;
        for (int k = 0; k < nbr_coefs(1); k++)
            base.bases_1d[0]->set(k) = LEG_EVEN;
    }
    // standard base
    void Domain_critic_inner::set_cheb_xodd_todd_base(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);

        base.allocate(nbr_coefs);
        base.def = true;
        base.bases_1d[1]->set(0) = COSSIN_ODD;
        for (int k = 0; k < nbr_coefs(1); k++)
            base.bases_1d[0]->set(k) = CHEB_ODD;
    }

    // standard base
    void Domain_critic_inner::set_legendre_xodd_todd_base(Base_spectral& base) const
    {
        assert(type_base == LEG_TYPE);

        base.allocate(nbr_coefs);
        base.def = true;
        base.bases_1d[1]->set(0) = COSSIN_ODD;
        for (int k = 0; k < nbr_coefs(1); k++)
            base.bases_1d[0]->set(k) = LEG_ODD;
    }

    // Rules for the multiplication of two basis.
    Base_spectral Domain_critic_inner::mult(const Base_spectral& a, const Base_spectral& b) const
    {

        assert(a.ndim == 2);
        assert(b.ndim == 2);

        Base_spectral res(2);
        bool res_def = true;

        if (!a.def)
            res_def = false;
        if (!b.def)
            res_def = false;

        if (res_def) {

            // Base in phi :
            res.bases_1d[1] = std::make_unique<Array<int>>(a.bases_1d[1]->get_dimensions());
            switch ((*a.bases_1d[1])(0)) {
                case COSSIN_EVEN:
                    switch ((*b.bases_1d[1])(0)) {
                        case COSSIN_EVEN:
                            res.bases_1d[1]->set(0) = COSSIN_EVEN;
                            break;
                        case COSSIN_ODD:
                            res.bases_1d[1]->set(0) = COSSIN_ODD;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case COSSIN_ODD:
                    switch ((*b.bases_1d[1])(0)) {
                        case COSSIN_EVEN:
                            res.bases_1d[1]->set(0) = COSSIN_ODD;
                            break;
                        case COSSIN_ODD:
                            res.bases_1d[1]->set(0) = COSSIN_EVEN;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                default:
                    res_def = false;
                    break;
            }

            res.bases_1d[0] = std::make_unique<Array<int>>(a.bases_1d[0]->get_dimensions());
            for (int k = 0; k < nbr_coefs(1); k++) {
                switch ((*a.bases_1d[0])(k)) {
                    case CHEB_EVEN:
                        switch ((*b.bases_1d[0])(k)) {
                            case CHEB_EVEN:
                                res.bases_1d[0]->set(k) = CHEB_EVEN;
                                break;
                            case CHEB_ODD:
                                res.bases_1d[0]->set(k) = CHEB_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG_EVEN:
                        switch ((*b.bases_1d[0])(k)) {
                            case LEG_EVEN:
                                res.bases_1d[0]->set(k) = LEG_EVEN;
                                break;
                            case LEG_ODD:
                                res.bases_1d[0]->set(k) = LEG_ODD;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case CHEB_ODD:
                        switch ((*b.bases_1d[0])(k)) {
                            case CHEB_EVEN:
                                res.bases_1d[0]->set(k) = CHEB_ODD;
                                break;
                            case CHEB_ODD:
                                res.bases_1d[0]->set(k) = CHEB_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG_ODD:
                        switch ((*b.bases_1d[0])(k)) {
                            case LEG_EVEN:
                                res.bases_1d[0]->set(k) = LEG_ODD;
                                break;
                            case LEG_ODD:
                                res.bases_1d[0]->set(k) = LEG_EVEN;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;

                    default:
                        res_def = false;
                        break;
                }
            }
        }

        if (!res_def)
            for (int dim = 0; dim < a.ndim; dim++)
                if (res.bases_1d[dim] != nullptr) {
                    res.bases_1d[dim].reset();
                    res.bases_1d[dim] = nullptr;
                }
        res.def = res_def;
        return res;
    }

    int Domain_critic_inner::give_place_var(char* p) const
    {
        int res = -1;
        if (strcmp(p, "X ") == 0)
            res = 0;
        if (strcmp(p, "T ") == 0)
            res = 1;
        return res;
    }
} // namespace Kadath
