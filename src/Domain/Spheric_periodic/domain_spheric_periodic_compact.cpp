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
#include "For_Kadath/Domain/spheric_periodic.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

namespace Kadath
{
    void coef_1d(int, Array<double>&);
    void coef_i_1d(int, Array<double>&);
    int der_1d(int, Array<double>&);

    // Standard constructor
    Domain_spheric_periodic_compact::Domain_spheric_periodic_compact(int num, int ttype, int typet, double rint,
                                                                     double oome, const Dim_array& nbr)
        : Domain(num, ttype, nbr), alpha(-0.5 / rint), ome(oome), type_time(typet)
    {
        assert(nbr.get_ndim() == 2);
        switch (type_time) {
            case TO_PI:
                maxt = M_PI;
                break;
            case TO_PI_OVER_2:
                maxt = M_PI / 2.;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        do_coloc();
    }

    // Constructor by copy
    Domain_spheric_periodic_compact::Domain_spheric_periodic_compact(const Domain_spheric_periodic_compact& so)
        : Domain(so), alpha(so.alpha), ome(so.ome), type_time(so.type_time), maxt(so.maxt)
    {
    }

    Domain_spheric_periodic_compact::Domain_spheric_periodic_compact(int num, BinarySource& source)
        : Domain(num, source)
    {
        alpha = source.read<double>();
        ome = source.read<double>();
        type_time = source.read<int>();
        switch (type_time) {
            case TO_PI:
                maxt = M_PI;
                break;
            case TO_PI_OVER_2:
                maxt = M_PI / 2.;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        do_coloc();
    }

    // Destructor
    Domain_spheric_periodic_compact::~Domain_spheric_periodic_compact() {}

    void Domain_spheric_periodic_compact::save(BinarySink& sink) const
    {
        nbr_points.save(sink);
        nbr_coefs.save(sink);
        sink.write<int>(ndim);
        sink.write<int>(type_base);
        sink.write<double>(alpha);
        sink.write<double>(ome);
        sink.write<int>(type_time);
    }

    ostream& Domain_spheric_periodic_compact::print(ostream& o) const
    {
        o << "Spheric-periodic compact" << endl;
        o << "Rint    = " << -0.5 / alpha << endl;
        o << "time goes to " << maxt << endl;
        o << "Omega   = " << ome << endl;
        o << "Nbr pts = " << nbr_points << endl;
        o << endl;
        return o;
    }

    Val_domain Domain_spheric_periodic_compact::der_normal(const Val_domain& so, int bound) const
    {

        Val_domain res(so.der_var(1));
        switch (bound) {
            case OUTER_BC:
                res = res.mult_xm1();
                res = res.mult_xm1();
                res *= -alpha;
                break;
            case INNER_BC:
                res = res.mult_xm1();
                res = res.mult_xm1();
                res *= -alpha;
                break;
            default:
                KADATH_THROW("Unknown boundary case in Domain_spheric_periodic_compact::der_normal");
        }
        return res;
    }

    // Computes the cartesian coordinates
    void Domain_spheric_periodic_compact::do_absol() const
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
            absol[0]->set(index) = 1. / alpha / ((*coloc[0])(index(0)) - 1);
            absol[1]->set(index) = (*coloc[1])(index(1)) / ome;
        } while (index.inc());
    }

    // Computes the radius
    void Domain_spheric_periodic_compact::do_radius() const
    {

        for (int i = 0; i < 2; i++)
            assert(coloc[i] != nullptr);
        assert(radius == nullptr);
        radius = new Val_domain(this);
        radius->allocate_conf();
        Index index(nbr_points);
        do
            radius->set(index) = 1. / alpha / ((*coloc[0])(index(0)) - 1);
        while (index.inc());
    }

    // Is a point inside this domain ?
    bool Domain_spheric_periodic_compact::is_in(const Point& xx, double prec) const
    {

        assert(xx.get_ndim() == 2);

        bool res = (xx(1) >= -0.5 / alpha + prec) ? true : false;
        if ((xx(2) < 0 - prec) || (xx(2) > maxt / ome + prec))
            res = false;
        return res;
    }

    // Convert absolute coordinates to numerical ones
    const Point Domain_spheric_periodic_compact::absol_to_num(const Point& abs) const
    {

        assert(is_in(abs));
        Point num(2);

        num.set(1) = 1 + 1. / abs(1) / alpha;
        num.set(2) = abs(2) * ome;

        return num;
    }

    double coloc_leg(int, int);
    void Domain_spheric_periodic_compact::do_coloc()
    {

        switch (type_base) {
            case CHEB_TYPE:
                nbr_coefs = nbr_points;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = -cos(M_PI * i / (nbr_points(0) - 1));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = maxt * j / (nbr_points(1) - 1);
                break;
            case LEG_TYPE:
                nbr_coefs = nbr_points;
                del_deriv();
                for (int i = 0; i < ndim; i++)
                    coloc[i] = new Array<double>(nbr_points(i));
                for (int i = 0; i < nbr_points(0); i++)
                    coloc[0]->set(i) = coloc_leg(i, nbr_points(0));
                for (int j = 0; j < nbr_points(1); j++)
                    coloc[1]->set(j) = maxt * j / (nbr_points(1) - 1);
                break;
            default:
                KADATH_THROW("Unknown type of basis in Domain_spheric_periodic_compact::do_coloc");
        }
    }

    // Base for a function symetric in z, using Chebyshev
    void Domain_spheric_periodic_compact::set_cheb_base(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        switch (type_time) {
            case TO_PI:
                base.bases_1d[1]->set(0) = COS;
                break;
            case TO_PI_OVER_2:
                base.bases_1d[1]->set(0) = COS_EVEN;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = CHEB;
    }
    void Domain_spheric_periodic_compact::set_cheb_base_odd(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        switch (type_time) {
            case TO_PI_OVER_2:
                base.bases_1d[1]->set(0) = COS_ODD;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = CHEB;
    }
    void Domain_spheric_periodic_compact::set_anti_cheb_base(Base_spectral& base) const
    {

        assert(type_base == CHEB_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());

        base.def = true;
        switch (type_time) {
            case TO_PI:
                base.bases_1d[1]->set(0) = COS;
                break;
            case TO_PI_OVER_2:
                base.bases_1d[1]->set(0) = COS_EVEN;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = CHEB;
    }

    // Base for a function symetric in z, using Legendre
    void Domain_spheric_periodic_compact::set_legendre_base_odd(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());
        base.def = true;
        switch (type_time) {
            case TO_PI_OVER_2:
                base.bases_1d[1]->set(0) = COS_ODD;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = LEG;
    }
    // Base for a function symetric in z, using Legendre
    void Domain_spheric_periodic_compact::set_legendre_base(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());
        base.def = true;
        switch (type_time) {
            case TO_PI:
                base.bases_1d[1]->set(0) = COS;
                break;
            case TO_PI_OVER_2:
                base.bases_1d[1]->set(0) = COS_EVEN;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = LEG;
    }
    void Domain_spheric_periodic_compact::set_anti_legendre_base(Base_spectral& base) const
    {

        assert(type_base == LEG_TYPE);
        base.allocate(nbr_coefs);

        Index index(base.bases_1d[0]->get_dimensions());
        base.def = true;
        switch (type_time) {
            case TO_PI:
                base.bases_1d[1]->set(0) = COS;
                break;
            case TO_PI_OVER_2:
                base.bases_1d[1]->set(0) = COS_EVEN;
                break;
            default:
                KADATH_THROW("Unknown case for type_time");
        }
        for (int j = 0; j < nbr_coefs(1); j++)
            base.bases_1d[0]->set(j) = LEG;
    }
    // Computes the derivatives with respect to rho,Z as a function of the numerical ones.
    void Domain_spheric_periodic_compact::do_der_abs_from_der_var(Val_domain** der_var, Val_domain** der_abs) const
    {
        // d/dr
        der_abs[0] = new Val_domain(-der_var[0]->mult_xm1());
        // d/dt :
        der_abs[1] = new Val_domain(*der_var[1] * ome);
    }

    // sets the value at infinity
    void Domain_spheric_periodic_compact::set_val_inf(Val_domain& so, double x) const
    {

        assert(so.get_domain() == this);

        so.coef_i();
        so.set_in_conf();
        Index inf(nbr_points);
        inf.set(0) = nbr_points(0) - 1;
        do
            so.set(inf) = x;
        while (inf.inc(1, 1));
    }

    // Rules for the multiplication of two basis.
    Base_spectral Domain_spheric_periodic_compact::mult(const Base_spectral& a, const Base_spectral& b) const
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

            // Bases in theta :
            res.bases_1d[1] = std::make_unique<Array<int>>(a.bases_1d[1]->get_dimensions());
            switch ((*a.bases_1d[1])(0)) {
                case COS:
                    switch ((*b.bases_1d[1])(0)) {
                        case COS:
                            res.bases_1d[1]->set(0) = COS;
                            break;
                        case SIN:
                            res.bases_1d[1]->set(0) = SIN;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;

                case SIN:
                    switch ((*b.bases_1d[1])(0)) {
                        case COS:
                            res.bases_1d[1]->set(0) = SIN;
                            break;
                        case SIN:
                            res.bases_1d[1]->set(0) = COS;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case COS_EVEN:
                    switch ((*b.bases_1d[1])(0)) {
                        case COS_EVEN:
                            res.bases_1d[1]->set(0) = COS_EVEN;
                            break;
                        case COS_ODD:
                            res.bases_1d[1]->set(0) = COS_ODD;
                            break;
                        case SIN_EVEN:
                            res.bases_1d[1]->set(0) = SIN_EVEN;
                            break;
                        case SIN_ODD:
                            res.bases_1d[1]->set(0) = SIN_ODD;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case COS_ODD:
                    switch ((*b.bases_1d[1])(0)) {
                        case COS_EVEN:
                            res.bases_1d[1]->set(0) = COS_ODD;
                            break;
                        case COS_ODD:
                            res.bases_1d[1]->set(0) = COS_EVEN;
                            break;
                        case SIN_EVEN:
                            res.bases_1d[1]->set(0) = SIN_ODD;
                            break;
                        case SIN_ODD:
                            res.bases_1d[1]->set(0) = SIN_EVEN;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case SIN_EVEN:
                    switch ((*b.bases_1d[1])(0)) {
                        case COS_EVEN:
                            res.bases_1d[1]->set(0) = SIN_EVEN;
                            break;
                        case COS_ODD:
                            res.bases_1d[1]->set(0) = SIN_ODD;
                            break;
                        case SIN_EVEN:
                            res.bases_1d[1]->set(0) = COS_EVEN;
                            break;
                        case SIN_ODD:
                            res.bases_1d[1]->set(0) = COS_ODD;
                            break;
                        default:
                            res_def = false;
                            break;
                    }
                    break;
                case SIN_ODD:
                    switch ((*b.bases_1d[1])(0)) {
                        case COS_EVEN:
                            res.bases_1d[1]->set(0) = SIN_ODD;
                            break;
                        case COS_ODD:
                            res.bases_1d[1]->set(0) = SIN_EVEN;
                            break;
                        case SIN_EVEN:
                            res.bases_1d[1]->set(0) = COS_ODD;
                            break;
                        case SIN_ODD:
                            res.bases_1d[1]->set(0) = COS_EVEN;
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

            // Base in r :
            Index index_0(a.bases_1d[0]->get_dimensions());
            res.bases_1d[0] = std::make_unique<Array<int>>(a.bases_1d[0]->get_dimensions());
            do {
                switch ((*a.bases_1d[0])(index_0)) {
                    case CHEB:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case CHEB:
                                res.bases_1d[0]->set(index_0) = CHEB;
                                break;
                            default:
                                res_def = false;
                                break;
                        }
                        break;
                    case LEG:
                        switch ((*b.bases_1d[0])(index_0)) {
                            case LEG:
                                res.bases_1d[0]->set(index_0) = LEG;
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
            } while (index_0.inc());
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
} // namespace Kadath
