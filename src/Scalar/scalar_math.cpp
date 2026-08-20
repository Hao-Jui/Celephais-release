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

#include "For_Kadath/Scalar/scalar.hpp"
namespace Kadath
{
    void Scalar::operator+=(const Scalar& so)
    {
        assert(&espace == &so.espace);
        *this = *this + so;
    }

    void Scalar::operator-=(const Scalar& so)
    {
        assert(&espace == &so.espace);
        *this = *this - so;
    }

    void Scalar::operator*=(const Scalar& so)
    {
        assert(&espace == &so.espace);
        *this = *this * so;
    }

    void Scalar::operator/=(const Scalar& so)
    {
        assert(&espace == &so.espace);
        *this = *this / so;
    }

    void Scalar::operator+=(double xx)
    {
        *this = *this + xx;
    }

    void Scalar::operator-=(double xx)
    {
        *this = *this - xx;
    }

    void Scalar::operator*=(double xx)
    {
        *this = *this * xx;
    }

    void Scalar::operator/=(double xx)
    {
        *this = *this / xx;
    }

    Scalar sin(const Scalar& so)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = sin(so(i));
        return res;
    }

    Scalar cos(const Scalar& so)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = cos(so(i));
        return res;
    }

    Scalar operator+(const Scalar& so)
    {
        Scalar res(so);
        return res;
    }

    Scalar operator-(const Scalar& so)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = -so(i);
        return res;
    }

    Scalar operator+(const Scalar& a, const Scalar& b)
    {
        assert(&a.espace == &b.espace);
        Scalar res(a.get_space());
        for (int i = 0; i < a.get_nbr_domains(); i++)
            res.set_domain(i) = a(i) + b(i);
        return res;
    }

    Scalar operator+(const Scalar& so, double x)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = so(i) + x;
        return res;
    }

    Scalar operator+(double x, const Scalar& so)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = so(i) + x;
        return res;
    }

    Scalar operator-(const Scalar& a, const Scalar& b)
    {
        assert(&a.espace == &b.espace);
        Scalar res(a.get_space());
        for (int i = 0; i < a.get_nbr_domains(); i++)
            res.set_domain(i) = a(i) - b(i);
        return res;
    }

    Scalar operator-(const Scalar& so, double x)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = so(i) - x;
        return res;
    }

    Scalar operator-(double x, const Scalar& so)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = x - so(i);
        return res;
    }

    Scalar operator*(const Scalar& a, const Scalar& b)
    {
        assert(&a.espace == &b.espace);
        Scalar res(a.get_space());
        for (int i = 0; i < a.get_nbr_domains(); i++)
            res.set_domain(i) = a(i) * b(i);
        return res;
    }

    Scalar operator*(const Scalar& so, double x)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = so(i) * x;
        return res;
    }

    Scalar operator*(double x, const Scalar& so)
    {
        return (so * x);
    }

    Scalar operator*(const Scalar& so, int m)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = so(i) * m;
        return res;
    }

    Scalar operator*(int m, const Scalar& so)
    {
        return (so * m);
    }

    Scalar operator*(const Scalar& so, long int m)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = so(i) * m;
        return res;
    }

    Scalar operator*(long int m, const Scalar& so)
    {
        return (so * m);
    }

    Scalar operator/(const Scalar& a, const Scalar& b)
    {
        assert(&a.espace == &b.espace);
        Scalar res(a.get_space());
        for (int i = 0; i < a.get_nbr_domains(); i++)
            res.set_domain(i) = a(i) / b(i);
        return res;
    }

    Scalar operator/(const Scalar& so, double x)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = so(i) / x;
        return res;
    }

    Scalar operator/(double x, const Scalar& so)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = x / so(i);
        return res;
    }

    Scalar pow(const Scalar& so, int n)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = pow(so(i), n);
        return res;
    }

    Scalar pow(const Scalar& so, double nn)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = pow(so(i), nn);
        return res;
    }

    Scalar sqrt(const Scalar& so)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = sqrt(so(i));
        return res;
    }

    Scalar exp(const Scalar& so)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = exp(so(i));
        return res;
    }

    Scalar atan(const Scalar& so)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = atan(so(i));
        return res;
    }

    Scalar atanh(const Scalar& so)
    {
        Scalar res(so.get_space());
        for (int i = 0; i < so.get_nbr_domains(); i++)
            res.set_domain(i) = atanh(so(i));
        return res;
    }

    double diffmax(const Scalar& aa, const Scalar& bb)
    {
        assert(&aa.espace == &bb.espace);
        double res = 0;
        for (int dd = 0; dd < aa.espace.get_nbr_domains(); dd++) {
            double courant = diffmax(aa(dd), bb(dd));
            if (courant > res)
                res = courant;
        }
        return res;
    }
} // namespace Kadath
