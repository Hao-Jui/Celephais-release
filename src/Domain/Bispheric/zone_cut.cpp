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
#include "For_Kadath/Utilities/utilities.hpp"
#include "For_Kadath/Param/param.hpp"

namespace Kadath
{
    double feta_zero(double eta, const Param& par)
    {
        double chi = par.get_double(0);
        double rsa = par.get_double(1);
        double denom = (cosh(eta) - cos(chi));
        return ((sinh(eta) * sinh(eta) + sin(chi) * sin(chi)) / denom / denom - rsa * rsa);
    }

    double eta_lim_chi(double chi, double rext, double a, double eta_c)
    {
        double res;
        Param par_func;
        par_func.add_double(chi, 0);
        par_func.add_double(rext / a, 1);
        double precis = PRECISION;
        int nitermax = 500;
        int niter;

        if (chi <= PRECISION)
            res = eta_c;
        else
            res = zerosec(feta_zero, par_func, 0, eta_c, precis, nitermax, niter);
        return res;
    }

    double fchi_zero(double chi, const Param& par)
    {
        double eta = par.get_double(0);
        double rsa = par.get_double(1);
        double denom = (cosh(eta) - cos(chi));
        return ((sinh(eta) * sinh(eta) + sin(chi) * sin(chi)) / denom / denom - rsa * rsa);
    }

    double chi_lim_eta(double eta, double rext, double a, double chi_c)
    {
        double res;
        Param par_func;
        par_func.add_double(eta, 0);
        par_func.add_double(rext / a, 1);
        double precis = PRECISION;
        int nitermax = 500;
        int niter;
        if (fabs(eta) <= PRECISION)
            res = chi_c;
        else
            res = zerosec(fchi_zero, par_func, 0, chi_c, precis, nitermax, niter);
        return res;
    }
} // namespace Kadath
