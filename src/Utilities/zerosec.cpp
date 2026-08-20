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
 * Search for a zero of a function in a given interval, by means of a
 *  secant method.
 *
 */

// Headers C
#include <math.h>
#include <assert.h>

// Headers Lorene
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Param/param.hpp"
//****************************************************************************
namespace Kadath
{
    double zerosec(double (*f)(double, const Param&), const Param& parf, double x1, double x2, double precis,
                   int nitermax, int& niter)
    {

        double f0_prec, f0, x0, x0_prec, dx, df;

        // Teste si un zero unique existe dans l'intervalle [x_1,x_2]

        //    bool warning = false ;

        f0_prec = f(x1, parf);
        f0 = f(x2, parf);
        if (f0 * f0_prec > 0.) {
            //	warning = true ;
            // cout <<
            //   "WARNING: zerosec: there does not exist a unique zero of the function"
            //<< endl ;
            // cout << "  between x1 = " << x1 << " ( f(x1)=" << f0_prec << " )" << endl ;
            // cout << "      and x2 = " << x2 << " ( f(x2)=" << f0 << " )" << endl ;
        }

        // Choisit la borne avec la plus petite valeur de |f(x)| comme la valeur la
        //  "plus recente" de x0

        if (fabs(f0) < fabs(f0_prec)) { // On a bien choisi f0_prec et f0
            x0_prec = x1;
            x0 = x2;
        } else { // il faut interchanger f0_prec et f0
            x0_prec = x2;
            x0 = x1;
            double swap = f0_prec;
            f0_prec = f0;
            f0 = swap;
        }

        // Debut des iterations de la methode de la secante

        niter = 0;
        do {
            df = f0 - f0_prec;
            assert(df != double(0));
            dx = (x0_prec - x0) * f0 / df;
            x0_prec = x0;
            f0_prec = f0;
            x0 += dx;
            f0 = f(x0, parf);
            niter++;
            if (niter > nitermax) {
                KADATH_THROW("zerosec: Maximum number of iterations has been reached ! ");
            }
        } while ((fabs(dx) > precis) && (fabs(f0) > PRECISION));

        //  if (warning) {
        //	cout << "      A zero has been found at x0 = " << x0 << endl ;
        // }

        return x0;
    }

} // namespace Kadath
