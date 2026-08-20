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

#include <math.h>
#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Array/array.hpp"
#define NBR_LEG_MAX 20

namespace Kadath
{
    void legendre(int n, double& poly, double& pder, double& polym1, double& pderm1, double& polym2, double& pderm2,
                  double x)
    {

        if (n == 0) {
            poly = 1;
            pder = 0;
        } else if (n == 1) {
            polym1 = 1;
            pderm1 = 0;
            poly = x;
            pder = 1;
        } else {
            polym1 = 1;
            pderm1 = 0;
            poly = x;
            pder = 1;
            for (int i = 1; i < n; i++) {
                polym2 = polym1;
                pderm2 = pderm1;
                polym1 = poly;
                pderm1 = pder;
                poly = ((2 * i + 1) * x * polym1 - i * polym2) / (i + 1);
                pder = ((2 * i + 1) * polym1 + (2 * i + 1) * x * pderm1 - i * pderm2) / (i + 1);
            }
        }
    }

    double leg(int n, double x)
    {

        double p, dp, p1, dp1, p2, dp2;
        legendre(n, p, dp, p1, dp1, p2, dp2, x);
        return p;
    }

    void gauss_lobato_legendre(int n, Array<double>& coloc, Array<double>& weight, double prec = PRECISION,
                               int itemax = 500)
    {
        // Verifs :
        assert(coloc.get_ndim() == 1);
        assert(coloc.get_size(0) == n);
        assert(weight.get_ndim() == 1);
        assert(weight.get_size(0) == n);

        double x_plus = 1;
        double x_moins = -1;
        double p_plus, dp_plus, p_plus_m1, dp_plus_m1, p_plus_m2, dp_plus_m2;
        double p_moins, dp_moins, p_moins_m1, dp_moins_m1, p_moins_m2, dp_moins_m2;
        double p, dp, p_m1, dp_m1, p_m2, dp_m2;

        legendre(n, p_plus, dp_plus, p_plus_m1, dp_plus_m1, p_plus_m2, dp_plus_m2, x_plus);
        legendre(n, p_moins, dp_moins, p_moins_m1, dp_moins_m1, p_moins_m2, dp_moins_m2, x_moins);

        double det = p_plus_m1 * p_moins_m2 - p_moins_m1 * p_plus_m2;
        double r_plus = -p_plus;
        double r_moins = -p_moins;
        double a = (r_plus * p_moins_m2 - r_moins * p_plus_m2) / det;
        double b = (r_moins * p_plus_m1 - r_plus * p_moins_m1) / det;

        coloc.set(n - 1) = 1;
        double dth = M_PI / (2 * n + 1);
        double cd = cos(2 * dth);
        double sd = sin(2 * dth);
        double cs = cos(dth);
        double ss = sin(dth);

        int borne_sup = (n % 2 == 0) ? n / 2 : (n + 1) / 2;

        for (int j = 1; j < borne_sup; j++) {
            double x = cs;
            bool loop = true;
            int ite = 0;
            while (loop) {
                legendre(n, p, dp, p_m1, dp_m1, p_m2, dp_m2, x);
                double poly = p + a * p_m1 + b * p_m2;
                double pder = dp + a * dp_m1 + b * dp_m2;
                double sum = 0;
                for (int i = 0; i < j; i++)
                    sum += 1. / (x - coloc(n - i - 1));

                double increm = -poly / (pder - sum * poly);

                x += increm;
                ite++;
                if ((fabs(increm) < prec) || (ite > itemax))
                    loop = false;
            }
            if (ite > itemax) {
                KADATH_THROW("Too many iterations...");
            }
            coloc.set(n - j - 1) = x;
            double auxi = cs * cd - ss * sd;
            ss = cs * sd + ss * cd;
            cs = auxi;
        }
        if (n % 2 == 1)
            coloc.set((n - 1) / 2) = 0;
        // Copy of the symetric ones :
        for (int i = 0; i < borne_sup; i++)
            coloc.set(i) = -coloc(n - i - 1);

        for (int i = 0; i < n; i++) {
            legendre(n - 1, p, dp, p_m1, dp_m1, p_m2, dp_m2, coloc(i));
            weight.set(i) = 2. / (n - 1) / n / p / p;
        }
    }

    double get_things_leg(int indice, int n, int thing)
    {

        static bool first = true;
        static int nbr_done = 0;
        static int n_done[NBR_LEG_MAX];
        static Array<double>** coloc = new Array<double>*[NBR_LEG_MAX];
        static Array<double>** weight = new Array<double>*[NBR_LEG_MAX];
        static Array<double>** gamma = new Array<double>*[NBR_LEG_MAX];

        // Initialisation
        if (first) {
            for (int i = 0; i < NBR_LEG_MAX; i++) {
                n_done[i] = 0;
                coloc[i] = nullptr;
                weight[i] = nullptr;
                gamma[i] = nullptr;
            }
            first = false;
        }

        // Has the computation been done ?
        int index = -1;
        for (int i = 0; i < nbr_done; i++)
            if (n_done[i] == n)
                index = i;

        if (index == -1) {

            if (nbr_done >= NBR_LEG_MAX) {
                KADATH_THROW("Too many different number of points for Legendre...");
            }
            // Computation must be done :
            Array<double> col(n);
            Array<double> poids(n);

            gauss_lobato_legendre(n, col, poids);
            coloc[nbr_done] = new Array<double>(col);
            weight[nbr_done] = new Array<double>(poids);
            gamma[nbr_done] = new Array<double>(n);
            (*gamma[nbr_done]) = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    gamma[nbr_done]->set(i) += pow(leg(i, (*coloc[nbr_done])(j)), 2) * (*weight[nbr_done])(j);
            n_done[nbr_done] = n;
            nbr_done++;
            index = nbr_done - 1;
        }

        switch (thing) {
            case 0:
                return (*coloc[index])(indice);
                break;
            case 1:
                return (*weight[index])(indice);
                break;
            case 2:
                return (*gamma[index])(indice);
                break;
            default:
                KADATH_THROW("Strange error in get_things_legendre...");
        }
    }

    double coloc_leg(int i, int nbr)
    {
        assert((i >= 0) && (i < nbr));
        return get_things_leg(i, nbr, 0);
    }

    double weight_leg(int i, int nbr)
    {
        assert((i >= 0) && (i < nbr));
        return get_things_leg(i, nbr, 1);
    }

    double gamma_leg(int i, int nbr)
    {
        assert((i >= 0) && (i < nbr));
        return get_things_leg(i, nbr, 2);
    }

    // Case given parity odd or even...

    double get_things_leg_parity(int indice, int n, int thing)
    {

        static bool first = true;
        static int nbr_done = 0;
        static int n_done[NBR_LEG_MAX];
        static Array<double>** coloc = new Array<double>*[NBR_LEG_MAX];
        static Array<double>** weight = new Array<double>*[NBR_LEG_MAX];

        // Initialisation
        if (first) {
            for (int i = 0; i < NBR_LEG_MAX; i++) {
                n_done[i] = 0;
                coloc[i] = nullptr;
                weight[i] = nullptr;
            }
            first = false;
        }

        // Has the computation been done ?
        int index = -1;
        for (int i = 0; i < nbr_done; i++)
            if (n_done[i] == n)
                index = i;

        if (index == -1) {

            if (nbr_done >= NBR_LEG_MAX) {
                KADATH_THROW("Too many number of points for Legendre...");
            }
            // Computation must be done :
            Array<double> col(2 * n - 1);
            Array<double> poids(2 * n - 1);

            gauss_lobato_legendre(2 * n - 1, col, poids);
            coloc[nbr_done] = new Array<double>(n);
            for (int i = 0; i < n; i++)
                coloc[nbr_done]->set(i) = col(i + n - 1);
            weight[nbr_done] = new Array<double>(n);
            weight[nbr_done]->set(0) = poids(n - 1) / 2.;
            for (int i = 1; i < n; i++)
                weight[nbr_done]->set(i) = poids(i + n - 1);
            n_done[nbr_done] = n;
            nbr_done++;
            index = nbr_done - 1;
        }

        switch (thing) {
            case 0:
                return (*coloc[index])(indice);
                break;
            case 1:
                return (*weight[index])(indice);
                break;
            default:
                KADATH_THROW("Strange error in get_things_legendre...");
        }
    }

    double coloc_leg_parity(int i, int nbr)
    {
        assert((i >= 0) && (i < nbr));
        return get_things_leg_parity(i, nbr, 0);
    }

    double weight_leg_parity(int i, int nbr)
    {
        assert((i >= 0) && (i < nbr));
        return get_things_leg_parity(i, nbr, 1);
    }

    double gamma_leg_even(int nn, int nbr)
    {
        assert((nn >= 0) && (nn < nbr));
        static bool first = true;
        static int nbr_done = 0;
        static int n_done[NBR_LEG_MAX];
        static Array<double>** gamma = new Array<double>*[NBR_LEG_MAX];

        if (first) {
            for (int i = 0; i < NBR_LEG_MAX; i++) {
                n_done[i] = 0;
                gamma[i] = nullptr;
            }
            first = false;
        }

        // Has the computation been done ?
        int index = -1;
        for (int i = 0; i < nbr_done; i++)
            if (n_done[i] == nbr)
                index = i;

        if (index == -1) {

            if (nbr_done >= NBR_LEG_MAX) {
                KADATH_THROW("Too many number of points for Legendre...");
            }
            gamma[nbr_done] = new Array<double>(nbr);
            *gamma[nbr_done] = 0;
            for (int i = 0; i < nbr; i++)
                for (int j = 0; j < nbr; j++)
                    gamma[nbr_done]->set(i) += pow(leg(i * 2, coloc_leg_parity(j, nbr)), 2) * weight_leg_parity(j, nbr);
            n_done[nbr_done] = nbr;
            nbr_done++;
            index = nbr_done - 1;
        }

        return (*gamma[index])(nn);
    }

    double gamma_leg_odd(int nn, int nbr)
    {
        assert((nn >= 0) && (nn < nbr));
        static bool first = true;
        static int nbr_done = 0;
        static int n_done[NBR_LEG_MAX];
        static Array<double>** gamma = new Array<double>*[NBR_LEG_MAX];

        if (first) {
            for (int i = 0; i < NBR_LEG_MAX; i++) {
                n_done[i] = 0;
                gamma[i] = nullptr;
            }
            first = false;
        }

        // Has the computation been done ?
        int index = -1;
        for (int i = 0; i < nbr_done; i++)
            if (n_done[i] == nbr)
                index = i;

        if (index == -1) {

            if (nbr_done >= NBR_LEG_MAX) {
                KADATH_THROW("Too many number of points for Legendre...");
            }
            gamma[nbr_done] = new Array<double>(nbr);
            *gamma[nbr_done] = 0;
            for (int i = 0; i < nbr; i++)
                for (int j = 0; j < nbr; j++)
                    gamma[nbr_done]->set(i) +=
                        pow(leg(i * 2 + 1, coloc_leg_parity(j, nbr)), 2) * weight_leg_parity(j, nbr);
            n_done[nbr_done] = nbr;
            nbr_done++;
            index = nbr_done - 1;
        }

        return (*gamma[index])(nn);
    }
} // namespace Kadath
