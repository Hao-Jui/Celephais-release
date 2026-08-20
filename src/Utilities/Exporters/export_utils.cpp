/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#include "For_Kadath/Utilities/exporter_utilities.hpp"
#include "For_Kadath/Array/point.hpp"
#include <cmath>

Point export_utils::point_spherical(double r, double theta, double phi, double shift_x)
{
    Point abs_coords(3);

    abs_coords.set(1) = r * std::sin(theta) * std::cos(phi) + shift_x;
    abs_coords.set(2) = r * std::sin(theta) * std::sin(phi);
    abs_coords.set(3) = r * std::cos(theta);

    return abs_coords;
}

double export_utils::lagrange_gen_k(int interp_order, double x, const double *xp, const double *yp)
{

    double temp = 0;
    for (int i = 0; i < interp_order; ++i) {
        double temp2 = 1.;
        for (int j = 0; j < interp_order; ++j) {
            if (i == j)
                continue;
            temp2 *= (x - xp[j]) / (xp[i] - xp[j]);
        }
        temp += yp[i] * temp2;
    }

    return temp;
}
