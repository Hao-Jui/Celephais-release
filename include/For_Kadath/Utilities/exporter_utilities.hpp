/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once

#include "For_Kadath/Array/point.hpp"
using namespace Kadath;
// class Point ;
/**
 * @namespace export_utils
 * Utilities for exporting Kadath initial data to evolution kits such as ETK
 */
namespace export_utils
{

    /// id quantities
    enum { PSI, ALP, BETX, BETY, BETZ, AXX, AXY, AXZ, AYY, AYZ, AZZ, NUM_VQUANTS };

    /// chained enumeration to include matter quantities
    enum id_matter_quants { H = NUM_VQUANTS, UX, UY, UZ, NUM_QUANTS };

    /// enumeration for quantities to export to evolution codes
    enum sim_vac_quants {
        ALPHA,
        BETAX,
        BETAY,
        BETAZ,
        GXX,
        GXY,
        GXZ,
        GYY,
        GYZ,
        GZZ,
        KXX,
        KXY,
        KXZ,
        KYY,
        KYZ,
        KZZ,
        NUM_VOUT
    };

    /// chained enums together to include matter quantities
    enum sim_matter_quants { RHO = NUM_VOUT, EPS, PRESS, VELX, VELY, VELZ, NUM_OUT };

    /**
     * generate the kth lagrange polynomial
     *
     * [input] interp_order: polynomial interpolation order
     * [input] x: current x
     * [input] xp: pointer to array of x points
     * [input] yp: pointer to array of y points
     */
    double lagrange_gen_k(int interp_order, double x, const double *xp, const double *yp);

    /**
     * Create a Kadath point in Cartesian coordinates from the given input spherical coordinates
     *
     * [input] r: radius
     * [input] theta: angle theta
     * [input] phi: angle phi
     * [input] shift_x: offset of x
     */
    Point point_spherical(double r, double theta, double phi, double shift_x);
} // namespace export_utils
