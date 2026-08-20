/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#include "For_Kadath/Kadath_point_h/kadath_bin_bh.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Utilities/exporter_utilities.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include <cmath>

using namespace Kadath;
using namespace export_utils;

std::array<std::vector<double>, NUM_VOUT> KadathExportBBH(int const npoints, double const* xx, double const* yy,
                                                          double const* zz, char const* fn,
                                                          double const interpolation_offset, int const interp_order,
                                                          double const delta_r_rel)
{
    std::string info_file(fn);
    kadath_config<BIN_INFO> bconfig(info_file);
    std::string kadath_fields = bconfig.space_filename();
    double yaxis = 0.;
    if (!std::isnan(bconfig.set(COMY)))
        yaxis = bconfig(COMY);

    BeFileSource fich(kadath_fields);
    Space_bin_bh space(fich);

    Scalar conf(space, fich);
    Scalar lapse(space, fich);
    Vector shift(space, fich);

    std::vector<std::reference_wrapper<const Scalar>> quants;
    for (int i = 0; i < NUM_VQUANTS; ++i)
        quants.push_back(std::cref(conf));
    quants[PSI] = std::cref(conf);
    quants[ALP] = std::cref(lapse);

    // reconstruct traceless extrinsic curvature
    Base_tensor base(shift.get_basis());
    Metric_flat fmet(space, base);
    System_of_eqs syst(space);
    fmet.set_system(syst, "f");

    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);
    syst.add_def("A_ij = (D_i bet_j + D_j bet_i - 2. / 3.* D^k bet_k * f_ij) /2. / N");

    quants[BETX] = std::cref(shift(1));
    quants[BETY] = std::cref(shift(2));
    quants[BETZ] = std::cref(shift(3));

    Tensor A(syst.give_val_def("A"));
    Index ind(A);

    quants[AXX] = std::cref(A(ind));
    ind.inc();
    quants[AXY] = std::cref(A(ind));
    ind.inc();
    quants[AXZ] = std::cref(A(ind));
    ind.inc();
    ind.inc();
    quants[AYY] = std::cref(A(ind));
    ind.inc();
    quants[AYZ] = std::cref(A(ind));
    ind.inc();
    ind.inc();
    ind.inc();
    quants[AZZ] = std::cref(A(ind));

    double xm = bco_utils::get_center(space, space.BH1);
    double xp = bco_utils::get_center(space, space.BH2);

    double rm = bco_utils::get_radius(space.get_domain(space.BH1 + 2), INNER_BC);
    double rp = bco_utils::get_radius(space.get_domain(space.BH2 + 2), INNER_BC);

    std::array<std::vector<double>, NUM_VOUT> out;
    for (auto& v : out)
        v.resize(npoints);

    for (int i = 0; i < npoints; ++i) {
        std::vector<double> quant_vals(NUM_VQUANTS);

        // get number of dimensions and construct index
        int ndim = 3;
        double x_shifted = xx[i] - bconfig(COM);  // shifting carpet [0,0] to the initial center of mass
        double y_shifted = yy[i] - yaxis;         // shifting carpet [0,0] to the initial center of mass
        double xxm = x_shifted - xm;
        double xxp = x_shifted - xp;

        double r2yz = y_shifted * y_shifted + zz[i] * zz[i];

        double r_minus = std::sqrt(xxm * xxm + r2yz);
        double r_plus = std::sqrt(xxp * xxp + r2yz);

        // lambda function for interpolation inside BH.
        auto interp_f = [&](auto& ah_r, auto& extrap_r, auto& bh_ori) {
            // ensure we don't divide by zero
            if (extrap_r == 0.)
                extrap_r = 1e-14;
            double xs = x_shifted - bh_ori;
            if (xs == 0.)
                xs = 1e-14;

            double theta = std::acos(zz[i] / extrap_r);
            double phi = std::atan2(y_shifted, xs); // atan2 is needed here

            std::vector<double> r_points(interp_order);
            for (int j = 0; j < interp_order; j++) {
                r_points[j] = (1. + interpolation_offset) * (1. + j * delta_r_rel) * ah_r;
            }

            // std::vector<double> psi6(interp_order);

            for (int k = 0; k < NUM_VQUANTS; ++k) {
                std::vector<double> vals(interp_order);

                for (int j = 0; j < interp_order; j++) {
                    vals[j] = quants[k].get().val_point(point_spherical(r_points[j], theta, phi, bh_ori));
                }

                quant_vals[k] = lagrange_gen_k(interp_order, extrap_r, r_points.data(), vals.data());
            }
        };

        if (r_minus <= (1. + interpolation_offset) * rm) {
            interp_f(rm, r_minus, xm);
        } else if (r_plus <= (1. + interpolation_offset) * rp) {
            interp_f(rp, r_plus, xp);
        } else {
            // construct Kadath point
            Point abs_coords(ndim);
            abs_coords.set(1) = x_shifted;
            abs_coords.set(2) = y_shifted;
            abs_coords.set(3) = zz[i];

            for (int k = 0; k < NUM_VQUANTS; ++k) {
                quant_vals[k] = quants[k].get().val_point(abs_coords);
            }
        }

        auto const psi = quant_vals[PSI];
        auto const psi2 = psi * psi;
        auto const psi4 = psi2 * psi2;

        out[ALPHA][i] = quant_vals[ALP];

        out[BETAX][i] = quant_vals[BETX];
        out[BETAY][i] = quant_vals[BETY];
        out[BETAZ][i] = quant_vals[BETZ];

        double g[3][3];
        g[0][0] = psi4;
        g[0][1] = 0.0;
        g[0][2] = 0.0;
        g[1][1] = psi4;
        g[1][2] = 0.0;
        g[2][2] = psi4;
        g[1][0] = g[0][1];
        g[2][0] = g[0][2];
        g[2][1] = g[1][2];

        out[GXX][i] = g[0][0];
        out[GXY][i] = g[0][1];
        out[GXZ][i] = g[0][2];
        out[GYY][i] = g[1][1];
        out[GYZ][i] = g[1][2];
        out[GZZ][i] = g[2][2];

        out[KXX][i] = quant_vals[AXX] * psi4;
        out[KXY][i] = quant_vals[AXY] * psi4;
        out[KXZ][i] = quant_vals[AXZ] * psi4;
        out[KYY][i] = quant_vals[AYY] * psi4;
        out[KYZ][i] = quant_vals[AYZ] * psi4;
        out[KZZ][i] = quant_vals[AZZ] * psi4;
    } // for i

    return out;
}
