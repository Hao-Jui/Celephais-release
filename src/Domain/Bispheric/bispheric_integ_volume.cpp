/*
    Copyright 2026 Celephais contributors

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

// Volume integration over the bispherical domains.
//
// Kadath historically left Domain::integ_volume unimplemented for the
// bispherical zones, so volume-integrated diagnostics (e.g. the L2 norm of a
// constraint violation in the binary readers) had to skip them. This file
// supplies integ_volume for all six bispherical domain classes
// (rect / chi_first / eta_first, each in its symmetric and nosym flavour).
//
// Geometry. Every bispherical domain maps the numerical cube to Cartesian space
// through the standard 3D bispherical chart with the binary axis along x:
//
//     x = x_0 + a sinh(eta) / D,   y = a sin(chi) cos(phi) / D,
//                                  z = a sin(chi) sin(phi) / D,
//     D = cosh(eta) - cos(chi),
//
// (x_0 is the chart offset origin_x for the nosym domains, 0 otherwise). The
// physical volume element is dV = a^3 sin(chi) / D^3 d(eta) d(chi) d(phi).
//
// Why not the spectral-coefficient quadrature used by the spherical domains.
// The geometric weight carries a single power of sin(chi), so the integrand is
// odd in chi at the axisymmetric (k=0) azimuthal mode; the bispherical spectral
// transform represents the k=0 mode in the chi-even sector only (regularity on
// the axis), so that integrand cannot be taken to coefficient space. The
// measure a^3 sin(chi)/D^3 is also non-polynomial, hence not exactly
// representable in any case.
//
// Approach. A Gauss-Legendre quadrature over the numerical cube, decoupled from
// the spectral collocation grid:
//
//     INT f dV = INT f * w_geom * |d(eta,chi)/d(xi0,xi1)| d(xi0) d(xi1) d(phi),
//     w_geom = a^3 sin(chi) / D^3.
//
// The field f (a regular, chi-even scalar) is evaluated at the Gauss nodes by
// spectral summation, the geometric weight and the numerical-coordinate
// Jacobian are evaluated analytically (the (eta, chi) <- (xi0, xi1) map is
// per-domain; for chi_first / eta_first one bound is a profile obtained from
// eta_lim_chi / chi_lim_eta), and the azimuthal integral runs over the full
// [0, 2*pi): for the symmetric domains the cosine series extends evenly past pi,
// so summation returns the correct mirror values with no special casing. The
// Gauss nodes cluster towards xi = +-1, i.e. towards chi = chi_min where the
// 1/D^3 measure peaks, giving spectral accuracy for the smooth integrand.

#include "For_Kadath/Array/headcpp.hpp"
#include "For_Kadath/Array/exceptions.hpp"

#include "For_Kadath/Domain/bispheric.hpp"
#include "For_Kadath/Domain/bispheric_nosym.hpp"
#include "For_Kadath/Array/array.hpp"
#include "For_Kadath/Val_domain/val_domain.hpp"

#include <cmath>
#include <vector>

namespace Kadath
{
// Bispherical coordinate cutoffs, defined in src/Domain/Bispheric/zone_cut.cpp:
// eta on the outer sphere as a function of chi, and chi as a function of eta.
double eta_lim_chi(double chi, double rext, double a, double eta_c);
double chi_lim_eta(double eta, double rext, double a, double chi_c);

namespace
{

// Number of Gauss-Legendre nodes per numerical direction. The integrand is
// smooth but the measure is sharply peaked near chi_min; the Gauss clustering
// at the interval ends resolves it, and 32 nodes give ~1e-6 relative accuracy.
constexpr int kGaussNodes = 32;

// Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration on P_n).
void gauss_legendre(int n, std::vector<double>& nodes, std::vector<double>& weights)
{
    nodes.assign(n, 0.);
    weights.assign(n, 0.);
    for (int i = 0; i < n; ++i) {
        double x = std::cos(M_PI * (i + 0.75) / (n + 0.5)); // initial guess
        double dp = 1.;
        for (int it = 0; it < 100; ++it) {
            double p0 = 1., p1 = x;
            for (int k = 2; k <= n; ++k) {
                const double p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            dp = n * (x * p1 - p0) / (x * x - 1.); // P_n'(x)
            const double dx = -p1 / dp;
            x += dx;
            if (std::fabs(dx) < 1e-15)
                break;
        }
        nodes[i] = x;
        weights[i] = 2. / ((1. - x * x) * dp * dp);
    }
}

// Result of the per-domain map (xi0, xi1) -> (eta, chi) plus the absolute value
// of the 2x2 numerical-coordinate Jacobian |d(eta,chi)/d(xi0,xi1)|.
struct MapPoint
{
    double eta;
    double chi;
    double abs_jac;
};

// Runs the Gauss-Legendre quadrature. xi0 spans the axis-0 numerical extent,
// xi1 the axis-1 extent, phi the full [0, 2*pi). `map` returns (eta, chi, |J|)
// for given (xi0, xi1); `so` is summed spectrally at each Gauss node.
template <typename MapFn>
double bispheric_gauss_volume(const Val_domain& so, double aa, double xi0_lo, double xi0_hi,
                              double xi1_lo, double xi1_hi, MapFn&& map)
{
    so.coef();
    const Array<double> cf = so.get_coef(); // one copy; summation reads it at every node
    const Base_spectral& base = so.get_base();

    std::vector<double> gn, gw;
    gauss_legendre(kGaussNodes, gn, gw);

    const double h0 = (xi0_hi - xi0_lo) / 2., m0 = (xi0_hi + xi0_lo) / 2.;
    const double h1 = (xi1_hi - xi1_lo) / 2., m1 = (xi1_hi + xi1_lo) / 2.;
    const double hp = M_PI, mp = M_PI; // phi in [0, 2*pi)

    Point num(3);
    double res = 0.;
    for (int i0 = 0; i0 < kGaussNodes; ++i0) {
        const double xi0 = m0 + h0 * gn[i0];
        num.set(1) = xi0;
        for (int i1 = 0; i1 < kGaussNodes; ++i1) {
            const double xi1 = m1 + h1 * gn[i1];
            const MapPoint pt = map(xi0, xi1);
            const double denom = std::cosh(pt.eta) - std::cos(pt.chi);
            const double w_geom = aa * aa * aa * std::sin(pt.chi) / (denom * denom * denom);
            const double weight_xy = gw[i0] * gw[i1] * h0 * h1 * w_geom * pt.abs_jac;
            num.set(2) = xi1;
            // Collapse the (eta, chi) dimensions once per (xi0, xi1); the phi
            // sweep then reuses this 1D partial via summation_last_dim instead of
            // re-running the full 3D summation (with its coefficient copy) at
            // every azimuthal node. Value-identical, ~kGaussNodes-fold cheaper.
            const Array<double> partial = base.summation_but_last(num, cf);
            for (int ip = 0; ip < kGaussNodes; ++ip) {
                res += base.summation_last_dim(mp + hp * gn[ip], partial) * weight_xy * gw[ip] * hp;
            }
        }
    }
    return res;
}

} // namespace

// ---------------------------------------------------------------------------
// Symmetric bispherical domains (chart origin at x = 0).
// ---------------------------------------------------------------------------

double Domain_bispheric_rect::integ_volume(const Val_domain& so) const
{
    if (so.check_if_zero())
        return 0.;
    const double slope_eta = (eta_plus - eta_minus) / 2.;
    const double mid_eta = (eta_plus + eta_minus) / 2.;
    const double abs_jac = std::fabs(slope_eta) * std::fabs(chi_min - M_PI);
    return bispheric_gauss_volume(so, aa, -1., 1., 0., 1., [&](double xi0, double xi1) {
        return MapPoint{slope_eta * xi0 + mid_eta, (chi_min - M_PI) * xi1 + M_PI, abs_jac};
    });
}

double Domain_bispheric_chi_first::integ_volume(const Val_domain& so) const
{
    if (so.check_if_zero())
        return 0.;
    const int signe = (eta_lim < 0) ? -1 : 1;
    return bispheric_gauss_volume(so, aa, -1., 1., 0., 1., [&](double xi0, double xi1) {
        const double chi = chi_max * xi1;
        const double bound = signe * eta_lim_chi(chi, r_ext, aa, eta_c);
        const double eta = (bound - eta_lim) / 2. * xi0 + (bound + eta_lim) / 2.;
        const double abs_jac = std::fabs((bound - eta_lim) / 2.) * std::fabs(chi_max);
        return MapPoint{eta, chi, abs_jac};
    });
}

double Domain_bispheric_eta_first::integ_volume(const Val_domain& so) const
{
    if (so.check_if_zero())
        return 0.;
    const double slope_eta = (eta_max - eta_min) / 2.;
    const double mid_eta = (eta_max + eta_min) / 2.;
    return bispheric_gauss_volume(so, aa, 0., 1., -1., 1., [&](double xi0, double xi1) {
        const double eta = slope_eta * xi1 + mid_eta;
        const double bound = chi_lim_eta(std::fabs(eta), r_ext, aa, chi_c);
        const double chi = (bound - M_PI) * xi0 + M_PI;
        const double abs_jac = std::fabs(slope_eta) * std::fabs(bound - M_PI);
        return MapPoint{eta, chi, abs_jac};
    });
}

// ---------------------------------------------------------------------------
// nosym bispherical domains (chart offset by origin_x along the binary axis).
// The offset is a pure translation; it leaves the volume element unchanged, and
// the field is evaluated in numerical coordinates, so origin_x does not appear.
// ---------------------------------------------------------------------------

double Domain_bispheric_rect_nosym::integ_volume(const Val_domain& so) const
{
    if (so.check_if_zero())
        return 0.;
    const double slope_eta = (eta_plus - eta_minus) / 2.;
    const double mid_eta = (eta_plus + eta_minus) / 2.;
    const double abs_jac = std::fabs(slope_eta) * std::fabs(chi_min - M_PI);
    return bispheric_gauss_volume(so, aa, -1., 1., 0., 1., [&](double xi0, double xi1) {
        return MapPoint{slope_eta * xi0 + mid_eta, (chi_min - M_PI) * xi1 + M_PI, abs_jac};
    });
}

double Domain_bispheric_chi_first_nosym::integ_volume(const Val_domain& so) const
{
    if (so.check_if_zero())
        return 0.;
    const int signe = (eta_lim < 0) ? -1 : 1;
    return bispheric_gauss_volume(so, aa, -1., 1., 0., 1., [&](double xi0, double xi1) {
        const double chi = chi_max * xi1;
        const double bound = signe * eta_lim_chi(chi, r_ext, aa, eta_c);
        const double eta = (bound - eta_lim) / 2. * xi0 + (bound + eta_lim) / 2.;
        const double abs_jac = std::fabs((bound - eta_lim) / 2.) * std::fabs(chi_max);
        return MapPoint{eta, chi, abs_jac};
    });
}

double Domain_bispheric_eta_first_nosym::integ_volume(const Val_domain& so) const
{
    if (so.check_if_zero())
        return 0.;
    const double slope_eta = (eta_max - eta_min) / 2.;
    const double mid_eta = (eta_max + eta_min) / 2.;
    return bispheric_gauss_volume(so, aa, 0., 1., -1., 1., [&](double xi0, double xi1) {
        const double eta = slope_eta * xi1 + mid_eta;
        const double bound = chi_lim_eta(std::fabs(eta), r_ext, aa, chi_c);
        const double chi = (bound - M_PI) * xi0 + M_PI;
        const double abs_jac = std::fabs(slope_eta) * std::fabs(bound - M_PI);
        return MapPoint{eta, chi, abs_jac};
    });
}

} // namespace Kadath
