//
// This file is part of Margherita, the light-weight EOS framework
//
//  Copyright (C) 2017, Elias Roland Most
//                      <emost@th.physik.uni-frankfurt.de>
//  Copyright (C) 2019, Ludwig Jens Papenfort
//                      <papenfort@th.physik.uni-frankfurt.de>
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 *   2026-06-16  Build the forward resampling interpolant with a monotone
 *               (Hyman-filtered) cubic spline (Interpolation/monotone_spline.hh)
 *               instead of the plain natural cubic spline. The natural spline
 *               overshoots across near-isobaric phase transitions, injecting
 *               nonpositive dP/drho and non-monotone log(P) into the resampled
 *               table; the monotone spline keeps full spline accuracy where the
 *               data is monotone and clamps only the overshoot nodes.
 */

#include "cold_table.hh"
#include "cold_table_implementation.hh"

#include "Interpolation/linear_interp.hh"
#include "Interpolation/monotone_spline.hh"

#include "lorene_io.hh"
#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>
namespace Kadath {
namespace Margherita {

void setup_Cold_Table(std::string cold_table_name, int cold_lintp_points, double h_cut, double mnuc_cgs) {
  auto vectors = Lorene_Table(cold_table_name, h_cut, mnuc_cgs);

  if (vectors[0].empty() || vectors[1].empty() || vectors[2].empty()) {
    throw std::runtime_error("Cold EOS table is empty");
  }
  if (vectors[0].front() <= 0 || vectors[2].front() <= 0) {
    throw std::runtime_error("Cold EOS table requires positive density and pressure");
  }

  Cold_Table::rhomin = vectors[0].front();
  Cold_Table::rhomax = vectors[0].back();

  Cold_Table::press_min = vectors[2].front();
  Cold_Table::press_max = vectors[2].back();

  double eps_min = vectors[1].front();
  double eps_max = vectors[1].back();

  double h_min = 1. + eps_min + Cold_Table::press_min / Cold_Table::rhomin;
  double h_max = 1. + eps_max + Cold_Table::press_max / Cold_Table::rhomax;

  Cold_Table::hmin = h_min;
  Cold_Table::hmax = h_max;

  [[maybe_unused]] double K_ext = (h_min - 1) / (2. * Cold_Table::rhomin);

  // Shift eps to ensure positivity before the log-space resample. The cold-crust
  // specific energy is non-monotone and can dip more negative AFTER the first kept
  // row, so the shift must clear the GLOBAL minimum, not just the front: a
  // front-based shift leaves the deeper-dip rows negative and log() poisons the
  // whole eps column with NaN.
  const double eps_global_min =
      *std::min_element(vectors[1].begin(), vectors[1].end());
  if (eps_global_min < 0) {
    Cold_Table::energy_shift = -2. * eps_global_min;
    for (auto &v : vectors[1]) v += Cold_Table::energy_shift;
  }

  // Log the table
  for (auto &v : vectors)
    for (auto &w : v) w = log(w);

  // Vector contains rho,eps and press vectors
  // These need to be moved into spline object
  auto rho_ptr = std::make_unique<double[]>(vectors[0].size());
  auto eps_ptr = std::make_unique<double[]>(vectors[1].size());
  auto press_ptr = std::make_unique<double[]>(vectors[2].size());

  for (int i = 0; i < static_cast<int>(vectors[0].size()); ++i) {
    rho_ptr[i] = vectors[0][i];
    eps_ptr[i] = vectors[1][i];
    press_ptr[i] = vectors[2][i];

  }

  // Interpolate the table onto a highly resolved uniform-log(rho) grid suitable
  // for fast linear interpolation at runtime. Use a monotone PCHIP interpolant
  // (not a natural cubic spline): the spline overshoots/rings across near-
  // isobaric stretches (first-order phase transitions), injecting nonpositive
  // dP/drho and non-monotone log(P) into the resampled forward table. PCHIP
  // preserves the data's monotonicity by construction, so press(rho)/eps(rho)
  // and dP/drho stay physical through a transition.

  auto forward_interp =
      monotone_spline_t<double, 2>(vectors[0].size(), std::move(rho_ptr),
                                   std::move(eps_ptr), std::move(press_ptr));

  auto rho_lin_ptr = std::make_unique<double[]>(cold_lintp_points);
  auto eps_lin_ptr = std::make_unique<double[]>(cold_lintp_points);
  auto press_lin_ptr = std::make_unique<double[]>(cold_lintp_points);

  const auto delta_rho = (log(Cold_Table::rhomax) - log(Cold_Table::rhomin)) /
                         (cold_lintp_points - 1);

  for (int nn = 0; nn < cold_lintp_points; ++nn) {
    const auto rhoL = log(Cold_Table::rhomin) + delta_rho * nn;
    rho_lin_ptr[nn] = rhoL;
    auto res = forward_interp.interpolate<0, 1>(rhoL);
    eps_lin_ptr[nn] = res[0];
    press_lin_ptr[nn] = res[1];
  }

  Cold_Table::lintp = linear_interp_t<double, 2>(
      cold_lintp_points, std::move(rho_lin_ptr), std::move(eps_lin_ptr),
      std::move(press_lin_ptr));

}
}
}
