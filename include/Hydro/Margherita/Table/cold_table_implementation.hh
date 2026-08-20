//
// This file is part of Margherita, the light-weight EOS framework
//
//  Copyright (C) 2017, Elias Roland Most
//                      <emost@th.physik.uni-frankfurt.de>
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
//
//  Modifications (Celephais):
//    2026-06-16  ensure_inverse_tables(): build the pressure->density inverse
//                with make_monotone_inverse_linear_interp() instead of the
//                strict make_inverse_linear_interp().  A first-order
//                phase-transition EOS (e.g. ZLA_a20_B160_g100) has a
//                near-isobaric plateau where consecutive stored log-pressures
//                collapse to equal in double precision (relative dP/P ~ 4e-15
//                falls below ULP(ln P) ~ 1.4e-14 near ln P ~ 82), which aborted
//                the strict builder.  The monotone variant keeps the increasing
//                (lowest-density) branch, matching the enthalpy inverse built
//                directly below.

#include "Interpolation/inverse_interp.hh"
#include "Interpolation/linear_interp.hh"
#include "Interpolation/spline.hh"
#include "cold_table.hh"

#pragma once

namespace Kadath {
namespace Margherita {

template <int extra_vars,template<typename,int> class interp_t>
inline bool Cold_Table_t<extra_vars,interp_t>::inverse_cache_matches() {
  return inverse_cache_points == lintp.size() &&
         inverse_cache_rhomin == rhomin &&
         inverse_cache_rhomax == rhomax &&
         inverse_cache_press_min == press_min &&
         inverse_cache_press_max == press_max &&
         inverse_cache_energy_shift == energy_shift;
}

template <int extra_vars,template<typename,int> class interp_t>
inline void Cold_Table_t<extra_vars,interp_t>::ensure_inverse_tables() {
  if (inverse_cache_matches()) {
    return;
  }

  const int points = lintp.size();
  if (points < 2) {
    throw std::runtime_error(
        "Cold_Table inverse interpolation requested before table setup");
  }

  auto log_rho_for_press = std::make_unique<double[]>(points);
  auto log_press = std::make_unique<double[]>(points);

  for (int i = 0; i < points; ++i) {
    const auto log_rho = lintp[i];
    const auto res = lintp.interpolate(
        log_rho, static_cast<int>(v_index::PRESS), static_cast<int>(v_index::EPS));

    log_rho_for_press[i] = log_rho;
    log_press[i] = res[static_cast<int>(v_index::PRESS)];
  }

  constexpr int enthalpy_inverse_subsamples = 8;
  const int h_points = (points - 1) * enthalpy_inverse_subsamples + 1;
  auto log_rho_for_h = std::make_unique<double[]>(h_points);
  auto h = std::make_unique<double[]>(h_points);
  const auto log_rhomin = log(rhomin);
  const auto delta_log_rho = (log(rhomax) - log_rhomin) / (h_points - 1);

  for (int i = 0; i < h_points; ++i) {
    const auto log_rho = log_rhomin + delta_log_rho * i;
    const auto res = lintp.interpolate(
        log_rho, static_cast<int>(v_index::PRESS), static_cast<int>(v_index::EPS));
    const auto pressure = exp(res[static_cast<int>(v_index::PRESS)]);
    const auto eps = exp(res[static_cast<int>(v_index::EPS)]) - energy_shift;
    const auto rho = exp(log_rho);

    log_rho_for_h[i] = log_rho;
    h[i] = 1. + eps + pressure / rho;
  }

  hmin = h[0];
  hmax = h[h_points - 1];
  // Tolerate first-order phase transitions: across a (near-)isobaric plateau the
  // stored log-pressure is non-strictly-increasing (sub-ULP-in-log density steps),
  // so the P->rho inverse is degenerate. Filter to the increasing branch exactly as
  // the enthalpy inverse below, instead of aborting on the flat segment.
  inverse_press_lintp = make_monotone_inverse_linear_interp(
      points, std::move(log_rho_for_press), std::move(log_press), "log rho",
      "log pressure");
  inverse_h_lintp = make_monotone_inverse_linear_interp(
      h_points, std::move(log_rho_for_h), std::move(h), "log rho",
      "specific enthalpy");

  inverse_cache_points = points;
  inverse_cache_rhomin = rhomin;
  inverse_cache_rhomax = rhomax;
  inverse_cache_press_min = press_min;
  inverse_cache_press_max = press_max;
  inverse_cache_energy_shift = energy_shift;
}

// ###################################################################################
// 	Public functions
// ###################################################################################

template <int extra_vars,template<typename,int> class interp_t>
inline double Cold_Table_t<extra_vars, interp_t>::press_cold_eps_cold__rho(
    double &eps_cold, double &rho, error_t &error) {
  const auto res = lintp.interpolate(log(rho), static_cast<int>(v_index::PRESS), static_cast<int>(v_index::EPS));

  eps_cold = exp(res[static_cast<int>(v_index::EPS)]) - energy_shift;
  return exp(res[static_cast<int>(v_index::PRESS)]);
}

template <int extra_vars, template<typename,int> class interp_t>
inline double Cold_Table_t<extra_vars, interp_t>::dpress_cold_drho__rho(double &rho,
                                                              error_t &error) {
  auto res = lintp.get_derivative(log(rho), static_cast<int>(v_index::PRESS));
  const auto res2 = lintp.interpolate(log(rho), static_cast<int>(v_index::PRESS));

  // res = dlog P/dlog \rho so need to multiply by P/rho
  return res[static_cast<int>(v_index::PRESS)] * exp(res2[static_cast<int>(v_index::PRESS)]) / rho;
}

template <int extra_vars,template<typename,int> class interp_t>
inline double Cold_Table_t<extra_vars,interp_t>::rho__press_cold(double &press_cold,
                                                        error_t &error) {
  ensure_inverse_tables();

  // Range check
  if (press_cold < press_min) {
    press_cold = press_min;
    error[0] = true;
    return rhomin;
  }

  // Range check
  if (press_cold > press_max) {
    press_cold = press_max;
    error[1] = true;
    return rhomax;
  }

  const auto lrho = inverse_press_lintp.interpolate(log(press_cold), 0);
  return exp(lrho[0]);
}

template <int extra_vars,template<typename,int> class interp_t>
inline double Cold_Table_t<extra_vars,interp_t>::rho__h_cold(double &h_cold, error_t
&error) {
  ensure_inverse_tables();

  // Range check
  if (h_cold < hmin) {
    h_cold = hmin;
    error[0] = true;
    return rhomin;
  }

  // Range check
  if (h_cold > hmax) {
    h_cold = hmax;
    error[1] = true;
    return rhomax;
  }

  const auto lrho = inverse_h_lintp.interpolate(h_cold, 0);
  return exp(lrho[0]);
}

template <int extra_vars,template<typename,int> class interp_t>
inline std::array<double, 2 + extra_vars>
Cold_Table_t<extra_vars,interp_t>::get_extra_quantities(double &rho, error_t &error) {
  return lintp.interpolate_all(log(rho));
}

template <int extra_vars,template<typename,int> class interp_t>
inline double
Cold_Table_t<extra_vars,interp_t>::rho_energy_dedp__press_cold(double &energy, double &dedp, double &press,
    error_t &error){

  auto rho = rho__press_cold(press,error);

  double eps;
  press_cold_eps_cold__rho(eps,rho,error);

  auto const dpdrho = dpress_cold_drho__rho(rho,error);

  energy = rho*(1.+eps);
  auto const rhoh = energy + press;

   dedp = rhoh/(dpdrho*rho);

   return rho;

}

}
}
