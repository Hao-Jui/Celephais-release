//
// This file is part of Margherita, the light-weight EOS framework
//
//  Copyright (C) 2017, Elias Roland Most
//                      <emost@th.physik.uni-frankfurt.de>
//  Copyright (C) 2017, Ludwig Jens Papenfort
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
 */

#include <array>
#include <bitset>
#include <limits>
#include "../margherita.hh"
#include "Interpolation/inverse_interp.hh"
#include "Interpolation/linear_interp.hh"
#include "Interpolation/spline.hh"

#pragma once

namespace Kadath {
namespace Margherita {

template <int extra_vars = 0, template<typename,int> class interp_t = linear_interp_t>
class Cold_Table_t {
 public:
  using error_t = std::bitset<2>;

  using this_interp_t = interp_t<double, 2 + extra_vars>;
  using inverse_interp_t = linear_interp_t<double, 1>;

  enum class v_index { EPS = 0, PRESS, YE, TEMP, ENTROPY, CS2, NUM_VARS };
  static constexpr int NUM_VARS_V = 7;

 private:
  static inline bool inverse_cache_matches();
  static inline void ensure_inverse_tables();

 public:
  // General definitions for cold EOS

  static inline double dpress_cold_drho__rho(double &rho, error_t &error);

  static inline double press_cold_eps_cold__rho(double &eps_cold, double &rho,
                                                error_t &error);

  static inline double rho__press_cold(double &press_cold, error_t &error);
  static inline double rho__h_cold(double &press_cold, error_t &error);
  static inline double rho_energy_dedp__press_cold(double &energy, double &dedp, double &press,
         error_t &error);

  // Hot Slice capabilities
  // Only used for ID
  static inline std::array<double, 2 + extra_vars> get_extra_quantities(
      double &rho, error_t &error);

  // Specific to Cold_Table
  inline static double energy_shift{};

  inline static this_interp_t lintp{};
  inline static inverse_interp_t inverse_press_lintp{};
  inline static inverse_interp_t inverse_h_lintp{};
  inline static int inverse_cache_points{};
  inline static double inverse_cache_rhomin{
      std::numeric_limits<double>::quiet_NaN()};
  inline static double inverse_cache_rhomax{
      std::numeric_limits<double>::quiet_NaN()};
  inline static double inverse_cache_press_min{
      std::numeric_limits<double>::quiet_NaN()};
  inline static double inverse_cache_press_max{
      std::numeric_limits<double>::quiet_NaN()};
  inline static double inverse_cache_energy_shift{
      std::numeric_limits<double>::quiet_NaN()};

  inline static double press_min{};
  inline static double press_max{};
  inline static double rhomin{};
  inline static double rhomax{};
  inline static double hmin{};
  inline static double hmax{};

};

using Cold_Table = Cold_Table_t<0>;
}
}
