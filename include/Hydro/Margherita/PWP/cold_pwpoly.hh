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
#include "../margherita.hh"

#pragma once

namespace Kadath {
namespace Margherita {

class Cold_PWPoly {
 public:
  using error_t = std::bitset<2>;

  static inline error_t rho_range();

 private:
  static inline error_t check_range(double &rho);

  static inline int find_piece__rho(double &rho, error_t &error);
  static inline int find_piece__h_cold(double &h_cold, error_t &error);

 public:
  // General definitions for cold EOS

  static inline double dpress_cold_drho__rho(double &rho, error_t &error);

  static inline double press_cold_eps_cold__rho(double &eps_cold, double &rho,
                                                error_t &error);

  static inline double rho__press_cold(double &press_cold, error_t &error);
  static inline double rho__h_cold(double &h_cold, error_t &error);
  static inline double rho_energy_dedp__press_cold(double &energy, double &dedp, double &press,
         error_t &error);

  // Specific to PWPoly

  static inline double gamma_cold__rho(double &rho, error_t &error);

  static inline double gamma_cold_eps_tab__rho(double &eps_tabL, double &rho,
                                               error_t &error);

  static constexpr int max_num_pieces = 2000; 

  inline static std::array<double, max_num_pieces> k_tab{};
  inline static std::array<double, max_num_pieces> gamma_tab{};
  inline static std::array<double, max_num_pieces> rho_tab{};
  inline static std::array<double, max_num_pieces> eps_tab{};
  inline static std::array<double, max_num_pieces> P_tab{};
  inline static std::array<double, max_num_pieces> h_tab{};
  inline static int num_pieces = 1;

  inline static double rhomin{};
  inline static double rhomax{};
};
}
}
