/*
 * This file is part of the KADATH library.
 * Copyright (C) 2019, Samuel Tootle
 *                     <tootle@th.physik.uni-frankfurt.de>
 * Copyright (C) 2019, Ludwig Jens Papenfort
 *                      <papenfort@th.physik.uni-frankfurt.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once
#include "Margherita/PWP/cold_pwpoly.hh"
#include "Margherita/PWP/cold_pwpoly_implementation.hh"
#include "Margherita/PWP/setup_polytrope.hh"
#include "Margherita/Table/cold_table.hh"
#include "Margherita/Table/cold_table_implementation.hh"
#include "Margherita/Table/setup_cold_table.hh"
#include "For_Kadath/Utilities/name_tools.hpp"
#include "celephais_paths.h"
#include <string>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include "For_Kadath/Val_domain/val_domain.hpp"
#include "For_Kadath/Term_eq/term_eq.hpp"
#include "For_Kadath/Scalar/scalar.hpp"

/** 
 * The various hydrodynamic quantities that can be obtained from this
 * interface for a given EOS
 */
enum class eos_var_t { PRESSURE, EPSILON, DENSITY, DHDRHO };

/**
 * This interface provides the user defined OPEs to provide hydrodynamic quantities as a function
 * of the specific enthalpy (\b h) to the Kadath System_of_eqs framework.\n 
 * The equation of state is managed by a modified, standalone version of Margherita: 
 * https://github.com/fil-grmhd/Margherita-EOS
 *
 * @tparam eos Margherita EOS type (e.g. Cold_PWPoly)
 * @tparam var EOS variable to update when action() is executed. see eos_var_t
 */
template <typename eos, eos_var_t var> class EOS {
private:
  /**
   * EOS::term_by_term_variation
   *
   * calculate the numerical variation of the dependent quantity (var), term by term,
   * as a function of a scalar quantity (currently h).
   *
   * @param [input] dom: domain to update
   * @param [input] so: previous numerical variatioon of the variable field (h)
   * @param [input] scalar: current value of the variable field (h)
   */
  static inline Kadath::Val_domain term_by_term_variation(
      int dom, const Kadath::Val_domain so, const Kadath::Val_domain scalar) {
    if (so.check_if_zero()) {
      return so;
    }
    typename eos::error_t err;

    // need to work in configuration space
    so.coef_i();
    Kadath::Val_domain res(so.get_domain());
    res.allocate_conf();
    Kadath::Index pos(so.get_conf().get_dimensions());

    do {
      double eps_cold = 0.0;
      double h = scalar(pos);
      double dh = so(pos);

      double rho = eos::rho__h_cold(h, err);
      double dpdrho = eos::dpress_cold_drho__rho(rho, err);
      double drho = rho * 1. / dpdrho * dh;

      if constexpr (var == eos_var_t::EPSILON) {
        double pressure = eos::press_cold_eps_cold__rho(eps_cold, rho, err);
        res.set(pos) = pressure / pow(rho, 2) * drho;

      } else if constexpr (var == eos_var_t::PRESSURE) {
        res.set(pos) = eos::dpress_cold_drho__rho(rho, err) * drho;

      } else if constexpr (var == eos_var_t::DENSITY) {
        res.set(pos) = drho;

      } else if constexpr (var == eos_var_t::DHDRHO) {
        res.set(pos) = 0;
      } else
        std::cerr << "Ill-defined variable in EOS class, please check."
                  << std::endl;
    } while (pos.inc());

    res.set_base() = so.get_base();
    return res;
  }

  /**
   * EOS::term_by_term
   *
   * calculate the value the dependent quantity (var), term by term,
   * as a function of a scalar quantity (currently h).
   *
   * @param [input] dom: domain to update
   * @param [input] so: current value of the variable field (h)
   */
  static inline Kadath::Val_domain term_by_term(int dom, const Kadath::Val_domain so) {
    if (so.check_if_zero()) {
      return so;
    }
    typename eos::error_t err;

    // need to work in configuration space
    so.coef_i();
    Kadath::Val_domain res(so.get_domain());
    res.allocate_conf();
    Kadath::Index pos(so.get_conf().get_dimensions());

    do {
      double eps_cold = 0.0;
      double h = so(pos);
      double rho = eos::rho__h_cold(h, err);
      double pressure = eos::press_cold_eps_cold__rho(eps_cold, rho, err);

      if constexpr (var == eos_var_t::EPSILON)
        res.set(pos) = eps_cold;
      else if constexpr (var == eos_var_t::DENSITY)
        res.set(pos) = rho;
      else if constexpr (var == eos_var_t::PRESSURE)
        res.set(pos) = pressure;
      else if constexpr (var == eos_var_t::DHDRHO) {
        res.set(pos) = 1. / h * eos::dpress_cold_drho__rho(rho, err);
      }
      else
        std::cerr << "Ill-defined variable in EOS class, please check."
                  << std::endl;
    } while (pos.inc());

    res.set_base() = so.get_base();
    return res;
  }

public:
  /**
   * EOS::init
   *
   * initialize the EOS setup before attempting to query the EOS or define OPEs.
   * @param [input] filename: filename of the EOS Table for file describing the polytrope.
   * @param [input] h_cut: specific enthalpy to cut the table with
   * @param [input] interp_pts: number of points to use when interpolating an EOS Table
   * @param [input] mnuc_cgs: nuclear mass unit (g) for the table's n->rho conversion;
   *                non-positive falls back to the built-in Margherita constant
   */
  static void init(std::string filename = "", const double h_cut = 0.0, const int interp_pts = 2000,
                   const double mnuc_cgs = 0.0) {
    //if no path is given, we set the default EOS diretory to look for the relevant table/polytrope
    if (filename.rfind("/") == std::string::npos) {
      // Prefer runtime env var override; fall back to compile-time CELEPHAIS_DATA_DIR
      const char* home_env = std::getenv("HOME_CELEPHAIS");
      const std::string data_dir = home_env ? std::string(home_env) + "/data" : CELEPHAIS_DATA_DIR;
      filename = data_dir + "/eos/" + filename;
    }

    if constexpr (std::is_same_v<eos, Kadath::Margherita::Cold_PWPoly>)
      Kadath::Margherita::Margherita_setup_polytrope(filename);
    else if constexpr (std::is_same_v<eos, Kadath::Margherita::Cold_Table>)
      Kadath::Margherita::setup_Cold_Table(filename, interp_pts, h_cut, mnuc_cgs);
  }

  /**
   * EOS::h_cold__rho
   *
   * compute specific enthalpy from a given density.  Used primarily in analysis codes.
   *
   * @param [input] rho: density
   */
  static double h_cold__rho(double rho) {
    typename eos::error_t err;

    double eps_cold = 0.;
    double pressure = eos::press_cold_eps_cold__rho(eps_cold, rho, err);
    double h = 1. + eps_cold + pressure / rho;

    return h;
  }

  /**
   * EOS::get
   *
   * for a given specific enthalpy, get the dependent quantity(var)
   *
   * @param [input] h: specific enthalpy
   */
  static double get(double h) {
    typename eos::error_t err;

    double eps_cold = 0.0;
    double rho = eos::rho__h_cold(h, err);
    double pressure = eos::press_cold_eps_cold__rho(eps_cold, rho, err);

    if constexpr (var == eos_var_t::EPSILON)
      return eps_cold;
    else if constexpr (var == eos_var_t::DENSITY)
      return rho;
    else if constexpr (var == eos_var_t::PRESSURE)
      return pressure;
  }

  /**
   * EOS::action
   *
   * This is called by the System of equations in order to generate the corresponding
   * Scalar field and its variation based on a definition using a user defined OPE.
   *
   * @param [input] term: term to get information from (i.e. specific enthalpy).
   * @param [input] p: Kadath parameter.  Not used, but required for user defined OPEs
   */
  static Kadath::Term_eq action(const Kadath::Term_eq &term, Kadath::Param *p) {
    Kadath::Term_eq target(term);

    int dom = term.get_dom();
    if (target.get_type_data() != Kadath::TERM_T) {
      throw std::invalid_argument("EOS only defined with respect to tensor terms");
    }

    Kadath::Scalar scalar(target.get_val_t());
    Kadath::Scalar result(scalar, false);

    Kadath::Val_domain value(scalar(dom));
    if (value.check_if_zero())
      result.set_domain(dom).set_zero();
    else
      result.set_domain(dom) = EOS::term_by_term(dom, value);

    if (target.get_p_der_t() == nullptr) {
      return Kadath::Term_eq(dom, result);
    }

    auto build_derivative = [&](const Kadath::Tensor& derivative_tensor) {
      Kadath::Scalar scalar_var(derivative_tensor, true);
      Kadath::Scalar result_var(scalar_var, false);

      Kadath::Val_domain value_var(scalar_var(dom));
      if (value.check_if_zero())
        result_var.set_domain(dom).set_zero();
      else
        result_var.set_domain(dom) =
            EOS::term_by_term_variation(dom, value_var, value);
      return result_var;
    };

    Kadath::Term_eq eos_term(dom, result, build_derivative(target.get_der_t()));
    eos_term.set_derivative_lane_count(target.get_derivative_lane_count());
    for (int lane = 1; lane < target.get_derivative_lane_count(); ++lane) {
      if (target.has_der_t(lane))
        eos_term.set_der_t(lane, build_derivative(target.get_der_t(lane)));
    }
    return eos_term;
  }
};
