/*
 * =====================================================================================
 *
 *       Filename:  linear_interp.hh
 *
 *    Description:  Linear interpolation
 *
 *        Version:  1.0
 *        Created:  03/05/2017 09:47:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), most@fias.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 */

/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

template <typename T, int num_vars, bool uniform>
class linterp_t {
 public:
  template <int N>
  using vec = std::array<T, N>;

  using vec_ptr = std::unique_ptr<T[]>;

  int nvars = num_vars;

 private:
  vec_ptr x;
  vec_ptr deltax;
  std::array<vec_ptr, num_vars> y;
  int num_points{};

  inline int find_index(const T &xin) const {
    if (!uniform) {
      int lower = 0;
      int upper = num_points - 1;
      // simple bisection should do it
      while (upper - lower > 1) {
        int tmp = lower + (upper - lower) / 2;
        if (xin < x[tmp])
          upper = tmp;
        else
          lower = tmp;
      }
      return lower;
    } else {
      const auto xL = xin - x[0];
      return (xin < x[num_points - 1]) * (xL > 0) *
                 static_cast<int>(xL / deltax[0]) +
             (num_points - 2) * (xin >= x[num_points - 1]);
    }
  }

 public:
  inline const T &operator[](const int i) const { return x[i]; }

  inline int size() const { return num_points; }

  template <int... vals>
  inline vec<sizeof...(vals)> interpolate(const T &xin) const {
    constexpr int N = sizeof...(vals);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    const int index = find_index(xin);
    const auto lambda = (xin - x[index]) / deltax[index];

    vec<N> V{};

    int m = 0;
    ((V[m++] = y[vals][index + 1] * lambda + y[vals][index] * (1 - lambda)), ...);
    return V;
  }

  template <typename... Args>
  inline vec<num_vars> interpolate(const T &xin, const Args &... vals) const {
    constexpr int N = sizeof...(Args);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    const int index = find_index(xin);
    const auto lambda = (xin - x[index]) / deltax[index];

    vec<num_vars> V{};

    ((V[vals] = y[vals][index + 1] * lambda + y[vals][index] * (1 - lambda)), ...);
    return V;
  }

  inline vec<num_vars> interpolate_all(const T &xin) const {
    const int index = find_index(xin);
    const auto lambda = (xin - x[index]) / deltax[index];
    vec<num_vars> V{};

    for (int vals = 0; vals < num_vars; ++vals)
      V[vals] = y[vals][index + 1] * lambda + y[vals][index] * (1 - lambda);

    return V;
  }

  template <int... vals>
  inline vec<sizeof...(vals)> get_derivative(const T &xin) const {
    constexpr int N = sizeof...(vals);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    const auto index = find_index(xin);

    vec<N> V{};
    int m = 0;
    ((V[m++] = (y[vals][index + 1] - y[vals][index]) / (deltax[index])), ...);
    return V;
  }

  template <typename... TArgs>
  inline vec<num_vars> get_derivative(const T &xin,
                                      const TArgs &... vals) const {
    constexpr int N = sizeof...(vals);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    const auto index = find_index(xin);

    vec<num_vars> V{};
    ((V[vals] = (y[vals][index + 1] - y[vals][index]) / (deltax[index])), ...);
    return V;
  }

  template <typename... Args>
  linterp_t(int __num_points, vec_ptr &&__x, Args &&... args)
      : x(std::move(__x)), num_points(__num_points) {
    static_assert(sizeof...(args) == num_vars,
                  "Need exactly num_vars arguments");
    int n = 0;
    ((y[n++] = std::forward<Args>(args)), ...);

    deltax = std::make_unique<T[]>(num_points - 1);
    for (int i = 0; i < num_points - 1; ++i) {
      deltax[i] = x[i + 1] - x[i];
      if (std::isnan(deltax[i]) || deltax[i] <= 0.) {
        throw std::invalid_argument("linear interpolation grid points must be strictly increasing");
      }
    }
  }

  // Dummy constructor, can't really be used
  linterp_t() = default;

  template <typename... Args>
  inline void replace_y(Args &&... args) {
    static_assert(sizeof...(args) == num_vars,
                  "Need exactly num_vars arguments");
    int n = 0;
    ((y[n++] = std::forward<Args>(args)), ...);
  }
};

template <typename T, int num_vars>
using linear_interp_t = linterp_t<T, num_vars, false>;

template <typename T, int num_vars>
using linear_interp_uniform_t = linterp_t<T, num_vars, true>;
