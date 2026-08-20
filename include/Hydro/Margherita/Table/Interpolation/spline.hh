/*
 * =====================================================================================
 *
 *       Filename:  spline.hh
 *
 *    Description:  Cubic spline interpolation.  The natural-spline second
 *                 derivatives are obtained by a Thomas tridiagonal solve
 *                 (O(num_points), exact for this symmetric strictly
 *                 diagonally-dominant system, no pivoting required).
 *
 *        Version:  1.0
 *        Created:  01/05/2017 00:14:10
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Elias Roland Most (ERM), most@fias.uni-frankfurt.de
 *   Organization:  Goethe University Frankfurt
 *
 * =====================================================================================
 *
 * This file is part of Margherita and is distributed under the GNU General
 * Public License v3 (see the Margherita Table sources for the full notice).
 *
 * Modifications (Celephais):
 *   2026-06-16  Replaced the dense O(num_points^3) LU decomposition in
 *               compute_ypp() with an in-place O(num_points) Thomas
 *               tridiagonal solve and removed the now-dead dense matrix
 *               builder.  Result is identical to rounding; table init on a
 *               7.5k-row Lorene table drops from ~30 s to ~0.2 s.
 *
 * =====================================================================================
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>
#include "LUP_old.cc"

template <typename T, int num_vars>
class cubic_spline_t {
 public:
  template <int N>
  using vec = std::array<T, N>;

  using vec_ptr = std::unique_ptr<T[]>;

  int nvars = num_vars;

  T minval;
  T maxval;

 private:
  vec_ptr x;
  std::array<vec_ptr, num_vars> y;
  vec_ptr deltax;
  std::array<vec_ptr, num_vars> ypp;

  int num_points;

  inline const T &operator[](const int i) const { return x[i]; }

  inline int size() const { return num_points; }

  inline void compute_rhs(int N, T rhs[]) {
    for (int i = 0; i < num_points - 2; ++i) {
      rhs[i] = (y[N][i + 2] - y[N][i + 1]) / deltax[i + 1] -
               (y[N][i + 1] - y[N][i]) / deltax[i];
    }
  }

  inline void compute_ypp() {
    // Natural cubic spline.  The second-derivative (ypp) system is symmetric,
    // strictly diagonally dominant, and TRIDIAGONAL, so it is solved directly
    // with the Thomas algorithm in O(num_points).  The previous code formed the
    // full dense (num_points-2)^2 matrix and ran an O(num_points^3) LU
    // decomposition with iterative polishing — for a 7.5k-row Lorene table that
    // cost ~30 s and ~1 GB before the table could even be queried.
    //
    // Row i (interior node i+1, i = 0 .. dim-1) of the system is
    //   sub[i]  * ypp[i] + diag[i] * ypp[i+1] + sup[i] * ypp[i+2] = rhs[i]
    // with
    //   sub[i]  = deltax[i]     / 6   (multiplies the (i-1)-th unknown, i > 0)
    //   diag[i] = (x[i+2]-x[i]) / 3
    //   sup[i]  = deltax[i+1]   / 6   (multiplies the (i+1)-th unknown, i < dim-1)
    const int dim = num_points - 2;

    // Natural (zero) second-derivative boundary conditions.
    for (int n = 0; n < num_vars; ++n) {
      ypp[n][0] = 0;
      ypp[n][num_points - 1] = 0;
    }
    if (dim <= 0) return;

    std::vector<T> cprime(dim);  // forward-swept super-diagonal
    std::vector<T> rhs(dim);

    for (int n = 0; n < num_vars; ++n) {
      compute_rhs(n, rhs.data());

      // Forward elimination.  dprime is accumulated in place in ypp[n][i+1].
      const T diag0 = (x[2] - x[0]) / 3.;
      cprime[0] = (dim > 1) ? (deltax[1] / 6.) / diag0 : T(0);
      ypp[n][1] = rhs[0] / diag0;
      for (int i = 1; i < dim; ++i) {
        const T sub  = deltax[i] / 6.;
        const T diag = (x[i + 2] - x[i]) / 3.;
        const T denom = diag - sub * cprime[i - 1];
        if (i < dim - 1) cprime[i] = (deltax[i + 1] / 6.) / denom;
        ypp[n][i + 1] = (rhs[i] - sub * ypp[n][i]) / denom;
      }

      // Back substitution into the interior nodes.
      for (int i = dim - 2; i >= 0; --i)
        ypp[n][i + 1] -= cprime[i] * ypp[n][i + 2];
    }
  }

  inline int find_index(const T &xin) const {
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
  }

  inline std::array<T, 4> interpolation_coefficients(const T &xin,
                                                     int &index) const {
    index = find_index(xin);
    std::array<T, 4> coeffs;
    coeffs[0] = (x[index + 1] - xin) / deltax[index];
    coeffs[1] = (xin - x[index]) / deltax[index];
    coeffs[2] = (coeffs[0] * coeffs[0] - 1.) *
                ((x[index + 1] - xin) * deltax[index]) / 6.;
    coeffs[3] =
        (coeffs[1] * coeffs[1] - 1.) * ((xin - x[index]) * deltax[index]) / 6.;

    return coeffs;
  }

 public:
  template <int... vals>
  inline vec<sizeof...(vals)> interpolate(const T &xin) const {
    constexpr int N = sizeof...(vals);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    int index;
    const auto coeffs = interpolation_coefficients(xin, index);

    vec<N> V{};

    int m = 0;
    ((V[m++] = coeffs[0] * y[vals][index] + coeffs[1] * y[vals][index + 1] +
                coeffs[2] * ypp[vals][index] + coeffs[3] * ypp[vals][index + 1]), ...);
    return V;
  }

  template <typename... TArgs>
  inline vec<num_vars> interpolate(const T &xin, const TArgs &... vals) const {
    constexpr int N = sizeof...(TArgs);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    int index;
    const auto coeffs = interpolation_coefficients(xin, index);

    vec<num_vars> V{};

    ((V[vals] = coeffs[0] * y[vals][index] +
                  coeffs[1] * y[vals][index + 1] +
                  coeffs[2] * ypp[vals][index] +
                  coeffs[3] * ypp[vals][index + 1]), ...);
    return V;
  }

  template <typename... TArgs>
  inline vec<num_vars> interpolate_strict(const T &xin,
                                          const TArgs &... vals) const {
    constexpr int N = sizeof...(TArgs);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    auto xinL = std::min(x[num_points - 1], std::max(x[0], xin));
    int index;
    const auto coeffs = interpolation_coefficients(xinL, index);

    vec<num_vars> V{};

    int m = 0;
    int tmp[] = {(V[vals] = coeffs[0] * y[vals][index] +
                            coeffs[1] * y[vals][index + 1] +
                            coeffs[2] * ypp[vals][index] +
                            coeffs[3] * ypp[vals][index + 1],
                  ++m)...};
    (void)tmp;
    return V;
  }

  inline vec<num_vars> interpolate_all(const T &xin) const {
    int index;
    const auto coeffs = interpolation_coefficients(xin, index);
    vec<num_vars> V{};

    for (int vals = 0; vals < num_vars; ++vals) {
      V[vals] = coeffs[0] * y[vals][index] + coeffs[1] * y[vals][index + 1] +
                coeffs[2] * ypp[vals][index] + coeffs[3] * ypp[vals][index + 1];
    }

    return V;
  }

  template <typename... TArgs>
  inline vec<num_vars> get_derivative(const T &xin,
                                      const TArgs &... vals) const {
    constexpr int N = sizeof...(vals);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");

    const auto index = find_index(xin);

    const auto A = (x[index + 1] - xin) / deltax[index];
    const auto B = (xin - x[index]) / deltax[index];

    vec<num_vars> V{};
    ((V[vals] = (y[vals][index + 1] - y[vals][index]) / (deltax[index]) -
                 (3. * A * A - 1.) / 6. * deltax[index] * ypp[vals][index] +
                 (3. * B * B - 1.) / 6. * deltax[index] * ypp[vals][index + 1]), ...);
    return V;
  }

  template <typename... Args>
  cubic_spline_t(int __num_points, vec_ptr &&__x, Args &&... args)
      : x(std::move(__x)), num_points(__num_points) {
    static_assert(sizeof...(args) == num_vars,
                  "Need exactly num_vars arguments");
    int n = 0;
    ((y[n] = std::forward<Args>(args),
      ypp[n] = std::make_unique<T[]>(num_points),
      ++n), ...);

    deltax = std::make_unique<T[]>(num_points - 1);
    for (int i = 0; i < num_points - 1; ++i) {
      deltax[i] = x[i + 1] - x[i];
      assert(!std::isnan(deltax[i]));
      assert(deltax[i] > 0.);
    }

    minval = x[0];
    maxval = x[num_points-1];

    compute_ypp();
  }

  // Dummy constructor, can't really be used
  cubic_spline_t() = default;

  template <typename... Args>
  inline void replace_y(Args &&... args) {
    static_assert(sizeof...(args) == num_vars,
                  "Need exactly num_vars arguments");
    int n = 0;
    ((y[n++] = std::forward<Args>(args)), ...);

    minval = x[0];
    maxval = x[num_points-1];

    compute_ypp();
  }
};
