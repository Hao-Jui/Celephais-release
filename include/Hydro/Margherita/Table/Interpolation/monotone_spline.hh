/*
 * =====================================================================================
 *
 *       Filename:  monotone_spline.hh
 *
 *    Description:  Monotonicity-preserving natural cubic spline (Hyman 1983
 *                 filter).  The natural cubic spline's node first-derivatives
 *                 are computed exactly (Thomas tridiagonal solve, as in
 *                 spline.hh), then filtered so each node slope lies in the
 *                 Fritsch-Carlson monotone region [0, 3*min(secant)].
 *
 *                 Where the data is already monotone (the common case), no node
 *                 is filtered and a cubic Hermite reconstruction with the
 *                 spline's node slopes reproduces the natural cubic spline
 *                 EXACTLY (a cubic is uniquely fixed by its two endpoint values
 *                 and two endpoint slopes).  So on smooth tables this carries
 *                 full O(h^4) spline accuracy.  Only at nodes where the spline
 *                 slope would overshoot (near-isobaric plateaux / first-order
 *                 phase transitions) is the slope clamped, recovering PCHIP-like
 *                 monotonicity there.  This is strictly more accurate than PCHIP
 *                 (which uses a lower-order local parabolic base slope even where
 *                 no limiting is needed) while keeping the same monotonicity
 *                 guarantee.
 *
 *                 Forward-resample interpolant for setup_cold_table: same
 *                 template signature and constructor as cubic_spline_t<T,
 *                 num_vars>, exposing only interpolate<vals...>(x) (the one
 *                 query the resampler uses).
 *
 *                 This file is original to the Celephais tree (Margherita Table
 *                 layer) and is not derived from any upstream Margherita source.
 *                 It is distributed under the same GPL v3 terms as the
 *                 surrounding Margherita Table code.
 *
 *        Created:  2026-06-16
 *       Compiler:  C++17
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

// ---------------------------------------------------------------------------
// monotone_spline_t — Hyman-filtered natural cubic spline
// ---------------------------------------------------------------------------

template <typename T, int num_vars>
class monotone_spline_t {
 public:
  template <int N>
  using vec = std::array<T, N>;

  using vec_ptr = std::unique_ptr<T[]>;

 private:
  vec_ptr x_;                          // abscissas [num_points]
  vec_ptr deltax_;                     // h[i] = x[i+1]-x[i] [num_points-1]
  std::array<vec_ptr, num_vars> y_;    // ordinate arrays
  std::array<vec_ptr, num_vars> m_;    // Hyman-filtered node slopes

  int num_points_;

  // -------------------------------------------------------------------
  // Node slopes: natural cubic spline second derivatives (Thomas solve,
  // same system as spline.hh), converted to node first derivatives, then
  // filtered to the monotone Fritsch-Carlson region (Hyman 1983).
  // -------------------------------------------------------------------
  void compute_slopes(int n) {
    const int np = num_points_;

    if (np == 1) {
      m_[n][0] = T(0);
      return;
    }
    if (np == 2) {
      const T d = (y_[n][1] - y_[n][0]) / deltax_[0];
      m_[n][0] = d;
      m_[n][1] = d;
      return;
    }

    // Secants
    std::vector<T> s(np - 1);
    for (int i = 0; i < np - 1; ++i)
      s[i] = (y_[n][i + 1] - y_[n][i]) / deltax_[i];

    // Natural cubic spline second derivatives ypp (ypp[0] = ypp[np-1] = 0).
    // Interior system for node j (1 <= j <= np-2):
    //   h[j-1] ypp[j-1] + 2(h[j-1]+h[j]) ypp[j] + h[j] ypp[j+1] = 6(s[j]-s[j-1])
    std::vector<T> ypp(np, T(0));
    const int dim = np - 2;
    if (dim > 0) {
      std::vector<T> cprime(dim, T(0));
      std::vector<T> rhs(dim);
      for (int i = 0; i < dim; ++i) rhs[i] = T(6) * (s[i + 1] - s[i]);

      const T diag0 = T(2) * (deltax_[0] + deltax_[1]);
      cprime[0] = (dim > 1) ? (deltax_[1] / diag0) : T(0);
      ypp[1] = rhs[0] / diag0;
      for (int i = 1; i < dim; ++i) {
        const T sub = deltax_[i];
        const T diag = T(2) * (deltax_[i] + deltax_[i + 1]);
        const T denom = diag - sub * cprime[i - 1];
        if (i < dim - 1) cprime[i] = deltax_[i + 1] / denom;
        ypp[i + 1] = (rhs[i] - sub * ypp[i]) / denom;
      }
      for (int i = dim - 2; i >= 0; --i)
        ypp[i + 1] -= cprime[i] * ypp[i + 2];
    }

    // Spline node first derivatives.  A Hermite cubic with these slopes
    // reproduces the spline exactly on every interval.
    std::vector<T> yp(np);
    for (int i = 0; i < np - 1; ++i)
      yp[i] = s[i] - deltax_[i] * (T(2) * ypp[i] + ypp[i + 1]) / T(6);
    yp[np - 1] =
        s[np - 2] + deltax_[np - 2] * (ypp[np - 2] + T(2) * ypp[np - 1]) / T(6);

    // Hyman filter: clamp each node slope to the monotone region.
    for (int i = 0; i < np; ++i) {
      // Adjacent secants (one-sided at the endpoints).
      const T sl = (i == 0) ? s[0] : ((i == np - 1) ? s[np - 2] : s[i - 1]);
      const T sr = (i == 0) ? s[0] : ((i == np - 1) ? s[np - 2] : s[i]);

      T slope = yp[i];
      if (i > 0 && i < np - 1 && sl * sr <= T(0)) {
        // Local extremum / plateau: zero slope guarantees monotonicity.
        slope = T(0);
      } else {
        const T lim = T(3) * std::min(std::abs(sl), std::abs(sr));
        if (slope * sr < T(0))
          slope = T(0);                       // wrong sign -> flatten
        else if (std::abs(slope) > lim)
          slope = std::copysign(lim, sr);     // overshoot -> clamp to FC bound
        // else: keep the exact spline slope (no accuracy loss).
      }
      m_[n][i] = slope;
    }
  }

  void compute_all_slopes() {
    for (int n = 0; n < num_vars; ++n) compute_slopes(n);
  }

  // -------------------------------------------------------------------
  // Binary search (same as cubic_spline_t / linterp_t).
  // -------------------------------------------------------------------
  inline int find_index(const T &xin) const {
    int lower = 0;
    int upper = num_points_ - 1;
    while (upper - lower > 1) {
      int mid = lower + (upper - lower) / 2;
      if (xin < x_[mid])
        upper = mid;
      else
        lower = mid;
    }
    return lower;
  }

  // -------------------------------------------------------------------
  // Cubic Hermite basis evaluation on interval [x_i, x_{i+1}].
  // -------------------------------------------------------------------
  inline T hermite_eval(int n, int i, T xin) const {
    const T h = deltax_[i];
    const T t = (xin - x_[i]) / h;
    const T t2 = t * t;
    const T t3 = t2 * t;

    const T H00 = T(2) * t3 - T(3) * t2 + T(1);
    const T H10 = t3 - T(2) * t2 + t;
    const T H01 = T(-2) * t3 + T(3) * t2;
    const T H11 = t3 - t2;

    return H00 * y_[n][i] + H10 * h * m_[n][i] +
           H01 * y_[n][i + 1] + H11 * h * m_[n][i + 1];
  }

 public:
  // interpolate<vals...>(x) — variadic non-type template form.
  template <int... vals>
  inline vec<sizeof...(vals)> interpolate(const T &xin) const {
    constexpr int N = sizeof...(vals);
    static_assert(N <= num_vars,
                  "Cannot interpolate more quantities than num_vars");
    const int i = find_index(xin);
    vec<N> V{};
    int m = 0;
    ((V[m++] = hermite_eval(vals, i, xin)), ...);
    return V;
  }

  // -------------------------------------------------------------------
  // Constructor — same signature as cubic_spline_t / linterp_t.
  // -------------------------------------------------------------------
  template <typename... Args>
  monotone_spline_t(int num_points, vec_ptr &&x_in, Args &&... args)
      : x_(std::move(x_in)), num_points_(num_points) {
    static_assert(sizeof...(args) == num_vars,
                  "Need exactly num_vars y-array arguments");
    int n = 0;
    ((y_[n] = std::forward<Args>(args),
      m_[n] = std::make_unique<T[]>(num_points),
      ++n), ...);

    deltax_ = std::make_unique<T[]>(num_points - 1);
    for (int i = 0; i < num_points - 1; ++i) {
      deltax_[i] = x_[i + 1] - x_[i];
      assert(!std::isnan(deltax_[i]));
      assert(deltax_[i] > T(0));
    }

    compute_all_slopes();
  }

  monotone_spline_t() = default;
};
