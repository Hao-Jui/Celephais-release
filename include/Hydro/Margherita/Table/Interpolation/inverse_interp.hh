#pragma once

#include "linear_interp.hh"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

template <typename T>
inline void require_finite_strictly_increasing(const T *values, const int count,
                                               const char *name) {
  if (count < 2) {
    throw std::invalid_argument(std::string(name) +
                                " needs at least two interpolation points");
  }

  if (!std::isfinite(values[0])) {
    throw std::invalid_argument(std::string(name) + " contains non-finite data");
  }

  for (int i = 1; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      throw std::invalid_argument(std::string(name) +
                                  " contains non-finite data");
    }
    if (!(values[i] > values[i - 1])) {
      throw std::invalid_argument(std::string(name) +
                                  " must be strictly increasing");
    }
  }
}

template <typename T>
inline linear_interp_t<T, 1> make_inverse_linear_interp(
    const int count, std::unique_ptr<T[]> &&forward_x,
    std::unique_ptr<T[]> &&forward_y, const char *x_name, const char *y_name) {
  require_finite_strictly_increasing(forward_x.get(), count, x_name);
  require_finite_strictly_increasing(forward_y.get(), count, y_name);
  return linear_interp_t<T, 1>(count, std::move(forward_y),
                               std::move(forward_x));
}

template <typename T>
inline linear_interp_t<T, 1> make_monotone_inverse_linear_interp(
    const int count, std::unique_ptr<T[]> &&forward_x,
    std::unique_ptr<T[]> &&forward_y, const char *x_name, const char *y_name) {
  require_finite_strictly_increasing(forward_x.get(), count, x_name);

  auto inverse_x = std::make_unique<T[]>(count);
  auto inverse_y = std::make_unique<T[]>(count);
  int used = 0;

  for (int i = 0; i < count; ++i) {
    if (!std::isfinite(forward_y[i])) {
      throw std::invalid_argument(std::string(y_name) +
                                  " contains non-finite data");
    }
    if (used == 0 || forward_y[i] > inverse_x[used - 1]) {
      inverse_x[used] = forward_y[i];
      inverse_y[used] = forward_x[i];
      ++used;
    }
  }

  if (used < 2) {
    throw std::invalid_argument(std::string(y_name) +
                                " has no increasing inverse branch");
  }

  auto trimmed_x = std::make_unique<T[]>(used);
  auto trimmed_y = std::make_unique<T[]>(used);
  for (int i = 0; i < used; ++i) {
    trimmed_x[i] = inverse_x[i];
    trimmed_y[i] = inverse_y[i];
  }

  return linear_interp_t<T, 1>(used, std::move(trimmed_x),
                               std::move(trimmed_y));
}
