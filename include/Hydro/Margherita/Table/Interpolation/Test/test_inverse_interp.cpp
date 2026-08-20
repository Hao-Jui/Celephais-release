#include "../inverse_interp.hh"
#include "../brent.hh"

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace {

std::unique_ptr<double[]> values(std::initializer_list<double> source) {
  auto ptr = std::make_unique<double[]>(source.size());
  int i = 0;
  for (const auto value : source) {
    ptr[i++] = value;
  }
  return ptr;
}

void exact_linear_round_trip() {
  auto log_rho = values({-2.0, -1.0, 0.0, 1.0, 2.0});
  auto log_press = values({-5.0, -2.5, 0.0, 2.5, 5.0});

  auto inverse = make_inverse_linear_interp(5, std::move(log_rho),
                                            std::move(log_press),
                                            "log rho", "log pressure");

  for (const auto expected_log_rho : {-2.0, -1.5, -0.25, 0.75, 2.0}) {
    const double log_pressure = 2.5 * expected_log_rho;
    const auto actual = inverse.interpolate(log_pressure, 0)[0];
    assert(std::abs(actual - expected_log_rho) < 1.0e-14);
  }
}

void pressure_inverse_matches_brent_diagnostic() {
  auto forward_log_rho = values({-2.0, -1.0, 0.0, 1.0, 2.0});
  auto forward_log_press = values({-5.0, -2.5, 0.0, 2.5, 5.0});
  linear_interp_t<double, 1> forward(5, std::move(forward_log_rho),
                                     std::move(forward_log_press));

  auto inverse_log_rho = values({-2.0, -1.0, 0.0, 1.0, 2.0});
  auto inverse_log_press = values({-5.0, -2.5, 0.0, 2.5, 5.0});
  auto inverse = make_inverse_linear_interp(5, std::move(inverse_log_rho),
                                            std::move(inverse_log_press),
                                            "log rho", "log pressure");

  for (const auto log_pressure : {-4.25, -1.25, 0.0, 1.25, 4.25}) {
    auto residual = [&](const double &log_rho) {
      return log_pressure - forward.interpolate(log_rho, 0)[0];
    };
    const auto brent_log_rho = zero_brent(-2.0, 2.0, 1.0e-13, residual);
    const auto inverse_log_rho_result = inverse.interpolate(log_pressure, 0)[0];
    assert(std::abs(inverse_log_rho_result - brent_log_rho) < 1.0e-13);
  }
}

void rejects_nonmonotone_forward_values() {
  auto log_rho = values({0.0, 1.0, 2.0});
  auto log_press = values({0.0, 1.0, 1.0});

  bool rejected = false;
  try {
    (void)make_inverse_linear_interp(3, std::move(log_rho), std::move(log_press),
                                     "log rho", "log pressure");
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}

void rejects_nonfinite_values() {
  auto log_rho = values({0.0, 1.0, 2.0});
  auto log_press = values({0.0, std::nan(""), 2.0});

  bool rejected = false;
  try {
    (void)make_inverse_linear_interp(3, std::move(log_rho), std::move(log_press),
                                     "log rho", "log pressure");
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}

void builds_from_noisy_monotone_branch() {
  auto log_rho = values({0.0, 1.0, 2.0, 3.0, 4.0});
  auto h = values({1.0, 0.9, 1.1, 1.2, 1.3});

  auto inverse = make_monotone_inverse_linear_interp(5, std::move(log_rho),
                                                     std::move(h), "log rho",
                                                     "specific enthalpy");

  const auto actual = inverse.interpolate(1.15, 0)[0];
  assert(std::abs(actual - 2.5) < 1.0e-14);
}

}  // namespace

int main() {
  exact_linear_round_trip();
  pressure_inverse_matches_brent_diagnostic();
  rejects_nonmonotone_forward_values();
  rejects_nonfinite_values();
  builds_from_noisy_monotone_branch();
}
