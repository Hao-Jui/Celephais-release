/*
 * =====================================================================================
 *       Filename:  tov.hh
 *    Description:  Margherita TOV: Simple TOV solving routines
 *        Version:  2.0
 *         Author:  Samuel David Tootle
 * =====================================================================================
 */
#pragma once

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace Kadath
{
    namespace Margherita
    {

        template <typename EOS> class MargheritaTOV
        {
          public:
            enum { RADIUS = 0, RHOB, PRESS, MASSR, MASSB, PHI, RISO, CONF, TIDALH, TIDALBETA, NUMVAR };

            using eos_t = EOS;

          private:
            using ary_t = std::array<double, NUMVAR>;
            using state_t = std::vector<ary_t>;

            static constexpr std::size_t max_iter = 40000;
            static constexpr double dr0 = 10.0 / 10000.0;
            static constexpr double loop_eps = 1e-8;

          public:
            bool compact_output{false};
            double radius{0};
            double arealr{0};
            double mass{0};
            double rhoc{0};
            double baryon_mass{0};
            double tidal_love_k2{0};
            double press_c{0};
            state_t state{};

            void reset(double rho_init)
            {
                state.clear();
                state.reserve(max_iter);
                radius = 0;
                arealr = 0;
                mass = 0;
                rhoc = rho_init;
                baryon_mass = 0;
                tidal_love_k2 = 0;
                press_c = 0;
            }

            ary_t gen_initial()
            {
                double eps_cold = 0.0;
                typename EOS::error_t error;
                press_c = EOS::press_cold_eps_cold__rho(eps_cold, rhoc, error);

                ary_t initial{};
                initial[RHOB] = rhoc;
                initial[PRESS] = press_c;
                initial[PHI] = 1.0;
                return initial;
            }

            ary_t evolve(const ary_t& input, double step_size) const
            {
                ary_t derivative{};
                if (input[RADIUS] == 0.0) {
                    return derivative;
                }

                typename EOS::error_t error;
                const double radius_value = input[RADIUS];
                const double radius_squared = radius_value * radius_value;
                const double mass_inside_radius = input[MASSR];
                double pressure = input[PRESS];

                double energy_density = 0.0;
                double energy_derivative_by_pressure = 0.0;
                const double rest_mass_density =
                    EOS::rho_energy_dedp__press_cold(energy_density, energy_derivative_by_pressure, pressure, error);

                const double radius_cubed = radius_squared * radius_value;
                const double schwarzschild_factor = radius_value / (radius_value - 2.0 * mass_inside_radius);

                // Tichy, arXiv:1610.03805, eqs. 178-180.
                derivative[MASSR] = 4.0 * M_PI * radius_squared * energy_density * step_size;
                derivative[PHI] = (mass_inside_radius + 4.0 * M_PI * radius_cubed * pressure) /
                                  (radius_value * (radius_value - 2.0 * mass_inside_radius)) * step_size;
                derivative[PRESS] = -derivative[PHI] * (energy_density + pressure);

                derivative[RISO] = (std::sqrt(schwarzschild_factor) - 1.0) / radius_value * step_size;
                derivative[MASSB] =
                    4.0 * M_PI * radius_squared * rest_mass_density * std::sqrt(schwarzschild_factor) * step_size;

                double tidal_h = input[TIDALH];
                double tidal_beta = input[TIDALBETA];
                if (state.size() == 1) {
                    tidal_h = radius_squared;
                    tidal_beta = 2.0 * radius_value;
                }

                // Hinderer, arXiv:0911.3535, eqs. 11-12.
                derivative[TIDALH] = tidal_beta * step_size;
                const double tidal_source = mass_inside_radius / radius_squared + 4.0 * M_PI * radius_value * pressure;
                derivative[TIDALBETA] =
                    2.0 * schwarzschild_factor * tidal_h *
                        (-2.0 * M_PI *
                             (5.0 * energy_density + 9.0 * pressure +
                              energy_derivative_by_pressure * (energy_density + pressure)) +
                         3.0 / radius_squared + 2.0 * schwarzschild_factor * tidal_source * tidal_source) +
                    2.0 * tidal_beta / radius_value * schwarzschild_factor *
                        (-1.0 + mass_inside_radius / radius_value +
                         2.0 * M_PI * radius_squared * (energy_density - pressure));
                derivative[TIDALBETA] *= step_size;

                return derivative;
            }

            void correct_phi()
            {
                const ary_t& surface = state.back();
                const double correction = 0.5 * std::log(1.0 - 2.0 * surface[MASSR] / surface[RADIUS]) - surface[PHI];

                for (ary_t& sample : state) {
                    sample[PHI] += correction;
                }
            }

            void correct_isotropic_r()
            {
                for (ary_t& sample : state) {
                    sample[RISO] = sample[RADIUS] * std::exp(sample[RISO]);
                }

                const ary_t& surface = state.back();
                const double correction =
                    (std::sqrt(surface[RADIUS] * surface[RADIUS] - 2.0 * surface[MASSR] * surface[RADIUS]) +
                     surface[RADIUS] - surface[MASSR]) /
                    (2.0 * surface[RISO]);

                for (ary_t& sample : state) {
                    sample[RISO] *= correction;
                }
            }

            void calculate_conformal_factor()
            {
                for (ary_t& sample : state) {
                    sample[CONF] = std::sqrt(sample[RADIUS] / sample[RISO]);
                }
            }

            void calculate_k2()
            {
                const ary_t& surface = state.back();
                const double y = surface[RADIUS] * surface[TIDALBETA] / surface[TIDALH];
                const double compactness = surface[MASSR] / surface[RADIUS];
                const double compactness2 = compactness * compactness;
                const double compactness3 = compactness2 * compactness;
                const double compactness5 = compactness3 * compactness2;
                const double redshift_factor = 1.0 - 2.0 * compactness;
                const double redshift_factor2 = redshift_factor * redshift_factor;

                tidal_love_k2 =
                    8.0 * compactness5 / 5.0 * redshift_factor2 * (2.0 + 2.0 * compactness * (y - 1.0) - y) /
                    (2.0 * compactness * (6.0 - 3.0 * y + 3.0 * compactness * (5.0 * y - 8.0)) +
                     4.0 * compactness3 *
                         (13.0 - 11.0 * y + compactness * (3.0 * y - 2.0) + 2.0 * compactness2 * (1.0 + y)) +
                     3.0 * redshift_factor2 * (2.0 - y + 2.0 * compactness * (y - 1.0)) * std::log(redshift_factor));
            }

            void solve(double rho_init, bool finish_isotropic_fields = true)
            {
                reset(rho_init);

                double step_size = dr0;
                state.push_back(gen_initial());

                std::size_t count = 0;
                while (count < max_iter) {
                    ary_t next = rk4_step(state.back(), step_size);
                    const double mass_change = std::abs(next[MASSR] - state.back()[MASSR]);
                    if ((mass_change < loop_eps && state.back()[PRESS] < loop_eps) || next[PRESS] < 0.0 ||
                        !std::isfinite(next[PRESS])) {
                        break;
                    }

                    state.push_back(next);
                    ++count;
                }

                if (count >= max_iter) {
                    throw std::runtime_error("Maximum TOV iterations reached; central density may not produce "
                                             "a stable static TOV solution");
                }

                if (count == 0) {
                    return;
                }

                state.erase(state.begin());
                finish_solution(finish_isotropic_fields);
            }

            friend std::ostream& operator<<(std::ostream& stream, const MargheritaTOV& tov)
            {
                if (tov.compact_output) {
                    stream << std::setprecision(12) << tov.press_c << "\t" << tov.mass << "\t" << tov.baryon_mass
                           << "\t" << tov.radius << "\t" << tov.tidal_love_k2 << std::endl;
                    return stream;
                }

                stream << " Central pressure: " << tov.press_c << std::endl;
                stream << " Central rest mass density: " << tov.rhoc << std::endl;
                stream << " Mass [Mo]: " << tov.mass << std::endl;
                stream << " Baryon mass [Mo]: " << tov.baryon_mass << std::endl;
                stream << " Radius [km]: " << tov.arealr * 1.4769994423016508 << std::endl;
                stream << " Isotropic Radius [km]: " << tov.radius * 1.4769994423016508 << std::endl;
                stream << " Love number k2: " << tov.tidal_love_k2 << std::endl;
                return stream;
            }

          private:
            static void add_scaled(ary_t& values, const ary_t& derivative, double weight)
            {
                for (int variable = 1; variable < NUMVAR; ++variable) {
                    values[variable] += weight * derivative[variable];
                }
            }

            ary_t rk4_step(const ary_t& input, double step_size) const
            {
                const ary_t k1 = evolve(input, step_size);

                ary_t trial = input;
                trial[RADIUS] += step_size / 2.0;
                add_scaled(trial, k1, 0.5);
                const ary_t k2 = evolve(trial, step_size);

                trial = input;
                trial[RADIUS] += step_size / 2.0;
                add_scaled(trial, k2, 0.5);
                const ary_t k3 = evolve(trial, step_size);

                trial = input;
                trial[RADIUS] += step_size;
                add_scaled(trial, k3, 1.0);
                const ary_t k4 = evolve(trial, step_size);

                ary_t result = input;
                for (int variable = 1; variable < NUMVAR; ++variable) {
                    result[variable] += (k1[variable] + 2.0 * k2[variable] + 2.0 * k3[variable] + k4[variable]) / 6.0;
                }

                finish_step(result, step_size);
                return result;
            }

            static void finish_step(ary_t& result, double step_size)
            {
                typename EOS::error_t error;
                result[RADIUS] += step_size;
                result[RHOB] = EOS::rho__press_cold(result[PRESS], error);
            }

            void finish_solution(bool finish_isotropic_fields)
            {
                if (finish_isotropic_fields) {
                    correct_phi();
                    correct_isotropic_r();
                    calculate_conformal_factor();
                    radius = state.back()[RISO];
                    calculate_k2();
                }

                arealr = state.back()[RADIUS];
                mass = state.back()[MASSR];
                baryon_mass = state.back()[MASSB];
            }
        };

    } // namespace Margherita
} // namespace Kadath
