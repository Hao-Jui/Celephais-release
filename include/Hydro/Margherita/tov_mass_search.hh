#pragma once

#include "tov.hh"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace Kadath
{
    namespace Margherita
    {

        template <typename Tov>
        bool solve_tov_for_adm_mass(Tov& tov, double target_mass, double rho_max = 1e-2, double rho_min_start = 5e-4,
                                    double tolerance = 1e-3)
        {
            constexpr std::size_t max_mass_search_iterations = 40000;

            const double density_step = (rho_max - rho_min_start) / 200.0;
            double lower_density = 0.0;
            double upper_density = 0.0;
            double best_mass = 0.0;
            double best_density = rho_min_start;
            bool target_bracketed = false;

            for (double density = rho_min_start; density < rho_max; density += density_step) {
                tov.solve(density, false);

                if (tov.mass > best_mass) {
                    best_mass = tov.mass;
                    best_density = density;
                }

                if (tov.mass < target_mass) {
                    lower_density = density;
                } else {
                    upper_density = density;
                    target_bracketed = true;
                    break;
                }
            }

            if (!target_bracketed) {
                using eos_t = typename Tov::eos_t;
                if constexpr (requires { eos_t::rhomax; }) {
                    const double eos_rho_max = eos_t::rhomax;
                    if (std::isfinite(eos_rho_max) && eos_rho_max > rho_max) {
                        return solve_tov_for_adm_mass(tov, target_mass, eos_rho_max, rho_max, tolerance);
                    }
                }

                std::cerr << "Maximum density (" << rho_max << ") reached.\n"
                          << "Maximum mass obtained: " << best_mass << ", with density: " << best_density << ".\n"
                          << "The density bracketing range may not be sufficiently constrained for the given EOS\n";

                if (best_mass > 1.0) {
                    tov.solve(best_density, true);
                    return true;
                }
                throw std::runtime_error("Unable to bracket a stable TOV maximum mass");
            }

            double density = 0.5 * (lower_density + upper_density);
            double mass_error = 1.0;
            std::size_t iteration = 0;
            while (std::fabs(mass_error) > tolerance && iteration < max_mass_search_iterations) {
                density = 0.5 * (lower_density + upper_density);
                tov.solve(density, false);
                mass_error = tov.mass - target_mass;

                if (mass_error > 0.0) {
                    upper_density = density;
                } else {
                    lower_density = density;
                }
                ++iteration;
            }

            if (iteration >= max_mass_search_iterations) {
                throw std::runtime_error("Maximum TOV mass-solve iterations reached; target mass may be too "
                                         "high or the density range too narrow");
            }

            tov.solve(density, true);
            return false;
        }

    } // namespace Margherita
} // namespace Kadath
