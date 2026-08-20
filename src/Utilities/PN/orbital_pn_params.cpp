#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Kadath::PN
{
    namespace
    {
        SpinVector operator+(SpinVector a, SpinVector b)
        {
            return {a.x + b.x, a.y + b.y, a.z + b.z};
        }

        SpinVector operator-(SpinVector a, SpinVector b)
        {
            return {a.x - b.x, a.y - b.y, a.z - b.z};
        }

        SpinVector operator*(double scale, SpinVector v)
        {
            return {scale * v.x, scale * v.y, scale * v.z};
        }

        SpinVector operator/(SpinVector v, double scale)
        {
            return {v.x / scale, v.y / scale, v.z / scale};
        }

        double dot(SpinVector a, SpinVector b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        SpinVector cross(SpinVector a, SpinVector b)
        {
            return {a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x};
        }

        double norm(SpinVector v)
        {
            return std::sqrt(dot(v, v));
        }

        SpinVector zero_vector()
        {
            return {0., 0., 0.};
        }

        bool is_finite(SpinVector v)
        {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        }

        void validate_algebraic_inputs(const AlgebraicPNParameters& params,
                                       double separation,
                                       double center_of_mass)
        {
            if (!std::isfinite(params.mass1) || !std::isfinite(params.mass2))
                throw std::invalid_argument("PN algebraic parameters require finite masses");
            if (params.mass1 <= 0. || params.mass2 <= 0.)
                throw std::invalid_argument("PN algebraic parameters require positive masses");
            if (!std::isfinite(separation) || separation <= 0.)
                throw std::invalid_argument("PN algebraic parameters require positive finite separation");
            if (!std::isfinite(center_of_mass))
                throw std::invalid_argument("PN algebraic parameters require finite center of mass");
        }

        void validate_precessing_frame(const OrbitalFrame& frame)
        {
            if (!is_finite(frame.n_hat) || !is_finite(frame.lambda_hat) || !is_finite(frame.l_hat))
                throw std::invalid_argument("PN precessing frame requires finite vectors");

            constexpr double tolerance = 1.e-10;
            if (std::abs(norm(frame.n_hat) - 1.) > tolerance ||
                std::abs(norm(frame.lambda_hat) - 1.) > tolerance ||
                std::abs(norm(frame.l_hat) - 1.) > tolerance)
                throw std::invalid_argument("PN precessing frame requires unit vectors");

            if (std::abs(dot(frame.n_hat, frame.lambda_hat)) > tolerance ||
                std::abs(dot(frame.n_hat, frame.l_hat)) > tolerance ||
                std::abs(dot(frame.lambda_hat, frame.l_hat)) > tolerance)
                throw std::invalid_argument("PN precessing frame requires orthogonal vectors");

            if (norm(cross(frame.n_hat, frame.lambda_hat) - frame.l_hat) > tolerance)
                throw std::invalid_argument("PN precessing frame must be right handed");
        }

        bool is_z_aligned(SpinVector spin)
        {
            const double scale = std::max(1., norm(spin));
            return std::abs(spin.x) <= 1.e-14 * scale && std::abs(spin.y) <= 1.e-14 * scale;
        }

        bool spins_are_z_aligned(SpinVector spin1, SpinVector spin2)
        {
            return is_z_aligned(spin1) && is_z_aligned(spin2);
        }

        double total_mass(const AlgebraicPNParameters& params)
        {
            return params.mass1 + params.mass2;
        }

        double reduced_mass(const AlgebraicPNParameters& params)
        {
            return params.mass1 * params.mass2 / total_mass(params);
        }

        double symmetric_mass_ratio(const AlgebraicPNParameters& params)
        {
            const double mass = total_mass(params);
            return params.mass1 * params.mass2 / (mass * mass);
        }

        double dimensionless_spin_z(double mass, SpinVector spin)
        {
            return spin.z / (mass * mass);
        }

        double leading_aligned_spin_orbit_inspiral_coefficient(const AlgebraicPNParameters& params,
                                                               SpinVector spin1,
                                                               SpinVector spin2)
        {
            if (!params.include_leading_spin_orbit)
                return 0.;

            const double mass = total_mass(params);
            const double eta = symmetric_mass_ratio(params);
            const double x1 = params.mass1 / mass;
            const double x2 = params.mass2 / mass;
            const double chi1 = dimensionless_spin_z(params.mass1, spin1);
            const double chi2 = dimensionless_spin_z(params.mass2, spin2);
            return (7. / 12.) *
                   ((19. * x1 * x1 + 15. * eta) * chi1 +
                    (19. * x2 * x2 + 15. * eta) * chi2);
        }

        SpinVector leading_spin_orbit_acceleration(const AlgebraicPNParameters& params,
                                                   double separation,
                                                   SpinVector n_hat,
                                                   SpinVector velocity,
                                                   SpinVector spin1,
                                                   SpinVector spin2)
        {
            // Kidder, Phys. Rev. D 52, 821 (1995), Eq. (2.2c), covariant SSC.
            const double mass = total_mass(params);
            const double delta = (params.mass1 - params.mass2) / mass;

            const SpinVector spin = spin1 + spin2;
            const SpinVector sigma = mass * (spin2 / params.mass2 - spin1 / params.mass1);

            const SpinVector spin_a = 2. * spin + delta * sigma;
            const SpinVector spin_b = 7. * spin + 3. * delta * sigma;
            const SpinVector spin_c = 3. * spin + delta * sigma;

            return (1. / std::pow(separation, 3)) *
                   (6. * dot(cross(n_hat, velocity), spin_a) * n_hat - cross(velocity, spin_b) +
                    3. * dot(n_hat, velocity) * cross(n_hat, spin_c));
        }

        SpinVector leading_spin_spin_acceleration(const AlgebraicPNParameters& params,
                                                  double separation,
                                                  SpinVector n_hat,
                                                  SpinVector spin1,
                                                  SpinVector spin2)
        {
            // Kidder, Phys. Rev. D 52, 821 (1995), Eq. (2.2e).
            const double mu = reduced_mass(params);
            const double n_s1 = dot(n_hat, spin1);
            const double n_s2 = dot(n_hat, spin2);
            const double s1_s2 = dot(spin1, spin2);
            const SpinVector bracket =
                s1_s2 * n_hat + n_s2 * spin1 + n_s1 * spin2 - 5. * n_s1 * n_s2 * n_hat;
            return (-3. / (mu * std::pow(separation, 4))) * bracket;
        }

        SpinVector conservative_circular_acceleration(const AlgebraicPNParameters& params,
                                                      double separation,
                                                      const OrbitalFrame& frame,
                                                      double omega,
                                                      SpinVector spin1,
                                                      SpinVector spin2,
                                                      double nonspinning_omega)
        {
            SpinVector acceleration = (-separation * nonspinning_omega * nonspinning_omega) * frame.n_hat;
            const SpinVector velocity = separation * omega * frame.lambda_hat;
            if (params.include_leading_spin_orbit)
                acceleration =
                    acceleration + leading_spin_orbit_acceleration(params,
                                                                   separation,
                                                                   frame.n_hat,
                                                                   velocity,
                                                                   spin1,
                                                                   spin2);
            if (params.include_leading_spin_spin)
                acceleration =
                    acceleration + leading_spin_spin_acceleration(params, separation, frame.n_hat, spin1, spin2);
            return acceleration;
        }

        double circular_force_balance_residual(const AlgebraicPNParameters& params,
                                               double separation,
                                               double omega,
                                               SpinVector spin1,
                                               SpinVector spin2,
                                               double nonspinning_omega)
        {
            const OrbitalFrame aligned_frame;
            const SpinVector acceleration = conservative_circular_acceleration(params,
                                                                               separation,
                                                                               aligned_frame,
                                                                               omega,
                                                                               spin1,
                                                                               spin2,
                                                                               nonspinning_omega);
            const double radial_acceleration = dot(acceleration, aligned_frame.n_hat);

            return radial_acceleration + separation * omega * omega;
        }

        double precessing_force_balance_residual(const AlgebraicPNParameters& params,
                                                 const PrecessingPNState& state,
                                                 double omega,
                                                 SpinVector spin1,
                                                 SpinVector spin2,
                                                 double nonspinning_omega)
        {
            const SpinVector acceleration = conservative_circular_acceleration(params,
                                                                               state.separation,
                                                                               state.frame,
                                                                               omega,
                                                                               spin1,
                                                                               spin2,
                                                                               nonspinning_omega);
            return dot(acceleration, state.frame.n_hat) + state.separation * omega * omega;
        }

        double solve_aligned_spin_circular_omega(const AlgebraicPNParameters& params,
                                                 double separation,
                                                 SpinVector spin1,
                                                 SpinVector spin2,
                                                 double nonspinning_omega)
        {
            if ((!params.include_leading_spin_orbit && !params.include_leading_spin_spin) ||
                (spin1.z == 0. && spin2.z == 0.))
                return nonspinning_omega;

            double low = 0.;
            double high = std::max(2. * nonspinning_omega, 1.e-16);
            double f_low =
                circular_force_balance_residual(params, separation, low, spin1, spin2, nonspinning_omega);
            double f_high =
                circular_force_balance_residual(params, separation, high, spin1, spin2, nonspinning_omega);

            for (int i = 0; i < 32 && f_low * f_high > 0.; ++i)
            {
                high *= 2.;
                f_high = circular_force_balance_residual(params, separation, high, spin1, spin2, nonspinning_omega);
            }

            if (f_low * f_high > 0.)
                return nonspinning_omega;

            for (int i = 0; i < 96; ++i)
            {
                const double mid = 0.5 * (low + high);
                const double f_mid =
                    circular_force_balance_residual(params, separation, mid, spin1, spin2, nonspinning_omega);
                if (f_low * f_mid <= 0.)
                {
                    high = mid;
                    f_high = f_mid;
                }
                else
                {
                    low = mid;
                    f_low = f_mid;
                }
            }

            return 0.5 * (low + high);
        }

        double solve_precessing_circular_omega(const AlgebraicPNParameters& params,
                                               const PrecessingPNState& state,
                                               SpinVector spin1,
                                               SpinVector spin2,
                                               double nonspinning_omega)
        {
            if ((!params.include_leading_spin_orbit && !params.include_leading_spin_spin) ||
                (norm(spin1) == 0. && norm(spin2) == 0.))
                return nonspinning_omega;

            double low = 0.;
            double high = std::max(2. * nonspinning_omega, 1.e-16);
            double f_low = precessing_force_balance_residual(params,
                                                             state,
                                                             low,
                                                             spin1,
                                                             spin2,
                                                             nonspinning_omega);
            double f_high = precessing_force_balance_residual(params,
                                                              state,
                                                              high,
                                                              spin1,
                                                              spin2,
                                                              nonspinning_omega);

            for (int i = 0; i < 32 && f_low * f_high > 0.; ++i)
            {
                high *= 2.;
                f_high = precessing_force_balance_residual(params,
                                                           state,
                                                           high,
                                                           spin1,
                                                           spin2,
                                                           nonspinning_omega);
            }

            if (f_low * f_high > 0.)
                return nonspinning_omega;

            for (int i = 0; i < 96; ++i)
            {
                const double mid = 0.5 * (low + high);
                const double f_mid = precessing_force_balance_residual(params,
                                                                       state,
                                                                       mid,
                                                                       spin1,
                                                                       spin2,
                                                                       nonspinning_omega);
                if (f_low * f_mid <= 0.)
                {
                    high = mid;
                    f_high = f_mid;
                }
                else
                {
                    low = mid;
                    f_low = f_mid;
                }
            }

            return 0.5 * (low + high);
        }

        AlgebraicPNOrbitalParameters circular_parameters_at_separation(const AlgebraicPNParameters& params,
                                                                       double separation,
                                                                       SpinVector spin1,
                                                                       SpinVector spin2,
                                                                       double center_of_mass)
        {
            validate_algebraic_inputs(params, separation, center_of_mass);

            const double body1_radius = std::abs(-separation / 2. + center_of_mass);
            const double body2_radius = std::abs(separation / 2. + center_of_mass);
            const double mass = total_mass(params);
            const double eta = symmetric_mass_ratio(params);
            const double gamma = mass / separation;
            const double mass_fraction1 = params.mass1 / mass;
            const double mass_fraction2 = params.mass2 / mass;
            const double log_scale =
                mass_fraction1 * std::log(body1_radius + 1.) + mass_fraction2 * std::log(body2_radius + 1.);

            const double omega_squared =
                mass / std::pow(separation, 3) *
                (1. + (-3. + eta) * gamma +
                 (6. + 41. / 4. * eta + std::pow(eta, 2)) * std::pow(gamma, 2) +
                 (-10. +
                  (-75707. / 840. + 41. / 64. * std::pow(M_PI, 2) +
                   22. * (std::log(separation) - log_scale)) *
                      eta +
                  19. / 2. * std::pow(eta, 2) + std::pow(eta, 3)) *
                     std::pow(gamma, 3));

            const double radial_velocity =
                -64. / 5. * std::pow(mass, 3) * eta / std::pow(separation, 3) *
                (1. + gamma * (-1751. / 336. - 7. / 4. * eta));
            AlgebraicPNOrbitalParameters orbital_params{
                std::sqrt(omega_squared),
                radial_velocity / separation,
                radial_velocity,
            };
            if (!spins_are_z_aligned(spin1, spin2))
                return orbital_params;

            orbital_params.omega =
                solve_aligned_spin_circular_omega(params, separation, spin1, spin2, orbital_params.omega);

            const double spin_orbit_inspiral =
                leading_aligned_spin_orbit_inspiral_coefficient(params, spin1, spin2);
            orbital_params.radial_velocity *= 1. - spin_orbit_inspiral * std::pow(gamma, 1.5);
            orbital_params.adot = orbital_params.radial_velocity / separation;
            return orbital_params;
        }

        SpinVector orbital_angular_momentum(const AlgebraicPNParameters& params,
                                            const PrecessingPNState& state,
                                            double omega)
        {
            const SpinVector position = state.separation * state.frame.n_hat;
            const SpinVector velocity = state.separation * omega * state.frame.lambda_hat;
            return reduced_mass(params) * cross(position, velocity);
        }

        SpinVector perpendicular_to(SpinVector vector, SpinVector axis)
        {
            return vector - dot(vector, axis) * axis;
        }

        SpinVector orbital_axis_derivative(const AlgebraicPNParameters& params,
                                           const PrecessingPNState& state,
                                           double omega,
                                           SpinVector acceleration)
        {
            const SpinVector position = state.separation * state.frame.n_hat;
            const SpinVector angular_momentum = orbital_angular_momentum(params, state, omega);
            const double angular_momentum_norm = norm(angular_momentum);
            if (angular_momentum_norm == 0.)
                return zero_vector();

            const SpinVector angular_momentum_dot = reduced_mass(params) * cross(position, acceleration);
            return perpendicular_to(angular_momentum_dot / angular_momentum_norm, state.frame.l_hat);
        }

        SpinVector leading_spin_orbit_precession(const AlgebraicPNParameters& params,
                                                 const PrecessingPNState& state,
                                                 double omega,
                                                 SpinVector dimensionful_spin,
                                                 double spin_mass,
                                                 bool first_body)
        {
            // Leading spin-orbit precession: dS/dt = Omega_SO x S.
            if (!params.include_leading_spin_orbit)
                return zero_vector();

            const double companion_mass = first_body ? params.mass2 : params.mass1;
            const double precession_weight = 2. + 1.5 * companion_mass / spin_mass;
            const SpinVector precession_frequency =
                (precession_weight / std::pow(state.separation, 3)) *
                orbital_angular_momentum(params, state, omega);
            return cross(precession_frequency, dimensionful_spin) / (spin_mass * spin_mass);
        }

        SpinVector tilted_dimensionless_spin(double dimensionless_spin, double spin_axis_degrees)
        {
            // Place the spin in the n_hat-l_hat plane at the requested tilt from the
            // orbital axis: in-plane (perpendicular) component along n_hat, aligned
            // (parallel) component along l_hat.
            const double radians = spin_axis_degrees * std::acos(-1.) / 180.;
            return {dimensionless_spin * std::sin(radians), 0., dimensionless_spin * std::cos(radians)};
        }
    } // namespace

    SpinVector make_dimensionful_spin(double mass, SpinVector dimensionless_spin)
    {
        const double mass2 = mass * mass;
        return mass2 * dimensionless_spin;
    }

    double dimensionless_spin_along_orbital_axis(double dimensionless_spin, double spin_axis_degrees)
    {
        if (!std::isfinite(dimensionless_spin) || !std::isfinite(spin_axis_degrees))
            throw std::invalid_argument("PN spin projection requires finite spin and axis angle");
        return dimensionless_spin * std::cos(spin_axis_degrees * std::acos(-1.) / 180.);
    }

    AlgebraicPNOrbitalParameters aligned_spin_algebraic_pn_parameters(double mass1,
                                                                      double mass2,
                                                                      double separation,
                                                                      SpinVector dimensionless_spin1,
                                                                      SpinVector dimensionless_spin2,
                                                                      double center_of_mass)
    {
        AlgebraicPNParameters params;
        params.mass1 = mass1;
        params.mass2 = mass2;
        return aligned_spin_algebraic_pn_parameters(params,
                                                    separation,
                                                    dimensionless_spin1,
                                                    dimensionless_spin2,
                                                    center_of_mass);
    }

    AlgebraicPNOrbitalParameters aligned_spin_algebraic_pn_parameters(const AlgebraicPNParameters& params,
                                                                      double separation,
                                                                      SpinVector dimensionless_spin1,
                                                                      SpinVector dimensionless_spin2,
                                                                      double center_of_mass)
    {
        if (!is_finite(dimensionless_spin1) || !is_finite(dimensionless_spin2))
            throw std::invalid_argument("PN aligned-spin parameters require finite dimensionless spins");
        if (!spins_are_z_aligned(dimensionless_spin1, dimensionless_spin2))
            throw std::invalid_argument("PN aligned-spin parameters require spins parallel to the z axis");

        return circular_parameters_at_separation(params,
                                                 separation,
                                                 make_dimensionful_spin(params.mass1, dimensionless_spin1),
                                                 make_dimensionful_spin(params.mass2, dimensionless_spin2),
                                                 center_of_mass);
    }

    PrecessingPNOrbitalParameters precessing_algebraic_pn_parameters(double mass1,
                                                                     double mass2,
                                                                     const PrecessingPNState& state)
    {
        AlgebraicPNParameters params;
        params.mass1 = mass1;
        params.mass2 = mass2;
        return precessing_algebraic_pn_parameters(params, state);
    }

    PrecessingPNOrbitalParameters precessing_algebraic_pn_parameters(const AlgebraicPNParameters& params,
                                                                     const PrecessingPNState& state)
    {
        validate_algebraic_inputs(params, state.separation, state.center_of_mass);
        validate_precessing_frame(state.frame);
        if (!is_finite(state.dimensionless_spin1) || !is_finite(state.dimensionless_spin2))
            throw std::invalid_argument("PN precessing parameters require finite dimensionless spins");

        const SpinVector spin1 = make_dimensionful_spin(params.mass1, state.dimensionless_spin1);
        const SpinVector spin2 = make_dimensionful_spin(params.mass2, state.dimensionless_spin2);

        const AlgebraicPNOrbitalParameters nonspinning =
            circular_parameters_at_separation(params,
                                              state.separation,
                                              zero_vector(),
                                              zero_vector(),
                                              state.center_of_mass);
        const double omega = solve_precessing_circular_omega(params,
                                                             state,
                                                             spin1,
                                                             spin2,
                                                             nonspinning.omega);

        const double aligned_chi1 = dot(state.dimensionless_spin1, state.frame.l_hat);
        const double aligned_chi2 = dot(state.dimensionless_spin2, state.frame.l_hat);
        const AlgebraicPNOrbitalParameters inspiral =
            circular_parameters_at_separation(params,
                                              state.separation,
                                              make_dimensionful_spin(params.mass1, {0., 0., aligned_chi1}),
                                              make_dimensionful_spin(params.mass2, {0., 0., aligned_chi2}),
                                              state.center_of_mass);

        const SpinVector acceleration = conservative_circular_acceleration(params,
                                                                           state.separation,
                                                                           state.frame,
                                                                           omega,
                                                                           spin1,
                                                                           spin2,
                                                                           nonspinning.omega);

        return {omega,
                inspiral.adot,
                inspiral.radial_velocity,
                acceleration,
                orbital_axis_derivative(params, state, omega, acceleration),
                leading_spin_orbit_precession(params, state, omega, spin1, params.mass1, true),
                leading_spin_orbit_precession(params, state, omega, spin2, params.mass2, false)};
    }

    PrecessingPNOrbitalParameters precessing_seed_from_tilt(double mass1,
                                                            double mass2,
                                                            double separation,
                                                            double dimensionless_spin1,
                                                            double spin1_axis_degrees,
                                                            double dimensionless_spin2,
                                                            double spin2_axis_degrees,
                                                            double center_of_mass)
    {
        if (!std::isfinite(dimensionless_spin1) || !std::isfinite(spin1_axis_degrees) ||
            !std::isfinite(dimensionless_spin2) || !std::isfinite(spin2_axis_degrees))
            throw std::invalid_argument("PN tilt seed requires finite spins and axis angles");

        AlgebraicPNParameters params;
        params.mass1 = mass1;
        params.mass2 = mass2;

        PrecessingPNState state;
        state.separation = separation;
        state.dimensionless_spin1 = tilted_dimensionless_spin(dimensionless_spin1, spin1_axis_degrees);
        state.dimensionless_spin2 = tilted_dimensionless_spin(dimensionless_spin2, spin2_axis_degrees);
        state.center_of_mass = center_of_mass;

        return precessing_algebraic_pn_parameters(params, state);
    }
} // namespace Kadath::PN
