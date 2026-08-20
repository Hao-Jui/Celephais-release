#pragma once

#include "For_Kadath/Config/config_binary.hpp"
#include <cmath>

using namespace Kadath;

namespace bco_utils
{
    inline double com_estimate(const double distance, const double M1, const double M2)
    {
        double half_dist = distance / 2.;
        double mass_sum = M1 + M2;
        return half_dist * (M1 - M2) / mass_sum;
    }

    template <typename config_t> double set_decay(config_t& bconfig, NODES bco)
    {
        if (std::isnan(bconfig.set(DECAY, bco)))
            bconfig.set(DECAY, bco) = bconfig(DIST) / 2.;
        const double weight4 = std::pow(bconfig(DECAY, bco), 4.);
        return 1. / weight4;
    }
} // namespace bco_utils

namespace Kadath::PN
{
    struct SpinVector
    {
        double x = 0.;
        double y = 0.;
        double z = 0.;
    };
    using CartesianVector = SpinVector;

    struct AlgebraicPNParameters
    {
        double mass1 = 1.;
        double mass2 = 1.;
        bool include_leading_spin_orbit = true;
        bool include_leading_spin_spin = true;
    };

    struct AlgebraicPNOrbitalParameters
    {
        double omega = 0.;
        double adot = 0.;
        double radial_velocity = 0.;
    };

    struct OrbitalFrame
    {
        CartesianVector n_hat{1., 0., 0.};
        CartesianVector lambda_hat{0., 1., 0.};
        CartesianVector l_hat{0., 0., 1.};
    };

    struct PrecessingPNState
    {
        double separation = 0.;
        SpinVector dimensionless_spin1{};
        SpinVector dimensionless_spin2{};
        OrbitalFrame frame{};
        double center_of_mass = 0.;
    };

    struct PrecessingPNOrbitalParameters
    {
        double omega = 0.;
        double adot = 0.;
        double radial_velocity = 0.;
        CartesianVector acceleration{};
        CartesianVector l_hat_dot{};
        SpinVector dimensionless_spin1_dot{};
        SpinVector dimensionless_spin2_dot{};
    };

    SpinVector make_dimensionful_spin(double mass, SpinVector dimensionless_spin);
    double dimensionless_spin_along_orbital_axis(double dimensionless_spin, double spin_axis_degrees);

    AlgebraicPNOrbitalParameters aligned_spin_algebraic_pn_parameters(double mass1,
                                                                      double mass2,
                                                                      double separation,
                                                                      SpinVector dimensionless_spin1,
                                                                      SpinVector dimensionless_spin2,
                                                                      double center_of_mass = 0.);

    AlgebraicPNOrbitalParameters aligned_spin_algebraic_pn_parameters(const AlgebraicPNParameters& params,
                                                                      double separation,
                                                                      SpinVector dimensionless_spin1,
                                                                      SpinVector dimensionless_spin2,
                                                                      double center_of_mass = 0.);

    PrecessingPNOrbitalParameters precessing_algebraic_pn_parameters(double mass1,
                                                                     double mass2,
                                                                     const PrecessingPNState& state);

    PrecessingPNOrbitalParameters precessing_algebraic_pn_parameters(const AlgebraicPNParameters& params,
                                                                     const PrecessingPNState& state);

    // Convenience seed used by every binary eccentricity-reduction / warm-up stage:
    // builds the precessing PN state from each body's dimensionless spin magnitude
    // and tilt angle (degrees from the orbital axis), placing the in-plane spin
    // component along n_hat ({chi_perp, 0, chi_par} in the canonical frame), then
    // returns the leading-order orbital parameters. Reduces to the aligned-spin
    // seed when both tilt angles are 0 (or 180) degrees.
    PrecessingPNOrbitalParameters precessing_seed_from_tilt(double mass1,
                                                            double mass2,
                                                            double separation,
                                                            double dimensionless_spin1,
                                                            double spin1_axis_degrees,
                                                            double dimensionless_spin2,
                                                            double spin2_axis_degrees,
                                                            double center_of_mass = 0.);
} // namespace Kadath::PN

namespace bco_utils
{
    template <typename config_t>
    Kadath::PN::PrecessingPNOrbitalParameters binary_pn_orbital_seed(config_t& bconfig,
                                                                     double mass1,
                                                                     double mass2)
    {
        const double raw_chi1 = bconfig.has_value(CHI, BCO1) ? static_cast<double>(bconfig(CHI, BCO1)) : 0.;
        const double raw_chi2 = bconfig.has_value(CHI, BCO2) ? static_cast<double>(bconfig(CHI, BCO2)) : 0.;
        const double spin_axis1 = bconfig.has_value(DEG, BCO1) ? static_cast<double>(bconfig(DEG, BCO1)) : 0.;
        const double spin_axis2 = bconfig.has_value(DEG, BCO2) ? static_cast<double>(bconfig(DEG, BCO2)) : 0.;
        return Kadath::PN::precessing_seed_from_tilt(mass1,
                                                     mass2,
                                                     bconfig(DIST),
                                                     raw_chi1,
                                                     spin_axis1,
                                                     raw_chi2,
                                                     spin_axis2,
                                                     bconfig(COM));
    }
} // namespace bco_utils
