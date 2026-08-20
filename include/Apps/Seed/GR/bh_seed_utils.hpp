#pragma once
// Space-generic isolated-BH field initialisers for GR seed data.
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "For_Kadath/Array/exceptions.hpp"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

namespace Kadath::Seed {

enum class BhSeedKind {
    LegacyConformal = 0,
    Trumpet = 1
};

template <typename config_t>
BhSeedKind select_bh_seed_kind(config_t& bconfig)
{
    if (!bconfig.has_value(TRUMPET_BH_SEED))
        return BhSeedKind::LegacyConformal;

    return bconfig.template value_as<int>(TRUMPET_BH_SEED) != 0
        ? BhSeedKind::Trumpet
        : BhSeedKind::LegacyConformal;
}

template <typename config_t>
std::vector<double> isolated_bh_seed_bounds(config_t& bconfig, bool use_config_vars)
{
    const int ndom = 4 + static_cast<int>(bconfig(NSHELLS));
    std::vector<double> bounds(ndom - 1);

    if (!use_config_vars) {
        bconfig.set(RIN)  = bconfig(MCH) * bco_utils::invpsisq;
        bconfig.set(RMID) = 2 * bconfig(MCH) * bco_utils::invpsisq;
        bconfig.set(ROUT) = 4 * bconfig(RMID);
    }
    bco_utils::set_isolated_BH_bounds(bounds, bconfig);
    return bounds;
}

template <typename space_t, typename config_t>
void write_legacy_conformal_bh_fields(space_t& space, config_t& bconfig)
{
    Base_tensor basis(space, CARTESIAN_BASIS);

    Scalar lapse(space);
    lapse = 1.;
    lapse.set_domain(0).annule_hard();
    lapse.set_domain(1).annule_hard();

    Scalar conf(lapse);
    conf.set_domain(2) = bco_utils::psi;
    conf.std_base();
    lapse.std_base();

    Vector shift(space, CON, basis);
    for (int i = 1; i <= 3; i++)
        shift.set(i).annule_hard();
    shift.std_base();

    bco_utils::save_to_file(space, bconfig, conf, lapse, shift);
}

class NonRotatingTrumpetProfile {
  public:
    explicit NonRotatingTrumpetProfile(double mass)
        : mass_(mass)
    {
        if (!(mass_ > 0.)) {
            KADATH_THROW("Non-rotating trumpet BH seed requires MCH > 0.");
        }
    }

    double horizon_isotropic_radius() const
    {
        return isotropic_radius_from_areal_radius(2. * mass_);
    }

    double limiting_areal_radius() const
    {
        return 1.5 * mass_;
    }

    double areal_radius(double isotropic_radius) const
    {
        const double r = std::max(isotropic_radius, 0.);
        if (!std::isfinite(r))
            return std::numeric_limits<double>::infinity();
        if (r == 0.)
            return limiting_areal_radius();

        double lower = limiting_areal_radius();
        double upper = std::max(2. * mass_, r + mass_ + 0.5 * mass_);
        while (isotropic_radius_from_areal_radius(upper) < r)
            upper *= 2.;

        for (int iter = 0; iter < 80; ++iter) {
            const double mid = 0.5 * (lower + upper);
            if (isotropic_radius_from_areal_radius(mid) < r)
                lower = mid;
            else
                upper = mid;
        }
        return 0.5 * (lower + upper);
    }

    double conformal_factor(double isotropic_radius) const
    {
        if (!std::isfinite(isotropic_radius))
            return 1.;
        const double r = positive_radius(isotropic_radius);
        return std::sqrt(areal_radius(r) / r);
    }

    double lapse(double isotropic_radius) const
    {
        if (!std::isfinite(isotropic_radius))
            return 1.;
        const double R = areal_radius(positive_radius(isotropic_radius));
        const double numerator =
            (2. * R - 3. * mass_) * std::sqrt(4. * R * R + 4. * mass_ * R + 3. * mass_ * mass_);
        return std::max(numerator / (4. * R * R), 0.);
    }

    double lapse_times_conformal_factor(double isotropic_radius) const
    {
        if (!std::isfinite(isotropic_radius))
            return 1.;
        return lapse(isotropic_radius) * conformal_factor(isotropic_radius);
    }

    double radial_shift(double isotropic_radius) const
    {
        const double r = isotropic_radius;
        if (!std::isfinite(r))
            return 0.;
        if (r <= 0.)
            return 0.;
        const double R = areal_radius(r);
        return 3. * std::sqrt(3.) * mass_ * mass_ * r / (4. * R * R * R);
    }

  private:
    double isotropic_radius_from_areal_radius(double areal_radius) const
    {
        const double R = std::max(areal_radius, limiting_areal_radius());
        if (R == limiting_areal_radius())
            return 0.;

        const double sqrt2 = std::sqrt(2.);
        const double first =
            (2. * R + mass_ + std::sqrt(4. * R * R + 4. * mass_ * R + 3. * mass_ * mass_)) / 4.;
        const double numerator = (4. + 3. * sqrt2) * (2. * R - 3. * mass_);
        const double denominator =
            8. * R + 6. * mass_ + 3. * std::sqrt(8. * R * R + 8. * mass_ * R + 6. * mass_ * mass_);
        return first * std::pow(numerator / denominator, 1. / sqrt2);
    }

    static double positive_radius(double isotropic_radius)
    {
        constexpr double min_radius = 1e-14;
        return std::max(isotropic_radius, min_radius);
    }

    double mass_;
};

template <typename config_t>
std::vector<double> isolated_trumpet_bh_seed_bounds(config_t& bconfig, bool use_config_vars)
{
    const int ndom = 4 + static_cast<int>(bconfig(NSHELLS));
    std::vector<double> bounds(ndom - 1);

    if (!use_config_vars) {
        NonRotatingTrumpetProfile profile(static_cast<double>(bconfig(MCH)));
        bconfig.set(RMID) = profile.horizon_isotropic_radius();
        bconfig.set(RIN) = 0.5 * bconfig(RMID);
        bconfig.set(ROUT) = 4. * bconfig(MCH);
    }
    bco_utils::set_isolated_BH_bounds(bounds, bconfig);
    return bounds;
}

template <typename space_t, typename config_t>
void write_trumpet_bh_fields(space_t& space, config_t& bconfig, int first_physical_domain = 2)
{
    if (std::abs(static_cast<double>(bconfig(CHI))) > 1e-14) {
        KADATH_THROW("The trumpet BH seed is currently implemented only for non-rotating BHs (CHI = 0).");
    }

    Base_tensor basis(space, CARTESIAN_BASIS);
    CoordFields<space_t> coords(space);
    Scalar radius(coords.radius());
    Vector radial_unit(coords.template e_rad<CON>());
    NonRotatingTrumpetProfile profile(static_cast<double>(bconfig(MCH)));

    Scalar lapse(space);
    lapse.annule_hard();
    Scalar conf(space);
    conf.annule_hard();
    Vector shift(space, CON, basis);
    for (int i = 1; i <= 3; i++)
        shift.set(i).annule_hard();

    const int ndom = space.get_nbr_domains();
    for (int dom = first_physical_domain; dom < ndom; ++dom) {
        Index pos(space.get_domain(dom)->get_nbr_points());
        do {
            const double r = radius(dom)(pos);
            conf.set_domain(dom).set(pos) = profile.conformal_factor(r);
            lapse.set_domain(dom).set(pos) = profile.lapse(r);
            const double beta_r = profile.radial_shift(r);
            for (int i = 1; i <= 3; ++i)
                shift.set(i).set_domain(dom).set(pos) = beta_r * radial_unit(i)(dom)(pos);
        } while (pos.inc());
    }

    conf.std_base();
    lapse.std_base();
    shift.std_base();

    bco_utils::save_to_file(space, bconfig, conf, lapse, shift);
}

template <typename space_t, typename config_t>
void write_bh_fields(space_t& space, config_t& bconfig, BhSeedKind kind)
{
    switch (kind) {
    case BhSeedKind::LegacyConformal:
        write_legacy_conformal_bh_fields(space, bconfig);
        return;
    case BhSeedKind::Trumpet:
        write_trumpet_bh_fields(space, bconfig);
        return;
    }

    KADATH_THROW("Unsupported BH seed kind.");
}

} // namespace Kadath::Seed
