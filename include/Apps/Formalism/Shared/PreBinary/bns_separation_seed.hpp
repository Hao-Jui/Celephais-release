#pragma once

// Separation-continuation seed projection for binary-NS regrid.
//
// The default regrid field transfer (Scalar::import) assumes
// the binary separation DIST is UNCHANGED between the old and new space: it
// samples the old field at each NEW collocation point's *absolute* Cartesian
// position. When DIST changes, the new star domains sit at new centres
// xc_i^new = +/- DIST_new/2 while the old field still has its matter at
// xc_i^old = +/- DIST_old/2, so a straight import reads vacuum where the star
// moved to and produces a seed Newton cannot recover from.
//
// This helper rebuilds the seed as a per-star rigid-translation re-superposition
// of the OLD BINARY solution -- the same partition-of-unity blend the cold seed
// uses (Apps/Formalism/Shared/PreBinary/bns_boosted_setup.hpp), but fed the old binary
// solution sampled through a translation that makes each star follow its centre:
//
//   delta_i      = xc_i^new - xc_i^old                 (each star's drift)
//   absol_i(P)   = (x - xc_i^new + xc_i^old, y, z)     (= P - delta_i)
//   decay_i(P)   = exp(-r4_i * invw_i)                 r4_i from offset to NEW centre i
//
// For fields with a meaningful exterior (conf, lapse, shift -- "background"
// mode) the untranslated old field is the partition-of-unity background, so the
// orbital/asymptotic tail is preserved and the blend reduces EXACTLY to a plain
// import when delta_i = 0 (same-DIST regrid -> no behaviour change):
//
//   f(P) = old.val_point(P)
//        + decay_1 * (old.val_point(absol_1) - old.val_point(P))
//        + decay_2 * (old.val_point(absol_2) - old.val_point(P))
//        = (1 - d1 - d2) old(P) + d1 old(absol_1) + d2 old(absol_2)
//
// For matter fields (logh, phi -- zero exterior, no "background" term) the blend
// is the pure decay-weighted translated sample; the exterior is then cleaned by
// the caller's ctx.zero_matter_outside_stars().
//
// The star-surface adapted mappings are already re-centred for the new DIST by
// the shared regrid core (bco_utils::interp_adapted_mapping in
// bns_xcts_regrid_impl_fields), so iterating the new space here yields collocation
// points at the correct new physical positions.
//
// Callers gate on bns_separation_changed() and fall back to a plain
// Scalar::import only when DIST is fixed.

#include "Apps/Formalism/Shared/Regrid/bns_regrid.hpp"
#include "Apps/Formalism/Shared/scalar_point_batch.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/Utilities/PN/orbital_pn_params.hpp"
#include "For_Kadath/Array/point.hpp"

#include <cmath>

namespace Kadath {
namespace bns_separation_seed {

// Per-star geometry for the translation blend.
struct two_center_drift {
    double xc_old[2];  // old star centres (x), [NS1, NS2]
    double xc_new[2];  // new star centres (x)
    double invw4[2];   // exponential decay 1/w^4 per star (bco_utils::set_decay)
};

// Old vs new separation, derived from the star centres of each space.
template <typename space_t>
double old_separation(const space_t& old_space)
{
    return bco_utils::get_center(old_space, old_space.NS2)
         - bco_utils::get_center(old_space, old_space.NS1);
}

template <typename space_t>
double new_separation(const space_t& space)
{
    return bco_utils::get_center(space, space.NS2)
         - bco_utils::get_center(space, space.NS1);
}

// True when the regrid moves the stars (separation continuation). tol is in the
// same length units as DIST; 1e-10 distinguishes a genuine drift from
// round-trip noise on a same-DIST p/h regrid.
template <typename space_t>
bool separation_changed(const space_t& old_space, const space_t& space, double tol = 1e-10)
{
    return std::abs(new_separation(space) - old_separation(old_space)) > tol;
}

template <typename config_t, typename space_t>
two_center_drift make_drift(const space_t& old_space, const space_t& space, config_t& bconfig)
{
    two_center_drift d;
    d.xc_old[0] = bco_utils::get_center(old_space, old_space.NS1);
    d.xc_old[1] = bco_utils::get_center(old_space, old_space.NS2);
    d.xc_new[0] = bco_utils::get_center(space, space.NS1);
    d.xc_new[1] = bco_utils::get_center(space, space.NS2);
    d.invw4[0] = bco_utils::set_decay(bconfig, BCO1);
    d.invw4[1] = bco_utils::set_decay(bconfig, BCO2);
    return d;
}

// Project one scalar field old_f (on old_space) onto new_f (on new_f's space)
// via the two-centre translation blend.
//   use_background = true  : conf/lapse/shift -- keep the untranslated old field
//                            as the partition-of-unity background; the
//                            compactified domain is pinned to `asymptotic`.
//   use_background = false : logh/phi (matter) -- pure decay-weighted translated
//                            sample, compactified domain set to `asymptotic`
//                            (0 for matter); exterior cleaned by the caller.
inline void blend_scalar(Scalar& new_f, const Scalar& old_f,
                         const two_center_drift& d,
                         bool use_background, double asymptotic)
{
    const Space& space = new_f.get_space();
    const int ndom = space.get_nbr_domains();

    for (int dom = 0; dom < ndom; ++dom) {
        Index new_pos(space.get_domain(dom)->get_nbr_points());
        const bool compactified = (dom == ndom - 1);
        do {
            if (compactified) {
                new_f.set_domain(dom).set(new_pos) = asymptotic;
                continue;
            }

            const double x = space.get_domain(dom)->get_cart(1)(new_pos);
            const double y = space.get_domain(dom)->get_cart(2)(new_pos);
            const double z = space.get_domain(dom)->get_cart(3)(new_pos);
            const double r2_perp = y * y + z * z;

            Point absol1(3), absol2(3);
            absol1.set(1) = x - d.xc_new[0] + d.xc_old[0];
            absol1.set(2) = y;
            absol1.set(3) = z;
            absol2.set(1) = x - d.xc_new[1] + d.xc_old[1];
            absol2.set(2) = y;
            absol2.set(3) = z;

            const double off1 = x - d.xc_new[0];
            const double off2 = x - d.xc_new[1];
            const double r4_1 = (off1 * off1 + r2_perp) * (off1 * off1 + r2_perp);
            const double r4_2 = (off2 * off2 + r2_perp) * (off2 * off2 + r2_perp);
            const double decay1 = std::exp(-r4_1 * d.invw4[0]);
            const double decay2 = std::exp(-r4_2 * d.invw4[1]);

            const double s1 = old_f.val_point(absol1);
            const double s2 = old_f.val_point(absol2);

            if (use_background) {
                Point pabs(3);
                pabs.set(1) = x;
                pabs.set(2) = y;
                pabs.set(3) = z;
                const double bg = old_f.val_point(pabs);
                new_f.set_domain(dom).set(new_pos) =
                    bg + decay1 * (s1 - bg) + decay2 * (s2 - bg);
            } else {
                new_f.set_domain(dom).set(new_pos) = decay1 * s1 + decay2 * s2;
            }
        } while (new_pos.inc());
    }
}

struct scalar_blend_field
{
    Scalar* target;
    const Scalar* source;
    bool use_background;
    double asymptotic;
};

// Field-batched form of blend_scalar. All lanes share the target traversal,
// translated/background points, decay weights, source-domain selection, and
// physical-to-numerical maps. The per-lane expression order remains identical
// to blend_scalar so this can be checked against it as an exact oracle.
inline void blend_scalar_batch(std::span<const scalar_blend_field> fields,
                               const two_center_drift& d)
{
    if (fields.empty())
        return;
    if (fields.front().target == nullptr || fields.front().source == nullptr)
        KADATH_THROW("blend_scalar_batch requires non-null fields");

    const Space& target_space = fields.front().target->get_space();
    const int target_ndim = fields.front().target->get_ndim();
    if (target_ndim != 3)
        KADATH_THROW("blend_scalar_batch requires three-dimensional fields");

    std::vector<const Scalar*> sources;
    std::vector<std::size_t> background_lanes;
    sources.reserve(fields.size());
    background_lanes.reserve(fields.size());
    for (std::size_t lane = 0; lane < fields.size(); ++lane) {
        const scalar_blend_field& field = fields[lane];
        if (field.target == nullptr || field.source == nullptr)
            KADATH_THROW("blend_scalar_batch requires non-null fields");
        if (&field.target->get_space() != &target_space ||
            field.target->get_ndim() != target_ndim)
            KADATH_THROW("blend_scalar_batch targets must share one space");
        sources.push_back(field.source);
        if (field.use_background)
            background_lanes.push_back(lane);
    }

    bns_field_transfer::scalar_source_batch source_batch(
        std::span<const Scalar* const>(sources.data(), sources.size()));
    if (source_batch.source_ndim() != target_ndim)
        KADATH_THROW("blend_scalar_batch source/target dimensions must match");

    std::vector<double> values1(fields.size());
    std::vector<double> values2(fields.size());
    std::vector<double> backgrounds(fields.size());
    const int ndom = target_space.get_nbr_domains();
    for (int dom = 0; dom < ndom; ++dom) {
        const Domain& target_domain = *target_space.get_domain(dom);
        Index new_pos(target_domain.get_nbr_points());
        const bool compactified = dom == ndom - 1;
        do {
            if (compactified) {
                for (const scalar_blend_field& field : fields)
                    field.target->set_domain(dom).set(new_pos) = field.asymptotic;
                continue;
            }

            const double x = target_domain.get_cart(1)(new_pos);
            const double y = target_domain.get_cart(2)(new_pos);
            const double z = target_domain.get_cart(3)(new_pos);
            const double r2_perp = y * y + z * z;

            Point absol1(3), absol2(3);
            absol1.set(1) = x - d.xc_new[0] + d.xc_old[0];
            absol1.set(2) = y;
            absol1.set(3) = z;
            absol2.set(1) = x - d.xc_new[1] + d.xc_old[1];
            absol2.set(2) = y;
            absol2.set(3) = z;

            const double off1 = x - d.xc_new[0];
            const double off2 = x - d.xc_new[1];
            const double r4_1 = (off1 * off1 + r2_perp) * (off1 * off1 + r2_perp);
            const double r4_2 = (off2 * off2 + r2_perp) * (off2 * off2 + r2_perp);
            const double decay1 = std::exp(-r4_1 * d.invw4[0]);
            const double decay2 = std::exp(-r4_2 * d.invw4[1]);

            source_batch.values(source_batch.locate(absol1), values1);
            source_batch.values(source_batch.locate(absol2), values2);

            if (!background_lanes.empty()) {
                Point pabs(3);
                pabs.set(1) = x;
                pabs.set(2) = y;
                pabs.set(3) = z;
                const auto background_point = source_batch.locate(
                    pabs, std::span<const std::size_t>(background_lanes.data(),
                                                      background_lanes.size()));
                for (const std::size_t lane : background_lanes)
                    backgrounds[lane] = source_batch.value(background_point, lane);
            }

            for (std::size_t lane = 0; lane < fields.size(); ++lane) {
                const scalar_blend_field& field = fields[lane];
                if (field.use_background) {
                    const double background = backgrounds[lane];
                    field.target->set_domain(dom).set(new_pos) =
                        background + decay1 * (values1[lane] - background)
                                   + decay2 * (values2[lane] - background);
                } else {
                    field.target->set_domain(dom).set(new_pos) =
                        decay1 * values1[lane] + decay2 * values2[lane];
                }
            }
        } while (new_pos.inc());
    }
}

inline void blend_vector(Vector& new_v, const Vector& old_v,
                         const two_center_drift& d)
{
    for (int i = 1; i <= 3; ++i)
        blend_scalar(new_v.set(i), old_v(i), d, /*use_background=*/true, /*asymptotic=*/0.0);
}

template <typename config_t, typename space_t>
void blend_gr_fields(bns_regrid_transfer_context<space_t>& ctx, config_t& bconfig,
                     const Scalar& old_conf, const Scalar& old_lapse,
                     const Vector& old_shift, const Scalar& old_logh,
                     const Scalar& old_phi,
                     Scalar& conf, Scalar& lapse, Vector& shift,
                     Scalar& logh, Scalar& phi,
                     std::span<const scalar_blend_field> extra_fields = {})
{
    const two_center_drift d = make_drift(ctx.old_space, ctx.space, bconfig);

    std::vector<scalar_blend_field> fields{
        {&conf, &old_conf, true, 1.0},
        {&lapse, &old_lapse, true, 1.0},
        {&shift.set(1), &old_shift(1), true, 0.0},
        {&shift.set(2), &old_shift(2), true, 0.0},
        {&shift.set(3), &old_shift(3), true, 0.0},
        {&logh, &old_logh, false, 0.0},
        {&phi, &old_phi, false, 0.0},
    };
    fields.insert(fields.end(), extra_fields.begin(), extra_fields.end());
    blend_scalar_batch(fields, d);
}

} // namespace bns_separation_seed
} // namespace Kadath
