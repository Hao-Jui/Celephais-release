#pragma once


#include "Apps/Formalism/Shared/Regrid/bns_regrid.hpp"
#include "Apps/Formalism/Shared/PreBinary/bns_separation_seed.hpp"

namespace Kadath {

// GR field transfer: conformal factor, lapse, shift, log-enthalpy, velocity
// potential. Default behavioural knobs (use_config_vars=false,
// ns1_include_outer_shells=true) reproduce the historical GR regrid exactly.
template <typename config_t, typename space_t = Space_bin_ns>
int bns_xcts_regrid_impl(config_t& bconfig, std::string output_fname,
                         const std::vector<Dim_array>& res_per_domain = {})
{
    auto transfer_fields = [](bns_regrid_transfer_context<space_t>& ctx, config_t& cfg) {
        Scalar old_conf(ctx.old_space, ctx.fin);
        Scalar old_lapse(ctx.old_space, ctx.fin);
        Vector old_shift(ctx.old_space, ctx.fin);
        Scalar old_logh(ctx.old_space, ctx.fin);
        Scalar old_phi(ctx.old_space, ctx.fin);

        for (int i = 0; i < 2; ++i) {
            int const dom = ctx.old_adapted_doms[i];
            bco_utils::update_adapted_field(old_phi, dom, dom + 1, ctx.old_inner_adapted[i], INNER_BC);
        }

        Scalar conf(ctx.space);
        conf = 1.;
        conf.std_base();

        Scalar lapse(ctx.space);
        lapse = 1.;
        lapse.std_base();

        Vector shift(ctx.space, CON, ctx.basis);
        for (int i = 1; i <= 3; i++)
            shift.set(i).annule_hard();
        shift.std_base();

        Scalar logh(ctx.space);
        logh.annule_hard();
        logh.std_base();

        Scalar phi(ctx.space);
        phi.annule_hard();
        phi.std_base();

        if (bns_separation_seed::separation_changed(ctx.old_space, ctx.space)) {
            // Binary separation changed (continuation in DIST): a plain import
            // would read vacuum where each star moved to. Re-superpose the old
            // binary with a per-star rigid translation so the matter follows its
            // domain. Reduces to import when the separation is unchanged, so the
            // branch above stays the fast path for p/h regrids.
            bns_separation_seed::blend_gr_fields(ctx, cfg,
                old_conf, old_lapse, old_shift, old_logh, old_phi,
                conf, lapse, shift, logh, phi);
        } else {
            const std::array import_fields{
                bns_field_transfer::import_field(conf, old_conf),
                bns_field_transfer::import_field(lapse, old_lapse),
                bns_field_transfer::import_field(logh, old_logh),
                bns_field_transfer::import_field(phi, old_phi),
                bns_field_transfer::import_field(shift.set(1), old_shift.set(1)),
                bns_field_transfer::import_field(shift.set(2), old_shift.set(2)),
                bns_field_transfer::import_field(shift.set(3), old_shift.set(3)),
            };
            bns_field_transfer::import_scalar_batch(import_fields);
        }

        ctx.zero_matter_outside_stars(logh, phi);

        lapse.std_base();
        conf.std_base();
        logh.std_base();
        shift.std_base();
        phi.std_base();

        bco_utils::save_to_file(ctx.space, cfg, conf, lapse, shift, logh, phi);
    };

    return bns_xcts_regrid_impl_fields<config_t, space_t>(
        bconfig, output_fname,
        /*use_config_vars=*/false,
        transfer_fields, res_per_domain);
}

} // namespace Kadath
