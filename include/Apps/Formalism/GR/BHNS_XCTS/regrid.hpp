#pragma once

// GR black-hole–neutron-star regrid: thin formalism wrapper over
// Shared/Regrid/bhns_regrid.hpp. The shared core owns all geometry/bounds/
// adapted-mapping/excision-setup; this file supplies only the GR field set
// (conf, lapse, shift, logh, phi). Templated on space_t so it also serves the
// nosym space (see BHNS_XCTS_nosym/regrid.hpp).

#include "Apps/Formalism/Shared/Regrid/bhns_regrid.hpp"

namespace Kadath {

using config_t = kadath_config<BIN_INFO>;

// GR field transfer: conformal factor, lapse, shift, log-enthalpy, velocity
// potential. old_conf is read by the shared core (the BH inner-radius estimate
// needs it); this callback reads the remaining old fields in the historical
// BeFileSource order (lapse, shift, logh, phi), then updates the BH/NS adapted
// fields, imports into the new space, zeroes matter/vel.pot. outside the star
// and inside the excision region, and saves.
template <typename space_t = Space_bhns>
int bhns_xcts_regrid_impl(config_t& bconfig, std::string output_fname, bool use_config_vars = false,
                          const std::vector<Dim_array>& res_per_domain = {})
{
    auto transfer_fields = [](bhns_regrid_transfer_context<space_t>& ctx, config_t& cfg) {
        Scalar& old_conf = ctx.old_conf;
        Scalar old_lapse(ctx.old_space, ctx.fin);
        Vector old_shift(ctx.old_space, ctx.fin);
        Scalar old_logh(ctx.old_space, ctx.fin);
        Scalar old_phi(ctx.old_space, ctx.fin);

        // update BH fields to help with import
        bco_utils::update_adapted_field(old_conf, ctx.old_space.ADAPTEDBH + 1, ctx.old_space.ADAPTEDBH,
                                        ctx.old_bh_outer, OUTER_BC);
        bco_utils::update_adapted_field(old_lapse, ctx.old_space.ADAPTEDBH + 1, ctx.old_space.ADAPTEDBH,
                                        ctx.old_bh_outer, OUTER_BC);
        for (int i = 1; i < 4; ++i)
            bco_utils::update_adapted_field(old_shift.set(i), ctx.old_space.ADAPTEDBH + 1, ctx.old_space.ADAPTEDBH,
                                            ctx.old_bh_outer, OUTER_BC);

        bco_utils::update_adapted_field(old_phi, ctx.old_space.ADAPTEDNS, ctx.old_space.ADAPTEDNS + 1,
                                        ctx.old_inner_adaptedNS, INNER_BC);

        // setup new fields
        Scalar conf(ctx.space);
        conf = 1.;

        Scalar lapse(ctx.space);
        lapse = 1.;

        Vector shift(ctx.space, CON, ctx.basis);
        for (int i = 1; i <= 3; i++)
            shift.set(i).annule_hard();

        Scalar logh(ctx.space);
        logh.annule_hard();

        Scalar phi(ctx.space);
        phi.annule_hard();
        // end setup

        // import old fields into new space
        conf.import(old_conf);
        lapse.import(old_lapse);
        logh.import(old_logh);
        phi.import(old_phi);

        shift.set(1).import(old_shift.set(1));
        shift.set(2).import(old_shift.set(2));
        shift.set(3).import(old_shift.set(3));
        // end import

        // make sure there is no matter or vel.pot. outside of the star
        logh.set_domain(ctx.space.ADAPTEDNS + 1).annule_hard();
        phi.set_domain(ctx.space.ADAPTEDNS + 1).annule_hard();
        for (int d = ctx.space.BH; d < ctx.space.get_nbr_domains(); ++d) {
            logh.set_domain(d).annule_hard();
            phi.set_domain(d).annule_hard();

            // make sure all fields are 0 inside the excision region
            if ((d >= ctx.space.BH) && (d < ctx.space.BH + 2)) {
                conf.set_domain(d).annule_hard();
                lapse.set_domain(d).annule_hard();
                for (int i = 1; i <= 3; i++)
                    shift.set(i).set_domain(d).annule_hard();
            }
        }

        lapse.std_base();
        conf.std_base();
        logh.std_base();
        shift.std_base();
        phi.std_base();

        bco_utils::save_to_file(ctx.space, cfg, conf, lapse, shift, logh, phi);
    };

    return bhns_xcts_regrid_impl_fields<config_t, space_t>(
        bconfig, output_fname, use_config_vars, transfer_fields, res_per_domain);
}

inline int bhns_xcts_regrid(config_t& bconfig, std::string output_fname, bool use_config_vars = false)
{
    return bhns_xcts_regrid_impl<Space_bhns>(bconfig, output_fname, use_config_vars);
}

// Per-domain AMR p-refinement overload: rebuild the new space with one Dim_array
// per domain (the geometry is recomputed from the converged old space, so it is
// preserved; only the resolution changes). Disambiguated from the bool overload
// by the vector argument.
inline int bhns_xcts_regrid(config_t& bconfig, std::string output_fname,
                            const std::vector<Dim_array>& res_per_domain)
{
    return bhns_xcts_regrid_impl<Space_bhns>(bconfig, output_fname, /*use_config_vars=*/false, res_per_domain);
}

} // namespace Kadath
