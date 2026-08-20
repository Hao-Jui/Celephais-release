#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include <sstream>
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/ns_bounds.hpp"
#include "Apps/Seed/seed_utils.hpp"

template <NODES s_type, typename config_t> void setup_co(config_t& bconfig, bool use_config_vars)
{
    // auto& fields = bconfig.return_fields();

    int type_coloc = CHEB_TYPE;
    Dim_array res(bconfig(DIM));
    res.set(0) = bconfig(BCO_RES);
    res.set(1) = bconfig(BCO_RES);

    Point center(bconfig(DIM));
    for (int i = 1; i <= bconfig(DIM); i++)
        center.set(i) = 0;

    if constexpr (s_type == NS) {
        const double h_cut = bconfig.template eos<double>(HCUT);
        const std::string eos_file = bconfig.template eos<std::string>(EOSFILE);
        const std::string eos_type = bconfig.template eos<std::string>(EOSTYPE);

        auto gen_NS = [&](auto tov) {
            auto bounds = bco_utils::make_NS_bounds(bconfig);
            Space_polar_adapted space(type_coloc, center, res, bounds);
            bco_utils::print_bounds("2D seed bounds", bounds);
            write_ns2d_init_setup_tofile_xcts(space, bconfig, *tov);
            cout << "generated a full single star space including compactification to infinity" << endl;
        };

        if (!use_config_vars) {
            if (eos_type == "Cold_PWPoly") {
                using eos_t = Kadath::Margherita::Cold_PWPoly;
                EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);
                auto tov = setup_ns_config_from_TOV<eos_t>(bconfig);
                gen_NS(std::move(tov));
            } else if (eos_type == "Cold_Table") {
                using eos_t = Kadath::Margherita::Cold_Table;

                const int interp_pts =
                    (bconfig.template eos<int>(INTERP_PTS) == 0) ? 2000 : bconfig.template eos<int>(INTERP_PTS);

                EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut, interp_pts,
                                                       bconfig.template eos<double>(MNUC_CGS));
                auto tov = setup_ns_config_from_TOV<eos_t>(bconfig);
                gen_NS(std::move(tov));
            } else {
                std::ostringstream oss;
                oss << eos_type << " is not recognized.\n";
                KADATH_THROW(oss.str());
            }
        }
    } else {
        std::cerr << "We are solving for NS now.\n";
    }
}

template <typename tov_t, typename config_t>
void write_ns2d_init_setup_tofile_xcts(Space_polar_adapted& space, config_t& bconfig, tov_t& tov)
{
    using eos_t = typename tov_t::eos_t;
    const int ndom = space.get_nbr_domains();
    const int ndim = 2;

    Scalar lapse(space);
    lapse = 1.;
    Scalar conf(lapse);
    Scalar logh(space);
    logh.annule_hard();
    Scalar shift(space);
    shift.annule_hard();
    Scalar Omg(space);
    Omg.annule_hard();

    auto lintp = setup_interpolator_from_TOV(tov);

    auto fill_fields = [&](const size_t dom) {
        // get number of collocation points (nradial × ntheta × nphi) in that domain.
        Index pos(space.get_domain(dom)->get_nbr_points());
        do {
            double rval = space.get_domain(dom)->get_radius()(pos);
            auto all_ltp = lintp.interpolate_all(rval);
            auto rho = (all_ltp[static_cast<int>(TovField::RHO)] <= 0) ? 1e-15
                                                                        : all_ltp[static_cast<int>(TovField::RHO)];

            auto h = EOS<eos_t, eos_var_t::DENSITY>::h_cold__rho(rho);
            if (pos(0) == 0 && pos(1) == 0) {
                bconfig.set(HC) = h;
            }
            logh.set_domain(dom).set(pos) = (h < 1) ? 0. : std::log(h);
            lapse.set_domain(dom).set(pos) = all_ltp[static_cast<int>(TovField::LAPSE)];
            conf.set_domain(dom).set(pos) = all_ltp[static_cast<int>(TovField::CONF)];
        } while (pos.inc());
    };

    for (int i = 0; i < ndim; ++i) {
        fill_fields(i);
    }
    for (int d = 2; d < ndom; ++d) {
        logh.set_domain(d).annule_hard();
    }

    shift.std_base();
    logh.std_base();
    conf.std_base();
    lapse.std_base();
    Omg.std_base();

    {
        bco_utils::save_to_file(space, bconfig, conf, lapse, shift, logh, Omg);
    }
}
