#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"
#include <sstream>
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "Apps/Bco_utils/bco_io.hpp"

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

    const int shells = (int)bconfig(NSHELLS);
    int ndom = 4 + shells;
    std::vector<double> bounds(ndom - 1);

    if constexpr (s_type == NS) {
        const double h_cut = bconfig.template eos<double>(HCUT);
        const std::string eos_file = bconfig.template eos<std::string>(EOSFILE);
        const std::string eos_type = bconfig.template eos<std::string>(EOSTYPE);

        auto gen_NS = [&](auto tov) {
            Array<double> bounds(ndom - 1);
            bounds.set(0) = bconfig(RIN);
            bounds.set(1) = bconfig(RMID);
            bounds.set(2) = bconfig(ROUT);
            for (int shell = 1, b = 3; shell <= shells; ++shell, ++b)
                bounds.set(b) = bounds(b - 1) * 2;
            Space_polar_adapted space(type_coloc, center, res, bounds);
            cout << "bounds: " << bounds << endl;
            write_ns2d_init_setup_tofile_msqi(space, bconfig, *tov);
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

                EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut, interp_pts);
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
void write_ns2d_init_setup_tofile_msqi(Space_polar_adapted& space, config_t& bconfig, tov_t& tov)
{
    using eos_t = typename tov_t::eos_t;
    enum class ltpQ { LAPSE = 0, RHO, CONF };
    const int ndom = space.get_nbr_domains();
    const int ndim = 2;

    Scalar lapse(space);
    lapse = 1.;
    Scalar conf(lapse);
    Scalar logh(space);
    logh.annule_hard();
    Scalar shift(space);
    shift.annule_hard();
    Scalar metQ(space);
    metQ.annule_hard();
    Scalar Omg(space);
    Omg.annule_hard();

    auto lintp = setup_interpolator_from_TOV(tov);

    auto fill_fields = [&](const size_t dom) {
        // get number of collocation points (nradial × ntheta × nphi) in that domain.
        Index pos(space.get_domain(dom)->get_nbr_points());
        do {
            double rval = space.get_domain(dom)->get_radius()(pos);
            auto all_ltp = lintp.interpolate_all(rval);
            auto rho = (all_ltp[static_cast<int>(ltpQ::RHO)] <= 0) ? 1e-15 : all_ltp[static_cast<int>(ltpQ::RHO)];

            auto h = EOS<eos_t, eos_var_t::DENSITY>::h_cold__rho(rho);
            if (pos(0) == 0 && pos(1) == 0) {
                bconfig.set(HC) = h;
            }
            logh.set_domain(dom).set(pos) = (h < 1) ? 0. : std::log(h);
            lapse.set_domain(dom).set(pos) = all_ltp[static_cast<int>(ltpQ::LAPSE)];
            conf.set_domain(dom).set(pos) = all_ltp[static_cast<int>(ltpQ::CONF)];
        } while (pos.inc());
    };

    for (int i = 0; i < ndim; ++i) {
        fill_fields(i);
    }
    for (int d = 2; d < ndom; ++d) {
        logh.set_domain(d).annule_hard();
    }

    shift.std_base();
    metQ.std_base();
    logh.std_base();
    conf.std_base();
    lapse.std_base();
    Omg.std_base();

    bco_utils::save_to_file(space, bconfig, conf, lapse, shift, metQ, logh, Omg);
}

template <typename eos_t, typename config_t> auto setup_ns_config_from_TOV(config_t& bconfig)
{
    using namespace Kadath::Margherita;
    auto tov = std::make_unique<MargheritaTOV<eos_t>>();
    bool use_Mmax = solve_tov_for_adm_mass(*tov, bconfig(MADM));

    if (use_Mmax)
        bconfig.set(MADM) = tov->mass;

    bconfig.set(NC) = tov->rhoc;
    bconfig.set(HC) = EOS<eos_t, eos_var_t::PRESSURE>::h_cold__rho(bconfig(NC));
    bconfig.set(MB) = tov->baryon_mass;

    // update surface radius estimate
    bconfig.set(RMID) = tov->radius;
    bconfig.set(RIN) = 0.5 * bconfig(RMID);
    bconfig.set(ROUT) = 2.0 * bconfig(RMID);

    return std::move(tov);
}

template <typename tov_t> auto setup_interpolator_from_TOV(tov_t& tov)
{
    const int max_iter = static_cast<int>(tov.state.size());

    std::unique_ptr<double[]> radius_lin_ptr{new double[max_iter]};
    std::unique_ptr<double[]> conf_lin_ptr{new double[max_iter]};
    std::unique_ptr<double[]> lapse_lin_ptr{new double[max_iter]};
    std::unique_ptr<double[]> rho_lin_ptr{new double[max_iter]};

    for (auto j = 0; j < max_iter; ++j) {
        auto phi = tov.state[j][tov.PHI];
        lapse_lin_ptr[j] = std::exp(phi);
        conf_lin_ptr[j] = tov.state[j][tov.CONF];
        radius_lin_ptr[j] = tov.state[j][tov.RISO];
        rho_lin_ptr[j] = tov.state[j][tov.RHOB];
    }

    // Linear interpolation of conf(r_isotropic), lapse(r_isotropic), and rho(r_isotropic).
    // The order here dictates the enum ltpQ enum.
    linear_interp_t<double, 3> ltp(max_iter, std::move(radius_lin_ptr), std::move(lapse_lin_ptr),
                                   std::move(rho_lin_ptr), std::move(conf_lin_ptr));
    return ltp;
}
