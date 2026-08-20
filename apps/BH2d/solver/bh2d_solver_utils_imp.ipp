
#include "coord_fields_2d.hpp"
#include "Apps/Bco_utils/bco_io.hpp"
#include "Apps/Bco_utils/bh_bounds.hpp"
#include "For_Kadath/Space/adapted_bh_polar.hpp"

template <NODES s_type, typename config_t> void setup_co(config_t& bconfig, bool use_config_vars)
{
    int type_coloc = CHEB_TYPE;
    Dim_array res(bconfig(DIM));
    res.set(0) = bconfig(BCO_RES);
    res.set(1) = 11; // bconfig(BCO_RES)-2;

    Point center(bconfig(DIM));
    for (int i = 1; i <= bconfig(DIM); i++) {
        center.set(i) = 0;
    }

    const int shells = static_cast<int>(bconfig(NSHELLS));
    int ndom = 4 + shells;

    if constexpr (s_type == BH) {
        Array<double> bounds(ndom - 1);
        if (!use_config_vars) {
            bconfig.set(RIN) = bconfig(MCH) * bco_utils::invpsisq;
            bconfig.set(RMID) = 2 * bconfig(MCH) * bco_utils::invpsisq;
            bconfig.set(ROUT) = 2 * bconfig(RMID);
        }
        bounds.set(0) = bconfig(RIN);
        bounds.set(1) = bconfig(RMID);
        bounds.set(2) = bconfig(ROUT);
        for (int shell = 1, b = 3; shell <= shells; ++shell, ++b) {
            bounds.set(b) = bounds(b - 1) * 1.709;
        }
        Space_adapted_bh_polar space(type_coloc, center, res, bounds);
        write_bh2d_init_setup_tofile_msqi(space, bconfig);
    } else {
        std::cerr << "We are solving for BH now.\n";
    }
}

template <typename space_t>
void debug_operators(System_of_eqs& syst_dbg, const space_t& space, const Scalar& alpha, double mass_scal)
{
    const int ndoms = space.get_nbr_domains();
    Scalar oneField(space);
    oneField = 1.;
    oneField.std_base();

    CoordFields2D<space_t> cfields(space);
    Scalar radius = cfields.radius();
    Scalar invr = cfields.inv_r();
    Scalar overexpmr = cfields.overexpmr_field(mass_scal);
    Vector gradr_i = cfields.gradr_i();

    syst_dbg.add_cst("alpha", alpha);
    syst_dbg.add_cst("invr", invr);
    syst_dbg.add_cst("overexpmr", overexpmr);
    syst_dbg.add_cst("gradr", gradr_i);
    syst_dbg.add_def("dinvr = scal(grad(invr), grad(invr))");

    for (int d = 1; d < ndoms; ++d) {
        syst_dbg.add_def(d, "gradinvr_i = -grad(invr)");
        syst_dbg.add_def(d, "dalphaerad = scal(gradr_i, gradr_i)");
    }

    Scalar dinvr = syst_dbg.give_val_def("dinvr");
    dinvr.std_base();
    Scalar dalphaerad = syst_dbg.give_val_def("dalphaerad");
    dalphaerad.std_base();

    for (int d = 1; d < ndoms; ++d) {
        const Dim_array nbr_pts = space.get_domain(d)->get_nbr_points();
        if (nbr_pts.get_ndim() < 2)
            continue;

        std::cout << "domain " << d << " along equator:\n";
        Index pos(nbr_pts);
        pos.set(1) = 0;
        for (int i = 0; i < nbr_pts(0); ++i) {
            pos.set(0) = i;
            std::cout << "  i=" << i << " r=" << radius(d)(pos) << " invr=" << invr(d)(pos)
                      << " dinvr=" << dinvr(d)(pos) << " overexpmr=" << overexpmr(d)(pos)
                      << " dalphaerad=" << dalphaerad(d)(pos) << "\n";
        }
    }
}

template <typename config_t> void write_bh2d_init_setup_tofile_msqi(Space_adapted_bh_polar& space, config_t& bconfig)
{

    int ndoms(space.get_nbr_domains());
    Scalar oneField(space);
    oneField = 1.;
    oneField.std_base();

    Scalar radius(oneField);
    Scalar rsint(oneField);
    for (int d = 0; d < ndoms; d++) {
        radius.set_domain(d) =
            sqrt(pow(space.get_domain(d)->get_cart(1), 2) + pow(space.get_domain(d)->get_cart(2), 2));
        rsint.set_domain(d) = space.get_domain(d)->get_cart(1);
    }
    radius.std_base();
    radius.coef();
    radius.coef_i();

    const double cm = bconfig(MCH);
    Scalar alpha_guess((2 * radius - cm) / (2 * radius + cm));
    Scalar conf_guess(1 + cm / (2 * radius));
    Scalar sphi_guess(oneField * 1e-3 * rsint * exp(-radius / 13.));

    const Dim_array nbr_pts = space.get_domain(ndoms - 1)->get_nbr_points();
    const int nr = nbr_pts(0);
    Index pos(nbr_pts);
    for (int j = 0; j < nbr_pts(1); ++j) {
        pos.set(0) = nr - 1;
        pos.set(1) = j;
        alpha_guess.set_domain(ndoms - 1).set(pos) = 1.0;
        sphi_guess.set_domain(ndoms - 1).set(pos) = 0.0;

        pos.set(0) = 0;
        conf_guess.set_domain(0).set(pos) = 1.;
    }

    Scalar lapse(space);
    Scalar metA(space);
    Scalar metB(space);
    Scalar beta(space);
    Scalar varscal(space);

    lapse = alpha_guess;
    lapse.std_base();

    metA = 2.;
    metA.std_base();

    metB = metA;
    metB.std_base();

    beta.annule_hard();
    beta.std_base();

    varscal = sphi_guess;
    varscal.std_base();

    bco_utils::save_to_file(space, bconfig, lapse, metA, metB, beta, varscal);

    // System_of_eqs syst_dbg(space); debug_operators(syst_dbg, space, alpha_guess, bconfig.template gravity<double>(GRAV_MASS_SCAL));
}
