/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#include "For_Kadath/Kadath_point_h/kadath_bin_ns.hpp"
#include <Hydro/EOS.hh>
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"
#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Utilities/exporter_utilities.hpp"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "For_Kadath/IO/be_file_source.hpp"
#include "Apps/Diagnostics/configured_eos.hpp"

#include <cmath>
#include <stdexcept>

using namespace Kadath;
using namespace export_utils;

std::array<std::vector<double>, NUM_OUT> KadathExportBNS(int const npoints, double const* xx, double const* yy,
                                                         double const* zz, char const* fn)
{
    std::string filename(fn);

    kadath_config<BIN_INFO> bconfig(filename);

    // get const EOS information - used for initializing EOS later
    const double h_cut = bconfig.eos<double>(HCUT, BCO1);
    const std::string eos_file = bconfig.eos<std::string>(EOSFILE, BCO1);
    const std::string eos_type = bconfig.eos<std::string>(EOSTYPE, BCO1);


    const double omega = bconfig(GOMEGA);
    const bool is_corotating = bconfig.control(COROT_BIN);
    double& ome1 = bconfig(OMEGA, BCO1);
    double& ang1 = bconfig(DEG, BCO1);
    double& ome2 = bconfig(OMEGA, BCO2);
    double& ang2 = bconfig(DEG, BCO2);
    const double axis = bconfig(COM);
    double yaxis = 0.;
    if (!std::isnan(bconfig.set(COMY)))
        yaxis = bconfig(COMY);

    /* file containing KADATH fields must have same name as config file
     * with only the extension being different */
    std::string kadath_filename = bconfig.space_filename();

    BeFileSource fin(kadath_filename);
    Space_bin_ns space(fin);
    Scalar conf(space, fin);
    Scalar lapse(space, fin);
    Vector shift(space, fin);
    Scalar logh(space, fin);
    Scalar phi(space, fin);

    std::vector<std::reference_wrapper<const Scalar>> quants;
    for (int i = 0; i < NUM_QUANTS; ++i)
        quants.push_back(std::cref(conf));

    quants[PSI] = std::cref(conf);
    quants[ALP] = std::cref(lapse);

    Base_tensor basis(shift.get_basis());

    int ndom = space.get_nbr_domains();

    double xc1 = bco_utils::get_center(space, space.NS1);
    double xc2 = bco_utils::get_center(space, space.NS2);
    double xo = bco_utils::get_center(space, ndom - 1);

    Metric_flat fmet(space, basis);

    // init coordinate fields
    CoordFields<Space_bin_ns> cfields(space);
    vec_ary_t coord_vectors{default_binary_vector_ary(space)};
    update_fields(cfields, coord_vectors, {}, xo, xc1, xc2);
    Vector cart(space, CON, basis);
    cart = cfields.cart();

    System_of_eqs syst(space, 0, ndom - 1);

    fmet.set_system(syst, "f");

    Param p;
    // add EOS user defined OPEs based on EOS type
    if (eos_type == "Cold_Table") {
        using namespace Kadath::Margherita;
        using eos_t = Kadath::Margherita::Cold_Table;

        KadathApps::init_configured_cold_table(bconfig, BCO1);
        syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
        syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
        syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    } else if (eos_type == "Cold_PWPoly") {
        using namespace Kadath::Margherita;
        using eos_t = Kadath::Margherita::Cold_PWPoly;

        EOS<eos_t, eos_var_t::PRESSURE>::init(eos_file, h_cut);

        syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
        syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
        syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    } else {
        throw std::invalid_argument("Unsupported EOS type for BNS export: " + eos_type);
    } // end adding EOS OPEs

    syst.add_cst("4piG", 4.0 * M_PI);
    syst.add_cst("PI", M_PI);

    syst.add_cst("omes1", ome1);
    syst.add_cst("omes2", ome2);

    syst.add_cst("angs1", ang1 * std::acos(-1.) / 180.);
    syst.add_cst("angs2", ang2 * std::acos(-1.) / 180.);

    syst.add_cst("mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_cst("mmx", *coord_vectors[to_int(coord_vector::BCO1_ROTx)]);
    syst.add_cst("mmz", *coord_vectors[to_int(coord_vector::BCO1_ROTz)]);
    syst.add_cst("mpx", *coord_vectors[to_int(coord_vector::BCO2_ROTx)]);
    syst.add_cst("mpz", *coord_vectors[to_int(coord_vector::BCO2_ROTz)]);

    syst.add_cst("ex", *coord_vectors[to_int(coord_vector::EX)]);
    syst.add_cst("ey", *coord_vectors[to_int(coord_vector::EY)]);
    syst.add_cst("ez", *coord_vectors[to_int(coord_vector::EZ)]);

    syst.add_cst("sm", *coord_vectors[to_int(coord_vector::S_BCO1)]);
    syst.add_cst("sp", *coord_vectors[to_int(coord_vector::S_BCO2)]);
    syst.add_cst("einf", *coord_vectors[to_int(coord_vector::S_INF)]);

    syst.add_cst("xaxis", axis);
    syst.add_cst("yaxis", yaxis);
    syst.add_cst("ome", omega);

    syst.add_cst("P", conf);
    syst.add_cst("N", lapse);
    syst.add_cst("bet", shift);
    syst.add_cst("phi", phi);

    syst.add_cst("H", logh);

    syst.add_def("NP = P*N");
    syst.add_def("Ntilde = N / P^6");

    syst.add_def("Morb^i = mg^i + xaxis * ey^i + yaxis * ex^i");
    std::string orbital_shift{"omega^i = bet^i + ome * Morb^i"};
    if (!std::isnan(bconfig.set(ADOT))) {
        syst.add_cst("adot", bconfig(ADOT));
        syst.add_cst("r", cart);
        syst.add_def("comr^i = r^i - xaxis * ex^i + yaxis * ey^i");
        orbital_shift += " + adot * comr^i";
    }
    syst.add_def(orbital_shift.c_str());

    for (int d = space.NS1; d <= space.ADAPTED1; ++d) {
        syst.add_def(d, "s^i  = omes1 * ( cos(angs1) * mmz^i + sin(angs1) * mmx^i ) ");
    }
    for (int d = space.NS2; d <= space.ADAPTED2; ++d) {
        syst.add_def(d, "s^i  = omes2 * ( cos(angs2) * mpz^i + sin(angs2) * mpx^i ) ");
    }

    syst.add_def("A_ij = (D_i bet_j + D_j bet_i - 2. / 3.* D^k bet_k * f_ij) /2. / N");

    syst.add_def("h = exp(H)");

    if (is_corotating) {
        syst.add_def("U^i = omega^i / N");
    } else {
        for (int d = 0; d < ndom; ++d) {
            if ((d <= space.ADAPTED1) || (d <= space.ADAPTED2 && d >= space.NS2))
                syst.add_def(d, "eta_i = D_i phi + P^4 * s_i");
            else
                syst.add_def(d, "eta_i = D_i phi");
        }

        syst.add_def("Wsquare = eta^i * eta_i / h^2 / P^4 + 1.");
        syst.add_def("W = sqrt(Wsquare)");

        syst.add_def("U^i = eta^i / P^4 / h / W");
    }

    Tensor A(syst.give_val_def("A"));
    Index ind(A);

    quants[AXX] = std::cref(A(ind));
    ind.inc();
    quants[AXY] = std::cref(A(ind));
    ind.inc();
    quants[AXZ] = std::cref(A(ind));
    ind.inc();
    ind.inc();
    quants[AYY] = std::cref(A(ind));
    ind.inc();
    quants[AYZ] = std::cref(A(ind));
    ind.inc();
    ind.inc();
    ind.inc();
    quants[AZZ] = std::cref(A(ind));

    quants[H] = std::cref(logh);

    Vector vel_kad(syst.give_val_def("U"));

    quants[UX] = std::cref(vel_kad(1));
    quants[UY] = std::cref(vel_kad(2));
    quants[UZ] = std::cref(vel_kad(3));

    quants[BETX] = std::cref(shift(1));
    quants[BETY] = std::cref(shift(2));
    quants[BETZ] = std::cref(shift(3));

    std::array<std::vector<double>, NUM_OUT> out;
    for (auto& v : out)
        v.resize(npoints);

    for (int i = 0; i < npoints; ++i) {
        std::vector<double> quant_vals(NUM_QUANTS);

        // get number of dimensions and construct index
        int ndim = 3;

        // construct Kadath point, shifted wrt the COM
        Point abs_coords(ndim);
        abs_coords.set(1) = xx[i] - axis;
        abs_coords.set(2) = yy[i] - yaxis;
        abs_coords.set(3) = zz[i];

        for (int k = 0; k < NUM_QUANTS; ++k) {
            quant_vals[k] = quants[k].get().val_point(abs_coords);
        }

        auto const psi = quant_vals[PSI];
        auto const psi2 = psi * psi;
        auto const psi4 = psi2 * psi2;

        out[ALPHA][i] = quant_vals[ALP];

        out[BETAX][i] = quant_vals[BETX];
        out[BETAY][i] = quant_vals[BETY];
        out[BETAZ][i] = quant_vals[BETZ];

        double g[3][3];
        g[0][0] = psi4;
        g[0][1] = 0.0;
        g[0][2] = 0.0;
        g[1][1] = psi4;
        g[1][2] = 0.0;
        g[2][2] = psi4;
        g[1][0] = g[0][1];
        g[2][0] = g[0][2];
        g[2][1] = g[1][2];

        out[GXX][i] = g[0][0];
        out[GXY][i] = g[0][1];
        out[GXZ][i] = g[0][2];
        out[GYY][i] = g[1][1];
        out[GYZ][i] = g[1][2];
        out[GZZ][i] = g[2][2];

        out[KXX][i] = quant_vals[AXX] * psi4;
        out[KXY][i] = quant_vals[AXY] * psi4;
        out[KXZ][i] = quant_vals[AXZ] * psi4;
        out[KYY][i] = quant_vals[AYY] * psi4;
        out[KYZ][i] = quant_vals[AYZ] * psi4;
        out[KZZ][i] = quant_vals[AZZ] * psi4;

        double h = std::exp(quant_vals[H]);

        // get quantities point-wise, since h is smoothest, and cut data at h=1
        if (h <= 1.) {
            out[RHO][i] = 0.;
            out[EPS][i] = 0.;
            out[PRESS][i] = 0.;
            out[VELX][i] = 0.;
            out[VELY][i] = 0.;
            out[VELZ][i] = 0.;
        } else {
            if (eos_type == "Cold_Table") {
                using namespace Kadath::Margherita;

                out[RHO][i] = EOS<Cold_Table, eos_var_t::DENSITY>::get(h);
                out[EPS][i] = EOS<Cold_Table, eos_var_t::EPSILON>::get(h);
                out[PRESS][i] = EOS<Cold_Table, eos_var_t::PRESSURE>::get(h);
            }

            if (eos_type == "Cold_PWPoly") {
                using namespace Kadath::Margherita;

                out[RHO][i] = EOS<Cold_PWPoly, eos_var_t::DENSITY>::get(h);
                out[EPS][i] = EOS<Cold_PWPoly, eos_var_t::EPSILON>::get(h);
                out[PRESS][i] = EOS<Cold_PWPoly, eos_var_t::PRESSURE>::get(h);
            }

            out[VELX][i] = quant_vals[UX];
            out[VELY][i] = quant_vals[UY];
            out[VELZ][i] = quant_vals[UZ];
        }
    } // for i

    return out;
}
