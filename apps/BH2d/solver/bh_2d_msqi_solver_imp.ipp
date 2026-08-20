
#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include "mpi.h"
#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "bh_2d_msqi_regrid.hpp"
#include "For_Kadath/Array/memory.hpp"
#include <cmath>
#include <cstdlib>

namespace Kadath {

static void isotropic_kerr(System_of_eqs& syst)
{
    syst.add_def("sint = multsint(ones)");
    syst.add_def("aKerr = CM * chi * ones");
    syst.add_def("rH = sqrt(CM^2 - aKerr^2) * ones / 2");
    syst.add_def("rK = rH * (1 + (CM + aKerr) / 2 / rH) * (1 + (CM - aKerr) / 2 / rH)");
    syst.add_def("acos2 = aKerr^2 * (1 - sint^2)");
    syst.add_def("sigma = rK^2 + acos2");
    syst.add_def("delta = rK^2 - 2*CM*rK + aKerr^2");
    syst.add_def("xx = (rK^2 + aKerr^2) * sigma + 2*CM*aKerr^2*rK*sint^2");
    syst.add_def("AsqHorizon = sigma / rH^2");
    syst.add_def("BsqHorizon = xx / sigma / rH^2");
}

template <typename config_t, typename space_t>
bh_2d_msqi_solver<config_t, space_t>::     // Scope (Class Name)
    bh_2d_msqi_solver(config_t& config_in, // Function Name (Must match Class Name for constructors)
                      space_t& space_in, Scalar& lapse_in, Scalar& metA_in, Scalar& metB_in, Scalar& beta_in,
                      Scalar& varscal_in)
    : Solver<config_t, space_t>(config_in, space_in), lapse(lapse_in), metA(metA_in), metB(metB_in), beta(beta_in),
      varscal(varscal_in), cfields(space_in)
{
    beta.affect_parameters();
    beta.set_parameters()->set_m_quant() = 1;
    beta.std_base();

    varscal.affect_parameters();
    varscal.set_parameters()->set_m_quant() = 1;
    varscal.std_base();
}

template <typename config_t, typename space_t> void bh_2d_msqi_solver<config_t, space_t>::syst_init(System_of_eqs& syst)
{

    const int ndom = space.get_nbr_domains();

    Scalar oneField(space);
    oneField = 1.;
    oneField.std_base();
    syst.add_cst("ones", oneField);
    syst.add_cst("4piG", 4.0 * M_PI);
    syst.add_cst("PI", M_PI);
    syst.add_cst("M", bconfig(MIRR));
    syst.add_cst("CM", bconfig(MCH));
    syst.add_cst("chi", bconfig(CHI));
    syst.add_var("ome", bconfig(OMEGA));
    syst.add_cst("mu", bconfig.template gravity<double>(GRAV_MASS_SCAL));

    // the basic fields, conformal factor, lapse and (log) enthalpy
    syst.add_var("N", lapse);
    syst.add_var("A", metA);
    syst.add_var("B", metB);
    syst.add_var("shift", beta);
    if (bconfig.template gravity<double>(GRAV_MASS_SCAL) == 0.0) {
        syst.add_def("sphi=0*ones");
    } else {
        syst.add_var("sphi", varscal);
    }

    Scalar radius(oneField);
    Scalar overR(oneField);
    for (int d = 0; d < ndom; d++) {
        radius.set_domain(d) =
            sqrt(pow(space.get_domain(d)->get_cart(1), 2) + pow(space.get_domain(d)->get_cart(2), 2));
        overR.set_domain(d) = 1 / radius(d);
    }
    radius.std_base();
    overR.std_base();

    syst.add_cst("r", radius);
    syst.add_cst("invr", overR);
}

template <typename config_t, typename space_t>
void bh_2d_msqi_solver<config_t, space_t>::add_common_defs(System_of_eqs& syst, bool fixed_lapse)
{

    if (fixed_lapse) {
        syst.add_cst("n0", bconfig(FIXED_LAPSE));
    }

    syst.add_def("winding = ones");
    syst.add_def("bet = divrsint(shift)");

    isotropic_kerr(syst);

    syst.add_def("gradN_i = grad(N)");
    syst.add_def("gradB_i = grad(B)");
    syst.add_def("gradbet_i = grad(bet)");
    syst.add_def("gradsphi_i = grad(sphi)");

    syst.add_def("DADA    = scal( grad(A), grad(A)  )");
    syst.add_def("DNDlnB  = scal( gradN  , gradB  ) / B");
    syst.add_def("DbetDbet= scal( gradbet, gradbet)");
    syst.add_def("DbetDN  = scal( gradbet, gradN  )");
    syst.add_def("DbetDlnB= scal( gradbet, gradB  ) / B");
    syst.add_def("DsphDsph= scal( gradsphi, gradsphi ) ");

    syst.add_def("wmb2= (ome*ones + bet*winding)^2");
    syst.add_def("aziscal = N^2 / B^2 * divrsint(sphi) * divrsint(sphi)");
    syst.add_def("NmdivincB2 = N^2 * winding^2 * divrsint(sphi) * divrsint(ones-A^2/B^2)");

    syst.add_def("NNJ = 2 * winding * N * (ome*ones + bet*winding) * sphi^2 ");
    syst.add_def("NNE = (  wmb2* sphi^2 + aziscal) + N^2 * ( DsphDsph / A^2 +   mu^2 * sphi^2 )");
    syst.add_def("NNS = (3*wmb2* sphi^2 - aziscal) - N^2 * ( DsphDsph / A^2 + 3*mu^2 * sphi^2 )");
    syst.add_def("NNSphSph = (wmb2* sphi^2 + aziscal) - N^2 * ( DsphDsph / A^2 +   mu^2 * sphi^2 )");

    syst.add_def("lapNterms = N * DNDlnB - B^2 * multrsint(multrsint(DbetDbet)) / 2");
    syst.add_def("sourceNterms = - 4piG * A^2 * (NNE + NNS)");

    syst.add_def("lapShifts = - multrsint(DbetDN) + 3 * N * multrsint(DbetDlnB)");
    syst.add_def("sourceShifts = - divrsint( 4 * 4piG * A^2 * NNJ ) / B^2");

    syst.add_def("sourceBterms = - 2 * 4piG * A^2 * B * (NNS - NNSphSph)");

    syst.add_def("lapAterms = N * A * lap2(N) - N * N * DADA / A - 3 * A * B^2 * multrsint(multrsint(DbetDbet)) / 4");
    syst.add_def("sourceAterms = - 2 * 4piG * A^3 * NNSphSph");

    syst.add_def("lapscalar = A^2 * wmb2 * sphi + NmdivincB2 "
                 "+ N * scal( gradsphi, gradN ) + N^2 * scal( gradsphi, gradB ) / B ");
    syst.add_def("sourcescalar = - A^2 * mu^2 * N^2 * sphi");

    syst.add_def(ndom - 1, "intJk = multrsint(multrsint(dr(bet))) / 4 / 4piG");
    syst.add_def(ndom - 1, "intMadm = -dr(A^2+B^2) / 4 / 4piG");
    syst.add_def(ndom - 1, "intMk = dr(N) / 4piG");

    // definitions of integrals on the excision surface
    syst.add_def("THETA = dr(A)/A + dr(B)/B + divr(2*ones)");
    syst.add_def("intMsq= A * B / 4 / 4piG");
}

template <typename config_t, typename space_t>
void bh_2d_msqi_solver<config_t, space_t>::add_domain_equations(System_of_eqs& syst, bool fixed_lapse)
{
    for (int d = space.HOMOTHETIC_INNER; d < space.get_nbr_domains(); d++) {
        if (d == space.HOMOTHETIC_INNER) {
            syst.add_def(d, "eqN  = N * lap(N)     + lapNterms + sourceNterms ");
            syst.add_def(d, "eqA  = N*N * lap2(A)  + lapAterms + sourceAterms ");
            syst.add_def(d, "eqB  = N * (2 * lap(N*B) - lap2(N*B)) + sourceBterms ");
            syst.add_def(d, "eqbet= N * lap(shift) + lapShifts + sourceShifts ");
            if (bconfig.template gravity<double>(GRAV_MASS_SCAL) != 0.0)
                syst.add_def(d, "eqphi= N*N * lap(sphi) + lapscalar + sourcescalar ");
        } else {
            syst.add_def(d, "eqN  = lap(N)       + (lapNterms + sourceNterms) / N ");
            syst.add_def(d, "eqA  = N * lap2(A)  + (lapAterms + sourceAterms) / N ");
            syst.add_def(d, "eqB  = (2 * lap(N*B) - lap2(N*B)) + sourceBterms / N ");
            syst.add_def(d, "eqbet= lap(shift)   + (lapShifts + sourceShifts) / N ");
            if (bconfig.template gravity<double>(GRAV_MASS_SCAL) != 0.0)
                syst.add_def(d, "eqphi= N * lap(sphi) + (lapscalar + sourcescalar) / N ");
        }
    }

    space.add_eq(syst, "eqN=0", "N", "dn(N)");
    space.add_eq(syst, "eqA=0", "A", "dn(A)");
    space.add_eq(syst, "eqB=0", "B", "dn(B)");
    space.add_eq(syst, "eqbet=0", "shift", "dn(shift)");
    if (bconfig.template gravity<double>(GRAV_MASS_SCAL) != 0.0)
        space.add_eq_free_horizon(syst, "eqphi=0", "sphi", "dn(sphi)");
}

template <typename config_t, typename space_t>
void bh_2d_msqi_solver<config_t, space_t>::add_boundary_conditions(System_of_eqs& syst, bool fixed_lapse)
{

    space.add_bc_inf(syst, "N=1");
    space.add_bc_inf(syst, "A=1");
    space.add_bc_inf(syst, "B=1");
    space.add_bc_inf(syst, "shift=0");
    if (bconfig.template gravity<double>(GRAV_MASS_SCAL) != 0.0)
        space.add_bc_inf(syst, "sphi=0");

    if (fixed_lapse) {
        space.add_bc_bh(syst, "N = n0");
    } else {
        space.add_bc_bh(syst, "dn(NP) = 0");
    }

    if (bconfig(FIXED_LAPSE) == 0) {
        space.add_bc_bh(syst, "THETA=0");
        space.add_bc_bh(syst, "dr(A)/A - dr(B)/B = 0");
    } else {
        space.add_bc_bh(syst, "A = sqrt(AsqHorizon)");
        space.add_bc_bh(syst, "B = sqrt(BsqHorizon)");
    }

    space.add_bc_bh(syst, "shift = multrsint( - ome / winding )");

    // In the homothetic inner domain, the mapping has one free DOF to make the integral match MIRR
    space.add_eq_int_horizon(syst, "integ(intMsq) - M * M = 0");
    space.add_eq_int_inf(syst, "integ(intJk) - chi * CM * CM = 0");

    for (int i : excluded_doms) {
        syst.add_eq_full(i, "N = 0");
        syst.add_eq_full(i, "A = 0");
        syst.add_eq_full(i, "B = 0");
        syst.add_eq_full(i, "shift = 0");
        if (bconfig.template gravity<double>(GRAV_MASS_SCAL) != 0.0)
            syst.add_eq_full(i, "sphi = 0");
    }
}

template <typename config_t, typename space_t> double bh_2d_msqi_solver<config_t, space_t>::compute_max_sphi() const
{
    int ndom = space.get_nbr_domains();
    double max_sphi = 0.0;
    for (int d = 2; d < ndom; ++d) {
        const Dim_array nbr_pts = space.get_domain(d)->get_nbr_points();
        if (nbr_pts.get_ndim() < 2) {
            continue;
        }
        Index pos(nbr_pts);
        for (int i = 0; i < nbr_pts(0); ++i) {
            pos.set(0) = i;
            for (int j = 0; j < nbr_pts(1); ++j) {
                pos.set(1) = j;
                double val = std::fabs(varscal.set_domain(d)(pos));
                if (val > max_sphi) {
                    max_sphi = val;
                }
            }
        }
    }
    return max_sphi;
}

template <typename config_t, typename space_t>
void bh_2d_msqi_solver<config_t, space_t>::print_diagnostics(System_of_eqs const& syst, const int ite,
                                                             const double conv) const
{

    int ndom = space.get_nbr_domains();
    double r = bco_utils::get_radius(space.get_domain(1), OUTER_BC);

    Val_domain integMsq(syst.give_val_def("intMsq")()(2));
    Val_domain integJk(syst.give_val_def("intJk")()(ndom - 1));
    Val_domain integMadm(syst.give_val_def("intMadm")()(ndom - 1));
    Val_domain integMk(syst.give_val_def("intMk")()(ndom - 1));

    double Mirrsq = space.get_domain(2)->integ(integMsq, INNER_BC);
    double J = space.get_domain(ndom - 1)->integ(integJk, OUTER_BC);
    double Madm = space.get_domain(ndom - 1)->integ(integMadm, OUTER_BC);
    double Mk = space.get_domain(ndom - 1)->integ(integMk, OUTER_BC);

    double Mirr = std::sqrt(Mirrsq);
    double Mch = Mirr * std::sqrt(1 + J * J / Mirrsq / Mirrsq / 4);

    double max_sphi = compute_max_sphi();
    double max_theta_inner = 0.0;
    const int inner_dom = space.HOMOTHETIC_INNER;
    const Dim_array theta_pts = space.get_domain(inner_dom)->get_nbr_points();
    if (theta_pts.get_ndim() >= 2) {
        Val_domain theta_def(syst.give_val_def("THETA")()(inner_dom));
        Index pos(theta_pts);
        pos.set(0) = 0;
        for (int j = 0; j < theta_pts(1); ++j) {
            pos.set(1) = j;
            double val = std::fabs(theta_def(pos));
            if (val > max_theta_inner) {
                max_theta_inner = val;
            }
        }
    }

#define FORMAT std::setw(13) << std::left << std::showpos
    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << "=======================================" << std::endl
              << FORMAT << "Iter: " << ite << std::endl
              << FORMAT << "Error: " << conv << std::endl
              << FORMAT << "Mirr: " << Mirr << std::endl
              << FORMAT << "R: " << r << std::endl
              << FORMAT << "Omega: " << bconfig(OMEGA) << std::endl
              << FORMAT << "J Komar: " << J << std::endl
              << FORMAT << "Mch: " << Mch << std::endl
              << FORMAT << "M adm: " << Madm << std::endl
              << FORMAT << "Virial err: " << (Mk - Madm) / (Mk + Madm) << std::endl
              << FORMAT << "Sphi: " << max_sphi << std::endl
              << FORMAT << "ThetaMax: " << max_theta_inner << std::endl;
    std::cout.flags(f);
    std::cout << "=======================================" << "\n\n";
#undef FORMAT
}

// standardized filename for each converged dataset at the end of each stage.
template <typename config_t, typename space_t>
std::string bh_2d_msqi_solver<config_t, space_t>::converged_filename(const std::string& stage) const
{
    return converged_bh2d_scalar_filename(bconfig, space, stage);
}


} // namespace Kadath
