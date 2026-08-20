#pragma once

#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"
#include <cmath>
#include <string>

namespace Kadath {

template <typename space_t, typename config_t>
void ns_3d_xcts_print_diagnostics(const space_t& space, config_t& bconfig,
                                  const System_of_eqs& syst, int ite, double conv)
{
    int ndom = space.get_nbr_domains();

    double baryonic_mass = 0.;
    for (int d = 0; d <= 1; ++d)
        baryonic_mass += syst.give_val_def_scalar_domain("intMb", d).integ_volume();

    // Domain-definition access returns a short-lived view. Integrate each view
    // before asking the system to evaluate another definition.
    double Madm = space.get_domain(ndom - 1)->integ(
        syst.give_val_def_scalar_domain("intMadm", ndom - 1), OUTER_BC);

    double Mk = space.get_domain(ndom - 1)->integ(
        syst.give_val_def_scalar_domain("intMk", ndom - 1), OUTER_BC);

    auto rs = bco_utils::get_rmin_rmax(space, 1);

    double Madmalt = space.get_domain(ndom - 1)->integ(
        syst.give_val_def_scalar_domain("intMadmalt", ndom - 1), OUTER_BC);

    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << "=======================================" << std::endl
              << FORMAT << "Iter: " << ite << std::endl
              << FORMAT << "Error: " << conv << std::endl
              << FORMAT << "Mb: " << baryonic_mass << std::endl
              << FORMAT << "Madm: " << Madm << std::endl
              << FORMAT << "Madm_ql: " << Madmalt << " [" << std::abs(Madm - Madmalt) / Madm << "]" << std::endl
              << FORMAT << "Mk: " << Mk << " [" << std::abs(Madm - Mk) / Madm << "]" << std::endl;
    std::cout << FORMAT << "R: " << rs[0] << " " << rs[1] << "\n";

    double J = space.get_domain(ndom - 1)->integ(
        syst.give_val_def_scalar_domain("intJ", ndom - 1), OUTER_BC);

    std::cout << FORMAT << "Jadm: " << J << std::endl
              << FORMAT << "Chi: " << J / Madm / Madm << " [" << bconfig(CHI) << "]\n"
              << FORMAT << "Omega: " << bconfig(OMEGA) << std::endl;
    std::cout.flags(f);
}

} // namespace Kadath
