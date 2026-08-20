#pragma once

#include "For_Kadath/Utilities/Exporters/bco_geometry.hpp"
#include "Apps/Helper/solver_base.hpp"
#include "For_Kadath/Kadath_point_h/kadath_adapted.hpp"

#include <string>

namespace Kadath {

template <typename space_t, typename config_t>
void bns_xcts_print_diagnostics(const space_t& space, config_t& bconfig,
                                const System_of_eqs& syst, int ite, double conv,
                                const char* scalar_label = nullptr,
                                double scalar_ns1 = 0., double scalar_ns2 = 0.)
{
    int ndom = space.get_nbr_domains();

    double baryonic_mass1 = 0., ql_mass1 = 0.;
    for (int d = space.NS1; d <= space.ADAPTED1; ++d) {
        baryonic_mass1 += syst.give_val_def_scalar_domain("intMb", d).integ_volume();
        ql_mass1 += syst.give_val_def_scalar_domain("intM", d).integ_volume();
    }

    double baryonic_mass2 = 0., ql_mass2 = 0.;
    for (int d = space.NS2; d <= space.ADAPTED2; ++d) {
        baryonic_mass2 += syst.give_val_def_scalar_domain("intMb", d).integ_volume();
        ql_mass2 += syst.give_val_def_scalar_domain("intM", d).integ_volume();
    }

    const Val_domain& integPx = syst.give_val_def_scalar_domain("intPx", ndom - 1);
    const Val_domain& integPy = syst.give_val_def_scalar_domain("intPy", ndom - 1);
    const Val_domain& integPz = syst.give_val_def_scalar_domain("intPz", ndom - 1);
    double Px = space.get_domain(ndom - 1)->integ(integPx, OUTER_BC);
    double Py = space.get_domain(ndom - 1)->integ(integPy, OUTER_BC);
    double Pz = space.get_domain(ndom - 1)->integ(integPz, OUTER_BC);

    const Val_domain& integS1 = syst.give_val_def_scalar_domain("intS1", space.ADAPTED1 + 1);
    double S1 = space.get_domain(space.ADAPTED1 + 1)->integ(integS1, OUTER_BC);
    double chi1 = S1 / bconfig(MADM, BCO1) / bconfig(MADM, BCO1);
    double chiql1 = S1 / ql_mass1 / ql_mass1;

    const Val_domain& integS2 = syst.give_val_def_scalar_domain("intS2", space.ADAPTED2 + 1);
    double S2 = space.get_domain(space.ADAPTED2 + 1)->integ(integS2, OUTER_BC);
    double chi2 = S2 / bconfig(MADM, BCO2) / bconfig(MADM, BCO2);
    double chiql2 = S2 / ql_mass2 / ql_mass2;

    auto print_rs = [&](int dom) {
        auto rs = bco_utils::get_rmin_rmax(space, dom);
        std::cout << FORMAT << "NS-Rs: " << rs[0] << " " << rs[1] << std::endl;
    };

    std::ios_base::fmtflags f(std::cout.flags());
    std::cout << "=======================================" << std::endl
              << FORMAT << "Iter: " << ite << std::endl
              << FORMAT << "Error: " << conv << std::endl
              << FORMAT << "Omega: " << bconfig(GOMEGA) << std::endl
              << FORMAT << "Axis: " << bconfig(COM) << std::endl
              << FORMAT << "P: [" << Px << ", " << Py << ", " << Pz << "]\n\n"

              << FORMAT << "NS1-Mb: " << baryonic_mass1 << std::endl
              << FORMAT << "NS1-Madm_ql: " << ql_mass1 << std::endl
              << FORMAT << "NS1-S: " << S1 << std::endl
              << FORMAT << "NS1-Chi: " << chi1 << std::endl
              << FORMAT << "NS1-Chi_ql: " << chiql1 << std::endl
              << FORMAT << "NS1-OmegaS: " << bconfig(OMEGA, BCO1) << "\n";
    if (scalar_label)
        std::cout << FORMAT << (std::string("NS1-") + scalar_label + ": ") << scalar_ns1 << "\n";
    print_rs(space.ADAPTED1);

    std::cout << "\n"
              << FORMAT << "NS2-Mb: " << baryonic_mass2 << std::endl
              << FORMAT << "NS2-Madm_ql: " << ql_mass2 << std::endl
              << FORMAT << "NS2-S: " << S2 << std::endl
              << FORMAT << "NS2-Chi: " << chi2 << std::endl
              << FORMAT << "NS2-Chi_ql: " << chiql2 << std::endl
              << FORMAT << "NS2-OmegaS: " << bconfig(OMEGA, BCO2) << std::endl;
    if (scalar_label)
        std::cout << FORMAT << (std::string("NS2-") + scalar_label + ": ") << scalar_ns2 << "\n";
    print_rs(space.ADAPTED2);

    std::cout.flags(f);
    std::cout << "=======================================" << "\n\n";
}

} // namespace Kadath
