#pragma once


#include "Hydro/EOS.hh"
#include "For_Kadath/Array/exceptions.hpp"

#include <cmath>

namespace Kadath::ns_2d_xcts_core {

inline void validate_spin(const double chi)
{
    if (!std::isfinite(chi))
        KADATH_THROW("NS2d XCTS spin chi must be finite");
}

template <class eos_t, typename config_t>
inline void add_base_system(System_of_eqs& syst, config_t& bconfig, Scalar& conf, Scalar& lapse,
                            Scalar& logh, Scalar& shift, Scalar& Omg)
{
    using namespace Kadath::Margherita;

    auto& space = syst.get_space();
    Scalar one_field(space);
    one_field = 1.;
    one_field.std_base();
    syst.add_cst("ones", one_field);
    syst.add_cst("4piG", 4.0 * M_PI);

    syst.add_var("P", conf);
    syst.add_var("N", lapse);
    syst.add_var("H", logh);
    syst.add_var("Omg", Omg);
    syst.add_var("brsint", shift);

    Param eos_parameters;
    syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &eos_parameters);
    syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &eos_parameters);
    syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &eos_parameters);
    syst.add_ope("dHdlnrho", &EOS<eos_t, eos_var_t::DHDRHO>::action, &eos_parameters);

    syst.add_cst("chi", bconfig(CHI));
    syst.add_var("ome", bconfig(OMEGA));
    if (bconfig.control(MB_FIXING)) {
        syst.add_cst("Mb", bconfig(MB));
        syst.add_var("Madm", bconfig(MADM));
    } else {
        syst.add_var("Mb", bconfig(MB));
        syst.add_cst("Madm", bconfig(MADM));
    }
}

inline void add_common_definitions(System_of_eqs& syst, const int ndom)
{
    syst.add_def("NP = P*N");
    syst.add_def("Ntilde = N / P^6");
    syst.add_def("bet = divrsint(brsint)");
    syst.add_def("gradbet_i = grad(bet)");
    syst.add_def("Asquare = multrsint(multrsint(scal(gradbet, gradbet))) / 2. / Ntilde^2");

    syst.add_def(ndom - 1, "intMadm = -dr(P) * 2 / 4piG");
    syst.add_def(ndom - 1, "intMk = dr(N)  / 4piG");
    syst.add_def(ndom - 1, "intJ = multrsint(multrsint(dr(bet))) / 4 / 4piG");

    syst.add_def("h = exp(H)");
    syst.add_def("rho = rho(h)");
    syst.add_def("eps = eps(h)");
    syst.add_def("press = press(h)");
    syst.add_def("dHdlnrho = dHdlnrho(h)");
    syst.add_def("delta = h - eps - 1.");
}

template <typename space_t>
inline void add_rotation_constraints(space_t& space, System_of_eqs& syst, const int ndom)
{
    const int adapted_inner = space.ADAPTED_INNER;
    syst.add_eq_matching(adapted_inner, INNER_BC, "Omg");
    syst.add_eq_matching(adapted_inner, INNER_BC, "dn(Omg)");
    syst.add_eq_inside(adapted_inner, "Omg = 0");
    for (int d = adapted_inner + 1; d < ndom; ++d)
        syst.add_eq_full(d, "Omg = 0");
}

template <typename space_t>
inline void add_global_conditions(space_t& space, System_of_eqs& syst, const int ndom)
{
    const int adapted_outer = space.ADAPTED_OUTER;
    syst.add_eq_bc(ndom - 1, OUTER_BC, "N=1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "P=1");
    syst.add_eq_bc(ndom - 1, OUTER_BC, "brsint=0");

    syst.add_eq_bc(adapted_outer, OUTER_BC, "H = 0");
    syst.add_eq_first_integral(0, adapted_outer, "firstint", "H - Hc");

    space.add_eq_int_volume(syst, adapted_outer + 1, "integvolume(intMb) = Mb");
    space.add_eq_int_inf(syst, "integ(intJ) - chi * Madm * Madm = 0");
    space.add_eq_int_inf(syst, "integ(intMadm) = Madm");
}

} // namespace Kadath::ns_2d_xcts_core
