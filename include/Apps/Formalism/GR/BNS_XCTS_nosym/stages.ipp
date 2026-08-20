// GR no-symmetry binary-NS stages: thin wrappers over the shared stage bodies in
// Apps/Formalism/Shared/Stages/bns_stages_common.ipp. The no-symmetry solver supplies
// the bordered rotational-gauge forcing on the in-star shift equation and the
// ADM z-momentum closure (the symmetric solver has neither — see that solver's
// stages.ipp for the trivial hooks).

#include "Apps/Formalism/Shared/Stages/bns_stages_common.ipp"
#include "Apps/Formalism/Shared/nosym_pz_closure.hpp"

#include <string>

namespace Kadath {

// nosym in-star shift equation: append the per-star rigid-rotation gauge forcing
// (muz1*mmz on NS1, muz2*mpz on NS2) when the rotational gauge is enabled, so the
// bordered multiplier gets a non-degenerate column. Gauge off => plain shift eq
// (bit-identical to the symmetric solver).
namespace {
inline std::string bns_nosym_matter_eqbet_def(bool gauge_enabled, int d, int ns1, int adapted1, int ns2,
                                              int adapted2)
{
    std::string eqbet_def = "eqbet^i = delta * D_j D^j bet^i + delta * D^i D_j bet^j / 3. "
                            "- 2. * delta * A^ij * D_j Ntilde - 4. * 4piG * N * P^4 * ptilde^i";
    if (gauge_enabled) {
        if (d >= ns1 && d <= adapted1)
            eqbet_def += " + muz1 * mmz^i";
        else if (d >= ns2 && d <= adapted2)
            eqbet_def += " + muz2 * mpz^i";
    }
    return eqbet_def;
}
} // namespace

template <class eos_t, typename config_t, typename space_t>
int bns_xcts_nosym_solver<eos_t, config_t, space_t>::hydrostatic_equilibrium_stage()
{
    return bns_run_force_balance_stage(
        *this,
        [this](int d) {
            return bns_nosym_matter_eqbet_def(rotational_gauge_enabled(), d, this->space.NS1, this->space.ADAPTED1,
                                              this->space.NS2, this->space.ADAPTED2);
        },
        [](System_of_eqs& syst, auto& bconfig) -> std::string {
            return nosym_closure::append_pz_closure_velocity(syst, bconfig);
        },
        [](auto& space, System_of_eqs& syst) { nosym_closure::add_pz_constraint(space, syst); });
}

template <class eos_t, typename config_t, typename space_t>
int bns_xcts_nosym_solver<eos_t, config_t, space_t>::ecc_red_stage()
{
    return hydro_rescaling_stage(ECC_RED, "ECC_RED");
}

template <class eos_t, typename config_t, typename space_t>
int bns_xcts_nosym_solver<eos_t, config_t, space_t>::hydro_rescaling_stage(STAGE stage, const std::string& stage_text)
{
    return bns_run_hydro_rescaling_stage(
        *this, stage, stage_text,
        [this](int d) {
            return bns_nosym_matter_eqbet_def(rotational_gauge_enabled(), d, this->space.NS1, this->space.ADAPTED1,
                                              this->space.NS2, this->space.ADAPTED2);
        },
        [](System_of_eqs& syst, auto& bconfig) -> std::string {
            return nosym_closure::append_pz_closure_velocity(syst, bconfig);
        },
        [](auto& space, System_of_eqs& syst) { nosym_closure::add_pz_constraint(space, syst); });
}

} // namespace Kadath
