// GR binary-NS stages: thin wrappers over the shared stage bodies in
// Apps/Formalism/Shared/Stages/bns_stages_common.ipp. The symmetric solver supplies the
// plain in-star shift definition (no rotational-gauge forcing) and no-op
// ADM z-momentum hooks (z-reflection symmetry forbids a net Pz).

#include "Apps/Formalism/Shared/Stages/bns_stages_common.ipp"

#include <string>

namespace Kadath {

// Symmetric BNS: under z-reflection symmetry the in-star shift equation carries
// no rotational-gauge forcing.
namespace {
inline std::string bns_sym_matter_eqbet_def(int /*d*/)
{
    return "eqbet^i = delta * D_j D^j bet^i + delta * D^i D_j bet^j / 3. "
           "- 2. * delta * A^ij * D_j Ntilde - 4. * 4piG * N * P^4 * ptilde^i";
}
} // namespace

template <class eos_t, typename config_t, typename space_t>
int bns_xcts_solver<eos_t, config_t, space_t>::hydrostatic_equilibrium_stage()
{
    return bns_run_force_balance_stage(
        *this, bns_sym_matter_eqbet_def,
        [](System_of_eqs&, auto&) -> std::string { return {}; },  // no Pz closure under z-reflection symmetry
        [](auto&, System_of_eqs&) {});                            // no Pz constraint
}

template <class eos_t, typename config_t, typename space_t>
int bns_xcts_solver<eos_t, config_t, space_t>::ecc_red_stage()
{
    return hydro_rescaling_stage(ECC_RED, "ECC_RED");
}

template <class eos_t, typename config_t, typename space_t>
int bns_xcts_solver<eos_t, config_t, space_t>::hydro_rescaling_stage(STAGE stage, const std::string& stage_text)
{
    return bns_run_hydro_rescaling_stage(
        *this, stage, stage_text, bns_sym_matter_eqbet_def,
        [](System_of_eqs&, auto&) -> std::string { return {}; },  // no Pz closure under z-reflection symmetry
        [](auto&, System_of_eqs&) {});                            // no Pz constraint
}

} // namespace Kadath
