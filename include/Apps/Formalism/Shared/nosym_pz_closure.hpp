#pragma once
// ADM z-momentum closure for nosym binaries.
//
// Spaces without z-reflection symmetry (the *_nosym binaries) admit a nonzero
// ADM linear momentum along z — e.g. tilted spins. The Px/Py closure via the
// center-of-mass unknowns (xaxis/yaxis) cannot reach Pz: both enter B^i only
// through the orbital rotation vector ome * Morb^i, and a rotation about z
// generates no z-velocity. Without a closure the solved data may carry a
// residual Pz and the binary drifts along z under evolution.
//
// Closure: append a uniform z-velocity  zvel * ez^i  to the co-moving frame
// velocity B^i and pin it with the momentum integral at infinity
// integ(intPz) = 0. A constant translation has vanishing conformal Killing
// operator, so zvel enters the system exclusively through the fluid velocity
// (U^i / V^i / firstint) — the same indirect mechanism by which xaxis/yaxis
// close Px/Py. One unknown, one integral condition: the system stays square.
//
// Method reference: this is the out-of-plane component of the boost-velocity
// center-of-mass control of Ossokine et al., Class. Quant. Grav. 32, 245010
// (2015) [arXiv:1506.01689] — there the uniform translation is added to the BBH
// shift outer boundary condition; here it enters the fluid velocity instead.
//
// Usage in a nosym hydro-rescaling (or equivalent) stage:
//   std::string bigB{"B^i = bet^i + ome * Morb^i"};
//   bigB += nosym_closure::append_pz_closure_velocity(syst, bconfig);
//   ...
//   space.add_eq_int_inf(syst, "integ(intPx) = 0");
//   space.add_eq_int_inf(syst, "integ(intPy) = 0");
//   nosym_closure::add_pz_constraint(space, syst);
//
// Prerequisites (already satisfied by the shared binary syst_init helpers):
//   - "ez" registered as a constant coordinate vector
//   - intPz defined at ndom-1 (add_extrinsic_curvature_and_adm_momenta)
// The converged zvel persists in bconfig(COM_VELZ) ("com_velz"), next to
// COM/COMY, so warm starts resume from the previous closure value.

#include "For_Kadath/Config/config_enums.hpp"
#include "For_Kadath/System_of_eqs/system_of_eqs.hpp"

#include <cmath>
#include <string>

namespace Kadath {
namespace nosym_closure {

// Registers the closure unknown "zvel" (persisted in bconfig(COM_VELZ)) and
// returns the velocity term to append to the B^i definition string.
template <typename config_t>
inline std::string append_pz_closure_velocity(System_of_eqs& syst, config_t& bconfig)
{
    if (std::isnan(bconfig.set(COM_VELZ)))
        bconfig.set(COM_VELZ) = 0.;
    syst.add_var("zvel", bconfig(COM_VELZ));
    return " + zvel * ez^i";
}

// Pins the closure unknown with the z ADM momentum integral at infinity.
template <typename space_t>
inline void add_pz_constraint(space_t& space, System_of_eqs& syst)
{
    space.add_eq_int_inf(syst, "integ(intPz) = 0");
    syst.set_last_eq_int_reflection_sector(-1);
}

} // namespace nosym_closure
} // namespace Kadath
