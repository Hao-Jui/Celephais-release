#pragma once

#include <cmath>

#include "Hydro/EOS.hh"
#include "For_Kadath/Utilities/Exporters/coord_fields.hpp"


namespace Kadath {
namespace xcts {

// EOS operator quartet. p is BORROWED — the caller must keep it alive as long as
// syst is used (same lifetime as the current inline `Param p` syst_init local).
template <class eos_t>
inline void add_eos_operators(System_of_eqs& syst, Param& p) {
    syst.add_ope("eps", &EOS<eos_t, eos_var_t::EPSILON>::action, &p);
    syst.add_ope("press", &EOS<eos_t, eos_var_t::PRESSURE>::action, &p);
    syst.add_ope("rho", &EOS<eos_t, eos_var_t::DENSITY>::action, &p);
    syst.add_ope("dHdlnrho", &EOS<eos_t, eos_var_t::DHDRHO>::action, &p);
    syst.declare_user_opes_target_only();
}

inline void add_thermodynamic_defs(System_of_eqs& syst) {
    syst.add_def("h = exp(H)");
    syst.add_def("rho = rho(h)");
    syst.add_def("eps = eps(h)");
    syst.add_def("press = press(h)");
    syst.add_def("dHdlnrho = dHdlnrho(h)");
    syst.add_def("delta = h - eps - 1.");
}

inline void add_conformal_combinations_and_adm_integrands(System_of_eqs& syst, int ndom) {
    syst.add_def("NP = P*N");
    syst.add_def("Ntilde = N / P^6");
    syst.add_def(ndom - 1, "intMadm = - einf^i * D_i P / 4piG * 2");
    syst.add_def(ndom - 1, "intMk = einf^i * D_i N / 4piG");
    syst.add_def(ndom - 1, "intMadmalt = -dr(P) * 2 / 4piG");
}

// Register the 4*pi*G coupling constant "4piG" used in the XCTS constraint
// sources. In geometric units (G=1) this is the fixed physical constant 4*pi
// — not a tunable parameter — so it is hardcoded rather than read from config.
inline void add_four_pi_g(System_of_eqs& syst) {
    syst.add_cst("4piG", 4.0 * M_PI);
}

// Register the flat conformal metric under the DSL name "f". One-liner shared
// verbatim by every GR matter caller (the member `fmet` is passed by reference).
inline void add_flat_metric(Metric_flat& fmet, System_of_eqs& syst) {
    fmet.set_system(syst, "f");
}

// Register the two fundamental scalar unknowns shared by every GR matter app:
// the conformal factor P and the lapse N. The binaries additionally register the
// shift vector "bet" *after* this call (the var-ordering — and therefore the
// Jacobian column ordering — is preserved: P, N are always emitted first, in
// this order, in all three callers).
inline void add_conformal_lapse_vars(System_of_eqs& syst, Scalar& conf, Scalar& lapse) {
    syst.add_var("P", conf);
    syst.add_var("N", lapse);
}

// Register the global-rotation field "mg" together with the BCO1 surface normal
// "sm" and the outer (infinity) surface normal "einf". These three add_cst calls
// are byte-identical across all GR matter callers. They are *contiguous* in the
// single-star NS body; in the binaries "mg" is bundled with the per-object
// rotation vectors and "sm"/"einf" are split by the BCO2 normal "sp", so wiring
// the binaries to this helper performs a COO-safe reorder (constants carry no
// Jacobian rows/columns; verified by an MD5-identical iter-0 residual gate).
// Templated on the coordinate-vector array type so the private SolverBase
// member typedef is matched without naming it here.
template <class cfary_t>
inline void add_global_rot_and_surface_coords(System_of_eqs& syst, const cfary_t& coord_vectors) {
    syst.add_cst("mg", *coord_vectors[to_int(coord_vector::GLOBAL_ROT)]);
    syst.add_cst("sm", *coord_vectors[to_int(coord_vector::S_BCO1)]);
    syst.add_cst("einf", *coord_vectors[to_int(coord_vector::S_INF)]);
}

// Binary-only (BNS + BHNS) conformal extrinsic curvature A^ij together with the
// three ADM linear-momentum surface integrands at infinity. The DSL strings are
// identical across the two binaries modulo parser-insignificant whitespace (the
// previous BHNS spelling padded "A^ij" and wrote "3.*"; canonicalised here to
// the BNS spacing — the System_of_eqs tokeniser treats "3. *" and "3.*"
// identically). The single-star NS never emits A^ij in syst_init (it builds the
// curvature per-stage), so this stays a 2-way binary-shared helper.
inline void add_extrinsic_curvature_and_adm_momenta(System_of_eqs& syst, int ndom) {
    syst.add_def("A^ij = (D^i bet^j + D^j bet^i - 2. / 3. * D_k bet^k * f^ij) / 2. / Ntilde");
    syst.add_def(ndom - 1, "intPx = A_i^j * ex_j * einf^i");
    syst.add_def(ndom - 1, "intPy = A_i^j * ey_j * einf^i");
    syst.add_def(ndom - 1, "intPz = A_i^j * ez_j * einf^i");
}

// Binary-only (BNS + BHNS) quasi-local spin surface integrands, evaluated just
// outside each compact object. The DSL strings are byte-identical between the
// two binaries; only the domain indices differ (BNS: ADAPTED1+1 / ADAPTED2+1;
// BHNS: ADAPTEDNS+1 / ADAPTEDBH+1), so the caller passes the two domain indices.
inline void add_quasilocal_spin_integrands(System_of_eqs& syst, int dom_spin1, int dom_spin2) {
    syst.add_def(dom_spin1, "intS1 = A_ij * mm^i * sm^j / 2. / 4piG");
    syst.add_def(dom_spin2, "intS2 = A_ij * mp^i * sp^j / 2. / 4piG");
}

// Refresh the central enthalpy / density entries of the configuration from the
// freshly solved central log-enthalpy. Byte-identical in the single-star and
// binary solvers; templated on the EOS so the density inversion uses the
// same branch the solver was instantiated with.
template <class eos_t, typename config_t>
inline void update_config_quantities(config_t& bconfig, const double& loghc)
{
    bconfig.set(HC) = std::exp(loghc);
    bconfig.set(NC) = EOS<eos_t, eos_var_t::DENSITY>::get(bconfig(HC));
}

} // namespace xcts
} // namespace Kadath
