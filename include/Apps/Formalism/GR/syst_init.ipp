#pragma once

#include "Hydro/EOS.hh"

// =============================================================================
// Shared GR (pure general-relativity) XCTS per-domain field-equation emission
//
// Factored out of both NS_3D_XCTS (single star: norot_stage.ipp,
// uniform_rot_stage.ipp, binary_boost_stage.ipp) and BNS_XCTS/stages.ipp
// (binary) so the per-domain XCTS constraint operators eqP / eqNP / eqbet are
// maintained in a single place. These operators are byte-identical (modulo
// parser-insignificant whitespace) across every GR caller; the equation
// strings here are the BNS canonical form, which the System_of_eqs DSL parser
// tokenises identically to the previous per-stage spellings.
//
// This free `inline` function takes the System_of_eqs by reference and emits
// only the eqP / eqNP / eqbet *equation* definitions. The matter source terms
// it references by name — Etilde, Stilde, ptilde — are emitted per-caller
// immediately before the call, because their definitions legitimately diverge
// between callers (the matter ENERGY source Stilde uses different leading
// coefficients and the corotating/irrotational velocity blocks differ). All
// names referenced in the DSL strings (P, NP, Ntilde, delta, A_ij, 4piG, N,
// and, in the matter branch, Etilde/Stilde/ptilde) must already be registered
// on `syst` by the calling solver before this helper runs.
//
//   has_shift : true  -> full XCTS with shift (A_ij / Ntilde / eqbet present)
//               false -> static no-shift form (A=0; no curvature, no eqbet)
//   has_matter: true  -> in-star, matter-sourced operators (multiplied by
//                        delta and carrying the Etilde/Stilde/ptilde sources)
//               false -> vacuum operators outside the stellar domains
// =============================================================================

namespace Kadath {
namespace gr_xcts {

inline void add_xcts_field_equations(System_of_eqs& syst, int d, bool has_matter, bool has_shift)
{
    if (has_shift) {
        if (has_matter) {
            syst.add_def(d, "eqP    = delta * D^i D_i P + A_ij * A^ij / P^7 / 8 * delta + 4piG / 2. * P^5 * Etilde");
            syst.add_def(d, "eqNP   = delta * D^i D_i NP - 7. / 8. * NP / P^8 * delta * A_ij * A^ij "
                            "- 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
            syst.add_def(d, "eqbet^i = delta * D_j D^j bet^i + delta * D^i D_j bet^j / 3. "
                            "- 2. * delta * A^ij * D_j Ntilde - 4. * 4piG * N * P^4 * ptilde^i");
        } else {
            syst.add_def(d, "eqP     = D^i D_i P + A_ij * A^ij / P^7 / 8");
            syst.add_def(d, "eqNP    = D^i D_i NP - 7. / 8. * NP / P^8 * A_ij * A^ij");
            syst.add_def(d, "eqbet^i = D_j D^j bet^i + D^i D_j bet^j / 3. - 2. * A^ij * D_j Ntilde");
        }
    } else {
        if (has_matter) {
            syst.add_def(d, "eqP    = delta * D^i D_i P + 4piG / 2. * P^5 * Etilde");
            syst.add_def(d, "eqNP   = delta * D^i D_i NP - 4piG / 2. * N * P^5 * (Etilde + 2. * Stilde)");
        } else {
            syst.add_def(d, "eqP = D^i D_i P");
            syst.add_def(d, "eqNP = D^i D_i NP");
        }
    }
}

} // namespace gr_xcts
} // namespace Kadath
