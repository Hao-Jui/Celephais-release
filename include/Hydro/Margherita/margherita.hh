/*
 * Modifications (Celephais):
 *   2026-06-16  Modified for the Celephais tree; see
 *               PATCHES-KADATH-UPSTREAM.md and LICENSE_SOURCE_AUDIT.tsv.
 */

#pragma once


namespace Margherita_constants {

constexpr double LENGTHGF = 6.77269222552442e-06;
constexpr double TIMEGF = 2.03040204956746e05;
constexpr double RHOGF = 1.61887093132742e-18;
constexpr double PRESSGF = 1.80123683248503e-39;
constexpr double EPSGF = 1.11265005605362e-21;
constexpr double INVRHOGF = 6.17714470405638e17;
constexpr double INVEPSGF = 8.98755178736818e20;
constexpr double INVPRESSGF = 5.55174079257738e38;
// for Temperature in polytropic EOS
constexpr double c2_cgs = 8.9875517873681764e+20;
constexpr double mnuc_MeV = 939.5; //931.494061;       // Atomic mass unit
constexpr double mnuc_cgs = 1.660539040e-24; //1.6748109230081286e-24 (939.5 MeV/c^2);  // Atomic mass unit cgs
constexpr double mnuc_Msun = mnuc_cgs * PRESSGF*(LENGTHGF*LENGTHGF*LENGTHGF)*c2_cgs;
constexpr double MeV_to_erg = 1.60217733e-6;
constexpr double cm3_to_fm3 = 1.0e39;
constexpr double avogadro = 6.0221367e23;
constexpr double m_neutron_MeV = 939.565379;


constexpr double hbarc_MeV_fm = 197.326978812;
}

namespace Margherita_helpers {
// Helper functions

constexpr int sgn(double A) { return (0 < A) - (A < 0); }

constexpr double max(double A, double B) {
  return (A > B) ? A : B;
}

template <typename... Targs>
constexpr double max(double A, double B, Targs... Fargs) {
  return max(A, max(B, Fargs...));
}

constexpr double min(double A, double B) {
  return (A > B) ? B : A;
}

template <typename... Targs>
constexpr double min(double A, double B, Targs... Fargs) {
  return min(A, min(B, Fargs...));
}
}  // namespace Margherita_helpers
