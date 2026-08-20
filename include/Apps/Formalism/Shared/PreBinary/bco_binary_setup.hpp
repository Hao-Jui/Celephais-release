#pragma once

#include "For_Kadath/Config/config_binary.hpp"
#include "For_Kadath/Config/config_bco.hpp"
namespace Kadath {

// Run the isolated NS solver for component `bco` in binary config `bconfig`.
// `ns_seed_solve(nsconfig, output_path, bconfig, bco)` performs the formalism-
// specific seed solve. Returns the path of the converged NS file.
template <typename config_t, typename NsSeedSolve>
std::string solve_NS_from_binary(config_t& bconfig, NODES bco, NsSeedSolve&& ns_seed_solve);

// Run the isolated BH solver for component `bco` in binary config `bconfig`.
// `bh_seed_solve(bhconfig, output_path, bconfig, bco)` performs the formalism-
// specific seed solve. Returns the path of the converged BH file.
template <typename config_t, typename BhSeedSolve>
std::string solve_BH_from_binary(config_t& bconfig, NODES bco, BhSeedSolve&& bh_seed_solve);

// One component of a binary, for the separation sanity check.
//   mass       : characteristic mass (MADM for a NS, MCH for a BH)
//   radius     : surface / excision coordinate radius (RMID); <= 0 if unknown
//   deformable : true for a NS (Roche / mass-shedding applies), false for a BH
struct BinaryComponent {
    double mass;
    double radius;
    bool   deformable;
};

// Assert the separation is physically reasonable. Two independent conditions:
//   * HARD (throws): dist <= overlap_safety * (R1 + R2) -- the adapted domains
//     would interpenetrate (numerically degenerate). EOS-aware via the per-star
//     radius. Falls back to 2.5 * (M1 + M2) when a radius is unset (<= 0).
//   * SOFT (warns only): dist < the Roche / mass-shedding onset of a deformable
//     star (Eggleton 1983). Physical but near the sequence endpoint, so it is a
//     warning, never a throw.
inline void check_dist(double dist, BinaryComponent c1, BinaryComponent c2,
                       double overlap_safety = 1.3);

// Legacy mass-only entry point: radii unknown, so the mass-scale fallback is
// used and the Roche warning is skipped. Throws if dist <= garbage_factor*(M1+M2).
inline void check_dist(double dist, double M1, double M2, double garbage_factor = 2.5);


} // namespace Kadath
#include "bco_binary_setup_imp.ipp"
