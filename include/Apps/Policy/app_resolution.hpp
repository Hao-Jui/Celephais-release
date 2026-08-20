#pragma once

#include "For_Kadath/Array/dim_array.hpp"
#include "For_Kadath/Array/exceptions.hpp"

namespace Kadath {

// =============================================================================
// App-side resolution policy -- the single "res = N" knob.
//
// Each app-side solve exposes one resolution knob, N, for the space it is
// building (TOML BCO_RES for isolated compact objects, BIN_RES for binaries).
// This header turns that local N into the per-direction (r, theta, phi) counts
// for the SPHERIC / adapted single-object spaces (isolated NS / BH), whose
// constructors take an explicit Dim_array (Phillipe "incline" / Mode-1 ctors).
// The GR seeds and the single-object regrid call these and pass the result to
// the Space ctor.
//
// In a BNS TOML, [binary].res drives only the binary continuation ladder and
// binary grid. [binary.ns1].res / [binary.ns2].res remain the final resolution
// targets for their isolated-star seed solves; those seed solves start from
// kDefaultInitialResolution unless the config overrides initial_resolution.
//
// The values MUST stay identical to the inline-set resolution in the src/Space
// builders (Phillipe convention), so a star seeded standalone has the same grid
// as the star side of a binary:
//
//   r = N,  theta = N,  phi = N-1   (both symmetries; Phillipe uses theta = N
//   for the full-theta nosym polar range too -- there is no 2N-5).
//
// Binary spaces (Bin_ns, Bhns, Bispheric, Bin_bh, ...) take an int nr and set
// BOTH their spheric res and bispheric res_bi INLINE in src/Space per Phillipe;
// those are intentionally not duplicated here.
// =============================================================================

// Default initial resolution for the continuation ladder when a config omits
// initial_resolution. Shared by the isolated-object solvers and the
// binary-component seed solves.
inline constexpr int kDefaultInitialResolution = 13;

inline int sym_theta_res(int n_r)   { return n_r; }
inline int nosym_theta_res(int n_r) { return n_r; }
inline int phi_res(int n_r)         { return n_r - 1; }

// (r, theta, phi) for spheric / adapted single-object domains; (N, N, N-1) for
// both symmetries (two names kept for call-site clarity).
inline Dim_array spheric_res(int n_r)
{
    Dim_array res(3);
    res.set(0) = n_r;
    res.set(1) = sym_theta_res(n_r);
    res.set(2) = phi_res(n_r);
    return res;
}

inline Dim_array spheric_res_nosym(int n_r)
{
    Dim_array res(3);
    res.set(0) = n_r;
    res.set(1) = nosym_theta_res(n_r);
    res.set(2) = phi_res(n_r);
    return res;
}

// =============================================================================
// Resolution validation + continuation ladder.
//
// Single home for the "resolution must be odd" rule and for the init->final
// continuation-ladder bootstrap that every app's resolution-stepping loop
// performs. Centralising these keeps the policy (which N values are legal, how
// the ladder is seeded) in one place instead of duplicated per app.
// =============================================================================

// The radial resolution N must be odd (9, 11, 13, ...). The collocation grid
// derived from N (Phillipe convention r = N, theta = N, phi = N - 1) is only
// well-formed for odd N.
inline void validate_resolution(int n_r)
{
    if (n_r % 2 == 0)
        KADATH_THROW("New Resolution is invalid.  Must be odd (9,11,13,etc)");
}

// Validated [init, final] continuation ladder. `init` is the resolution the
// continuation starts at; `final` is the target. Both must be odd. If the
// target is below init there is no ladder to climb, so the target is raised
// to init (a single-rung ladder) rather than treated as an error.
struct ResolutionLadder
{
    int init;
    int final;
};

inline ResolutionLadder make_resolution_ladder(int init, int final)
{
    if (init > final)
        final = init;
    validate_resolution(init);
    validate_resolution(final);
    return {init, final};
}

} // namespace Kadath
