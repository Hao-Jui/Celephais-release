# Apps/Formalism/Shared



Layering (outermost → innermost):



## ns ↔ bns symmetric pairs

The two genuinely parallel single-star / binary concerns mirror by name:

| Concern | NS (single star) | BNS (binary) |
|---|---|---|
| Single-solve **stage dispatch** + runtime diagnostics | `ns_driver_common.hpp` | `binary_driver_common.hpp` |
| Resolution **regrid** core (space traits + templated field transfer) | `Regrid/ns_regrid.hpp` | `Regrid/bns_regrid.hpp` |

`binary_driver_common.hpp` provides the shared BNS/BHNS
`QUASI_EQUIL -> FORCE_BALANCE -> ECC_RED` dispatch plus the BNS diagnostics;
both BNS and BHNS solvers include it directly. `*_regrid.hpp` provides a `*_space_traits` map and a
templated regrid driven by a per-formalism field-set callback.



## `Stages/` subfolder — shared per-stage solve bodies

The per-stage equation-setup bodies that the sym/nosym solver pairs share live in
`Stages/`:



Each `*_stages_common.ipp` holds the stage body as a free function templated on the
solver type and befriended by both solver classes; the per-class `stages.ipp` are
thin wrappers binding `*this` plus callbacks for the sym/nosym divergences (the
in-star eqbet gauge forcing and, for BNS, the ADM z-momentum closure). This stays
innermost (Shared core); the per-formalism solver headers include these bodies, not
the other way round. Stage **dispatch** (`{ns_driver,binary_driver}_common.hpp` —
which stage runs in what order) is a thinner layer and stays at the directory root.

## `PreBinary/` subfolder — binary initial-guess construction

The seed-construction step that runs *before* the binary solve stages — building the
binary's initial guess from converged isolated-object seeds — lives in `PreBinary/`:

| File | Role |
|---|---|
| `PreBinary/bco_binary_setup.{hpp,ipp}` | Solve an isolated NS/BH seed from a binary config slot (callback-driven, formalism-agnostic) |
| `PreBinary/bns_boosted_setup.hpp` | Superimpose two isolated-NS seeds (exponential-decay weight) into the binary guess |
| `PreBinary/bns_separation_seed.hpp` | Separation seed (superposition + regrid to the binary grid) |

This is *setup*, not a solve stage (no `System_of_eqs` / Newton loop), and is distinct
from the top-level `Apps/Seed/` module (single-object TOV/seed utilities) — hence
`PreBinary/` rather than a `Seed/` that would collide. `ns_3d_xcts_binary_boost_stage.hpp`
is the binary-boost *solve stage* and correctly lives under `Stages/`, not here.

## Scope singletons (no ns/bns twin — by design)

These are **not** ns/bns pairs; each covers a scope where only one side exists.

- **`omega_mode.hpp`** — binary-scope. The `OmegaMode` enum (`Fixed`/`Free`)
  selecting a binary stage's orbital-frequency treatment: `Fixed` holds `ome`
  constant and drops the quasi-equilibrium constraint (warm-up pass); `Free`
  solves for `ome` and adds the constraint (physical pass). The warm-up decision
  is the caller's; the saved `FIXED_GOMEGA` config flag is never mutated.
  No NS analog — an isolated star has no orbital Ω.

- **`PreBinary/bco_binary_setup.{hpp,ipp}`** — binary-scope. Solve an isolated NS/BH
  **seed** from a binary config slot, then write results back. Theory-agnostic: the
  formalism driver (and `BIN_BOOST` enablement) is supplied by a `*_seed_solve`
  callback (typically `NsPolicyCommon::solve_ns_xcts<…NsDriverTraits>`), so this
  header pulls **no** formalism solver. (See the `PreBinary/` section below for
  the rest of the binary initial-guess construction.)

- **`Stages/bns_stage_diagnostics.hpp`** — BNS-lane only. `env_flag/env_double/env_int`
  config helpers + Jacobian/equation probes used by the BNS solver stages and
  policies. (Used exclusively in the BNS lane; hence the `bns_` prefix. Grouped
  with the stage bodies under `Stages/`.)

- **`xcts_syst_init.ipp`** — cross-cutting (NS **and** BNS). The `xcts::` helpers
  that build the common pieces of a `System_of_eqs`: flat metric, EOS operators,
  conformal/lapse vars, thermodynamic defs, ADM integrands, quasi-local spin, etc.

## Why it is not "more symmetric"

Only stage-dispatch and regrid are genuine single-star/binary counterparts, and
those already mirror. The singletons above are asymmetric because the underlying
physics/workflow is: a binary has an orbital Ω and a seed-from-slot step that an
isolated star lacks; an isolated star has a chi-continuation ladder whose fused
state machine has no binary equivalent. Forcing matching `ns_`/`bns_` filenames
onto those would name files after a symmetry that does not exist.
