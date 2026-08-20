# Binary hp-refinement (adaptive mesh refinement)



- **h-refinement** inserts a shell into a flagged region (the default first move
  — more small domains at the base resolution keep the per-domain dense blocks
  cheap to assemble and factor),
- **p-refinement** adds collocation points to a flagged domain+direction once
  the region's shell budget is spent.

## Files



The gate is invoked from `run_bns_amr_gate`
(`include/Apps/Workflow/bns_app_workflow.hpp`): up to `max_cycles` rounds of
*evaluate → regrid → re-solve*, stopping when no tail exceeds the thresholds.

## How a cycle decides



Regions are independent: only a flagged region below `max_shells` gets a shell;
a flagged region already at the cap waits for the p-fallback.

## What keeps its angular resolution locked

`nr` varies freely per domain — radial-seam conditions are sized by angular
modes only, so differing radial counts across a seam stay square. Angular
(`n_theta`, `n_phi`) refinement is allowed per domain too, now that
non-conforming seams are coupled with coefficient-space (`import`) matching — but
two structures stay **locked** because a differing angular count there would make
Kadath's `Eq_matching` tau system ill-defined:

- **Each star/compact-object core** — the triple `{nucleus, adapted-outer,
  adapted-inner}` is forced to one shared `(n_theta, n_phi)` after refinement
  (`lock_star_core`). The two adapted domains share the deformable surface (its
  angular representation must be single-valued), and the nucleus's
  parity-restricted basis needs a square import seam against its neighbour.
- **The five bispheric domains** share one `Dim_array` and refine as a **locked
  block in PHI (axis 2) only**, when `refine_bispheric` is on (default). Phi is
  azimuthal — the same physical coordinate in every bipolar variant — so a
  uniform `+2` keeps the conforming internal seams square. Axes 0/1 (the bipolar
  chi/eta, whose slot maps to a different physical direction per variant) are not
  refinable: bumping them unbalances those seams. The block has no h-fallback (no
  shell fits inside the chimera), so phi-p is its only refinement path.

All BNS and BHNS variants allow per-domain angular p moves subject to those
locks. The actionability mask remains in the core for layouts that still have
non-refinable axes: such tails are measured in the demand diagnostics but are
recorded as `blocked_by_structure` rather than regridded. If the dominant
post-h tail is such a blocked target, the gate stops instead of taking a
lower-priority p move that would not address the dominant error.

## Why r_bisph is separation-blended

Binary setup chooses the bispheric matching radius from the separation instead
of growing it from the shell count. `bco_utils::make_binary_NS_bounds` therefore
uses

```
r_bisph = RMID + rbisph_fill · (dist/2 − RMID)        (default fill = 1/3)
```

With no extra shells, `r_bisph == ROUT`. With extra shells, the vector layout is
`[RIN, RMID, ROUT, shell1, ..., r_bisph]`, so the fixed band
`(RMID, r_bisph]` is subdivided and the final matching radius stays put. Because
`r_bisph` is recomputed from `RMID`/`DIST`/`fill` (never from the mutated `ROUT`),
adding a shell leaves it invariant. An absent `rbisph_fill` key means the default
blend; the explicit `rbisph_fill = 0` uses the config `ROUT` as the fixed outer
target (and conflicts with `mode = "hp"`, which the dispatch rejects).

## Configuration

```toml
[binary]
rbisph_fill = 0.33333333333333331   # optional; absent = 1/3, 0 = legacy bounds

[adaptive_mesh_refinement]
enabled = true        # default false
max_cycles = 2        # rounds of evaluate -> regrid -> re-solve (default 2)
mode = "hp"           # "hp" (default): gated shell layout moves, then bounded p; "p": points only
max_shells = 3        # hp only: per-region shell cap; h candidates still need post-h gain (default 3)
max_nr = 15           # per-direction +2 ceiling of the p-move, BOTH modes; over-cap domains drop out (default 15)
refine_bispheric = true   # phi-only locked-block refinement of the 5 bispheric domains (default true)
h_gain_min = 2.0      # after h, require moved-region R_new <= R_old / h_gain_min
h_stall_cycles = 1    # legacy/inert compatibility key
h_saturation_linf = 0.95      # legacy/inert compatibility key
h_angular_dominance = 1.0     # skip h for this cycle when A/R is at least this value
p_dof_growth_limit = 2.0      # accept p candidates while projected DOF growth <= limit; 0 disables
p_fallback_dof_growth_limit = 1.1   # tighter staged p budget after an h solve
p_fallback_max_candidates = 1       # post-h p candidates per AMR cycle; 0 disables
tail_width = 2        # highest modes counted as "tail"
l2_tail_threshold = 1e-8
linf_tail_threshold = 1e-7
norm_floor = 0.0      # domains with total norm <= floor report zero ratios
```

The gate runs only after the solver chain completes — a job that dies or times
out mid-chain never reaches it. A current binary always prints either the
refine banner or `spectral tails below thresholds; stopping`.

## Banner

Ordinary h and p banners:

```
AMR cycle 0 per-direction check: flagged domains radial=2 theta=0 phi=0 | R=530 A=0 | max tail field=logh domain=1 NS1 adapted-outer axis=0 l2=4.2e-05 linf=5.3e-05 | thresholds l2=1e-08 linf=1e-07
AMR h-refinement cycle 0: radial tail above threshold -> adding shells: NS1 NSHELLS->1; regrid uniform at base nr=9 (angular axes re-evaluated next cycle)

AMR h-layout probe cycle 1: moved-region radial demand gain=1.4 [compact1=1.4] required>=2 | R=380 A=25 A/R=0.066 | radial flags 2->2
AMR h-refinement cycle 1: h gate closed (moved-region radial demand gain=1.4 [compact1=1.4] (<2)); considering bounded p-refinement
AMR p-refinement cycle 1 (hp fallback): max tail field=logh domain=1 NS1 adapted-outer l2=4.2e-05 linf=5.3e-05 | p candidates accepted=1 skipped_by_dof_trust=0 skipped_by_batch_cap=0 skipped_by_structure=0 dof_growth=1.08
  p staging: post_h=true dof_limit=1.1 candidate_limit=1
  changed domains (nr,nt,np) before -> after:
    domain  1  NS1 adapted-outer       (9,9,8) -> (11,9,8)
    domain  2  NS1 adapted-inner       (9,9,8) -> (11,9,8)
  -> max radial=11
```

Structural stop banner:

```
AMR cycle 1 per-direction check: flagged domains radial=1 theta=0 phi=0 | blocked_by_structure radial=0 theta=1 phi=0 | R=10 A=200 | max tail field=phi domain=3 NS shell 0 axis=1 l2=1e-03 linf=1 | thresholds l2=1e-08 linf=1e-07
AMR cycle 1: dominant tail is structurally unsupported by per-domain p-refinement in this space (field=phi domain=3 NS shell 0 axis=1); stopping
```

## Gates / tests

- `tests/unit/test_bns_per_domain_p_refinement.cpp` — per-domain `Space_bin_ns`
  constructor patterns, save/load roundtrip (the space file format is already
  per-domain), mixed-resolution Laplace square-system gate (eqs == unknowns,
  roundoff-zero residual across every seam type, including an NSHELLS=1
  layout), decision-layer and p/hp dispatch tests.
- `tests/unit/test_ns_bounds_rbisph.cpp` — blended-r_bisph formula, fixed-band
  subdivision layout, legacy opt-out parity, dispatch semantics.
- Production probes (d=35, χ=0.3 BNS): p-cycle refined 5/12 domains 9→11 and
  reconverged quadratically to 1.6e-11; forced hp probe (`max_nr = 9`) ran two
  h-cycles (shells 0→1→2 per star + exterior) at fixed `r_bisph`, reconverging
  to 5.7e-9 — shell counts that are geometrically impossible at d=35 under the
  legacy bounds.

## Known limits

