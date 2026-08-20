# Margherita Table EOS Flow

`Cold_Table` provides a cold tabulated EOS for Margherita TOV seeding and EOS
helper calls. The table branch is intentionally small: it reads one EOS table,
builds the runtime interpolation objects, and serves forward and inverse cold
EOS queries.

## Files

- `cold_table.hh`: public `Cold_Table_t` state and EOS helper declarations.
- `cold_table_implementation.hh`: forward and inverse EOS helper logic.
- `setup_cold_table.cc`: table loading and forward interpolation setup.
- `lorene_io.hh`: Lorene-format table reader.
- `interpolation/`: local interpolation primitives used by `Cold_Table`.

## Setup Path

The runtime setup path is:

1. `setup_Cold_Table()` receives the Lorene table path, interpolation point
   count, and optional `h_cut`.
2. `Lorene_Table()` reads the original Lorene table and applies `h_cut` as a
   row filter. For each row it computes
   `h = 1 + eps + P / rho`; only rows with `h >= h_cut` are retained.
   The default `h_cut = 0.0` keeps ordinary positive-enthalpy rows.
3. The retained table columns are converted to log-space:
   `log(rho)`, `log(eps)`, and `log(P)`.
4. A monotone (Hyman-filtered) cubic spline is built from the retained
   log-space table: `log(rho) -> log(eps), log(P)`. A plain natural cubic spline
   overshoots across near-isobaric stretches (first-order phase transitions),
   injecting nonpositive `dP/drho` and non-monotone `log(P)` into the resampled
   table; the Hyman filter keeps the exact spline slopes where the data is
   monotone (so smooth tables retain full spline accuracy) and clamps only the
   nodes that would overshoot. See `Interpolation/monotone_spline.hh`.
5. That interpolant is sampled onto a uniform `log(rho)` grid with
   `cold_lintp_points` samples.
6. `Cold_Table::lintp` stores this sampled table as a linear interpolator.

After setup, ordinary forward EOS calls use `Cold_Table::lintp`; they do not
query the original Lorene data directly.

Because `h_cut` is applied before interpolation setup, it also defines the
runtime lower table edge: the first retained row becomes `rhomin`, `press_min`,
and the low-end enthalpy used by inverse table construction. Raising `h_cut`
therefore trims low-density/low-enthalpy data from both forward and inverse EOS
queries.

## Forward Queries

Forward calls are evaluated from the resampled linear table:

- `press_cold_eps_cold__rho(rho)`:
  `log(rho) -> log(P), log(eps)`, then exponentiates back to physical units.
- `dpress_cold_drho__rho(rho)`:
  differentiates the linear `log(P)` representation and converts
  `d log(P) / d log(rho)` to `dP / drho`.
- `get_extra_quantities(rho)`:
  returns interpolated table quantities from the same forward table.

## Solver-Facing EOS Variables

`System_of_eqs` does not call the `Cold_Table` primitives directly. It registers
four operators through the `EOS<eos_var_t>` wrapper (`Hydro/EOS.hh`), each a
function of the specific enthalpy `h`. The solver works in `H = log(h)` and
passes `h = exp(H)`:

| operator      | `eos_var_t` | returns |
| ------------- | ----------- | ------- |
| `rho(h)`      | `DENSITY`   | rest-mass density `rho` |
| `eps(h)`      | `EPSILON`   | specific internal energy `eps` |
| `press(h)`    | `PRESSURE`  | pressure `P` |
| `dHdlnrho(h)` | `DHDRHO`    | `(1/h) * dP/drho` = sound speed squared `cs^2` |

Every one of these first maps `h -> rho` with the enthalpy inverse
(`rho__h_cold`), then reads `P`, `eps`, and `dP/drho` from the forward
`log(rho)` table. So the Newton-step EOS evaluations depend on the **enthalpy
inverse** and the **forward table**, and never on the pressure inverse. The
pressure inverse (`rho__press_cold`, `rho_energy_dedp__press_cold`) is used only
by the Margherita TOV seeding, which integrates in pressure.

Note the `DHDRHO` naming: it returns `cs^2 = (1/h) dP/drho`, not the
mathematical `dh/drho = (1/rho) dP/drho`; the two differ by `rho/h`. The
Jacobian contribution for `DHDRHO` is hardcoded to zero (it enters the equations
as a coefficient, not a varied field).

## Inverse Queries

The inverse path is cached and built lazily on the first inverse query after
setup:

1. `ensure_inverse_tables()` checks whether the cached inverse tables match the
   current `Cold_Table::lintp` and EOS bounds.
2. `rho__press_cold(P)` builds `log(P) -> log(rho)` from the existing forward
   linear table, keeping only the increasing branch
   (`make_monotone_inverse_linear_interp`); see below.
3. `rho__h_cold(h)` samples the enthalpy expression
   `h = 1 + eps + P / rho` from the existing forward linear table on a denser
   `log(rho)` grid, then builds `h -> log(rho)`.
4. Later inverse calls use these cached inverse linear interpolators directly.

The inverse tables do not spline or re-read the original EOS data. They invert
the same forward runtime representation used by the rest of `Cold_Table`.

Both inverses keep only the increasing branch of the resampled forward table
(`make_monotone_inverse_linear_interp`): a sample becomes an inverse node only if
its `log(P)` (resp. `h`) is strictly greater than the last kept value; equal or
decreasing samples are skipped. For a smooth, strictly increasing table this
keeps every sample and reproduces the old Brent solve to roundoff.

Filtering is still required at a genuine first-order phase transition. The
forward interpolant is the monotone (Hyman) cubic spline, so it no longer
*overshoots*: `dP/drho` stays nonnegative and `log(P)` never spuriously
decreases. But across a true Maxwell plateau `P` is flat over a finite density
range, so consecutive resampled `log(P)` values become bit-identical once the
per-step change falls below the ULP of `log(P)` (~1e-14 near `log(P) ~ 80`).
That flat, equal-valued segment is still not strictly increasing, so a strict
inverse would abort (the historical `log pressure must be strictly increasing`
error). The monotone branch instead keeps the lowest-density edge of the plateau
and proceeds — it selects the hadronic-onset branch at the transition pressure.

Historically, with a plain natural cubic spline forward table there was a second
failure mode: spline overshoot producing genuinely *decreasing* `log(P)` and
negative `dP/drho`. Switching the forward interpolant to the monotone spline
eliminated that mode at the source; only the flat-plateau ULP collapse remains,
and the inverse filter handles it.

## Legacy Brent Inversion

Before the cached inverse tables, inverse calls solved scalar root problems at
runtime with Brent's method:

- `rho__press_cold(P)` searched in `log(rho)` and evaluated
  `log(P_target) - log(P(log(rho)))`.
- `rho__h_cold(h)` searched in `log(rho)` and evaluated
  `h_target - (1 + eps(log(rho)) + P(log(rho)) / rho)`.

Both residuals used `Cold_Table::lintp`, not the original Lorene data and not
the original cubic spline. In other words, Brent inverted the same resampled
linear runtime table used by forward calls.

The Brent path had two costs:

1. Every inverse call required many forward interpolations plus `exp` and `log`
   evaluations.
2. The enthalpy solve could converge to a low-density extrapolated root when
   the sampled enthalpy curve was not strictly monotone near the table edge.

The cached inverse-table path makes each inverse call a single interpolation.
For smooth tables it reproduces the Brent pressure result to roundoff; for
non-monotone regions both inverses keep the monotone in-table branch instead of
solving for an extrapolated or multivalued root.

## Robustness Rules

An EOS table provided to `Cold_Table` must satisfy these properties:

- `rho` values are finite, positive, and strictly increasing.
- `P` values are finite, positive, and strictly increasing with `rho`.
- `eps` values are finite. If the first `eps` value is negative,
  `setup_Cold_Table()` applies a uniform positive shift before taking logs.
- The table contains at least two usable rows after any read-time filtering.
- The physically relevant enthalpy branch,
  `h = 1 + eps + P / rho`, is increasing over the density range used by TOV
  seeding and inverse EOS helper calls.

The interpolation setup then enforces the runtime requirements:

- all log-space interpolation coordinates must be finite;
- forward interpolation coordinates (`log(rho)`) must be strictly increasing;
- the forward interpolant is monotone (Hyman cubic spline), so `dP/drho` stays
  nonnegative and the resampled `log(P)` never overshoots;
- both inverses still keep the increasing in-table branch and skip the bit-equal
  samples left by ULP collapse at a genuine flat (Maxwell) plateau, matching the
  EOS helper's bounded table semantics.
