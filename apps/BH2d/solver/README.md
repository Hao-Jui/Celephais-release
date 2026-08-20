isoBH notes

Space_adapted_bh_polar add-eq helpers (2D axisymmetric) vs Space_adapted_bh (3D)

- add_bc_bh: inner BH boundary. 2D uses HOMOTHETIC_INNER (domain 2); 3D uses domain 2.
- add_bc_inf: outer boundary at infinity. Both use nbr_domains-1 and OUTER_BC.
- add_eq: adds eqs inside domains and matching across OUTER_BC for each domain from
  inner BH domain to sys.get_dom_max(), then adds inside for the compactified domain.
- add_eq_int_inf: integral constraint at infinity. 2D uses Domain_polar_compact,
  3D uses Domain_compact; logic is otherwise identical.
- add_eq_int_horizon (2D) is the analogue of add_eq_int_bh (3D):
  adds a single-part Eq_int on INNER_BC for the inner BH domain.
- add_eq_int_volume: volume integral over the first nz domains plus a subtraction term.
- add_eq_zero_mode_inf: enforces zero mode at infinity; 2D only sets k index if ndim > 2.

Key mapping summary

- Space_adapted_bh_polar::add_eq_int_horizon <-> Space_adapted_bh::add_eq_int_bh
- Domain index for the BH inner boundary is 2 in both; 2D uses HOMOTHETIC_INNER.

## Irreducible Mass (`MIRR`) and Horizon Constraints

In the `isoBH` solver (2D axisymmetric), `MIRR` (denoted as `M` in the system) is declared as a **constant**:

```cpp
syst.add_cst("M", bconfig(MIRR));
```

Similar to the 3D solver, users may observe that the "measured" irreducible mass changes during iteration or that `MIRR` seems to update. The explanation is identical:

### 1. `M` is a Target
`M` is fixed in the system of equations. It serves as the target value for the horizon area constraint:

```cpp
// Area - 16*pi*M^2 = 0
space.add_eq_int_horizon(syst, "integ(intMsq) - M * M = 0");
```

### 2. What Updates?
To satisfy this constraint, the solver adjusts:
*   **Domain Mapping (Horizon Shape)**: The physical coordinate radius of the horizon is defined as a function $R(\theta)$ (in 2D axisymmetric). The spectral coefficients of this function are unknowns in the system.
    *   During each Newton step, the solver updates these coefficients.
    *   The adapted domain recalculates the mapping from the numerical grid to physical coordinates.
    *   This physically moves the grid points of the inner boundary to match the required horizon area and shape.
*   **Metric Fields**: The conformal factor and other fields update, changing the physical area.

The solver iterates until the physical horizon area matches the target `M`.

### 3. Consistency Updates
In `bh_2d_msqi_stages.cpp`, `bconfig(MIRR)` is updated before system initialization to ensure consistency with `MCH` and `CHI`:

```cpp
bconfig.set(MIRR) = bco_utils::mirr_from_mch(bconfig(CHI), bconfig(MCH));
```

### 4. Diagnostics
The output printed during solving (e.g., `Mirr: ...`) represents the **current measured value** of the irreducible mass from the fields at that iteration. It converges to the target `M`.
