# BH 3D XCTS Solver Notes

## Irreducible Mass (`MIRR`) and Horizon Constraints

In the solver, `MIRR` (often denoted as `M` in the system of equations) is declared as a **constant**:

```cpp
syst.add_cst("M", bconfig(MIRR));
```

However, users often observe that the "measured" irreducible mass changes during the iteration or that `MIRR` seems to update. Here is the explanation:

### 1. `M` is a Target, Not a Variable
`M` is fixed in the system of equations. It does not have a corresponding unknown in the Newton-Raphson vector. It serves as the target value for the horizon area constraint:

```cpp
space.add_eq_int_bh(syst, "integ(intMsq) - M * M = 0");
```

### 2. What Updates?
To satisfy this constraint, the solver adjusts:
*   **Domain Mapping (Horizon Shape)**: The physical coordinate radius of the horizon is defined as a function $R(\theta, \varphi)$ expanded in spherical harmonics. The coefficients of this expansion are treated as unknowns in the system.
    *   During each Newton step, the solver computes corrections for these coefficients.
    *   The `Domain_shell_inner_adapted` class updates its internal `inner_radius` variable and recalculates the coordinate mapping.
    *   This effectively "moves" the physical location of the grid points at the inner boundary to satisfy the area constraint and the apparent horizon boundary conditions.
*   **Metric Fields**: The conformal factor and other fields update, changing the physical area.

The solver iterates until the physical horizon area matches the target `M`.

### 3. Consistency Updates
In some stages (e.g., `von_Neumann_stage`), `bconfig(MIRR)` itself is updated *before* the system is initialized to ensure consistency with the user-specified Christodoulou mass (`MCH`) and spin (`CHI`):

```cpp
bconfig.set(MIRR) = bco_utils::mirr_from_mch(bconfig(CHI), bconfig(MCH));
```

### 4. Diagnostics
The output printed during solving (e.g., `Mirr: 1.400...`) represents the **current measured value** of the irreducible mass from the fields at that iteration. It converges to the target `M` as the residual goes to zero.