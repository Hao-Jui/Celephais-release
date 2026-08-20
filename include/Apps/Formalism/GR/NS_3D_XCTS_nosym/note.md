# NS_nosym math-related bugs fixed

This note records only the math-related issues fixed while checking isolated
`NS_nosym` orientation invariance for `chi=0.3` at `deg=0,30,90`.

## Bugs fixed

1. No-sym theta Galerkin coefficient logic was inconsistent across domains.

   The full-theta no-sym coefficient arrays interleave COS/SIN regularity
   subspaces. Several count, affect, export, one-coefficient, and boundary
   paths used local approximations of which theta coefficients are physical
   and which are Galerkin anchors. Those approximations disagreed for modes
   such as constrained COS `j=1` and SIN `j=2`, which could make imported
   matching rows and domain unknown/condition counts inconsistent.

   The regularity logic is now centralized in
   `include/For_Kadath/Domain/spheric_nosym_regularization.hpp`, including the
   true coefficient predicate, anchor index, basis anchor weight, and export
   anchor weight.

2. Adapted no-sym matching import counted stale constrained coefficients.

   `eq_matching_import.cpp` had its own adapted-no-sym coefficient selection
   rules. They counted coefficients that the no-sym export path skipped, so
   the imported matching block could have a stale row count. It now uses the
   same no-sym regularization helper as the domain affect/export/count code.

3. Selected Cartesian tensor component `(2,1)` was counted as `(1,1)`.

   Several no-sym and adapted-no-sym boundary count dispatchers repeated the
   `(1,1)` predicate where they meant `(2,1)`. This could undercount or
   miscount selected tensor-component boundary conditions. The predicate now
   dispatches `(2,1)` to `tt(2,1)` in the affected no-sym boundary count
   files.

4. Adapted no-sym areal-radius surface determinant used wrong derivatives and
   sine weights.

   The area element for a deformed surface `r(theta,phi)` should use
   `dr/dtheta = der_var(2)`, `dr/dphi = der_var(3)`, and
   `det(q) = R^2 * (R^2 sin^2(theta) + sin^2(theta) R_theta^2 + R_phi^2)`.
   The old code used the wrong derivative indices and placed the `sin^2`
   factor on the wrong derivative term. This made the areal radius
   orientation-dependent.

5. The x-axis rotation vector used a uniform basis assignment.

   `rot_x = (0, -z, y)` has an antisymmetric z component in the no-sym theta
   basis. Calling `rot_x.std_base()` assigned the same base to every component.
   The components now use `std_base`, `std_anti_base`, and `std_base`
   respectively.

6. Outer adapted no-sym Cartesian coordinate fields lacked component bases.

   The outer adapted no-sym coordinate builders filled the Cartesian component
   values but did not assign the component bases. They now match the inner
   adapted no-sym convention: x/y use `std_base`, z uses `std_anti_base`.

## Verification

The registered orientation-invariance regression passed with `np=4` at
`res=9`:

```text
deg   final_error      Madm          Mb            Jadm          Chi           ArealR       CentralOmega
0     2.45609e-07      1.35000000    1.48972936    0.54675003    0.300000016   7.88613552   0.0170236544
30    2.97693e-07      1.35000000    1.48972972    0.54675003    0.300000016   7.88612569   0.0170233396
90    3.75746e-07      1.35000000    1.48973225    0.54675003    0.300000018   7.88603391   0.0170207778
```

The remaining central-Omega spread at `res=9` is spectral truncation: a
higher-resolution `res=11` endpoint check for `deg=0,90` passed the stricter
default invariant tolerance, reducing the central-Omega difference from the
`res=9` level to approximately `2.14e-8`.
