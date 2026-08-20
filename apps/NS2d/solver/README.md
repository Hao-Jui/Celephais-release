# isoNS notes

## Domain-1 outer boundary adjustment (ns_2d_msqi*)

The outer boundary of domain 1 is an adapted mapping variable, so it is updated
every Newton iteration from the solver's correction vector.

Key flow:
- The Newton step produces `xx`, which is unpacked in
  `Space_polar_adapted::xx_to_vars_variable_domains`.
- The outer boundary correction is stored in `cor_outer`, used to update all
  fields in that domain, and then `update_mapping(cor_outer)` is called on the
  adapted outer domain to move the boundary each iteration.
- The update itself is `*outer_radius += cor`, followed by mapping rebuilds in
  `Domain_polar_shell_outer_adapted::update_mapping`.
- The boundary condition driving the adjustment for domain 1 in ns_2d_msqi is
  `H = 0` on `OUTER_BC`.

Pointers (boundary adjustment):
- `src/Space/Adapted_polar/space_polar_adapted.cpp`
- `src/Domain/Adapted_polar/domain_polar_outer_adapted.cpp`
- `include/Apps/TwoD/IsoNS/ns_2d_msqi_solver_imp.ipp`
- `src/System_of_eqs/do_newton.cpp`

## How `xx` is produced in the Newton step

`xx` is the global Newton correction vector assembled in
`System_of_eqs::do_newton` by solving the linearized system

```
J * x = F
```

where `J` is the Jacobian of all equations with respect to the unknowns and
`F` is the current residual (the “second member” from `sec_member()`).

The workflow in `src/System_of_eqs/do_newton.cpp` is:
- Build the residual vector `second` and assemble the Jacobian columns with
  `do_col_J`.
- Solve the distributed linear system with ScaLAPACK `pdgesv_` (or the normal
  equations for Gauss-Newton). The solution is gathered into a full vector `X`.
- `X` is the `xx` passed into `xx_to_vars_variable_domains` and `xx_to_vars`,
  which apply the correction to the adapted mapping and the field variables.

Notes:
- The correction is applied as “old minus delta” (see the `*var = *old - *var`
  update after `xx_to_vars`), so the sign convention is embedded there.
- For isoNS, this is the vector that drives the adapted outer boundary update
  described above.
- In isoNS, `System_of_eqs::do_newton` dispatches to the space instance’s
  `xx_to_vars_variable_domains`; because the solver uses `Space_polar_adapted`,
  this resolves to `Space_polar_adapted::xx_to_vars_variable_domains` in
  `src/Space/Adapted_polar/space_polar_adapted.cpp` (declared in
  `adapted_polar.hpp`). Other implementations (e.g., `bin_ns.hpp`) are for
  different space types and are not used here.

## How `derive_t` participates in Newton

`derive_t` is not called directly from `do_newton`; it is used inside the
domain operator algebra when assembling the Jacobian:
- `System_of_eqs::do_newton` builds Jacobian columns via
  `System_of_eqs::do_col_J`.
- `do_col_J` triggers `update_terms_from_variable_domains`, which calls each
  domain’s `update_term_eq`.
- When equation operators (e.g., `lap_term_eq`, `flat_grad_spher`, `lap2_term_eq`)
  are applied to a `Term_eq`, they use `derive_t` to compute the theta-derivative
  of both `val_t` and `der_t`. These contributions are what end up in the
  Jacobian entries for adapted domains.

Pointers (derive_t/Jacobian):
- `src/System_of_eqs/solver.cpp`
- `src/Domain/Adapted_polar/domain_polar_outer_adapted_term_eq.cpp`

## Shift field spectral parameters (msqi 2D)

The msqi 2D solvers set the shift field to use the axisymmetric azimuthal
mode `m = 1` and rebuild the basis after changing that setting:
- `shift.affect_parameters()` ensures the parameter block is attached.
- `shift.set_parameters()->set_m_quant() = 1;` enforces the `m = 1` Fourier
  mode for the phi-direction shift in the axisymmetric formulation.
- `shift.std_base();` rebuilds the standard spectral basis after changing
  parameters.

Pointers:
- `include/Apps/TwoD/IsoNS/ns_2d_msqi_solver_imp.ipp`
- `include/For_Kadath/tensor.hpp`
- `src/Tensor/tensor.cpp`
- `src/Tensor/param_tensor.cpp`
