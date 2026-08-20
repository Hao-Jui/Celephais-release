# Space_three_body Parameter Mapping

`Space_three_body` represents a nested, collinear three-body layout:

- a parent five-domain bispheric aggregate wraps one spherical body and one spherical child aggregate;
- the child aggregate is itself a five-domain bispheric aggregate wrapping two spherical bodies;
- the parent outer boundary and the child aggregate outer boundary are both spheres.

The parameter mapping is extended by composing two independent bispheric maps.

## Inputs

The constructor accepts:

```cpp
Space_three_body(
    int ttype,
    double parent_dist,
    double child_dist,
    const std::vector<double>& body_bounds,
    const std::vector<double>& child1_bounds,
    const std::vector<double>& child2_bounds,
    double child_outer_radius,
    const std::vector<double>& outer_bounds,
    int nr);
```

The three body bound arrays follow the adapted-star convention:

```text
[rin, rmid, rout]
```

or, with extra fixed shells:

```text
[rin, rmid, shell_0_outer, ..., rout]
```

`outer_bounds[0]` is the parent bispheric outer radius. Any later entries in
`outer_bounds` become fixed outer spherical shells, followed by a compactified
domain.

## Shared Bispheric Map

Each bispheric aggregate is built from:

```text
(r_minus, r_plus, distance, outer_radius)
```

The coordinate scale `a` is solved from:

```text
sqrt(a^2 + r_minus^2) + sqrt(a^2 + r_plus^2) = distance
```

Then the two exposed sphere centers in that map's local coordinate system are:

```text
eta_minus   = -asinh(a / r_minus)
eta_plus    =  asinh(a / r_plus)
minus_center = a * cosh(eta_minus) / sinh(eta_minus)
plus_center  = a * cosh(eta_plus)  / sinh(eta_plus)
```

The spherical outer boundary sets the transition limits:

```text
chi_c   = 2 * atan(a / outer_radius)
eta_c   = log((1 + outer_radius / a) / (outer_radius / a - 1))
eta_lim = eta_c / 2
chi_lim = chi_lim_eta(eta_lim, outer_radius, a, chi_c)
```

## Composition

The parent map treats the child aggregate as a single exposed sphere:

```text
parent_map = make_bispheric_map(
    body_bounds.back(),
    child_outer_radius,
    parent_dist,
    outer_bounds[0])
```

The child map is solved in its own local coordinates:

```text
child_map = make_bispheric_map(
    child1_bounds.back(),
    child2_bounds.back(),
    child_dist,
    child_outer_radius)
```

The parent map's plus-side center becomes the child aggregate origin:

```text
child_origin_x = parent_map.plus_center
```

The actual body centers in global coordinates are:

```text
body_center_x   = parent_map.minus_center
child1_center_x = child_origin_x + child_map.minus_center
child2_center_x = child_origin_x + child_map.plus_center
```

The five child bispheric domains are first constructed from `child_map` in local
coordinates and then translated by `child_origin_x`. The parent bispheric
domains remain centered at the global origin.

## Domain Order

After any fixed spherical shells around the bodies, domain indices are assigned
as:

```text
BODY
ADAPTED_BODY, ADAPTED_BODY + 1, optional body shells
CHILD1
ADAPTED_CHILD1, ADAPTED_CHILD1 + 1, optional child1 shells
CHILD2
ADAPTED_CHILD2, ADAPTED_CHILD2 + 1, optional child2 shells
CHILD_BI ... CHILD_BI + 4
PARENT_BI ... PARENT_BI + 4
OUTER ... compact_domain
```

`CHILD_BI ... CHILD_BI + 4` are the child aggregate's five translated
bispheric domains. `PARENT_BI ... PARENT_BI + 4` are the parent aggregate's
five global bispheric domains.

## Variable-Domain Mapping

The variable-domain unknowns are the adapted surfaces of all three spherical
bodies:

```text
ADAPTED_BODY / ADAPTED_BODY + 1
ADAPTED_CHILD1 / ADAPTED_CHILD1 + 1
ADAPTED_CHILD2 / ADAPTED_CHILD2 + 1
```

`nbr_unknowns_from_variable_domains`, `affecte_coef_to_variable_domains`,
`xx_to_ders_variable_domains`, and `xx_to_vars_variable_domains` all iterate
over these three adapted pairs. This extends the two-body pattern by adding the
second child-body adapted pair while keeping the child bispheric aggregate
placement controlled by `child_origin_x`.
