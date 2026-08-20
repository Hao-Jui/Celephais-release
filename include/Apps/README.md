# Apps — solver architecture reference

App solver headers sit under `include/Apps`. They may use public Kadath
library utilities, but they must not depend on library-side Newton linear
algebra internals under `include/Linear_algebra`.

**Policy vs Workflow** — the load-bearing distinction:

- **Policy = knobs/defaults you set.** Tunable values and the rules that bound
  them: the default initial resolution, the odd-resolution rule, the
  `ResolutionLadder` value type. Edit here to change *what* a solve targets.
- **Workflow = mechanism you don't toggle.** The loops that consume those
  values: the +2 continuation stepper, the warm-start ladder floor, the
  per-app solve/regrid drivers. Mechanism depends on Policy, never the reverse.



| Layer | Purpose |
|---|---|
| **Policy** | Knobs + defaults: `app_resolution.hpp` (default init res, ladder type, odd-res rule), `bns_policy_common.hpp`. The place to set tunables. |
| **Workflow** | Mechanism, no knobs: continuation stepper + warm-start floor (`solver_sequence.hpp`, `resume_ladder.hpp`) and the per-app solve/regrid drivers (`*_app_workflow.hpp`, `stage_outcome.hpp`, `dataset_resume.hpp`). |
| **Startup** | Config file -> space + field loading. Nothing else. |
| **Helper** | App solver base class and MPI-consensus Newton loop wrapper. |
| **Seed** | Generate initial `.dat` from physics (TOV → fields). |
| **Formalism** | 2D/3D Newton stages, regrid wrappers, per-theory solvers. |
| **Shared** | Cross-solver sequencing (incl. the NS driver state machine), shared regrid templates. |
| **Bco_utils** | App-only compact-object bounds/regrid/IO. |
| **AMR** | Adaptive-refinement policies + hp indicators. |
| **Diagnostics** | Templated snapshot/reader thin-mains. |
| **Evolution** | Spectral time-evolution (CCZ4/Z4c, GRHD, SBP, PHE). |

BCO helpers are split by who depends on them. App-only bounds/regrid/IO
(`ns_bounds`, `bco_regrid`, `bco_io`, `bh_bounds`) live in `Apps/Bco_utils/`.
Helpers the library exporters in `src/` also use stay under `For_Kadath/`:
geometry + coord fields in `For_Kadath/Utilities/Exporters/`, the PN orbital
parameters in `For_Kadath/Utilities/PN/orbital_pn_params.hpp`.

---

## Directory map



---

## Include chain for a 3-D GR solver

```
<solver>_solver_imp.ipp
  └─ <solver>_regrid.hpp
       ├─ Apps/Formalism/Shared/Regrid/ns_regrid.hpp
       ├─ Apps/Seed/GR/<object>_seed.hpp
       ├─ Apps/Helper/solver_base.hpp
       └─ Apps/Bco_utils/*.hpp   (bounds/regrid/IO; pulls For_Kadath/Utilities/Exporters/bco_geometry.hpp)
```

---

## BCO seed architecture

### Key types

```cpp
namespace Kadath::Seed {

struct GR {};                         // theory tag; add ScalarTensor, EdGB, … here

// Primary template — unspecialised combinations are a compile error.
template <NODES s_type, typename space_t, typename theory_t = GR>
struct ObjectSeeder;

// Public entry point.
template <NODES s_type, typename space_t, typename config_t, typename theory_t = GR>
void setup_co(config_t& bconfig);     // delegates to ObjectSeeder::execute

}
```

### How a 3-D solver calls it

```cpp
// Formalism/GR/ns_3d_xcts_nosym/solver_imp.ipp (rank-0 guard):
Kadath::Seed::setup_co<NS, Space_spheric_adapted_nosym>(bconfig);

// Formalism/GR/ns_3d_xcts/solver_imp.ipp:
Kadath::Seed::setup_co<NS, Space_spheric_adapted>(bconfig);

// Formalism/GR/bh_3d_xcts/bh_3d_xcts_solver_imp.ipp:
Kadath::Seed::setup_co<BH, Space_adapted_bh>(bconfig);
```

The isolated-BH seed has two field profiles behind the same `ObjectSeeder`
specialisation:

| `[bh].trumpet_bh_seed` | profile |
|------------------------|---------|
| unset or `false` | legacy conformal seed (`P` seeded from `bco_utils::psi`, zero shift) |
| `true` | non-rotating stationary trumpet seed from arXiv:2212.10891 (`R(r=0)=3MCH/2`, horizon at `r≈0.779327 MCH`) |

Both profiles write the same `{space, conf, lapse, shift}` data layout consumed
by the GR BH XCTS solver. The trumpet option is currently restricted to
`CHI = 0`.

`apps/Trumpet` uses the same `[bh].trumpet_bh_seed = true` profile, but loads it
into `Space_trumpet`: an NS-style adapted spherical space with no excised
domains. Its adapted interface is initialised at the paper-profile horizon and
then held on the paper horizon level set

```cpp
lev = r^2 - r_h^2 = 0
```

applied at the exterior shell side of the horizon (`domain 2 / INNER_BC`).
The interior domains remain populated by the analytic regular trumpet profile;
the Newton solve is restricted to the exterior `P` and `N` fields because the
paper's regular puncture variable is `W = psi^-2`, while Kadath's XCTS scalar is
`P = psi`, which is singular at the puncture.

The 2-D polar solvers (`isoNS`, `isoBH`, `excisionBH`, `ms_BH`) are standalone:
they generate their own initial data from TOV and solve in isolation, with no
dependency on another solver's converged output.  They use `::setup_co<s_type>`
from `2D/Seed/co_solver_utils.hpp`.  ObjectSeeder does not apply to them.

---

## Extension protocol

### New space variant (same theory, same object type)

1. Create `Seed/GR/ns_nosym_ring_seed.hpp` (hypothetical):

```cpp
#pragma once
#include "../seed_traits.hpp"
#include "../seed_utils.hpp"
#include "spheric_adapted_nosym_ring.hpp"

namespace Kadath::Seed {
template <>
struct ObjectSeeder<NS, Space_spheric_adapted_nosym_ring, GR> {
    template <typename config_t>
    static void execute(config_t& bconfig) {
        // ... Space_spheric_adapted_nosym_ring space(...);
        //     write_ns_fields(space, bconfig, *tov);
    }
};
} // namespace Kadath::Seed
```

2. Include from the solver's regrid file:

```cpp
// Formalism/GR/ns_3d_xcts_nosym_ring/regrid.hpp:
#include "Apps/Seed/GR/ns_nosym_ring_seed.hpp"
```

3. Update the solver driver call:

```cpp
Kadath::Seed::setup_co<NS, Space_spheric_adapted_nosym_ring>(bconfig);
```

**Zero changes to existing files.**

---

### New gravitational theory

1. Add theory tag to `Seed/seed_traits.hpp`:

```cpp
namespace Kadath::Seed { struct ScalarTensor {}; }
```

2. Create `Seed/ScalarTensor/ns_seed.hpp`:

```cpp
#pragma once
#include "../seed_traits.hpp"
#include "../seed_utils.hpp"
#include "kadath_adapted.hpp"



3. Caller:

```cpp
Kadath::Seed::setup_co<NS, Space_spheric_adapted, ScalarTensor>(bconfig);
```

**Existing GR code is untouched.**

---

### Adding a field to all NS seeds

Modify `Seed/seed_utils.hpp` — `write_ns_fields<space_t>` constructs and saves
all fields.  A field needed only in one specialisation should be written inline
in that specialisation's `execute()` instead.

---

### Note on 2-D solvers and ObjectSeeder

2-D polar solvers are standalone by design: they solve in isolation and generate
their own initial data from TOV.  ObjectSeeder is for solvers that receive a seed
produced by another solver (e.g. isolated NS seeded before being inserted into a
binary).  Do not wire 2-D solvers into ObjectSeeder unless the architecture changes
to require an external seed handoff.
