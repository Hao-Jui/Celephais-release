# Configurator

The parameter path is:

```text
TOML file  <->  ConfigTree  <->  kadath_config<ParamContainer>  <->  solver
read_toml_config_tree()      open_config()/write_config()           bconfig(KEY)
```

Runtime solver code should read `.toml` directly through `kadath_config`.
Example TOML files and updated solver TOML snapshots are also written through
`kadath_config::write_config()`. Do not add app-specific bridge layers, typed
TOML mirrors, or hand-written key tables for new parameters. The enum registry
is the source of truth for accepted runtime keys, enum order, generated maps,
and enum-family registration:

- `include/For_Kadath/Config/config_enum_entries.hpp`

To add, rename, or retire a Configurator key, edit the relevant
`KADATH_*_ENTRIES` list there. To add a new enum family, add its entry-list
macro and one row in `KADATH_CONFIG_ENUMS`.

## Read Flow

`kadath_config<...>::open_config()` parses TOML into `ConfigTree`, reads the
node branch (`bh`, `ns`, or `binary`), then fills enum-indexed storage arrays:

| TOML section | Runtime storage | Registry map |
|--------------|-----------------|--------------|
| `bh`, `ns`, `binary` | compact-object or binary parameter arrays | `MBCO_PARAMS()`, `MBIN_PARAMS()`, `MEOS_PARAMS()`, ... |
| `fields` | field toggles | `MBCO_FIELDS()` |
| `stages` | stage toggles | `MSTAGE()` plus container stage subsets |
| `sequence_controls` | sequence booleans | `MCONTROLS()` |
| `sequence_settings` | sequence numeric settings | `MSEQ_SETTINGS()` |
| `gravity` | gravity theory selection | `MGRAVITY_PARAMS()` |

The solver-facing API remains the existing enum-indexed access pattern:

```cpp
using config_t = kadath_config<BCO_BH_INFO>;
config_t bconfig{input_toml_path};

const auto resolution = bconfig(BCO_RES);
const bool checkpoint = bconfig.control(CHECKPOINT);
const bool def_theory = bconfig.is_def();
```

The app solve entry points and readers use this direct runtime path.

## Write Flow

`kadath_config<...>::write_config()` is the only Configurator TOML writer. It
serializes the same enum-indexed storage arrays that `open_config()` reads:

1. `container.return_branch()` builds the compact-object or binary branch from
   the container maps (`MBCO_PARAMS()`, `MBIN_PARAMS()`, `MEOS_PARAMS()`,
   and `MDIFFROT_PARAMS()`).
2. `build_branch()` serializes `fields`, `stages`, `sequence_controls`,
   `sequence_settings`, and `gravity` from their registry maps.
3. `write_toml_config_tree()` emits TOML from the resulting `ConfigTree`.

Example setup files follow the same route. The app entry point constructs the
runtime `kadath_config`, calls `set_defaults()` or a small app-specific default
customizer, then calls `write_config()` through `write_example_toml()`.
App-specific customizers may choose compact-object types or dimensional defaults,
but they must use enum accessors and must not introduce key-string tables.

## Restart Sidecars

The Configurator accepts only `.toml` config paths. Readers may accept a saved
solution `.dat` path as a convenience, but only by resolving the matching
`.toml` sidecar before constructing `kadath_config`. The binary data path then
comes from `kadath_config::space_filename()`. Do not reconstruct restart data
paths manually from `config_filename()` and `config_outputdir()`.

## File Layout

- `config_enum_entries.hpp` defines the X-macro registry:
  `KADATH_BIN_PARAM_ENTRIES`, `KADATH_BCO_PARAM_ENTRIES`,
  `KADATH_EOS_PARAM_ENTRIES`, `KADATH_CONTROL_ENTRIES`, `KADATH_GRAVITY_ENTRIES`,
  stage entries, node entries, field entries, and the aggregate
  `KADATH_CONFIG_ENUMS` registry.
- `config_enums.hpp` expands the registry into enum classes, count constants,
  and map-accessor declarations.
- `src/Utilities/Configurator/config_enums.cpp` expands the same registry into
  `ConfigEnumEntry` arrays, string-to-enum maps, and count `static_assert`s.
- `config_tree.hpp` is the minimal TOML-backed tree representation used between
  parsing and runtime container loading.
- `config_utils_toml.hpp` and `config_utils_toml.inl` provide TOML read/write
  helpers.
- `config_bco.hpp`, `config_binary.hpp`, and `configurator_toml.hpp` define the
  binary and compact-object containers that consume the registry maps.

## Parameter Containers

The containers are storage views over registry-defined keys. They should not
own independent key lists.

### `BCO_INFO`

```text
BCO_INFO
|-- node type: "bh" by default
|-- bco_params: Array<ConfigSlot>
|   `-- map: MBCO_PARAMS()
`-- accessors
    `-- operator()(idx)
```

### `BCO_BH_INFO`

```text
BCO_BH_INFO : BCO_INFO
`-- node type: "bh"
```

Black-hole containers read compact-object maps.

### `BCO_NS_INFO`

```text
BCO_NS_INFO : BCO_INFO
|-- node type: "ns"
|-- eos_params: EOSArray
|   `-- map: MEOS_PARAMS()
|-- diffrot_params: DIFFROT_ary
|   `-- map: MDIFFROT_PARAMS()
`-- accessors
    |-- set_eos_param(idx), get_eos_param<T>(idx)
    `-- set_diffrot_param(idx), get_diffrot_param<T>(idx)
```

Neutron-star containers read compact-object, EOS, and differential rotation maps
from the `ns` branch.

### `BIN_INFO`

```text
BIN_INFO
|-- node type: "binary"
|-- bin_params: Array<double>
|   `-- map: MBIN_PARAMS()
|-- bin_stages: std::map<string, STAGE>
|   `-- map: MSTAGE()
`-- BCOS[2]: unique_ptr<BCO_INFO>
    |-- BCO_BH_INFO when the compact-object type is "bh"
    `-- BCO_NS_INFO when the compact-object type is "ns"
```

## Maintenance Rules

- Do not duplicate TOML key tables in documentation. Link to
  `config_enum_entries.hpp` instead.
- Read and write through `kadath_config<...>`. Do not reintroduce typed
  app-config wrappers, bridge adapters, or parallel TOML reader/writer layers.
- Keep enum order stable unless migration code or fixtures are updated with the
  change.
- Empty key strings in an entry list are internal or retired slots retained for
  array-index compatibility; they are not runtime TOML keys.
- Curated subset maps, such as stage subsets, should reference enum values from
  the registry rather than repeating key strings.
- After registry changes, run
  `ctest --test-dir build --output-on-failure -R 'configurator|config_layer_guard'`.

## Gravity Theory Selection



```toml
# GR (default — section may be omitted entirely)
[gravity]
theory = "GR"



Solver dispatch:



## Per-Parameter Touch Surface

Existing groups are intentionally cheap to maintain. The normal touch surface is
one registry row, plus an optional runtime default when generated examples
should emit the value. New groups are still structural additions, but they are
centralized in the same runtime Configurator path rather than mirrored through
typed TOML readers, writers, or bridges.

### Add To Existing Group

Use this path when solver code will read the value through `bconfig(KEY)`,
`binary_config(KEY)`, `control(KEY)`, `set_stage(KEY)`, `seq_setting(KEY)`,
`gravity<T>(KEY)`, or a related enum-indexed accessor.

1. Add one enum/key row to the matching `KADATH_*_ENTRIES` list in
   `config_enum_entries.hpp`.

   ```cpp
   #define KADATH_BCO_PARAM_ENTRIES(APPLY, CTX) \
       /* existing rows */                      \
       APPLY(CTX, HOGE_RADIUS, "hoge_radius")
   ```

   That row automatically updates the enum value, `NUM_*_V`, the generated
   key-to-enum map, TOML reads through `open_config()`, and TOML writes through
   `write_config()`.

2. Use the generated enum in runtime code.

   ```cpp
   const auto hoge_radius = bconfig(HOGE_RADIUS);
   bconfig.set(HOGE_RADIUS) = 1.0;
   ```

3. If generated examples should include the value, set a runtime default in the
   relevant container (`config_bco.hpp` or `config_binary.hpp`).

   ```cpp
   bconfig.set(HOGE_RADIUS) = 1.0;
   ```

4. If the value belongs to a curated subset, such as an app-specific stage list,
   update only the enum list in `src/Utilities/Configurator/config_enums.cpp`.
   Do not repeat the TOML key string.

   ```cpp
   static constexpr std::array<STAGE, 4> neutron_star_stages = {
       STAGE::NOROT,
       STAGE::UNIROT,
       STAGE::BIN_BOOST,
       STAGE::HOGE_STAGE,
   };
   ```

5. Update TOML fixtures or focused tests that intentionally cover the new key,
   then run the Configurator tests and guard.

### Create New Group

Use this path when the new data does not belong in any existing storage group
(`bh`/`ns`/`binary`, `fields`, `stages`, `sequence_controls`, or
`sequence_settings`).

1. Add a new entry-list macro in `config_enum_entries.hpp`.

   ```cpp
   #define KADATH_HOGE_SETTING_ENTRIES(APPLY, CTX) \
       APPLY(CTX, HOGE_ALPHA, "hoge_alpha")        \
       APPLY(CTX, HOGE_BETA, "hoge_beta")
   ```

2. Register the group in `KADATH_CONFIG_ENUMS`.

   ```cpp
   X(HOGE_SETTINGS, NUM_HOGE_SETTINGS, KADATH_HOGE_SETTING_ENTRIES,
     hoge_setting_entries, MHOGE_SETTINGS, 1)
   ```

   This emits the enum class, count constant, entry array, map accessor, and
   compile-time drift check.

3. Add storage and accessors in `configurator_toml.hpp`.

   ```cpp
   std::array<double, NUM_HOGE_SETTINGS_V> hoge_settings{};

   template <typename E> auto& hoge_setting(E idx)
   {
       return hoge_settings[static_cast<int>(idx)];
   }
   ```

4. Wire read and write in `configurator_toml.inl`.

   ```cpp
   read_keys(MHOGE_SETTINGS(), hoge_settings, read_branch(tree, "hoge_settings"));

   new_tree.push_back(std::make_pair(
       "hoge_settings",
       build_branch<ConfigTree>(MHOGE_SETTINGS(), hoge_settings)));
   ```

5. Add defaults if examples should emit the group, and add focused read/write
   tests.

### Verification

After registry or group changes, run:

```sh
cmake --build build --target celephais_unit_tests
ctest --test-dir build --output-on-failure -R 'configurator|config_layer_guard'
```

There is no second typed config surface. A parameter that is read or written by
the Configurator must be represented by the runtime enum registry and consumed
through `kadath_config`.

## `sequence_controls` Flag Reference



## `stages` Flag Reference

Stage names are per-physics: the same gate position means a different solve in
each app family, so each family has its own enum entries. Binary BNS/BHNS solve
order is explicit in `Apps/Formalism/Shared/binary_driver_common.hpp`
(`QUASI_EQUIL -> FORCE_BALANCE -> ECC_RED`) rather than inferred from enum
order. `BIN_BOOST` must stay last.

| Flag | TOML key | Apps | Solve |
|------|----------|------|-------|
| `NOROT` | `norot` | NS | Non-rotating (TOV-like) warm-up solve. |
| `UNIROT` | `uniform_rot` | NS | Uniform rotation production solve; chi continuation toward `FINAL_CHI`. |
| `DIRICHLET_LAPSE` | `dirichlet_lapse` | BH | Fixed (Dirichlet) lapse on the excision. |
| `VON_NEUMANN` | `von_neumann` | BH | Neumann lapse condition; production solve with chi ladder. |
| `TRUMPET` | `trumpet` | KSBH/Trumpet | Trumpet BH with resolved horizon. |
| `QUASI_EQUIL` | `quasi_equilibrium` | BNS/BHNS/BBH/3NS | Fixed-PN-Omega hydro-rescale initializer: enthalpy is rescaled to the baryon-mass targets. In BHNS this stage uses the fixed-lapse BH boundary condition. |
| `FORCE_BALANCE` | `force_balance` | BNS/BHNS | Hydrostatic equilibrium with Omega solved by force balance (`hydrostatic_equilibrium_stage`): H solved with the Euler first integral, Mb via the central-enthalpy unknowns. In BHNS this stage uses the von-Neumann BH lapse condition. |
| `ECC_RED` | `ecc_red` | binaries | Eccentricity-reduction hydro-rescale at fixed Omega (PN via `use_pn`, or user `ecc_omega`/`adot`). BNS/BHNS use the same rescale family as `QUASI_EQUIL` under the `ECC_RED` label; BNS/BHNS add the eccentricity-reduction radial term and ADM-momentum closures here. In BHNS this stage uses the von-Neumann BH lapse condition. |
| `BIN_BOOST` | `binary_boost` | NS/BH seeds | Boosted-CO continuation, driven by the binary workflow. |
| `FIXED_OMEGA`, `COROT_EQUAL`, `TOTAL_FIXED_COM`, `TESTING`, `GRAV`, `VEL_POT_ONLY` | — | none | Legacy slots; readable but not gated by any current solver. |

Converged-solution filenames embed the uppercase flag name
(`converged_NS_UNIROT.*`, `converged_BNS_QUASI_EQUIL.*`, ...);
`developer/migrate_stage_filenames.sh` renames pre-rename artifacts
(convergence caches, run directories) in place.

### Legacy v1 stage keys

v1 configs wrote `norot_bc`/`total`/`total_bc`, whose meaning depends on the
app, so they are **rejected at read time** (`open_config()` throws naming the
key and the file). Migrate a dataset once with
`developer/migrate_stage_filenames.sh <dir>` — it renames converged artifacts
and rewrites the `[stages]` keys in every TOML using this mapping:

| v1 key | NS | BH | KSBH (by hand) | BNS/BHNS/BBH/3NS |
|--------|----|----|----------------|------------------|
| `norot_bc` | `norot` | — | — | — |
| `total` | — | `dirichlet_lapse` | — | `force_balance` |
| `total_bc` | `uniform_rot` | `von_neumann` | `trumpet` | `quasi_equilibrium` |

(Trumpet/KSBH configs also use a `[bh]` node and cannot be auto-detected;
edit those by hand.)
