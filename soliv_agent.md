# Celephais AI Onboarding

AI-only contract. Minimize time to first build/run on laptop, workstation, or
HPC. Prefer the generic `env` path for unknown machines; named presets are
examples/legacy conveniences, not the scaling strategy.

> Machine inspecific guideline.

## Contract

- Production build requires MUMPS. Dense/Schur paths are debug-only.
- Do not guess scheduler partition/account/qos/resources. Generate a guarded
  Slurm template and ask the human to fill only those scheduler facts.
- Installation is incomplete until a user can rebuild from a fresh login shell
  with bare `./compile.sh`. After the first successful build, persist the machine
  setup and update `compile.sh` as required by step 4; do not leave the user
  dependent on the agent's temporary shell state or a target argument.
- Completion is invalid unless the final prompt/chat message pastes the exact
  stdout of `apps/submission_scripts/celephais_submission_usage.sh <target>`.
  Terminal output or log paths do not count.

## Routes

Preflight (step 2) reports `TARGET_CLASS`, which picks the route. Same build for
both — only the run/handoff differs.

- **direct** — laptop / workstation / inside an allocation (`macbook`, `snowman`,
  or `env` with no `sbatch`). Steps **1 → 2 → 3 → 4 → 6**; run the binary right
  there. **Skip step 5** (no scheduler). Laptop-install is this route.
- **slurm** — HPC scheduler login node. Steps **1 → 2 → 3 → 4 → 5 → 6**; step 5
  generates the guarded site script the human fills, then submit via `sbatch`.

Completion is the step-6 fixed working-directory usage paste in either case.

## Prerequisites

Install the toolchain once; the State Machine assumes it is present. MUMPS and
METIS are built in-tree — never install them.

- **macOS laptop (`macbook` preset):** Homebrew, then
  `brew install ninja ccache open-mpi fftw openblas scalapack`.
  The toolchain resolves each via `brew --prefix`; without `open-mpi` CMake
  aborts at configure. `ninja` is required (every preset uses it); `ccache` is
  optional (speeds rebuilds).
- **Linux workstation / HPC (`env` path):** supply a C/C++/Fortran compiler +
  MPI, plus FFTW, BLAS/LAPACK (OpenBLAS or MKL) and ScaLAPACK, with `ninja`
  (+ optional `ccache`) on `PATH` — from Lmod modules or the package manager
  (`apt install ninja-build`, `dnf`, `pacman`). Export the stack in step 1.

## State Machine

1. Prepare one coherent compiler/MPI/math environment.
   - Laptop: use the maintained `macbook` preset when appropriate.
   - Unknown Linux workstation/HPC: load/export the site stack, then use `env`.
   - Required useful exports: `CC`, `CXX`, `FC`, `FFTW_HOME`; and
     optional `SCALAPACK_DIR`, `MKLROOT`, `CELEPHAIS_MUMPS_EXTRA_LIBRARIES`
     (`BOOST_ROOT`/`BOOST_HOME` are only an rpath hint — Boost is not a build
     dependency). MUMPS is built automatically from
     `third_party/mumps` — no `MUMPS_DIR` or `PETSC_DIR` needed.
   - **MUMPS Fortran-MPI link tail.** The bundled MUMPS is Fortran+MPI; its
     archives are pulled into a C++-driven final link that must resolve the MPI
     *Fortran* bindings (`libmpi_mpifh`). Undefined `mpi_bcast_`/`mpi_pack_`/
     `mpi_isend_` at the final link ⟹ that tail is missing. Two compiler setups —
     pick what your MPI accepts:
     - *Plain compilers* (`gcc/g++/gfortran`): if `find_package(MPI)` validates
       (`MPI_<lang>_WORKS` passes), it fills `MPI::MPI_Fortran` with the Fortran
       libs automatically — nothing else needed (the macbook path).
     - *Wrapper compilers* (`mpicc/mpicxx/mpif90`): some clusters need the
       wrapper AS the compiler because plain-gcc + FindMPI flag-extraction fails
       `MPI_<lang>_WORKS` — e.g. BinAC2's OpenMPI wrapper adds UCX/libfabric
       rpath that `-showme` does not reproduce. FindMPI then returns an EMPTY
       `MPI::MPI_Fortran`, so set `CELEPHAIS_MUMPS_EXTRA_LIBRARIES` to the
       Fortran-MPI tail (the same site-tail the ncsa/delta/snowman toolchains
       use). Derive it from the wrapper so it follows the module, e.g.:
       `D=$(dirname $(dirname $(command -v mpif90)))/lib64;
       export CELEPHAIS_MUMPS_EXTRA_LIBRARIES="$D/libmpi_mpifh.so;$D/libmpi.so;gfortran"`
   - Lmod is hierarchical: load the **compiler first, then MPI, then libraries**,
     in a **login shell**. Wrong order or a non-login shell makes `module load`
     a silent no-op (`module list` stays empty though it returns success).

2. Normalize and preflight:

```bash
source cmake/env/env.sh
tools/celephais_preflight.sh --target env
```

Read only:

```text
READY=
TARGET_CLASS=
BLOCKER=
NEXT=
```

If `READY=0`, do `NEXT`. Ask the human only when `NEXT` explicitly requires
human data.

3. Build:

```bash
./compile.sh macbook   # laptop; bare `./compile.sh` auto-detects Darwin -> macbook
./compile.sh env       # unknown Linux workstation / HPC (after step 2)
```

CMake must configure, print `Celephais stack report` (with `MUMPS provider: bundled`),
define `CELEPHAIS_USE_MUMPS`, and build the in-tree MUMPS from `third_party/mumps`.
After the build, `compile.sh` installs the complete relocatable CMake package to
`build/install` by default. This includes the library, public headers,
`CelephaisConfig.cmake`, and `CelephaisTargets.cmake`; a consumer can configure
with `-DCMAKE_PREFIX_PATH=/absolute/path/to/Celephais/build/install` and link
`Celephais::solution`. Set `CELEPHAIS_INSTALL_PREFIX` before invoking
`compile.sh` only when a different writable installation prefix is required.
The script must fail if `CelephaisTargets.cmake` is absent after installation.

**Build bundled METIS/MUMPS serially if the parallel build fails.** The
`metis_bundled` / `mumps_bundled` ExternalProjects run a nested `cmake --build`;
under a parent GNU-Make jobserver (`cmake --build --parallel N`, which is what
`compile.sh` does) an older make (< 4.4) aborts them with
`read jobs pipe: Bad file descriptor`. Build the two deps without the jobserver
first, then parallel-build the rest:

```bash
cmake --preset env
cmake --build build --target metis_bundled   # serial: no jobserver
cmake --build build --target mumps_bundled   # serial: no jobserver
cmake --build build --parallel 8             # deps stamped; sources compile -jN
```

**Bundled METIS is built 64-bit (`IDXTYPEWIDTH=64`).** `BundledMetis.cmake` patches
the copied `metis.h` to a 64-bit `idx_t` (`cmake/patch_metis_idx64.cmake`) so the
MUMPS analysis phase can order matrices with more than 2^31 nonzeros. Without it,
high-resolution runs (res15 BNS/BHNS, nnz ~ 2e9) die in analysis with MUMPS
`INFOG(1)=-51` "integer overflow in ordering". This widens the **ordering only**:
MUMPS uses its `mumps_metis_kway_*_mixedto64` wrappers, `MUMPS_INT` stays 32-bit
(matrix order N never nears 2^31; NNZ is already 64-bit in MUMPS's mixed mode), and
BLAS/(Sca)LAPACK/MPI stay LP64 — **no ILP64 build needed**. Changing the METIS
integer width on an existing build tree requires clearing BOTH the copied tree and
the ExternalProject stamps:
```bash
rm -rf build/metis build/metis_bundled-prefix build/mumps build/mumps_bundled-prefix
```
(removing only `build/metis` leaves the cached download stamp, so the patch step
runs against a missing header and fails).

4. Make later rebuilds one command. From a fresh login shell, bare `./compile.sh`
   must select the successful target and recreate its compiler/MPI/math
   environment without manual `module load`, `source`, exports, or a target
   argument.
   - Keep machine-specific module loads and exports in `cmake/env/<site>.sh`, and
     add or reuse the matching CMake preset/toolchain.
   - For an unregistered host, update `detect_target()` in `compile.sh` to map
     its stable hostname pattern to `<site>`. Updating only the environment file
     or preset is insufficient because bare `./compile.sh` would still select
     `env`.
   - An already registered route such as `macbook` needs no redundant edit, but
     it must pass the same fresh-shell check.

Verify from outside the agent's prepared shell (replace the path with this
checkout's absolute path):

```bash
bash -lc 'cd /absolute/path/to/Celephais && ./compile.sh'
```

5. Slurm route only (`TARGET_CLASS=slurm`) — direct route skips to step 6:

```bash
tools/generate_submission_template.sh <site>
tools/celephais_preflight.sh --target <site> --submit
```

If the blocker is `SBATCH_RESOURCES`, ask the human:

```text
Please edit apps/submission_scripts/<site>.sh: fill partition/account/qos if required, nodes, ntasks-per-node, cpus-per-task, time, then set CELEPHAIS_SCHEDULER_READY=1 in that file.
```

After the human fills it:

```bash
tools/celephais_preflight.sh --target <site> --submit
```

6. Handoff to human (both routes). Always paste this command's exact stdout in
the final prompt/chat message. `<target>` is the preset used in step 3
(`macbook` for a laptop, the site name on HPC); it remains an accepted argument
for caller compatibility, while the printed usage is target-neutral:

```bash
apps/submission_scripts/celephais_submission_usage.sh <target> <external-plugin-path> | tee build-logs/<target>-usage.txt
```

When the second argument is present, the helper prints the private-worker
location and external-plugin path immediately after the banner. `compile.sh`
passes the installed `CelephaisTargets.cmake` path automatically.

## Expected Use

Run from `apps/<APP>/sbatch_outputs/` after build:

```bash
./sub_Celephais.sh input.toml
./sub_Celephais.sh solve input.toml ./output_dir/
./sub_Celephais.sh 2sacra input.toml ./exported_levels/
```

One arg means the public `bin/celephais` controller. Two or more args mean
`program-or-path`, `input.toml`, then solver args. On Slurm login,
`sub_Celephais.sh -> sbatch <site>.sh -> celephais_sbatch_entry.sh -> celephais_run_solve.sh`.
On direct machines or inside an allocation,
`sub_Celephais.sh -> celephais_run_solve.sh`.

**OpenMPI under `srun` needs an explicit PMI (not machine-specific).** Inside a
Slurm allocation `celephais_run_solve.sh` launches the solver with `srun`. If the
job aborts immediately at `MPI_Init` with `OPAL ERROR: Unreachable in
ext3x_client.c` / "OMPI was not built with SLURM's PMI support", the cluster's
`MpiDefault` is unset (`scontrol show config | grep MpiDefault`) so a bare
`srun` negotiates no PMI. Fix in the site env: `export SLURM_MPI_TYPE=pmix`
(run `srun --mpi=list` to confirm; fall back to `pmi2` if `pmix` is absent).
OpenMPI must be built `--with-pmix`/`--with-slurm` (`ompi_info | grep -i pmix`).

## Failure Reply

Use this shape:

```text
Status: blocked at <BLOCKER>.
Evidence: <short command/log fact>.
Need: <exact NEXT or one human datum>.
```
