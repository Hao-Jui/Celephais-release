# Kadath AEI MUMPS Patch Ledger

Vendor baseline:

- Package: MUMPS 5.9.0
- Source archive: `https://mumps-solver.org/MUMPS_5.9.0.tar.gz`
- SHA256:
  `02c6efdb91749ec0f82351d40f3f860547272a1eb1d899126a4265b4d6bcc4ca`
- Imported into: `third_party/mumps`

## Local Changes

No Kadath-local edits have been made to any **retained** source/header/license
file — every file we ship is byte-for-byte upstream (the SHA256 above is of the
full upstream tarball). The drop has been trimmed to the double-precision subset
Kadath builds:

- 2026-06-13 — Removed the single/complex/complex-double arithmetic source files
  (`src/[scz]*.{F,c,h}`, 306 files; the `s`-leading arith-independent commons
  `sol_common.F`, `sol_ds_common_m.F`, `sol_omp_common_m.F` were KEPT) plus the
  unused `MATLAB/`, `SCILAB/`, `examples/` interfaces and `doc/userguide_5.9.0.pdf`.
  Kadath builds only `ARITH=d` (double) + the arithmetic-independent common
  objects + PORD, so the removed files are never compiled. CeCILL-C (and the BSD-3
  AMD/LAPACK and PORD components) all permit redistributing a modified/partial
  copy provided license texts and per-file notices are retained — they are
  (`LICENSE`, `doc/CeCILL-C_V1-*.txt`, `PORD/README`, and all surviving file
  headers are intact). Kadath-specific size trim (24 MB → ~7.8 MB); nothing to
  propose upstream. Verified: clean `make ARITH=d` build of
  `libdmumps.a`/`libmumps_common.a`/`libpord.a` and full Kadath link/run.

Future changes to files under `third_party/mumps` must be recorded here with:

- Date
- Changed files
- Reason for the change
- Whether the change is Kadath-specific or should be proposed upstream
- Verification performed

Build-system integration outside `third_party/mumps` should be documented in the
Kadath AEI build files or solver documentation, not as a MUMPS source patch.

## Build-system integration (no source patches)

This vendor tree is consumed without modification by `cmake/BundledMUMPS.cmake`,
which copies it into the build directory, generates a `Makefile.inc` (from
`cmake/mumps_Makefile.inc.in`: double arithmetic, PORD ordering plus METIS
ordering, real MPI, no OpenMP, static archives), and drives MUMPS's own
Makefiles to build `libdmumps.a` / `libmumps_common.a` / `libpord.a`. MUMPS
uses both PORD (bundled within MUMPS) and METIS (from the in-tree
`third_party/metis`; see `third_party/metis/PATCHES.md`) for fill-reducing
ordering. MUMPS is built this way unconditionally on every target — there is no
system/PETSc-MUMPS discovery path (the legacy `find_package(MUMPS)` route has
been removed). The copy keeps this committed tree byte-for-byte upstream — no
build artifacts are written here.
