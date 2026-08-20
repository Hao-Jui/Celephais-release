# Kadath AEI METIS Patch Ledger

Vendor baseline:

- Package: METIS 5.1.0
- Source archive: `https://karypis.github.io/glaros/files/sw/metis/metis-5.1.0.tar.gz`
- Mirror: `https://ftp.mcs.anl.gov/pub/pdetools/spack-pkgs/metis-5.1.0.tar.gz`
- SHA256:
  `76faebe03f6c963127dbb73c13eab58c9a3faeae48779f049066a21c087c5db2`
- Imported into: `third_party/metis`
- License: METIS core is Apache-2.0 (`third_party/metis/LICENSE.txt`; Copyright
  1995–2013 Regents of the University of Minnesota). The bundled **GKlib is a
  license mix** (all compiled into `libmetis.a`): GKlib core Apache-2.0;
  `GKlib/gkregex.{c,h}` glibc **LGPL-2.1-or-later**; `GKlib/random.c` Mersenne
  Twister **BSD-3-Clause**; `GKlib/ms_inttypes.h`,`ms_stdint.h` (Chemeris MSVC
  shims) **BSD-3-Clause**. All permit in-tree redistribution; LGPL §6 is met by
  shipping the full corresponding source. See the repo-root `THIRD_PARTY_NOTICES.md`.
- Integer/real width: `IDXTYPEWIDTH=32`, `REALTYPEWIDTH=32` (matches
  MUMPS 32-bit `MUMPS_INT`)

## Local Changes

No Kadath-local **source** changes have been applied to this METIS vendor drop;
every source/header/CMake/license file is byte-for-byte upstream (the SHA256
above is of the full upstream tarball).

- 2026-06-13 — Removed the `graphs/` directory (~11 MB of sample input graphs
  shipped only as `gpmetis`/`ndmetis` demo data). It is not referenced by the
  METIS build or by `cmake/BundledMetis.cmake` (which builds only the `metis`
  library target), carries no license headers, and is pure repository bloat.
  Kadath-specific size trim; nothing to propose upstream. Verified: clean
  configure+build of `libmetis.a` and full MUMPS link/run unaffected.

Future changes to files under `third_party/metis` must be recorded here with:

- Date
- Changed files
- Reason for the change
- Whether the change is Kadath-specific or should be proposed upstream
- Verification performed

Build-system integration outside `third_party/metis` should be documented in the
Kadath AEI build files or solver documentation, not as a METIS source patch.

## Build-system integration (no source patches)

This vendor tree is consumed without modification by `cmake/BundledMetis.cmake`,
which copies it into the build directory and builds the static `metis` target via
METIS's own CMake. The `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` escape hatch is
passed because METIS ships `cmake_minimum_required(VERSION 2.8)`: that is a
FATAL_ERROR on CMake >= 4.0 (compatibility with CMake < 3.5 was removed in 4.0;
the escape-hatch variable was itself added in 4.0), and a non-fatal deprecation
on CMake 3.25–3.30 (where the unrecognized flag is harmlessly ignored). METIS is
built unconditionally as a dependency
of the in-tree MUMPS (wired via `cmake/BundledMUMPS.cmake`) to provide METIS
fill-reducing ordering; see `third_party/mumps/PATCHES.md`. The copy keeps this
committed tree byte-for-byte upstream — no build artifacts are written here.
