# Third-Party Notices

Celephais is distributed under the **GNU General Public License, version 3 or
later** (see [`LICENSE`](LICENSE)). It bundles the third-party components listed
below in-tree and builds them from source as part of the normal build. Each
component is redistributed under its own license; the full license texts are
located as noted per component. The combined work is lawfully redistributable
under GPLv3-or-later — see the **GPLv3 Compatibility** section at the end for the
per-license reasoning.

No Kadath-local **source** edits are applied to the vendored trees
(`third_party/mumps`, `third_party/metis`); each is byte-for-byte upstream.
Both differ from upstream only by file *removals* (size trims) recorded in the
respective `PATCHES.md`. All retained per-file copyright/license headers are
intact.

Kadath/FUKa/Phillipe-derived source under `include/` and `src/` is not treated
as an untouched vendored tree. Its provenance audit and patch notice are tracked
separately in `LICENSE_SOURCE_AUDIT.tsv` and
`PATCHES-KADATH-UPSTREAM.md`. Those files identify modified derivatives whose
upstream headers must be preserved or restored before adding dated Celephais
modification notices, including the 2026-06-17 follow-up rows for the
`Space_spheric_homothetic` NS NOROT-stage space and the
`include/For_Kadath/Domain/adapted_polar.hpp` Newton line-search
snapshot/restore overrides.

---

## MUMPS 5.9.0 — `third_party/mumps`

MUltifrontal Massively Parallel Solver; the production sparse direct linear
solver in Celephais.

- **Source archive:** `https://mumps-solver.org/MUMPS_5.9.0.tar.gz`
- **Archive SHA256:**
  `02c6efdb91749ec0f82351d40f3f860547272a1eb1d899126a4265b4d6bcc4ca`
- **Copyright:** 1991–2026 CERFACS, CNRS, ENS Lyon, INP Toulouse, Inria,
  Mumps Technologies, University of Bordeaux.

### License (primary): CeCILL-C v1

MUMPS is released under the **CeCILL-C Free Software License Agreement v1**
(SPDX: `CECILL-C`). The full license texts are bundled:

- English: `third_party/mumps/doc/CeCILL-C_V1-en.txt`
- French: `third_party/mumps/doc/CeCILL-C_V1-fr.txt`
- Online: `https://cecill.info/licences/Licence_CeCILL-C_V1-en.html`

The top-level notice is `third_party/mumps/LICENSE`; per-file copyright/license
notices are retained in the source headers. CeCILL-C is a weak-copyleft
(LGPL-analog) license: its copyleft is confined to the MUMPS files, and its
Article 5.3.3 expressly permits the surrounding derivative work (Kadath) to be
distributed under another license (here GPLv3-or-later) provided the MUMPS
portion remains under CeCILL-C, its full corresponding source stays available,
and the Article 6.4 rights notice is preserved. **The CeCILL-C-licensed MUMPS
files are NOT relicensed under the GPL**; only the combined work as a whole is
distributed under GPLv3-or-later.

### Embedded carve-outs (per the MUMPS `LICENSE`)

- **BSD-3-Clause** — variants of **AMD ordering** (©1996–2016 Timothy A. Davis,
  Patrick R. Amestoy, Iain S. Duff; **including the MUMPS-modified AMD variant
  routines**, e.g. `MUMPS_AMD_ELT`/`HAMD`/`HAMF4`/`QAMD`/`CST_AMF`) in
  `third_party/mumps/src/ana_orderings.F` (full BSD-3 text in the header, UC
  Berkeley clause), and **`[sdcz]MUMPS_TRUNCATED_RRQR`** (rank-revealing QR
  derived from LAPACK; ©U. Tennessee / U. California Berkeley / U. Colorado
  Denver, 1992–2017) in `third_party/mumps/src/dlr_core.F` (full BSD-3 text in
  the routine header). The BSD-3 copyright notices, the list of conditions, and
  the disclaimers are reproduced in those source headers.
- **PORD fill-reducing ordering** (optional) — **public domain**. Extracted from
  the SPACE-1.0 package (Juergen Schulze, University of Paderborn; partly based
  on SPOOLES by Cleve Ashcraft). `third_party/mumps/PORD/README` is the sole and
  sufficient license notice and states SPACE-1.0 (including PORD) is *"totally
  within the public domain; there are absolutely no licensing restrictions."*
  The PORD/Schulze authorship acknowledgement is preserved in `PORD/README`.

### Trimmed-to-double subset

The drop is trimmed to the double-precision subset Kadath builds: the
single/complex/complex-double arithmetic sources (`src/[scz]*.{F,c,h}`, 306
files) and the unused `MATLAB/`, `SCILAB/`, `examples/` interfaces and
`doc/userguide_5.9.0.pdf` were removed. The three `s`-leading
arithmetic-independent commons (`sol_common.F`, `sol_ds_common_m.F`,
`sol_omp_common_m.F`) were **kept**. CeCILL-C (and the BSD-3 AMD/LAPACK and
public-domain PORD components) all permit redistributing a modified/partial copy
provided all license texts and per-file notices are retained — they are. Details:
`third_party/mumps/PATCHES.md`.

### Acknowledgment (citation request)

The MUMPS `LICENSE` asks users to acknowledge the package in scientific
publications that depend on it, using:

- [1] P. R. Amestoy, I. S. Duff, J. Koster, J.-Y. L'Excellent,
  *SIAM J. Matrix Anal. Appl.* **23**(1):15–41 (2001).
- [2] P. R. Amestoy, A. Buttari, J.-Y. L'Excellent, T. Mary,
  *ACM Trans. Math. Software* **45**(1):2:1–2:26 (2019).

Users are also asked to use reasonable endeavours to notify the authors. The
upstream author, contributor, and funding acknowledgments are preserved in
`third_party/mumps/CREDITS`.

---

## METIS 5.1.0 (incl. bundled GKlib) — `third_party/metis`

Graph partitioning / fill-reducing ordering, used by MUMPS for METIS ordering.

- **Source archive:**
  `https://karypis.github.io/glaros/files/sw/metis/metis-5.1.0.tar.gz`
  (mirror: `https://ftp.mcs.anl.gov/pub/pdetools/spack-pkgs/metis-5.1.0.tar.gz`)
- **Archive SHA256:**
  `76faebe03f6c963127dbb73c13eab58c9a3faeae48779f049066a21c087c5db2`

### METIS core + GKlib core: Apache License 2.0

- **Copyright:** 1995–2013 Regents of the University of Minnesota.
- Short notice: `third_party/metis/LICENSE.txt`. **Full Apache-2.0 text:**
  `third_party/LICENSES/Apache-2.0.txt` (complete, 202 lines), satisfying
  Apache-2.0 §4(a).
- METIS source files are byte-for-byte upstream (only the `graphs/` sample-data
  directory was removed; see `third_party/metis/PATCHES.md`), so no §4(b)
  change-notice obligation is triggered. METIS ships **no `NOTICE` file**
  (verified), so Apache-2.0 §4(d) imposes nothing to propagate.
- The GKlib core files (e.g. `GKlib/GKlib.h`, `GKlib/memory.c`, `csr.c`,
  `graph.c`) carry **no per-file SPDX/license header**; they are Apache-2.0 by
  virtue of the top-level `third_party/metis/LICENSE.txt` blanket notice
  (Apache-2.0 permits a top-level notice).

### GKlib is a license MIX (all compiled into `libmetis.a`)

- **`GKlib/gkregex.c`, `GKlib/gkregex.h`** — glibc extended-regex routines
  (©2002, 2003, 2005 Free Software Foundation, Inc.; contributed by Isamu
  Hasegawa), **LGPL-2.1-or-later**. The GNU LGPL v2.1-or-later permission
  notice is retained in the file headers; the full LGPL-2.1 license text is
  provided at `third_party/LICENSES/LGPL-2.1.txt`. LGPL-2.1 §6
  (relinking) is satisfied in substance: the complete corresponding source of
  these files is shipped in-tree and METIS is built entirely from source via
  `cmake/BundledMetis.cmake`, so a recipient can modify `gkregex` and relink
  `libmetis.a`.
- **`GKlib/random.c`** — Mersenne Twister MT19937-64 (©2004 Makoto Matsumoto &
  Takuji Nishimura), **BSD-3-Clause**. *Note:* the in-tree file reproduces only
  the copyright line and the warranty-disclaimer paragraph; the three enumerated
  BSD-3 redistribution conditions are **not present in the file**. This is the
  upstream GKlib state (the file is byte-for-byte upstream). The full, identical
  3-clause BSD conditions text (no advertising clause) is reproduced in-tree in
  the sibling GKlib file `third_party/metis/GKlib/ms_inttypes.h` and applies
  equally to `random.c`. BSD-3 is GPL-compatible and the combination is lawful.
- **`GKlib/ms_inttypes.h`, `GKlib/ms_stdint.h`** — MSVC ISO C9x compatibility
  shims (©2006 Alexander Chemeris), **BSD-3-Clause** (full 3-clause text incl.
  no-endorsement clause present in the headers). Windows-only; inert on the
  Kadath macOS/Linux build.

- Integer/real width: `IDXTYPEWIDTH=32`, `REALTYPEWIDTH=32` (matches the MUMPS
  32-bit `MUMPS_INT`). Provenance + build integration:
  `third_party/metis/PATCHES.md`.

---

## Bundled single-header libraries — `include/For_Kadath/Third_Party/`

### `flat_hash_map.hpp` — Boost Software License 1.0

- `ska::flat_hash_map` / `bytell_hash_map`, **Copyright Malte Skarupke 2017**,
  distributed under the **Boost Software License, Version 1.0** (BSL-1.0).
- The header reproduces the 3-line Boost copyright/permission notice and the
  `http://www.boost.org/LICENSE_1_0.txt` URL. BSL-1.0 requires the **full**
  license text be reproduced in source distributions; the canonical full text is
  provided at **`third_party/LICENSES/BSL-1.0.txt`** and applies to this
  component.

### `Tomlplusplus/include/toml.hpp` — MIT License

- **toml++ v3.4.0** (a TOML parser), **Copyright (c) Mark Gillard
  `<mark.gillard@outlook.com.au>`**, **MIT License**. SPDX
  (`SPDX-License-Identifier: MIT`) is present at `toml.hpp` line 5; the full MIT
  permission notice is reproduced inline in `toml.hpp` and the complete MIT text
  ships at `include/For_Kadath/Third_Party/Tomlplusplus/LICENSE` (with
  `Tomlplusplus/README.md`).
- **Embedded sub-component:** a UTF-8 DFA decoder (**Copyright (c) 2008-2009
  Bjoern Hoehrmann `<bjoern@hoehrmann.de>`**, MIT) is incorporated inside
  `toml.hpp` (`struct utf8_decoder`, ~line 8954) under toml++'s MIT terms with
  its copyright line retained inline.

---

## Margherita EOS framework — `include/Hydro/Margherita/`

Light-weight equation-of-state framework, vendored in-tree and compiled/linked
into the BNS / BHNS / NS2d / ThreeBody apps via `include/Hydro/EOS.hh`.

- **License:** **GPL-3.0-or-later** (per-file headers; *"This file is part of
  Margherita, the light-weight EOS framework ... GNU General Public License ...
  version 3 ... or (at your option) any later version"*). No standalone
  `LICENSE`/`COPYING` file ships inside the subtree; the per-file headers are the
  notice and are retained.
- **Copyright:** (C) 2017 Elias Roland Most
  `<emost@th.physik.uni-frankfurt.de>` and Ludwig Jens Papenfort
  `<papenfort@th.physik.uni-frankfurt.de>` (Goethe University Frankfurt / FIAS);
  `tov.hh` (v2.0) and the `Table/setup` files also (C) Samuel David Tootle.
- **Coverage:** `margherita.hh`, `tov.hh`, `tov_mass_search.hh`, `PWP/`
  (`cold_pwpoly.hh`, `cold_pwpoly_implementation.hh`, `setup_polytrope.{cc,hh}`),
  `Table/` (`cold_table.hh`, `cold_table_implementation.hh`,
  `setup_cold_table.{cc,hh}`, `lorene_io.hh`, `Interpolation/{spline.hh,
  linear_interp.hh, inverse_interp.hh, LUP_old.cc}`).
- Because Margherita is GPL-3.0-or-later — the same license as the Kadath
  umbrella — there is **no compatibility gap**; it is acknowledged here for its
  distinct third-party copyright holders.

### `Table/Interpolation/brent.hh` — GNU LGPL (distinct sub-component)

- Brent root-finder `zero_brent`: original FORTRAN77 by **Richard Brent**, C++
  port by **John Burkardt** (transcription correction credited to Thomas
  Secretin). In-file statement: *"This code is distributed under the GNU LGPL
  license."* This file carries no Most/Papenfort copyright line; its LGPL
  attribution is a separate requirement and is retained in the file header.

---

## GPLv3 Compatibility

The combined, statically-linked Celephais work (GPL-3.0-or-later) is lawfully
redistributable with all bundled component licenses, with **no blocking
incompatibility**, provided the per-component notice/source obligations above are
met. Per-license reasoning (sources: FSF license list
`https://www.gnu.org/licenses/license-list.html`, FSF license-compatibility
`https://www.gnu.org/licenses/license-compatibility.html`, ASF
`https://www.apache.org/licenses/GPL-compatibility.html`, CeCILL FAQ
`http://www.cecill.info/faq.en.html`):

- **CeCILL-C (MUMPS core).** The FSF labels CeCILL-C *"incompatible with the GNU
  GPL"* — but that means CeCILL-C code cannot be **relicensed into** the GPL; it
  does **not** block bundling. CeCILL-C is a weak-copyleft, LGPL-analog license:
  its Article 5.3.3 expressly permits the surrounding *Derivative Software*
  (Kadath) to be distributed *"under a license agreement other than this
  Agreement"* (here GPLv3-or-later) provided (a) the MUMPS files stay under
  CeCILL-C and are not relicensed, (b) full MUMPS source remains available for
  the whole distribution period at no more than transfer cost, and (c) the
  Article 6.4 rights notice and all MUMPS IP notices are preserved. Kadath's
  vendoring model (recompiled-but-unmodified `third_party/mumps`, license docs
  shipped) satisfies this. **Operational rule:** treat MUMPS exactly like an LGPL
  dependency — a separable, source-available, separately-licensed module; never
  fold CeCILL-C source into a GPL-headered file.
  > This software incorporates MUMPS (MUltifrontal Massively Parallel Solver),
  > version 5.9.0, Copyright 1991-2026 CERFACS, CNRS, ENS Lyon, INP Toulouse,
  > Inria, Mumps Technologies, University of Bordeaux. MUMPS is licensed under
  > the CeCILL-C Free Software License Agreement v1 (see
  > `third_party/mumps/doc/CeCILL-C_V1-en.txt` and
  > `https://cecill.info/licences/Licence_CeCILL-C_V1-en.html`). MUMPS remains
  > governed by CeCILL-C; its complete corresponding source code is provided in
  > `third_party/mumps` and is available for the entire distribution period. The
  > MUMPS intellectual-property notices have been preserved unmodified. The
  > CeCILL-C-licensed portions are not relicensed under the GPL; only the
  > combined work as a whole is distributed under the GNU GPL v3-or-later as
  > permitted by CeCILL-C Article 5.3.3, with the rights notice required by
  > Article 6.4.

- **Apache-2.0 (METIS / GKlib core).** GPLv3-compatible — and **GPLv2-INcompatible**
  (Apache-2.0's patent-termination clause is a "further restriction" GPLv2 cannot
  accommodate). Kadath being GPL-3.0-**or-later** picks up the v3 leg, so the
  combination is clean. Direction is correct: Apache-2.0 code is pulled INTO the
  GPLv3 umbrella (the result is governed by GPLv3). Required Apache-2.0
  attribution/copyright/patent notices are retained; the full text ships at
  `third_party/LICENSES/Apache-2.0.txt`; METIS ships no `NOTICE` file.

- **LGPL-2.1-or-later (gkregex).** GPL-compatible: the *"or later"* election plus
  LGPL §3 permit converting the LGPL-covered code into a GPL-covered work, so it
  folds cleanly into the GPLv3+ result. LGPL §6 (relinking) is satisfied in
  substance — the complete corresponding source of `gkregex.{c,h}` is shipped and
  METIS is built from source (`cmake/BundledMetis.cmake`), so a recipient can
  modify and relink. The FSF copyright and LGPL permission notices are retained.

- **BSD-3-Clause (MUMPS AMD/LAPACK snippets; GKlib `random.c`, `ms_*.h`).** The
  3-clause ("new"/modified) BSD license — **without** the advertising clause — is
  GPL-compatible with all GPL versions (distinct from the GPL-incompatible
  4-clause "original" BSD). The MUMPS headers cite the LAPACK/AMD BSD-3 (no
  advertising clause); the GKlib BSD-3 parts likewise carry no advertising
  clause. Per-file BSD headers are retained (with the `random.c` caveat noted
  above).

- **MIT / Expat (toml++) and Boost-1.0 (flat_hash_map).** FSF-listed
  GPL-compatible permissive licenses; they impose only copyright/permission-notice
  retention, with no copyleft and no source-availability or NOTICE-file
  obligation. Notices retained; toml++ full MIT text ships at
  `Tomlplusplus/LICENSE`; the BSL-1.0 full text is provided at
  `third_party/LICENSES/BSL-1.0.txt`.

- **PORD (public domain).** SPACE-1.0/PORD carries no licensing restrictions and
  is freely incorporable into a GPLv3+ work; the `PORD/README` authorship
  acknowledgement is preserved as provenance.

- **Margherita EOS (GPL-3.0-or-later) and `brent.hh` (GNU LGPL).** Margherita is
  the same GPL-3.0-or-later license as the Kadath umbrella — no compatibility
  question. `brent.hh` under the GNU LGPL folds into the GPLv3+ work via the
  standard LGPL→GPL combination path; its Brent/Burkardt attribution is retained.

**Build-fetched, not redistributed:** Catch2 v3.7.1 is declared via CMake
`FetchContent` in `tests/unit/CMakeLists.txt` (downloaded at build/test time,
never committed to git, test-only, not linked into `libcelephais.a` or any app). It
is **not** part of the source distribution and requires no source-tree notice
entry. (Catch2 is BSL-1.0 if attribution is ever wanted in a binary test
distribution.)
