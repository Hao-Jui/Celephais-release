# BundledMetis.cmake -- build the in-tree METIS 5.1.0 (third_party/metis) from
# source so the bundled MUMPS can use METIS fill-reducing ordering (recovering
# the factor performance the PORD-only build gave up).
#
# METIS 5.1.0 is Apache-2.0 licensed (third_party/metis/LICENSE.txt) and
# self-contained: GKlib is bundled and compiled into libmetis.a. As with the
# bundled MUMPS, the vendor tree is copied into the build directory and built
# there so the committed source stays pristine and parallel per-job builds never
# share object files.
#
# The copied include/metis.h is patched to IDXTYPEWIDTH=64 (patch_metis_idx64.cmake)
# so libmetis.a uses a 64-bit idx_t. This lets the bundled MUMPS analysis phase
# order matrices with more than 2^31 nonzeros, fixing MUMPS INFOG(1)=-51 ("integer
# overflow in ordering") at res15 scale (nnz ~ 2e9). It widens the ORDERING only:
# MUMPS_INT stays 32-bit (matrix order N never nears 2^31; NNZ is already 64-bit in
# MUMPS's default mixed-mode) and BLAS/LAPACK/ScaLAPACK/MPI stay LP64 -- the
# factorization never sees a METIS idx_t. REALTYPEWIDTH stays 32.
#
# METIS ships an ancient `cmake_minimum_required(VERSION 2.8)` that modern CMake
# rejects; -DCMAKE_POLICY_VERSION_MINIMUM=3.5 is the supported escape hatch. We
# build only the static `metis` target (not the gpmetis/ndmetis programs).

include(ExternalProject)

set(METIS_VENDOR_DIR "${PROJECT_SOURCE_DIR}/third_party/metis")
if(NOT EXISTS "${METIS_VENDOR_DIR}/include/metis.h")
    message(FATAL_ERROR
        "In-tree METIS source not found at ${METIS_VENDOR_DIR}. The bundled MUMPS "
        "uses METIS ordering; restore third_party/metis.")
endif()

set(METIS_BUILD_DIR  "${CMAKE_BINARY_DIR}/metis")        # copy of the vendor tree
set(METIS_CMAKE_DIR  "${METIS_BUILD_DIR}/_build")         # out-of-source METIS build
set(METIS_INCLUDE_DIR "${METIS_BUILD_DIR}/include")
set(METIS_LIBRARY    "${METIS_CMAKE_DIR}/libmetis/libmetis.a")

# METIS is plain C (no MPI); use the MPI C wrapper when available for toolchain
# consistency, else the project C compiler.
set(_METIS_CC "${MPI_C_COMPILER}")
if(NOT _METIS_CC)
    set(_METIS_CC "${CMAKE_C_COMPILER}")
endif()

# Configure args for METIS's own CMake. Mirror the MUMPS build's archiver
# handling (honour CMAKE_AR/CMAKE_RANLIB for cross/Cray toolchains; harmless when
# unset). The -DCMAKE_POLICY_VERSION_MINIMUM=3.5 escape hatch is REQUIRED on
# CMake >= 4.0 (METIS ships cmake_minimum_required(VERSION 2.8), a fatal error
# there) and is an unrecognized/unused var on CMake 3.25-3.30, so gate it on 4.0
# to avoid a spurious "unused variable" warning on older HPC CMake.
set(_METIS_CFG_ARGS
    -G "${CMAKE_GENERATOR}"
    "-DGKLIB_PATH=${METIS_BUILD_DIR}/GKlib"
    "-DSHARED=0"
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
    "-DCMAKE_C_COMPILER=${_METIS_CC}"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
# ASan instrumentation: inject if the parent build is using ASan so that
# METIS heap allocations are tracked and OOB writes inside GKlib are caught
# at the write site rather than deferred to gk_free / Darwin xzm.
if(CMAKE_C_FLAGS MATCHES "-fsanitize=address" OR
   CMAKE_C_FLAGS_RELWITHDEBINFO MATCHES "-fsanitize=address")
    list(APPEND _METIS_CFG_ARGS
        "-DCMAKE_C_FLAGS=-fsanitize=address -fno-omit-frame-pointer -g")
    message(STATUS "METIS: injecting -fsanitize=address for heap-OOB localisation")
endif()
if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
    list(APPEND _METIS_CFG_ARGS "-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
endif()
if(CMAKE_AR)
    list(APPEND _METIS_CFG_ARGS "-DCMAKE_AR=${CMAKE_AR}")
endif()
if(CMAKE_RANLIB)
    list(APPEND _METIS_CFG_ARGS "-DCMAKE_RANLIB=${CMAKE_RANLIB}")
endif()

# METIS index width. 64-bit by default (orders >2^31 nonzeros at high resolution).
# On macOS the 64-bit GKlib path corrupts the heap during nested-dissection
# refinement and Darwin's hardened allocator traps it (SIGTRAP in gk_free); Linux
# glibc silently tolerates the same bug. Fall back to 32-bit on Apple in normal
# builds. KADATH_METIS_FORCE_IDX64 overrides this for ASan root-cause builds.
set(_METIS_IDXWIDTH 64)
if(APPLE AND NOT KADATH_METIS_FORCE_IDX64)
    set(_METIS_IDXWIDTH 32)
    message(STATUS "METIS: IDXTYPEWIDTH=32 on Apple (GKlib 64-bit heap trap under Darwin hardened malloc)")
elseif(APPLE AND KADATH_METIS_FORCE_IDX64)
    message(STATUS "METIS: IDXTYPEWIDTH=64 forced on Apple (KADATH_METIS_FORCE_IDX64=ON) -- ASan debug build")
endif()

ExternalProject_Add(metis_bundled
    SOURCE_DIR        "${METIS_BUILD_DIR}"
    DOWNLOAD_COMMAND  "${CMAKE_COMMAND}" -E copy_directory
                      "${METIS_VENDOR_DIR}" "${METIS_BUILD_DIR}"
    PATCH_COMMAND     "${CMAKE_COMMAND}" "-DMETIS_HEADER=${METIS_INCLUDE_DIR}/metis.h"
                      "-DMETIS_IDXWIDTH=${_METIS_IDXWIDTH}"
                      -P "${PROJECT_SOURCE_DIR}/cmake/patch_metis_idx64.cmake"
    CONFIGURE_COMMAND "${CMAKE_COMMAND}" -S "${METIS_BUILD_DIR}" -B "${METIS_CMAKE_DIR}"
                      ${_METIS_CFG_ARGS}
    BUILD_COMMAND     "${CMAKE_COMMAND}" --build "${METIS_CMAKE_DIR}" --target metis
    BUILD_IN_SOURCE   0
    INSTALL_COMMAND   ""
    BUILD_BYPRODUCTS  "${METIS_LIBRARY}"
    LOG_DOWNLOAD      1
    LOG_CONFIGURE     1
    LOG_BUILD         1)

message(STATUS "METIS provider: bundled (third_party/metis 5.1.0, Apache-2.0, built in ${METIS_CMAKE_DIR})")
