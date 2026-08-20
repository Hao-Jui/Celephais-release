cmake_minimum_required(VERSION 3.25)
project(Celephais)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Canonical repository root used by the Celephais library and applications.
set(CELEPHAIS_ROOT_DIR ${PROJECT_SOURCE_DIR} CACHE PATH "Celephais repository root" FORCE)

# Compile-time paths baked into celephais_paths.h (env var takes precedence at runtime)
set(CELEPHAIS_DATA_DIR "${PROJECT_SOURCE_DIR}/data")
configure_file(cmake/celephais_paths.h.in ${CMAKE_BINARY_DIR}/generated/celephais_paths.h @ONLY)

# Default to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

cmake_policy(SET CMP0109 NEW)
if(POLICY CMP0156)
    cmake_policy(SET CMP0156 NEW)
endif()
if(POLICY CMP0179)
    cmake_policy(SET CMP0179 NEW)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(CELEPHAIS_SUPPRESS_COMPILE_COMMANDS_LINK
    "Do not repoint the source-tree compile_commands.json link (integration harnesses)"
    OFF)

# Symlink compile_commands.json to repo root so clangd / LLM editors find it
# without being launched from the build dir.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND
   NOT CELEPHAIS_SUPPRESS_COMPILE_COMMANDS_LINK)
    file(CREATE_LINK
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_SOURCE_DIR}/compile_commands.json"
        SYMBOLIC
    )
endif()

# Optional compiler cache to speed up rebuilds.
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# ── Build options ───────────────────────────────────────────────────────────
option(BUILD_APPS "Build Celephais physics application executables" OFF)
option(BUILD_TESTING "Build test suite" OFF)
option(CELEPHAIS_USE_PCH "Precompile hot standard-library/MPI headers for the solution archive" OFF)
option(CELEPHAIS_ENABLE_FFTW_ORACLE
    "Enable the test-only FFTW differential oracle (never required by production packages)"
    OFF)


set(CELEPHAIS_ARCH "" CACHE STRING
    "Release target ISA passed to -march (for example native, x86-64-v3, or skylake-avx512)")
set(CELEPHAIS_CPU "" CACHE STRING
    "Release target CPU passed to -mcpu (for example native or apple-m4)")

if(NOT CELEPHAIS_ARCH STREQUAL "" AND NOT CELEPHAIS_CPU STREQUAL "")
    message(FATAL_ERROR "Set only one of CELEPHAIS_ARCH and CELEPHAIS_CPU")
endif()

set(CELEPHAIS_TARGET_CLASS "unknown" CACHE STRING "Target launcher class: direct, slurm, or unknown")
set_property(CACHE CELEPHAIS_TARGET_CLASS PROPERTY STRINGS direct slurm unknown)
set(_CELEPHAIS_TARGET_CLASSES direct slurm unknown)
if(NOT CELEPHAIS_TARGET_CLASS IN_LIST _CELEPHAIS_TARGET_CLASSES)
    message(FATAL_ERROR "CELEPHAIS_TARGET_CLASS must be direct, slurm, or unknown; got '${CELEPHAIS_TARGET_CLASS}'")
endif()

# ── Dependencies ────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

enable_language(Fortran)
find_package(MPI REQUIRED COMPONENTS CXX Fortran)
if(CELEPHAIS_ENABLE_FFTW_ORACLE)
    find_package(FFTW REQUIRED)
endif()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(SCALAPACK QUIET)

# MUMPS is built from the self-contained in-tree source (third_party/mumps) as
# part of the Celephais build — there is no system/PETSc MUMPS to discover. The
# static archives do not exist until build time, so there is no configure-time
# link probe; the Celephais/app links are the link test.
include(BundledMUMPS)

set(SPARSE_SOLVER_FOUND TRUE)
message(STATUS "MUMPS sparse solver enabled")

include(CelephaisStackReport)
celephais_print_stack_report()
include(CelephaisRuntimePathCheck)
celephais_warn_missing_runtime_paths()

# ── Celephais spectral library ─────────────────────────────────────────────────
file(GLOB_RECURSE CELEPHAIS_SOURCES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)
# The Chebyshev derivative kernels expose their current Release rounding order
# as a byte-level compatibility contract. Disable implicit contraction in the
# two translation units implementing that contract; intended fused steps use
# std::fma explicitly, so Debug and Release execute the same recurrence.
if(CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang|GNU)$")
    set_source_files_properties(
        ${PROJECT_SOURCE_DIR}/src/Ope_1d/der_1d_cheb_contract.cpp
        ${PROJECT_SOURCE_DIR}/src/Ope_1d/ope_1d.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off"
    )
else()
    message(FATAL_ERROR
        "The exact Chebyshev derivative contract requires -ffp-contract=off support")
endif()
# Margherita EOS implementation lives under include/Hydro (vendored), so it is
# not caught by the src/*.cpp glob above and must be listed explicitly.
list(APPEND CELEPHAIS_SOURCES ${PROJECT_SOURCE_DIR}/include/Hydro/Margherita/PWP/setup_polytrope.cc)
list(APPEND CELEPHAIS_SOURCES ${PROJECT_SOURCE_DIR}/include/Hydro/Margherita/Table/setup_cold_table.cc)

add_library(celephais STATIC ${CELEPHAIS_SOURCES})
add_library(Celephais::solution ALIAS celephais)
set(CELEPHAIS_LIB_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/lib" CACHE PATH
    "Output directory for libcelephais.a; override for parallel-safe per-job builds")
set_target_properties(celephais PROPERTIES
    OUTPUT_NAME celephais
    EXPORT_NAME solution
    ARCHIVE_OUTPUT_DIRECTORY ${CELEPHAIS_LIB_OUTPUT_DIR}
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(celephais
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_BINARY_DIR}/generated
        ${MUMPS_INCLUDE_DIRS}
)
if(CELEPHAIS_ENABLE_FFTW_ORACLE)
    target_include_directories(celephais PRIVATE ${FFTW_INCLUDES})
endif()

# System_of_eqs keeps two private implementation headers (coloring_internals.hpp,
# equation_parser.hpp) at the directory root. Its .cpp files are grouped into
# assembly/ coloring/ parsing/ diagnostics/ subfolders and quote-include those
# headers, so expose the directory PRIVATEly to resolve the includes across
# subfolders without leaking the internal headers to Celephais consumers.
target_include_directories(celephais PRIVATE
    ${PROJECT_SOURCE_DIR}/src/System_of_eqs
)

target_compile_features(celephais PUBLIC cxx_std_20)

# Precompiled headers (opt-in). Covers libc++ and MPI hot paths, plus FFTW only
# in an explicit differential-oracle build.
# Frontend dominates Celephais cold builds (~79% in -ftime-trace probe). Pre-baking
# these collapses ~0.5s/TU shared header cost.
if(CELEPHAIS_USE_PCH)
    set(CELEPHAIS_PCH_HEADERS
        <vector>
        <memory>
        <array>
        <complex>
        <cmath>
        <string>
        <map>
        <unordered_map>
        <algorithm>
        <iostream>
        <fstream>
        <sstream>
        <cassert>
        <cstdint>
        <cstdlib>
        <cstring>
        <mpi.h>
    )
    if(CELEPHAIS_ENABLE_FFTW_ORACLE)
        list(APPEND CELEPHAIS_PCH_HEADERS <fftw3.h>)
    endif()
    target_precompile_headers(celephais PRIVATE ${CELEPHAIS_PCH_HEADERS})
endif()

# Compiler warnings — suppress -Wunused-parameter (19K+ from virtual method hierarchy)
target_compile_options(celephais PRIVATE
    -Wall -Wextra
    -Wno-unused-parameter
    -Woverloaded-virtual
)
# Clang-specific: GCC covers this via -Wall's -Wdelete-non-virtual-dtor
target_compile_options(celephais PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wdelete-non-abstract-non-virtual-dtor>
)

# The installed solution reader deliberately excludes the nonlinear-solver
# tail. Current domain vtables require MPI C++ and LAPACK/BLAS, but loading and
# evaluating a checkpoint does not require FFTW, MUMPS, ScaLAPACK, an MPI
# Fortran binding, or its compiler runtime.
target_compile_definitions(celephais PRIVATE CELEPHAIS_USE_MUMPS)
target_link_libraries(celephais PUBLIC
    MPI::MPI_CXX
    LAPACK::LAPACK
    BLAS::BLAS
)
if(CELEPHAIS_ENABLE_FFTW_ORACLE)
    set_source_files_properties(
        ${PROJECT_SOURCE_DIR}/src/Coef/native_r2hc.cpp
        PROPERTIES COMPILE_DEFINITIONS CELEPHAIS_ENABLE_FFTW_ORACLE)
    target_link_libraries(celephais PRIVATE FFTW::FFTW)
endif()
# The in-tree MUMPS generates its headers (mumps_int_def.h) at build time, so
# the implementation archive must wait before compiling its MUMPS-using
# sources, even though the installed solution-reader interface omits MUMPS.
add_dependencies(celephais mumps_bundled)

# Existing applications and tests need the full sparse-solver closure. Keep the
# solver target as a build-tree-only interface; only celephais is
# exported and installed, under the public name Celephais::solution.
add_library(celephais_solver INTERFACE)
target_compile_definitions(celephais_solver INTERFACE CELEPHAIS_USE_MUMPS)
target_include_directories(celephais_solver INTERFACE
    "${CMAKE_BINARY_DIR}/generated"
)
target_link_libraries(celephais_solver INTERFACE
    celephais
    MUMPS::MUMPS
)

set(CELEPHAIS_SPARSE_LIBRARIES ${MUMPS_LIBRARIES} CACHE INTERNAL "Sparse solver libraries")
set(CELEPHAIS_EXTRA_LIBS ${CELEPHAIS_SPARSE_LIBRARIES} CACHE STRING "Extra libraries needed by Celephais" FORCE)

# Release flags.
#
# -march=native bakes the BUILD host's exact instruction set into the binary, so
# the binary SIGILLs (illegal instruction) if a run node has a poorer ISA than
# the build node. That is safe when build==run (a workstation), but unsafe on
# HPC clusters whose login/build node and compute node — or whose compute
# partition — differ in microarchitecture. Pick the arch flag by deployment:
#   -DCELEPHAIS_CPU=<cpu>   -> -mcpu=<cpu>     (cache-native CPU choice)
#   -DCELEPHAIS_ARCH=<arch> -> -march=<arch>   (cache-native ISA choice)
#   CELEPHAIS_ARCH=<arch>   -> -march=<arch>   (environment choice)
# Use an explicit architecture only when every deployment node supports it.
#   -DCELEPHAIS_TARGET_CLASS=direct (build==run) -> -march=native
# On the measured Apple M4 build, Apple Clang 21 maps -march=native to apple-m1,
# but that target is faster for the Jacobian critical path than its apple-m4
# -mcpu=native model. CELEPHAIS_CPU remains available for explicit CPU A/Bs.
#   otherwise (slurm / unknown) -> portable baseline ISA, -mtune=native for
#                                  host-tuned scheduling without restricting the
#                                  instruction set.
if (NOT CELEPHAIS_CPU STREQUAL "")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -mcpu=${CELEPHAIS_CPU}")
elseif (NOT CELEPHAIS_ARCH STREQUAL "")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -march=${CELEPHAIS_ARCH}")
elseif (NOT "$ENV{CELEPHAIS_ARCH}" STREQUAL "")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -march=$ENV{CELEPHAIS_ARCH}")
elseif (CELEPHAIS_TARGET_CLASS STREQUAL "direct")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -march=native")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -mtune=native")
endif()
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -DNDEBUG")

# Debug-only Array Index validation (dimension-match + per-axis bounds). When
# on it is the hottest leaf in the Jacobian assembly (~12% of the first-J
# build), so it is gated OFF in optimized builds. Kept on its own macro rather
# than NDEBUG so it controls only the array checks, not asserts; asan/ubsan
# (RelWithDebInfo) still catches the downstream out-of-bounds access.
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -DCELEPHAIS_ARRAY_BOUNDS_CHECK")

if(BUILD_TESTING)
    enable_testing()
endif()

# ── Physics applications (opt-in) ──────────────────────────────────────────
if(BUILD_APPS)
    add_subdirectory(apps)
endif()

# ── Testing (opt-in) ──────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# ── Installed solution-reader package ──────────────────────────────────────
set(CELEPHAIS_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/Celephais")

configure_package_config_file(
    cmake/CelephaisConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/CelephaisConfig.cmake"
    INSTALL_DESTINATION "${CELEPHAIS_INSTALL_CMAKEDIR}"
)

install(TARGETS celephais
    EXPORT CelephaisTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(DIRECTORY include/Celephais
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(EXPORT CelephaisTargets
    FILE CelephaisTargets.cmake
    NAMESPACE Celephais::
    DESTINATION "${CELEPHAIS_INSTALL_CMAKEDIR}"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CelephaisConfig.cmake"
    DESTINATION "${CELEPHAIS_INSTALL_CMAKEDIR}"
)
if(CELEPHAIS_ENABLE_FFTW_ORACLE)
    install(FILES cmake/FindFFTW.cmake
        DESTINATION "${CELEPHAIS_INSTALL_CMAKEDIR}")
endif()
install(FILES LICENSE THIRD_PARTY_NOTICES.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/Celephais"
)

# ── API documentation (Doxygen, opt-in target) ─────────────────────────────
# Run `cmake --build build --target docs-api` to render HTML into Doxygen/api/.
# Target is created only when Doxygen is installed; missing tool is non-fatal.
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Doxygen/api)
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_LATEX NO)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
    set(DOXYGEN_WARN_LOGFILE ${PROJECT_SOURCE_DIR}/Doxygen/api/doxygen-warnings.log)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_EXTRACT_PRIVATE NO)
    set(DOXYGEN_EXTRACT_STATIC YES)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_EXCLUDE_PATTERNS
        "*/flat_hash_map.hpp"
        "*/SpECTRE/*"
        "*/Elliptica/*"
        "*/build*/*"
        "*/Testing/*"
        "*/.git/*"
        "*/__pycache__/*")
    set(DOXYGEN_USE_MATHJAX YES)
    set(DOXYGEN_HAVE_DOT NO)
    set(DOXYGEN_PROJECT_NAME "Celephais")
    set(DOXYGEN_PROJECT_BRIEF "Spectral methods library for numerical relativity")
    set(DOXYGEN_FULL_PATH_NAMES NO)
    set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
    set(DOXYGEN_QT_AUTOBRIEF YES)
    set(DOXYGEN_MULTILINE_CPP_IS_BRIEF YES)
    set(DOXYGEN_SORT_MEMBER_DOCS NO)
    set(DOXYGEN_SOURCE_BROWSER YES)
    set(DOXYGEN_REFERENCED_BY_RELATION YES)
    set(DOXYGEN_REFERENCES_RELATION YES)

    # ALIASES neutralise stray LaTeX commands that appear unwrapped in
    # legacy comments (e.g. `\d`, `\theta`). They render as literal text
    # instead of emitting "Found unknown command" warnings. Real LaTeX
    # blocks should still use \f$ ... \f$ / \f[ ... \f].
    set(DOXYGEN_ALIASES
        [[d=\\d]]
        [[theta=\\theta]]
        [[varphi=\\varphi]]
        [[gamma_ij=\\gamma_{ij}]]
        [[eta_ij=\\eta_{ij}]]
        [[C=\\C]]
        [[cIndex=\\cIndex]])

    doxygen_add_docs(docs-api
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/Newton
        ${PROJECT_SOURCE_DIR}/Doxygen/api-groups.dox
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Generating Doxygen API reference into Doxygen/api/"
    )
endif()
