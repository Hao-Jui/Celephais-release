# BundledMUMPS.cmake -- build the in-tree MUMPS (third_party/mumps) from source
# and expose the MUMPS::MUMPS imported target.
#
# Rationale: ship a self-contained, reproducible MUMPS with the repository
# instead of discovering a system/PETSc/Homebrew build (the legacy
# find_package(MUMPS) path has been retired). The vendor tree under
# third_party/mumps stays byte-for-byte upstream (see third_party/mumps/PATCHES.md);
# it is copied into the build directory and built there so that the source stays
# pristine and parallel per-job builds never share MUMPS object files.
#
# MUMPS uses its own Makefile build system (not CMake). We generate a
# Makefile.inc from CMake-resolved compilers/flags and drive `make` for the
# double-precision arithmetic, PORD ordering, real-MPI, static-archive
# configuration that Kadath needs. ScaLAPACK/LAPACK/BLAS are NOT linked into the
# static archives (archiving only); they are appended to MUMPS::MUMPS as the
# post-MUMPS dependency tail.

include(ExternalProject)
include(ProcessorCount)

# Build the in-tree METIS first; MUMPS uses it for fill-reducing ordering. This
# defines metis_bundled + METIS_LIBRARY / METIS_INCLUDE_DIR.
include(BundledMetis)

set(MUMPS_VENDOR_DIR "${PROJECT_SOURCE_DIR}/third_party/mumps")
if(NOT EXISTS "${MUMPS_VENDOR_DIR}/src/dmumps_driver.F")
    message(FATAL_ERROR
        "In-tree MUMPS source not found at ${MUMPS_VENDOR_DIR}. "
        "Celephais builds MUMPS from this vendored source; restore third_party/mumps "
        "(e.g. via the git submodule / archive it ships in).")
endif()

# In-build copy of the vendor tree; MUMPS only supports in-source builds.
set(MUMPS_BUILD_DIR "${CMAKE_BINARY_DIR}/mumps")
set(MUMPS_BUILD_LIB "${MUMPS_BUILD_DIR}/lib")
set(MUMPS_BUILD_INC "${MUMPS_BUILD_DIR}/include")

set(_DMUMPS_A       "${MUMPS_BUILD_LIB}/libdmumps.a")
set(_MUMPS_COMMON_A "${MUMPS_BUILD_LIB}/libmumps_common.a")
set(_PORD_A         "${MUMPS_BUILD_LIB}/libpord.a")
set(_MUMPS_INT_DEF  "${MUMPS_BUILD_INC}/mumps_int_def.h")

# CMake checks that an IMPORTED target's INTERFACE_INCLUDE_DIRECTORIES exist at
# configure time, but the headers are populated during the build. Create the
# directory now (empty) so the check passes; the build fills it.
file(MAKE_DIRECTORY "${MUMPS_BUILD_INC}")

# ── Build tools ───────────────────────────────────────────────────────────────
# The generator is Ninja, so CMAKE_MAKE_PROGRAM is ninja, not make. MUMPS needs
# a POSIX/GNU make to drive its own Makefiles.
find_program(MUMPS_MAKE_PROGRAM NAMES gmake make)
if(NOT MUMPS_MAKE_PROGRAM)
    message(FATAL_ERROR "BundledMUMPS: no 'make'/'gmake' found to build third_party/mumps.")
endif()
ProcessorCount(_MUMPS_NJOBS)
if(_MUMPS_NJOBS EQUAL 0)
    set(_MUMPS_NJOBS 1)
endif()

# Prefer the MPI compiler wrappers (they inject the MPI include/lib flags MUMPS
# needs); fall back to the project compilers.
set(KADATH_MUMPS_CC "${MPI_C_COMPILER}")
if(NOT KADATH_MUMPS_CC)
    set(KADATH_MUMPS_CC "${CMAKE_C_COMPILER}")
endif()
set(KADATH_MUMPS_FC "${MPI_Fortran_COMPILER}")
if(NOT KADATH_MUMPS_FC)
    set(KADATH_MUMPS_FC "${CMAKE_Fortran_COMPILER}")
endif()

# Archiver/ranlib: honour the CMake-detected toolchain binaries when present
# (cross/Cray toolchains set CMAKE_AR/CMAKE_RANLIB to the correct wrappers);
# fall back to plain ar/ranlib otherwise. Trailing spaces are significant: the
# Makefile rules write `$(AR)$@` and `$(OUTC)$*.o`, so AR/OUTC/OUTF must separate
# options from the operand.
set(_mumps_ar "ar")
if(CMAKE_AR)
    set(_mumps_ar "${CMAKE_AR}")
endif()
set(KADATH_MUMPS_AR "${_mumps_ar} cr ")
set(KADATH_MUMPS_OUTC "-o ")
set(KADATH_MUMPS_OUTF "-o ")
set(KADATH_MUMPS_RANLIB "ranlib")
if(CMAKE_RANLIB)
    set(KADATH_MUMPS_RANLIB "${CMAKE_RANLIB}")
endif()

# Optimization flags. No OpenMP (see Makefile.inc header note). gfortran >= 10
# rejects MUMPS's legacy mismatched MPI interface calls without
# -fallow-argument-mismatch.
set(KADATH_MUMPS_OPTF "-O3")
set(KADATH_MUMPS_OPTC "-O3")
set(KADATH_MUMPS_OPTL "-O3")
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    string(APPEND KADATH_MUMPS_OPTF " -fallow-argument-mismatch")
endif()

# MPI include flags for INCPAR. Harmless duplication when CC/FC are wrappers; the
# safety net when they are not. The MUMPS source is 242 .F files that #include
# mpif.h plus C files that #include mpi.h, so feed BOTH the Fortran and C MPI
# include trees (they differ on Intel-MPI/Cray); deduped so the wrapper case
# stays a harmless no-op.
set(KADATH_MUMPS_INCPAR "")
set(_mumps_seen_inc "")
foreach(_inc IN LISTS MPI_C_INCLUDE_DIRS MPI_Fortran_INCLUDE_DIRS)
    if(_inc AND NOT _inc IN_LIST _mumps_seen_inc)
        list(APPEND _mumps_seen_inc "${_inc}")
        string(APPEND KADATH_MUMPS_INCPAR " -I${_inc}")
    endif()
endforeach()

# METIS ordering: -Dmetis enables MUMPS's metis interface (mumps_metis.c); IMETIS
# puts metis.h on the compile line; LMETIS is the static libmetis for the final
# link (folded into the MUMPS link tail below). MUMPS auto-ordering (ICNTL(7)=7)
# then prefers metis over PORD.
set(KADATH_MUMPS_ORDERINGSF "-Dpord -Dmetis")
set(KADATH_MUMPS_IMETIS "-I${METIS_INCLUDE_DIR}")
set(KADATH_MUMPS_LMETIS "-L${METIS_CMAKE_DIR}/libmetis -lmetis")

# Numerical libraries (only used by a manual shared/examples build; see template).
string(REPLACE ";" " " KADATH_MUMPS_LAPACK "${LAPACK_LIBRARIES}")
string(REPLACE ";" " " KADATH_MUMPS_BLAS   "${BLAS_LIBRARIES}")
set(KADATH_MUMPS_SCALAP "")
if(SCALAPACK_LIBRARIES)
    string(REPLACE ";" " " KADATH_MUMPS_SCALAP "${SCALAPACK_LIBRARIES}")
endif()
set(KADATH_MUMPS_LIBOTHERS "-lpthread")

set(_MUMPS_MAKEFILE_INC "${CMAKE_BINARY_DIR}/mumps-Makefile.inc")
configure_file("${PROJECT_SOURCE_DIR}/cmake/mumps_Makefile.inc.in"
    "${_MUMPS_MAKEFILE_INC}" @ONLY)

# ── Build the static archives via MUMPS's own Makefiles ─────────────────────────
# Build only the libraries (PORD, common, double), never the example drivers
# (which would need a full ScaLAPACK/BLAS link we deliberately defer).
ExternalProject_Add(mumps_bundled
    SOURCE_DIR        "${MUMPS_BUILD_DIR}"
    DOWNLOAD_COMMAND  "${CMAKE_COMMAND}" -E copy_directory
                      "${MUMPS_VENDOR_DIR}" "${MUMPS_BUILD_DIR}"
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE   1
    # copy_if_different runs every build (BUILD_ALWAYS), so the active
    # Makefile.inc always reaches the MUMPS tree. The refresh_configuration step
    # below invalidates objects when that generated configuration changes.
    BUILD_COMMAND     "${CMAKE_COMMAND}" -E copy_if_different
                      "${_MUMPS_MAKEFILE_INC}" "${MUMPS_BUILD_DIR}/Makefile.inc"
            COMMAND   "${MUMPS_MAKE_PROGRAM}" "-j${_MUMPS_NJOBS}" lib/libpord.a
            COMMAND   "${MUMPS_MAKE_PROGRAM}" "-j${_MUMPS_NJOBS}" -C src libcommon
            COMMAND   "${MUMPS_MAKE_PROGRAM}" "-j${_MUMPS_NJOBS}" -C src ARITH=d ../lib/libdmumps.a
    BUILD_ALWAYS      1
    INSTALL_COMMAND   ""
    BUILD_BYPRODUCTS  "${_DMUMPS_A}" "${_MUMPS_COMMON_A}" "${_PORD_A}" "${_MUMPS_INT_DEF}"
    LOG_DOWNLOAD      1
    LOG_CONFIGURE     1
    LOG_BUILD         1
    USES_TERMINAL_BUILD 1)

# MUMPS object rules do not depend on Makefile.inc. Without an explicit
# invalidation step, changing compiler flags or ordering backends updates the
# file in the copied tree but silently reuses archives compiled with the old
# configuration. Tie a one-time clean to the generated Makefile's timestamp;
# ExternalProject's step stamp then keeps ordinary BUILD_ALWAYS invocations
# incremental while any real configure_file content change rebuilds all MUMPS
# objects and archives coherently.
ExternalProject_Add_Step(mumps_bundled refresh_configuration
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_MUMPS_MAKEFILE_INC}" "${MUMPS_BUILD_DIR}/Makefile.inc"
    COMMAND "${MUMPS_MAKE_PROGRAM}" -C src clean
    COMMAND "${MUMPS_MAKE_PROGRAM}" -C PORD/lib clean
    COMMAND "${CMAKE_COMMAND}" -E rm -f
            "${_DMUMPS_A}" "${_MUMPS_COMMON_A}" "${_PORD_A}"
            "${MUMPS_BUILD_DIR}/PORD/lib/libpord.a" "${_MUMPS_INT_DEF}"
    WORKING_DIRECTORY "${MUMPS_BUILD_DIR}"
    DEPENDEES download
    DEPENDERS build
    DEPENDS "${_MUMPS_MAKEFILE_INC}"
    COMMENT "Invalidating bundled MUMPS objects after a configuration change")

# metis.h + libmetis.a must exist before MUMPS compiles mumps_metis.c / links.
add_dependencies(mumps_bundled metis_bundled)

# ── Post-MUMPS dependency tail ──────────────────────────────────────────────────
# Order matters: libdmumps -> libmumps_common -> libpord is a clean dependency
# DAG (PORD never calls back into MUMPS), so a single ordered pass resolves it.
# On linkers that support a rescanned link group (GNU ld --start-group) wrap the
# archives in one anyway, as a belt-and-suspenders guard against any back-edge;
# AppleClang/ld64 does not support the RESCAN feature for CXX (and rescans
# archives natively), so fall back to the plain ordered list there.
# libmetis follows libmumps_common (mumps_metis.c calls METIS_NodeND); it is
# self-contained (GKlib folded in). The DAG stays acyclic: common -> metis.
set(_MUMPS_CORE_LIBRARIES "${_DMUMPS_A}" "${_MUMPS_COMMON_A}"
    "${_PORD_A}" "${METIS_LIBRARY}")
set(_MUMPS_CORE_LINK_LIBRARIES ${_MUMPS_CORE_LIBRARIES})
list(LENGTH _MUMPS_CORE_LIBRARIES _MUMPS_CORE_COUNT)
set(_MUMPS_RESCAN_OK FALSE)
if(DEFINED CMAKE_CXX_LINK_GROUP_USING_RESCAN_SUPPORTED AND CMAKE_CXX_LINK_GROUP_USING_RESCAN_SUPPORTED)
    set(_MUMPS_RESCAN_OK TRUE)
elseif(DEFINED CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED AND CMAKE_LINK_GROUP_USING_RESCAN_SUPPORTED)
    set(_MUMPS_RESCAN_OK TRUE)
endif()
if(_MUMPS_CORE_COUNT GREATER 1 AND _MUMPS_RESCAN_OK)
    string(REPLACE ";" "," _MUMPS_CORE_GROUP "${_MUMPS_CORE_LIBRARIES}")
    set(_MUMPS_CORE_LINK_LIBRARIES "$<LINK_GROUP:RESCAN,${_MUMPS_CORE_GROUP}>")
endif()

set(_MUMPS_DEPENDENCY_LIBRARIES)
if(SCALAPACK_LIBRARIES)
    list(APPEND _MUMPS_DEPENDENCY_LIBRARIES ${SCALAPACK_LIBRARIES})
endif()
if(LAPACK_LIBRARIES)
    list(APPEND _MUMPS_DEPENDENCY_LIBRARIES ${LAPACK_LIBRARIES})
endif()
if(BLAS_LIBRARIES)
    list(APPEND _MUMPS_DEPENDENCY_LIBRARIES ${BLAS_LIBRARIES})
endif()
if(TARGET MPI::MPI_Fortran)
    list(APPEND _MUMPS_DEPENDENCY_LIBRARIES MPI::MPI_Fortran)
elseif(MPI_Fortran_LIBRARIES)
    list(APPEND _MUMPS_DEPENDENCY_LIBRARIES ${MPI_Fortran_LIBRARIES})
endif()

# Fortran runtime + threads needed by the static MUMPS archives at final link.
if(APPLE)
    find_library(QUADMATH_LIB quadmath
        PATHS /opt/homebrew/opt/gcc/lib/gcc/current NO_DEFAULT_PATH)
    find_library(GFORTRAN_LIB gfortran
        PATHS /opt/homebrew/opt/gcc/lib/gcc/current NO_DEFAULT_PATH)
    foreach(_lib IN ITEMS QUADMATH_LIB GFORTRAN_LIB)
        if(${_lib})
            list(APPEND _MUMPS_DEPENDENCY_LIBRARIES "${${_lib}}")
        endif()
    endforeach()
    if(NOT GFORTRAN_LIB)
        list(APPEND _MUMPS_DEPENDENCY_LIBRARIES gfortran)
    endif()
    find_library(MPI_MPIFH_LIB mpi_mpifh
        PATHS /opt/homebrew/opt/open-mpi/lib NO_DEFAULT_PATH)
    if(MPI_MPIFH_LIB)
        list(APPEND _MUMPS_DEPENDENCY_LIBRARIES "${MPI_MPIFH_LIB}")
    endif()
elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    list(APPEND _MUMPS_DEPENDENCY_LIBRARIES gfortran)
endif()

find_package(Threads QUIET)
if(TARGET Threads::Threads)
    list(APPEND _MUMPS_DEPENDENCY_LIBRARIES Threads::Threads)
endif()

# libmetis (GKlib) uses the C math library; needed at the final link on Linux
# (on macOS libm lives in libSystem, so -lm is a harmless no-op).
list(APPEND _MUMPS_DEPENDENCY_LIBRARIES m)

# Site-specific post-MUMPS link tail wired by the HPC toolchains (Cray/Intel
# Fortran runtime, MPI Fortran binding, etc.). Appended so a cluster keeps its
# required final-link closure. No-op on macbook (variable unset).
if(CELEPHAIS_MUMPS_EXTRA_LIBRARIES)
    list(APPEND _MUMPS_DEPENDENCY_LIBRARIES ${CELEPHAIS_MUMPS_EXTRA_LIBRARIES})
endif()

if(_MUMPS_DEPENDENCY_LIBRARIES)
    list(REMOVE_DUPLICATES _MUMPS_DEPENDENCY_LIBRARIES)
endif()

# ── Public contract (MUMPS_* variables + MUMPS::MUMPS consumed by the build) ────
set(MUMPS_FOUND TRUE)
set(MUMPS_INCLUDE_DIR "${MUMPS_BUILD_INC}")
set(MUMPS_INCLUDE_DIRS "${MUMPS_BUILD_INC}")
set(MUMPS_LIBRARIES ${_MUMPS_CORE_LINK_LIBRARIES} ${_MUMPS_DEPENDENCY_LIBRARIES})
set(MUMPS_LINK_LIBRARIES ${MUMPS_LIBRARIES} CACHE STRING "Resolved MUMPS link libraries" FORCE)
set(MUMPS_PROVIDER "bundled" CACHE STRING "Resolved MUMPS provider class" FORCE)

if(NOT TARGET MUMPS::MUMPS)
    add_library(MUMPS::MUMPS INTERFACE IMPORTED GLOBAL)
endif()
set_target_properties(MUMPS::MUMPS PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MUMPS_BUILD_INC}"
    INTERFACE_LINK_LIBRARIES "${MUMPS_LIBRARIES}")
# add_dependencies() cannot target an IMPORTED library, so build ordering is
# established by the consumer (add_dependencies(celephais mumps_bundled) in the top
# CMakeLists, so the generated headers exist before Celephais compiles) and by the
# BUILD_BYPRODUCTS above (Ninja knows the archives are produced by mumps_bundled
# and orders every link that consumes them accordingly).

message(STATUS "MUMPS provider: bundled (third_party/mumps, built in ${MUMPS_BUILD_DIR})")
