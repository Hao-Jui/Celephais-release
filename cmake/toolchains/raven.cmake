# raven.cmake — CMake toolchain for Raven HPC cluster (MPCDF)
#
# Modules are loaded automatically by compile.sh (sources cmake/env/raven.sh).
# Manual setup: source cmake/env/raven.sh && cmake --preset raven

# ── Compilers (Intel MPI wrappers, version set by module) ────────────────────
set(CMAKE_CXX_COMPILER "mpig++" CACHE FILEPATH "C++ compiler")
set(CMAKE_C_COMPILER   "mpigcc" CACHE FILEPATH "C compiler")
set(CMAKE_Fortran_COMPILER "mpifort" CACHE FILEPATH "Fortran compiler")

# ── MKL (replaces OpenBLAS/ScaLAPACK/LAPACK) ─────────────────────────────────
set(MKL_VERSION ON CACHE BOOL "Use Intel MKL instead of SCALAPACK/BLAS/LAPACK")

# ── Serial FFTW (provided by the fftw-mpi module) ────────────────────────────
if(DEFINED ENV{FFTW_HOME})
    set(FFTW_INCLUDE_DIRS "$ENV{FFTW_HOME}/include" CACHE PATH "FFTW include dir")
    set(FFTW_LIBRARIES
        "$ENV{FFTW_HOME}/lib/libfftw3.so"
        CACHE STRING "FFTW libraries" FORCE)
endif()

# MUMPS is built in-tree (third_party/mumps); no system/PETSc MUMPS is discovered.

if(DEFINED ENV{PETSC_ARCH})
    set(PETSC_ARCH "$ENV{PETSC_ARCH}" CACHE STRING "PETSc architecture")
endif()
