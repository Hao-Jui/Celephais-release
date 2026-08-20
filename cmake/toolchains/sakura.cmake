# sakura.cmake — CMake toolchain for Sakura HPC cluster (MPCDF)
#
# Modules are loaded automatically by compile.sh (sources cmake/env/sakura.sh).
# Manual setup: source cmake/env/sakura.sh && cmake --preset sakura

# ── Compilers (Intel MPI wrappers, version set by module) ────────────────────
set(CMAKE_CXX_COMPILER "mpigxx" CACHE FILEPATH "C++ compiler")
set(CMAKE_C_COMPILER   "mpigcc" CACHE FILEPATH "C compiler")

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
