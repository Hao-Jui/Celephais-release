# macbook.cmake — CMake toolchain for macOS with Homebrew
#
# Detects Homebrew-installed packages dynamically via `brew --prefix`.
# Works on both Apple Silicon (/opt/homebrew) and Intel (/usr/local) Macs.

# ── Homebrew detection ────────────────────────────────────────────────────────
find_program(BREW brew REQUIRED)
if(NOT BREW)
    message(FATAL_ERROR "Homebrew not found. Install from https://brew.sh")
endif()

macro(brew_prefix _pkg _out)
    execute_process(
        COMMAND ${BREW} --prefix ${_pkg}
        OUTPUT_VARIABLE ${_out}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endmacro()

# ── Compilers (from Open MPI) ─────────────────────────────────────────────────
brew_prefix(open-mpi OMPI_PREFIX)
if(NOT OMPI_PREFIX OR NOT EXISTS "${OMPI_PREFIX}/bin/mpicc")
    message(FATAL_ERROR "open-mpi not found via Homebrew. Run: brew install open-mpi")
endif()

set(CMAKE_C_COMPILER   "${OMPI_PREFIX}/bin/mpicc"  CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${OMPI_PREFIX}/bin/mpicxx" CACHE FILEPATH "C++ compiler")

# ── FFTW ──────────────────────────────────────────────────────────────────────
brew_prefix(fftw FFTW_PREFIX)
if(FFTW_PREFIX)
    set(FFTW_INCLUDE_DIRS "${FFTW_PREFIX}/include" CACHE PATH "FFTW include dir")
    set(FFTW_LIBRARIES
        "${FFTW_PREFIX}/lib/libfftw3.dylib"
        CACHE STRING "FFTW libraries" FORCE)
endif()

# ── Apple Accelerate (BLAS + LAPACK) ──────────────────────────────────────────
# BLA_VENDOR is also set by the macbook preset so FindBLAS/FindLAPACK retain the
# backend choice if their discovery logic is used elsewhere.  Force the concrete
# framework here so an existing build directory cannot retain the former
# Homebrew OpenBLAS cache entries.
set(ACCELERATE_FRAMEWORK "/System/Library/Frameworks/Accelerate.framework")
set(BLAS_LIBRARIES
    "${ACCELERATE_FRAMEWORK}"
    CACHE STRING "BLAS library" FORCE)
set(LAPACK_LIBRARIES
    "${ACCELERATE_FRAMEWORK}"
    CACHE STRING "LAPACK library" FORCE)

# ── ScaLAPACK ─────────────────────────────────────────────────────────────────
brew_prefix(scalapack SCALAPACK_PREFIX)
if(SCALAPACK_PREFIX)
    set(SCALAPACK_LIBRARIES "${SCALAPACK_PREFIX}/lib/libscalapack.dylib" CACHE STRING "ScaLAPACK library")
endif()

# ── MUMPS ─────────────────────────────────────────────────────────────────────
# Built in-tree from third_party/mumps by cmake/BundledMUMPS.cmake; the gfortran
# / quadmath runtime tail it needs at link is resolved there. No system MUMPS and
# no hardcoded path.
