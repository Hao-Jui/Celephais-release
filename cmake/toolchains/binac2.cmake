# CMake toolchain for bwForCluster BinAC 2.
#
# Manual setup: source cmake/env/binac2.sh && cmake --preset binac2

set(CMAKE_C_COMPILER "mpicc" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "mpicxx" CACHE FILEPATH "C++ compiler")
set(CMAKE_Fortran_COMPILER "mpifort" CACHE FILEPATH "Fortran compiler")
set(MPI_C_COMPILER "mpicc" CACHE FILEPATH "MPI C compiler")
set(MPI_CXX_COMPILER "mpicxx" CACHE FILEPATH "MPI CXX compiler")
set(MPI_Fortran_COMPILER "mpifort" CACHE FILEPATH "MPI Fortran compiler")

set(MKL_VERSION OFF CACHE BOOL "BinAC2 uses PETSc/OpenBLAS/ScaLAPACK")
set(CMAKE_MAKE_PROGRAM "ninja" CACHE FILEPATH "Ninja")

execute_process(
    COMMAND mpifort --showme:libdirs
    OUTPUT_VARIABLE _BINAC2_MPI_LIBDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(_BINAC2_MPI_LIBDIR)
    string(REPLACE " " ";" _BINAC2_MPI_LIBDIRS "${_BINAC2_MPI_LIBDIR}")
    list(GET _BINAC2_MPI_LIBDIRS 0 _BINAC2_MPI_LIB)
endif()
if(_BINAC2_MPI_LIB)
    set(_BINAC2_MPI_FORTRAN_LIBRARIES
        "${_BINAC2_MPI_LIB}/libmpi_usempif08.so"
        "${_BINAC2_MPI_LIB}/libmpi_usempi_ignore_tkr.so"
        "${_BINAC2_MPI_LIB}/libmpi_mpifh.so"
        "${_BINAC2_MPI_LIB}/libmpi.so")
endif()
set(_BINAC2_MUMPS_EXTRA_LIBRARIES)
if(_BINAC2_MPI_FORTRAN_LIBRARIES)
    list(APPEND _BINAC2_MUMPS_EXTRA_LIBRARIES ${_BINAC2_MPI_FORTRAN_LIBRARIES})
endif()
list(APPEND _BINAC2_MUMPS_EXTRA_LIBRARIES gfortran)

if(DEFINED ENV{FFTW_HOME})
    set(FFTW_INCLUDE_DIRS "$ENV{FFTW_HOME}/include" CACHE PATH "FFTW include dir")
    set(FFTW_LIBRARIES
        "$ENV{FFTW_HOME}/lib/libfftw3.so"
        CACHE STRING "FFTW libraries" FORCE)
endif()

# The BinAC2 PETSc module ships the numerical stack (OpenBLAS, ScaLAPACK,
# (par)metis); its prefix is taken from $MUMPS_DIR or $PETSC_DIR. MUMPS itself is
# built in-tree (third_party/mumps), so the prefix is used only for BLAS/ScaLAPACK
# and the MUMPS link tail below — no system MUMPS is discovered.
if(DEFINED ENV{MUMPS_DIR})
    set(_BINAC2_PETSC "$ENV{MUMPS_DIR}")
elseif(DEFINED ENV{PETSC_DIR})
    set(_BINAC2_PETSC "$ENV{PETSC_DIR}")
endif()

if(_BINAC2_PETSC)
    set(BLAS_LIBRARIES
        "${_BINAC2_PETSC}/lib/libopenblas.so"
        CACHE STRING "PETSc OpenBLAS")
    set(LAPACK_LIBRARIES
        "${_BINAC2_PETSC}/lib/libopenblas.so"
        CACHE STRING "PETSc OpenBLAS LAPACK")
    set(SCALAPACK_LIBRARY
        "${_BINAC2_PETSC}/lib/libscalapack.so"
        CACHE FILEPATH "PETSc ScaLAPACK")
    set(SCALAPACK_LIBRARIES
        "${_BINAC2_PETSC}/lib/libscalapack.so"
        CACHE STRING "PETSc ScaLAPACK")
    list(PREPEND _BINAC2_MUMPS_EXTRA_LIBRARIES
        "${_BINAC2_PETSC}/lib/libscalapack.so"
        "${_BINAC2_PETSC}/lib/libopenblas.so"
        "${_BINAC2_PETSC}/lib/libparmetis.so"
        "${_BINAC2_PETSC}/lib/libmetis.so")
    list(APPEND _BINAC2_MUMPS_EXTRA_LIBRARIES pthread m)
endif()

if(_BINAC2_MUMPS_EXTRA_LIBRARIES)
    set(CELEPHAIS_MUMPS_EXTRA_LIBRARIES
        ${_BINAC2_MUMPS_EXTRA_LIBRARIES}
        CACHE STRING "Extra BinAC2 libraries required when linking bundled MUMPS" FORCE)
endif()
