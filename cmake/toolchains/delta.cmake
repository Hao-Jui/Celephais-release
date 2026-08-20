# CMake toolchain for NCSA Delta.
#
# Manual setup: source cmake/env/delta.sh && cmake --preset delta

set(CMAKE_C_COMPILER "cc" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "CC" CACHE FILEPATH "C++ compiler")
set(CMAKE_Fortran_COMPILER "ftn" CACHE FILEPATH "Fortran compiler")
set(MPI_Fortran_COMPILER "ftn" CACHE FILEPATH "MPI Fortran compiler")

set(MKL_VERSION OFF CACHE BOOL "Delta uses Cray LibSci, not MKL")
set(CMAKE_MAKE_PROGRAM "gmake" CACHE FILEPATH "GNU Make")

if(DEFINED ENV{FFTW_HOME})
    set(FFTW_INCLUDE_DIRS "$ENV{FFTW_HOME}/include" CACHE PATH "FFTW include dir")
    set(FFTW_LIBRARIES
        "$ENV{FFTW_HOME}/lib/libfftw3.so"
        CACHE STRING "FFTW libraries" FORCE)
endif()

if(DEFINED ENV{CRAY_LIBSCI_PREFIX})
    set(_DELTA_LIBSCI "$ENV{CRAY_LIBSCI_PREFIX}/lib/libsci_gnu_mpi.so.6.0")
    set(BLAS_LIBRARIES "${_DELTA_LIBSCI}" CACHE STRING "Cray LibSci BLAS")
    set(LAPACK_LIBRARIES "${_DELTA_LIBSCI}" CACHE STRING "Cray LibSci LAPACK")
    set(SCALAPACK_LIBRARY "${_DELTA_LIBSCI}" CACHE FILEPATH "Cray LibSci ScaLAPACK")
    set(SCALAPACK_LIBRARIES "${_DELTA_LIBSCI}" CACHE STRING "Cray LibSci ScaLAPACK")
endif()

set(CELEPHAIS_MUMPS_EXTRA_LIBRARIES
    "/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib/libmpifort_gnu_112.so"
    "/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib/libmpi_gnu_112.so"
    "/opt/cray/pe/dsmml/0.3.1/dsmml/lib/libdsmml.so"
    CACHE STRING "Extra Delta libraries required when linking static MUMPS from C++" FORCE)
