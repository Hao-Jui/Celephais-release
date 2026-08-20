# CMake toolchain for the snowman workstation.
#
# Manual setup: source cmake/env/snowman.sh && cmake --preset snowman

set(CMAKE_C_COMPILER "mpicc" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "mpicxx" CACHE FILEPATH "C++ compiler")
set(CMAKE_Fortran_COMPILER "mpifort" CACHE FILEPATH "Fortran compiler")
set(MPI_C_COMPILER "mpicc" CACHE FILEPATH "MPI C compiler")
set(MPI_CXX_COMPILER "mpicxx" CACHE FILEPATH "MPI CXX compiler")
set(MPI_Fortran_COMPILER "mpifort" CACHE FILEPATH "MPI Fortran compiler")

set(MKL_VERSION OFF CACHE BOOL "snowman uses system BLAS/LAPACK")
set(CMAKE_MAKE_PROGRAM "make" CACHE FILEPATH "Make")

set(_SNOWMAN_DEPS "$ENV{KADATH_DEPS_ROOT}")
if(NOT _SNOWMAN_DEPS)
    set(_SNOWMAN_DEPS "$ENV{HOME}/scratch3/kadath-local")
endif()
set(_SNOWMAN_USR "${_SNOWMAN_DEPS}/usr")
set(_SNOWMAN_LIB "${_SNOWMAN_USR}/lib/x86_64-linux-gnu")

set(FFTW_ROOT "${_SNOWMAN_USR}" CACHE PATH "FFTW root")
if(EXISTS "${_SNOWMAN_USR}/include/fftw3.h")
    set(FFTW_INCLUDES "${_SNOWMAN_USR}/include" CACHE PATH "FFTW include dir")
    set(FFTW_LIB "${_SNOWMAN_LIB}/libfftw3.so" CACHE FILEPATH "FFTW library")
    set(FFTW_INCLUDE_DIRS "${_SNOWMAN_USR}/include" CACHE PATH "FFTW include dir")
    set(FFTW_LIBRARIES "${_SNOWMAN_LIB}/libfftw3.so" CACHE STRING "FFTW libraries")
endif()

set(BLAS_LIBRARIES "/usr/lib/x86_64-linux-gnu/libblas.so" CACHE STRING "BLAS library")
set(LAPACK_LIBRARIES "/usr/lib/x86_64-linux-gnu/liblapack.so" CACHE STRING "LAPACK library")
set(SCALAPACK_LIBRARY "${_SNOWMAN_LIB}/libscalapack-mpich.so" CACHE FILEPATH "MPICH ScaLAPACK")
set(SCALAPACK_LIBRARIES "${_SNOWMAN_LIB}/libscalapack-mpich.so" CACHE STRING "MPICH ScaLAPACK")

# MUMPS is built in-tree (third_party/mumps); only its link tail is wired here.
set(CELEPHAIS_MUMPS_EXTRA_LIBRARIES
    "${_SNOWMAN_LIB}/libscalapack-mpich.so"
    "/usr/lib/x86_64-linux-gnu/liblapack.so"
    "/usr/lib/x86_64-linux-gnu/libblas.so"
    "/usr/lib/x86_64-linux-gnu/libmpichfort.so"
    "/usr/lib/x86_64-linux-gnu/libmpich.so"
    CACHE STRING "Extra snowman libraries required when linking static MUMPS" FORCE)
