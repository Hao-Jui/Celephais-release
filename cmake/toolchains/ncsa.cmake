# CMake toolchain for NCSA Campus Cluster.
#
# Manual setup: source cmake/env/ncsa.sh && cmake --preset ncsa

set(CMAKE_C_COMPILER "mpicc" CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "mpicxx" CACHE FILEPATH "C++ compiler")
set(CMAKE_Fortran_COMPILER "mpifort" CACHE FILEPATH "Fortran compiler")
set(MPI_C_COMPILER "mpicc" CACHE FILEPATH "MPI C compiler")
set(MPI_CXX_COMPILER "mpicxx" CACHE FILEPATH "MPI CXX compiler")
set(MPI_Fortran_COMPILER "mpifort" CACHE FILEPATH "MPI Fortran compiler")

set(MKL_VERSION OFF CACHE BOOL "Use explicit MKL libraries on NCSA")
set(CMAKE_MAKE_PROGRAM "ninja" CACHE FILEPATH "Ninja")

execute_process(
    COMMAND mpifort --showme:libdirs
    OUTPUT_VARIABLE _NCSA_MPI_LIBDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(_NCSA_MPI_LIBDIR)
    string(REPLACE " " ";" _NCSA_MPI_LIBDIRS "${_NCSA_MPI_LIBDIR}")
    list(GET _NCSA_MPI_LIBDIRS 0 _NCSA_MPI_LIB)
elseif(EXISTS "/sw/apps/mpi/openmpi/5.0.1/gcc/13.3.0/lib")
    set(_NCSA_MPI_LIB "/sw/apps/mpi/openmpi/5.0.1/gcc/13.3.0/lib")
endif()
if(_NCSA_MPI_LIB)
    set(_NCSA_MPI_FORTRAN_LIBRARIES
        "${_NCSA_MPI_LIB}/libmpi_usempif08.so"
        "${_NCSA_MPI_LIB}/libmpi_usempi_ignore_tkr.so"
        "${_NCSA_MPI_LIB}/libmpi_mpifh.so"
        "${_NCSA_MPI_LIB}/libmpi.so")
endif()

if(DEFINED ENV{FFTW_HOME})
    set(FFTW_INCLUDE_DIRS "$ENV{FFTW_HOME}/include" CACHE PATH "FFTW include dir")
    set(FFTW_LIBRARIES
        "$ENV{FFTW_HOME}/lib/libfftw3.so"
        CACHE STRING "FFTW libraries" FORCE)
endif()

if(DEFINED ENV{MKLROOT})
    set(_NCSA_MKL "$ENV{MKLROOT}/lib")
    set(_NCSA_MKL_CORE_LIBRARIES
        "${_NCSA_MKL}/libmkl_gf_lp64.so"
        "${_NCSA_MKL}/libmkl_sequential.so"
        "${_NCSA_MKL}/libmkl_core.so"
        pthread
        m
        dl)
    set(BLAS_LIBRARIES ${_NCSA_MKL_CORE_LIBRARIES} CACHE STRING "MKL BLAS")
    set(LAPACK_LIBRARIES ${_NCSA_MKL_CORE_LIBRARIES} CACHE STRING "MKL LAPACK")
    set(SCALAPACK_LIBRARIES
        "${_NCSA_MKL}/libmkl_scalapack_lp64.so"
        "${_NCSA_MKL}/libmkl_blacs_openmpi_lp64.so"
        ${_NCSA_MKL_CORE_LIBRARIES}
        CACHE STRING "MKL ScaLAPACK/OpenMPI BLACS")
    set(CELEPHAIS_MUMPS_EXTRA_LIBRARIES
        "${_NCSA_MKL}/libmkl_scalapack_lp64.so"
        "${_NCSA_MKL}/libmkl_blacs_openmpi_lp64.so"
        ${_NCSA_MKL_CORE_LIBRARIES}
        ${_NCSA_MPI_FORTRAN_LIBRARIES}
        CACHE STRING "Extra NCSA libraries required when linking static MUMPS" FORCE)
endif()
