# Generic toolchain for an already prepared environment.
#
# Expected inputs are ordinary environment variables such as CC/CXX/FC,
# FFTW_HOME, BOOST_ROOT/BOOST_HOME, and optional
# BLAS_LIBRARIES/LAPACK_LIBRARIES/SCALAPACK_LIBRARIES tails. MUMPS is built
# in-tree from third_party/mumps — no MUMPS/PETSc prefix is needed.

function(_celephais_env_set_from_env _cmake_var _env_var _type _doc)
    if(DEFINED ENV{${_env_var}} AND NOT "$ENV{${_env_var}}" STREQUAL "")
        set(${_cmake_var} "$ENV{${_env_var}}" CACHE ${_type} "${_doc}" FORCE)
    endif()
endfunction()

function(_celephais_env_find_program _cmake_var _doc)
    if(DEFINED ${_cmake_var} AND NOT "${${_cmake_var}}" STREQUAL "")
        return()
    endif()
    find_program(_celephais_env_found_program NAMES ${ARGN})
    if(_celephais_env_found_program)
        set(${_cmake_var} "${_celephais_env_found_program}" CACHE FILEPATH "${_doc}" FORCE)
    endif()
    unset(_celephais_env_found_program CACHE)
endfunction()

function(_celephais_env_first_prefix _out)
    foreach(_env_var IN LISTS ARGN)
        if(DEFINED ENV{${_env_var}} AND NOT "$ENV{${_env_var}}" STREQUAL "")
            set(${_out} "$ENV{${_env_var}}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${_out} "" PARENT_SCOPE)
endfunction()

_celephais_env_set_from_env(CMAKE_C_COMPILER CC FILEPATH "C compiler from active environment")
_celephais_env_set_from_env(CMAKE_CXX_COMPILER CXX FILEPATH "C++ compiler from active environment")
_celephais_env_set_from_env(CMAKE_Fortran_COMPILER FC FILEPATH "Fortran compiler from active environment")

_celephais_env_find_program(CMAKE_C_COMPILER "C compiler from active environment"
    mpicc mpigcc mpiicc cc)
_celephais_env_find_program(CMAKE_CXX_COMPILER "C++ compiler from active environment"
    mpicxx mpig++ mpigxx mpiicpx CC c++)
_celephais_env_find_program(CMAKE_Fortran_COMPILER "Fortran compiler from active environment"
    mpifort mpiifort ifort ifx ftn gfortran)

if(CMAKE_C_COMPILER)
    set(MPI_C_COMPILER "${CMAKE_C_COMPILER}" CACHE FILEPATH "MPI C compiler" FORCE)
endif()
if(CMAKE_CXX_COMPILER)
    set(MPI_CXX_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "MPI CXX compiler" FORCE)
endif()
if(CMAKE_Fortran_COMPILER)
    set(MPI_Fortran_COMPILER "${CMAKE_Fortran_COMPILER}" CACHE FILEPATH "MPI Fortran compiler" FORCE)
endif()

# Some clusters expose the serial library from a provider whose legacy prefix
# variable is FFTW_MPI_HOME. Accept that prefix, but link only libfftw3.
_celephais_env_first_prefix(_CELEPHAIS_ENV_FFTW_PREFIX FFTW_HOME FFTW_MPI_HOME FFTW_ROOT FFTWDIR)
if(_CELEPHAIS_ENV_FFTW_PREFIX)
    set(FFTW_ROOT "${_CELEPHAIS_ENV_FFTW_PREFIX}" CACHE PATH "FFTW prefix" FORCE)
    set(FFTW_INCLUDE_DIRS "${_CELEPHAIS_ENV_FFTW_PREFIX}/include" CACHE PATH "FFTW include dir" FORCE)
    if(EXISTS "${_CELEPHAIS_ENV_FFTW_PREFIX}/lib/libfftw3${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(FFTW_LIBRARIES
            "${_CELEPHAIS_ENV_FFTW_PREFIX}/lib/libfftw3${CMAKE_SHARED_LIBRARY_SUFFIX}"
            CACHE STRING "FFTW libraries" FORCE)
    endif()
endif()

# MUMPS is built in-tree (third_party/mumps); no system/PETSc MUMPS is discovered.
_celephais_env_set_from_env(BLAS_LIBRARIES BLAS_LIBRARIES STRING "BLAS libraries")
_celephais_env_set_from_env(LAPACK_LIBRARIES LAPACK_LIBRARIES STRING "LAPACK libraries")
_celephais_env_set_from_env(SCALAPACK_LIBRARIES SCALAPACK_LIBRARIES STRING "ScaLAPACK libraries")
_celephais_env_set_from_env(SCALAPACK_DIR SCALAPACK_DIR PATH "ScaLAPACK prefix")
_celephais_env_set_from_env(CELEPHAIS_MUMPS_EXTRA_LIBRARIES CELEPHAIS_MUMPS_EXTRA_LIBRARIES STRING "Extra MUMPS link tail")

if(DEFINED ENV{CELEPHAIS_TARGET_CLASS} AND NOT "$ENV{CELEPHAIS_TARGET_CLASS}" STREQUAL "")
    set(CELEPHAIS_TARGET_CLASS "$ENV{CELEPHAIS_TARGET_CLASS}" CACHE STRING "Target launcher class" FORCE)
endif()
