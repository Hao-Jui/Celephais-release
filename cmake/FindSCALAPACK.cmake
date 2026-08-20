set(_SCALAPACK_ROOT_HINTS)
foreach(_root IN ITEMS "${SCALAPACK_DIR}" "${SCALAPACK_ROOT}" "$ENV{SCALAPACK_DIR}" "$ENV{SCALAPACK_ROOT}")
    if(_root)
        list(APPEND _SCALAPACK_ROOT_HINTS "${_root}")
    endif()
endforeach()

set(_SCALAPACK_LIB_HINTS)
foreach(_root IN LISTS _SCALAPACK_ROOT_HINTS)
    list(APPEND _SCALAPACK_LIB_HINTS "${_root}/lib" "${_root}/lib64")
endforeach()

if(DEFINED ENV{MKLROOT})
    list(APPEND _SCALAPACK_LIB_HINTS
        "$ENV{MKLROOT}/lib"
        "$ENV{MKLROOT}/lib/intel64"
    )
endif()

# Reference ScaLAPACK: plain name (source/`make install`) plus Debian/Ubuntu
# MPI-flavoured package names (libscalapack-openmpi.so / libscalapack-mpich.so).
# No NO_DEFAULT_PATH on purpose — CMake's default search already covers the
# multiarch dirs (/usr/lib/<triple>, /usr/lib64), so this stays portable across
# arches/distros instead of a hardcoded x86_64 path list.
find_library(SCALAPACK_LIBRARY
    NAMES scalapack scalapack-openmpi scalapack-mpich
    PATHS ${_SCALAPACK_LIB_HINTS}
)

find_library(MKL_SCALAPACK_LIBRARY
    NAMES mkl_scalapack_lp64 mkl_scalapack_ilp64
    PATHS ${_SCALAPACK_LIB_HINTS}
)

if(DEFINED ENV{I_MPI_ROOT})
    set(_MKL_BLACS_NAMES mkl_blacs_intelmpi_lp64 mkl_blacs_openmpi_lp64 mkl_blacs_lp64)
else()
    set(_MKL_BLACS_NAMES mkl_blacs_openmpi_lp64 mkl_blacs_intelmpi_lp64 mkl_blacs_lp64)
endif()

find_library(MKL_BLACS_LIBRARY
    NAMES ${_MKL_BLACS_NAMES}
    PATHS ${_SCALAPACK_LIB_HINTS}
)

if(MKL_SCALAPACK_LIBRARY AND MKL_BLACS_LIBRARY)
    set(SCALAPACK_LIBRARIES ${MKL_SCALAPACK_LIBRARY} ${MKL_BLACS_LIBRARY})
elseif(SCALAPACK_LIBRARY)
    set(SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCALAPACK DEFAULT_MSG SCALAPACK_LIBRARIES)

if(SCALAPACK_FOUND AND NOT TARGET SCALAPACK::SCALAPACK)
    add_library(SCALAPACK::SCALAPACK INTERFACE IMPORTED)
    target_link_libraries(SCALAPACK::SCALAPACK INTERFACE ${SCALAPACK_LIBRARIES})
endif()

mark_as_advanced(
    SCALAPACK_LIBRARY
    MKL_SCALAPACK_LIBRARY
    MKL_BLACS_LIBRARY
)
