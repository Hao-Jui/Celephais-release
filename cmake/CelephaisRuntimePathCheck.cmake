option(CELEPHAIS_RUNTIME_PATH_WARN
    "Warn when shared dependency directories are missing from the runtime loader path"
    ON)

function(_celephais_append_existing_dir _out _dir)
    set(_dirs ${${_out}})
    if(_dir AND IS_DIRECTORY "${_dir}")
        list(APPEND _dirs "${_dir}")
    endif()
    set(${_out} ${_dirs} PARENT_SCOPE)
endfunction()

function(_celephais_collect_shared_lib_dirs _out)
    set(_dirs)
    foreach(_var IN LISTS ARGN)
        if(NOT DEFINED ${_var})
            continue()
        endif()
        foreach(_item IN LISTS ${_var})
            if(IS_ABSOLUTE "${_item}" AND _item MATCHES "\\.(so|dylib)(\\.|$)")
                get_filename_component(_dir "${_item}" DIRECTORY)
                _celephais_append_existing_dir(_dirs "${_dir}")
            elseif(_item MATCHES "^-L(.+)")
                _celephais_append_existing_dir(_dirs "${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endforeach()
    if(_dirs)
        list(REMOVE_DUPLICATES _dirs)
    endif()
    set(${_out} ${_dirs} PARENT_SCOPE)
endfunction()

function(celephais_warn_missing_runtime_paths)
    if(NOT CELEPHAIS_RUNTIME_PATH_WARN)
        return()
    endif()
    if(APPLE)
        set(CELEPHAIS_RUNTIME_PATH_MISSING ""
            CACHE STRING "Shared dependency directories absent from runtime loader path"
            FORCE)
        return()
    endif()

    _celephais_collect_shared_lib_dirs(_runtime_dirs
        BLAS_LIBRARIES
        LAPACK_LIBRARIES
        SCALAPACK_LIBRARIES
        MUMPS_LINK_LIBRARIES
    )

    if(CELEPHAIS_ENABLE_FFTW_ORACLE)
        _celephais_collect_shared_lib_dirs(_fftw_runtime_dirs FFTW_LIBRARIES)
        list(APPEND _runtime_dirs ${_fftw_runtime_dirs})
    endif()

    foreach(_prefix IN ITEMS
            "$ENV{BOOST_ROOT}"
            "$ENV{BOOST_HOME}"
            "$ENV{MKLROOT}"
            "$ENV{I_MPI_ROOT}")
        if(NOT _prefix)
            continue()
        endif()
        _celephais_append_existing_dir(_runtime_dirs "${_prefix}/lib")
        _celephais_append_existing_dir(_runtime_dirs "${_prefix}/lib64")
        _celephais_append_existing_dir(_runtime_dirs "${_prefix}/lib/intel64")
        _celephais_append_existing_dir(_runtime_dirs "${_prefix}/lib/release")
    endforeach()
    if(CELEPHAIS_ENABLE_FFTW_ORACLE)
        foreach(_prefix IN ITEMS "$ENV{FFTW_HOME}" "$ENV{FFTW_MPI_HOME}")
            _celephais_append_existing_dir(_runtime_dirs "${_prefix}/lib")
            _celephais_append_existing_dir(_runtime_dirs "${_prefix}/lib64")
        endforeach()
    endif()

    if(_runtime_dirs)
        list(REMOVE_DUPLICATES _runtime_dirs)
    endif()

    set(_loader_path ":$ENV{LD_LIBRARY_PATH}:")
    if(APPLE)
        string(APPEND _loader_path "$ENV{DYLD_LIBRARY_PATH}:")
    endif()

    set(_missing)
    foreach(_dir IN LISTS _runtime_dirs)
        string(FIND "${_loader_path}" ":${_dir}:" _found_at)
        if(_found_at EQUAL -1)
            list(APPEND _missing "${_dir}")
        endif()
    endforeach()

    if(_missing)
        list(REMOVE_DUPLICATES _missing)
        set(CELEPHAIS_RUNTIME_PATH_MISSING "${_missing}"
            CACHE STRING "Shared dependency directories absent from runtime loader path"
            FORCE)
        message(WARNING
            "Runtime loader path is missing shared dependency directories: ${_missing}\n"
            "Source cmake/env/<target>.sh or add these directories to LD_LIBRARY_PATH "
            "before building/running. This is a pre-build warning; ldd remains the final proof.")
    else()
        set(CELEPHAIS_RUNTIME_PATH_MISSING ""
            CACHE STRING "Shared dependency directories absent from runtime loader path"
            FORCE)
    endif()
endfunction()
