# Kadath physics application suite.
# Included by root CMakeLists.txt when BUILD_APPS=ON.
# CMakeExec.cmake is included once here so each app subdirectory can use
# kadath_add_exec() without re-including it.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMakeExec.cmake)

set(KADATH_APPS NS NS_nosym NS2d NS2d_xcts BH BH2d BHNS BNS BNS_nosym)
# Non-app subdirectories that legitimately carry a CMakeLists.txt or sources.
set(KADATH_APPS_EXCLUDED cmake shared submission_scripts)

foreach(APP ${KADATH_APPS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${APP}/CMakeLists.txt")
        add_subdirectory(${APP})
    else()
        message(WARNING "App '${APP}' not found at ${CMAKE_CURRENT_SOURCE_DIR}/${APP}, skipping.")
    endif()
endforeach()

# Guard against the silent-skip trap: an app directory present on disk but
# missing from KADATH_APPS would otherwise be dropped from the build without
# any diagnostic.
file(GLOB _kadath_app_dirs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
foreach(_app_cmake ${_kadath_app_dirs})
    get_filename_component(_app_dir "${_app_cmake}" DIRECTORY)
    if(NOT _app_dir IN_LIST KADATH_APPS AND NOT _app_dir IN_LIST KADATH_APPS_EXCLUDED)
        message(FATAL_ERROR "App directory 'apps/${_app_dir}' has a CMakeLists.txt but is not listed in KADATH_APPS (apps/CMakeLists.txt). Add it to the list or to KADATH_APPS_EXCLUDED.")
    endif()
endforeach()
unset(_kadath_app_dirs)

if(TARGET bns_nosym_solve AND TARGET ns_solve)
    add_dependencies(bns_nosym_solve ns_solve)
endif()
