# CMakeExec.cmake — helper included by apps/CMakeLists.txt
# All dependencies (FFTW, LAPACK, MPI, MUMPS, SCALAPACK) come
# transitively from the celephais_solver target. Apps only need to call
# kadath_add_exec() and set APP_NAME before doing so.

# Optional compiler cache to speed up rebuilds.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

set(KADATH_APP_OUTPUT_BASE "${CELEPHAIS_ROOT_DIR}/apps" CACHE PATH
    "Base directory for app binaries; override for parallel-safe per-job builds")

function(kadath_set_app_runtime_output TARGET)
    if(NOT DEFINED APP_NAME OR APP_NAME STREQUAL "")
        message(FATAL_ERROR "APP_NAME must be set before adding app target '${TARGET}'")
    endif()

    set_target_properties(${TARGET} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${KADATH_APP_OUTPUT_BASE}/${APP_NAME}/bin/$<CONFIG>"
    )
endfunction()

# kadath_add_exec(TARGET SOURCE)
# Creates an executable that links against the full Celephais solver stack.
# APP_NAME must be set by the calling CMakeLists.txt so Release binaries land in
#   ${CELEPHAIS_ROOT_DIR}/apps/${APP_NAME}/bin/Release/
function(kadath_add_exec TARGET SOURCE)
    add_executable(${TARGET} ${SOURCE})
    target_link_libraries(${TARGET} PRIVATE celephais_solver)
    target_compile_options(${TARGET} PRIVATE
        -Wall -Wextra -Wno-unused-parameter -Woverloaded-virtual
    )
    # ${CELEPHAIS_ROOT_DIR}/include  -> central Apps/* + For_Kadath/* headers
    # ${CMAKE_CURRENT_SOURCE_DIR} -> the calling app's own dir, so single-app
    #   headers co-located under apps/<APP>/ (e.g. "solver/solver.hpp", policy
    #   headers) resolve without living in the shared include/ tree.
    target_include_directories(${TARGET} PRIVATE
        ${CELEPHAIS_ROOT_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    kadath_set_app_runtime_output(${TARGET})
endfunction()

function(kadath_add_configured_reader TARGET SPACE_HEADER ENTRYPOINT USAGE)
    set(KADATH_READER_SPACE_HEADER "${SPACE_HEADER}")
    set(KADATH_READER_ENTRYPOINT "${ENTRYPOINT}")
    set(KADATH_READER_USAGE "${USAGE}")
    set(GENERATED_READER_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_main.cpp")
    configure_file(
        "${CELEPHAIS_ROOT_DIR}/apps/shared/reader_main.cpp.in"
        "${GENERATED_READER_SOURCE}"
        @ONLY
    )
    kadath_add_exec(${TARGET} "${GENERATED_READER_SOURCE}")
    target_sources(${TARGET} PRIVATE
        "${CELEPHAIS_ROOT_DIR}/include/Apps/Diagnostics/reader_entrypoints.hpp"
        "${CELEPHAIS_ROOT_DIR}/include/Apps/Diagnostics/reader_impl.hpp"
    )
endfunction()

function(kadath_add_single_eos_reader TARGET SPACE_HEADER SPACE_TYPE FORMALISM USAGE)
    set(ENTRYPOINT "KadathApps::reader_single_eos_entrypoint<${SPACE_TYPE}, ${FORMALISM}>")
    kadath_add_configured_reader(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}" "${USAGE}")
endfunction()

function(kadath_add_2d_eos_reader TARGET SPACE_HEADER SPACE_TYPE FORMALISM USAGE)
    set(ENTRYPOINT "KadathApps::reader_2d_eos_entrypoint<${SPACE_TYPE}, ${FORMALISM}>")
    kadath_add_configured_reader(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}" "${USAGE}")
endfunction()

function(kadath_add_single_bh_reader TARGET SPACE_HEADER SPACE_TYPE USAGE)
    set(ENTRYPOINT "KadathApps::reader_single_bh_entrypoint<${SPACE_TYPE}>")
    kadath_add_configured_reader(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}" "${USAGE}")
endfunction()

function(kadath_add_binary_eos_reader TARGET SPACE_HEADER SPACE_TYPE FORMALISM USAGE)
    set(ENTRYPOINT "KadathApps::reader_binary_eos_entrypoint<${SPACE_TYPE}, ${FORMALISM}>")
    kadath_add_configured_reader(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}" "${USAGE}")
endfunction()

function(kadath_add_bhns_eos_reader TARGET SPACE_HEADER SPACE_TYPE USAGE)
    set(ENTRYPOINT "KadathApps::reader_bhns_eos_entrypoint<${SPACE_TYPE}>")
    kadath_add_configured_reader(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}" "${USAGE}")
endfunction()

function(kadath_add_three_body_eos_reader TARGET SPACE_HEADER SPACE_TYPE FORMALISM USAGE)
    set(ENTRYPOINT "KadathApps::reader_three_body_eos_entrypoint<${SPACE_TYPE}, ${FORMALISM}>")
    kadath_add_configured_reader(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}" "${USAGE}")
endfunction()

function(kadath_add_configured_xz_snapshot TARGET SPACE_HEADER ENTRYPOINT)
    set(KADATH_XZ_SNAPSHOT_SPACE_HEADER "${SPACE_HEADER}")
    set(KADATH_XZ_SNAPSHOT_ENTRYPOINT "${ENTRYPOINT}")
    set(GENERATED_XZ_SNAPSHOT_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_main.cpp")
    configure_file(
        "${CELEPHAIS_ROOT_DIR}/apps/shared/xz_snapshot_main.cpp.in"
        "${GENERATED_XZ_SNAPSHOT_SOURCE}"
        @ONLY
    )
    kadath_add_exec(${TARGET} "${GENERATED_XZ_SNAPSHOT_SOURCE}")
    target_sources(${TARGET} PRIVATE
        "${CELEPHAIS_ROOT_DIR}/include/Apps/Diagnostics/xz_snapshot_impl.hpp"
    )
endfunction()

function(kadath_add_binary_xz_snapshot TARGET SPACE_HEADER SPACE_TYPE FIELD_TRAITS)
    set(ENTRYPOINT "KadathApps::xz_snapshot_binary_main<${SPACE_TYPE}, ${FIELD_TRAITS}>")
    kadath_add_configured_xz_snapshot(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()

function(kadath_add_ns_xz_snapshot TARGET SPACE_HEADER SPACE_TYPE FIELD_TRAITS)
    set(ENTRYPOINT "KadathApps::xz_snapshot_ns_main<${SPACE_TYPE}, ${FIELD_TRAITS}>")
    kadath_add_configured_xz_snapshot(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()

function(kadath_add_three_body_xz_snapshot TARGET SPACE_HEADER SPACE_TYPE FIELD_TRAITS)
    set(ENTRYPOINT "KadathApps::xz_snapshot_three_body_main<${SPACE_TYPE}, ${FIELD_TRAITS}>")
    kadath_add_configured_xz_snapshot(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()

# xy_snapshot — orbital-plane (z == 0) counterpart of xz_snapshot. Same
# configure-a-thin-main pattern: a generated main instantiates one of the
# templated entry points in Apps/Diagnostics/xy_snapshot_impl.hpp with the app's
# Space type and field-set trait. The Release binary lands in
# apps/${APP_NAME}/bin/Release/ as xy_snapshot, alongside xz_snapshot.
function(kadath_add_configured_xy_snapshot TARGET SPACE_HEADER ENTRYPOINT)
    set(KADATH_XY_SNAPSHOT_SPACE_HEADER "${SPACE_HEADER}")
    set(KADATH_XY_SNAPSHOT_ENTRYPOINT "${ENTRYPOINT}")
    set(GENERATED_XY_SNAPSHOT_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_main.cpp")
    configure_file(
        "${CELEPHAIS_ROOT_DIR}/apps/shared/xy_snapshot_main.cpp.in"
        "${GENERATED_XY_SNAPSHOT_SOURCE}"
        @ONLY
    )
    kadath_add_exec(${TARGET} "${GENERATED_XY_SNAPSHOT_SOURCE}")
    target_sources(${TARGET} PRIVATE
        "${CELEPHAIS_ROOT_DIR}/include/Apps/Diagnostics/xy_snapshot_impl.hpp"
    )
endfunction()

function(kadath_add_binary_xy_snapshot TARGET SPACE_HEADER SPACE_TYPE FIELD_TRAITS)
    set(ENTRYPOINT "KadathApps::xy_snapshot_binary_main<${SPACE_TYPE}, ${FIELD_TRAITS}>")
    kadath_add_configured_xy_snapshot(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()

function(kadath_add_ns_xy_snapshot TARGET SPACE_HEADER SPACE_TYPE FIELD_TRAITS)
    set(ENTRYPOINT "KadathApps::xy_snapshot_ns_main<${SPACE_TYPE}, ${FIELD_TRAITS}>")
    kadath_add_configured_xy_snapshot(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()

function(kadath_add_three_body_xy_snapshot TARGET SPACE_HEADER SPACE_TYPE FIELD_TRAITS)
    set(ENTRYPOINT "KadathApps::xy_snapshot_three_body_main<${SPACE_TYPE}, ${FIELD_TRAITS}>")
    kadath_add_configured_xy_snapshot(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()

# coeff_read — per-app spectral coefficient dump (reusable successor of the
# BNS_nosym-only tail_probe). Same configure-a-thin-main pattern as the reader /
# xz_snapshot diagnostics: a generated main instantiates one of the templated
# entry points in Apps/Diagnostics/coeff_reader_impl.hpp with the app's Space type
# and field-set trait. The Release binary lands in apps/${APP_NAME}/bin/Release/.
function(kadath_add_configured_coeff_reader TARGET SPACE_HEADER ENTRYPOINT)
    set(KADATH_COEFF_READER_SPACE_HEADER "${SPACE_HEADER}")
    set(KADATH_COEFF_READER_ENTRYPOINT "${ENTRYPOINT}")
    set(GENERATED_COEFF_READER_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_main.cpp")
    configure_file(
        "${CELEPHAIS_ROOT_DIR}/apps/shared/coeff_reader_main.cpp.in"
        "${GENERATED_COEFF_READER_SOURCE}"
        @ONLY
    )
    kadath_add_exec(${TARGET} "${GENERATED_COEFF_READER_SOURCE}")
    target_sources(${TARGET} PRIVATE
        "${CELEPHAIS_ROOT_DIR}/include/Apps/Diagnostics/coeff_reader_impl.hpp"
    )
endfunction()

function(kadath_add_binary_coeff_reader TARGET SPACE_HEADER SPACE_TYPE FIELD_TRAITS)
    set(ENTRYPOINT "KadathApps::coeff_reader_binary_main<${SPACE_TYPE}, ${FIELD_TRAITS}>")
    kadath_add_configured_coeff_reader(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()

function(kadath_add_ns_coeff_reader TARGET SPACE_HEADER SPACE_TYPE FIELD_TRAITS)
    set(ENTRYPOINT "KadathApps::coeff_reader_ns_main<${SPACE_TYPE}, ${FIELD_TRAITS}>")
    kadath_add_configured_coeff_reader(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()

function(kadath_add_three_body_coeff_reader TARGET SPACE_HEADER SPACE_TYPE FIELD_TRAITS)
    set(ENTRYPOINT "KadathApps::coeff_reader_three_body_main<${SPACE_TYPE}, ${FIELD_TRAITS}>")
    kadath_add_configured_coeff_reader(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()

# x_proflle — raw saved-field profiles along the positive x-axis for
# axisymmetric neutron-star and black-hole solutions. A generated main selects
# the app's positional on-disk field layout. Release binaries land in
# apps/${APP_NAME}/bin/Release/ as x_proflle.
function(kadath_add_configured_x_profile TARGET SPACE_HEADER ENTRYPOINT)
    set(KADATH_X_PROFILE_SPACE_HEADER "${SPACE_HEADER}")
    set(KADATH_X_PROFILE_ENTRYPOINT "${ENTRYPOINT}")
    set(GENERATED_X_PROFILE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_main.cpp")
    configure_file(
        "${CELEPHAIS_ROOT_DIR}/apps/shared/x_profile_main.cpp.in"
        "${GENERATED_X_PROFILE_SOURCE}"
        @ONLY
    )
    kadath_add_exec(${TARGET} "${GENERATED_X_PROFILE_SOURCE}")
    target_sources(${TARGET} PRIVATE
        "${CELEPHAIS_ROOT_DIR}/include/Apps/Diagnostics/x_profile.hpp"
    )
endfunction()

function(kadath_add_ns2d_x_profile TARGET SPACE_HEADER SPACE_TYPE FIELD_LAYOUT)
    set(ENTRYPOINT "KadathApps::x_profile_ns_main<${SPACE_TYPE}, ${FIELD_LAYOUT}>")
    kadath_add_configured_x_profile(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()

function(kadath_add_bh2d_x_profile TARGET SPACE_HEADER SPACE_TYPE FIELD_LAYOUT)
    set(ENTRYPOINT "KadathApps::x_profile_bh_main<${SPACE_TYPE}, ${FIELD_LAYOUT}>")
    kadath_add_configured_x_profile(${TARGET} "${SPACE_HEADER}" "${ENTRYPOINT}")
endfunction()
