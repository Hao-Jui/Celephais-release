if(NOT DEFINED CELEPHAIS_SOURCE_DIR OR NOT DEFINED CONSUMER_SOURCE_DIR OR
   NOT DEFINED TEST_ROOT)
    message(FATAL_ERROR "solution package test requires source, consumer, and test paths")
endif()

if(MAKE_PROGRAM AND NOT IS_ABSOLUTE "${MAKE_PROGRAM}")
    find_program(CELEPHAIS_MAKE_PROGRAM_PATH
        NAMES "${MAKE_PROGRAM}"
        REQUIRED)
    set(MAKE_PROGRAM "${CELEPHAIS_MAKE_PROGRAM_PATH}")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")
set(CELEPHAIS_BUILD_DIR "${TEST_ROOT}/source-build")
set(STAGING_PREFIX "${TEST_ROOT}/staging")
set(INSTALL_PREFIX "${TEST_ROOT}/relocated")
set(CONSUMER_BUILD_DIR "${TEST_ROOT}/consumer-build")

if(TOOLCHAIN_FILE AND NOT IS_ABSOLUTE "${TOOLCHAIN_FILE}")
    set(TOOLCHAIN_FILE "${CELEPHAIS_SOURCE_DIR}/${TOOLCHAIN_FILE}")
endif()

# This gate deliberately starts from source.  Disabling package discovery makes
# any accidental production find_package(FFTW) call a configure-time failure.
set(SOURCE_CONFIGURE_COMMAND "${CMAKE_COMMAND}" -S "${CELEPHAIS_SOURCE_DIR}"
    -B "${CELEPHAIS_BUILD_DIR}" -G "${GENERATOR}"
    "-DBUILD_TESTING=OFF"
    "-DBUILD_APPS=OFF"
    "-DCELEPHAIS_ENABLE_FFTW_ORACLE=OFF"
    "-DCELEPHAIS_SUPPRESS_COMPILE_COMMANDS_LINK=ON"
    "-DCMAKE_DISABLE_FIND_PACKAGE_FFTW=TRUE"
    "-DFFTW_ROOT=${TEST_ROOT}/intentionally-missing-fftw"
    "-DCELEPHAIS_LIB_OUTPUT_DIR=${CELEPHAIS_BUILD_DIR}/lib"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
if(MAKE_PROGRAM)
    list(APPEND SOURCE_CONFIGURE_COMMAND "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
if(BUILD_TYPE)
    list(APPEND SOURCE_CONFIGURE_COMMAND "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
endif()
if(TOOLCHAIN_FILE)
    list(APPEND SOURCE_CONFIGURE_COMMAND "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}")
endif()
if(BLA_VENDOR_HINT)
    list(APPEND SOURCE_CONFIGURE_COMMAND "-DBLA_VENDOR=${BLA_VENDOR_HINT}")
endif()

execute_process(
    COMMAND ${SOURCE_CONFIGURE_COMMAND}
    RESULT_VARIABLE source_configure_result
    OUTPUT_VARIABLE source_configure_stdout
    ERROR_VARIABLE source_configure_stderr
)
if(NOT source_configure_result EQUAL 0)
    message(FATAL_ERROR
        "FFTW-disabled Celephais configure failed (${source_configure_result})\n"
        "${source_configure_stdout}\n${source_configure_stderr}")
endif()

set(SOURCE_BUILD_COMMAND "${CMAKE_COMMAND}" --build "${CELEPHAIS_BUILD_DIR}"
    --target celephais --parallel 4)
if(BUILD_TYPE)
    list(APPEND SOURCE_BUILD_COMMAND --config "${BUILD_TYPE}")
endif()
execute_process(
    COMMAND ${SOURCE_BUILD_COMMAND}
    RESULT_VARIABLE source_build_result
    OUTPUT_VARIABLE source_build_stdout
    ERROR_VARIABLE source_build_stderr
)
if(NOT source_build_result EQUAL 0)
    message(FATAL_ERROR
        "FFTW-disabled Celephais build failed (${source_build_result})\n"
        "${source_build_stdout}\n${source_build_stderr}")
endif()

set(INSTALL_COMMAND "${CMAKE_COMMAND}" --install "${CELEPHAIS_BUILD_DIR}"
    --prefix "${STAGING_PREFIX}")
if(BUILD_TYPE)
    list(APPEND INSTALL_COMMAND --config "${BUILD_TYPE}")
endif()
execute_process(
    COMMAND ${INSTALL_COMMAND}
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_stdout
    ERROR_VARIABLE install_stderr
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR
        "Celephais install failed (${install_result})\n${install_stdout}\n${install_stderr}")
endif()

file(RENAME "${STAGING_PREFIX}" "${INSTALL_PREFIX}")

file(GLOB_RECURSE installed_package_files
    "${INSTALL_PREFIX}/*/cmake/Celephais/*.cmake")
if(NOT installed_package_files)
    message(FATAL_ERROR "CelephaisTargets.cmake was not installed")
endif()
foreach(package_file IN LISTS installed_package_files)
    file(READ "${package_file}" package_contents)
    string(FIND "${package_contents}" "${CELEPHAIS_SOURCE_DIR}" source_path_position)
    string(FIND "${package_contents}" "${CELEPHAIS_BUILD_DIR}" build_path_position)
    string(FIND "${package_contents}" "${STAGING_PREFIX}" staging_path_position)
    if(NOT source_path_position EQUAL -1 OR NOT build_path_position EQUAL -1)
        message(FATAL_ERROR "installed package is not relocatable: ${package_file}")
    endif()
    if(NOT staging_path_position EQUAL -1)
        message(FATAL_ERROR "installed package retained its original prefix: ${package_file}")
    endif()
endforeach()

set(CONFIGURE_COMMAND "${CMAKE_COMMAND}" -S "${CONSUMER_SOURCE_DIR}"
    -B "${CONSUMER_BUILD_DIR}" -G "${GENERATOR}"
    "-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}"
    "-DCMAKE_DISABLE_FIND_PACKAGE_FFTW=TRUE"
    "-DFFTW_ROOT=${TEST_ROOT}/intentionally-missing-fftw"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
if(MAKE_PROGRAM)
    list(APPEND CONFIGURE_COMMAND "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
if(BUILD_TYPE)
    list(APPEND CONFIGURE_COMMAND "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
endif()
if(TOOLCHAIN_FILE)
    list(APPEND CONFIGURE_COMMAND "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}")
endif()
if(BLA_VENDOR_HINT)
    list(APPEND CONFIGURE_COMMAND "-DBLA_VENDOR=${BLA_VENDOR_HINT}")
endif()

execute_process(
    COMMAND ${CONFIGURE_COMMAND}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR
        "external consumer configure failed (${configure_result})\n"
        "${configure_stdout}\n${configure_stderr}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${CONSUMER_BUILD_DIR}" --verbose
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_stdout
    ERROR_VARIABLE build_stderr
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR
        "external consumer build failed (${build_result})\n${build_stdout}\n${build_stderr}")
endif()

string(TOLOWER "${build_stdout}\n${build_stderr}" link_output)
if(link_output MATCHES "libfftw|(^|[ ;])-lfftw|fftw::")
    message(FATAL_ERROR
        "Celephais::solution leaked FFTW into the consumer link\n${link_output}")
endif()
foreach(forbidden IN ITEMS mumps scalapack gfortran quadmath)
    string(FIND "${link_output}" "${forbidden}" forbidden_position)
    if(NOT forbidden_position EQUAL -1)
        message(FATAL_ERROR
            "Celephais::solution leaked '${forbidden}' into the consumer link\n${link_output}")
    endif()
endforeach()

set(CONSUMER_EXE "${CONSUMER_BUILD_DIR}/solution_reader")
if(WIN32)
    set(CONSUMER_EXE "${CONSUMER_EXE}.exe")
endif()
if(NOT EXISTS "${CONSUMER_EXE}" AND BUILD_TYPE)
    set(CONSUMER_EXE "${CONSUMER_BUILD_DIR}/${BUILD_TYPE}/solution_reader")
    if(WIN32)
        string(APPEND CONSUMER_EXE ".exe")
    endif()
endif()
if(NOT EXISTS "${CONSUMER_EXE}")
    message(FATAL_ERROR "external consumer executable not found")
endif()

file(GLOB installed_archives "${INSTALL_PREFIX}/*/libcelephais.a")
if(NOT installed_archives)
    message(FATAL_ERROR "installed Celephais archive not found")
endif()
if(NM_TOOL)
    execute_process(
        COMMAND "${NM_TOOL}" -u ${installed_archives}
        RESULT_VARIABLE nm_result
        OUTPUT_VARIABLE nm_stdout
        ERROR_VARIABLE nm_stderr)
    string(TOLOWER "${nm_stdout}\n${nm_stderr}" undefined_symbols)
    if(NOT nm_result EQUAL 0 OR undefined_symbols MATCHES "fftw")
        message(FATAL_ERROR
            "installed archive FFTW-symbol gate failed (${nm_result})\n${nm_stdout}\n${nm_stderr}")
    endif()
endif()

if(APPLE)
    execute_process(COMMAND otool -L "${CONSUMER_EXE}"
        RESULT_VARIABLE dependency_result
        OUTPUT_VARIABLE dependency_stdout
        ERROR_VARIABLE dependency_stderr)
elseif(UNIX)
    execute_process(COMMAND ldd "${CONSUMER_EXE}"
        RESULT_VARIABLE dependency_result
        OUTPUT_VARIABLE dependency_stdout
        ERROR_VARIABLE dependency_stderr)
endif()
if(DEFINED dependency_result)
    string(TOLOWER "${dependency_stdout}\n${dependency_stderr}" dependencies)
    if(NOT dependency_result EQUAL 0 OR dependencies MATCHES "libfftw")
        message(FATAL_ERROR
            "external consumer FFTW dependency gate failed (${dependency_result})\n"
            "${dependency_stdout}\n${dependency_stderr}")
    endif()
endif()

# The ignored full 3D fixture is optional in partial/public checkouts. When it
# is present, prove that the installed package loads and evaluates it.
if(EXISTS "${FIXTURE}")
    execute_process(
        COMMAND "${CONSUMER_EXE}" ns_nosym "${FIXTURE}" 0 0 0
        RESULT_VARIABLE run_result
        OUTPUT_VARIABLE run_stdout
        ERROR_VARIABLE run_stderr
    )
    if(NOT run_result EQUAL 0 OR NOT run_stdout MATCHES "conf=")
        message(FATAL_ERROR
            "external consumer load failed (${run_result})\n${run_stdout}\n${run_stderr}")
    endif()

    execute_process(
        COMMAND "${CONSUMER_EXE}" bns_nosym "${FIXTURE}" 0 0 0
        RESULT_VARIABLE family_refusal_result
        OUTPUT_VARIABLE family_refusal_stdout
        ERROR_VARIABLE family_refusal_stderr
    )
    if(family_refusal_result EQUAL 0 OR
       NOT family_refusal_stderr MATCHES "SolutionKind does not match the checkpoint family")
        message(FATAL_ERROR
            "wrong-family refusal gate failed (${family_refusal_result})\n"
            "${family_refusal_stdout}\n${family_refusal_stderr}")
    endif()

    string(REGEX REPLACE "\\.toml$" ".dat" fixture_data "${FIXTURE}")
    if(EXISTS "${fixture_data}")
        set(explicit_gr_config "${TEST_ROOT}/explicit_gr.toml")
        set(explicit_gr_data "${TEST_ROOT}/explicit_gr.dat")
        file(READ "${FIXTURE}" explicit_gr_contents)
        string(APPEND explicit_gr_contents "\n[gravity]\ntheory = \"GR\"\n")
        file(WRITE "${explicit_gr_config}" "${explicit_gr_contents}")
        file(COPY_FILE "${fixture_data}" "${explicit_gr_data}")
        execute_process(
            COMMAND "${CONSUMER_EXE}" ns_nosym "${explicit_gr_config}" 0 0 0
            RESULT_VARIABLE explicit_gr_result
            OUTPUT_VARIABLE explicit_gr_stdout
            ERROR_VARIABLE explicit_gr_stderr
        )
        if(NOT explicit_gr_result EQUAL 0 OR NOT explicit_gr_stdout MATCHES "conf=")
            message(FATAL_ERROR
                "explicit GR checkpoint load failed (${explicit_gr_result})\n"
                "${explicit_gr_stdout}\n${explicit_gr_stderr}")
        endif()
    endif()

    execute_process(
        COMMAND "${CONSUMER_EXE}" ns_nosym "${FIXTURE}" nan 0 0
        RESULT_VARIABLE finite_refusal_result
        OUTPUT_VARIABLE finite_refusal_stdout
        ERROR_VARIABLE finite_refusal_stderr
    )
    if(finite_refusal_result EQUAL 0 OR
       NOT finite_refusal_stderr MATCHES "point coordinates must be finite")
        message(FATAL_ERROR
            "non-finite point refusal gate failed (${finite_refusal_result})\n"
            "${finite_refusal_stdout}\n${finite_refusal_stderr}")
    endif()
endif()


execute_process(
    COMMAND "${CONSUMER_EXE}" ns_nosym "${TEST_ROOT}/missing.toml" 0 0 0
    RESULT_VARIABLE refusal_result
    OUTPUT_VARIABLE refusal_stdout
    ERROR_VARIABLE refusal_stderr
)
if(refusal_result EQUAL 0 OR NOT refusal_stderr MATCHES "companion config not found")
    message(FATAL_ERROR
        "missing-file refusal gate failed (${refusal_result})\n"
        "${refusal_stdout}\n${refusal_stderr}")
endif()

message(STATUS "external Celephais::solution package consumer passed")
