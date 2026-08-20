add_subdirectory(unit)
add_subdirectory(smoke)
add_subdirectory(eos)

if(NOT CELEPHAIS_ENABLE_FFTW_ORACLE)
    add_test(NAME solution_package_consumer
        COMMAND ${CMAKE_COMMAND}
            -DCELEPHAIS_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            -DCONSUMER_SOURCE_DIR=${PROJECT_SOURCE_DIR}/examples/solution_consumer
            -DTEST_ROOT=${PROJECT_BINARY_DIR}/tests/solution_package_consumer
            -DGENERATOR=${CMAKE_GENERATOR}
            -DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
            -DBUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DTOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DNM_TOOL=${CMAKE_NM}
            -DBLA_VENDOR_HINT=${BLA_VENDOR}
            -DFIXTURE=${PROJECT_SOURCE_DIR}/data/convergence/converged_NS_NOSYM_UNIROT.dd2.1.35.0.3.z30.0.11.toml
            -P ${CMAKE_CURRENT_SOURCE_DIR}/solution_package_consumer.cmake
    )
    set_tests_properties(solution_package_consumer PROPERTIES
        LABELS "integration;solution"
        TIMEOUT 900
    )
endif()
