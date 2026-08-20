# Guard test: every committed convergence baseline must carry the input
# snapshot that defines the numerical problem being compared.

if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

set(BASELINE_DIR "${SOURCE_DIR}/tests/regression/baselines")

file(GLOB BASELINE_FILES
    "${BASELINE_DIR}/*.txt"
)

set(MISSING_INPUTS "")
foreach(BASELINE_FILE IN LISTS BASELINE_FILES)
    get_filename_component(BASELINE_NAME "${BASELINE_FILE}" NAME)
    if(BASELINE_NAME MATCHES "\\.stdout\\.txt$")
        continue()
    endif()

    get_filename_component(BASELINE_STEM "${BASELINE_FILE}" NAME_WE)
    set(INPUT_SNAPSHOT "${BASELINE_DIR}/${BASELINE_STEM}.toml")
    if(NOT EXISTS "${INPUT_SNAPSHOT}")
        list(APPEND MISSING_INPUTS "${BASELINE_NAME} -> ${BASELINE_STEM}.toml")
    endif()
endforeach()

if(MISSING_INPUTS)
    list(JOIN MISSING_INPUTS "\n" MISSING_TEXT)
    message(FATAL_ERROR
        "Regression convergence baselines are missing TOML input snapshots:\n${MISSING_TEXT}"
    )
endif()

message(STATUS "Regression baseline input guard passed: convergence baselines have TOML snapshots")
