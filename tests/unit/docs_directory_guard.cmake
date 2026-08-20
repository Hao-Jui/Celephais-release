set(DOCS_DIR "${SOURCE_DIR}/docs")
set(README "${SOURCE_DIR}/README.md")

foreach(RETIRED_DIR
        "${DOCS_DIR}/adr"
        "${DOCS_DIR}/api"
        "${DOCS_DIR}/solver")
    if(IS_DIRECTORY "${RETIRED_DIR}")
        message(FATAL_ERROR
            "docs/ must not contain retired generated/archive subdirectory: ${RETIRED_DIR}")
    endif()
endforeach()

execute_process(
    COMMAND git -C "${SOURCE_DIR}" ls-files --cached --others --exclude-standard "docs/"
    OUTPUT_VARIABLE GIT_DOC_FILES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_RC
)

if(NOT GIT_RC EQUAL 0 OR GIT_DOC_FILES STREQUAL "")
    file(GLOB_RECURSE DOC_FILES
        LIST_DIRECTORIES false
        RELATIVE "${SOURCE_DIR}"
        "${DOCS_DIR}/*")
else()
    string(REPLACE "\n" ";" DOC_FILES "${GIT_DOC_FILES}")
endif()

file(READ "${README}" README_TEXT)

foreach(DOC_FILE IN LISTS DOC_FILES)
    get_filename_component(DOC_EXT "${DOC_FILE}" EXT)
    if(NOT DOC_EXT STREQUAL ".md" AND NOT DOC_EXT STREQUAL ".html")
        message(FATAL_ERROR
            "docs/ may only contain markdown files indexed by top-level README.md; found ${DOC_FILE}")
    endif()

    string(FIND "${README_TEXT}" "${DOC_FILE}" DOC_INDEX)
    if(DOC_INDEX EQUAL -1)
        message(FATAL_ERROR
            "docs markdown is not indexed by top-level README.md: ${DOC_FILE}")
    endif()
endforeach()
