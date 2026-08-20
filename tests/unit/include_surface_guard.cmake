# Guard test: Kadath exposes include/ as its public header root. For_Kadath
# module directories must stay addressable through qualified includes instead
# of being re-added as public flat include roots.

if(NOT DEFINED SOURCE_DIR)
    message(FATAL_ERROR "SOURCE_DIR is required")
endif()

file(READ "${SOURCE_DIR}/CMakeLists.txt" ROOT_CMAKE)

set(FORBIDDEN_PUBLIC_INCLUDE_TOKENS
    "include/For_Kadath/Kadath_point_h"
    "include/For_Kadath/Config"
    "include/For_Kadath/Domain"
    "include/For_Kadath/IO"
    "include/For_Kadath/Matrice"
    "include/For_Kadath/Array"
    "include/For_Kadath/Space"
    "include/For_Kadath/Third_Party"
    "include/For_Kadath/Utilities"
    "include/For_Kadath/System_of_eqs"
    "include/For_Kadath/List_comp"
    "include/For_Kadath/Metric"
    "include/For_Kadath/Ope_eq"
    "include/For_Kadath/Param"
    "include/For_Kadath/Scalar"
    "include/For_Kadath/Tensor"
    "include/For_Kadath/Term_eq"
    "include/For_Kadath/Val_domain"
    "include/For_Kadath/Base_spectral"
    "include/For_Kadath/Base_tensor"
    "include/For_Kadath/Coef"
    "include/For_Kadath/Ope_1d"
    "src/Array"
    "KADATH_TOMLPLUSPLUS_INCLUDE_DIR"
)

foreach(TOKEN IN LISTS FORBIDDEN_PUBLIC_INCLUDE_TOKENS)
    string(FIND "${ROOT_CMAKE}" "${TOKEN}" FOUND_AT)
    if(NOT FOUND_AT EQUAL -1)
        message(FATAL_ERROR
            "Legacy public include surface '${TOKEN}' is present in CMakeLists.txt. "
            "Use qualified includes through ${SOURCE_DIR}/include instead."
        )
    endif()
endforeach()

file(GLOB_RECURSE FOR_KADATH_HEADERS
    "${SOURCE_DIR}/include/For_Kadath/*.h"
    "${SOURCE_DIR}/include/For_Kadath/*.hh"
    "${SOURCE_DIR}/include/For_Kadath/*.hpp"
    "${SOURCE_DIR}/include/For_Kadath/*.inl"
)

set(LEGACY_FOR_KADATH_INCLUDE_ROOTS "${SOURCE_DIR}/include/For_Kadath")
set(FOR_KADATH_BASENAMES "")
foreach(HEADER IN LISTS FOR_KADATH_HEADERS)
    get_filename_component(BASENAME "${HEADER}" NAME)
    list(APPEND FOR_KADATH_BASENAMES "${BASENAME}")
    get_filename_component(HEADER_DIR "${HEADER}" DIRECTORY)
    list(APPEND LEGACY_FOR_KADATH_INCLUDE_ROOTS "${HEADER_DIR}")
endforeach()
list(REMOVE_DUPLICATES FOR_KADATH_BASENAMES)
list(REMOVE_DUPLICATES LEGACY_FOR_KADATH_INCLUDE_ROOTS)

file(GLOB_RECURSE SCANNED_FILES
    "${SOURCE_DIR}/apps/*.cpp"
    "${SOURCE_DIR}/apps/*.h"
    "${SOURCE_DIR}/apps/*.hh"
    "${SOURCE_DIR}/apps/*.hpp"
    "${SOURCE_DIR}/apps/*.ipp"
    "${SOURCE_DIR}/include/*.h"
    "${SOURCE_DIR}/include/*.hh"
    "${SOURCE_DIR}/include/*.hpp"
    "${SOURCE_DIR}/include/*.inl"
    "${SOURCE_DIR}/src/*.cpp"
    "${SOURCE_DIR}/src/*.h"
    "${SOURCE_DIR}/src/*.hh"
    "${SOURCE_DIR}/src/*.hpp"
    "${SOURCE_DIR}/src/*.ipp"
    "${SOURCE_DIR}/tests/*.cpp"
    "${SOURCE_DIR}/tests/*.h"
    "${SOURCE_DIR}/tests/*.hh"
    "${SOURCE_DIR}/tests/*.hpp"
    "${SOURCE_DIR}/tests/*.ipp"
)

set(VIOLATIONS "")
foreach(SCANNED_FILE IN LISTS SCANNED_FILES)
    file(STRINGS "${SCANNED_FILE}" INCLUDE_LINES REGEX "^[ \t]*#[ \t]*include[ \t]+\"[^\"]+\"")
    foreach(INCLUDE_LINE IN LISTS INCLUDE_LINES)
        string(REGEX REPLACE "^[ \t]*#[ \t]*include[ \t]+\"([^\"]+)\".*$" "\\1" INCLUDE_PATH "${INCLUDE_LINE}")
        if(INCLUDE_PATH MATCHES "^For_Kadath/")
            continue()
        endif()

        get_filename_component(INCLUDE_BASENAME "${INCLUDE_PATH}" NAME)
        list(FIND FOR_KADATH_BASENAMES "${INCLUDE_BASENAME}" MATCH_INDEX)
        if(MATCH_INDEX EQUAL -1)
            continue()
        endif()

        get_filename_component(SCANNED_DIR "${SCANNED_FILE}" DIRECTORY)
        if(EXISTS "${SCANNED_DIR}/${INCLUDE_PATH}")
            continue()
        endif()

        foreach(LEGACY_ROOT IN LISTS LEGACY_FOR_KADATH_INCLUDE_ROOTS)
            if(EXISTS "${LEGACY_ROOT}/${INCLUDE_PATH}")
                list(APPEND VIOLATIONS "${SCANNED_FILE}: ${INCLUDE_LINE}")
                break()
            endif()
        endforeach()
    endforeach()
endforeach()

if(VIOLATIONS)
    list(JOIN VIOLATIONS "\n" VIOLATION_TEXT)
    message(FATAL_ERROR
        "Legacy non-local For_Kadath includes remain. Include through For_Kadath/... instead:\n${VIOLATION_TEXT}"
    )
endif()

message(STATUS "Kadath include surface guard passed: public root is include/ and no legacy non-local For_Kadath includes remain")
