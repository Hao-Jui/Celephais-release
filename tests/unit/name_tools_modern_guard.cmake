# Guard test: fail loud if the modern value-returning name_tools helpers
# get removed, renamed, or have their declared signatures regressed.
#
# Pure regex match against include/for_Kadath/utilities/name_tools.hpp. No runtime build.
# The legacy C-helpers (trim_spaces, get_parts, is_tensor, etc.) are intentionally
# NOT guarded here — they may be deleted later when call site migration completes.

set(NAME_TOOLS_HPP "${SOURCE_DIR}/include/for_Kadath/utilities/name_tools.hpp")

if(NOT EXISTS "${NAME_TOOLS_HPP}")
    message(FATAL_ERROR "name_tools.hpp not found at ${NAME_TOOLS_HPP}")
endif()

file(READ "${NAME_TOOLS_HPP}" NAME_TOOLS_TEXT)

# Required structs. Match field-by-field so a struct with a renamed/missing
# field is also rejected.
set(REQUIRED_STRUCT_PATTERNS
    "struct[ \t]+BinarySplit"
    "BinarySplit[^;]*\n[^}]*std::string[ \t]+lhs"
    "BinarySplit[^;]*\n[^}]*std::string[ \t]+rhs"
    "struct[ \t]+TensorParse"
    "TensorParse[^;]*\n[^}]*std::string[ \t]+base_name"
    "TensorParse[^;]*\n[^}]*int[ \t]+valence"
    "TensorParse[^;]*\n[^}]*std::string[ \t]+indices"
    "TensorParse[^;]*\n[^}]*std::vector<int>[ \t]+types"
)

# Required free function declarations. Match return type + name + first arg
# (string_view input) — leaves room for argument-name renames but locks types.
set(REQUIRED_FUNCTION_PATTERNS
    "std::optional<BinarySplit>[ \t]+split_binary_operator\\([ \t]*std::string_view"
    "std::pair<std::string,[ \t]*std::string>[ \t]+split_first\\([ \t]*std::string_view"
    "std::optional<BinarySplit>[ \t]+split_nth\\([ \t]*std::string_view"
    "std::optional<TensorParse>[ \t]+parse_tensor\\([ \t]*std::string_view"
)

# Required header includes (modern helpers depend on them).
set(REQUIRED_INCLUDE_PATTERNS
    "#include[ \t]+<optional>"
    "#include[ \t]+<string_view>"
    "#include[ \t]+<utility>"
    "#include[ \t]+<vector>"
)

foreach(PATTERN IN LISTS REQUIRED_STRUCT_PATTERNS REQUIRED_FUNCTION_PATTERNS REQUIRED_INCLUDE_PATTERNS)
    if(NOT NAME_TOOLS_TEXT MATCHES "${PATTERN}")
        message(FATAL_ERROR
            "name_tools.hpp lost a required modern-helper element: ${PATTERN}\n"
            "If you intentionally removed a helper, also remove the matching pattern from this guard."
        )
    endif()
endforeach()

message(STATUS "name_tools modern helpers present: 4 functions + 2 structs + 4 includes")
