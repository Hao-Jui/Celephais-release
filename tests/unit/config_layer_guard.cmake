set(FORBIDDEN_CONFIG_PATHS
    "include/for_Kadath/config/app_config"
    "src/AppConfig"
)

foreach(path IN LISTS FORBIDDEN_CONFIG_PATHS)
    if(EXISTS "${SOURCE_DIR}/${path}")
        message(FATAL_ERROR "Retired typed config path still exists: ${path}")
    endif()
endforeach()

set(FORBIDDEN_CONFIG_TOKENS
    "BUILD_APP_CONFIG_TESTS"
    "kadath_app_config"
    "src/AppConfig"
    "for_Kadath/config/app_config"
    "AppStartupConfig"
    "ConfigCompatAdapter"
    "ConfigTomlReader"
    "ConfigTomlWriter"
    "LegacyBridge"
    "load_bh2d_startup_config"
    "load_ns_startup_config"
    "load_ns2d_startup_config"
    "load_bns_startup_config"
    "load_bhns_startup_config"
)

file(GLOB_RECURSE CONFIG_LAYER_FILES
    "${SOURCE_DIR}/CMakeLists.txt"
    "${SOURCE_DIR}/apps/*.cpp"
    "${SOURCE_DIR}/apps/CMakeLists.txt"
    "${SOURCE_DIR}/apps/*/CMakeLists.txt"
    "${SOURCE_DIR}/include/*.hpp"
    "${SOURCE_DIR}/include/*.hh"
    "${SOURCE_DIR}/include/*.h"
    "${SOURCE_DIR}/src/*.cpp"
    "${SOURCE_DIR}/src/*.hpp"
)

foreach(file IN LISTS CONFIG_LAYER_FILES)
    file(READ "${file}" content)
    foreach(token IN LISTS FORBIDDEN_CONFIG_TOKENS)
        string(FIND "${content}" "${token}" found_at)
        if(NOT found_at EQUAL -1)
            message(FATAL_ERROR "Retired typed config token '${token}' found in ${file}")
        endif()
    endforeach()
endforeach()
