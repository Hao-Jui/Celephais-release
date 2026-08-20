set(REQUIRED_FILES
    "cmake/env/env.sh"
    "cmake/env/celephais_env_common.sh"
    "cmake/toolchains/env.cmake"
    "tools/celephais_preflight.sh"
    "tools/generate_submission_template.sh"
)

foreach(path IN LISTS REQUIRED_FILES)
    if(NOT EXISTS "${SOURCE_DIR}/${path}")
        message(FATAL_ERROR "Missing generic onboarding file: ${path}")
    endif()
endforeach()

file(READ "${SOURCE_DIR}/CMakePresets.json" PRESETS_TEXT)
if(NOT PRESETS_TEXT MATCHES "\"name\"[ \t\r\n]*:[ \t\r\n]*\"env\"")
    message(FATAL_ERROR "CMakePresets.json must define generic env preset")
endif()
if(NOT PRESETS_TEXT MATCHES "\"toolchainFile\"[ \t\r\n]*:[ \t\r\n]*\"cmake/toolchains/env\\.cmake\"")
    message(FATAL_ERROR "env preset must use cmake/toolchains/env.cmake")
endif()

file(READ "${SOURCE_DIR}/tools/celephais_preflight.sh" PREFLIGHT_TEXT)
foreach(token "READY=" "TARGET_CLASS=" "BLOCKER=" "NEXT=")
    if(NOT PREFLIGHT_TEXT MATCHES "${token}")
        message(FATAL_ERROR "celephais_preflight.sh must emit ${token}")
    endif()
endforeach()

file(READ "${SOURCE_DIR}/tools/generate_submission_template.sh" SUBMISSION_GENERATOR_TEXT)
if(NOT SUBMISSION_GENERATOR_TEXT MATCHES "CELEPHAIS_SCHEDULER_READY=0")
    message(FATAL_ERROR "submission generator must create guarded templates")
endif()
if(NOT SUBMISSION_GENERATOR_TEXT MATCHES "celephais_sbatch_entry\\.sh")
    message(FATAL_ERROR "submission generator must delegate to celephais_sbatch_entry.sh")
endif()

execute_process(
    COMMAND bash "${SOURCE_DIR}/apps/submission_scripts/celephais_submission_usage.sh" macbook
    RESULT_VARIABLE SUBMISSION_USAGE_RESULT
    OUTPUT_VARIABLE SUBMISSION_USAGE_TEXT
)
set(EXPECTED_SUBMISSION_USAGE
    "  ╔══════════════════════════════════════╗\n  ║          C E L E P H A I S           ║\n  ║  Spectral Solver for Compact Binary  ║\n  ╚══════════════════════════════════════╝\n\nRun from the working directory.\n./sub_Celephais.sh parameter.toml\n\nOne arg: bin/celephais with the top-level run specification.\nTwo+ args: private-worker name or path=$1, INPUT_FILE=$2, rest are worker args.\n")
if(NOT SUBMISSION_USAGE_RESULT EQUAL 0 OR
   NOT SUBMISSION_USAGE_TEXT STREQUAL EXPECTED_SUBMISSION_USAGE)
    message(FATAL_ERROR "celephais_submission_usage.sh emitted an unexpected build handoff")
endif()

file(READ "${SOURCE_DIR}/soliv_agent.md" SOLIV_AGENT_TEXT)
foreach(token
        "tools/celephais_preflight.sh"
        "tools/generate_submission_template.sh"
        "celephais_submission_usage.sh"
        "CelephaisTargets.cmake"
        "CELEPHAIS_INSTALL_PREFIX"
        "fresh login shell"
        "detect_target")
    if(NOT SOLIV_AGENT_TEXT MATCHES "${token}")
        message(FATAL_ERROR "soliv_agent.md must point agents at ${token}")
    endif()
endforeach()

file(READ "${SOURCE_DIR}/compile.sh" COMPILE_TEXT)
foreach(token
        "cmake --install"
        "CELEPHAIS_INSTALL_PREFIX"
        "CelephaisTargets.cmake")
    if(NOT COMPILE_TEXT MATCHES "${token}")
        message(FATAL_ERROR "compile.sh must install the package via ${token}")
    endif()
endforeach()

foreach(path
        "CMakeLists.txt"
        "tools/celephais_preflight.sh"
        "soliv_agent.md")
    file(READ "${SOURCE_DIR}/${path}" PROJECT_NAMING_TEXT)
    string(TOLOWER "${PROJECT_NAMING_TEXT}" PROJECT_NAMING_TEXT)
    if(PROJECT_NAMING_TEXT MATCHES "kadath")
        message(FATAL_ERROR "${path} still mentions the retired project name")
    endif()
endforeach()
