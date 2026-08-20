set(ALLOWED_CONFIG_CPP "${SOURCE_DIR}/src/System_of_eqs/solver_runtime_config.cpp")
set(NEWTON_LOOP_RUNNER_HPP "${SOURCE_DIR}/include/Apps/Helper/newton_loop_runner.hpp")

file(READ "${NEWTON_LOOP_RUNNER_HPP}" NEWTON_LOOP_RUNNER_TEXT)
if(NEWTON_LOOP_RUNNER_TEXT MATCHES "System_of_eqs::SOLVER[ \t\r\n]+solver")
    message(FATAL_ERROR "legacy newton_step_with_consensus overload must stay removed")
endif()
if(NEWTON_LOOP_RUNNER_TEXT MATCHES "syst\\.do_newton\\(")
    message(FATAL_ERROR "newton_step_with_consensus must dispatch through do_newton_configured")
endif()
if(NOT NEWTON_LOOP_RUNNER_TEXT MATCHES "syst\\.do_newton_configured\\(precision, error, config\\)")
    message(FATAL_ERROR "newton_step_with_consensus must honor CELEPHAIS_SOLVER dispatch")
endif()

file(GLOB_RECURSE CANDIDATE_FILES
    "${SOURCE_DIR}/src/*.cpp"
    "${SOURCE_DIR}/src/Newton/*.cpp"
    "${SOURCE_DIR}/include/Linear_algebra/*.cpp"
    "${SOURCE_DIR}/include/Linear_algebra/*.hpp"
    "${SOURCE_DIR}/include/Linear_algebra/*.ipp"
    "${SOURCE_DIR}/include/Apps/Helper/*.cpp"
    "${SOURCE_DIR}/include/Apps/Helper/*.hpp"
    "${SOURCE_DIR}/include/Apps/Helper/*.ipp"
    "${SOURCE_DIR}/include/Apps/Formalism/*.cpp"
    "${SOURCE_DIR}/include/Apps/Formalism/*.hpp"
    "${SOURCE_DIR}/include/Apps/Formalism/*.ipp"
    "${SOURCE_DIR}/include/Apps/Seed/*.cpp"
    "${SOURCE_DIR}/include/Apps/Seed/*.hpp"
    "${SOURCE_DIR}/include/Apps/Seed/*.ipp"
    "${SOURCE_DIR}/include/Apps/TwoD/*.cpp"
    "${SOURCE_DIR}/include/Apps/TwoD/*.hpp"
    "${SOURCE_DIR}/include/Apps/TwoD/*.ipp"
)

set(SOLVER_RUNTIME_ENV_PATTERN
    "std::getenv\\(\"(CELEPHAIS_SOLVER|CELEPHAIS_TIMING|DEF_FILTER|DROP_TOL|MUMPS_ORDERING|MUMPS_OOC|MUMPS_BLR|MUMPS_TREE_CACHE|JFNK_MAX_ITERS|JFNK_MUMPS_PC_REFRESH|JFNK_MUMPS_PC_ADAPTIVE|JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS|JFNK_RTOL)\"\\)"
    "std::getenv\\(\"(CELEPHAIS_SOLVER|CELEPHAIS_TIMING|DEF_FILTER|DROP_TOL|MUMPS_ORDERING|MUMPS_OOC|MUMPS_OOC_TOUCH|MUMPS_OOC_SAFETY|MUMPS_OOC_BUDGET_MB|MUMPS_RANKS_PER_NODE|MUMPS_BLR|JFNK_MAX_ITERS|JFNK_MUMPS_PC_REFRESH|JFNK_MUMPS_PC_ADAPTIVE|JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS|JFNK_RTOL)\"\\)"
)

set(RETIRED_DEVELOPMENT_ENV_PATTERN
    "(getenv|env_[A-Za-z0-9_]+)\\([ \t\r\n]*\"(COLORING_|MEM_TRACE_|MUMPS_MATCHING|MUMPS_SCALING|MUMPS_ERROR_ANALYSIS|MUMPS_ROOT_SCALAPACK|MUMPS_NULL_PIVOT|JFNK_(PRECOND|PRECONDITIONER|DIAGNOSTIC|DUMP|SHIFT|SPECTRAL|GCRO|SCHWARZ|BLOCK|ORACLE)|PHYSICS|PRECOND_ORACLE|BMF|WPC|JV_)"
)

foreach(FILE IN LISTS CANDIDATE_FILES)
    if(FILE STREQUAL ALLOWED_CONFIG_CPP)
        continue()
    endif()
    file(READ "${FILE}" TEXT)
    if(TEXT MATCHES "${SOLVER_RUNTIME_ENV_PATTERN}")
        message(FATAL_ERROR
            "Solver runtime env access must go through SolverRuntimeConfig::from_environment(): ${FILE}")
    endif()
    if(TEXT MATCHES "${RETIRED_DEVELOPMENT_ENV_PATTERN}")
        message(FATAL_ERROR "Retired development runtime flag remains in production source: ${FILE}")
    endif()
endforeach()

file(GLOB_RECURSE SOLVER_STAGE_FILES
    "${SOURCE_DIR}/include/Apps/Formalism/*.cpp"
    "${SOURCE_DIR}/include/Apps/Formalism/*.hpp"
    "${SOURCE_DIR}/include/Apps/Formalism/*.ipp"
    "${SOURCE_DIR}/include/Apps/TwoD/*.cpp"
    "${SOURCE_DIR}/include/Apps/TwoD/*.hpp"
    "${SOURCE_DIR}/include/Apps/TwoD/*.ipp"
)
foreach(FILE IN LISTS SOLVER_STAGE_FILES)
    if(FILE STREQUAL NEWTON_LOOP_RUNNER_HPP)
        continue()
    endif()
    file(READ "${FILE}" TEXT)
    if(TEXT MATCHES "(\\.|->)do_newton\\(")
        message(FATAL_ERROR
            "App solver loops must use newton_step_with_consensus with SolverRuntimeConfig: ${FILE}")
    endif()
endforeach()
