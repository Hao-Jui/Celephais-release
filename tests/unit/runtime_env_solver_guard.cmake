set(RUNTIME_ENV_SH "${SOURCE_DIR}/apps/submission_scripts/celephais_runtime_env.sh")
file(READ "${RUNTIME_ENV_SH}" RUNTIME_ENV_TEXT)

if(NOT RUNTIME_ENV_TEXT MATCHES "dense[ \t\r\n|]+mumps[ \t\r\n|]+jfnk-mumps")
    message(FATAL_ERROR "celephais_runtime_env.sh must accept dense|mumps|jfnk-mumps")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "CELEPHAIS_SOLVER_DEFAULT:-jfnk-mumps")
    message(FATAL_ERROR "celephais_runtime_env.sh must default to CELEPHAIS_SOLVER=jfnk-mumps")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "JFNK_MAX_ITERS:-48")
    message(FATAL_ERROR "celephais_runtime_env.sh must default to JFNK_MAX_ITERS=48")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "JFNK_MUMPS_PC_REFRESH:-10")
    message(FATAL_ERROR "celephais_runtime_env.sh must default to JFNK_MUMPS_PC_REFRESH=10")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "MUMPS_TREE_CACHE:-0")
    message(FATAL_ERROR "celephais_runtime_env.sh must default the MUMPS tree cache off")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "JFNK_MUMPS_PC_ADAPTIVE:-0")
    message(FATAL_ERROR "celephais_runtime_env.sh must default adaptive PC refresh off")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS:-20")
    message(FATAL_ERROR "celephais_runtime_env.sh must default adaptive PC max steps to 20")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "MUMPS_OOC:-auto")
    message(FATAL_ERROR "celephais_runtime_env.sh must default MUMPS OOC policy to auto")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "MUMPS_OOC_TOUCH:-1.3")
    message(FATAL_ERROR "celephais_runtime_env.sh must default MUMPS OOC touch to 1.3")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "MUMPS_OOC_SAFETY:-0.7")
    message(FATAL_ERROR "celephais_runtime_env.sh must default MUMPS OOC safety to 0.7")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "MUMPS_OOC_BUDGET_MB:-}")
    message(FATAL_ERROR "celephais_runtime_env.sh must leave the test-only OOC budget unset")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "JACOBIAN_STRUCTURAL_PLAN_CACHE:-1")
    message(FATAL_ERROR "celephais_runtime_env.sh must default the Jacobian structural-plan cache on")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "SPARSE_MUMPS_ANALYZE_REUSE:-0")
    message(FATAL_ERROR "celephais_runtime_env.sh must default sparse-direct analyze reuse off")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "SPARSE_PARITY_MASK:-1")
    message(FATAL_ERROR "celephais_runtime_env.sh must default sparse parity masking on")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "JACOBIAN_FUSED_PARITY_MASK:-1")
    message(FATAL_ERROR "celephais_runtime_env.sh must default fused parity masking on")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "SPARSE_SECTOR_REDUCE:-1")
    message(FATAL_ERROR "celephais_runtime_env.sh must default sparse sector reduction on")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "SPARSE_PARITY_SPLIT_SOLVE:-1")
    message(FATAL_ERROR "celephais_runtime_env.sh must default the parity split solve on")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "SPARSE_CHORD_REUSE:-1")
    message(FATAL_ERROR "celephais_runtime_env.sh must default sparse-direct chord reuse on")
endif()

if(NOT RUNTIME_ENV_TEXT MATCHES "SPARSE_MUMPS_PATTERN_DROP_TOL:--1")
    message(FATAL_ERROR "celephais_runtime_env.sh must track the fixed numerical threshold by default")
endif()

set(RETIRED_DEVELOPMENT_FLAG_PATTERN
    "matrixfree|matrix_free|mfree|bordered_mf|JFNK_(PRECOND|PRECONDITIONER|DIAGNOSTIC|DUMP|SHIFT|SPECTRAL|GCRO|SCHWARZ|BLOCK|ORACLE)|PHYSICS|PRECOND_ORACLE|BMF|WPC|JV_|COLORING_|MEM_TRACE_|MUMPS_MATCHING|MUMPS_SCALING|MUMPS_ERROR_ANALYSIS|MUMPS_ROOT_SCALAPACK|MUMPS_NULL_PIVOT"
)

if(RUNTIME_ENV_TEXT MATCHES "${RETIRED_DEVELOPMENT_FLAG_PATTERN}")
    message(FATAL_ERROR "celephais_runtime_env.sh still exports retired matrix-free flags")
endif()

set(RUN_SOLVE_SH "${SOURCE_DIR}/apps/submission_scripts/celephais_run_solve.sh")
file(READ "${RUN_SOLVE_SH}" RUN_SOLVE_TEXT)

foreach(VAR
        CELEPHAIS_SOLVER
        CELEPHAIS_TIMING
        DEF_FILTER
        JACOBIAN_ASSEMBLER_PROFILE
        JACOBIAN_STRUCTURAL_PLAN_CACHE
        SPARSE_PARITY_MASK
        JACOBIAN_FUSED_PARITY_MASK
        SPARSE_SECTOR_REDUCE
        SPARSE_PARITY_SPLIT_SOLVE
        DROP_TOL
        MUMPS_BLR
        MUMPS_BLR_DROP_TOL
        MUMPS_OOC
        MUMPS_OOC_TOUCH
        MUMPS_OOC_SAFETY
        MUMPS_OOC_BUDGET_MB
        MUMPS_ORDERING
        MUMPS_TREE_CACHE
        MUMPS_RANKS_PER_NODE
        SPARSE_MUMPS_ANALYZE_REUSE
        SPARSE_CHORD_REUSE
        SPARSE_MUMPS_PATTERN_DROP_TOL
        SPARSE_MUMPS_SUPERSET_MAX_NNZ_RATIO
        DIRECT_REPLAY_CAPTURE
        DIRECT_REPLAY_CAPTURE_ORDINAL
        MUMPS_INFOG_TRACE
        JFNK_MAX_ITERS
        JFNK_MUMPS_PC_REFRESH
        JFNK_MUMPS_PC_ADAPTIVE
        JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS
        JFNK_RTOL
        MEMORY_MAPPER_PHASE_PROFILE)
    if(NOT RUN_SOLVE_TEXT MATCHES "[ \t\r\n]${VAR}[ \t\r\n]")
        message(FATAL_ERROR "celephais_run_solve.sh must forward ${VAR} through mpirun")
    endif()
endforeach()

if(RUN_SOLVE_TEXT MATCHES "${RETIRED_DEVELOPMENT_FLAG_PATTERN}")
    message(FATAL_ERROR "celephais_run_solve.sh still forwards retired development flags")
endif()

if(NOT RUN_SOLVE_TEXT MATCHES "MPI_ENV_ARGS\\+=\\(-x[ \t\r\n]+\"\\$env_var\"\\)")
    message(FATAL_ERROR "celephais_run_solve.sh must convert MPI_ENV_VARS entries into mpirun -x arguments")
endif()

foreach(PLACEMENT_CONTROL
        MPI_BIND_TO
        MPI_MAP_BY
        SLURM_CPU_BIND)
    if(NOT RUN_SOLVE_TEXT MATCHES "${PLACEMENT_CONTROL}")
        message(FATAL_ERROR
            "celephais_run_solve.sh must expose optional ${PLACEMENT_CONTROL} placement control")
    endif()
endforeach()

if(NOT RUN_SOLVE_TEXT MATCHES "Darwin")
    message(FATAL_ERROR
        "celephais_run_solve.sh must refuse unsupported Open MPI binding on macOS")
endif()

set(SHARED_SUB_CELEPHAIS "${SOURCE_DIR}/apps/submission_scripts/sub_Celephais.sh")
file(READ "${SHARED_SUB_CELEPHAIS}" SHARED_SUB_CELEPHAIS_TEXT)

if(NOT SHARED_SUB_CELEPHAIS_TEXT MATCHES "celephais_runtime_env\\.sh")
    message(FATAL_ERROR "Shared sub_Celephais.sh must source celephais_runtime_env.sh")
endif()
if(NOT SHARED_SUB_CELEPHAIS_TEXT MATCHES "celephais_run_solve\\.sh")
    message(FATAL_ERROR "Shared sub_Celephais.sh must exec celephais_run_solve.sh")
endif()

file(GLOB APP_LAUNCHERS
    "${SOURCE_DIR}/apps/*/sbatch_outputs/sub_Celephais.sh"
    "${SOURCE_DIR}/apps/*/sbatch_outputs/binac.sh"
)
foreach(LAUNCHER IN LISTS APP_LAUNCHERS)
    file(READ "${LAUNCHER}" LAUNCHER_TEXT)
    if(NOT LAUNCHER_TEXT MATCHES "submission_scripts/sub_Celephais\\.sh|celephais_runtime_env\\.sh")
        message(FATAL_ERROR "App launcher must delegate to shared sub_Celephais.sh or source celephais_runtime_env.sh: ${LAUNCHER}")
    endif()
endforeach()
