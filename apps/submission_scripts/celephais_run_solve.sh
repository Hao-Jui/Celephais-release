#!/bin/bash
# Shared solve-launcher used by every apps/<APP>/sbatch_outputs/sub_Celephais.sh.
#
# Usage:
#   celephais_run_solve.sh <solve_bin> <input.toml> [solver args...]
#
# Assumes celephais_runtime_env.sh has already been sourced in the parent shell
# so explicit Celephais runtime flags plus OMP_* / VECLIB_* controls are exported. The list below
# forwards the production SolverRuntimeConfig knobs plus narrow diagnostic
# knobs consumed directly by MUMPS/Jacobian internals.
#
# Optional env vars:
#   MPI_NP        — mpirun -np value (default 2)
#   MPI_BIND_TO — Open MPI --bind-to policy (unset by default)
#   MPI_MAP_BY  — Open MPI --map-by policy (unset by default)
#   SLURM_CPU_BIND — srun --cpu-bind policy (unset by default)
#   PROFILE_MODE  — none|time|sample (default none); sample writes
#                   sample_solve_rank0.txt via macOS `sample(1)`.

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <solve_bin> <input.toml> [solver args...]" >&2
    exit 1
fi

SOLVE_BIN="$1"
CONFIG_FILE="$2"
shift 2

if [ ! -x "$SOLVE_BIN" ]; then
    echo "Solve binary not found or not executable: $SOLVE_BIN" >&2
    exit 1
fi

PROFILE_MODE="${PROFILE_MODE:-none}"
MPI_NP="${MPI_NP:-2}"
TIME_CMD=()
if [ -x /usr/bin/time ]; then
    TIME_CMD=(/usr/bin/time)
    if /usr/bin/time -l true >/dev/null 2>&1; then
        TIME_CMD=(/usr/bin/time -l)
    elif /usr/bin/time -v true >/dev/null 2>&1; then
        TIME_CMD=(/usr/bin/time -v)
    fi
fi

MPI_ENV_VARS=(
    LD_LIBRARY_PATH
    HOME_CELEPHAIS
    CELEPHAIS_SOLVER
    CELEPHAIS_TIMING
    DEF_FILTER
    DEF_SCALAR_BRANCH_TARGET
    DEF_FREEZE_SCALAR
    JACOBIAN_ASSEMBLER_PROFILE
    JACOBIAN_STRUCTURAL_PLAN_CACHE
    SPARSE_PARITY_MASK
    JACOBIAN_FUSED_PARITY_MASK
    SPARSE_SECTOR_REDUCE
    SPARSE_PARITY_SPLIT_SOLVE
    JACOBIAN_LOCAL_COO_BLOCKS
    KERNEL_PROFILE
    OPE_ACTION_PROFILE
    OPE_MULT_PROFILE
    DEF_COMPUTE_PER_DEF
    PARTNER_STAGING_PROFILE
    JBUILD_PROFILE_AND_EXIT
    EQBC_DIRECT_EMIT_CENSUS
    MEMORY_MAPPER_PROFILE
    MEMORY_MAPPER_PHASE_PROFILE
    ARRAY_COPY_PROFILE
    JACOBIAN_WLANE32
    JACOBIAN_WLANE16
    JACOBIAN_WLANE8
    JACOBIAN_WLANE4
    JACOBIAN_WLANE2
    JACOBIAN_GLOBAL_GROUP_PLAN
    JACOBIAN_VARDOM_WLANE2
    JACOBIAN_VARDOM_MAX_WIDTH
    MATCHING_LANE_EXPORT
    MATCHING_IMPORT_LANE_BATCH
    DIRECT_COLUMN_ORACLE
    DIRECT_COLUMN_ORACLE_MAX_FAILURES
    W4_BUCKET_CENSUS
    W8_BUCKET_CENSUS
    W16_BUCKET_CENSUS
    W32_BUCKET_CENSUS
    ROW_DISJOINT_BUCKET_CENSUS
    ROW_DISJOINT_BUCKET_CENSUS_EXIT_AFTER
    ROW_DISJOINT_BUCKET_CLASS
    ROW_DISJOINT_BUCKET_DOMAIN
    ROW_DISJOINT_BUCKET_VAR
    ROW_DISJOINT_BUCKET_MAX_COLUMNS
    ROW_DISJOINT_BUCKET_VERBOSE
    ROW_DISJOINT_BUCKET_ORACLE
    ROW_DISJOINT_BUCKET_ORACLE_THROW
    ROW_DISJOINT_BUCKET_ORACLE_MAX_GROUP_SIZE
    JACOBIAN_COO_BYTE_HASH
    JACOBIAN_CANONICAL_COO_BYTE_HASH
    JACOBIAN_COO_BYTE_HASH_EXIT_AFTER_FIRST
    VAL_DOMAIN_DER_ABS_CACHE
    DERIVATIVE_LANE_TILING
    OPE_DER_BATCH
    OPE_DER_AOSOA
    OPE1D_WORKSPACE
    OPE1D_WORKSPACE_MAX_BYTES
    OPE_ACTION_WRITE_INTO
    PER_BUCKET_DER_ABS_FLUSH
    RELEASE_ALLOCATOR_PAGES
    DO_JX_DER_ABS_CACHE
    DO_JX_MPI
    DO_JX_MPI_SELFTEST
    DO_JX_TERM_CLOSURE
    SEC_MEMBER_MPI
    RESIDUAL_FORWARD
    RESIDUAL_FORWARD_SELFTEST
    DEF_CLOSURE_CROSSCHECK
    DO_JX_PROFILE
    DO_JX_PROFILE_EVERY
    DROP_TOL
    MUMPS_BLR
    MUMPS_BLR_DROP_TOL
    MUMPS_BLR_MIXED
    MUMPS_SINGLE_FACTOR
    MUMPS_NATIVE_VERBOSE
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
    RSS_PHASE_PROFILE
    JFNK_MAX_ITERS
    JFNK_MUMPS_PC_REFRESH
    JFNK_MUMPS_PC_ADAPTIVE
    JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS
    JFNK_RTOL
    JFNK_EW
    JFNK_EW_GAMMA
    JFNK_EW_ALPHA
    JFNK_EW_MAX
    JFNK_EW_MIN
    JFNK_LINESEARCH
    JFNK_LS_SELFTEST
    JFNK_LS_ARMIJO
    JFNK_LS_ALPHA
    JFNK_LS_MAX_BACKTRACK
    JFNK_LS_MIN_LAMBDA
    GMRES_PROFILE
    PRIMAL_TANGENT_PROBE
    PRIMAL_TANGENT_PROBE_REPS
    VAL_CACHE_RSS_PROBE
    SCHUR_AGGREGATES
    SCHUR_SPLIT_EXTERIOR_TAIL
    SCHUR_PROBE_CONTRACT_ONLY
    SCHUR_PROBE_RCOND
    SCHUR_PROBE_STRICT
    SCHUR_PROBE_NULLITY
    SCHUR_PROBE_NULLITY_AGGREGATES
    SCHUR_PROBE_NULLITY_DIM_MAX
    SCHUR_PROBE_NULLVEC
    SCHUR_PROBE_NULLVEC_MAX
    SCHUR_PROBE_NULLVEC_DIM_MAX
    SCHUR_PROBE_PINNING
    SCHUR_PC_GMRES
    SCHUR_PC_GMRES_MAXIT
    SCHUR_PC_COARSE
    SCHUR_PC_COARSE_GAUGE
    SCHUR_PC_COARSE_REG
    SCHUR_PC_COARSE_ORACLE
    SCHUR_PC_ORACLE_K
    SCHUR_PC_ORACLE_ITERS
    OMP_NUM_THREADS
    MKL_NUM_THREADS
    OPENBLAS_NUM_THREADS
    VECLIB_MAXIMUM_THREADS
)

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Inside a SLURM allocation: use srun. Rank count and env are inherited
    # from the allocation, so no -np or -x flags are needed.
    SLURM_PLACEMENT_ARGS=()
    if [ -n "${SLURM_CPU_BIND:-}" ]; then
        SLURM_PLACEMENT_ARGS+=("--cpu-bind=${SLURM_CPU_BIND}")
    fi
    RUN_CMD=(srun "${SLURM_PLACEMENT_ARGS[@]}" "$SOLVE_BIN" "$CONFIG_FILE" "$@")
else
    MPI_ENV_ARGS=()
    MPI_PLACEMENT_ARGS=()
    if mpirun --version 2>&1 | grep -qi 'open[[:space:]-]*mpi'; then
        if [ -n "${MPI_BIND_TO:-}" ]; then
            if [ "$(uname -s)" = "Darwin" ]; then
                echo "MPI_BIND_TO is unsupported by Open MPI on macOS; leave it unset." >&2
                exit 2
            fi
            MPI_PLACEMENT_ARGS+=(--bind-to "$MPI_BIND_TO")
        fi
        if [ -n "${MPI_MAP_BY:-}" ]; then
            MPI_PLACEMENT_ARGS+=(--map-by "$MPI_MAP_BY")
        fi
        for env_var in "${MPI_ENV_VARS[@]}"; do
            if [ "${!env_var+x}" = "x" ]; then
                MPI_ENV_ARGS+=(-x "$env_var")
            fi
        done
    elif [ -n "${MPI_BIND_TO:-}${MPI_MAP_BY:-}" ]; then
        echo "MPI_BIND_TO/MPI_MAP_BY require an Open MPI launcher." >&2
        exit 2
    fi

    RUN_CMD=(
        mpirun -np "$MPI_NP"
            "${MPI_PLACEMENT_ARGS[@]}"
            "${MPI_ENV_ARGS[@]}"
            "$SOLVE_BIN"
            "$CONFIG_FILE"
            "$@"
    )
fi

if [ "$PROFILE_MODE" = "sample" ]; then
    "${RUN_CMD[@]}" &
    job_pid=$!
    sleep 2
    solve_pid=$(pgrep -n -f '/bin/Release/solve')
    if [ -z "$solve_pid" ]; then
        echo "PROFILE_MODE=sample: could not find solve PID; skipping sample."
    else
        echo "PROFILE_MODE=sample: sampling solve pid $solve_pid -> sample_solve_rank0.txt"
        sample "$solve_pid" 10 -file sample_solve_rank0.txt
    fi
    wait "$job_pid"
else
    if [ "${#TIME_CMD[@]}" -gt 0 ]; then
        "${TIME_CMD[@]}" "${RUN_CMD[@]}"
    else
        "${RUN_CMD[@]}"
    fi
fi
