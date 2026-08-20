#!/bin/bash
# Universal solve launcher — auto-detects platform and solve binary.
# Symlinked from apps/<APP>/sbatch_outputs/sub_Celephais.sh.
#
# Usage:  ./sub_Celephais.sh                                       (writes the app example input)
#         ./sub_Celephais.sh <input.toml>                          (runs ../bin/Release/solve)
#         ./sub_Celephais.sh 2sacra <input.toml> [extra args...]   (runs ../bin/Release/2sacra)
#         ./sub_Celephais.sh /path/to/bin <input.toml> [extra...]  (explicit binary)
#
# On macbook:           runs locally via mpirun
# On workstation:       runs locally via mpirun
# On HPC + allocation:  runs via srun (SLURM detected)
# On HPC + no alloc:    submits batch job via sbatch

# CALLER_DIR = where the symlink lives (the app's sbatch_outputs/)
CALLER_DIR="$(cd "$(dirname "$0")" && pwd)"
# CELEPHAIS_SUBMISSION_DIR = where this script actually lives (resolve symlink chain)
_source="${BASH_SOURCE[0]}"
while [[ -L "$_source" ]]; do
    _dir="$(cd "$(dirname "$_source")" && pwd)"
    _source="$(readlink "$_source")"
    [[ "$_source" != /* ]] && _source="$_dir/$_source"
done
CELEPHAIS_SUBMISSION_DIR="$(cd "$(dirname "$_source")" && pwd)"
CELEPHAIS_REPO_ROOT="$(cd "$CELEPHAIS_SUBMISSION_DIR/../.." && pwd)"

detect_target() {
    if [[ "$(uname -s)" == "Darwin" ]]; then echo "macbook"; return; fi
    local host="${HOSTNAME:-$(hostname -s 2>/dev/null || echo "")}"
    case "$host" in
        viper*)  echo "viper"  ;;
        sakura*) echo "sakura" ;;
        raven*)  echo "raven"  ;;
        dt-login*|dt-*.delta.ncsa.illinois.edu) echo "delta" ;;
        cc-login*|*.campuscluster.illinois.edu) echo "ncsa" ;;
        snowman*) echo "snowman" ;;
        login*.cluster) echo "${CELEPHAIS_SITE_NAME:-env}" ;;
        *)
            echo "${CELEPHAIS_SITE_NAME:-env}"
            ;;
    esac
}

CELEPHAIS_TARGET="${CELEPHAIS_TARGET:-$(detect_target)}"

resolve_path() {
    local path="$1"
    if [[ "$path" == /* ]]; then
        printf '%s\n' "$path"
        return 0
    fi
    local dir base
    dir="$(dirname "$path")"
    base="$(basename "$path")"
    dir="$(cd "$dir" 2>/dev/null && pwd)" || return 1
    printf '%s/%s\n' "$dir" "$base"
}

target_class() {
    case "$1" in
        macbook|snowman) echo "direct"; return ;;
    esac
    if [[ "${CELEPHAIS_TARGET_CLASS:-}" == "direct" || "${CELEPHAIS_TARGET_CLASS:-}" == "slurm" ]]; then
        echo "$CELEPHAIS_TARGET_CLASS"
    elif command -v sbatch >/dev/null 2>&1; then
        echo "slurm"
    else
        echo "direct"
    fi
}

if [[ $# -eq 0 ]]; then
    if ! SOLVE_BIN="$(resolve_path "$CALLER_DIR/../bin/Release/solve")"; then
        echo "Could not resolve app solve binary from: $CALLER_DIR" >&2
        exit 1
    fi
    exec "$SOLVE_BIN"
fi

# Resolve solve binary and input file
BIN_DIR="$CALLER_DIR/../bin/Release"
if [[ $# -ge 2 ]]; then
    if [[ "$1" == */* ]]; then
        SOLVE_BIN="$1"
    else
        SOLVE_BIN="$BIN_DIR/$1"
    fi
    INPUT_FILE="$2"
    shift 2
elif [[ $# -eq 1 ]]; then
    SOLVE_BIN="$BIN_DIR/solve"
    INPUT_FILE="$1"
    shift 1
fi
EXTRA_ARGS=("$@")

if ! SOLVE_BIN="$(resolve_path "$SOLVE_BIN")"; then
    echo "Could not resolve solve binary path: $SOLVE_BIN" >&2
    exit 1
fi
if ! INPUT_FILE="$(resolve_path "$INPUT_FILE")"; then
    echo "Could not resolve input file path: $INPUT_FILE" >&2
    exit 1
fi

# Scheduler target without active allocation -> submit batch job
RESOLVED_TARGET_CLASS="$(target_class "$CELEPHAIS_TARGET")"
if [[ "$RESOLVED_TARGET_CLASS" == "slurm" && -z "${SLURM_JOB_ID:-}" ]]; then
    PLATFORM_SCRIPT="$CELEPHAIS_SUBMISSION_DIR/${CELEPHAIS_TARGET}.sh"
    if [[ ! -f "$PLATFORM_SCRIPT" ]]; then
        echo "No sbatch script/template for target '$CELEPHAIS_TARGET': $PLATFORM_SCRIPT" >&2
        echo "Run: tools/generate_submission_template.sh ${CELEPHAIS_TARGET}" >&2
        echo "Then ask the human to fill scheduler resources and set CELEPHAIS_SCHEDULER_READY=1." >&2
        exit 1
    fi
    if grep -Eq '^[[:space:]]*CELEPHAIS_SCHEDULER_READY=0([[:space:]]|$)' "$PLATFORM_SCRIPT"; then
        echo "Sbatch template is not ready: $PLATFORM_SCRIPT" >&2
        echo "Ask the human to edit that file: fill SBATCH partition/account/qos/resources, then change CELEPHAIS_SCHEDULER_READY=0 to CELEPHAIS_SCHEDULER_READY=1." >&2
        exit 1
    fi
    if grep -Eq '^#SBATCH[[:space:]].*(<[^>]+>|FILL_ME|TODO)' "$PLATFORM_SCRIPT"; then
        echo "Sbatch template still contains an active placeholder: $PLATFORM_SCRIPT" >&2
        echo "Ask the human to replace placeholder SBATCH values before submission." >&2
        exit 1
    fi
    echo "Submitting batch job via generated site script: sbatch ${CELEPHAIS_TARGET}.sh"
    exec sbatch \
        --export=ALL,CELEPHAIS_TARGET="$CELEPHAIS_TARGET",CELEPHAIS_REPO_ROOT="$CELEPHAIS_REPO_ROOT",CELEPHAIS_SUBMISSION_DIR="$CELEPHAIS_SUBMISSION_DIR" \
        "$PLATFORM_SCRIPT" "$SOLVE_BIN" "$INPUT_FILE" "${EXTRA_ARGS[@]}"
fi

# Local or inside allocation — run directly
ENV_FILE="$CELEPHAIS_REPO_ROOT/cmake/env/${CELEPHAIS_TARGET}.sh"
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
elif [[ -f "$CELEPHAIS_REPO_ROOT/cmake/env/env.sh" ]]; then
    source "$CELEPHAIS_REPO_ROOT/cmake/env/env.sh"
fi

source "$CELEPHAIS_SUBMISSION_DIR/celephais_runtime_env.sh"

exec "$CELEPHAIS_SUBMISSION_DIR/celephais_run_solve.sh" "$SOLVE_BIN" "$INPUT_FILE" "${EXTRA_ARGS[@]}"
