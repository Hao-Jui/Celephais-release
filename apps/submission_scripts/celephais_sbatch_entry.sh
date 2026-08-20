#!/bin/bash
# Shared body for human-owned SLURM scripts.
#
# apps/submission_scripts/{cluster}.sh should contain only site SBATCH lines and:
#   exec "${CELEPHAIS_SUBMISSION_DIR:?submit via sub_Celephais.sh}/celephais_sbatch_entry.sh" "$@"

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: celephais_sbatch_entry.sh <solve_bin> <input.toml> [solver args...]" >&2
    exit 1
fi

CELEPHAIS_TARGET="${CELEPHAIS_TARGET:-}"
if [ -z "$CELEPHAIS_TARGET" ]; then
    echo "CELEPHAIS_TARGET is unset. Submit through sub_Celephais.sh or export CELEPHAIS_TARGET=<cluster>." >&2
    exit 1
fi

CELEPHAIS_SUBMISSION_DIR="${CELEPHAIS_SUBMISSION_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CELEPHAIS_REPO_ROOT="${CELEPHAIS_REPO_ROOT:-$(cd "$CELEPHAIS_SUBMISSION_DIR/../.." && pwd)}"
export HOME_CELEPHAIS="${HOME_CELEPHAIS:-$CELEPHAIS_REPO_ROOT}"

SOLVE_BIN="$1"
INPUT_FILE="$2"
shift 2

ENV_FILE="$CELEPHAIS_REPO_ROOT/cmake/env/${CELEPHAIS_TARGET}.sh"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
elif [ -f "$CELEPHAIS_REPO_ROOT/cmake/env/env.sh" ]; then
    source "$CELEPHAIS_REPO_ROOT/cmake/env/env.sh"
else
    echo "Missing environment file: $ENV_FILE or $CELEPHAIS_REPO_ROOT/cmake/env/env.sh" >&2
    exit 1
fi

source "$CELEPHAIS_SUBMISSION_DIR/celephais_runtime_env.sh"

exec "$CELEPHAIS_SUBMISSION_DIR/celephais_run_solve.sh" "$SOLVE_BIN" "$INPUT_FILE" "$@"
