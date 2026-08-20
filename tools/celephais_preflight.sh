#!/bin/bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="env"
ENV_FILE="${CELEPHAIS_ENV_FILE:-}"
CHECK_SUBMIT=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --submit) CHECK_SUBMIT=1; shift ;;
        --target) TARGET="${2:?missing target}"; shift 2 ;;
        --env-file) ENV_FILE="${2:?missing env file}"; shift 2 ;;
        -h|--help)
            echo "Usage: tools/celephais_preflight.sh [--submit] [--target name] [--env-file path] [target]"
            exit 0
            ;;
        *) TARGET="$1"; shift ;;
    esac
done

emit() {
    local ready="$1"
    local target_class="$2"
    local blocker="$3"
    local next="$4"
    printf 'READY=%s\n' "$ready"
    printf 'TARGET_CLASS=%s\n' "$target_class"
    printf 'BLOCKER=%s\n' "$blocker"
    printf 'NEXT=%s\n' "$next"
}

source_env() {
    local candidate="$1"
    [ -n "$candidate" ] || return 0
    [ -f "$candidate" ] || return 0
    # shellcheck disable=SC1090
    source "$candidate"
}

if [ -z "$ENV_FILE" ] && [ -f "$ROOT_DIR/cmake/env/${TARGET}.sh" ]; then
    ENV_FILE="$ROOT_DIR/cmake/env/${TARGET}.sh"
fi
source_env "$ENV_FILE"
if [ -f "$ROOT_DIR/cmake/env/celephais_env_common.sh" ]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/cmake/env/celephais_env_common.sh"
    celephais_add_default_runtime_paths
fi

detect_target_class() {
    if [ "${CELEPHAIS_TARGET_CLASS:-}" = "direct" ] || [ "${CELEPHAIS_TARGET_CLASS:-}" = "slurm" ]; then
        printf '%s\n' "$CELEPHAIS_TARGET_CLASS"
    elif [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
        printf 'direct\n'
    elif command -v sbatch >/dev/null 2>&1; then
        printf 'slurm\n'
    else
        printf 'direct\n'
    fi
}

TARGET_CLASS="$(detect_target_class)"

cmake_version_ok() {
    command -v cmake >/dev/null 2>&1 || return 1
    local version major minor
    version="$(LC_ALL=C cmake --version 2>/dev/null | awk 'NR==1 {print $3}')"
    [[ "$version" =~ ^([0-9]+)\.([0-9]+) ]] || return 1
    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    (( major > 3 || (major == 3 && minor >= 25) ))
}

pick_command() {
    local explicit="$1"
    shift
    if [ -n "$explicit" ]; then
        command -v "$explicit" 2>/dev/null || printf '%s\n' "$explicit"
        return 0
    fi
    local candidate
    for candidate in "$@"; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}


runtime_path_missing() {
    [ "$(uname -s 2>/dev/null)" != "Darwin" ] || return 1
    local prefix subdir dir
    for prefix in \
        "${FFTW_HOME:-}" "${FFTW_MPI_HOME:-}" "${BOOST_ROOT:-}" "${BOOST_HOME:-}" \
        "${SCALAPACK_DIR:-}" "${SCALAPACK_ROOT:-}" "${MKLROOT:-}" "${I_MPI_ROOT:-}"; do
        [ -n "$prefix" ] || continue
        for subdir in lib lib64 lib/intel64 lib/release lib/x86_64-linux-gnu; do
            dir="$prefix/$subdir"
            [ -d "$dir" ] || continue
            case ":${LD_LIBRARY_PATH:-}:" in
                *":$dir:"*) ;;
                *) printf '%s\n' "$dir"; return 0 ;;
            esac
        done
    done
    return 1
}

if ! cmake_version_ok; then
    emit 0 "$TARGET_CLASS" "CMAKE_VERSION" "load or install CMake >= 3.25"
    exit 2
fi

if ! pick_command "${CXX:-}" mpicxx mpig++ mpigxx mpiicpx CC c++ >/dev/null; then
    emit 0 "$TARGET_CLASS" "CXX_MPI_WRAPPER" "load MPI compiler wrappers or export CXX"
    exit 2
fi
if ! pick_command "${FC:-}" mpifort mpiifort ifort ifx ftn gfortran >/dev/null; then
    emit 0 "$TARGET_CLASS" "FORTRAN_WRAPPER" "load a Fortran/MPI wrapper or export FC"
    exit 2
fi

# MUMPS ships in-tree (third_party/mumps) and is built from source as part of the
# Celephais build, so the source itself supplies MUMPS — no system/PETSc provider is
# required or discovered. The only requirement is that the vendored source exists.
if [ ! -f "$ROOT_DIR/third_party/mumps/src/dmumps_driver.F" ]; then
    emit 0 "$TARGET_CLASS" "MUMPS_SOURCE" "in-tree MUMPS missing; restore third_party/mumps"
    exit 2
fi

missing_runtime="$(runtime_path_missing || true)"
if [ -n "$missing_runtime" ]; then
    emit 0 "$TARGET_CLASS" "RUNTIME_PATH" "source cmake/env/env.sh or add ${missing_runtime} to LD_LIBRARY_PATH"
    exit 2
fi

if [ "$CHECK_SUBMIT" = 1 ] && [ "$TARGET_CLASS" = "slurm" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
    site_script="$ROOT_DIR/apps/submission_scripts/${TARGET}.sh"
    if [ ! -f "$site_script" ]; then
        emit 0 "$TARGET_CLASS" "SBATCH_TEMPLATE" "tools/generate_submission_template.sh ${TARGET}"
        exit 2
    fi
    if grep -Eq '^[[:space:]]*CELEPHAIS_SCHEDULER_READY=0([[:space:]]|$)' "$site_script"; then
        emit 0 "$TARGET_CLASS" "SBATCH_RESOURCES" "ask human to fill ${site_script} and set CELEPHAIS_SCHEDULER_READY=1"
        exit 2
    fi
fi

emit 1 "$TARGET_CLASS" "" "LC_ALL=C cmake --preset ${TARGET}"
