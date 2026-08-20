#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Detect platform ──────────────────────────────────────────────────────────
detect_target() {
    if [[ $# -ge 1 && -n "${1:-}" ]]; then
        echo "$1"
        return
    fi

    if [[ "$(uname -s)" == "Darwin" ]]; then
        echo "macbook"
        return
    fi

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

TARGET="$(detect_target "${1:-}")"

print_submission_handoff() {
    # Optional: absent in slimmed distributions that ship no submission scripts.
    [[ -x "${ROOT_DIR}/apps/submission_scripts/celephais_submission_usage.sh" ]] || return 0
    mkdir -p "${ROOT_DIR}/build-logs"
    local handoff_target="${CELEPHAIS_SITE_NAME:-$TARGET}"
    "${ROOT_DIR}/apps/submission_scripts/celephais_submission_usage.sh" \
        "${handoff_target}" "${TARGETS_FILE}" \
        | tee "${ROOT_DIR}/build-logs/${handoff_target}-usage.txt"
}

refresh_submission_links() {
    # Optional: absent in slimmed distributions that ship no submission scripts.
    [[ -e "${ROOT_DIR}/apps/submission_scripts/sub_Celephais.sh" ]] || return 0
    local app_dir
    for app_dir in "${ROOT_DIR}"/apps/*; do
        [[ -d "$app_dir/sbatch_outputs" ]] || continue
        ln -sf ../../submission_scripts/sub_Celephais.sh "${app_dir}/sbatch_outputs/sub_Celephais.sh"
    done
}

require_cmake_minimum() {
    local version major minor
    if ! command -v cmake >/dev/null 2>&1; then
        echo "CMake is required. Load a cmake module or install CMake >= 3.25 for target '${TARGET}'." >&2
        exit 1
    fi

    version="$(LC_ALL=C cmake --version 2>/dev/null | awk 'NR==1 {print $3}')"
    if [[ ! "$version" =~ ^([0-9]+)\.([0-9]+) ]]; then
        echo "Could not parse CMake version '${version}'. Need CMake >= 3.25 for target '${TARGET}'." >&2
        exit 1
    fi

    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    if (( major < 3 || (major == 3 && minor < 25) )); then
        echo "CMake ${version} is too old. Load a cmake module or install CMake >= 3.25 for target '${TARGET}'." >&2
        exit 1
    fi
}

# ── Load modules on HPC ──────────────────────────────────────────────────────
ENV_FILE="${ROOT_DIR}/cmake/env/${TARGET}.sh"
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading modules for ${TARGET}..."
    source "$ENV_FILE"
fi

require_cmake_minimum

if [[ -x "${ROOT_DIR}/tools/celephais_preflight.sh" ]]; then
    echo "Preflight for ${TARGET}..."
    PREFLIGHT_OUTPUT="$("${ROOT_DIR}/tools/celephais_preflight.sh" --target "${TARGET}" || true)"
    echo "$PREFLIGHT_OUTPUT"
    if grep -q '^READY=0$' <<<"$PREFLIGHT_OUTPUT"; then
        exit 2
    fi
fi

# ── Detect core count ────────────────────────────────────────────────────────
detect_jobs() {
    if [[ -n "${JOBS:-}" ]]; then
        echo "$JOBS"
        return
    fi

    local jobs=""
    if command -v nproc &>/dev/null; then
        jobs="$(nproc 2>/dev/null || true)"
    fi
    if [[ -z "$jobs" || "$jobs" -le 1 ]] && command -v getconf &>/dev/null; then
        jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    fi
    if [[ -z "$jobs" || "$jobs" -le 1 ]] && [[ "$(uname -s)" == "Darwin" ]]; then
        jobs="$(sysctl -n hw.ncpu)"
    fi
    if [[ -z "$jobs" || "$jobs" -le 1 ]]; then
        jobs=4
    fi
    echo "$jobs"
}

JOBS="$(detect_jobs)"

# ── Configure + Build ────────────────────────────────────────────────────────
# Generic `env` target: prefer the Ninja generator when ninja is on PATH. Ninja
# is faster than Unix Makefiles and avoids the GNU-make jobserver vs nested
# ExternalProject failure (bundled METIS/MUMPS), so it needs no serial-deps
# workaround. Falls back to the Make-based `env` preset when ninja is absent.
PRESET="${TARGET}"
if [ "${TARGET}" = "env" ] && command -v ninja >/dev/null 2>&1; then
    PRESET="env-ninja"
    echo "ninja found; using '${PRESET}' preset"
fi

BUILD_DIR="${ROOT_DIR}/build"
INSTALL_PREFIX="${CELEPHAIS_INSTALL_PREFIX:-${BUILD_DIR}/install}"

echo "Building Celephais [target=${PRESET}, jobs=${JOBS}]"
LC_ALL=C cmake --preset "${PRESET}"
LC_ALL=C cmake --build --preset "${PRESET}" --parallel "${JOBS}"

echo "Installing Celephais CMake package [prefix=${INSTALL_PREFIX}]"
LC_ALL=C cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}"

TARGETS_FILE="$(find "${INSTALL_PREFIX}" -type f -path '*/cmake/Celephais/CelephaisTargets.cmake' -print -quit)"
if [[ -z "${TARGETS_FILE}" ]]; then
    echo "CelephaisTargets.cmake was not installed under ${INSTALL_PREFIX}." >&2
    exit 1
fi

echo ""
echo "Build complete. Public entry: bin/celephais"
refresh_submission_links
print_submission_handoff
