#!/bin/bash
# Shared helpers for machine-local Celephais environment files.

celephais_module_init() {
    if type module >/dev/null 2>&1; then
        return 0
    fi

    local init_file
    for init_file in \
        /etc/profile.d/modules.sh \
        /usr/share/lmod/lmod/init/bash \
        /usr/share/Modules/init/bash; do
        if [ -r "$init_file" ]; then
            # shellcheck disable=SC1090
            source "$init_file"
            break
        fi
    done
}

celephais_prepend_path_var() {
    local var_name="$1"
    local path_value="$2"
    [ -n "$path_value" ] && [ -d "$path_value" ] || return 0
    eval "case \":\${${var_name}:-}:\" in *\":${path_value}:\"*) ;; *) export ${var_name}=\"${path_value}\${${var_name}:+:\$${var_name}}\" ;; esac"
}

celephais_apply_prefix_aliases() {
    if [ -z "${BOOST_ROOT:-}" ] && [ -n "${BOOST_HOME:-}" ]; then
        export BOOST_ROOT="$BOOST_HOME"
    fi
    # Legacy provider modules may set FFTW_MPI_HOME even when consumers use
    # only the serial libfftw3 library from that prefix.
    if [ -z "${FFTW_HOME:-}" ] && [ -n "${FFTW_MPI_HOME:-}" ]; then
        export FFTW_HOME="$FFTW_MPI_HOME"
    fi
}

celephais_add_runtime_paths() {
    [ "$(uname -s 2>/dev/null)" != "Darwin" ] || return 0

    local prefix subdir
    for prefix in "$@"; do
        [ -n "$prefix" ] || continue
        for subdir in lib lib64 lib/intel64 lib/release lib/x86_64-linux-gnu; do
            celephais_prepend_path_var LD_LIBRARY_PATH "$prefix/$subdir"
        done
    done
}

celephais_add_default_runtime_paths() {
    celephais_apply_prefix_aliases
    celephais_add_runtime_paths \
        "${FFTW_HOME:-}" \
        "${FFTW_MPI_HOME:-}" \
        "${BOOST_ROOT:-}" \
        "${BOOST_HOME:-}" \
        "${SCALAPACK_DIR:-}" \
        "${SCALAPACK_ROOT:-}" \
        "${MKLROOT:-}" \
        "${I_MPI_ROOT:-}"
}
