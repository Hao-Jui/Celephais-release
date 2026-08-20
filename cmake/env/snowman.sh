#!/bin/bash
# Workstation environment for snowman. No scheduler/modules are required.

export KADATH_DEPS_ROOT="${KADATH_DEPS_ROOT:-${HOME:?HOME is required}/scratch3/kadath-local}"

export CC="${CC:-mpicc}"
export CXX="${CXX:-mpicxx}"
export FC="${FC:-mpifort}"

export FFTW_HOME="${FFTW_HOME:-$KADATH_DEPS_ROOT/usr}"
export FFTW_ROOT="${FFTW_ROOT:-$FFTW_HOME}"
export FFTWDIR="${FFTWDIR:-$FFTW_HOME}"
export SCALAPACK_DIR="${SCALAPACK_DIR:-$KADATH_DEPS_ROOT/usr}"
export MUMPS_DIR="${MUMPS_DIR:-$KADATH_DEPS_ROOT/MUMPS_5.5.1}"

_kadath_snowman_ld_paths=(
    "$KADATH_DEPS_ROOT/usr/lib/x86_64-linux-gnu"
    "$MUMPS_DIR/lib"
)
for _kadath_snowman_path in "${_kadath_snowman_ld_paths[@]}"; do
    if [[ -d "$_kadath_snowman_path" ]]; then
        export LD_LIBRARY_PATH="${_kadath_snowman_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
unset _kadath_snowman_path _kadath_snowman_ld_paths
