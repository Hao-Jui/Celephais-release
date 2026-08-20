#!/bin/bash
# Module environment for Raven HPC (MPCDF)
# Sourced by compile.sh and apps/submission_scripts/raven.sh.

if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
        [ -r "$f" ] && source "$f" && break
    done
fi

module purge
module load gcc/14
module load ninja/1.11
module load boost/1.83
module load impi/2021.11
module load fftw-mpi/3.3.10
module load mkl/2025.2
module load petsc-real-double/3.22

if [ -z "${PETSC_DIR:-}" ] && [ -n "${PETSC_HOME:-}" ]; then
    export PETSC_DIR="$PETSC_HOME"
fi

_kadath_prepend_lib_path() {
    [ -d "$1" ] || return 0
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$1:"*) ;;
        *) export LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}

for _kadath_prefix in \
    "${BOOST_ROOT:-}" \
    "${BOOST_HOME:-}" \
    "${FFTW_HOME:-}" \
    "${MKLROOT:-}" \
    "${I_MPI_ROOT:-}" \
    "${PETSC_DIR:-}"; do
    [ -n "$_kadath_prefix" ] || continue
    _kadath_prepend_lib_path "$_kadath_prefix/lib"
    _kadath_prepend_lib_path "$_kadath_prefix/lib64"
    _kadath_prepend_lib_path "$_kadath_prefix/lib/intel64"
    _kadath_prepend_lib_path "$_kadath_prefix/lib/release"
done
unset _kadath_prefix
