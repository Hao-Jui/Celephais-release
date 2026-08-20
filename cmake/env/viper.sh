#!/bin/bash
# Module environment for Viper HPC (MPCDF)
# Sourced by compile.sh and apps/submission_scripts/viper.sh

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
module load metis/5.1
module load parmetis/4.0
module load mumps-32-noomp/5.6

if [ -z "${MUMPS_DIR:-}" ] && [ -n "${MUMPS_HOME:-}" ]; then
    export MUMPS_DIR="$MUMPS_HOME"
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
    "${FFTW_HOME:-}" \
    "${MKLROOT:-}" \
    "${METIS_HOME:-}" \
    "${PARMETIS_HOME:-}" \
    "${MUMPS_DIR:-}"; do
    [ -n "$_kadath_prefix" ] || continue
    _kadath_prepend_lib_path "$_kadath_prefix/lib"
    _kadath_prepend_lib_path "$_kadath_prefix/lib64"
    _kadath_prepend_lib_path "$_kadath_prefix/lib/intel64"
done
unset _kadath_prefix
