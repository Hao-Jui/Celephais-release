#!/bin/bash
# Module environment for NCSA Campus Cluster.
# Sourced by compile.sh and human-provided NCSA submission scripts.

if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -r /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    fi
fi

module purge
module load gcc/13.3.0
module load openmpi/5.0.1-gcc-13.3.0
module load fftw/3.3.10
module load boost/1.8.7
module load intel/tbb/2022.0
module load intel/compiler-rt/2025.0.4
module load intel/mkl/2025.0

export CC=mpicc
export CXX=mpicxx
export FC=mpifort

export BOOST_ROOT="${BOOST_ROOT:-/sw/apps/boost/1.87.0}"
export SCALAPACK_DIR="${SCALAPACK_DIR:-$MKLROOT}"
export MUMPS_DIR="${MUMPS_DIR:-/u/hjkuan/scratch/local/MUMPS_5.5.1}"

_kadath_ncsa_ld_paths=(
    "${MUMPS_DIR}/lib"
    "${MKLROOT:-}/lib"
    "${FFTW_HOME:-}/lib"
    "${BOOST_ROOT}/lib"
)
for _kadath_ncsa_path in "${_kadath_ncsa_ld_paths[@]}"; do
    if [[ -d "$_kadath_ncsa_path" ]]; then
        export LD_LIBRARY_PATH="${_kadath_ncsa_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
unset _kadath_ncsa_path _kadath_ncsa_ld_paths
