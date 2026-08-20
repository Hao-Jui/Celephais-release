#!/bin/bash
# Module environment for NCSA Delta.
# Sourced by compile.sh and human-provided Delta submission scripts.
module purge
module load gcc-native/13.2
module load craype/2.7.34
module load libfabric/1.22.0
module load craype-network-ofi
module load cray-mpich/8.1.32
module load cray-libsci/25.03.0
module load PrgEnv-gnu/8.6.0
module load craype-x86-milan
module load fftw/3.3.10-gcc13.3.1
module load boost/1.88.0-gcc13.3.1
module load gmake/4.4.1-gcc13.3.1

export CC=cc
export CXX=CC
export FC=ftn

export CRAY_LIBSCI_PREFIX=/opt/cray/pe/libsci/25.03.0/GNU/12.2/x86_64
export SCALAPACK_DIR="${CRAY_LIBSCI_PREFIX}"
export MUMPS_DIR="${MUMPS_DIR:-/projects/bdur/hjkuan/local/MUMPS_5.5.1}"

_kadath_delta_ld_paths=(
    "${FFTW_HOME:-}/lib"
    "${CRAY_LIBSCI_PREFIX}/lib"
    "${MUMPS_DIR}/lib"
    "/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib"
    "/opt/cray/pe/dsmml/0.3.1/dsmml/lib"
)
for _kadath_delta_path in "${_kadath_delta_ld_paths[@]}"; do
    if [[ -d "$_kadath_delta_path" ]]; then
        export LD_LIBRARY_PATH="${_kadath_delta_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
unset _kadath_delta_path _kadath_delta_ld_paths
