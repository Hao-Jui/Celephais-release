#!/bin/bash
# Module environment for bwForCluster BinAC 2.
# Sourced by compile.sh (build) and the Slurm entry script (runtime), so the
# build and the run see an identical stack.
#
# CC/CXX/FC are the OpenMPI WRAPPERS on purpose. On BinAC2, find_package(MPI)
# only passes its MPI_<lang>_WORKS compile/link test when the compiler IS the
# wrapper; plain gcc + FindMPI flag-extraction fails (the wrapper pulls
# UCX/libfabric rpath that `mpif90 -showme` does not fully reproduce). The price
# of wrapper-as-compiler is that FindMPI returns an EMPTY MPI::MPI_Fortran, so
# the bundled (Fortran) MUMPS loses its mpi_bcast_/mpi_pack_ bindings at the
# C++-driven final link. We restore them through CELEPHAIS_MUMPS_EXTRA_LIBRARIES
# below — the same site-tail mechanism the ncsa/delta/snowman toolchains use.

if ! type module >/dev/null 2>&1; then
    if [ -r /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -r /usr/share/lmod/lmod/init/bash ]; then
        source /usr/share/lmod/lmod/init/bash
    elif [ -r /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    fi
fi

# Lmod is hierarchical: compiler first, then MPI, then the libs under that
# compiler+MPI branch. mkl lives under the intel toolkit, so load it after.
module purge
module load compiler/gnu/13.3
module load mpi/openmpi/4.1-gnu-13.3
module load numlib/fftw/3.3.10-openmpi-4.1-gnu-13.3
module load lib/boost/1.88.0-openmpi-4.1-gnu-13.3
module load devel/cmake/3.29
module load devel/ninja/1.12.1
module load toolkit/intel/OneAPI-2025.0
module load mkl/2025.0          # MKL BLAS/LAPACK/ScaLAPACK + BLACS-openmpi

export CC=mpicc
export CXX=mpicxx
export FC=mpif90

# srun must launch OpenMPI via PMIx: SLURM's MpiDefault is unset on BinAC2, so a
# bare `srun` negotiates no PMI and OpenMPI 4.1's ext3x client aborts MPI_Init
# ("OMPI was not built with SLURM's PMI support"). pmix_v3 matches ext3x.
export SLURM_MPI_TYPE=pmix

export FFTW_HOME="${FFTW_HOME:-${FFTW_MPI_HOME:-/opt/bwhpc/common/numlib/fftw-mpi/3.3.10-openmpi-4.1-gnu-13.3}}"
export BOOST_ROOT="${BOOST_ROOT:-${BOOST_HOME:-/opt/bwhpc/common/lib/boost/1.88.0-openmpi-4.1-gnu-13.3}}"
# MUMPS is built in-tree from third_party/mumps; MKL supplies ScaLAPACK/BLAS/LAPACK.
# No MUMPS_DIR/PETSC_DIR (system/PETSc-MUMPS discovery was removed).
export SCALAPACK_DIR="${SCALAPACK_DIR:-$MKLROOT}"

# The EOS/data tree must resolve inside THIS checkout. A stale HOME_CELEPHAIS in
# the user profile is carried into the job by sbatch --export=ALL, and
# celephais_sbatch_entry only sets it :-default, so the stale value would win and
# the solver would read another tree's EOS. Force it to this checkout's root.
export HOME_CELEPHAIS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Fortran MPI bindings + gfortran runtime that the bundled MUMPS archives need at
# the final C++ link (MPI::MPI_Fortran is empty under wrapper-compilers, above).
# Derived from the loaded mpif90 so it follows the MPI module, not a pinned path.
if [ -z "${CELEPHAIS_MUMPS_EXTRA_LIBRARIES:-}" ]; then
    _ompi_prefix="$(dirname "$(dirname "$(command -v mpif90)")")"
    for _d in lib64 lib; do
        if [ -f "${_ompi_prefix}/${_d}/libmpi_mpifh.so" ]; then
            export CELEPHAIS_MUMPS_EXTRA_LIBRARIES="${_ompi_prefix}/${_d}/libmpi_mpifh.so;${_ompi_prefix}/${_d}/libmpi.so;gfortran"
            break
        fi
    done
    unset _ompi_prefix _d
fi
