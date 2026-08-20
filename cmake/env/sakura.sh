#!/bin/bash
# Module environment for Sakura HPC (MPCDF)
# Sourced by compile.sh and apps/submission_scripts/sakura.sh
module purge
module load git/2.50
module load gcc/14
module load ninja/1.11
module load gsl/2.7
module load impi/2021.11
module load mkl/2025.2
