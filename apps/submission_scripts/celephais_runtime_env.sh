#!/bin/bash
# Shared runtime environment for all Celephais app launchers.
#
# Source from apps/<APP>/sbatch_outputs/sub_Celephais.sh after setting optional
# per-app defaults such as CELEPHAIS_SOLVER_DEFAULT=dense.

if [ -z "${HOME_CELEPHAIS:-}" ]; then
    _celephais_runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if ! HOME_CELEPHAIS="$(git -C "$_celephais_runtime_dir" rev-parse --show-toplevel 2>/dev/null)"; then
        HOME_CELEPHAIS="$(cd "$_celephais_runtime_dir/../.." && pwd)"
    fi
    unset _celephais_runtime_dir
fi
export HOME_CELEPHAIS

# Solver selection. Default to the measured production winner: exact Jv with
# sparse MUMPS right preconditioning.
export CELEPHAIS_SOLVER="${CELEPHAIS_SOLVER:-${CELEPHAIS_SOLVER_DEFAULT:-jfnk-mumps}}"

# Production diagnostics.
export CELEPHAIS_TIMING="${CELEPHAIS_TIMING:-0}"
export DEF_FILTER="${DEF_FILTER:-1}"
export JACOBIAN_ASSEMBLER_PROFILE="${JACOBIAN_ASSEMBLER_PROFILE:-0}"
# Reuse immutable direct-column planning and classified column metadata across
# repeated JacobianAssembler instances on one unchanged System_of_eqs.
export JACOBIAN_STRUCTURAL_PLAN_CACHE="${JACOBIAN_STRUCTURAL_PLAN_CACHE:-1}"
# Measured y-reflection cross-sector masking. Unsupported spaces and coupled
# systems self-disable; exact 0 selects the unmasked legacy path.
export SPARSE_PARITY_MASK="${SPARSE_PARITY_MASK:-1}"
# Emission-time form of the parity mask (default ON since 2026-08-07 proof:
# zero fused defects across direct r13 QE and JFNK res7 arms). J1 fuses
# optimistically when complete structural labels exist; exact 0 opts out.
export JACOBIAN_FUSED_PARITY_MASK="${JACOBIAN_FUSED_PARITY_MASK:-1}"
# Certified exact-sector Newton reduction. Exact 0 selects the unreduced path.
export SPARSE_SECTOR_REDUCE="${SPARSE_SECTOR_REDUCE:-1}"
# Factor the two disconnected parity sector blocks of a fused-mask Jacobian one
# at a time, so peak factor residency is the larger block instead of the sum.
export SPARSE_PARITY_SPLIT_SOLVE="${SPARSE_PARITY_SPLIT_SOLVE:-1}"
# Experimental append-only rank-local COO storage. Fixed-capacity blocks avoid
# geometric row/value-vector relocation during the first Jacobian assembly.
export JACOBIAN_LOCAL_COO_BLOCKS="${JACOBIAN_LOCAL_COO_BLOCKS:-0}"
# W=32 is the production packed-column path. W=16/W=8/W=4/W=2 stay enabled as
# deterministic fallbacks for rank-fragmented buckets and leftover groups.
export JACOBIAN_WLANE32="${JACOBIAN_WLANE32:-1}"
export JACOBIAN_WLANE16="${JACOBIAN_WLANE16:-1}"
export JACOBIAN_WLANE8="${JACOBIAN_WLANE8:-1}"
export JACOBIAN_WLANE4="${JACOBIAN_WLANE4:-1}"
export JACOBIAN_WLANE2="${JACOBIAN_WLANE2:-1}"
# Experimental global group ownership. Keep off until rank-balance, canonical
# COO hash, RSS, and end-to-end convergence gates have been measured.
export JACOBIAN_GLOBAL_GROUP_PLAN="${JACOBIAN_GLOBAL_GROUP_PLAN:-0}"
# Packed adapted-geometry tangents are restricted to the explicitly supported
# no-symmetry BNS space. W=2 is the qualified production default; wider lanes
# remain available for workload-specific A/B through VARDOM_MAX_WIDTH.
export JACOBIAN_VARDOM_WLANE2="${JACOBIAN_VARDOM_WLANE2:-1}"
export JACOBIAN_VARDOM_MAX_WIDTH="${JACOBIAN_VARDOM_MAX_WIDTH:-2}"
# Matching-equation lane export and no-symmetry point-major import batching.
# Both are qualified defaults and remain independently overridable for ablations.
export MATCHING_LANE_EXPORT="${MATCHING_LANE_EXPORT:-1}"
export MATCHING_IMPORT_LANE_BATCH="${MATCHING_IMPORT_LANE_BATCH:-1}"
# Val_domain::compute_der_abs cache. Memoises the spectral der_abs result keyed
# on coefficient bits. Default-on in assembly and exact-Jv GMRES scopes after
# full BNS convergence gates cleared.
export VAL_DOMAIN_DER_ABS_CACHE="${VAL_DOMAIN_DER_ABS_CACHE:-1}"
export DO_JX_DER_ABS_CACHE="${DO_JX_DER_ABS_CACHE:-1}"
# Derivative-lane tiling, batched one-dimensional traversal, and its bounded
# workspace are qualified production defaults. Each remains independently
# overridable for exact legacy-path ablations.
# The cap bounds logical workspace ownership per rank for one Jacobian
# assembly; scope teardown releases that ownership, while MemoryMapper may
# retain one legacy-sized line backing block for reuse.
export DERIVATIVE_LANE_TILING="${DERIVATIVE_LANE_TILING:-1}"
export OPE_DER_BATCH="${OPE_DER_BATCH:-1}"
# Coefficient-major, lane-contiguous kernels for the BNS Chebyshev/Fourier
# basis set. Kept independently opt-in until a full convergence/RSS gate.
export OPE_DER_AOSOA="${OPE_DER_AOSOA:-0}"
export OPE1D_WORKSPACE="${OPE1D_WORKSPACE:-1}"
export OPE1D_WORKSPACE_MAX_BYTES="${OPE1D_WORKSPACE_MAX_BYTES:-1048576}"
# Reuse compatible persistent Term_eq result storage. Set to 0 for the legacy
# deep-assignment path during workload-specific parity or performance ablation.
export OPE_ACTION_WRITE_INTO="${OPE_ACTION_WRITE_INTO:-1}"

# MUMPS knobs.
export DROP_TOL="${DROP_TOL:-1e-14}"
export MUMPS_BLR="${MUMPS_BLR:-0}"
export MUMPS_BLR_DROP_TOL="${MUMPS_BLR_DROP_TOL:-0.0}"
# Exact tri-state: 0 forces in-core, 1 forces out-of-core, and auto retains the
# factor-time memory-budget policy. Invalid values fall back to auto in config.
export MUMPS_OOC="${MUMPS_OOC:-auto}"
export MUMPS_OOC_TOUCH="${MUMPS_OOC_TOUCH:-1.3}"
export MUMPS_OOC_SAFETY="${MUMPS_OOC_SAFETY:-0.7}"
# Test-only node-available-memory override. Empty means use the platform probe;
# production must not invent a numeric budget when that probe is unreadable.
export MUMPS_OOC_BUDGET_MB="${MUMPS_OOC_BUDGET_MB:-}"
export MUMPS_ORDERING="${MUMPS_ORDERING:-7}"
# Persist the first JFNK preconditioner's matching/ordering pair beside the
# stage solution and replay it on later refreshes. Default OFF (dormant
# insurance; production stages rarely refresh). Opt in with
# MUMPS_TREE_CACHE=1; an invalid replay falls back to the ordinary
# fresh analyze path.
export MUMPS_TREE_CACHE="${MUMPS_TREE_CACHE:-0}"
# Sparse-direct symbolic reuse stays opt-in after production testing found
# support growth at the fixed numerical threshold. The -1 pattern setting tracks
# that threshold, avoiding an RSS-heavy 1e-16 superset when reuse is requested.
export SPARSE_MUMPS_ANALYZE_REUSE="${SPARSE_MUMPS_ANALYZE_REUSE:-0}"
export SPARSE_CHORD_REUSE="${SPARSE_CHORD_REUSE:-1}"
export SPARSE_MUMPS_PATTERN_DROP_TOL="${SPARSE_MUMPS_PATTERN_DROP_TOL:--1}"
export SPARSE_MUMPS_SUPERSET_MAX_NNZ_RATIO="${SPARSE_MUMPS_SUPERSET_MAX_NNZ_RATIO:-2}"
# Diagnostic-only exact sparse-direct COO/RHS capture. The empty path keeps the
# writer completely inactive; the ordinal selects one assembly in staged runs.
export DIRECT_REPLAY_CAPTURE="${DIRECT_REPLAY_CAPTURE:-}"
export DIRECT_REPLAY_CAPTURE_ORDINAL="${DIRECT_REPLAY_CAPTURE_ORDINAL:-1}"

# JFNK-MUMPS controls. The sparse MUMPS factor is the right preconditioner;
# exact Jv is applied matrix-free inside GMRES.
export JFNK_MAX_ITERS="${JFNK_MAX_ITERS:-48}"
# Cadence 10 (was 5): the res11 z40 ladder measured both FB-stage age-5 refreshes as
# pure waste (identical Newton trajectory with and without; -31% e2e without, HANDOFF
# 2026-08-05). 10 keeps a bounded backstop for stages that genuinely degrade their PC;
# the 10x-error-growth guard and recovery_retry rebuild paths are unaffected.
export JFNK_MUMPS_PC_REFRESH="${JFNK_MUMPS_PC_REFRESH:-10}"
# The measured cost-aware cadence remains opt-in until full-workflow gates have
# established that it reduces wall without changing convergence.
export JFNK_MUMPS_PC_ADAPTIVE="${JFNK_MUMPS_PC_ADAPTIVE:-0}"
export JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS="${JFNK_MUMPS_PC_ADAPTIVE_MAX_STEPS:-20}"
export JFNK_RTOL="${JFNK_RTOL:-1e-8}"
# Eisenstat-Walker forcing-term controller (choice 2). Default on after BNS G0
# full-convergence A/B passed at gamma=0.9, alpha=2.0, eta_max=1e-3 (-5.95 %
# total wall on BNS, -3.6 % on NS, no extra Newton or J-build). Opt out with
# JFNK_EW=0; JFNK_RTOL stays the floor on eta.
export JFNK_EW="${JFNK_EW:-1}"
export JFNK_EW_GAMMA="${JFNK_EW_GAMMA:-0.9}"
export JFNK_EW_ALPHA="${JFNK_EW_ALPHA:-2.0}"
export JFNK_EW_MAX="${JFNK_EW_MAX:-1e-3}"
# Guarded backtracking line search. Default on — a strict no-op on healthy
# solves (Newton trajectory bit-identical to the full-step path); only engages
# when a full Newton step would increase |F|. Safe on every space now that the
# 2D polar adapted snapshot/restore is bit-exact. Set to 0 to disable.
export JFNK_LINESEARCH="${JFNK_LINESEARCH:-1}"
export JFNK_LS_ARMIJO="${JFNK_LS_ARMIJO:-0}"
export JFNK_LS_ALPHA="${JFNK_LS_ALPHA:-1e-4}"
export JFNK_LS_MAX_BACKTRACK="${JFNK_LS_MAX_BACKTRACK:-3}"
export JFNK_LS_MIN_LAMBDA="${JFNK_LS_MIN_LAMBDA:-0.1}"

# BLAS / LAPACK thread caps (prevent over-subscription against MPI ranks).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# macOS libmalloc retains freed per-Jacobian rank-0 arrays in its large-block
# death-row cache (~0.8 GiB at BNS res7/np4, zero wall cost; HANDOFF 2026-08-09).
if [ "$(uname -s)" = "Darwin" ]; then
    export MallocLargeCache="${MallocLargeCache:-0}"
fi

# OpenMPI / PMIx launcher noise reduction (controls launcher, not children).
export PMIX_MCA_pmix_base_tool_support="${PMIX_MCA_pmix_base_tool_support:-0}"

case "$CELEPHAIS_SOLVER" in
    dense | mumps | jfnk-mumps) ;;
    *)
        echo "Invalid CELEPHAIS_SOLVER='$CELEPHAIS_SOLVER' (expected: dense|mumps|jfnk-mumps)" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
