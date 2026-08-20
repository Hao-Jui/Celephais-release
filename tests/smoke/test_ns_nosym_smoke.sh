#!/bin/bash
# test_ns_nosym_smoke.sh — NS_nosym infrastructure smoke test.
# Verifies: binary launches, input parses, 3D nosym spherical domains construct,
# EOS loads, MPI initializes, and solver reaches pre-Newton setup phase.
# Does NOT check Newton convergence, Jacobian assembly, or residuals.
# Exit 0 = PASS, 1 = FAIL
set -euo pipefail

REPO_ROOT="${HOME_CELEPHAIS:?HOME_CELEPHAIS must be set}"
APP_DIR="$REPO_ROOT/apps/NS_nosym"
SANDBOX_DIR="${CELEPHAIS_APP_SANDBOX:-$APP_DIR/sandbox}"
INPUT="${NS_NOSYM_SMOKE_INPUT:-deg30_res9.toml}"

if [ ! -f "$SANDBOX_DIR/$INPUT" ]; then
    echo "SKIP: input file not found: $SANDBOX_DIR/$INPUT"
    exit 0
fi

if [ ! -x "$APP_DIR/bin/Release/solve" ]; then
    echo "SKIP: NS_nosym solve binary missing: $APP_DIR/bin/Release/solve"
    exit 0
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
cp -R "$SANDBOX_DIR"/. "$WORK_DIR"/
EXPECT_COLD_SEED=0
if [ ! -f "$WORK_DIR/${INPUT%.toml}.dat" ]; then
    EXPECT_COLD_SEED=1
fi

echo "Running NS_nosym smoke test..."
STDOUT="$(mktemp)"
cd "$WORK_DIR"
SMOKE_TIMEOUT="${NS_NOSYM_SMOKE_TIMEOUT:-60}"
SOLVER_RC=0
timeout "$SMOKE_TIMEOUT" mpirun -np 1 \
    -x HOME_CELEPHAIS \
    "$APP_DIR/bin/Release/solve" "$INPUT" . > "$STDOUT" 2>&1 || SOLVER_RC=$?

# Crash signals: SIGSEGV=139, SIGABRT=134, SIGFPE=136, SIGBUS=138
if [ "$SOLVER_RC" -ne 0 ] && [ "$SOLVER_RC" -ne 124 ]; then
    echo "FAIL: NS_nosym crashed (exit $SOLVER_RC)"
    tail -20 "$STDOUT" >&2
    rm -f "$STDOUT"
    exit 1
fi

# Pre-Newton marker: printed during driver setup before any Newton loop
if grep -q "Solutions will be stored in:" "$STDOUT"; then
    echo "  ok  pre-Newton setup marker found (3D nosym spherical domains constructed)"
else
    echo "  FAIL  'Solutions will be stored in:' not found — solver crashed during domain construction"
    tail -20 "$STDOUT" >&2
    rm -f "$STDOUT"
    exit 1
fi

if [ "$EXPECT_COLD_SEED" -eq 1 ]; then
    if grep -q "Generating NS_nosym seed from a converged 2D XCTS solve at resolution" "$STDOUT" &&
       grep -q "Resolution of lifted 3D space:" "$STDOUT"; then
        echo "  ok  tilted 2D XCTS cold-seed solve and no-sym lift completed"
    else
        echo "FAIL: tilted 2D XCTS cold-seed markers not found"
        tail -30 "$STDOUT" >&2
        rm -f "$STDOUT"
        exit 1
    fi
fi

# Sanitizer check
if grep -qE "AddressSanitizer|UndefinedBehaviorSanitizer|runtime error:" "$STDOUT"; then
    echo "FAIL: sanitizer report detected"
    grep -E "AddressSanitizer|UndefinedBehaviorSanitizer|runtime error:" "$STDOUT" | head -5 >&2
    rm -f "$STDOUT"
    exit 1
fi

rm -f "$STDOUT"
echo "PASS: NS_nosym smoke (launched, 3D nosym domains constructed, no crash, no sanitizer)"
exit 0
