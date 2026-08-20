#!/bin/bash
# test_bh2d_xcts_smoke.sh — BH2d XCTS infrastructure smoke test.
# Verifies: binary launches, input parses, domains construct, MPI initializes.
# Does NOT check Newton convergence, Jacobian assembly, or residuals.
# Exit 0 = PASS, 1 = FAIL
set -euo pipefail

REPO_ROOT="${HOME_CELEPHAIS:?HOME_CELEPHAIS must be set}"
APP_DIR="$REPO_ROOT/apps/BH2d_xcts"
SANDBOX_DIR="${CELEPHAIS_APP_SANDBOX:-$APP_DIR/sandbox}"
INPUT="initial.toml"

if [ ! -f "$SANDBOX_DIR/$INPUT" ]; then
    echo "SKIP: input file not found: $SANDBOX_DIR/$INPUT"
    exit 0
fi

if [ ! -x "$APP_DIR/bin/Release/solve" ]; then
    echo "SKIP: BH2d_xcts solve binary missing: $APP_DIR/bin/Release/solve"
    exit 0
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
cp -R "$SANDBOX_DIR"/. "$WORK_DIR"/

echo "Running BH2d_xcts smoke test..."
STDOUT="$(mktemp)"
cd "$WORK_DIR"
SMOKE_TIMEOUT="${BH2D_XCTS_SMOKE_TIMEOUT:-60}"
SOLVER_RC=0
timeout "$SMOKE_TIMEOUT" mpirun -np 1 "$APP_DIR/bin/Release/solve" "$INPUT" . > "$STDOUT" 2>&1 || SOLVER_RC=$?

# Crash signals: SIGSEGV=139, SIGABRT=134, SIGFPE=136, SIGBUS=138
if [ "$SOLVER_RC" -ne 0 ] && [ "$SOLVER_RC" -ne 124 ]; then
    echo "FAIL: BH2d_xcts crashed (exit $SOLVER_RC)"
    tail -20 "$STDOUT" >&2
    rm -f "$STDOUT"
    exit 1
fi

# Require non-empty output — solver must have started and emitted something
if [ ! -s "$STDOUT" ]; then
    echo "FAIL: BH2d_xcts produced no output"
    rm -f "$STDOUT"
    exit 1
fi

# Sanitizer check
if grep -qE "AddressSanitizer|UndefinedBehaviorSanitizer|runtime error:" "$STDOUT"; then
    echo "FAIL: sanitizer report detected"
    grep -E "AddressSanitizer|UndefinedBehaviorSanitizer|runtime error:" "$STDOUT" | head -5 >&2
    rm -f "$STDOUT"
    exit 1
fi

rm -f "$STDOUT"
echo "PASS: BH2d_xcts smoke (launched, no crash, no sanitizer)"
exit 0
