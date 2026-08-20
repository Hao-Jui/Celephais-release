#!/bin/bash
# test_ns2d_smoke.sh — NS2d infrastructure smoke test.
# Verifies: binary launches, input parses, EOS loads, domains construct, MPI initializes.
# Does NOT check Newton convergence, Jacobian assembly, or residuals.
# Exit 0 = PASS, 1 = FAIL
set -euo pipefail

REPO_ROOT="${HOME_CELEPHAIS:?HOME_CELEPHAIS must be set}"
APP_DIR="$REPO_ROOT/apps/NS2d"
SANDBOX_DIR="${CELEPHAIS_APP_SANDBOX:-$APP_DIR/sandbox}"
INPUT="initial.toml"

if [ ! -f "$SANDBOX_DIR/$INPUT" ]; then
    echo "SKIP: input file not found: $SANDBOX_DIR/$INPUT"
    exit 0
fi

if [ ! -x "$APP_DIR/bin/Release/solve" ]; then
    echo "SKIP: NS2d solve binary missing: $APP_DIR/bin/Release/solve"
    exit 0
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
cp -R "$SANDBOX_DIR"/. "$WORK_DIR"/

echo "Running NS2d smoke test..."
STDOUT="$(mktemp)"
cd "$WORK_DIR"
SMOKE_TIMEOUT="${NS2D_SMOKE_TIMEOUT:-30}"
SOLVER_RC=0
timeout "$SMOKE_TIMEOUT" mpirun -np 1 "$APP_DIR/bin/Release/solve" "$INPUT" . > "$STDOUT" 2>&1 || SOLVER_RC=$?

# Crash signals: SIGSEGV=139, SIGABRT=134, SIGFPE=136, SIGBUS=138
# Max-iterations is a non-crash solver exit — tolerate it since this
# test does NOT check convergence (only infrastructure: launch, parse,
# EOS load, domain construct, MPI init).
if [ "$SOLVER_RC" -ne 0 ] && [ "$SOLVER_RC" -ne 124 ]; then
    if grep -q "Maximum iterations exceeded" "$STDOUT"; then
        echo "  note  solver hit max iterations (non-crash, continuing)"
    else
        echo "FAIL: NS2d crashed (exit $SOLVER_RC)"
        tail -20 "$STDOUT" >&2
        rm -f "$STDOUT"
        exit 1
    fi
fi

# Require non-empty output — solver must have started and emitted something
if [ ! -s "$STDOUT" ]; then
    echo "FAIL: NS2d produced no output"
    rm -f "$STDOUT"
    exit 1
fi

# Domain construction marker — the NS2d solver prints iteration lines
# once the space is built and the Newton loop starts.
if grep -qE "Iter:|generated a full" "$STDOUT"; then
    echo "  ok  domain construction marker found"
else
    echo "  FAIL  domain construction marker not found (solver may have crashed during init)"
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
echo "PASS: NS2d smoke (launched, domains constructed, no crash, no sanitizer)"
exit 0
