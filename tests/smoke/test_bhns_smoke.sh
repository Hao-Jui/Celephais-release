#!/bin/bash
# test_bhns_smoke.sh — BHNS infrastructure smoke test.
# Verifies: binary launches, input parses, MPI initializes, and the solver reaches
# the BHNS setup phase (binary-config setup) without an early crash. Runs with
# BHNS_STOP_AFTER_SETUP=1, which skips the two heavy component seed solves
# (a full NS + BH XCTS Newton solve each — minutes, and the source of the old
# timeout-kill flakiness) and exits cleanly in seconds.
# Does NOT check Newton convergence, Jacobian assembly, residuals, or the seed solves.
# Exit 0 = PASS, 1 = FAIL
set -euo pipefail

REPO_ROOT="${HOME_CELEPHAIS:?HOME_CELEPHAIS must be set}"
APP_DIR="$REPO_ROOT/apps/BHNS"
SANDBOX_DIR="${CELEPHAIS_APP_SANDBOX:-$APP_DIR/sandbox}"
INPUT="initial.toml"

if [ ! -f "$SANDBOX_DIR/$INPUT" ]; then
    echo "SKIP: input file not found: $SANDBOX_DIR/$INPUT"
    exit 0
fi

if [ ! -x "$APP_DIR/bin/Release/solve" ]; then
    echo "SKIP: BHNS solve binary missing: $APP_DIR/bin/Release/solve"
    exit 0
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
cp -R "$SANDBOX_DIR"/. "$WORK_DIR"/

echo "Running BHNS smoke test..."
STDOUT="$(mktemp)"
cd "$WORK_DIR"
SMOKE_TIMEOUT="${BHNS_SMOKE_TIMEOUT:-90}"
SOLVER_RC=0
# BHNS_STOP_AFTER_SETUP=1 returns cleanly right after the 3D bispheric
# space is constructed, before the (compute-bound) Newton solve. This exercises
# launch + input parse + domain construction + MPI init and exits 0 in seconds
# — deterministic, no reliance on a timeout kill (whose SIGTERM used to drop the
# early banner from block-buffered stdout, making this test flaky).
timeout "$SMOKE_TIMEOUT" mpirun -np 1 \
    -x HOME_CELEPHAIS \
    -x BHNS_STOP_AFTER_SETUP=1 \
    "$APP_DIR/bin/Release/solve" "$INPUT" . > "$STDOUT" 2>&1 || SOLVER_RC=$?

# A clean setup-only run exits 0. Non-zero = a crash (SIGSEGV=139, SIGABRT=134,
# SIGFPE=136, SIGBUS=138) or a timeout (124 = setup hung / far too slow) — both
# are failures here.
if [ "$SOLVER_RC" -ne 0 ]; then
    echo "FAIL: BHNS setup did not exit cleanly (exit $SOLVER_RC)"
    tail -20 "$STDOUT" >&2
    rm -f "$STDOUT"
    exit 1
fi

# Positive marker printed (rank 0, flushed) once setup is reached, before the
# skipped component seed solves.
if grep -q "BHNS setup phase reached" "$STDOUT"; then
    echo "  ok  launch + MPI + parse + binary-config setup reached cleanly"
else
    echo "  FAIL  setup marker not found — solver exited 0 without reaching the setup stop"
    tail -20 "$STDOUT" >&2
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
echo "PASS: BHNS smoke (launched, MPI + parse + binary-config setup, clean exit, no sanitizer)"
exit 0
