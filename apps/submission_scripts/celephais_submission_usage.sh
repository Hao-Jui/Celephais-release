#!/bin/bash
# Print the human-facing build handoff.

external_plugin="${2:-}"

cat <<'EOF'
  ╔══════════════════════════════════════╗
  ║          C E L E P H A I S           ║
  ║  Spectral Solver for Compact Binary  ║
  ╚══════════════════════════════════════╝
EOF

if [[ -n "${external_plugin}" ]]; then
    printf '\nPrivate workers: libexec/celephais/\n'
    printf 'External plugin: %s\n' "${external_plugin}"
fi
printf '\n'

cat <<'EOF'
Run from the working directory.
./sub_Celephais.sh parameter.toml

One arg: bin/celephais with the top-level run specification.
Two+ args: private-worker name or path=$1, INPUT_FILE=$2, rest are worker args.
EOF
