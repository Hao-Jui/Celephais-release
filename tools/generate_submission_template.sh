#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE=0
NAME=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        -h|--help)
            echo "Usage: tools/generate_submission_template.sh [--force] <name>"
            exit 0
            ;;
        *) NAME="$1"; shift ;;
    esac
done

if [[ ! "$NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Usage: tools/generate_submission_template.sh [--force] <name>" >&2
    exit 2
fi

OUT="$ROOT_DIR/apps/submission_scripts/${NAME}.sh"
if [ -e "$OUT" ] && [ "$FORCE" != 1 ]; then
    printf 'READY=0\n'
    printf 'TARGET_CLASS=slurm\n'
    printf 'BLOCKER=SBATCH_TEMPLATE_EXISTS\n'
    printf 'NEXT=review %s or rerun with --force\n' "$OUT"
    exit 2
fi

cat > "$OUT" <<EOF
#!/bin/bash -l
# Local Slurm entry script. Submit through apps/<APP>/sbatch_outputs/sub_Celephais.sh.

#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
#SBATCH -D ./
#SBATCH -J Celephais
##SBATCH --partition=<partition>
##SBATCH --account=<account>
##SBATCH --qos=<qos>
##SBATCH --nodes=<nodes>
##SBATCH --ntasks-per-node=<tasks-per-node>
##SBATCH --cpus-per-task=<cpus-per-task>
##SBATCH --time=<HH:MM:SS>

# After reviewing/filling the SBATCH lines above, change this exact line to:
# CELEPHAIS_SCHEDULER_READY=1
CELEPHAIS_SCHEDULER_READY=0
[ "\$CELEPHAIS_SCHEDULER_READY" = 1 ] || {
    echo "Edit apps/submission_scripts/${NAME}.sh: fill SBATCH resources, then set CELEPHAIS_SCHEDULER_READY=1." >&2
    exit 2
}

exec "\${CELEPHAIS_SUBMISSION_DIR:?submit via sub_Celephais.sh}/celephais_sbatch_entry.sh" "\$@"
EOF

chmod +x "$OUT"

printf 'READY=0\n'
printf 'TARGET_CLASS=slurm\n'
printf 'BLOCKER=SBATCH_RESOURCES\n'
printf 'NEXT=ask human to fill %s and set CELEPHAIS_SCHEDULER_READY=1\n' "$OUT"
