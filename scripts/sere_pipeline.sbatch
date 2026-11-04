#!/usr/bin/env bash
#SBATCH --job-name=sere-pipeline
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --chdir=/zfsauton2/home/anqix/projects/stamp
#SBATCH --output=/zfsauton/scratch/anqix/stamp/slurm/sere-%j.out
#SBATCH --error=/zfsauton/scratch/anqix/stamp/slurm/sere-%j.err
#SBATCH --open-mode=append

# Submit with submit_sere_pipeline_tmux.sh (creates log directory first).
# Do not start a detached tmux training process inside a batch allocation:
# ending the batch script would release its GPUs.
set -Eeuo pipefail

SERE_REPO_DIR="${SERE_REPO_DIR:-/zfsauton2/home/anqix/projects/stamp}"
SERE_PYTHON="${SERE_PYTHON:-/zfsauton2/home/anqix/miniconda3/envs/stamp2/bin/python}"
SERE_CONFIG="${SERE_CONFIG:-configs/sere_temporal_lopo.json}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd "$SERE_REPO_DIR"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo 'Submit this file with sbatch; do not run it with bash on a login node.' >&2
    exit 2
fi
[[ -x "$SERE_PYTHON" && -f "$SERE_CONFIG" ]]

active_step=''
phase='preflight'
stop_job() {
    trap '' TERM INT HUP
    echo "Stopping during ${phase}; resubmit to resume saved training epochs."
    if [[ -n "$active_step" ]]; then
        kill -TERM "$active_step" 2>/dev/null || true
        wait "$active_step" || true
    fi
    exit 143
}
trap stop_job TERM INT HUP
trap 'echo "Command error during ${phase} (exit $?). Check subsequent log output for job progress." >&2' ERR

# Slurm supplies the compute-node name; the optional hostname utility may not
# be installed. Avoid an ERR trap from a nonessential command substitution.
echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME:-${HOSTNAME:-unknown-node}}; started $(date -Is)"
echo "Repository: $SERE_REPO_DIR"
echo "Python: $SERE_PYTHON"
# Respect Slurm's CUDA_VISIBLE_DEVICES. cuda:0..2 are allocation-local IDs.
"$SERE_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
    raise SystemExit('This pipeline requires three Slurm-visible CUDA GPUs.')
print('Allocated GPUs:', [torch.cuda.get_device_name(i) for i in range(3)], flush=True)
PY

run_phase() {
    local reduction="$1"
    local action="$2"
    local -a device_args
    phase="${reduction}/${action}"
    if [[ "$action" == select || "$action" == fit ]]; then
        device_args=(--devices cuda:0 cuda:1 cuda:2)
    else
        device_args=(--device cuda:0)
    fi
    echo "=== Starting ${phase} at $(date -Is) ==="
    srun --ntasks=1 --unbuffered --kill-on-bad-exit=1 \
        "$SERE_PYTHON" -u scripts/run_sere_lopo.py "$action" \
        --config "$SERE_CONFIG" --reduction "$reduction" "${device_args[@]}" &
    active_step=$!
    wait "$active_step"
    active_step=''
    echo "=== Finished ${phase} at $(date -Is) ==="
}

# prepare skips completed caches; in the current none run it builds L128.
# select/fit resume matching seed-654 runs. test reports EVERY active length.
for reduction in none mean; do
    for action in prepare select fit test; do
        run_phase "$reduction" "$action"
    done
done
echo "Both reduction workflows completed at $(date -Is)."
