#!/usr/bin/env bash
#SBATCH --job-name=pennaction-video
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --chdir=/zfsauton2/home/anqix/projects/stamp
#SBATCH --output=/zfsauton/scratch/anqix/stamp/slurm/pennaction-%j.out
#SBATCH --error=/zfsauton/scratch/anqix/stamp/slurm/pennaction-%j.err
#SBATCH --open-mode=append
set -Eeuo pipefail
PENN_REPO_DIR="${PENN_REPO_DIR:-/zfsauton2/home/anqix/projects/stamp}"
PENN_PYTHON="${PENN_PYTHON:-/zfsauton2/home/anqix/miniconda3/envs/stamp2/bin/python}"
PENN_CONFIG="${PENN_CONFIG:-configs/pennaction_video.json}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
cd "$PENN_REPO_DIR"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit using sbatch.' >&2; exit 2; }
active_step=''
stop_job() {
    trap '' TERM INT HUP
    if [[ -n "$active_step" ]]; then
        kill -TERM "$active_step" 2>/dev/null || true
        wait "$active_step" || true
    fi
    echo 'Stopped; resubmit the same script to resume.'
    exit 143
}
trap stop_job TERM INT HUP
trap 'echo "PennAction pipeline stopped on error; later phases were not started." >&2' ERR
"$PENN_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
    raise SystemExit('This job requires three Slurm-visible GPUs.')
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(3)], flush=True)
PY
# Respect Slurm CUDA_VISIBLE_DEVICES. One worker per allocation-local GPU.
srun --ntasks=1 --unbuffered --kill-on-bad-exit=1 \
    "$PENN_PYTHON" -u scripts/run_pennaction.py all --config "$PENN_CONFIG" \
    --reduction both --devices cuda:0 cuda:1 cuda:2 &
active_step=$!
wait "$active_step"
active_step=''
echo 'PennAction baseline/SupCon, both reductions, all five seeds complete.'
