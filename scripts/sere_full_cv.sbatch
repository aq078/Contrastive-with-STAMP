#!/usr/bin/env bash
#SBATCH --job-name=sere-full-cv
#SBATCH --partition=legacy
#SBATCH --qos=qos_legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --chdir=/zfsauton2/home/anqix/projects/stamp
#SBATCH --output=/zfsauton/scratch/anqix/stamp/slurm/sere-full-cv-%j.out
#SBATCH --error=/zfsauton/scratch/anqix/stamp/slurm/sere-full-cv-%j.err
#SBATCH --open-mode=append
set -Eeuo pipefail
SERE_CV_REPO_DIR="${SERE_CV_REPO_DIR:-/zfsauton2/home/anqix/projects/stamp}"
SERE_CV_PYTHON="${SERE_CV_PYTHON:-/zfsauton2/home/anqix/miniconda3/envs/stamp2/bin/python}"
SERE_CV_CONFIG="${SERE_CV_CONFIG:-configs/sere_full_patient_cv.json}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
cd "$SERE_CV_REPO_DIR"
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
trap 'echo "SERE full-trial CV pipeline stopped on error; later phases were not started." >&2' ERR
"$SERE_CV_PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
    raise SystemExit('This job requires three Slurm-visible GPUs.')
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(3)], flush=True)
PY
# Respect Slurm CUDA_VISIBLE_DEVICES. One worker per allocation-local GPU.
srun --ntasks=1 --unbuffered --kill-on-bad-exit=1 \
    "$SERE_CV_PYTHON" -u scripts/run_sere_full_cv.py all --config "$SERE_CV_CONFIG" \
    --reduction both --devices cuda:0 cuda:1 cuda:2 &
active_step=$!
wait "$active_step"
active_step=''
echo 'SERE full-trial CV baseline/SupCon, both reductions, all five test folds complete.'
