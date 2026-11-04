#!/usr/bin/env bash
#SBATCH --job-name=sere-small-ce-cv
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
#SBATCH --output=/zfsauton/scratch/anqix/stamp/slurm/sere-early-cv-%j.out
#SBATCH --error=/zfsauton/scratch/anqix/stamp/slurm/sere-early-cv-%j.err
#SBATCH --open-mode=append
set -Eeuo pipefail
SERE_CV_REPO_DIR="${SERE_CV_REPO_DIR:-/zfsauton2/home/anqix/projects/stamp}"
SERE_CV_PYTHON="${SERE_CV_PYTHON:-/zfsauton2/home/anqix/miniconda3/envs/stamp2/bin/python}"
SERE_CV_CONFIG="${SERE_CV_CONFIG:-configs/sere_full_patient_early.json}"
# Defaults: L8/16/32/48/64, stride 8, unweighted CE, seed 654.
# Five patient test folds with inner validation for early stopping.
# Runs baseline and SupCon, first without and then with MOMENT reduction.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
cd "$SERE_CV_REPO_DIR"
export PYTHONPATH="$SERE_CV_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
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
# Run diagnostics and the workflow in the SAME Slurm step and Python process.
# Keep Slurm's GPU visibility unchanged; cuda:0..2 are allocation-local IDs.
srun --ntasks=1 --unbuffered --kill-on-bad-exit=1 \
    "$SERE_CV_PYTHON" -u - "$SERE_CV_CONFIG" <<'PY' &
import json
import os
import platform
import runpy
import shutil
import subprocess
import sys
import torch
config_path = sys.argv[1]
print('Node:', platform.node(), 'job:', os.environ.get('SLURM_JOB_ID'),
      'step:', os.environ.get('SLURM_STEP_ID'), flush=True)
print('Python:', sys.executable, 'PyTorch:', torch.__version__,
      'PyTorch CUDA:', torch.version.cuda, flush=True)
for name in ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS',
             'LD_LIBRARY_PATH', 'CUDA_HOME'):
    print(f'{name}={os.environ.get(name, "<unset>")}', flush=True)
nvidia_smi = shutil.which('nvidia-smi')
if nvidia_smi:
    try:
        status = subprocess.run([nvidia_smi], timeout=20, check=False)
        print('nvidia-smi exit code:', status.returncode, flush=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print('nvidia-smi diagnostic failed:', exc, flush=True)
else:
    print('nvidia-smi not found on PATH', flush=True)
with open(config_path) as stream:
    config = json.load(stream)
print('Config:', sys.argv[1], flush=True)
print('Windows:', config['window_lengths'], 'stride:', config['stride'],
      'seeds:', config['seeds'], flush=True)
print('CE weighting:', {name: config['recipes'][name].get('class_weighting', 'none')
                        for name in ('baseline', 'supcon_stage2')}, flush=True)
print('Output:', config['output_dir'], flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
    raise SystemExit('CUDA preflight failed inside srun; inspect node/GPU diagnostics above. '
                     'This job requires three usable Slurm-visible GPUs.')
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(3)], flush=True)
for i in range(3):
    probe = torch.ones(1, device=f'cuda:{i}') + 1
    torch.cuda.synchronize(i)
    del probe
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
print('CUDA computation passed on all three GPUs. Starting workflow.', flush=True)
sys.argv = ['scripts/run_sere_early_cv.py', 'all', '--config', config_path,
            '--reduction', 'both', '--devices', 'cuda:0', 'cuda:1', 'cuda:2']
runpy.run_path(sys.argv[0], run_name='__main__')
PY
active_step=$!
wait "$active_step"
active_step=''
echo 'SERE full-trial CV baseline/SupCon, both reductions, all five test folds complete.'
