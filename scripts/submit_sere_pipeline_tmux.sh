#!/usr/bin/env bash
# Optional sbatch overrides may be passed as arguments, e.g. --partition=general.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
export SERE_REPO_DIR="${SERE_REPO_DIR:-$repo_dir}"
log_dir="${SERE_LOG_DIR:-/zfsauton/scratch/anqix/stamp/slurm}"
session="${SERE_TMUX_SESSION:-sere-pipeline}"

if [[ "${1:-}" == --monitor ]]; then
    shift
    mkdir -p "$log_dir"
    # sbatch returns immediately; Slurm owns the job even if tmux/login host exits.
    if ! submitted="$(sbatch --parsable "$@" --chdir="$SERE_REPO_DIR" \
        --output="$log_dir/sere-%j.out" --error="$log_dir/sere-%j.err" \
        "$script_dir/sere_pipeline.sbatch")"; then
        echo 'Submission failed; inspect the Slurm error above.' >&2
        exit 1
    fi
    job_id="${submitted%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch output: $submitted" >&2; exit 1; }
    printf '%s\n' "$job_id" > "$log_dir/latest_job_id"
    printf '%s\n' "Submitted Slurm job $job_id" \
        "Status: squeue -j $job_id" "Cancel: scancel $job_id" \
        "Logs: $log_dir/sere-$job_id.{out,err}" \
        'Detach from tmux with Ctrl+B, then D. Ctrl+C here stops log viewing, not the job.'
    touch "$log_dir/sere-$job_id.out" "$log_dir/sere-$job_id.err"
    exec tail -n 40 -F "$log_dir/sere-$job_id.out" "$log_dir/sere-$job_id.err"
fi

command -v tmux >/dev/null
command -v sbatch >/dev/null
if tmux has-session -t "=$session" 2>/dev/null; then
    echo "Session $session already exists; no duplicate job submitted."
    echo "Attach: tmux attach -t $session"
    exit 0
fi
mkdir -p "$log_dir"
# %q shell-quotes every argument, including paths or overrides containing spaces.
printf -v monitor_command '%q ' env "SERE_REPO_DIR=$SERE_REPO_DIR" "SERE_LOG_DIR=$log_dir" \
    "SERE_PYTHON=${SERE_PYTHON:-/zfsauton2/home/anqix/miniconda3/envs/stamp2/bin/python}" \
    "SERE_CONFIG=${SERE_CONFIG:-configs/sere_temporal_lopo.json}" \
    bash "$script_dir/submit_sere_pipeline_tmux.sh" --monitor "$@"
tmux new-session -d -s "$session" -c "$SERE_REPO_DIR"
# Keep submission failures visible instead of immediately destroying the session.
tmux set-option -w -t "=$session:0" remain-on-exit on >/dev/null
tmux respawn-pane -k -t "=$session:0.0" "$monitor_command"
echo "Submission and log monitor started in tmux session $session."
echo "Attach: tmux attach -t $session"
