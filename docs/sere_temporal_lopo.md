# SERE temporal 80/20 evaluation with five patient-grouped folds

The current workflow uses **five patient-grouped folds**, not LOPO. The script
and config filenames retain `lopo` for command compatibility. It evaluates
baseline STAMP and STAMP+SupCon separately for each MOMENT representation:

- `--reduction none`: P temporal patches × 99 joint-coordinate series.
- `--reduction mean`: 3 coordinates × 33 joints, after averaging valid patches.

All four tracks evaluate **all six window lengths**, using the same temporal
splits, patient groups, and **seed 654 throughout**. Validation selects training
duration separately for each method/length; it does not choose a winning length.

## Current evaluation: 16, 32, 64, 80, 96, and 128

`configs/sere_temporal_lopo.selection.json` sets the active lengths to
**16, 32, 64, 80, 96, and 128**, for both baseline and SupCon in both reduction modes.
Use the same prepare/select/fit/test and resume commands below. Recovery and
preparation process these six lengths. L16 has been reinstated and L96 added
before final fitting/testing. L128 is now included as an explicitly approved
new length. Compatible saved seed-654 runs are reused. Run `prepare` before
resuming `select` to build L128 embeddings; completed caches are skipped. Legacy
cache recovery deliberately leaves new lengths such as L128 to ordinary prepare.
An already running process retains its original candidate list. Stop it in its
terminal and rerun the same command to adopt the restriction. Training resumes
from the last completed epoch; recovery skips completed embedding lengths.

The base config retains its original five lengths and five-seed inventory to
preserve existing protocol and checkpoint identities. Do not edit those original
lists for this ongoing run. The sidecar sets `selection_seeds: [654]`, applied to
CV, final fitting, and testing. Other seeds remain saved but are not scheduled
or included in the new comparisons.
After normal source/model/config/code verification, the command applies the
separate restriction and records `selection_policy.json` in each reduction's
output directory, along with `selection_seed_policy.json`. The final
`validation_results.json` contains all six lengths for both methods.
Individual fits, folds, schedules, and resume signatures are
unchanged, so saved L32 work resumes normally. No training implementation or
saved checkpoint is rewritten. This is an explicit amendment to the candidate
search, made before final fitting or test evaluation.

Existing baseline and SupCon stage-1 curves for seed 654 are reused. A changed
stage-1 duration requires replay and stage-2 training from that new checkpoint.
Those runs use separate `<stage>_seed654_stage1epoch<N>` directories if an older
run conflicts. For example, L64 stage 1 changes from epoch 7 (five-seed selection)
to epoch 9 (seed-654 selection), so the old epoch-7 stage-2 scores are not reused.
No existing checkpoint or five-seed validation summary is overwritten.

To save a progress report without training or touching test data:

```bash
python -u scripts/run_sere_lopo.py status --reduction none --device cpu
```

This writes `selection_progress_seed654.json` and per-length
`validation_summary_seed654.json`. `select` completes the missing single-seed
CV jobs and writes `validation_results.json`; `fit` retrains **both methods at
all six lengths**, with one seed; `test` evaluates all twelve final models after
all their final fits are complete. No test-based choice of window is made.

The sidecar name follows the config: `NAME.json` uses `NAME.selection.json`.
Once recorded, the policy is also loaded from scratch if the sidecar is absent.
Explicit expansion (`allow_window_expansion: true`) is supported before final
fitting/test access. Previous policies and completed validation summaries are
archived under `policy_history/`; expansion retains the earlier active summary
until the full expanded set completes. `fit` refuses an incomplete expanded
plan. New lengths outside the immutable base inventory additionally require
`allow_new_window_lengths: true` (enabled for L128). Other conflicting policies
are rejected. For a fresh dataset/experiment,
set the desired lengths directly in its new base config and use a new output root.

## Scratch storage and recovery of the deleted cache

The default config now stores large files under `/zfsauton/scratch/anqix/stamp/`:

- `dataset/SERE_dataset_SHAREABLE_skeletons/`: the SERE source dataset.
- `moment/`: local MOMENT weights.
- `experiments/sere_temporal_5fold_v2/`: current checkpoints, results, and new caches.
- `embeddings/sere_development/`: recovered development caches used by five-fold training.

The repository's `dataset/SERE_dataset_SHAREABLE_skeletons`, `moment`, and `experiments/sere_temporal_5fold_v2`
paths are compatibility symlinks to scratch. Run commands from the repository
as before. Future preparation, selection, fitting, and test outputs use scratch.
The same scratch mount must be accessible when resuming on another machine.

The legacy directory's name does **not** mean its data was unused: only its
training runs were superseded. Five-fold training still reused its development
embeddings. After that directory was deleted, the saved five-fold requests
retained enough preprocessing metadata to rebuild compatible caches. Before
resuming the current without-reduction run, execute:

```bash
python -u scripts/recover_sere_embeddings.py --reduction none --device cuda:0
python -u scripts/run_sere_lopo.py select --reduction none --devices cuda:0 cuda:1 cuda:2
```

Recovery uses one GPU. It verifies the unchanged scientific config, source files,
MOMENT weights, and model/preprocessing code, rebuilds only development data, and
compares fold sample-key hashes against saved training requests. It preserves
the historical cache metadata those requests require; ordinary `prepare` would
instead create a new cache identity that existing requests reject. It does not
modify training checkpoints or validation curves. `recovery.json` records the
reconstruction explicitly: this is regeneration, not a byte-for-byte backup
restore, and different hardware/library versions may change floating-point
embeddings slightly. Use the original environment and GPU type when possible.

Rerun the recovery command after interruption: complete lengths are skipped;
an incomplete staged length is rebuilt. `--raw-only` validates/rebuilds raw
samples without loading MOMENT. This option does **not** produce embeddings.
The with-reduction experiment has no saved training progress and uses ordinary
`prepare --reduction mean` as shown below.

## Run with three GPUs

### Submit the complete sequence through Slurm and monitor with tmux

From your normal SSH/login shell:

```bash
cd /zfsauton2/home/anqix/projects/stamp
bash scripts/submit_sere_pipeline_tmux.sh
tmux attach -t sere-pipeline
```

The batch job requests one node, three GPUs, 12 CPU cores, 64 GiB memory, and
48 hours on `general`. This partition/time limit was checked with `sinfo`.
The scripts passed shell syntax and mocked orchestration tests. Slurm's
`sbatch --test-only` returned `Access/permission denied` from the Codex session,
so actual submission permissions must be checked from your normal login shell.
Nothing was submitted during setup.

Execution order is exactly:

```text
none: prepare (including L128) -> select -> fit -> test
mean: prepare -> select -> fit -> test
```

All six active lengths and both methods use seed 654. Existing compatible caches
and training results are reused. Each phase must succeed before the next starts.
Prepare/test use one GPU; select/fit distribute independent jobs across the three
allocated GPUs. No physical GPU IDs are forced; CUDA respects Slurm's allocation.
Stop any previous interactive workflow using the same output directories before
submitting, or its run lock will reject the batch process.

**Slurm keeps the computation running after SSH disconnects.** tmux hosts the
submission command and log monitor, not a detached process inside the allocation.
Detach with Ctrl+B then D. Reconnect to the same login host and run
`tmux attach -t sere-pipeline`. If that tmux server is lost, the Slurm job still
continues and its scratch logs can be viewed from another login session.

Logs: `/zfsauton/scratch/anqix/stamp/slurm/sere-<JOBID>.out` and `.err`.
The launcher writes `latest_job_id` in that directory. Use `squeue -u "$USER"`
to inspect your jobs and `scancel JOBID` to cancel one. Ctrl+C in the log viewer
only stops viewing; it does not cancel the Slurm job.

Site-specific sbatch overrides can be passed to the launcher, for example:

```bash
bash scripts/submit_sere_pipeline_tmux.sh --account=YOUR_ACCOUNT --partition=general
```

After an interrupted/failed job has exited, resubmit with a new monitor name:

```bash
SERE_TMUX_SESSION=sere-resume bash scripts/submit_sere_pipeline_tmux.sh
```

Completed training runs are skipped and incomplete training resumes at its last
complete epoch. Newly generated embedding LMDBs do not support mid-cache resume;
an interrupted ordinary `prepare` may need its incomplete cache handled before
resubmission. The batch stops and leaves the error in `.err` rather than deleting
or overwriting a cache. GPU reservation ends at 48 hours even if tmux remains open.

Optional environment overrides: `SERE_REPO_DIR`, `SERE_PYTHON`, `SERE_CONFIG`,
`SERE_LOG_DIR`, and `SERE_TMUX_SESSION`. The default Python is the existing stamp2
environment's executable, so manual conda activation is unnecessary.

### Run individual phases interactively

Run from the repository root in the existing STAMP environment. Confirm that
`cuda:0`, `cuda:1`, and `cuda:2` are visible in your allocated environment.

```bash
conda activate stamp2

# Without reduction
python -u scripts/run_sere_lopo.py prepare --reduction none --device cuda:0
python -u scripts/run_sere_lopo.py select --reduction none --devices cuda:0 cuda:1 cuda:2
python -u scripts/run_sere_lopo.py fit --reduction none --devices cuda:0 cuda:1 cuda:2

# With reduction
python -u scripts/run_sere_lopo.py prepare --reduction mean --device cuda:0
python -u scripts/run_sere_lopo.py select --reduction mean --devices cuda:0 cuda:1 cuda:2
python -u scripts/run_sere_lopo.py fit --reduction mean --devices cuda:0 cuda:1 cuda:2
```

After both modes finish selection and fitting successfully:

```bash
python -u scripts/run_sere_lopo.py test --reduction none --device cuda:0
python -u scripts/run_sere_lopo.py test --reduction mean --device cuda:0
```

Selection and fit distribute independent fold/seed jobs across the specified
GPUs, one worker per GPU. They do not use data-parallel training within a model.
Prepare and test use one GPU. With `--devices`, they use the first listed device.
Do not run both reduction modes simultaneously using the same three GPUs.

`all` performs prepare, select, fit, and test for one representation. Each `all`
command includes test evaluation; when comparing all four tracks, the separate
commands above let you finish all selection before viewing any test results.

`--config PATH` selects a different config. `--device cuda:0` uses a single GPU;
`--device cpu` is available for small checks.

## Resume after SSH loss, interruption, or moving machines

**Run the same command again. No resume flag is needed.** For example:

```bash
cd /zfsauton2/home/anqix/projects/stamp
conda activate stamp2
python -u scripts/run_sere_lopo.py select --reduction none --devices cuda:0 cuda:1 cuda:2
```

- Completed runs are verified and skipped.
- Each active baseline, SupCon stage-1, stage-1 replay, or stage-2 run saves an
  atomic `resume.pt` after every complete epoch, including validation.
- The checkpoint includes weights, optimizer state, OneCycle scheduler state,
  completed epoch, metric curves, Python/NumPy/PyTorch random states, the active
  GPU's random state, and the training-loader shuffle generator state.
- A partial epoch is repeated. At most the current unsaved epoch is lost per
  worker. Final fitting has the same resume behavior.
- An incomplete checkpoint write never replaces the previous complete one.
- Changing the GPU IDs or worker count is allowed. For example, resume with
  `--device cuda:0` if the new machine has only one GPU.
- Checkpoint paths are relative, so the project can move to another mount path.
  Update physical path fields in the config if needed. Source/model contents,
  scientific settings, folds, and implementation must remain the same.
- Cross-machine continuation requires access to the output directory, raw
  source files, local MOMENT weights, and the embedding caches. If storage is
  not shared, copy them before resuming. When reusing old caches, include the
  `prepared_data_dir` tree as well.
- Different GPU models or library versions may change floating-point results;
  restoring random states does not promise bitwise equality across hardware.

The CLI handles SIGHUP/SIGTERM by shutting down its worker pool. Run locks
prevent overlapping writers. If a job is still running under `tmux` or a job
scheduler, reconnect to it rather than starting a duplicate job.

For continued execution even when SSH disconnects:

```bash
tmux new -s sere
python -u scripts/run_sere_lopo.py select --reduction none --devices cuda:0 cuda:1 cuda:2 > sere_select_none.log 2>&1
```

Detach with Ctrl+B, then D. Reattach with `tmux attach -t sere`. Checkpoints also
allow restart if the process or machine stops, independently of tmux.

## Existing LOPO data and runs

The default output root is now `/zfsauton/scratch/anqix/stamp/experiments/sere_temporal_5fold_v2/`, containing
`without_reduction/` and `with_reduction/`. Old LOPO runs remain untouched.

The config's `prepared_data_dir` points to `/zfsauton/scratch/anqix/stamp/embeddings/sere_development/`.
Completed development embeddings there are reused **read-only**, after checking
source/manifest fingerprints, MOMENT weight fingerprints, stride, labels,
normalization, padding, reduction, and patient membership. Changing patient folds
alone does not require recomputing embeddings. Selection can directly use these
verified old caches; otherwise run `prepare` first. Missing or incomplete old
caches are rebuilt in the new output root, without modifying the old files.

Old LOPO training runs/checkpoints cannot resume as five-fold fits, since their
training/validation membership differs. Stop an old LOPO job in its original
terminal before starting the new search if it occupies the GPUs you need.

## Configuration and fold definition

Edit `configs/sere_temporal_lopo.json` before starting a new experiment:

- `cv.n_folds`: 5; `cv.seed`: 42.
- Original window-length inventory: 16, 32, 64, 80, 96; current eligible candidates:
  **16, 32, 64, 80, 96, 128**, through the selection sidecar; stride: 16.
- Active seed for CV, final training, and testing: **654**. Original saved
  inventory: 654, 114, 25, 759, 281; other seeds are retained only as historical results.
- Baseline epoch budget: **65**, matching the contents of the requested
  `baseline_val_acc_...ne50...` config rather than its directory name.
- SupCon stage 1: 80 epochs; stage 2: 30 epochs.
- Stage 2 trains **only MHAP and classifier**. The frozen backbone stays in
  evaluation mode to disable its dropout.

Patient IDs are sorted, shuffled deterministically using the fold seed, and
assigned to five balanced-size groups: 4, 4, 4, 3, 3 patients for SERE's 18
patients. Every patient is validated exactly once per seed/length, and all of a
patient's trials/exercises remain together. Group assignment uses no labels or
test statistics and is saved in `folds.json`. Groups match across lengths and
reduction modes. The restricted baseline search uses 6 lengths × 5 folds × 1 seed = 30 fits
per reduction mode.

Copied recipe settings are explicit in the config; `recipe_sources` documents
historical origins and is not reloaded. Scientific settings, source/model
contents, fold membership, and implementation code are fingerprinted. Changing
these requires a new output directory. Physical paths and GPU choices are not
part of the scientific fingerprint.

## Temporal split, missing data, and masks

For every original inclusive trial interval:

```text
cut = frame_init + floor(0.8 * original_frame_count)
development = [frame_init, cut)
test        = [cut, frame_end + 1)
```

The cut is fixed before filtering or windowing and is independent of window
length. Shared frames between overlapping trial intervals are excluded from all
such intervals, without moving their cuts. Missing/nonfinite skeleton frames
and excluded boundaries break runs; remaining frames are never concatenated
across gaps. E3–E5 missing `affected` fields map to SERE's `-1` sentinel.
Identical duplicate rows are deduplicated; conflicting duplicate frames fail.

Sliding windows are generated independently inside each contiguous run. A run
shorter than the candidate length produces one right-padded sample, using no
frames from another portion. Longer runs use stride-aligned full windows;
trailing incomplete windows are not added. Labels use any known compensation
frame. Padding is excluded from labeling; samples without known labels are
excluded. Coverage reports identify unrepresented trials. Missing patients fail
instead of silently disappearing from the aggregate.

Normalization is per-frame hip-centering and shoulder scaling, retaining the
legacy loader's `/100` scale. MOMENT RevIN uses only observed frames. Normalized
padding is zero; attention includes partially observed patches and excludes
fully padded patches. Without reduction, the patch-validity mask reaches
STAMP's normalization, criss-cross mixing, and pooling. With reduction, valid
patches are averaged per scalar series (each partial patch counts once), then
joint-major `(33,3)` data is transposed to `(3,33)`. All reduced tokens are valid.

## Patient-equal selection and final fit

Each grouped-fold model records validation results **separately for each held-out
patient**, including separate contrastive validation batches per patient.
At every epoch, average the seed-654 validation BA equally across all 18
patients. The selection mean is still patient-equal; folds have unequal patient
counts (4,4,4,3,3).

At the selected common epoch, compute each fold's mean patient BA, then report
**sample SD of those five fold scores (`ddof=1`)**. This is variability across
patient groups, not across seeds or a confidence interval. Reports include the
five `fold_scores`, `fold_mean_ba`, and `fold_std_ba`. The unweighted fold mean can
differ from the patient-equal validation BA; they are labeled separately.

BA is mean recall over classes present in a patient's labels. Single-class
patients contribute their class recall and are flagged. Class support and
confusion matrices appear under each epoch's `patients` record.

Baseline selects the epoch with highest patient-equal BA. SupCon first selects
the epoch with lowest patient-equal contrastive loss, then replays each fold's
stage 1 to that common epoch. Stage 2 starts from its matching fold/seed
checkpoint and selects its epoch by patient-equal BA. This happens independently
for each method, representation, and length. All lengths proceed to final fitting
and testing. Epoch ties choose the earliest epoch. All folds run complete curves; there is no per-fold early
stopping. Validation RNG is isolated from training to support exact replay.

Final models are fresh fits on all development samples at selected durations.
They have no validation or test loader. OneCycle retains the original CV epoch
horizon (65/80/30), recalculated in steps for the final training-set size, and
training stops at the selected epoch rather than compressing the cycle.

Only after both methods at **all six lengths** in a representation have
completed final fits can test samples/embeddings be generated. Test metrics include
patient-equal BA, per-patient support/confusion matrices,
per-window predictions/probabilities, and separate pooled-window BA. No seed or
hyperparameter is selected using test results. Final testing measures later
portions of known patients, not unseen-patient generalization.

`evaluation_results.csv` and `.json` pair validation BA, validation fold SD, and
test BA for each method/length. There is no test seed SD estimate with one seed.
The underlying legacy per-window test artifact may contain a zero `seed_std`
field for its single model; that is not an uncertainty estimate.

## Output layout

Inside each representation's root:

```text
protocol.json, manifest.json, folds.json
selection_policy.json, selection_seed_policy.json
[data/L<length>/development/{raw,embeddings,coverage.json}]
cv/L<length>/<stage>/fold_<fold>/seed_654/
    request.json, resume.pt, checkpoint.pt, run.json
cv/L<length>/validation_summary_seed654.json
selection_progress_seed654.json
validation_results.json
all_window_final_models.json
window_evaluation/L<length>/
    protocol.json, selection.json, final_models.json
    final/<method>/L<length>/<stage>/seed_654/
        request.json, resume.pt, checkpoint.pt, run.json
    data/L<length>/test/{raw,embeddings,coverage.json}
    test/<method>/seed_654.json
    test_results.json
evaluation_results.json, evaluation_results.csv
```

Development caches may live in `prepared_data_dir` instead of the new root.
Resume support applies to training and fitting. An interrupted newly generated
LMDB cache is still rejected as incomplete; the old completed caches remain
usable. Tests: `python -B -m unittest discover -s tests -v`.
