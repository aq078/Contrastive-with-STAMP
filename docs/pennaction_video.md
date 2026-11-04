# PennAction: one video per instance

This isolated workflow leaves the SERE implementation, running-job fingerprints,
old PennAction LMDBs and old experiment directories unchanged.

## What inspection found

Raw data: `dataset/Penn_Action/labels/*.mat`, with 13 two-dimensional joints and
15 actions. Raw annotations contain only official train/test flags. The existing
LMDB's `__sequence_split__` supplies the fixed validation assignment. The selected
split source is `dataset/processed_pennaction/pennaction_xy_bboxnorm_L32_S16.lmdb`;
despite its filename, its metadata actually describes 64-frame windows. Only its
split/class metadata is used. Old window samples and embeddings are not reused.

There are 1,068 train, 189 validation and 1,068 test videos. The existing exclusion
of video `0038` is preserved. The importer checks window-key split consistency,
raw-video coverage, class mapping and official train/test flags. It never makes
a new split for a different seed. Historical `experiments/Penn_Action` folders
in this checkout have no saved experiment configs or results.

## Data flow and training

1. Import and fingerprint the supplied video split and raw annotations.
2. Normalize each complete video using bbox center and maximum bbox side.
   Repair malformed boxes from joint coordinates and interpolate invisible joints
   within that video; entirely unobserved joint coordinates use the bbox center.
   There is no extra division by 100, windowing, truncation or temporal resampling.
3. Extract frozen MOMENT patch embeddings for train and validation only. The last
   partial patch is retained and masked. Cache one tensor file per video, allowing
   prepare to resume after each completed video.
4. Without reduction, retain all patch embeddings as `P x 26 x 1024`. The loader
   pads to `84 x 26 x 1024` and supplies the valid-token mask. With reduction,
   average valid patch embeddings per coordinate and reshape to `2 x 13 x 1024`.
   The 672-frame padding capacity accommodates the longest source video (663).
   This is a fixed storage/model capacity, not a candidate window length. A video
   longer than capacity raises an error instead of being truncated.
5. Train baseline and SupCon for seeds **654, 114, 25, 759, 281**, using only the
   provided training videos. Each seed selects its own earliest best validation
   checkpoint: maximum multiclass balanced accuracy for baseline/stage 2, minimum
   validation SupCon loss for stage 1. SupCon validation uses a fixed shuffled
   batching order and fixed augmentation RNG each epoch, avoiding batches grouped
   by action/video ID. Stage 2 loads that seed's best stage 1 and
   trains **MHAP + classifier only**, with the frozen backbone in evaluation mode.
6. Once both methods have completed all five seeds for a reduction, prepare and
   evaluate test videos using their validation-selected models. No train+val
   refit is performed; the supplied validation split stays separate.
7. Report each seed's validation/test BA and accuracy, then mean and sample SD
   across the five seeds (ddof=1), for all four method/reduction combinations.
   BA is mean recall over the 15 actions. Detailed test reports contain per-class
   recall, support, prediction counts, confusion matrices and one prediction per
   video. Test metrics never select checkpoints.

The initial config uses the current STAMP architecture and recipe budgets from
SERE (baseline 65 epochs, SupCon stage 1 80, stage 2 30), because no PennAction
configs were recoverable here. All training/validation batch sizes are 16 for
whole-video memory use. The classifiers use ordinary cross-entropy with label
smoothing 0.1; SERE's binary class weighting and patient aggregation are not
implicitly transferred to this 15-class dataset. Change hyperparameters before
starting a run, or choose a new output directory afterwards.

## Run

From the repository, first validate source data without GPU work or output writes:

```bash
cd /zfsauton2/home/anqix/projects/stamp
/zfsauton2/home/anqix/miniconda3/envs/stamp2/bin/python scripts/run_pennaction.py audit
```

Submit the complete three-GPU workflow (both reductions, both methods, five seeds):

```bash
cd /zfsauton2/home/anqix/projects/stamp
mkdir -p /zfsauton/scratch/anqix/stamp/slurm
sbatch scripts/pennaction_pipeline.sbatch
```

Slurm continues after SSH logout; tmux is optional. Resubmit the same command
after an interrupted or time-limited job. Checkpoints atomically save each full
epoch, including model, best-so-far model, optimizer, scheduler and RNG states.
Completed seed/stage runs are skipped. Never submit a duplicate concurrent run
for the same output directory. The orchestrator and individual runs are locked.

For an existing GPU allocation, run individual phases/modes:

```bash
python -u scripts/run_pennaction.py prepare --reduction both --devices cuda:0
python -u scripts/run_pennaction.py train --reduction none --devices cuda:0 cuda:1 cuda:2
python -u scripts/run_pennaction.py train --reduction mean --devices cuda:0 cuda:1 cuda:2
python -u scripts/run_pennaction.py test --reduction both --devices cuda:0
```

Or run `all --reduction both --devices cuda:0 cuda:1 cuda:2`. Resume can use a
single GPU (`--devices cuda:0`) or a different allocation. The five independent
seed fits are distributed across the available GPUs; one fit uses one GPU.

Config: `configs/pennaction_video.json`. Output:

```
/zfsauton/scratch/anqix/stamp/experiments/pennaction_video_v1/
  without_reduction/
  with_reduction/
```

Each reduction directory contains `manifest.json`, `protocol.json`, `embeddings/`,
`runs/{baseline,supcon,head}/seed_*/{resume.pt,best.pt,run.json}`, detailed `test/`
predictions, and `evaluation_results.{json,csv}` / `per_seed_results.csv`.
Changing data, code, model weights or hyperparameters requires a new output
root; the workflow refuses to mix or overwrite incompatible runs.
