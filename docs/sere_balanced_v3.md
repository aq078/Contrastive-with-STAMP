# SERE class-balanced experiment (v3)

Use `configs/sere_temporal_balanced.json`, with its matching `.selection.json`.
The original config and all v2 experiment artifacts are preserved. The new
configuration writes to:

```
/zfsauton/scratch/anqix/stamp/experiments/sere_temporal_5fold_balanced_v3/
  without_reduction/
  with_reduction/
```

Both baseline and STAMP+SupCon use the same patient-grouped five folds, seed 654,
and lengths 16, 32, 64, 80, 96, 128. Every length is fitted and tested separately.
The temporal split, label rule, padding and model architecture are unchanged.

## Changes

Epoch selection and final test BA now use:

```
0.5 * (mean negative recall over patients with negative examples
       + mean positive recall over patients with positive examples)
```

A missing class does not contribute a recall for that patient. Both classes must
exist across the evaluated group; otherwise evaluation raises an explicit error.
Always-positive predictions score 0.5. The selection score uses all held-out
patients, while the sample SD uses the five fold scores at the selected common
epoch (ddof=1). Fold means can differ from the overall patient-based score.

Reports also include negative recall, positive recall, pooled BA, predicted
positive rate, confusion matrices (`[[TN, FP], [FN, TP]]`), and the always-positive
reference. Per-patient `ba` remains the historical present-class diagnostic;
it is **not** averaged to select epochs in v3.

Baseline and SupCon stage 2 use cross-entropy weights `N / (2 * N_class)`.
Counts come exclusively from the current training view: other patients'
development data in CV, all patients' development data in the final fit.
Weights and counts are saved in each training `request.json` and participate in
its checkpoint identity. Label smoothing remains 0.1. Stage 1 still uses the
original SupCon loss and ordinary batches; stage 2 trains only MHAP+classifier.
No threshold tuning or balanced SupCon sampling was added.

Frozen development embeddings are verified and read without modification from
the shared cache and v2 (including L128). Training checkpoints start fresh under
v3 and resume only matching v3 runs. Test preparation still waits until final
models are complete. No test data determine weights or epochs.

## Submit and resume

From the repository root, submit both reduction modes, prepare through test:

```bash
SERE_CONFIG=configs/sere_temporal_balanced.json \
  sbatch --job-name=sere-balanced-v3 scripts/sere_pipeline.sbatch
```

For the optional detached tmux submission and log monitor:

```bash
SERE_CONFIG=configs/sere_temporal_balanced.json \
SERE_TMUX_SESSION=sere-balanced-v3 \
  bash scripts/submit_sere_pipeline_tmux.sh --job-name=sere-balanced-v3
```

Resubmit the same command after a stopped/expired job to resume completed epochs.
Do not submit duplicate jobs while one is running. Slurm itself survives SSH
logout. An existing tmux monitor session is reused without submitting another
job; direct `sbatch` is the simplest way to resubmit a stopped job.

Inside an existing three-GPU allocation, run either mode directly:

```bash
python -u scripts/run_sere_lopo.py all \
  --config configs/sere_temporal_balanced.json \
  --reduction none --devices cuda:0 cuda:1 cuda:2

python -u scripts/run_sere_lopo.py all \
  --config configs/sere_temporal_balanced.json \
  --reduction mean --devices cuda:0 cuda:1 cuda:2
```

Replace `all` with `prepare`, `select`, `fit`, `test`, or `status` as needed.
The config argument is required: the default remains the historical config.
The source-code fingerprint has changed, so historical v2 execution correctly
refuses reuse under this code; its saved reports remain readable and intact.

## Saved-results diagnostic

`reports/sere_balanced_v3/previous_validation_rescored.json` recalculates old
unweighted CV curves under the new metric, including revised best epochs.
It does not represent newly trained weighted models, does not evaluate test,
and cannot be used as v3's completed selection artifact.

To create another independent report (the output must not already exist):

```bash
python scripts/rescore_sere_validation.py \
  --source /zfsauton/scratch/anqix/stamp/experiments/sere_temporal_5fold_v2 \
  --output reports/sere_balanced_v3/another_validation_rescore.json
```

The existing test set has already been inspected. Subsequent results on that
same test set are exploratory; a fresh untouched evaluation is needed for a
confirmatory performance claim.
