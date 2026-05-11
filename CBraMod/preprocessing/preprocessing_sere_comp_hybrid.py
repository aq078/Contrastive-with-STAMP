#!/usr/bin/env python3
"""
SERE preprocessing for hybrid compensation setup:

- TRAIN:
    positive samples = contiguous compensation events (with optional context),
                       resampled to fixed length EVENT_T
    negative samples = random fixed-length windows sampled from trials,
                       restricted to contain no positive frames

- VAL / TEST:
    samples = random fixed-length windows sampled from full trials,
              labels aggregated from frame labels inside the window

This keeps train positives "clean" while evaluating on realistic random windows.

Output LMDB:
    key = "{E}_pid{pid}_aff{affected}_trial{trial}_{sample_type}_{idx}"
    value = pickle({
        "sample": [33,3,L] float32,
        "label":  int 0/1,
        "mask":   [33*3, L] bool,
        "meta":   dict
    })

Edit CONFIG below. No CLI arguments.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lmdb
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# =========================
# CONFIG (EDIT HERE)
# =========================

ROOT_DIR = Path("dataset/SERE_dataset_SHAREABLE_skeletons")
OUT_LMDB = Path("dataset/processed_sere/sere_comp_hybrid_train_event_valtest_random_xyz.lmdb")

WORLD_DIR = ROOT_DIR / "MediaPipe_skeletons" / "WorldLandmarks"
FRAME_LABEL_DIR = ROOT_DIR / "Labels" / "frame_level" / "compensation"
VIDEO_LABEL_DIR = ROOT_DIR / "Labels" / "video_level" / "compensation"

EXERCISES = ["E1", "E2", "E3", "E4", "E5"]

WORLD_FILENAME_FMT = "{ex}_mp_world_landmarks.csv"
FRAME_LABEL_FILENAME_FMT = "{ex}_frame_labels.csv"
VIDEO_LABEL_FILENAME_FMT = "{ex}_labels_comp.csv"

# Columns
PID_COL = "pid"
AFFECTED_COL = "affected"
TRIAL_COL = "trial"
FRAME_COL = "frame"
FRAME_INIT_COL = "frame_init"
FRAME_END_COL = "frame_end"

FRAME_COMP_COL = "comp"

# TRAIN positive event settings
EVENT_T = 128
MIN_POS_EVENT_FRAMES = 5
MERGE_GAP_FRAMES = 3
POS_CONTEXT_BEFORE = 8
POS_CONTEXT_AFTER = 8

# TRAIN negative window settings
TRAIN_NEG_WINDOW_L = 64
TRAIN_NEG_WINDOWS_PER_TRIAL = 2
TRAIN_NEG_REQUIRE_ALL_ZERO = True

# VAL / TEST random window settings
EVAL_WINDOW_L = 64
EVAL_WINDOWS_PER_TRIAL = 8
EVAL_AGGREGATION = "any"   # "any" | "majority" | "mean"
MIN_VALID_LABELS_IN_WINDOW = 1

# General filtering
MIN_TRIAL_FRAMES = 64
SKIP_SEGMENTS_WITH_NANS = True
NORMALIZE = True

# Subject split
SEED = 42
SPLIT = {"train": 0.6, "val": 0.15, "test": 0.25}

LMDB_MAP_SIZE = 8_000_000_000

# =========================
# Constants
# =========================
POSE_N = 33
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


# =========================
# Helpers
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def split_pids_balanced_by_label(
    pid_to_counts: dict,
    split_spec: str,
    seed: int,
) -> dict[str, set]:
    parts = {}
    for tok in split_spec.strip().split():
        name, frac = tok.split(":")
        parts[name] = float(frac)
    if abs(sum(parts.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1. Got: {parts}")

    total0 = sum(pid_to_counts[pid].get(0, 0) for pid in pid_to_counts)
    total1 = sum(pid_to_counts[pid].get(1, 0) for pid in pid_to_counts)
    total = total0 + total1
    if total == 0:
        raise ValueError("No labels found to balance.")

    target_ratio1 = total1 / total
    target_segments = {k: int(round(parts[k] * total)) for k in parts}
    drift = total - sum(target_segments.values())
    if drift != 0:
        target_segments["train"] += drift

    rng = random.Random(seed)
    pids = list(pid_to_counts.keys())
    rng.shuffle(pids)
    pids.sort(key=lambda pid: pid_to_counts[pid].get(0, 0) + pid_to_counts[pid].get(1, 0), reverse=True)

    splits = {k: set() for k in parts}
    cur = {k: {"n0": 0, "n1": 0, "n": 0} for k in parts}

    def score(assign_split: str, pid) -> float:
        n0 = cur[assign_split]["n0"] + pid_to_counts[pid].get(0, 0)
        n1 = cur[assign_split]["n1"] + pid_to_counts[pid].get(1, 0)
        n = n0 + n1
        rate_penalty = 0.0 if n == 0 else abs((n1 / n) - target_ratio1)

        size_penalty = 0.0
        if cur[assign_split]["n"] + (pid_to_counts[pid].get(0, 0) + pid_to_counts[pid].get(1, 0)) > target_segments[assign_split]:
            size_penalty = 1.0
        return rate_penalty + 0.1 * size_penalty

    for pid in pids:
        best = min(parts.keys(), key=lambda k: score(k, pid))
        splits[best].add(pid)
        cur[best]["n0"] += pid_to_counts[pid].get(0, 0)
        cur[best]["n1"] += pid_to_counts[pid].get(1, 0)
        cur[best]["n"] += pid_to_counts[pid].get(0, 0) + pid_to_counts[pid].get(1, 0)

    return splits


def print_split_summary(pid_split: dict, pid_to_counts: dict) -> None:
    def summarize(split_name: str):
        pids = pid_split[split_name]
        n0 = sum(pid_to_counts[p].get(0, 0) for p in pids)
        n1 = sum(pid_to_counts[p].get(1, 0) for p in pids)
        tot = n0 + n1
        pos = (n1 / tot) if tot else float("nan")
        print(f"{split_name}: pids={len(pids)} samples={tot} pos_rate={pos:.4f} n0={n0} n1={n1}", flush=True)

    print("\n=== Split summary ===", flush=True)
    for s in ["train", "val", "test"]:
        summarize(s)


def normalize_xyz(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = X.astype(np.float32, copy=True)
    xyz = out[..., :3]

    root = 0.5 * (xyz[:, LEFT_HIP, :] + xyz[:, RIGHT_HIP, :])
    xyz = xyz - root[:, None, :]

    shoulder_vec = xyz[:, LEFT_SHOULDER, :] - xyz[:, RIGHT_SHOULDER, :]
    scale = np.linalg.norm(shoulder_vec, axis=-1)
    scale = np.maximum(scale, eps)
    xyz = xyz / scale[:, None, None]

    out[..., :3] = xyz
    return out


def segment_has_nans(Xs: np.ndarray) -> bool:
    return bool(np.isnan(Xs).any())


def resample_time(X: np.ndarray, T_new: int) -> np.ndarray:
    T_old = X.shape[0]
    if T_old == T_new:
        return X.astype(np.float32, copy=False)

    t_old = np.linspace(0.0, 1.0, T_old, endpoint=True)
    t_new = np.linspace(0.0, 1.0, T_new, endpoint=True)

    flat = X.reshape(T_old, -1)
    f = interp1d(
        t_old,
        flat,
        kind="linear",
        axis=0,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    flat_new = f(t_new).astype(np.float32)
    return flat_new.reshape(T_new, POSE_N, 3)


def aggregate_label(y_window: np.ndarray, agg: str) -> int:
    valid = ~np.isnan(y_window)
    if valid.sum() < MIN_VALID_LABELS_IN_WINDOW:
        return -1

    y = y_window[valid]
    agg = agg.lower().strip()
    if agg == "any":
        return int(np.any(y >= 0.5))
    if agg == "majority":
        return int(np.mean(y) >= 0.5)
    if agg == "mean":
        return int(np.round(np.mean(y)))
    raise ValueError(f"Unknown EVAL_AGGREGATION='{agg}'")


def contiguous_runs(frames: np.ndarray, labels: np.ndarray, target_value: int) -> List[Tuple[int, int]]:
    runs = []
    start = None
    prev_frame = None

    for f, y in zip(frames, labels):
        y_bin = int(y >= 0.5)
        if y_bin == target_value:
            if start is None:
                start = f
            elif prev_frame is not None and f != prev_frame + 1:
                runs.append((start, prev_frame))
                start = f
        else:
            if start is not None:
                runs.append((start, prev_frame))
                start = None
        prev_frame = f

    if start is not None:
        runs.append((start, prev_frame))
    return runs


def merge_close_runs(runs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not runs:
        return []

    runs = sorted(runs)
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        gap = s - pe - 1
        if gap <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def interval_length(iv: Tuple[int, int]) -> int:
    s, e = iv
    return e - s + 1


def event_segments_for_trial(
    frames: np.ndarray,
    y: np.ndarray,
    frame_init: int,
    frame_end: int,
) -> List[Tuple[int, int]]:
    pos_runs = contiguous_runs(frames, y, target_value=1)
    pos_runs = merge_close_runs(pos_runs, MERGE_GAP_FRAMES)
    pos_runs = [iv for iv in pos_runs if interval_length(iv) >= MIN_POS_EVENT_FRAMES]

    pos_segments = []
    for s, e in pos_runs:
        pos_segments.append(
            (max(frame_init, s - POS_CONTEXT_BEFORE), min(frame_end, e + POS_CONTEXT_AFTER))
        )
    return merge_close_runs(sorted(pos_segments), max_gap=0)


def sample_random_windows(
    frames: np.ndarray,
    y: np.ndarray,
    window_l: int,
    n_windows: int,
    rng: random.Random,
    require_all_zero: bool | None = None,
    max_tries: int = 200,
) -> List[Tuple[int, int, int]]:
    """
    Returns list of (start_idx, end_idx, label) in index space over frames/y.
    If require_all_zero=True, only keep windows whose valid labels are all 0.
    If require_all_zero=None, aggregate labels normally using EVAL_AGGREGATION.
    """
    out = []
    T = len(frames)
    if T < window_l:
        return out

    for _ in range(n_windows):
        accepted = False
        for _try in range(max_tries):
            start = rng.randint(0, T - window_l)
            end = start + window_l
            yw = y[start:end]

            if require_all_zero is True:
                valid = ~np.isnan(yw)
                if valid.sum() == 0:
                    continue
                if np.any(yw[valid] >= 0.5):
                    continue
                label = 0
            else:
                label = aggregate_label(yw, EVAL_AGGREGATION)
                if label < 0:
                    continue

            out.append((start, end, int(label)))
            accepted = True
            break

        if not accepted:
            continue

    return out


# =========================
# Loading WorldLandmarks (wide)
# =========================
def load_worldlandmarks_wide(path: Path) -> pd.DataFrame:
    df = lower_cols(pd.read_csv(path))

    for c in [PID_COL, AFFECTED_COL, FRAME_COL]:
        if c not in df.columns:
            raise ValueError(f"[world] Missing '{c}' in {path}. Found: {list(df.columns)}")

    for j in range(POSE_N):
        for suf in ["x", "y", "z"]:
            col = f"{j}{suf}"
            if col not in df.columns:
                raise ValueError(f"[world] Missing '{col}' in {path}")

    df[FRAME_COL] = pd.to_numeric(df[FRAME_COL], errors="coerce")
    df = df.dropna(subset=[FRAME_COL])
    df[FRAME_COL] = df[FRAME_COL].astype(int)
    return df


def build_xyz_for_frames(df_world: pd.DataFrame, pid: Any, affected: Any, frames: np.ndarray) -> np.ndarray:
    sub = df_world[(df_world[PID_COL] == pid) & (df_world[AFFECTED_COL] == affected)].copy()
    if sub.empty:
        return np.zeros((0, POSE_N, 3), dtype=np.float32)

    sub = sub.set_index(FRAME_COL, drop=False)
    frames = np.asarray(frames, dtype=int)

    rows = sub.reindex(index=frames, copy=False)
    T = len(rows)

    X = np.zeros((T, POSE_N, 3), dtype=np.float32)
    for j in range(POSE_N):
        X[:, j, 0] = rows[f"{j}x"].to_numpy(dtype=np.float32)
        X[:, j, 1] = rows[f"{j}y"].to_numpy(dtype=np.float32)
        X[:, j, 2] = rows[f"{j}z"].to_numpy(dtype=np.float32)
    return X


# =========================
# Loading Labels
# =========================
def load_frame_labels(path: Path) -> pd.DataFrame:
    df = lower_cols(pd.read_csv(path))
    needed = [PID_COL, AFFECTED_COL, FRAME_COL, FRAME_COMP_COL]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"[frame labels] Missing '{c}' in {path}. Found: {list(df.columns)}")

    df[FRAME_COL] = pd.to_numeric(df[FRAME_COL], errors="coerce")
    df[FRAME_COMP_COL] = pd.to_numeric(df[FRAME_COMP_COL], errors="coerce")
    df = df.dropna(subset=[FRAME_COL])
    df[FRAME_COL] = df[FRAME_COL].astype(int)
    return df


def load_video_labels(path: Path) -> pd.DataFrame:
    df = lower_cols(pd.read_csv(path))

    required = [PID_COL, TRIAL_COL, FRAME_INIT_COL, FRAME_END_COL]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"[video labels] Missing '{c}' in {path}. Found: {list(df.columns)}")

    if AFFECTED_COL not in df.columns:
        df[AFFECTED_COL] = np.nan

    df[TRIAL_COL] = pd.to_numeric(df[TRIAL_COL], errors="coerce")
    df[FRAME_INIT_COL] = pd.to_numeric(df[FRAME_INIT_COL], errors="coerce")
    df[FRAME_END_COL] = pd.to_numeric(df[FRAME_END_COL], errors="coerce")

    df = df.dropna(subset=[TRIAL_COL, FRAME_INIT_COL, FRAME_END_COL])
    df[TRIAL_COL] = df[TRIAL_COL].astype(int)
    df[FRAME_INIT_COL] = df[FRAME_INIT_COL].astype(int)
    df[FRAME_END_COL] = df[FRAME_END_COL].astype(int)

    df = df[df[FRAME_END_COL] >= df[FRAME_INIT_COL]].copy()
    return df


def compute_pid_sample_counts() -> dict:
    """
    Dry-run pass: compute per-pid counts under this hybrid protocol.
    """
    pid_to_counts = {}
    rng = random.Random(SEED)

    for ex in EXERCISES:
        flf = FRAME_LABEL_DIR / FRAME_LABEL_FILENAME_FMT.format(ex=ex)
        vlf = VIDEO_LABEL_DIR / VIDEO_LABEL_FILENAME_FMT.format(ex=ex)

        if not flf.exists():
            raise FileNotFoundError(f"Missing frame label file: {flf}")
        if not vlf.exists():
            raise FileNotFoundError(f"Missing video label file: {vlf}")

        print(f"\n[dry-run] === {ex} ===", flush=True)

        fdf = load_frame_labels(flf)
        vdf = load_video_labels(vlf)

        f_by = {}
        for (pid, affected), g in fdf.groupby([PID_COL, AFFECTED_COL], sort=False):
            f_by[(pid, affected)] = g.sort_values(FRAME_COL).reset_index(drop=True)

        vdf = vdf.sort_values([PID_COL, AFFECTED_COL, TRIAL_COL]).reset_index(drop=True)

        for row in vdf.itertuples(index=False):
            pid = getattr(row, PID_COL)
            affected = getattr(row, AFFECTED_COL)
            frame_init = int(getattr(row, FRAME_INIT_COL))
            frame_end = int(getattr(row, FRAME_END_COL))

            if (pid, affected) not in f_by:
                continue

            g = f_by[(pid, affected)]
            sub = g[(g[FRAME_COL] >= frame_init) & (g[FRAME_COL] <= frame_end)].copy()
            if len(sub) < MIN_TRIAL_FRAMES:
                continue

            frames = sub[FRAME_COL].to_numpy(dtype=int)
            y = sub[FRAME_COMP_COL].to_numpy(dtype=np.float32)

            if pid not in pid_to_counts:
                pid_to_counts[pid] = {0: 0, 1: 0}

            # train protocol counts
            pos_segments = event_segments_for_trial(frames, y, frame_init, frame_end)
            pid_to_counts[pid][1] += len(pos_segments)

            neg_windows = sample_random_windows(
                frames=frames,
                y=y,
                window_l=TRAIN_NEG_WINDOW_L,
                n_windows=TRAIN_NEG_WINDOWS_PER_TRIAL,
                rng=rng,
                require_all_zero=TRAIN_NEG_REQUIRE_ALL_ZERO,
            )
            pid_to_counts[pid][0] += len(neg_windows)

            # eval protocol counts too, so split balancing sees mixed sample population
            eval_windows = sample_random_windows(
                frames=frames,
                y=y,
                window_l=EVAL_WINDOW_L,
                n_windows=EVAL_WINDOWS_PER_TRIAL,
                rng=rng,
                require_all_zero=None,
            )
            for _, _, label in eval_windows:
                pid_to_counts[pid][int(label)] += 1

    return pid_to_counts


def main() -> None:
    global PID_COL, AFFECTED_COL, TRIAL_COL, FRAME_COL, FRAME_INIT_COL, FRAME_END_COL, FRAME_COMP_COL
    PID_COL = PID_COL.lower()
    AFFECTED_COL = AFFECTED_COL.lower()
    TRIAL_COL = TRIAL_COL.lower()
    FRAME_COL = FRAME_COL.lower()
    FRAME_INIT_COL = FRAME_INIT_COL.lower()
    FRAME_END_COL = FRAME_END_COL.lower()
    FRAME_COMP_COL = FRAME_COMP_COL.lower()

    set_seed(SEED)
    rng = random.Random(SEED)

    # -------------------------
    # PASS 1: compute pid->(n0,n1)
    # -------------------------
    pid_to_counts = compute_pid_sample_counts()
    all_pids = sorted(pid_to_counts.keys())

    if not all_pids:
        raise RuntimeError("No pids found in dry-run; check your input paths / filtering settings.")

    split_spec = " ".join(f"{k}:{v}" for k, v in SPLIT.items())
    pid_split = split_pids_balanced_by_label(
        pid_to_counts=pid_to_counts,
        split_spec=split_spec,
        seed=SEED,
    )

    print_split_summary(pid_split, pid_to_counts)

    # -------------------------
    # PASS 2: write LMDB
    # -------------------------
    OUT_LMDB.parent.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(str(OUT_LMDB), map_size=int(LMDB_MAP_SIZE))

    dataset_keys: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    total_samples = 0
    kept_samples = 0
    skipped_trials = 0

    with db.begin(write=True) as txn:
        for ex in EXERCISES:
            wf = WORLD_DIR / WORLD_FILENAME_FMT.format(ex=ex)
            flf = FRAME_LABEL_DIR / FRAME_LABEL_FILENAME_FMT.format(ex=ex)
            vlf = VIDEO_LABEL_DIR / VIDEO_LABEL_FILENAME_FMT.format(ex=ex)

            print(f"\n=== {ex} ===", flush=True)
            print(f"World:       {wf}", flush=True)
            print(f"Frame labels:{flf}", flush=True)
            print(f"Video labels:{vlf}", flush=True)

            wdf = load_worldlandmarks_wide(wf)
            fdf = load_frame_labels(flf)
            vdf = load_video_labels(vlf)

            f_by = {}
            for (pid, affected), g in fdf.groupby([PID_COL, AFFECTED_COL], sort=False):
                f_by[(pid, affected)] = g.sort_values(FRAME_COL).reset_index(drop=True)

            vdf = vdf.sort_values([PID_COL, AFFECTED_COL, TRIAL_COL]).reset_index(drop=True)

            for row in vdf.itertuples(index=False):
                pid = getattr(row, PID_COL)
                affected = getattr(row, AFFECTED_COL)
                trial = int(getattr(row, TRIAL_COL))
                frame_init = int(getattr(row, FRAME_INIT_COL))
                frame_end = int(getattr(row, FRAME_END_COL))

                if (pid, affected) not in f_by:
                    skipped_trials += 1
                    continue

                g = f_by[(pid, affected)]
                sub = g[(g[FRAME_COL] >= frame_init) & (g[FRAME_COL] <= frame_end)].copy()
                if len(sub) < MIN_TRIAL_FRAMES:
                    skipped_trials += 1
                    continue

                frames = sub[FRAME_COL].to_numpy(dtype=int)
                y = sub[FRAME_COMP_COL].to_numpy(dtype=np.float32)

                if pid in pid_split["train"]:
                    split = "train"
                elif pid in pid_split["val"]:
                    split = "val"
                else:
                    split = "test"

                samples_to_write = []

                if split == "train":
                    # positives: event-centered segments
                    pos_segments = event_segments_for_trial(frames, y, frame_init, frame_end)
                    for idx, (s, e) in enumerate(pos_segments):
                        seg_frames = np.arange(s, e + 1, dtype=int)
                        samples_to_write.append({
                            "sample_type": "train_pos",
                            "idx": idx,
                            "label": 1,
                            "start": s,
                            "end": e,
                            "frame_ids": seg_frames,
                            "target_len": EVENT_T,
                        })

                    # negatives: random all-zero windows
                    neg_windows = sample_random_windows(
                        frames=frames,
                        y=y,
                        window_l=TRAIN_NEG_WINDOW_L,
                        n_windows=TRAIN_NEG_WINDOWS_PER_TRIAL,
                        rng=rng,
                        require_all_zero=TRAIN_NEG_REQUIRE_ALL_ZERO,
                    )
                    for idx, (start_i, end_i, label) in enumerate(neg_windows):
                        seg_frames = frames[start_i:end_i]
                        samples_to_write.append({
                            "sample_type": "train_neg",
                            "idx": idx,
                            "label": label,
                            "start": int(seg_frames[0]),
                            "end": int(seg_frames[-1]),
                            "frame_ids": seg_frames,
                            "target_len": EVENT_T,
                        })
                else:
                    # val/test: random windows with aggregated labels
                    eval_windows = sample_random_windows(
                        frames=frames,
                        y=y,
                        window_l=EVAL_WINDOW_L,
                        n_windows=EVAL_WINDOWS_PER_TRIAL,
                        rng=rng,
                        require_all_zero=None,
                    )
                    for idx, (start_i, end_i, label) in enumerate(eval_windows):
                        seg_frames = frames[start_i:end_i]
                        samples_to_write.append({
                            "sample_type": f"{split}_rand",
                            "idx": idx,
                            "label": label,
                            "start": int(seg_frames[0]),
                            "end": int(seg_frames[-1]),
                            "frame_ids": seg_frames,
                            "target_len": EVENT_T,
                        })

                for item in samples_to_write:
                    Xs = build_xyz_for_frames(wdf, pid, affected, item["frame_ids"])
                    total_samples += 1

                    if Xs.shape[0] < 2:
                        continue

                    if SKIP_SEGMENTS_WITH_NANS and segment_has_nans(Xs):
                        continue

                    if NORMALIZE:
                        Xs = normalize_xyz(Xs)

                    orig_len = int(Xs.shape[0])
                    Xs = resample_time(Xs, item["target_len"])

                    Xs_st = np.transpose(Xs, (1, 2, 0)).astype(np.float32, copy=False)
                    mask = np.zeros((Xs_st.shape[0] * Xs_st.shape[1], Xs_st.shape[2]), dtype=np.bool_)

                    key = f"{ex}_pid{pid}_aff{affected}_trial{trial}_{item['sample_type']}_{item['idx']}"
                    meta = {
                        "exercise": ex,
                        "pid": pid,
                        "affected": affected,
                        "trial": trial,
                        "frame_init": frame_init,
                        "frame_end": frame_end,
                        "sample_type": item["sample_type"],
                        "segment_start_frame": item["start"],
                        "segment_end_frame": item["end"],
                        "orig_len": orig_len,
                        "train_protocol": "event_pos_random_neg" if split == "train" else "random_window_eval",
                        "eval_agg": EVAL_AGGREGATION if split in {"val", "test"} else None,
                    }

                    value = pickle.dumps(
                        {"sample": Xs_st, "label": int(item["label"]), "mask": mask, "meta": meta},
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                    txn.put(key.encode("utf-8"), value)
                    dataset_keys[split].append(key)
                    kept_samples += 1

            print(
                f"[{ex}] kept/total samples so far: {kept_samples}/{total_samples} | skipped_trials: {skipped_trials}",
                flush=True
            )

        txn.put(b"__keys__", pickle.dumps(dataset_keys, protocol=pickle.HIGHEST_PROTOCOL))

    db.close()

    print("\nDone.", flush=True)
    print(f"Total samples:  {total_samples}", flush=True)
    print(f"Kept samples:   {kept_samples}", flush=True)
    print(f"Skipped trials: {skipped_trials}", flush=True)
    print(f"Output LMDB:    {OUT_LMDB}", flush=True)


if __name__ == "__main__":
    main()