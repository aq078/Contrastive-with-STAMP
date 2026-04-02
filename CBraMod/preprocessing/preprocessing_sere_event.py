#!/usr/bin/env python3
"""
SERE preprocessing for FRAME-LEVEL compensation EVENT-CENTERED samples,
using video-level trial segmentation (frame_init/frame_end) from E*_labels_comp.csv.

Inputs per exercise (E1..E5):
  - MediaPipe world landmarks (wide): MediaPipe_skeletons/WorldLandmarks/E1_mp_world_landmarks.csv
  - Frame labels (per frame):         Labels/frame_level/compensation/E1_frame_labels.csv
  - Video labels + segmentation:      Labels/video_level/compensation/E1_labels_comp.csv
        contains: pid, affected, trial, frame_init, frame_end, ...

Output:
  - LMDB where each record is an event-centered segment:
        key = "{E}_pid{pid}_aff{affected}_trial{trial}_{pos|neg}_{idx}"
        value = pickle({
            "sample": [33,3,T] float32 (joints, xyz, time),
            "label":  int 0/1,
            "meta":   dict
        })
  - "__keys__" entry holding train/val/test splits (subject-disjoint by pid).

Positive samples:
  - contiguous runs of comp==1 inside each trial
  - nearby gaps can be merged
  - each event gets context frames before/after
  - variable-length event is resampled to fixed length RESAMPLE_T

Negative samples:
  - drawn from comp==0 regions in the same trial
  - matched approximately to positive event lengths
  - also resampled to fixed length RESAMPLE_T

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

ROOT_DIR = Path("dataset/SERE_dataset_SHAREABLE_skeletons")  # <-- EDIT
OUT_LMDB = Path("dataset/processed_sere/sere_framecomp_event_xyz_T128.lmdb")  # <-- EDIT

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

# Choose which frame-level label column to supervise with:
# "comp" (overall) or one subtype like "comp_tr"
FRAME_COMP_COL = "comp"

# Event-centered segmentation
RESAMPLE_T = 128
MIN_TRIAL_FRAMES = 16

MIN_POS_EVENT_FRAMES = 5
MIN_NEG_EVENT_FRAMES = 5
MERGE_GAP_FRAMES = 3

POS_CONTEXT_BEFORE = 8
POS_CONTEXT_AFTER = 8

NEG_CONTEXT_BEFORE = 8
NEG_CONTEXT_AFTER = 8
MAX_NEG_PER_POS = 1

# Normalization
NORMALIZE = True

# Filtering
SKIP_SEGMENTS_WITH_NANS = True

# Subject split (pid-based)
SEED = 42
SPLIT = {"train": 0.6, "val": 0.15, "test": 0.25}

# LMDB
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
    """
    Group split by pid, trying to match overall label distribution in each split.
    pid_to_counts: {pid: {0: n0, 1: n1}} (counts over event segments)
    """
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
        if n == 0:
            rate_penalty = 0.0
        else:
            rate_penalty = abs((n1 / n) - target_ratio1)

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


def normalize_xyz(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    X: [T,33,3] (x,y,z).
    Root-center by hip midpoint, scale by shoulder distance per-frame.
    """
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
    """
    X: [T_old,33,3] -> [T_new,33,3]
    """
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


def run_length(iv: Tuple[int, int]) -> int:
    s, e = iv
    return e - s + 1


def contiguous_runs(frames: np.ndarray, labels: np.ndarray, target_value: int) -> List[Tuple[int, int]]:
    """
    Returns runs in absolute frame coordinates:
      [(start_frame, end_frame), ...]
    Requires both:
      - target label
      - consecutive frame numbers
    """
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


def subtract_intervals(
    whole: Tuple[int, int],
    blocked: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    whole minus blocked intervals, all in closed integer frame coordinates.
    """
    ws, we = whole
    blocked = sorted(blocked)
    result = []
    cur = ws

    for bs, be in blocked:
        if be < cur:
            continue
        if bs > we:
            break
        if bs > cur:
            result.append((cur, bs - 1))
        cur = max(cur, be + 1)

    if cur <= we:
        result.append((cur, we))

    return result


def event_segments_for_trial(
    frames: np.ndarray,
    y: np.ndarray,
    frame_init: int,
    frame_end: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Build positive event-centered segments and candidate negative segments.
    """
    pos_runs = contiguous_runs(frames, y, target_value=1)
    pos_runs = merge_close_runs(pos_runs, MERGE_GAP_FRAMES)
    pos_runs = [iv for iv in pos_runs if run_length(iv) >= MIN_POS_EVENT_FRAMES]

    pos_segments = []
    for s, e in pos_runs:
        seg = (max(frame_init, s - POS_CONTEXT_BEFORE), min(frame_end, e + POS_CONTEXT_AFTER))
        pos_segments.append(seg)

    pos_segments = merge_close_runs(sorted(pos_segments), max_gap=0)

    neg_candidates = subtract_intervals((frame_init, frame_end), pos_segments)
    neg_candidates = [
        (max(frame_init, s - NEG_CONTEXT_BEFORE), min(frame_end, e + NEG_CONTEXT_AFTER))
        for s, e in neg_candidates
    ]
    neg_candidates = merge_close_runs(sorted(neg_candidates), max_gap=0)
    neg_candidates = [iv for iv in neg_candidates if run_length(iv) >= MIN_NEG_EVENT_FRAMES]

    return pos_segments, neg_candidates


def sample_negative_segments(
    pos_segments: List[Tuple[int, int]],
    neg_candidates: List[Tuple[int, int]],
    rng: random.Random,
) -> List[Tuple[int, int]]:
    """
    For each positive segment, sample up to MAX_NEG_PER_POS matched negatives
    from candidate negative regions.
    """
    negatives = []

    if not pos_segments or not neg_candidates:
        return negatives

    for pos_iv in pos_segments:
        pos_len = run_length(pos_iv)
        usable = [iv for iv in neg_candidates if run_length(iv) >= pos_len]
        if not usable:
            continue

        for _ in range(MAX_NEG_PER_POS):
            base = rng.choice(usable)
            bs, be = base
            max_start = be - pos_len + 1
            start = rng.randint(bs, max_start)
            end = start + pos_len - 1
            negatives.append((start, end))

    return negatives


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
    """
    Align skeleton rows to the provided frame ids.
    Missing frames become NaNs via reindex.
    """
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
    """
    Video-level segmentation:
      required: pid, trial, frame_init, frame_end
      optional: affected
    """
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


def compute_pid_segment_counts() -> dict:
    """
    Dry-run pass: compute per-pid counts of event-centered segment labels.
    Returns: pid_to_counts = {pid: {0: n0, 1: n1}}
    """
    pid_to_counts = {}

    rng = random.Random(SEED)

    for ex in EXERCISES:
        wf = WORLD_DIR / WORLD_FILENAME_FMT.format(ex=ex)
        flf = FRAME_LABEL_DIR / FRAME_LABEL_FILENAME_FMT.format(ex=ex)
        vlf = VIDEO_LABEL_DIR / VIDEO_LABEL_FILENAME_FMT.format(ex=ex)

        if not wf.exists():
            raise FileNotFoundError(f"Missing world file: {wf}")
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

            pos_segments, neg_candidates = event_segments_for_trial(frames, y, frame_init, frame_end)
            neg_segments = sample_negative_segments(pos_segments, neg_candidates, rng)

            n0 = len(neg_segments)
            n1 = len(pos_segments)

            if (n0 + n1) == 0:
                continue

            if pid not in pid_to_counts:
                pid_to_counts[pid] = {0: 0, 1: 0}
            pid_to_counts[pid][0] += n0
            pid_to_counts[pid][1] += n1

    return pid_to_counts


def print_split_summary(pid_split: dict, pid_to_counts: dict) -> None:
    def summarize(split_name: str):
        pids = pid_split[split_name]
        n0 = sum(pid_to_counts[p].get(0, 0) for p in pids)
        n1 = sum(pid_to_counts[p].get(1, 0) for p in pids)
        tot = n0 + n1
        pos = (n1 / tot) if tot else float("nan")
        print(f"{split_name}: pids={len(pids)} segments={tot} pos_rate={pos:.4f} n0={n0} n1={n1}", flush=True)

    print("\n=== Split summary (by EVENT labels) ===", flush=True)
    for s in ["train", "val", "test"]:
        summarize(s)


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
    # PASS 1: compute pid->(n0,n1) over *event segments*
    # -------------------------
    pid_to_counts = compute_pid_segment_counts()
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
    total_segments = 0
    kept_segments = 0
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

                pos_segments, neg_candidates = event_segments_for_trial(frames, y, frame_init, frame_end)
                neg_segments = sample_negative_segments(pos_segments, neg_candidates, rng)

                if pid in pid_split["train"]:
                    split = "train"
                elif pid in pid_split["val"]:
                    split = "val"
                else:
                    split = "test"

                pos_idx = 0
                neg_idx = 0

                all_segments = [(1, seg) for seg in pos_segments] + [(0, seg) for seg in neg_segments]

                for label, (s, e) in all_segments:
                    seg_frames = np.arange(s, e + 1, dtype=int)
                    Xs = build_xyz_for_frames(wdf, pid, affected, seg_frames)
                    total_segments += 1

                    if Xs.shape[0] < 2:
                        continue

                    if SKIP_SEGMENTS_WITH_NANS and segment_has_nans(Xs):
                        continue

                    if NORMALIZE:
                        Xs = normalize_xyz(Xs)

                    orig_len = int(Xs.shape[0])
                    Xs = resample_time(Xs, RESAMPLE_T)

                    # STAMP expects (n_spatial, n_temporal, seq_len) = (33, 3, T)
                    Xs_st = np.transpose(Xs, (1, 2, 0)).astype(np.float32, copy=False)
                    mask = np.zeros((Xs_st.shape[0] * Xs_st.shape[1], Xs_st.shape[2]), dtype=np.bool_)

                    if label == 1:
                        seg_name = f"pos_{pos_idx}"
                        pos_idx += 1
                    else:
                        seg_name = f"neg_{neg_idx}"
                        neg_idx += 1

                    key = f"{ex}_pid{pid}_aff{affected}_trial{trial}_{seg_name}"
                    meta = {
                        "exercise": ex,
                        "pid": pid,
                        "affected": affected,
                        "trial": trial,
                        "frame_init": frame_init,
                        "frame_end": frame_end,
                        "segment_type": "positive" if label == 1 else "negative",
                        "segment_start_frame": int(s),
                        "segment_end_frame": int(e),
                        "orig_len": orig_len,
                        "source": "event_centered_compensation",
                    }

                    value = pickle.dumps(
                        {"sample": Xs_st, "label": int(label), "mask": mask, "meta": meta},
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                    txn.put(key.encode("utf-8"), value)
                    dataset_keys[split].append(key)
                    kept_segments += 1

            print(
                f"[{ex}] kept/total segments so far: {kept_segments}/{total_segments} | skipped_trials: {skipped_trials}",
                flush=True
            )

        txn.put(b"__keys__", pickle.dumps(dataset_keys, protocol=pickle.HIGHEST_PROTOCOL))

    db.close()

    print("\nDone.", flush=True)
    print(f"Total segments: {total_segments}", flush=True)
    print(f"Kept segments:  {kept_segments}", flush=True)
    print(f"Skipped trials: {skipped_trials}", flush=True)
    print(f"Output LMDB:    {OUT_LMDB}", flush=True)


if __name__ == "__main__":
    main()