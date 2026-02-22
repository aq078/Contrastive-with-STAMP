#!/usr/bin/env python3
"""
SERE preprocessing for FRAME-LEVEL compensation baseline (Option B: windowed samples),
using video-level trial segmentation (frame_init/frame_end) from E*_labels_comp.csv.

Inputs per exercise (E1..E5):
  - MediaPipe world landmarks (wide): MediaPipe_skeletons/WorldLandmarks/E1_mp_world_landmarks.csv
  - Frame labels (per frame):         Labels/frame_level/compensation/E1_frame_labels.csv
  - Video labels + segmentation:      Labels/video_level/compensation/E1_labels_comp.csv
        contains: pid, affected, trial, frame_init, frame_end, ...

Output:
  - LMDB where each record is a window:
        key = "{E}_pid{pid}_aff{affected}_trial{trial}_w{start}"
        value = pickle({
            "sample": [L,33,4] float32 (x,y,z,visibility),
            "label":  int 0/1  (aggregated from frame-level comp),
            "meta":   dict
        })
  - "__keys__" entry holding train/val/test splits (subject-disjoint by pid).

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

# =========================
# CONFIG (EDIT HERE)
# =========================

ROOT_DIR = Path("dataset/SERE_dataset_SHAREABLE_skeletons")  # <-- EDIT
OUT_LMDB = Path("dataset/processed_sere/sere_framecomp_world_xyzv_L64_S16.lmdb")  # <-- EDIT

WORLD_DIR = ROOT_DIR / "MediaPipe_skeletons" / "WorldLandmarks"
FRAME_LABEL_DIR = ROOT_DIR / "Labels" / "frame_level" / "compensation"
VIDEO_LABEL_DIR = ROOT_DIR / "Labels" / "video_level" / "compensation"

EXERCISES = ["E1", "E2", "E3", "E4", "E5"]

WORLD_FILENAME_FMT = "{ex}_mp_world_landmarks.csv"
FRAME_LABEL_FILENAME_FMT = "{ex}_frame_labels.csv"
VIDEO_LABEL_FILENAME_FMT = "{ex}_labels_comp.csv" 
VIDEO_LABEL_FILENAME = "_labels_comp.csv" 
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

# Windowing
WINDOW_L = 64
STRIDE = 16
AGGREGATION = "any"  # "any" | "majority" | "mean"

# Normalization
NORMALIZE = True

# Filtering
MIN_TRIAL_FRAMES = 64              # skip short trials
MIN_VALID_LABELS_IN_WINDOW = 1     # if labels are missing (NaN), skip windows with too few labels
SKIP_WINDOWS_WITH_NANS = True      # skip windows where skeleton has NaNs due to missing frames

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
    pid_to_counts: {pid: {0: n0, 1: n1}} (counts over trial segments/windows)
    """
    parts = {}
    for tok in split_spec.strip().split():
        name, frac = tok.split(":")
        parts[name] = float(frac)
    if abs(sum(parts.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1. Got: {parts}")

    # totals
    total0 = sum(pid_to_counts[pid].get(0, 0) for pid in pid_to_counts)
    total1 = sum(pid_to_counts[pid].get(1, 0) for pid in pid_to_counts)
    total = total0 + total1
    if total == 0:
        raise ValueError("No labels found to balance.")

    target_ratio1 = total1 / total  # desired positive rate

    # target sizes by number of segments (not pids)
    target_segments = {k: int(round(parts[k] * total)) for k in parts}
    # fix rounding drift
    drift = total - sum(target_segments.values())
    if drift != 0:
        # push drift into train by default
        target_segments["train"] += drift

    # state
    rng = random.Random(seed)
    pids = list(pid_to_counts.keys())
    rng.shuffle(pids)

    # sort hard cases first (big pids first)
    pids.sort(key=lambda pid: pid_to_counts[pid].get(0,0) + pid_to_counts[pid].get(1,0), reverse=True)

    splits = {k: set() for k in parts}
    cur = {k: {"n0": 0, "n1": 0, "n": 0} for k in parts}

    def score(assign_split: str, pid) -> float:
        # penalty = how far split’s pos rate would be from global + size overflow penalty
        n0 = cur[assign_split]["n0"] + pid_to_counts[pid].get(0, 0)
        n1 = cur[assign_split]["n1"] + pid_to_counts[pid].get(1, 0)
        n = n0 + n1
        if n == 0:
            rate_penalty = 0.0
        else:
            rate_penalty = abs((n1 / n) - target_ratio1)

        size_penalty = 0.0
        if cur[assign_split]["n"] + (pid_to_counts[pid].get(0,0)+pid_to_counts[pid].get(1,0)) > target_segments[assign_split]:
            size_penalty = 1.0  # discourage overflow

        return rate_penalty + 0.1 * size_penalty

    for pid in pids:
        # choose best split by score
        best = min(parts.keys(), key=lambda k: score(k, pid))
        splits[best].add(pid)
        cur[best]["n0"] += pid_to_counts[pid].get(0, 0)
        cur[best]["n1"] += pid_to_counts[pid].get(1, 0)
        cur[best]["n"] += pid_to_counts[pid].get(0, 0) + pid_to_counts[pid].get(1, 0)

    return splits

def split_pids(pids: List[Any], split: Dict[str, float], seed: int) -> Dict[str, set]:
    if abs(sum(split.values()) - 1.0) > 1e-6:
        raise ValueError(f"SPLIT fractions must sum to 1. Got {split}")

    uniq = sorted(set(pids))
    set_seed(seed)
    random.shuffle(uniq)

    n = len(uniq)
    n_train = int(split["train"] * n)
    n_val = int(split["val"] * n)
    train = set(uniq[:n_train])
    val = set(uniq[n_train : n_train + n_val])
    test = set(uniq[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


def normalize_xyzv(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    X: [T,33,4] (x,y,z,vis).
    Root-center by hip midpoint, scale by shoulder distance per-frame.
    """
    out = X.astype(np.float32, copy=True)
    xyz = out[..., :3]  # [T,33,3]

    root = 0.5 * (xyz[:, LEFT_HIP, :] + xyz[:, RIGHT_HIP, :])  # [T,3]
    xyz = xyz - root[:, None, :]

    shoulder_vec = xyz[:, LEFT_SHOULDER, :] - xyz[:, RIGHT_SHOULDER, :]
    scale = np.linalg.norm(shoulder_vec, axis=-1)  # [T]
    scale = np.maximum(scale, eps)
    xyz = xyz / scale[:, None, None]

    out[..., :3] = xyz
    return out


def window_has_nans(Xw: np.ndarray) -> bool:
    return bool(np.isnan(Xw).any())


def aggregate_label(y_window: np.ndarray, agg: str) -> int:
    """
    y_window: [L] float values (0/1 or NaN).
    Returns 0/1; -1 if unusable due to missing labels.
    """
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
    raise ValueError(f"Unknown AGGREGATION='{agg}'")


# =========================
# Loading WorldLandmarks (wide)
# =========================
def load_worldlandmarks_wide(path: Path) -> pd.DataFrame:
    df = lower_cols(pd.read_csv(path))

    for c in [PID_COL, AFFECTED_COL, FRAME_COL]:
        if c not in df.columns:
            raise ValueError(f"[world] Missing '{c}' in {path}. Found: {list(df.columns)}")

    for j in range(POSE_N):
        for suf in ["x", "y", "z", "v"]:
            col = f"{j}{suf}"
            if col not in df.columns:
                raise ValueError(f"[world] Missing '{col}' in {path}")

    df[FRAME_COL] = pd.to_numeric(df[FRAME_COL], errors="coerce")
    df = df.dropna(subset=[FRAME_COL])
    df[FRAME_COL] = df[FRAME_COL].astype(int)
    return df


def build_xyzv_for_frames(df_world: pd.DataFrame, pid: Any, affected: Any, frames: np.ndarray) -> np.ndarray:
    """
    Align skeleton rows to the provided frame ids.
    Missing frames become NaNs via reindex.
    """
    sub = df_world[(df_world[PID_COL] == pid) & (df_world[AFFECTED_COL] == affected)].copy()
    if sub.empty:
        return np.zeros((0, POSE_N, 4), dtype=np.float32)

    sub = sub.set_index(FRAME_COL, drop=False)
    frames = np.asarray(frames, dtype=int)

    rows = sub.reindex(index=frames, copy=False)  # NaNs where missing
    T = len(rows)

    X = np.zeros((T, POSE_N, 4), dtype=np.float32)
    for j in range(POSE_N):
        X[:, j, 0] = rows[f"{j}x"].to_numpy(dtype=np.float32)
        X[:, j, 1] = rows[f"{j}y"].to_numpy(dtype=np.float32)
        X[:, j, 2] = rows[f"{j}z"].to_numpy(dtype=np.float32)
        X[:, j, 3] = rows[f"{j}v"].to_numpy(dtype=np.float32)
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

    Some exercises may not have 'affected' in the video-level file (e.g., E3).
    """
    df = lower_cols(pd.read_csv(path))

    required = [PID_COL, TRIAL_COL, FRAME_INIT_COL, FRAME_END_COL]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"[video labels] Missing '{c}' in {path}. Found: {list(df.columns)}")

    # affected is optional
    if AFFECTED_COL not in df.columns:
        df[AFFECTED_COL] = np.nan  # fill; handled later

    df[TRIAL_COL] = pd.to_numeric(df[TRIAL_COL], errors="coerce")
    df[FRAME_INIT_COL] = pd.to_numeric(df[FRAME_INIT_COL], errors="coerce")
    df[FRAME_END_COL] = pd.to_numeric(df[FRAME_END_COL], errors="coerce")

    df = df.dropna(subset=[TRIAL_COL, FRAME_INIT_COL, FRAME_END_COL])
    df[TRIAL_COL] = df[TRIAL_COL].astype(int)
    df[FRAME_INIT_COL] = df[FRAME_INIT_COL].astype(int)
    df[FRAME_END_COL] = df[FRAME_END_COL].astype(int)

    df = df[df[FRAME_END_COL] >= df[FRAME_INIT_COL]].copy()
    return df



def compute_pid_window_counts() -> dict:
    """
    Dry-run pass: compute per-pid counts of aggregated window labels.
    Returns: pid_to_counts = {pid: {0: n0, 1: n1}}
    """
    pid_to_counts = {}  # pid -> {0: n0, 1: n1}

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

        wdf = load_worldlandmarks_wide(wf)
        fdf = load_frame_labels(flf)
        vdf = load_video_labels(vlf)

        # Index frame labels by (pid, affected)
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
                continue

            g = f_by[(pid, affected)]
            sub = g[(g[FRAME_COL] >= frame_init) & (g[FRAME_COL] <= frame_end)].copy()
            if len(sub) < max(MIN_TRIAL_FRAMES, WINDOW_L):
                continue

            frames = sub[FRAME_COL].to_numpy(dtype=int)
            y = sub[FRAME_COMP_COL].to_numpy(dtype=np.float32)

            X = build_xyzv_for_frames(wdf, pid, affected, frames)
            if X.shape[0] < WINDOW_L:
                continue

            if NORMALIZE:
                X = normalize_xyzv(X)

            T = X.shape[0]
            for start in range(0, T - WINDOW_L + 1, STRIDE):
                end = start + WINDOW_L
                Xw = X[start:end]
                yw = y[start:end]

                if SKIP_WINDOWS_WITH_NANS and window_has_nans(Xw):
                    continue

                y_agg = aggregate_label(yw, AGGREGATION)
                if y_agg < 0:
                    continue

                if pid not in pid_to_counts:
                    pid_to_counts[pid] = {0: 0, 1: 0}
                pid_to_counts[pid][int(y_agg)] += 1

    return pid_to_counts


def print_split_summary(pid_split: dict, pid_to_counts: dict) -> None:
    def summarize(split_name: str):
        pids = pid_split[split_name]
        n0 = sum(pid_to_counts[p].get(0, 0) for p in pids)
        n1 = sum(pid_to_counts[p].get(1, 0) for p in pids)
        tot = n0 + n1
        pos = (n1 / tot) if tot else float("nan")
        print(f"{split_name}: pids={len(pids)} windows={tot} pos_rate={pos:.4f} n0={n0} n1={n1}", flush=True)

    print("\n=== Split summary (by WINDOW labels) ===", flush=True)
    for s in ["train", "val", "test"]:
        summarize(s)


def main() -> None:
    # Lowercase configured column names to match lower_cols()
    global PID_COL, AFFECTED_COL, TRIAL_COL, FRAME_COL, FRAME_INIT_COL, FRAME_END_COL, FRAME_COMP_COL
    PID_COL = PID_COL.lower()
    AFFECTED_COL = AFFECTED_COL.lower()
    TRIAL_COL = TRIAL_COL.lower()
    FRAME_COL = FRAME_COL.lower()
    FRAME_INIT_COL = FRAME_INIT_COL.lower()
    FRAME_END_COL = FRAME_END_COL.lower()
    FRAME_COMP_COL = FRAME_COMP_COL.lower()

    set_seed(SEED)

    # -------------------------
    # PASS 1: compute pid->(n0,n1) over *windows*
    # -------------------------
    pid_to_counts = compute_pid_window_counts()
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
    total_windows = 0
    kept_windows = 0
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
                if len(sub) < max(MIN_TRIAL_FRAMES, WINDOW_L):
                    skipped_trials += 1
                    continue

                frames = sub[FRAME_COL].to_numpy(dtype=int)
                y = sub[FRAME_COMP_COL].to_numpy(dtype=np.float32)

                X = build_xyzv_for_frames(wdf, pid, affected, frames)
                if X.shape[0] < WINDOW_L:
                    skipped_trials += 1
                    continue

                if NORMALIZE:
                    X = normalize_xyzv(X)

                if pid in pid_split["train"]:
                    split = "train"
                elif pid in pid_split["val"]:
                    split = "val"
                else:
                    split = "test"

                T = X.shape[0]
                for start in range(0, T - WINDOW_L + 1, STRIDE):
                    end = start + WINDOW_L
                    Xw = X[start:end]
                    yw = y[start:end]
                    total_windows += 1

                    if SKIP_WINDOWS_WITH_NANS and window_has_nans(Xw):
                        continue

                    y_agg = aggregate_label(yw, AGGREGATION)
                    if y_agg < 0:
                        continue

                    key = f"{ex}_pid{pid}_aff{affected}_trial{trial}_w{start}"
                    meta = {
                        "exercise": ex,
                        "pid": pid,
                        "affected": affected,
                        "trial": trial,
                        "frame_init": frame_init,
                        "frame_end": frame_end,
                        "start": int(start),
                        "end": int(end),
                        "frames_start": int(frames[start]),
                        "frames_end": int(frames[end - 1]),
                        "agg": AGGREGATION,
                    }
                    ### STAMP COMPAT: reorder sample to (spatial, temporal, seq_len) and add mask ###
                    # Xw is currently (L, 33, 4) = (frames, joints, features)
                    # STAMP expects per-sample shape (n_spatial, n_temporal, seq_len) = (33, 4, L)
                    Xw_st = np.transpose(Xw, (1, 2, 0)).astype(np.float32, copy=False)  # (33,4,L)

                    # Optional but recommended: provide an explicit mask so STAMP doesn't infer a broken one.
                    # Shape (33*4, L) matches "channels x time" after flattening joints/features.
                    mask = np.zeros((Xw_st.shape[0] * Xw_st.shape[1], Xw_st.shape[2]), dtype=np.bool_)
                    ### END STAMP COMPAT ###
                    value = pickle.dumps(
                        {"sample": Xw_st, "label": int(y_agg), "mask": mask, "meta": meta},
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    

                    txn.put(key.encode("utf-8"), value)
                    dataset_keys[split].append(key)
                    kept_windows += 1

            print(f"[{ex}] kept/total windows so far: {kept_windows}/{total_windows} | skipped_trials: {skipped_trials}", flush=True)

        txn.put(b"__keys__", pickle.dumps(dataset_keys, protocol=pickle.HIGHEST_PROTOCOL))

    db.close()

    print("\nDone.", flush=True)
    print(f"Total windows:  {total_windows}", flush=True)
    print(f"Kept windows:   {kept_windows}", flush=True)
    print(f"Skipped trials: {skipped_trials}", flush=True)
    print(f"Output LMDB:    {OUT_LMDB}", flush=True)
    
if __name__=="__main__":
    main()
