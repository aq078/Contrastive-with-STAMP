#!/usr/bin/env python3

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Dict, List

import lmdb
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


ROOT_DIR = Path("dataset/SERE_dataset_SHAREABLE_skeletons")
OUT_DIR = Path("dataset/processed_sere")

WORLD_DIR = ROOT_DIR / "MediaPipe_skeletons" / "WorldLandmarks"
VIDEO_LABEL_ROOT = ROOT_DIR / "Labels" / "video_level"

EXERCISES = ["E1", "E2", "E3", "E4", "E5"]
WORLD_FILENAME_FMT = "{ex}_mp_world_landmarks.csv"

TASK_SPECS = [
    {
        "dataset_name": "sere_video_comp",
        "label_type": "compensation",
        "label_dir": "compensation",
        "label_file": "labels_comp.csv",
        "label_col": "comp",
        "out_lmdb": OUT_DIR / "sere_video_comp_world_xyz_T256.lmdb",
    },
    {
        "dataset_name": "sere_video_rom",
        "label_type": "rom",
        "label_dir": "ROM",
        "label_file": "labels_rom.csv",
        "label_col": "rom",
        "out_lmdb": OUT_DIR / "sere_video_rom_world_xyz_T256.lmdb",
    },
    {
        "dataset_name": "sere_video_smooth",
        "label_type": "smoothness",
        "label_dir": "smoothness",
        "label_file": "labels_smooth.csv",
        "label_col": "smooth",
        "out_lmdb": OUT_DIR / "sere_video_smooth_world_xyz_T256.lmdb",
    },
    {
        "dataset_name": "sere_video_spast",
        "label_type": "spasticity",
        "label_dir": "spasticity",
        "label_file": "labels_spast.csv",
        "label_col": "spast",
        "out_lmdb": OUT_DIR / "sere_video_spast_world_xyz_T256.lmdb",
    },
]

FEATURES = ("x", "y", "z")
FEAT_D = len(FEATURES)

RESAMPLE_T = 256
NORMALIZE = True
MIN_TRIAL_FRAMES = 5
SKIP_TRIALS_WITH_NANS = True

SEED = 42
SPLIT = {"train": 0.5, "val": 0.25, "test": 0.25}

LMDB_MAP_SIZE = 8_000_000_000

POSE_N = 33
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

PID_COL = "pid"
AFFECTED_COL = "affected"
TRIAL_COL = "trial"
FRAME_COL = "frame"
FRAME_INIT_COL = "frame_init"
FRAME_END_COL = "frame_end"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


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
    val = set(uniq[n_train:n_train + n_val])
    test = set(uniq[n_train + n_val:])
    return {"train": train, "val": val, "test": test}


def normalize_xyz_or_xyzv(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
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
    return flat_new.reshape(T_new, POSE_N, FEAT_D)


def trial_has_nans(X: np.ndarray) -> bool:
    return bool(np.isnan(X).any())


def load_worldlandmarks_wide(path: Path) -> pd.DataFrame:
    df = lower_cols(pd.read_csv(path))

    for c in [PID_COL, AFFECTED_COL, FRAME_COL]:
        if c not in df.columns:
            raise ValueError(f"[world] Missing '{c}' in {path}")

    for j in range(POSE_N):
        for suf in FEATURES:
            col = f"{j}{suf}"
            if col not in df.columns:
                raise ValueError(f"[world] Missing '{col}' in {path}")

    df[FRAME_COL] = pd.to_numeric(df[FRAME_COL], errors="coerce")
    df = df.dropna(subset=[FRAME_COL])
    df[FRAME_COL] = df[FRAME_COL].astype(int)
    return df


def load_video_labels(path: Path, label_col: str) -> pd.DataFrame:
    df = lower_cols(pd.read_csv(path))

    required = [PID_COL, TRIAL_COL, FRAME_INIT_COL, FRAME_END_COL]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"[video labels] Missing '{c}' in {path}")

    if AFFECTED_COL not in df.columns:
        df[AFFECTED_COL] = np.nan

    label_col = label_col.lower().strip()
    if label_col not in df.columns:
        raise ValueError(f"[video labels] Missing label_col='{label_col}' in {path}")

    df[TRIAL_COL] = pd.to_numeric(df[TRIAL_COL], errors="coerce")
    df[FRAME_INIT_COL] = pd.to_numeric(df[FRAME_INIT_COL], errors="coerce")
    df[FRAME_END_COL] = pd.to_numeric(df[FRAME_END_COL], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    df = df.dropna(subset=[TRIAL_COL, FRAME_INIT_COL, FRAME_END_COL, label_col])
    df[TRIAL_COL] = df[TRIAL_COL].astype(int)
    df[FRAME_INIT_COL] = df[FRAME_INIT_COL].astype(int)
    df[FRAME_END_COL] = df[FRAME_END_COL].astype(int)
    df = df[df[FRAME_END_COL] >= df[FRAME_INIT_COL]].copy()

    return df


def build_segment_for_trial(
    df_world: pd.DataFrame,
    pid: Any,
    affected: Any,
    frame_init: int,
    frame_end: int,
) -> np.ndarray:
    sub = df_world[df_world[PID_COL] == pid].copy()

    if AFFECTED_COL in sub.columns and not pd.isna(affected):
        sub = sub[sub[AFFECTED_COL] == affected].copy()

    if sub.empty:
        return np.zeros((0, POSE_N, FEAT_D), dtype=np.float32)

    sub = sub[(sub[FRAME_COL] >= frame_init) & (sub[FRAME_COL] <= frame_end)].copy()
    if sub.empty:
        return np.zeros((0, POSE_N, FEAT_D), dtype=np.float32)

    sub = sub.sort_values(FRAME_COL)
    T0 = len(sub)

    X = np.zeros((T0, POSE_N, FEAT_D), dtype=np.float32)
    for j in range(POSE_N):
        for k, suf in enumerate(FEATURES):
            X[:, j, k] = sub[f"{j}{suf}"].to_numpy(dtype=np.float32)

    return X


def process_one_task(task_spec: Dict[str, Any]) -> None:
    dataset_name = task_spec["dataset_name"]
    label_type = task_spec["label_type"]
    label_dir_name = task_spec["label_dir"]
    label_file_suffix = task_spec["label_file"]
    label_col = task_spec["label_col"].lower().strip()
    out_lmdb = Path(task_spec["out_lmdb"])

    print(f"\n==============================")
    print(f"Building {dataset_name}")
    print(f"label_type={label_type} label_col={label_col}")
    print(f"==============================\n")

    all_pids: List[Any] = []
    for ex in EXERCISES:
        label_path = VIDEO_LABEL_ROOT / label_dir_name / f"{ex}_{label_file_suffix}"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        ldf = load_video_labels(label_path, label_col)
        all_pids.extend(ldf[PID_COL].unique().tolist())

    pid_split = split_pids(all_pids, SPLIT, SEED)

    out_lmdb.parent.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(str(out_lmdb), map_size=int(LMDB_MAP_SIZE))

    dataset_keys: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    total_trials = 0
    kept_trials = 0
    skipped_trials = 0

    with db.begin(write=True) as txn:
        for ex in EXERCISES:
            wf = WORLD_DIR / WORLD_FILENAME_FMT.format(ex=ex)
            vlf = VIDEO_LABEL_ROOT / label_dir_name / f"{ex}_{label_file_suffix}"

            if not wf.exists():
                raise FileNotFoundError(f"Missing world file: {wf}")
            if not vlf.exists():
                raise FileNotFoundError(f"Missing video label file: {vlf}")

            print(f"\n=== {ex} ===")
            print(f"World:       {wf}")
            print(f"Video labels:{vlf}")

            wdf = load_worldlandmarks_wide(wf)
            vdf = load_video_labels(vlf, label_col)
            vdf = vdf.sort_values([PID_COL, AFFECTED_COL, TRIAL_COL]).reset_index(drop=True)

            for row in vdf.itertuples(index=False):
                pid = getattr(row, PID_COL)
                affected = getattr(row, AFFECTED_COL)
                trial = int(getattr(row, TRIAL_COL))
                frame_init = int(getattr(row, FRAME_INIT_COL))
                frame_end = int(getattr(row, FRAME_END_COL))
                y = getattr(row, label_col)

                total_trials += 1

                X = build_segment_for_trial(
                    wdf,
                    pid=pid,
                    affected=affected,
                    frame_init=frame_init,
                    frame_end=frame_end,
                )

                if X.shape[0] < MIN_TRIAL_FRAMES:
                    skipped_trials += 1
                    continue

                if SKIP_TRIALS_WITH_NANS and trial_has_nans(X):
                    skipped_trials += 1
                    continue

                if NORMALIZE:
                    X = normalize_xyz_or_xyzv(X)

                orig_num_frames = int(X.shape[0])
                X = resample_time(X, RESAMPLE_T)

                X_st = np.transpose(X, (1, 2, 0)).astype(np.float32, copy=False)
                mask = np.zeros((X_st.shape[0] * X_st.shape[1], X_st.shape[2]), dtype=np.bool_)

                if pid in pid_split["train"]:
                    split = "train"
                elif pid in pid_split["val"]:
                    split = "val"
                else:
                    split = "test"

                key = f"{ex}_pid{pid}_aff{affected}_trial{trial}"
                meta = {
                    "dataset_name": dataset_name,
                    "task": label_type,
                    "label_type": label_type,
                    "label_col": label_col,
                    "exercise": ex,
                    "pid": pid,
                    "affected": affected,
                    "trial": trial,
                    "frame_init": frame_init,
                    "frame_end": frame_end,
                    "orig_num_frames": orig_num_frames,
                    "features": list(FEATURES),
                }

                value = pickle.dumps(
                    {
                        "sample": X_st,
                        "label": int(y) if float(y).is_integer() else float(y),
                        "mask": mask,
                        "meta": meta,
                    },
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

                txn.put(key.encode("utf-8"), value)
                dataset_keys[split].append(key)
                kept_trials += 1

            print(
                f"[{ex}] kept/total trials so far: "
                f"{kept_trials}/{total_trials} | skipped_trials: {skipped_trials}",
                flush=True,
            )

        txn.put(b"__keys__", pickle.dumps(dataset_keys, protocol=pickle.HIGHEST_PROTOCOL))

    db.close()

    print("\nDone.")
    print(f"{dataset_name}")
    print(f"Total trials:   {total_trials}")
    print(f"Kept trials:    {kept_trials}")
    print(f"Skipped trials: {skipped_trials}")
    print({k: len(v) for k, v in dataset_keys.items()})
    print(f"Output LMDB:    {out_lmdb}")


def main() -> None:
    set_seed(SEED)
    for task_spec in TASK_SPECS:
        process_one_task(task_spec)


if __name__ == "__main__":
    main()