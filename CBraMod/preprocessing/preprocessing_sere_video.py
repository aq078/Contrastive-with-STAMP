#!/usr/bin/env python3
"""
STAMP-style preprocessing for SERE MediaPipe WorldLandmarks (xyz + visibility) -> LMDB.

Dataset layout (as in your diagram)
-----------------------------------
ROOT/
  MediaPipe_skeletons/
    WorldLandmarks/
      E1_mp_world_landmarks.csv
      ...
      E5_mp_world_landmarks.csv
  Labels/
    video_level/
      compensation/ E1_labels_comp.csv ... E5_labels_comp.csv
      ROM/          E1_labels_rom.csv  ... E5_labels_rom.csv
      smoothness/   E1_labels_smooth.csv ... E5_labels_smooth.csv
      spasticity/   E1_labels_spast.csv ... E5_labels_spast.csv

WorldLandmarks header (wide)
----------------------------
pid, affected, frame, 0x,0y,0z,0v,0p, 1x,1y,1z,1v,1p, ... 32x,32y,32z,32v,32p

We keep xyz+visibility => (x,y,z,v) for each of 33 landmarks => sample shape [T,33,4].
We ignore 'p' (presence).

Video-level label headers (wide)
--------------------------------
compensation: pid affected trial frame_init frame_end comp comp_sh comp_sh_ele comp_sh_abd comp_tr comp_hd
ROM:          pid affected trial frame_init frame_end rom rom_sh rom_eb rom_wr rom_hp rom_kn rom_tr
smoothness:   pid affected trial frame_init frame_end smooth
spasticity:   pid affected trial frame_init frame_end spast spast_sh spast_hp spast_kn spast_ak spast_eb spast_wr

Output LMDB
-----------
key:   "{E}_pid{pid}_trial{trial}"
value: pickle.dumps({
          "sample": np.float32 [T,33,4],
          "label":  int/float (from label_col),
          "meta":   dict
       })
Also stores "__keys__" with {"train":[...], "val":[...], "test":[...]} subject-disjoint by pid.

Usage example
-------------
python preprocess_sere_worldlandmarks_lmdb.py \
  --root_dir /path/to/SERE \
  --out_lmdb /path/to/processed_sere/world_xyzv.lmdb \
  --label_type ROM \
  --label_col rom \
  --T 256 \
  --normalize 1 \
  --seed 42 \
  --split "train:0.7 val:0.15 test:0.15"
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lmdb
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


POSE_N = 33
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def normalize_xyzv(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    X: [T,33,4] (xyzv). Root-center by hip midpoint and scale by shoulder distance.
    Leaves visibility unchanged.
    """
    out = X.astype(np.float32, copy=True)
    xyz = out[..., :3]

    root = 0.5 * (xyz[:, LEFT_HIP, :] + xyz[:, RIGHT_HIP, :])  # [T,3]
    xyz = xyz - root[:, None, :]

    shoulder_vec = xyz[:, LEFT_SHOULDER, :] - xyz[:, RIGHT_SHOULDER, :]
    scale = np.linalg.norm(shoulder_vec, axis=-1)  # [T]
    scale = np.maximum(scale, eps)
    xyz = xyz / scale[:, None, None]

    out[..., :3] = xyz
    return out


def resample_time(X: np.ndarray, T_new: int) -> np.ndarray:
    """
    Linear interpolation along time.
    X: [T,33,4] -> [T_new,33,4]
    """
    T_old = X.shape[0]
    if T_old == T_new:
        return X.astype(np.float32)

    t_old = np.linspace(0.0, 1.0, T_old, endpoint=True)
    t_new = np.linspace(0.0, 1.0, T_new, endpoint=True)

    flat = X.reshape(T_old, -1)  # [T_old, 33*4]
    f = interp1d(t_old, flat, kind="linear", axis=0, fill_value="extrapolate", assume_sorted=True)
    flat_new = f(t_new).astype(np.float32)
    return flat_new.reshape(T_new, POSE_N, 4)


def split_pids(pids: List[Any], split_spec: str, seed: int) -> Dict[str, set]:
    """
    split_spec: "train:0.7 val:0.15 test:0.15"
    """
    parts: Dict[str, float] = {}
    for tok in split_spec.strip().split():
        name, frac = tok.split(":")
        parts[name] = float(frac)
    if abs(sum(parts.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1. Got: {parts}")

    uniq = sorted(set(pids))
    set_seed(seed)
    random.shuffle(uniq)

    n = len(uniq)
    n_train = int(round(parts.get("train", 0.0) * n))
    n_val = int(round(parts.get("val", 0.0) * n))

    train = set(uniq[:n_train])
    val = set(uniq[n_train:n_train + n_val])
    test = set(uniq[n_train + n_val:])
    return {"train": train, "val": val, "test": test}


# -----------------------------
# WorldLandmarks parsing (WIDE)
# -----------------------------
def load_worldlandmarks_wide(path: Path) -> pd.DataFrame:
    """
    Returns df with lowercase columns.
    Required: pid, affected, frame, and for j=0..32: {j}x,{j}y,{j}z,{j}v (visibility)
    (We ignore {j}p presence)
    """
    df = lower_cols(pd.read_csv(path))

    required = ["pid", "affected", "frame"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"[world] Missing column '{c}' in {path}. Found: {list(df.columns)[:30]}")

    # Spot-check a few expected keypoint columns
    for c in ["0x", "0y", "0z", "0v", "32x", "32y", "32z", "32v"]:
        if c not in df.columns:
            raise ValueError(f"[world] Missing expected column '{c}' in {path}. "
                             f"Your file should contain 0x..32v. Found: {list(df.columns)[:30]}...")

    return df


def build_xyzv_segment(df_world: pd.DataFrame, pid: Any, frame_init: int, frame_end: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Slice frames in [frame_init, frame_end] for a given pid.
    Returns:
      X: [T0,33,4] float32 (xyzv)
      meta: frame range and counts
    """
    sub = df_world[df_world["pid"] == pid].copy()
    if sub.empty:
        return np.zeros((0, POSE_N, 4), dtype=np.float32), {"empty": True}

    sub = sub[(sub["frame"] >= frame_init) & (sub["frame"] <= frame_end)].copy()
    if sub.empty:
        return np.zeros((0, POSE_N, 4), dtype=np.float32), {"empty": True}

    sub = sub.sort_values("frame")
    frames = sub["frame"].to_numpy()
    T0 = len(sub)

    X = np.zeros((T0, POSE_N, 4), dtype=np.float32)
    # Fill landmark data
    for j in range(POSE_N):
        X[:, j, 0] = sub[f"{j}x"].to_numpy(dtype=np.float32)
        X[:, j, 1] = sub[f"{j}y"].to_numpy(dtype=np.float32)
        X[:, j, 2] = sub[f"{j}z"].to_numpy(dtype=np.float32)
        X[:, j, 3] = sub[f"{j}v"].to_numpy(dtype=np.float32)  # visibility

    meta = {
        "frames_min": int(frames.min()),
        "frames_max": int(frames.max()),
        "num_frames": int(T0),
    }
    return X, meta


# -----------------------------
# Labels parsing
# -----------------------------
LABEL_FILE_BY_TYPE = {
    "compensation": "labels_comp.csv",
    "ROM": "labels_rom.csv",
    "smoothness": "labels_smooth.csv",
    "spasticity": "labels_spast.csv",
}


def load_video_labels(path: Path, label_col: str) -> pd.DataFrame:
    """
    Video-level labels are one row per trial segment:
      pid affected trial frame_init frame_end <label_col> ...
    """
    df = lower_cols(pd.read_csv(path))
    for c in ["pid", "affected", "trial", "frame_init", "frame_end"]:
        if c not in df.columns:
            raise ValueError(f"[labels] Missing '{c}' in {path}. Found: {list(df.columns)}")

    label_col = label_col.lower().strip()
    if label_col not in df.columns:
        raise ValueError(f"[labels] label_col='{label_col}' not found in {path}. Available: {list(df.columns)}")

    return df


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    CONFIG = {
        # SERE root directory
        "root_dir": "dataset/SERE_dataset_SHAREABLE_skeletons",

        # Output LMDB directory (will be created)
        "out_lmdb": "dataset/processed_sere/world_xyzv.lmdb",

        # One of: "compensation", "ROM", "smoothness", "spasticity"
        "label_type": "ROM",

        # Which label column to use from the label CSV
        # Examples:
        #   compensation: "comp" (or "comp_tr", "comp_hd", ...)
        #   ROM: "rom" (or "rom_sh", "rom_eb", ...)
        #   smoothness: "smooth"
        #   spasticity: "spast" (or "spast_wr", ...)
        "label_col": "rom",

        # Resample each trial segment to fixed length T
        "T": 256,

        # 1 to apply root-center+scale normalization, 0 to disable
        "normalize": 1,

        # Subject-disjoint split (by pid)
        "seed": 42,
        "split": "train:0.7 val:0.15 test:0.15",

        # LMDB size in bytes (increase if you store many samples)
        "map_size": 8_000_000_000,
    }

    # -----------------------------
    # (Optional) basic validation
    # -----------------------------
    required = ["root_dir", "out_lmdb", "label_type", "label_col", "T", "normalize", "seed", "split", "map_size"]
    missing = [k for k in required if k not in CONFIG]
    if missing:
        raise ValueError(f"Missing CONFIG keys: {missing}")

    class Args:
        pass

    args = Args()
    for k, v in CONFIG.items():
        setattr(args, k, v)

    # ---- everything below is the same as before ----
    root = Path(args.root_dir)

    world_dir = root / "MediaPipe_skeletons" / "WorldLandmarks"
    label_dir = root / "Labels" / "video_level" / args.label_type

    world_files = sorted(world_dir.glob("E*_mp_world_landmarks.csv"))
    if not world_files:
        raise FileNotFoundError(f"No world files found: {world_dir / 'E*_mp_world_landmarks.csv'}")

    # Collect pids for subject split
    all_pids: List[Any] = []
    for wf in world_files:
        wdf = load_worldlandmarks_wide(wf)
        all_pids.extend(list(wdf["pid"].unique()))
    pid_split = split_pids(all_pids, args.split, args.seed)

    # Open LMDB
    out_lmdb = Path(args.out_lmdb)
    out_lmdb.parent.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(str(out_lmdb), map_size=int(args.map_size))

    dataset_keys: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for wf in world_files:
        ex = wf.stem.split("_")[0]  # E1..E5
        label_name = LABEL_FILE_BY_TYPE[args.label_type]
        label_path = label_dir / f"{ex}_{label_name}"

        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found for {ex}: {label_path}")

        print(f"\n=== {ex} ===")
        print(f"World:  {wf}")
        print(f"Labels: {label_path}")

        wdf = load_worldlandmarks_wide(wf)
        ldf = load_video_labels(label_path, args.label_col)

        for _, row in ldf.iterrows():
            pid = row["pid"]
            trial = row["trial"]
            frame_init = int(row["frame_init"])
            frame_end = int(row["frame_end"])
            y = row[args.label_col.lower().strip()]
            y = int(y)

            split = "train" if pid in pid_split["train"] else "val" if pid in pid_split["val"] else "test"

            X, seg_meta = build_xyzv_segment(wdf, pid, frame_init, frame_end)
            if X.shape[0] < 5:
                continue

            if int(args.normalize) == 1:
                X = normalize_xyzv(X)

            X = resample_time(X, int(args.T))  # [T,33,4]

            # Store as [n_spatial, n_temporal, T] = [33,4,T] for STAMP loader
            sample_sct = np.transpose(X, (1, 2, 0)).astype(np.float32)  # [33,4,T]

            key = f"{ex}_pid{pid}_trial{trial}"
            meta = {
                "exercise": ex,
                "pid": pid,
                "trial": trial,
                "affected": row["affected"],
                "frame_init": frame_init,
                "frame_end": frame_end,
                "label_type": args.label_type,
                "label_col": args.label_col.lower().strip(),
                "orig_num_frames": int(seg_meta.get("num_frames", -1)),
            }

            data_dict = {
                "sample": sample_sct,
                "label": int(y) if isinstance(y, (np.integer, int)) else y,
                "meta": meta,
            }


           

            txn = db.begin(write=True)
            txn.put(key.encode(), pickle.dumps(data_dict))
            txn.commit()

            dataset_keys[split].append(key)

    txn = db.begin(write=True)
    txn.put(b"__keys__", pickle.dumps(dataset_keys))
    txn.commit()
    db.close()

    print("\nDone.")
    print({k: len(v) for k, v in dataset_keys.items()})
    print(f"LMDB saved to: {out_lmdb}")
    
if __name__ == '__main__':
    main()
