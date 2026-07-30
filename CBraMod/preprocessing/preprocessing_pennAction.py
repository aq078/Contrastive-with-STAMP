"""
Preprocess PennAction 2D skeleton annotations into a STAMP-compatible LMDB.

Design:
  - Uses PennAction's official split:
        train ==  1 -> official training pool
        train == -1 -> official test set
  - Creates a stratified validation split from the official training sequences.
  - Splits are assigned at the SEQUENCE level before windowing, so windows from
    one video can never leak across train/val/test.
  - Interpolates joints over frames where visibility == 0.
  - Normalizes each frame using its [x1, y1, x2, y2] person bounding box.
  - Creates fixed-length windows and includes a tail-aligned final window.
  - Edge-pads sequences shorter than WINDOW_L.
  - Stores each sample in the same basic format used by the SERE STAMP pipeline:

        {
            "sample": [13, 2, WINDOW_L] float32,
            "label":  int in [0, 14],
            "mask":   [26, WINDOW_L] bool,
            "meta":   dict
        }

  - Stores:
        "__keys__"          -> {"train": [...], "val": [...], "test": [...]}
        "__class_to_idx__"  -> action-name to integer mapping
        "__idx_to_class__"  -> inverse mapping
        "__sequence_split__"-> sequence id to split
        "__dataset_meta__"  -> preprocessing configuration and statistics

This version is intended for PennAction experiments with MOMENT reduction=None.

The LMDB deliberately preserves the two semantic axes:
    spatial channels  = 13 joints
    temporal channels = 2 coordinates (x, y)

Each raw sample is stored as [13, 2, L]. The downstream MOMENT embedding
generator should preserve/restore these axes rather than treating the 26
joint-coordinate series as the final STAMP channel layout.

Note: the compatibility mask remains [13*2, L], because MOMENT may temporarily
flatten the joint-coordinate axes when processing the time series. That does
not change the intended STAMP layout of spatial=13, temporal=2.

Run from the STAMP repository root:
    python scripts/preprocess_pennaction.py
"""

from __future__ import annotations

import pickle
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import lmdb
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


# =============================================================================
# CONFIG: edit these paths/settings if needed
# =============================================================================

ROOT_DIR = Path("dataset/Penn_Action")
LABEL_DIR = ROOT_DIR / "labels"
FRAME_DIR = ROOT_DIR / "frames"  # Used only for path validation/metadata.
WINDOW_L = 128
STRIDE = 16
SEED = 42
OUT_LMDB = Path(
    "dataset/processed_pennaction/"
    f"pennaction_xy_bboxnorm_nored_L{WINDOW_L}_S{STRIDE}.lmdb"
)



# Fraction of PennAction's official training sequences used for validation.
VAL_FRACTION = 0.15

# Visibility handling:
#   "interpolate" -> linearly interpolate each joint coordinate over visible frames.
#                    If a joint is never visible, fill it with the bbox center before
#                    normalization, which becomes approximately zero afterward.
#   "keep"        -> retain the coordinates stored by PennAction.
VISIBILITY_MODE = "interpolate"

NORMALIZE_BY_BBOX = True
INCLUDE_TAIL_WINDOW = True
PAD_SHORT_SEQUENCES = True

# The STAMP SERE preprocessing reference stores an all-False boolean mask.
# Preserve that convention here. Padding validity is separately stored in meta.
STAMP_MASK_ALL_FALSE = True

# Remove an existing LMDB directory before writing.
OVERWRITE = True

# Manually exclude known problematic sequences before split construction.
SKIP_SEQUENCE_IDS = {
    "0038",
}

# Filled during scanning when a sequence is skipped because its annotation
# contains invalid bounding boxes.
AUTO_SKIPPED_SEQUENCE_IDS: List[str] = []

# Frame images are not needed for skeleton preprocessing. Set True only when
# you want to require a matching frames/<sequence_id>/ directory.
VALIDATE_FRAME_DIRECTORIES = False

LMDB_MAP_SIZE = 8_000_000_000
COMMIT_EVERY = 2_000


# Exact strings found in PennAction annotation files.
ACTION_NAMES = [
    "baseball_pitch",
    "baseball_swing",
    "bench_press",
    "bowl",
    "clean_and_jerk",
    "golf_swing",
    "jump_rope",
    "jumping_jacks",
    "pullup",
    "pushup",
    "situp",
    "squat",
    "strum_guitar",
    "tennis_forehand",
    "tennis_serve",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(ACTION_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

N_JOINTS = 13
N_COORDS = 2


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    mat_path: Path
    action: str
    label: int
    pose: str
    official_train_flag: int
    nframes: int
    dimensions: Tuple[int, int, int]


# =============================================================================
# General helpers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_config() -> None:
    if not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"PennAction label directory not found: {LABEL_DIR}")

    if not FRAME_DIR.is_dir():
        raise FileNotFoundError(f"PennAction frame directory not found: {FRAME_DIR}")

    if WINDOW_L <= 0:
        raise ValueError(f"WINDOW_L must be positive, got {WINDOW_L}")

    if STRIDE <= 0:
        raise ValueError(f"STRIDE must be positive, got {STRIDE}")

    if not 0.0 < VAL_FRACTION < 1.0:
        raise ValueError(
            f"VAL_FRACTION must be between 0 and 1, got {VAL_FRACTION}"
        )

    if VISIBILITY_MODE not in {"interpolate", "keep"}:
        raise ValueError(
            "VISIBILITY_MODE must be 'interpolate' or 'keep', "
            f"got {VISIBILITY_MODE!r}"
        )


def scalar_int(value: Any, field: str, path: Path) -> int:
    arr = np.asarray(value).squeeze()
    if arr.size != 1:
        raise ValueError(
            f"{path}: field {field!r} must be scalar, got shape {arr.shape}"
        )
    return int(arr.item())


def clean_matlab_string(value: Any) -> str:
    value = np.asarray(value).squeeze()

    if value.ndim == 0:
        return str(value.item()).strip()

    if value.dtype.kind in {"U", "S"}:
        return "".join(value.tolist()).strip()

    return str(value).strip()


# =============================================================================
# Annotation loading and validation
# =============================================================================

def load_annotation(path: Path) -> Dict[str, Any]:
    """
    Load one PennAction annotation.

    Confirmed PennAction field layout:
      action: str
      pose: str
      x, y, visibility: [T, 13]
      train: scalar, either 1 or -1
      bbox: [T, 4] in [x1, y1, x2, y2]
      dimensions: [height, width, T]
      nframes: scalar T
    """
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)

    required = {
        "action",
        "pose",
        "x",
        "y",
        "visibility",
        "train",
        "bbox",
        "dimensions",
        "nframes",
    }
    missing = sorted(required.difference(mat.keys()))
    if missing:
        raise KeyError(f"{path}: missing fields {missing}")

    action = clean_matlab_string(mat["action"])
    pose = clean_matlab_string(mat["pose"])
    train_flag = scalar_int(mat["train"], "train", path)
    nframes = scalar_int(mat["nframes"], "nframes", path)

    x = np.asarray(mat["x"], dtype=np.float32)
    y = np.asarray(mat["y"], dtype=np.float32)
    visibility = np.asarray(mat["visibility"], dtype=np.bool_)
    bbox = np.asarray(mat["bbox"], dtype=np.float32)
    dimensions = np.asarray(mat["dimensions"], dtype=np.int64).reshape(-1)

    if x.shape != (nframes, N_JOINTS):
        raise ValueError(
            f"{path}: x shape {x.shape}; expected {(nframes, N_JOINTS)}"
        )
    if y.shape != x.shape:
        raise ValueError(f"{path}: y shape {y.shape}; expected {x.shape}")
    if visibility.shape != x.shape:
        raise ValueError(
            f"{path}: visibility shape {visibility.shape}; expected {x.shape}"
        )
    if bbox.ndim != 2 or bbox.shape[1] != 4:
        raise ValueError(
            f"{path}: bbox shape {bbox.shape}; expected [T, 4]"
        )
    if dimensions.shape != (3,):
        raise ValueError(
            f"{path}: dimensions shape {dimensions.shape}; expected (3,)"
        )
    if int(dimensions[2]) != nframes:
        raise ValueError(
            f"{path}: dimensions[2]={dimensions[2]} but nframes={nframes}"
        )
    if action not in CLASS_TO_IDX:
        raise ValueError(
            f"{path}: unknown action {action!r}. "
            f"Expected one of {sorted(CLASS_TO_IDX)}"
        )
    if train_flag not in {-1, 1}:
        raise ValueError(
            f"{path}: unexpected train flag {train_flag}; expected -1 or 1"
        )

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{path}: x/y contain NaN or infinite values")

    # PennAction occasionally has a bbox sequence that is one or a few frames
    # shorter/longer than the skeleton sequence. Align it to nframes rather
    # than discarding an otherwise valid action sequence.
    original_bbox_frames = int(bbox.shape[0])
    bbox_length_adjustment = 0

    if original_bbox_frames < nframes:
        missing = nframes - original_bbox_frames
        if original_bbox_frames == 0:
            raise ValueError(f"{path}: bbox array is empty")

        print(
            f"[bbox repair] {path.stem}: bbox has {original_bbox_frames} "
            f"frames but skeleton has {nframes}; padding {missing} frame(s)",
            flush=True,
        )
        bbox = np.concatenate(
            [bbox, np.repeat(bbox[-1:, :], missing, axis=0)],
            axis=0,
        )
        bbox_length_adjustment = missing

    elif original_bbox_frames > nframes:
        extra = original_bbox_frames - nframes
        print(
            f"[bbox repair] {path.stem}: bbox has {original_bbox_frames} "
            f"frames but skeleton has {nframes}; trimming {extra} frame(s)",
            flush=True,
        )
        bbox = bbox[:nframes]
        bbox_length_adjustment = -extra

    widths = bbox[:, 2] - bbox[:, 0]
    heights = bbox[:, 3] - bbox[:, 1]
    valid_bbox = (
        np.isfinite(bbox).all(axis=1)
        & (widths > 0)
        & (heights > 0)
    )
    invalid_bbox_frames = int((~valid_bbox).sum())

    if invalid_bbox_frames > 0:
        valid_indices = np.flatnonzero(valid_bbox)

        if valid_indices.size == 0:
            raise ValueError(
                f"{path}: all {nframes} frames have invalid bounding boxes"
            )

        all_indices = np.arange(nframes)
        for coordinate in range(4):
            bbox[:, coordinate] = np.interp(
                all_indices,
                valid_indices,
                bbox[valid_indices, coordinate],
            ).astype(np.float32)

        print(
            f"[bbox repair] {path.stem}: interpolated "
            f"{invalid_bbox_frames}/{nframes} invalid frame(s)",
            flush=True,
        )

    # Defensive verification after padding/trimming/interpolation.
    widths = bbox[:, 2] - bbox[:, 0]
    heights = bbox[:, 3] - bbox[:, 1]
    if (
        bbox.shape != (nframes, 4)
        or not np.isfinite(bbox).all()
        or np.any(widths <= 0)
        or np.any(heights <= 0)
    ):
        raise ValueError(f"{path}: bounding boxes remain invalid after repair")

    skeleton = np.stack([x, y], axis=-1)  # [T, 13, 2]

    return {
        "sequence_id": path.stem,
        "action": action,
        "label": CLASS_TO_IDX[action],
        "pose": pose,
        "train_flag": train_flag,
        "nframes": nframes,
        "dimensions": tuple(int(v) for v in dimensions),
        "skeleton": skeleton,
        "visibility": visibility,
        "bbox": bbox,
        "bbox_original_frames": original_bbox_frames,
        "bbox_length_adjustment": bbox_length_adjustment,
        "bbox_interpolated_frames": invalid_bbox_frames,
    }


def scan_sequences() -> List[SequenceRecord]:
    paths = sorted(LABEL_DIR.glob("*.mat"))
    if not paths:
        raise RuntimeError(f"No .mat files found under {LABEL_DIR}")

    records: List[SequenceRecord] = []
    seen_ids = set()

    skipped = 0

    for i, path in enumerate(paths, start=1):
        sequence_id = path.stem

        if sequence_id in SKIP_SEQUENCE_IDS:
            print(
                f"[skip] {sequence_id}: manually excluded before splitting",
                flush=True,
            )
            skipped += 1
            continue

        try:
            ann = load_annotation(path)
        except ValueError as exc:
            message = str(exc)

            # Skip only the known annotation-quality problem. Other validation
            # errors are re-raised so genuine bugs or unexpected formats are
            # not silently hidden.
            if "all" in message and "frames have invalid bounding boxes" in message:
                print(
                    f"[skip] {sequence_id}: {message}",
                    flush=True,
                )
                skipped += 1
                AUTO_SKIPPED_SEQUENCE_IDS.append(sequence_id)
                continue

            raise

        sequence_id = ann["sequence_id"]

        if sequence_id in seen_ids:
            raise ValueError(f"Duplicate sequence id: {sequence_id}")
        seen_ids.add(sequence_id)

        if VALIDATE_FRAME_DIRECTORIES:
            frame_dir = FRAME_DIR / sequence_id
            if not frame_dir.is_dir():
                raise FileNotFoundError(
                    f"{path}: matching frame directory not found: {frame_dir}"
                )

        records.append(
            SequenceRecord(
                sequence_id=sequence_id,
                mat_path=path,
                action=ann["action"],
                label=int(ann["label"]),
                pose=ann["pose"],
                official_train_flag=int(ann["train_flag"]),
                nframes=int(ann["nframes"]),
                dimensions=ann["dimensions"],
            )
        )

        if i % 250 == 0 or i == len(paths):
            print(
                f"[scan] inspected {i}/{len(paths)} annotation files; "
                f"kept={len(records)} skipped={skipped}",
                flush=True,
            )

    print(
        f"[scan] complete: kept {len(records)} sequences, "
        f"skipped {skipped}",
        flush=True,
    )
    return records


# =============================================================================
# Split construction
# =============================================================================

def make_sequence_splits(
    records: Sequence[SequenceRecord],
) -> Dict[str, List[SequenceRecord]]:
    """
    Preserve the official PennAction test split and form validation only from
    official-training sequences. Stratification is by action class.
    """
    official_train = [r for r in records if r.official_train_flag == 1]
    official_test = [r for r in records if r.official_train_flag == -1]

    if not official_train or not official_test:
        raise RuntimeError(
            "Expected non-empty official train and test sequence sets"
        )

    indices = np.arange(len(official_train))
    labels = np.asarray([r.label for r in official_train], dtype=np.int64)

    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_FRACTION,
        random_state=SEED,
        shuffle=True,
        stratify=labels,
    )

    train = [official_train[int(i)] for i in sorted(train_indices)]
    val = [official_train[int(i)] for i in sorted(val_indices)]
    test = sorted(official_test, key=lambda r: r.sequence_id)

    split_records = {
        "train": sorted(train, key=lambda r: r.sequence_id),
        "val": sorted(val, key=lambda r: r.sequence_id),
        "test": test,
    }

    assert_split_integrity(split_records)
    return split_records


def assert_split_integrity(
    split_records: Mapping[str, Sequence[SequenceRecord]],
) -> None:
    ids = {
        split: {record.sequence_id for record in records}
        for split, records in split_records.items()
    }

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = ids[a].intersection(ids[b])
        if overlap:
            raise AssertionError(
                f"Sequence leakage between {a} and {b}: "
                f"{sorted(overlap)[:10]}"
            )

    for record in split_records["test"]:
        if record.official_train_flag != -1:
            raise AssertionError(
                f"Non-official-test sequence entered test: {record.sequence_id}"
            )

    for split in ("train", "val"):
        for record in split_records[split]:
            if record.official_train_flag != 1:
                raise AssertionError(
                    f"Official-test sequence entered {split}: "
                    f"{record.sequence_id}"
                )


# =============================================================================
# Skeleton processing
# =============================================================================

def interpolate_track(
    values: np.ndarray,
    visible: np.ndarray,
    fallback_value: float,
) -> np.ndarray:
    """
    Interpolate one coordinate track over visible frames.

    np.interp linearly fills internal gaps and extends the nearest visible value
    through leading/trailing invisible frames.
    """
    values = np.asarray(values, dtype=np.float32)
    visible = np.asarray(visible, dtype=np.bool_)
    valid_indices = np.flatnonzero(visible)

    if valid_indices.size == 0:
        return np.full(values.shape, fallback_value, dtype=np.float32)

    if valid_indices.size == 1:
        return np.full(
            values.shape,
            float(values[valid_indices[0]]),
            dtype=np.float32,
        )

    frame_indices = np.arange(values.shape[0], dtype=np.float32)
    return np.interp(
        frame_indices,
        valid_indices.astype(np.float32),
        values[valid_indices],
    ).astype(np.float32)


def interpolate_invisible_joints(
    skeleton: np.ndarray,
    visibility: np.ndarray,
    bbox: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      processed skeleton [T, 13, 2]
      ever-visible mask [13]

    A never-visible joint is filled with the per-frame bbox center, so bbox
    normalization maps it near zero rather than injecting a corner coordinate.
    """
    out = skeleton.astype(np.float32, copy=True)

    bbox_center_x = 0.5 * (bbox[:, 0] + bbox[:, 2])
    bbox_center_y = 0.5 * (bbox[:, 1] + bbox[:, 3])
    ever_visible = visibility.any(axis=0)

    for joint_idx in range(N_JOINTS):
        if ever_visible[joint_idx]:
            # The fallback is unused when at least one point is visible.
            out[:, joint_idx, 0] = interpolate_track(
                out[:, joint_idx, 0],
                visibility[:, joint_idx],
                fallback_value=0.0,
            )
            out[:, joint_idx, 1] = interpolate_track(
                out[:, joint_idx, 1],
                visibility[:, joint_idx],
                fallback_value=0.0,
            )
        else:
            out[:, joint_idx, 0] = bbox_center_x
            out[:, joint_idx, 1] = bbox_center_y

    return out, ever_visible.astype(np.bool_)


def normalize_by_bbox(
    skeleton: np.ndarray,
    bbox: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Frame-wise translation/scale normalization.

    bbox is [x1, y1, x2, y2]. Coordinates are centered by the bbox center and
    both x/y are divided by max(width, height), preserving aspect ratio.
    """
    out = skeleton.astype(np.float32, copy=True)

    center = np.stack(
        [
            0.5 * (bbox[:, 0] + bbox[:, 2]),
            0.5 * (bbox[:, 1] + bbox[:, 3]),
        ],
        axis=-1,
    ).astype(np.float32)

    width = bbox[:, 2] - bbox[:, 0]
    height = bbox[:, 3] - bbox[:, 1]
    scale = np.maximum(np.maximum(width, height), eps).astype(np.float32)

    out -= center[:, None, :]
    out /= scale[:, None, None]

    return out


def process_sequence_skeleton(
    annotation: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    skeleton = np.asarray(annotation["skeleton"], dtype=np.float32)
    visibility = np.asarray(annotation["visibility"], dtype=np.bool_)
    bbox = np.asarray(annotation["bbox"], dtype=np.float32)

    if VISIBILITY_MODE == "interpolate":
        skeleton, ever_visible = interpolate_invisible_joints(
            skeleton,
            visibility,
            bbox,
        )
    else:
        skeleton = skeleton.copy()
        ever_visible = visibility.any(axis=0).astype(np.bool_)

    if NORMALIZE_BY_BBOX:
        skeleton = normalize_by_bbox(skeleton, bbox)

    if not np.isfinite(skeleton).all():
        raise ValueError(
            f"{annotation['sequence_id']}: processed skeleton contains "
            "NaN or infinite values"
        )

    return skeleton.astype(np.float32, copy=False), ever_visible


# =============================================================================
# Window generation
# =============================================================================

def window_starts(
    nframes: int,
    window_l: int,
    stride: int,
    include_tail: bool,
) -> List[int]:
    if nframes <= window_l:
        return [0]

    starts = list(range(0, nframes - window_l + 1, stride))

    if include_tail:
        tail_start = nframes - window_l
        if starts[-1] != tail_start:
            starts.append(tail_start)

    return starts


def extract_window(
    skeleton: np.ndarray,
    visibility: np.ndarray,
    start: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Returns:
      Xw               [WINDOW_L, 13, 2]
      frame_valid_mask [WINDOW_L], True for original frames
      visibility_w     [WINDOW_L, 13], original visibility flags
      valid_length     number of unpadded frames
    """
    nframes = skeleton.shape[0]
    end = min(start + WINDOW_L, nframes)
    valid_length = end - start

    Xw = skeleton[start:end]
    visibility_w = visibility[start:end]

    frame_valid_mask = np.zeros(WINDOW_L, dtype=np.bool_)
    frame_valid_mask[:valid_length] = True

    if valid_length < WINDOW_L:
        if not PAD_SHORT_SEQUENCES:
            raise RuntimeError(
                "Encountered a short sequence while PAD_SHORT_SEQUENCES=False"
            )

        pad_length = WINDOW_L - valid_length

        Xw = np.pad(
            Xw,
            pad_width=((0, pad_length), (0, 0), (0, 0)),
            mode="edge",
        )

        # Repeated padded frames are not original observations.
        visibility_w = np.pad(
            visibility_w,
            pad_width=((0, pad_length), (0, 0)),
            mode="constant",
            constant_values=False,
        )

    if Xw.shape != (WINDOW_L, N_JOINTS, N_COORDS):
        raise AssertionError(f"Unexpected window shape: {Xw.shape}")

    return Xw, frame_valid_mask, visibility_w, valid_length


def make_stamp_mask(
    frame_valid_mask: np.ndarray,
) -> np.ndarray:
    """
    STAMP compatibility mask with shape [13*2, WINDOW_L].

    The SERE reference script writes an all-False mask, so this script does the
    same by default. The true padding mask is retained in meta["frame_valid_mask"].

    Change STAMP_MASK_ALL_FALSE only after verifying the exact mask semantics in
    the embedding-generation code.
    """
    if STAMP_MASK_ALL_FALSE:
        return np.zeros(
            (N_JOINTS * N_COORDS, WINDOW_L),
            dtype=np.bool_,
        )

    # Alternative masked-position convention: True means padded/masked.
    padded = ~frame_valid_mask
    return np.broadcast_to(
        padded[None, :],
        (N_JOINTS * N_COORDS, WINDOW_L),
    ).copy()


# =============================================================================
# Reporting
# =============================================================================

def count_classes(
    records: Iterable[SequenceRecord],
) -> Counter:
    return Counter(record.action for record in records)


def print_sequence_split_summary(
    split_records: Mapping[str, Sequence[SequenceRecord]],
) -> None:
    print("\n=== Sequence split summary ===", flush=True)
    for split in ("train", "val", "test"):
        records = split_records[split]
        print(f"{split:5s}: {len(records):4d} sequences", flush=True)
        counts = count_classes(records)
        for action in ACTION_NAMES:
            print(
                f"  {action:20s}: {counts[action]:4d}",
                flush=True,
            )


def print_window_summary(
    dataset_keys: Mapping[str, Sequence[str]],
    window_class_counts: Mapping[str, Counter],
) -> None:
    print("\n=== Window summary ===", flush=True)
    for split in ("train", "val", "test"):
        print(
            f"{split:5s}: {len(dataset_keys[split]):5d} windows",
            flush=True,
        )
        counts = window_class_counts[split]
        for action in ACTION_NAMES:
            print(
                f"  {action:20s}: {counts[action]:5d}",
                flush=True,
            )


# =============================================================================
# LMDB writing
# =============================================================================

def prepare_output_path() -> None:
    OUT_LMDB.parent.mkdir(parents=True, exist_ok=True)

    if OUT_LMDB.exists():
        if not OVERWRITE:
            raise FileExistsError(
                f"Output already exists: {OUT_LMDB}\n"
                "Set OVERWRITE = True to replace it."
            )

        if OUT_LMDB.is_dir():
            shutil.rmtree(OUT_LMDB)
        else:
            OUT_LMDB.unlink()


def write_lmdb(
    split_records: Mapping[str, Sequence[SequenceRecord]],
) -> None:
    prepare_output_path()

    dataset_keys: Dict[str, List[str]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    sequence_split: Dict[str, str] = {}
    window_class_counts: Dict[str, Counter] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }
    sequence_window_counts: Dict[str, int] = {}
    padded_window_counts: Counter = Counter()

    total_windows = 0
    txn_writes = 0

    db = lmdb.open(
        str(OUT_LMDB),
        map_size=int(LMDB_MAP_SIZE),
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=True,
    )

    txn = db.begin(write=True)

    try:
        for split in ("train", "val", "test"):
            records = split_records[split]

            for seq_idx, record in enumerate(records, start=1):
                annotation = load_annotation(record.mat_path)
                skeleton, ever_visible = process_sequence_skeleton(annotation)
                visibility = np.asarray(
                    annotation["visibility"],
                    dtype=np.bool_,
                )

                starts = window_starts(
                    nframes=record.nframes,
                    window_l=WINDOW_L,
                    stride=STRIDE,
                    include_tail=INCLUDE_TAIL_WINDOW,
                )

                sequence_split[record.sequence_id] = split
                sequence_window_counts[record.sequence_id] = len(starts)

                for start in starts:
                    Xw, frame_valid_mask, visibility_w, valid_length = (
                        extract_window(skeleton, visibility, start)
                    )

                    # STAMP expects:
                    #   (n_spatial, n_temporal, seq_len) = (13, 2, WINDOW_L)
                    Xw_stamp = np.transpose(
                        Xw,
                        (1, 2, 0),
                    ).astype(np.float32, copy=False)

                    # reduction=None experiment:
                    # preserve semantic axes as [spatial=13, temporal=2, time=L].
                    expected_shape = (N_JOINTS, N_COORDS, WINDOW_L)
                    if Xw_stamp.shape != expected_shape:
                        raise AssertionError(
                            f"Expected no-reduction sample layout "
                            f"{expected_shape}, got {Xw_stamp.shape}"
                        )

                    stamp_mask = make_stamp_mask(frame_valid_mask)

                    end_exclusive = min(start + WINDOW_L, record.nframes)
                    key = f"{record.sequence_id}_w{start:04d}"

                    meta = {
                        "dataset": "PennAction",
                        "sequence_id": record.sequence_id,
                        "action": record.action,
                        "label": int(record.label),
                        "pose": record.pose,
                        "split": split,
                        "official_train_flag": int(
                            record.official_train_flag
                        ),
                        "nframes": int(record.nframes),
                        "dimensions": tuple(record.dimensions),
                        "window_start_idx": int(start),
                        "window_end_idx_exclusive": int(end_exclusive),
                        # PennAction image filenames are one-based.
                        "frame_number_start": int(start + 1),
                        "frame_number_end": int(end_exclusive),
                        "valid_length": int(valid_length),
                        "padded_frames": int(WINDOW_L - valid_length),
                        "frame_valid_mask": frame_valid_mask,
                        "original_visibility": visibility_w,
                        "ever_visible_joints": ever_visible,
                        "normalization": (
                            "bbox_center_max_side"
                            if NORMALIZE_BY_BBOX
                            else "none"
                        ),
                        "visibility_mode": VISIBILITY_MODE,
                        "window_length": int(WINDOW_L),
                        "stride": int(STRIDE),
                    }

                    value = {
                        "sample": Xw_stamp,
                        "label": int(record.label),
                        "mask": stamp_mask,
                        "meta": meta,
                    }

                    if Xw_stamp.shape != (
                        N_JOINTS,
                        N_COORDS,
                        WINDOW_L,
                    ):
                        raise AssertionError(
                            f"{key}: sample shape {Xw_stamp.shape}"
                        )
                    if stamp_mask.shape != (
                        N_JOINTS * N_COORDS,
                        WINDOW_L,
                    ):
                        raise AssertionError(
                            f"{key}: mask shape {stamp_mask.shape}"
                        )

                    inserted = txn.put(
                        key.encode("utf-8"),
                        pickle.dumps(
                            value,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        ),
                        overwrite=False,
                    )
                    if not inserted:
                        raise KeyError(f"Duplicate LMDB key: {key}")

                    dataset_keys[split].append(key)
                    window_class_counts[split][record.action] += 1
                    total_windows += 1
                    txn_writes += 1

                    if valid_length < WINDOW_L:
                        padded_window_counts[split] += 1

                    if txn_writes >= COMMIT_EVERY:
                        txn.commit()
                        txn = db.begin(write=True)
                        txn_writes = 0
                        print(
                            f"[write] committed {total_windows} windows",
                            flush=True,
                        )

                if seq_idx % 100 == 0 or seq_idx == len(records):
                    print(
                        f"[{split}] processed "
                        f"{seq_idx}/{len(records)} sequences; "
                        f"{len(dataset_keys[split])} windows",
                        flush=True,
                    )

        dataset_meta = {
            "dataset_name": "pennaction",
            "root_dir": str(ROOT_DIR),
            "window_length": int(WINDOW_L),
            "stride": int(STRIDE),
            "seed": int(SEED),
            "validation_fraction_of_official_train": float(
                VAL_FRACTION
            ),
            "n_classes": len(ACTION_NAMES),
            "raw_n_spatial_channels": N_JOINTS,
            "raw_n_temporal_channels": N_COORDS,
            "raw_sample_shape": (
                N_JOINTS,
                N_COORDS,
                WINDOW_L,
            ),
            # Intended STAMP channel layout for reduction=None.
            # Raw samples are already [spatial, temporal, time] = [13, 2, L].
            "intended_n_spatial_channels": N_JOINTS,
            "intended_n_temporal_channels": N_COORDS,
            "intended_stamp_channel_layout": (
                N_JOINTS,
                N_COORDS,
            ),
            "moment_reduction": "none",
            # MOMENT/embedding code may temporarily flatten 13*2 series for
            # batched processing, but it must preserve/restore the semantic
            # [13, 2] axes for the downstream STAMP experiment.
            "moment_input_series_count": (
                N_JOINTS * N_COORDS
            ),
            "normalization": (
                "bbox_center_max_side"
                if NORMALIZE_BY_BBOX
                else "none"
            ),
            "visibility_mode": VISIBILITY_MODE,
            "include_tail_window": bool(INCLUDE_TAIL_WINDOW),
            "pad_short_sequences": bool(PAD_SHORT_SEQUENCES),
            "stamp_mask_all_false": bool(STAMP_MASK_ALL_FALSE),
            "manually_skipped_sequence_ids": sorted(SKIP_SEQUENCE_IDS),
            "automatically_skipped_sequence_ids": sorted(
                AUTO_SKIPPED_SEQUENCE_IDS
            ),
            "all_skipped_sequence_ids": sorted(
                set(SKIP_SEQUENCE_IDS)
                | set(AUTO_SKIPPED_SEQUENCE_IDS)
            ),
            "validate_frame_directories": bool(
                VALIDATE_FRAME_DIRECTORIES
            ),
            "sequence_counts": {
                split: len(split_records[split])
                for split in ("train", "val", "test")
            },
            "window_counts": {
                split: len(dataset_keys[split])
                for split in ("train", "val", "test")
            },
            "padded_window_counts": {
                split: int(padded_window_counts[split])
                for split in ("train", "val", "test")
            },
            "sequence_window_counts": sequence_window_counts,
        }

        metadata_items = {
            b"__keys__": dataset_keys,
            b"__class_to_idx__": CLASS_TO_IDX,
            b"__idx_to_class__": IDX_TO_CLASS,
            b"__sequence_split__": sequence_split,
            b"__dataset_meta__": dataset_meta,
        }

        for key, obj in metadata_items.items():
            txn.put(
                key,
                pickle.dumps(
                    obj,
                    protocol=pickle.HIGHEST_PROTOCOL,
                ),
            )

        txn.commit()
        txn = None
        db.sync()

    except Exception:
        if txn is not None:
            txn.abort()
        raise
    finally:
        db.close()

    print_window_summary(dataset_keys, window_class_counts)

    print("\n=== Padding summary ===", flush=True)
    for split in ("train", "val", "test"):
        print(
            f"{split:5s}: {padded_window_counts[split]} "
            "padded windows",
            flush=True,
        )

    print("\nDone.", flush=True)
    print(f"Total windows: {total_windows}", flush=True)
    print(f"Output LMDB:   {OUT_LMDB}", flush=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    AUTO_SKIPPED_SEQUENCE_IDS.clear()
    validate_config()
    set_seed(SEED)

    records = scan_sequences()

    print("\n=== Official split flags ===", flush=True)
    official_counts = Counter(r.official_train_flag for r in records)
    for flag in sorted(official_counts):
        print(f"{flag:2d}: {official_counts[flag]}", flush=True)

    print("\n=== Full dataset action counts ===", flush=True)
    full_counts = count_classes(records)
    for action in ACTION_NAMES:
        print(f"{action:20s}: {full_counts[action]:4d}", flush=True)

    lengths = np.asarray([r.nframes for r in records], dtype=np.int64)
    print("\n=== Sequence lengths ===", flush=True)
    print(f"count:  {len(lengths)}", flush=True)
    print(f"min:    {lengths.min()}", flush=True)
    print(f"max:    {lengths.max()}", flush=True)
    print(f"mean:   {lengths.mean():.3f}", flush=True)
    print(f"median: {np.median(lengths):.1f}", flush=True)
    print(
        f"< WINDOW_L ({WINDOW_L}): "
        f"{int(np.sum(lengths < WINDOW_L))}",
        flush=True,
    )

    split_records = make_sequence_splits(records)
    print_sequence_split_summary(split_records)
    write_lmdb(split_records)


if __name__ == "__main__":
    main()