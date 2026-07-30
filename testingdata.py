"""
Inspect PennAction MOMENT embeddings produced for STAMP.

The script checks:
  - train/val/test LMDB directories exist
  - number of samples in each split
  - embedding shapes and dtypes
  - labels are in [0, 14]
  - embeddings contain no NaN or Inf
  - each sample contains 26 reduced MOMENT embeddings
  - embedding dimension is 1024
  - one stacked STAMP batch can be formed as [B, 1, 26, 1024]

Default expected layout:
    embeddings/Penn_Action/MOMENT-1-large/
        train/
        val/
        test/

Usage:
    python inspect_pennaction_embeddings.py

Specify another embedding root:
    python inspect_pennaction_embeddings.py \
        --embedding-root embeddings/Penn_Action/MOMENT-1-large

Inspect more samples:
    python inspect_pennaction_embeddings.py --max-samples-per-split 100
"""

from __future__ import annotations
import re
import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import lmdb
import numpy as np


EXPECTED_SPLIT_COUNTS = {
    "train": 4024,
    "val": 780,
    "test": 4179,
}
EXPECTED_NUM_CHANNELS = 26
EXPECTED_EMBEDDING_DIM = 1024
EXPECTED_NUM_CLASSES = 15


def decode_value(raw: bytes):
    """
    Decode one PennAction embedding sample stored as raw float32 bytes.

    Expected shape:
        [26, 1024]
    """
    expected_values = 26 * 1024
    expected_bytes = expected_values * np.dtype(np.float32).itemsize

    if len(raw) != expected_bytes:
        raise ValueError(
            f"Unexpected raw byte length: {len(raw)}. "
            f"Expected {expected_bytes} bytes for a "
            f"(26, 1024) float32 embedding."
        )

    embedding = np.frombuffer(
        raw,
        dtype=np.float32,
    ).reshape(26, 1024)

    return embedding


def normalize_key(key: bytes) -> str:
    try:
        return key.decode("utf-8")
    except UnicodeDecodeError:
        return repr(key)


def is_metadata_key(key: bytes) -> bool:
    key_text = normalize_key(key)
    return key_text.startswith("__")


def get_first(
    item: Any,
    candidate_names: Iterable[str],
) -> Any | None:
    if not isinstance(item, dict):
        return None

    for name in candidate_names:
        if name in item:
            return item[name]

    return None


def extract_embedding(item: Any) -> np.ndarray:
    """
    Extract embeddings from several common storage structures.
    """
    if isinstance(item, np.ndarray):
        return item

    if isinstance(item, dict):
        value = get_first(
            item,
            (
                "embedding",
                "embeddings",
                "features",
                "feature",
                "x",
                "sample",
            ),
        )
        if value is None:
            raise KeyError(
                "Could not find embeddings. Available dictionary fields: "
                f"{sorted(item.keys())}"
            )
        return np.asarray(value)

    if isinstance(item, (tuple, list)) and len(item) > 0:
        return np.asarray(item[0])

    raise TypeError(
        f"Unsupported LMDB item type for embeddings: {type(item)}"
    )

def extract_label_from_key(key: str) -> int:
    match = re.search(r"_y(\d+)$", key)

    if match is None:
        raise ValueError(
            f"Could not extract label from key: {key}"
        )

    label = int(match.group(1))

    if not 0 <= label < 15:
        raise ValueError(
            f"Label {label} from key {key} is outside [0, 14]"
        )

    return label
def extract_label(item: Any) -> int | None:
    if isinstance(item, dict):
        value = get_first(
            item,
            (
                "label",
                "y",
                "target",
                "original_label",
            ),
        )
        if value is None:
            return None
        return int(np.asarray(value).reshape(-1)[0])

    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return int(np.asarray(item[1]).reshape(-1)[0])

    return None


def convert_to_26_by_1024(
    embedding: np.ndarray,
    key: str,
) -> np.ndarray:
    """
    Normalize expected PennAction embedding layouts to [26, 1024].

    Accepted shapes include:
      [26, 1024]
      [1, 26, 1024]
      [2, 13, 1024]
      [13, 2, 1024]
      [26 * 1024]
    """
    embedding = np.asarray(embedding)

    if embedding.shape == (
        EXPECTED_NUM_CHANNELS,
        EXPECTED_EMBEDDING_DIM,
    ):
        return embedding

    if embedding.shape == (
        1,
        EXPECTED_NUM_CHANNELS,
        EXPECTED_EMBEDDING_DIM,
    ):
        return embedding[0]

    if embedding.shape == (2, 13, EXPECTED_EMBEDDING_DIM):
        return embedding.reshape(
            EXPECTED_NUM_CHANNELS,
            EXPECTED_EMBEDDING_DIM,
        )

    if embedding.shape == (13, 2, EXPECTED_EMBEDDING_DIM):
        return embedding.reshape(
            EXPECTED_NUM_CHANNELS,
            EXPECTED_EMBEDDING_DIM,
        )

    if embedding.size == (
        EXPECTED_NUM_CHANNELS * EXPECTED_EMBEDDING_DIM
    ):
        return embedding.reshape(
            EXPECTED_NUM_CHANNELS,
            EXPECTED_EMBEDDING_DIM,
        )

    raise AssertionError(
        f"{key}: embedding shape {embedding.shape} cannot be converted "
        f"to ({EXPECTED_NUM_CHANNELS}, {EXPECTED_EMBEDDING_DIM})"
    )


def open_lmdb(path: Path) -> lmdb.Environment:
    if not path.exists():
        raise FileNotFoundError(f"LMDB path does not exist: {path}")

    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        subdir=path.is_dir(),
    )


def inspect_split(
    split: str,
    path: Path,
    max_samples: int | None,
    batch_size: int,
    print_samples: int,
    enforce_counts: bool,
) -> None:
    print(f"\n{'=' * 70}")
    print(f"Split: {split}")
    print(f"Path:  {path}")
    print(f"{'=' * 70}")

    env = open_lmdb(path)

    checked = 0
    labels: list[int] = []
    batch_embeddings: list[np.ndarray] = []
    shapes: dict[tuple[int, ...], int] = {}
    dtypes: dict[str, int] = {}
    total_entries = 0
    metadata_entries = 0

    try:
        with env.begin(write=False) as txn:
            stats = txn.stat()
            total_lmdb_entries = int(stats["entries"])

            cursor = txn.cursor()
            for raw_key, raw_value in cursor:
                if is_metadata_key(raw_key):
                    metadata_entries += 1
                    continue

                total_entries += 1

                if max_samples is not None and checked >= max_samples:
                    continue

                key = normalize_key(raw_key)

                raw_embedding = decode_value(raw_value)
                label = extract_label_from_key(key)

                shapes[tuple(raw_embedding.shape)] = (
                    shapes.get(tuple(raw_embedding.shape), 0) + 1
                )
                dtype_name = str(raw_embedding.dtype)
                dtypes[dtype_name] = dtypes.get(dtype_name, 0) + 1

                embedding = convert_to_26_by_1024(
                    raw_embedding,
                    key=key,
                )

                if not np.issubdtype(
                    embedding.dtype,
                    np.floating,
                ):
                    raise AssertionError(
                        f"{key}: embedding dtype is {embedding.dtype}; "
                        "expected floating point"
                    )

                if not np.isfinite(embedding).all():
                    bad = int((~np.isfinite(embedding)).sum())
                    raise AssertionError(
                        f"{key}: embedding contains {bad} NaN/Inf values"
                    )

                if label is not None:
                    if not 0 <= label < EXPECTED_NUM_CLASSES:
                        raise AssertionError(
                            f"{key}: label {label} is outside "
                            f"[0, {EXPECTED_NUM_CLASSES - 1}]"
                        )
                    labels.append(label)

                if len(batch_embeddings) < batch_size:
                    batch_embeddings.append(
                        embedding.astype(np.float32, copy=False)
                    )

                if checked < print_samples:
                    print(f"\nSample {checked + 1}")
                    print(f"  key:            {key}")
                    print(f"  stored shape:   {raw_embedding.shape}")
                    print(f"  normalized:     {embedding.shape}")
                    print(f"  dtype:          {embedding.dtype}")
                    print(
                        "  value range:    "
                        f"[{embedding.min():.6f}, "
                        f"{embedding.max():.6f}]"
                    )
                    print(
                        f"  mean/std:       "
                        f"{embedding.mean():.6f} / "
                        f"{embedding.std():.6f}"
                    )
                    print(f"  label:           {label}")

                checked += 1

            print("\nLMDB summary")
            print(f"  all LMDB entries:   {total_lmdb_entries}")
            print(f"  sample entries:     {total_entries}")
            print(f"  metadata entries:   {metadata_entries}")
            print(f"  samples validated:  {checked}")
            print(f"  stored shapes:      {shapes}")
            print(f"  stored dtypes:      {dtypes}")

            expected_count = EXPECTED_SPLIT_COUNTS[split]
            if total_entries != expected_count:
                message = (
                    f"{split}: found {total_entries} samples, "
                    f"expected {expected_count}"
                )
                if enforce_counts:
                    raise AssertionError(message)
                print(f"  WARNING: {message}")
            else:
                print(
                    f"  sample count:       correct "
                    f"({expected_count})"
                )

            if labels:
                unique, counts = np.unique(
                    np.asarray(labels),
                    return_counts=True,
                )
                label_counts = {
                    int(label): int(count)
                    for label, count in zip(unique, counts)
                }
                print(f"  checked labels:     {label_counts}")
            else:
                print(
                    "  labels:             not stored in these "
                    "embedding entries"
                )

            if not batch_embeddings:
                raise RuntimeError(
                    f"No sample embeddings found in {path}"
                )

            batch = np.stack(batch_embeddings, axis=0)
            stamp_batch = batch[:, None, :, :]

            print("\nBatch shape check")
            print(f"  reduced MOMENT batch: {batch.shape}")
            print(f"  STAMP input batch:    {stamp_batch.shape}")
            print(
                "  expected STAMP shape: "
                f"[B, 1, {EXPECTED_NUM_CHANNELS}, "
                f"{EXPECTED_EMBEDDING_DIM}]"
            )

            expected = (
                len(batch_embeddings),
                1,
                EXPECTED_NUM_CHANNELS,
                EXPECTED_EMBEDDING_DIM,
            )
            if stamp_batch.shape != expected:
                raise AssertionError(
                    f"STAMP batch shape {stamp_batch.shape}; "
                    f"expected {expected}"
                )

    finally:
        env.close()

    print(f"\n{split}: all embedding checks passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect PennAction MOMENT embedding LMDBs."
    )
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path(
            "embeddings/Penn_Action/MOMENT-1-large"
        ),
        help=(
            "Directory containing train, val, and test embedding "
            "LMDB directories."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="Split to inspect.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help=(
            "Maximum samples to fully validate per split. "
            "Default validates every sample."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used for the final shape check.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=2,
        help="Number of detailed sample summaries per split.",
    )
    parser.add_argument(
        "--enforce-counts",
        action="store_true",
        help=(
            "Fail when split counts do not match 4024/780/4179. "
            "Without this flag, count differences produce warnings."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (
        args.max_samples_per_split is not None
        and args.max_samples_per_split <= 0
    ):
        raise ValueError(
            "--max-samples-per-split must be positive"
        )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    if args.print_samples < 0:
        raise ValueError("--print-samples cannot be negative")

    splits = (
        ("train", "val", "test")
        if args.split == "all"
        else (args.split,)
    )

    for split in splits:
        split_path = args.embedding_root / split

        # Some pipelines save train.lmdb rather than train/.
        if not split_path.exists():
            lmdb_variant = (
                args.embedding_root / f"{split}.lmdb"
            )
            if lmdb_variant.exists():
                split_path = lmdb_variant

        inspect_split(
            split=split,
            path=split_path,
            max_samples=args.max_samples_per_split,
            batch_size=args.batch_size,
            print_samples=args.print_samples,
            enforce_counts=args.enforce_counts,
        )

    print("\n" + "=" * 70)
    print("All requested PennAction embedding checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()