from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat


LABEL_DIR = Path("dataset/Penn_Action/labels")


split_counts = Counter()
action_counts = Counter()
action_split_counts = defaultdict(Counter)
frame_lengths = []

for mat_path in sorted(LABEL_DIR.glob("*.mat")):
    mat = loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
    )

    action = str(mat["action"]).strip()
    train_flag = int(mat["train"])
    nframes = int(mat["nframes"])

    split_counts[train_flag] += 1
    action_counts[action] += 1
    action_split_counts[action][train_flag] += 1
    frame_lengths.append(nframes)


print("Split flag counts:")
for flag, count in sorted(split_counts.items()):
    print(f"  {flag}: {count}")

print("\nTotal action counts:")
for action, count in sorted(action_counts.items()):
    print(f"  {action:20s}: {count}")

print("\nAction counts by split:")
for action in sorted(action_split_counts):
    counts = action_split_counts[action]

    print(
        f"  {action:20s} "
        f"train=1: {counts[1]:4d}  "
        f"test=-1: {counts[-1]:4d}"
    )

frame_lengths = np.asarray(frame_lengths)

print("\nSequence-length statistics:")
print("  count:", len(frame_lengths))
print("  min:", frame_lengths.min())
print("  max:", frame_lengths.max())
print("  mean:", frame_lengths.mean())
print("  median:", np.median(frame_lengths))
print("  < 16 frames:", np.sum(frame_lengths < 16))
print("  < 32 frames:", np.sum(frame_lengths < 32))
print("  < 64 frames:", np.sum(frame_lengths < 64))
mat = loadmat(
    "dataset/Penn_Action/labels/0001.mat",
    squeeze_me=True,
    struct_as_record=False,
)

print("dimensions:", mat["dimensions"])
print("first 10 bbox rows:")
print(mat["bbox"][:10])

print("\nFirst-frame joint coordinate ranges:")
print("x min/max:", mat["x"][0].min(), mat["x"][0].max())
print("y min/max:", mat["y"][0].min(), mat["y"][0].max())
print("bbox first row:", mat["bbox"][0])