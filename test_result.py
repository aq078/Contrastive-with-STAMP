from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.io import loadmat


LABEL_DIR = Path("dataset/Penn_Action/labels")

lengths = []
lengths_by_action = defaultdict(list)


for mat_path in sorted(LABEL_DIR.glob("*.mat")):
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    nframes = int(np.asarray(mat["nframes"]).squeeze())

    action_raw = np.asarray(mat["action"]).squeeze()
    if action_raw.ndim == 0:
        action = str(action_raw.item()).strip()
    elif action_raw.dtype.kind in {"U", "S"}:
        action = "".join(action_raw.tolist()).strip()
    else:
        action = str(action_raw).strip()

    lengths.append(nframes)
    lengths_by_action[action].append(nframes)


lengths = np.asarray(lengths)


print("=" * 60)
print("PennAction Sequence Length Statistics")
print("=" * 60)

print(f"Number of sequences: {len(lengths)}")
print(f"Min:    {lengths.min()} frames")
print(f"Max:    {lengths.max()} frames")
print(f"Mean:   {lengths.mean():.2f} frames")
print(f"Median: {np.median(lengths):.2f} frames")
print(f"Std:    {lengths.std():.2f} frames")

print("\nPercentiles:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p:2d}%: {np.percentile(lengths, p):7.2f} frames")


print("\n" + "=" * 60)
print("Length ranges")
print("=" * 60)

ranges = [
    (0, 31),
    (32, 63),
    (64, 95),
    (96, 127),
    (128, 159),
    (160, 191),
    (192, 255),
    (256, 511),
    (512, np.inf),
]

for low, high in ranges:
    if np.isinf(high):
        count = np.sum(lengths >= low)
        label = f">= {low}"
    else:
        count = np.sum((lengths >= low) & (lengths <= high))
        label = f"{low:3d}-{high:3d}"

    pct = 100 * count / len(lengths)
    print(f"{label:10s}: {count:4d} ({pct:6.2f}%)")


print("\n" + "=" * 60)
print("Relative to candidate window lengths")
print("=" * 60)

for window in [16, 32, 64, 128, 256]:
    shorter = np.sum(lengths < window)
    pct = 100 * shorter / len(lengths)

    print(
        f"< {window:3d} frames: "
        f"{shorter:4d}/{len(lengths)} ({pct:6.2f}%)"
    )


print("\n" + "=" * 60)
print("Per-action statistics")
print("=" * 60)

print(
    f"{'Action':22s} "
    f"{'N':>5s} "
    f"{'Mean':>8s} "
    f"{'Median':>8s} "
    f"{'Min':>6s} "
    f"{'Max':>6s}"
)

print("-" * 60)

for action in sorted(lengths_by_action):
    x = np.asarray(lengths_by_action[action])

    print(
        f"{action:22s} "
        f"{len(x):5d} "
        f"{x.mean():8.1f} "
        f"{np.median(x):8.1f} "
        f"{x.min():6d} "
        f"{x.max():6d}"
    )