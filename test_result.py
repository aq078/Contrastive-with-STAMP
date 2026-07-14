import lmdb
import pickle
from collections import Counter

LMDB_PATH = "dataset/processed_sere/sere_framecomp_world_xyz_L64_S16.lmdb"


def print_stats(split_name, labels):
    counter = Counter(labels)
    total = len(labels)

    print(f"\n=== {split_name.upper()} ===")
    print(f"Total samples: {total}")

    for cls in sorted(counter.keys()):
        n = counter[cls]
        print(f"Class {cls}: {n} ({100*n/total:.2f}%)")

    if 0 in counter and 1 in counter:
        ratio = counter[1] / counter[0]
        print(f"Positive/Negative ratio: {ratio:.3f}")

        # Suggested CE weights
        total = counter[0] + counter[1]
        w0 = total / (2 * counter[0])
        w1 = total / (2 * counter[1])

        print("\nSuggested CrossEntropy weights:")
        print(f"weight = [{w0:.4f}, {w1:.4f}]")


env = lmdb.open(
    LMDB_PATH,
    readonly=True,
    lock=False,
    readahead=False,
)

with env.begin() as txn:
    dataset_keys = pickle.loads(txn.get(b"__keys__"))

    for split in ["train", "val", "test"]:
        labels = []

        for key in dataset_keys[split]:
            data = pickle.loads(txn.get(key.encode()))
            labels.append(int(data["label"]))

        print_stats(split, labels)

env.close()