#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path("dataset/SERE_dataset_SHAREABLE_skeletons")
FRAME_LABEL_DIR = ROOT_DIR / "Labels" / "frame_level" / "compensation"

EXERCISES = ["E1", "E2", "E3", "E4", "E5"]
FRAME_LABEL_FILENAME_FMT = "{ex}_frame_labels.csv"

PID_COL = "pid"
AFFECTED_COL = "affected"
FRAME_COL = "frame"
LABEL_COL = "comp"

MERGE_GAP = 3  # set 0 to disable merging


def lower_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_frame_labels(path):
    df = lower_cols(pd.read_csv(path))
    df = df[[PID_COL, AFFECTED_COL, FRAME_COL, LABEL_COL]].copy()

    df[FRAME_COL] = pd.to_numeric(df[FRAME_COL], errors="coerce")
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

    df = df.dropna()
    df[FRAME_COL] = df[FRAME_COL].astype(int)
    return df


def get_runs(frames, labels):
    runs = []
    start = None
    prev_f = None

    for f, y in zip(frames, labels):
        y = int(y >= 0.5)

        if y == 1:
            if start is None:
                start = f
            elif f != prev_f + 1:
                runs.append((start, prev_f))
                start = f
        else:
            if start is not None:
                runs.append((start, prev_f))
                start = None

        prev_f = f

    if start is not None:
        runs.append((start, prev_f))

    return runs


def merge_runs(runs, max_gap):
    if max_gap == 0 or len(runs) <= 1:
        return runs

    merged = [runs[0]]

    for s, e in runs[1:]:
        ps, pe = merged[-1]
        gap = s - pe - 1

        if gap <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    return merged


def lengths(runs):
    return [e - s + 1 for s, e in runs]


def summarize(name, arr):
    if len(arr) == 0:
        print(f"{name}: empty")
        return

    arr = np.array(arr)
    print(f"\n{name}")
    print(f"count:  {len(arr)}")
    print(f"mean:   {arr.mean():.2f}")
    print(f"median: {np.median(arr):.2f}")
    print(f"p90:    {np.percentile(arr, 90):.2f}")
    print(f"p95:    {np.percentile(arr, 95):.2f}")
    print(f"max:    {arr.max()}")

def plot_hist_log(lengths, title, save_path):
    lengths = np.array(lengths)

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=50)
    plt.yscale("log")  # 👈 key
    plt.xlabel("Event length (frames)")
    plt.ylabel("Count (log scale)")
    plt.title(title)
    plt.grid()

    plt.savefig(save_path)
    plt.close()
def plot_sorted_lengths_log(lengths, title, save_path):
    lengths = np.sort(np.array(lengths))

    plt.figure(figsize=(8, 5))
    plt.plot(lengths)
    plt.yscale("log")  # 👈 important
    plt.xlabel("Event index (sorted)")
    plt.ylabel("Event length (frames, log)")
    plt.title(title)
    plt.grid()

    plt.savefig(save_path)
    plt.close()


def main():
    all_raw = []
    all_merged = []

    for ex in EXERCISES:
        path = FRAME_LABEL_DIR / FRAME_LABEL_FILENAME_FMT.format(ex=ex)
        print(f"\n=== {ex} ===")

        df = load_frame_labels(path)

        ex_raw = []
        ex_merged = []

        for (pid, aff), g in df.groupby([PID_COL, AFFECTED_COL]):
            g = g.sort_values(FRAME_COL)

            frames = g[FRAME_COL].to_numpy()
            labels = g[LABEL_COL].to_numpy()

            runs = get_runs(frames, labels)
            merged = merge_runs(runs, MERGE_GAP)

            ex_raw.extend(lengths(runs))
            ex_merged.extend(lengths(merged))

        summarize(f"{ex} raw", ex_raw)
        summarize(f"{ex} merged", ex_merged)

        all_raw.extend(ex_raw)
        all_merged.extend(ex_merged)

    summarize("ALL raw", all_raw)
    summarize("ALL merged", all_merged)

    
plot_sorted_lengths(all_raw, "Sorted raw event lengths", "sorted_raw.png")
plot_sorted_lengths(all_merged, "Sorted merged event lengths", "sorted_merged.png")
if __name__ == "__main__":
    main()