import argparse
import os
import pickle
from collections import Counter

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def print_dict(d, indent=0):
    sp = " " * indent
    for k, v in d.items():
        if isinstance(v, float):
            print(f"{sp}{k}: {v:.4f}")
        else:
            print(f"{sp}{k}: {v}")


def confusion(y_true, y_pred, n_classes=2):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    return cm


def safe_get_label_pred_cols(df):
    """
    Try common column names produced by different pipelines.
    Returns (y_true_col, y_pred_col, prob_cols)
    """
    cols = list(df.columns)
    # common truth columns
    for yt in ["y_true", "label", "target", "y"]:
        if yt in cols:
            y_true_col = yt
            break
    else:
        y_true_col = None

    # common pred columns
    for yp in ["y_pred", "pred", "prediction", "yhat"]:
        if yp in cols:
            y_pred_col = yp
            break
    else:
        y_pred_col = None

    # probability columns (optional)
    prob_cols = [c for c in cols if "prob" in c.lower() or "logit" in c.lower() or c.lower().startswith("p_")]
    return y_true_col, y_pred_col, prob_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", default=".", help="Experiment directory containing results/ (default: current dir)")
    ap.add_argument("--save_csv", action="store_true", help="Save per-seed metrics CSV to results/summary_metrics.csv")
    args = ap.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    results_dir = os.path.join(exp_dir, "results")

    if not os.path.isdir(results_dir):
        raise SystemExit(f"Could not find results/ in: {exp_dir}")

    files = [
        "mean_performance_metrics.pkl",
        "std_performance_metrics.pkl",
        "performance_metrics_per_seed.pkl",
        "pred_df_per_seed.pkl",
        "extra_info_per_seed.pkl",
    ]

    print("=" * 80)
    print("Experiment directory:", exp_dir)
    print("Results directory:", results_dir)
    print("=" * 80)

    # ---- Mean/std metrics ----
    mean_path = os.path.join(results_dir, "mean_performance_metrics.pkl")
    std_path = os.path.join(results_dir, "std_performance_metrics.pkl")

    if os.path.exists(mean_path):
        mean_metrics = load_pkl(mean_path)
        print("\n[MEAN METRICS]")
        if isinstance(mean_metrics, dict):
            print_dict(mean_metrics, indent=2)
        else:
            print(type(mean_metrics), mean_metrics)
    else:
        mean_metrics = None
        print("\n[MEAN METRICS] missing")

    if os.path.exists(std_path):
        std_metrics = load_pkl(std_path)
        print("\n[STD METRICS]")
        if isinstance(std_metrics, dict):
            print_dict(std_metrics, indent=2)
        else:
            print(type(std_metrics), std_metrics)
    else:
        std_metrics = None
        print("\n[STD METRICS] missing")

    # ---- Per-seed metrics ----
    per_seed_path = os.path.join(results_dir, "performance_metrics_per_seed.pkl")
    per_seed = None
    if os.path.exists(per_seed_path):
        per_seed = load_pkl(per_seed_path)
        print("\n[PER-SEED METRICS]")
        if isinstance(per_seed, dict):
            for seed, md in per_seed.items():
                print(f"  seed={seed}")
                if isinstance(md, dict):
                    print_dict(md, indent=4)
                else:
                    print("    ", type(md), md)
        else:
            print(type(per_seed), per_seed)
    else:
        print("\n[PER-SEED METRICS] missing")

    # ---- Predictions ----
    pred_path = os.path.join(results_dir, "pred_df_per_seed.pkl")
    if os.path.exists(pred_path):
        pred_obj = load_pkl(pred_path)
        print("\n[PREDICTIONS]")
        print("Type:", type(pred_obj))

        # Often this is dict: seed -> dataframe
        if isinstance(pred_obj, dict):
            items = pred_obj.items()
        else:
            # sometimes a single dataframe
            items = [("single", pred_obj)]

        for seed, df in items:
            print("\n" + "-" * 80)
            print("Seed:", seed)
            if pd is not None and hasattr(df, "columns"):
                print("Columns:", list(df.columns))
                yt_col, yp_col, prob_cols = safe_get_label_pred_cols(df)

                if yt_col is None or yp_col is None:
                    print("Could not infer y_true/y_pred columns. Showing head():")
                    print(df.head())
                    continue

                y_true = df[yt_col].to_numpy()
                y_pred = df[yp_col].to_numpy()

                # Distribution checks
                ct_true = Counter(map(int, y_true))
                ct_pred = Counter(map(int, y_pred))
                print("y_true distribution:", dict(ct_true))
                print("y_pred distribution:", dict(ct_pred))

                # Confusion matrix
                n_classes = int(max(np.max(y_true), np.max(y_pred))) + 1 if len(y_true) else 2
                n_classes = max(n_classes, 2)
                cm = confusion(y_true, y_pred, n_classes=n_classes)
                print("Confusion matrix (rows=true, cols=pred):")
                print(cm)

                # Per-class recall
                recall = []
                for c in range(n_classes):
                    denom = cm[c, :].sum()
                    r = (cm[c, c] / denom) if denom > 0 else float("nan")
                    recall.append(r)
                print("Per-class recall:", {c: (f"{r:.4f}" if np.isfinite(r) else "nan") for c, r in enumerate(recall)})
                bal_acc = float(np.nanmean(recall))
                print(f"Balanced accuracy (from preds): {bal_acc:.4f}")

                if prob_cols:
                    print("Prob/logit columns detected:", prob_cols[:10])
            else:
                # Not a dataframe, just show structure
                if isinstance(df, dict):
                    print("Dict keys:", list(df.keys())[:20])
                else:
                    print("Object:", df)
    else:
        print("\n[PREDICTIONS] missing")

    # ---- Extra info ----
    extra_path = os.path.join(results_dir, "extra_info_per_seed.pkl")
    if os.path.exists(extra_path):
        extra = load_pkl(extra_path)
        print("\n[EXTRA INFO]")
        print("Type:", type(extra))
        if isinstance(extra, dict):
            print("Keys:", list(extra.keys())[:50])
    else:
        print("\n[EXTRA INFO] missing")

    # ---- Optional: save per-seed CSV ----
    if args.save_csv and pd is not None and isinstance(per_seed, dict):
        rows = []
        for seed, md in per_seed.items():
            if isinstance(md, dict):
                row = {"seed": seed}
                row.update(md)
                rows.append(row)
        if rows:
            out_df = pd.DataFrame(rows)
            out_path = os.path.join(results_dir, "summary_metrics.csv")
            out_df.to_csv(out_path, index=False)
            print("\nSaved per-seed metrics CSV to:", out_path)
        else:
            print("\nNo per-seed metrics rows to save.")
    elif args.save_csv and pd is None:
        print("\nCannot save CSV: pandas not installed.")


if __name__ == "__main__":
    main()