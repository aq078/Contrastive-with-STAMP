import os
import pickle
import numpy as np

# ============================================================
# EDIT THIS LIST EACH TIME
# ============================================================

EXPERIMENT_NAMES = [
    "baseline_val_acc_MOMENT-1-large_nrs5_ne50_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0",
    "MOMENT-1-large_nrs5_ne80_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0_stage2_linear"
]

EXPERIMENTS_DIR = "experiments/sere"


# ============================================================
# Extract validation balanced accuracy
# ============================================================

all_results = []

for experiment_name in EXPERIMENT_NAMES:

    pkl_path = os.path.join(
        EXPERIMENTS_DIR,
        experiment_name,
        "results",
        "extra_info_per_seed.pkl",
    )

    if not os.path.exists(pkl_path):
        print(f"[NOT FOUND] {experiment_name}")
        print(f"  Expected: {pkl_path}\n")
        continue

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    per_seed = []
    val_bas = []

    for seed, info in data.items():

        best_epoch = info["best_epoch"]
        val_ba_list = info["val_balanced_acc_list"]

        # Validation BA at the checkpoint selected by training
        val_ba = val_ba_list[best_epoch]

        per_seed.append((seed, best_epoch, val_ba))
        val_bas.append(val_ba)

    val_bas = np.asarray(val_bas, dtype=float)

    mean_ba = np.mean(val_bas)
    std_ba = np.std(val_bas, ddof=0)

    all_results.append({
        "experiment": experiment_name,
        "mean": mean_ba,
        "std": std_ba,
        "per_seed": per_seed,
    })


# ============================================================
# Detailed results
# ============================================================

for result in all_results:

    print("=" * 80)
    print(result["experiment"])
    print("-" * 80)

    for seed, best_epoch, val_ba in result["per_seed"]:
        print(
            f"seed={seed:<5} | "
            f"best_epoch={best_epoch:<3} | "
            f"val_balanced_accuracy={val_ba:.4f}"
        )

    print(
        f"\nValidation balanced accuracy: "
        f"{result['mean']:.4f} ± {result['std']:.4f}"
    )
    print()


# ============================================================
# Compact comparison
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION BALANCED ACCURACY SUMMARY")
print("=" * 80)

for result in all_results:
    print(
        f"{result['mean']:.4f} ± {result['std']:.4f}"
        f"  |  {result['experiment']}"
    )