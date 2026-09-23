import os
import pickle
import numpy as np

EXPERIMENTS_DIR = "experiments/sere"

EXPERIMENT_NAMES = [
    "baseline_val_acc_MOMENT-1-large_nrs5_ne50_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0",

    "MOMENT-1-large_nrs5_ne80_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0_stage2_linear",
]


for experiment_name in EXPERIMENT_NAMES:

    pkl_path = os.path.join(
        EXPERIMENTS_DIR,
        experiment_name,
        "results",
        "extra_info_per_seed.pkl",
    )

    print("=" * 100)
    print(f"Experiment: {experiment_name}")

    if not os.path.exists(pkl_path):
        print(f"ERROR: File not found:\n{pkl_path}\n")
        continue

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    val_accs = []

    for seed, info in data.items():

        best_epoch = info["best_epoch"]
        val_acc_list = info["val_balanced_acc_list"]

        # Validation balanced accuracy at selected checkpoint
        val_acc = val_acc_list[best_epoch]

        val_accs.append(val_acc)

        print(
            f"Seed {seed}: "
            f"best epoch = {best_epoch}, "
            f"validation BA = {val_acc:.4f}"
        )

    val_accs = np.array(val_accs)

    mean = np.mean(val_accs)
    std = np.std(val_accs, ddof=0)

    print("-" * 100)
    print(f"Validation balanced accuracy: {mean:.4f} ± {std:.4f}")
    print()


print("\nSUMMARY")
print("=" * 100)

for experiment_name in EXPERIMENT_NAMES:

    pkl_path = os.path.join(
        EXPERIMENTS_DIR,
        experiment_name,
        "results",
        "extra_info_per_seed.pkl",
    )

    if not os.path.exists(pkl_path):
        continue

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    val_accs = [
        info["val_balanced_acc_list"][info["best_epoch"]]
        for info in data.values()
    ]

    print(
        f"{np.mean(val_accs):.4f} ± {np.std(val_accs, ddof=0):.4f}"
        f" | {experiment_name}"
    )