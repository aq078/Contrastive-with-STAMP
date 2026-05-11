import pickle
import matplotlib.pyplot as plt


def load_first_seed_curve(pkl_path, key):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    first_seed = list(data.keys())[0]  # 👈 take first seed
    return data[first_seed][key]


# ======== CHANGE THESE ========
exp1_path = "experiments/sere/baseline_val_acc_MOMENT-1-large_nrs5_ne50_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0/results/extra_info_per_seed.pkl"
exp2_path = "experiments/sere/MOMENT-1-large_nrs5_ne80_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0_stage2_linear/results/extra_info_per_seed.pkl"

label1 = "Baseline"
label2 = "SupCon"
# =============================


# -------- TRAIN --------
train1 = load_first_seed_curve(exp1_path, "train_balanced_acc_list")
train2 = load_first_seed_curve(exp2_path, "train_balanced_acc_list")

plt.figure(figsize=(8, 5))
plt.plot(train1, label=label1)
plt.plot(train2, label=label2)
plt.axvline(x=190, color='r', linestyle='--', label='best_epoch')
plt.xlabel("Epoch")
plt.ylabel("Train Balanced Accuracy")
plt.title("Train Comparison (First Seed)")
plt.legend()
plt.grid()
plt.ylim(0.5, 1.0)  # optional
plt.savefig("compare_train_seed0.png")
plt.close()


# -------- VAL --------
val1 = load_first_seed_curve(exp1_path, "val_balanced_acc_list")
val2 = load_first_seed_curve(exp2_path, "val_balanced_acc_list")

plt.figure(figsize=(8, 5))
plt.plot(val1, label=label1)
plt.plot(val2, label=label2)
plt.axvline(x=190, color='r', linestyle='--', label='best_epoch')
plt.xlabel("Epoch")
plt.ylabel("Validation Balanced Accuracy")
plt.title("Validation Comparison (First Seed)")
plt.legend()
plt.grid()
plt.ylim(0.5, 1.0)  # optional
plt.savefig("compare_val_seed0.png")
plt.close()