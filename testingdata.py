import pickle

path = "experiments/sere/MOMENT-1-large_nrs5_ne80_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0_stage2_linear/results/performance_metrics_per_seed.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

for seed, metrics in data.items():
    print(f"Seed {seed}: balanced_accuracy = {metrics['balanced_accuracy']:.4f}")
    
import pickle

mean_path = "experiments/sere/MOMENT-1-large_nrs5_ne80_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0_stage2_linear/results/mean_performance_metrics.pkl"
std_path  = "experiments/sere/MOMENT-1-large_nrs5_ne80_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0_stage2_linear/results/std_performance_metrics.pkl"

mean = pickle.load(open(mean_path, "rb"))
std  = pickle.load(open(std_path, "rb"))

print(f"Balanced Accuracy: {mean['balanced_accuracy']:.4f} ± {std['balanced_accuracy']:.4f}")