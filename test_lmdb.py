import argparse, ast, json, lmdb, pickle, random
from collections import Counter
import numpy as np

def load_keys(txn):
    raw = txn.get(b"__keys__")
    if raw is None:
        raise RuntimeError("No __keys__ found.")
    # pickle?
    try:
        return pickle.loads(raw)
    except Exception:
        pass
    s = raw.decode("utf-8", errors="replace").strip()
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except Exception:
            return ast.literal_eval(s)
    raise RuntimeError("Could not parse __keys__.")

def load_item(txn, key):
    kb = key.encode() if isinstance(key, str) else key
    raw = txn.get(kb)
    if raw is None:
        raise KeyError(f"Missing key {key}")
    return pickle.loads(raw)

def find_sample(item):
    for k in ["sample","x","data","features","inputs"]:
        if k in item:
            return np.asarray(item[k]), k
    return None, None

def find_mask(item):
    for k in ["mask","attention_mask","pad_mask","valid_mask"]:
        if k in item:
            m = np.asarray(item[k])
            if m.dtype != np.bool_:
                m = (m != 0)
            return m, k
    return None, None

def find_label(item, key_str):
    for k in ["label","y","target"]:
        if k in item:
            return int(np.asarray(item[k]).item()), k
    if "_y" in key_str:
        try:
            return int(key_str.split("_y")[-1]), "parsed_from_key"
        except Exception:
            pass
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb_dir", required=True, help="Directory containing data.mdb/lock.mdb")
    ap.add_argument("--n", type=int, default=200, help="How many items to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = lmdb.open(args.lmdb_dir, readonly=True, lock=False, readahead=False, max_readers=256)
    rng = random.Random(args.seed)

    with env.begin() as txn:
        keys_obj = load_keys(txn)

        # keys may be list OR dict of splits
        if isinstance(keys_obj, dict):
            print("Detected dict __keys__ with fields:", list(keys_obj.keys()))
            # If it already contains splits, summarize each split
            for split, keys in keys_obj.items():
                sample_keys = keys if len(keys) <= args.n else rng.sample(keys, args.n)
                summarize(txn, split, sample_keys)
        else:
            keys = keys_obj
            sample_keys = keys if len(keys) <= args.n else rng.sample(keys, args.n)
            summarize(txn, "ALL", sample_keys)

def summarize(txn, name, sample_keys):
    label_ctr = Counter()
    sample_shapes = Counter()
    mask_shapes = Counter()
    sample_nan = sample_inf = sample_all_nan = 0
    mask_missing = mask_all_false = mask_present = 0
    mask_true_frac_sum = 0.0
    sample_min = np.inf
    sample_max = -np.inf

    for k in sample_keys:
        k_str = k if isinstance(k, str) else k.decode("utf-8", errors="replace")
        item = load_item(txn, k)

        y, _ = find_label(item, k_str)
        if y is not None:
            label_ctr[y] += 1

        x, _ = find_sample(item)
        if x is not None:
            x = np.asarray(x)
            sample_shapes[tuple(x.shape)] += 1
            finite = np.isfinite(x)
            if np.isnan(x).any(): sample_nan += 1
            if np.isinf(x).any(): sample_inf += 1
            if np.isnan(x).all(): sample_all_nan += 1
            xf = x[finite]
            if xf.size:
                sample_min = min(sample_min, float(xf.min()))
                sample_max = max(sample_max, float(xf.max()))

        m, _ = find_mask(item)
        if m is None:
            mask_missing += 1
        else:
            mask_present += 1
            mask_shapes[tuple(m.shape)] += 1
            tf = float(m.mean())
            mask_true_frac_sum += tf
            if tf == 0.0:
                mask_all_false += 1

    print("\n" + "="*80)
    print("SPLIT/SET:", name)
    print("checked_items:", len(sample_keys))
    print("label_distribution:", dict(label_ctr))
    print("sample_shape_counts:", dict(sample_shapes))
    print("sample_items_with_nan:", sample_nan)
    print("sample_items_with_inf:", sample_inf)
    print("sample_items_all_nan:", sample_all_nan)
    print("sample_min_finite:", None if sample_min == np.inf else sample_min)
    print("sample_max_finite:", None if sample_max == -np.inf else sample_max)
    print("mask_missing_count:", mask_missing)
    print("mask_shape_counts:", dict(mask_shapes))
    print("mask_items_all_false:", mask_all_false)
    print("mask_mean_true_fraction:",
          None if mask_present == 0 else (mask_true_frac_sum / mask_present))

if __name__ == "__main__":
    main()
