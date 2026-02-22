import argparse
import lmdb, pickle, random
import numpy as np

def get_keys(txn):
    # STAMP-style: __keys__ stored as pickled dict or list
    raw = txn.get(b"__keys__")
    if raw is None:
        raise RuntimeError("No __keys__ found. This LMDB doesn't look like STAMP format.")
    keys_obj = pickle.loads(raw)
    return keys_obj

def load_item(txn, key):
    if isinstance(key, str):
        k = key.encode()
    else:
        k = key
    raw = txn.get(k)
    if raw is None:
        raise KeyError(f"Missing key {key}")
    return pickle.loads(raw)

def find_embedding_array(obj):
    # Common field names across pipelines
    for candidate in ["embeddings", "embedding", "sample", "x", "features"]:
        if candidate in obj:
            arr = obj[candidate]
            try:
                arr = np.asarray(arr)
                if arr.size > 0:
                    return arr, candidate
            except Exception:
                pass
    # Sometimes nested
    if "data" in obj:
        try:
            arr = np.asarray(obj["data"])
            if arr.size > 0:
                return arr, "data"
        except Exception:
            pass
    raise KeyError(f"Couldn't find embedding array field in item keys={list(obj.keys())}")

def summarize_split(split_dir, max_items=2000, seed=0):
    env = lmdb.open(split_dir, readonly=True, lock=False, readahead=False, max_readers=256)
    rng = random.Random(seed)

    with env.begin() as txn:
        keys_obj = get_keys(txn)

        # keys_obj may be dict with split keys OR a flat list (already split per LMDB)
        if isinstance(keys_obj, dict):
            # If someone stored dict even inside split LMDB, try common names
            if "train" in keys_obj or "val" in keys_obj or "test" in keys_obj:
                # choose the first available split list
                key_list = next(iter(keys_obj.values()))
            else:
                key_list = list(keys_obj.values())[0]
        else:
            key_list = keys_obj

        n = len(key_list)
        if n == 0:
            return {"n": 0}

        sample_keys = key_list
        if max_items and n > max_items:
            sample_keys = rng.sample(key_list, max_items)

        # Streaming stats
        count = 0
        nan_count = 0
        inf_count = 0
        min_v = np.inf
        max_v = -np.inf
        sum_v = 0.0
        sumsq_v = 0.0
        field_name = None
        shape_example = None

        for k in sample_keys:
            obj = load_item(txn, k)
            arr, fname = find_embedding_array(obj)
            field_name = field_name or fname
            shape_example = shape_example or tuple(arr.shape)

            finite_mask = np.isfinite(arr)
            nan_count += int(np.isnan(arr).any())
            inf_count += int(np.isinf(arr).any())

            arr_f = arr[finite_mask]
            if arr_f.size == 0:
                continue

            min_v = min(min_v, float(arr_f.min()))
            max_v = max(max_v, float(arr_f.max()))
            sum_v += float(arr_f.sum())
            sumsq_v += float((arr_f * arr_f).sum())
            count += int(arr_f.size)

        mean = sum_v / count if count else float("nan")
        var = (sumsq_v / count - mean * mean) if count else float("nan")
        std = float(np.sqrt(max(var, 0.0))) if np.isfinite(var) else float("nan")

        return {
            "n_items_total": n,
            "n_items_sampled": len(sample_keys),
            "field": field_name,
            "shape_example": shape_example,
            "n_values": count,
            "min": min_v,
            "max": max_v,
            "mean": mean,
            "std": std,
            "items_with_nan": nan_count,
            "items_with_inf": inf_count,
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True,
                    help="Parent directory containing train/val/test LMDB env dirs")
    ap.add_argument("--max_items", type=int, default=2000)
    args = ap.parse_args()

    for split in ["train", "val", "test"]:
        split_dir = f"{args.emb_dir.rstrip('/')}/{split}"
        try:
            stats = summarize_split(split_dir, max_items=args.max_items)
            print(f"\n== {split} ==")
            for k, v in stats.items():
                print(f"{k}: {v}")
        except Exception as e:
            print(f"\n== {split} ==")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
