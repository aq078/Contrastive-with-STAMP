import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from stamp.modeling.stamp import STAMP
from stamp.local import get_local_config


def load_exp_config(exp_dir: str):
    with open(Path(exp_dir) / "exp_config.json", "r") as f:
        return json.load(f)


def build_model_from_exp_config(exp_config: dict, device: str):
    p = exp_config["modeling_approach_config"]["params"]

    model = STAMP(
        input_dim=p["input_dim"],
        D=p["D"],
        n_temporal_channels=p["n_temporal_channels"],
        n_spatial_channels=p["n_spatial_channels"],
        encoder_aggregation=p["encoder_aggregation"],
        n_classes=p["n_classes"],
        initial_proj_params=p["initial_proj_params"],
        final_classifier_params=p["final_classifier_params"],
        use_batch_norm=p["use_batch_norm"],
        use_instance_norm=p["use_instance_norm"],
        pe_params=p["pe_params"],
        transformer_params=p["transformer_params"],
        gated_mlp_params=p["gated_mlp_params"],
        mhap_params=p["mhap_params"],
        n_cls_tokens=p.get("n_cls_tokens", None),
        supcon_params=p.get("supcon_params", None),
    )
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model


def open_embedding_lmdb(split_dir: str):
    env = lmdb.open(
        split_dir,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=256,
    )
    return env


def read_keys(env):
    with env.begin(write=False) as txn:
        keys_bytes = txn.get(b"__keys__")
        keys_str = json.loads(keys_bytes.decode())
    return keys_str


def fetch_embedding_sample(txn, key: str, n_temporal: int, n_spatial: int):
    data_bytes = txn.get(key.encode())
    x = np.frombuffer(data_bytes, dtype=np.float32)

    channel_product = n_temporal * n_spatial
    emb_dim = x.shape[0] // channel_product

    x = x.reshape(n_spatial, n_temporal, emb_dim)   # from embedding LMDB layout
    x = np.transpose(x, (1, 0, 2))                  # -> (T, S, C)
    return x


def parse_label_from_key(key: str, dataset_name: str):
    parts = key.split("_")
    label = int(parts[-1][1:])
    if dataset_name == "tuev":
        label -= 1
    return label


@torch.no_grad()
def extract_features_for_split(model, split_dir: str, dataset_name: str, n_temporal: int, n_spatial: int, device: str):
    env = open_embedding_lmdb(split_dir)
    keys = read_keys(env)

    X = []
    y = []

    with env.begin(write=False) as txn:
        for key in keys:
            arr = fetch_embedding_sample(txn, key, n_temporal, n_spatial)
            label = parse_label_from_key(key, dataset_name)

            x = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)  # (1,T,S,C)

            logits, features, _ = model(x, return_attention=False, return_features=True)
            feat = features.squeeze(0).detach().cpu().numpy()

            X.append(feat)
            y.append(label)

    env.close()
    return np.stack(X), np.array(y)


def main():
    local_config = get_local_config()

    dataset_name = "sere"

    stage1_exp_name = "MOMENT-1-large_nrs5_ne50_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0_stage1_supcon"
    seed = 654

    exp_dir = f"{local_config.tsfm_experiments_dir}/{dataset_name}/{stage1_exp_name}"
    exp_config = load_exp_config(exp_dir)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    p = exp_config["modeling_approach_config"]["params"]
    n_temporal = p["n_temporal_channels"]
    n_spatial = p["n_spatial_channels"]

    model = build_model_from_exp_config(exp_config, device=device)

    base_exp_name = stage1_exp_name.replace("_stage1_supcon", "")
    ckpt_path = f"checkpoints/{dataset_name}/MOMENT-1-large_nrs5_ne50_D128_ip-full-dr0.3_pe-basicNST_gmlp-tcriss_cross-nl8-dff256-dr0.3_cmconcat_nonrec_mhap-A4-dr0.3-Q8-qcweighted_sum_lres0.1_ls0.1gcT_inorm_tdr1.0/supcon/best.pth"
    model = load_checkpoint(model, ckpt_path, device=device)

    embeddings_root = f"embeddings/{dataset_name}/{exp_config['embedding_model_name']}"
    train_dir = f"{embeddings_root}/train"
    val_dir = f"{embeddings_root}/val"
    test_dir = f"{embeddings_root}/test"

    print("Extracting train features...")
    X_train, y_train = extract_features_for_split(
        model, train_dir, dataset_name, n_temporal, n_spatial, device
    )
    print("Extracting val features...")
    X_val, y_val = extract_features_for_split(
        model, val_dir, dataset_name, n_temporal, n_spatial, device
    )
    print("Extracting test features...")
    X_test, y_test = extract_features_for_split(
        model, test_dir, dataset_name, n_temporal, n_spatial, device
    )

    print("Feature shapes:")
    print("train:", X_train.shape, y_train.shape)
    print("val:  ", X_val.shape, y_val.shape)
    print("test: ", X_test.shape, y_test.shape)

    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    val_bacc = balanced_accuracy_score(y_val, val_pred)
    test_bacc = balanced_accuracy_score(y_test, test_pred)

    print("\nLogistic regression probe results")
    print(f"Val balanced acc:  {val_bacc:.4f}")
    print(f"Test balanced acc: {test_bacc:.4f}")

    print("\nVal confusion matrix:")
    print(confusion_matrix(y_val, val_pred))

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_pred))


if __name__ == "__main__":
    main()