from types import SimpleNamespace
import torch

from stamp.datasets import lmdb_np_dataset
from stamp.local import get_local_config


def main():
    local_config = get_local_config()

    dataset_name = "sere"
    embedding_model_name = "MOMENT-1-large"

    # This should point to embeddings/sere/MOMENT-1-large
    dataset_dir = f"embeddings/{dataset_name}/{embedding_model_name}"

    params = SimpleNamespace(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        batch_size=4,
        return_mask=True,
        pad_to_len=0,
        reshape_data=False,
        orig_seq_len=1024,
        tdr=1.0,
        seed=0,
    )

    dataset = lmdb_np_dataset.LoadDataset(params)
    loaders = dataset.get_data_loader()

    batch = next(iter(loaders["train"]))

    print("Number of returned items:", len(batch))

    if len(batch) == 5:
        x, y, token_labels, mask, keys = batch
    else:
        raise RuntimeError(f"Expected 5 items, got {len(batch)}")

    print("\n=== Shapes ===")
    print("x:", x.shape)
    print("y:", y.shape)
    print("token_labels:", token_labels.shape)
    print("mask:", mask.shape)
    print("num keys:", len(keys))

    print("\n=== Dtypes ===")
    print("x dtype:", x.dtype)
    print("y dtype:", y.dtype)
    print("token_labels dtype:", token_labels.dtype)
    print("mask dtype:", mask.dtype)

    print("\n=== Values ===")
    print("y:", y.tolist())
    print("unique token labels:", torch.unique(token_labels, return_counts=True))
    print("first key:", keys[0])
    print("first sample token labels shape:", token_labels[0].shape)
    print("first sample first 40 token labels:", token_labels[0, :40].tolist())

    print("\n=== Reshape sanity ===")
    B = y.shape[0]
    expected_T = 8
    expected_S = 99
    expected_D = 1024
    expected_N = expected_T * expected_S

    assert x.shape == (B, expected_T, expected_S, expected_D), (
        f"Expected x shape {(B, expected_T, expected_S, expected_D)}, got {tuple(x.shape)}"
    )

    assert token_labels.shape == (B, expected_N), (
        f"Expected token_labels shape {(B, expected_N)}, got {tuple(token_labels.shape)}"
    )

    print("token_labels reshaped [T=8, S=99], first sample first 10 spatial labels per temporal token:")
    print(token_labels[0].reshape(expected_T, expected_S)[:, :10])

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()