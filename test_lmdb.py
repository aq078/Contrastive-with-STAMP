from types import SimpleNamespace
import torch
import numpy as np

from stamp.datasets import lmdb_pickle_dataset
from stamp.local import get_local_config


def main():
    local_config = get_local_config()

    dataset_name = "sere"
    processed_data_dir = local_config.processed_data_dirs[dataset_name]

    params = SimpleNamespace(
        dataset_name=dataset_name,
        dataset_dir=processed_data_dir,
        batch_size=4,
        return_mask=True,
        pad_to_len=64,
        reshape_data=True,
        orig_seq_len=64,
        embedding_model_name="MOMENT-1-large",
        tdr=1.0,
        seed=0,
        temporal_channel_selection=None,
    )

    dataset = lmdb_pickle_dataset.LoadDataset(params)
    loaders = dataset.get_data_loader()

    batch = next(iter(loaders["train"]))

    print("Number of returned items:", len(batch))

    if len(batch) == 5:
        x, y, token_labels, mask, metadata = batch
    else:
        raise RuntimeError(f"Expected 5 batch items, got {len(batch)}")

    print("\n=== Shapes ===")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("token_labels shape:", token_labels.shape)
    print("mask shape:", mask.shape)
    print("metadata length:", len(metadata))

    print("\n=== Dtypes ===")
    print("x dtype:", x.dtype)
    print("y dtype:", y.dtype)
    print("token_labels dtype:", token_labels.dtype)
    print("mask dtype:", mask.dtype)

    print("\n=== Basic checks ===")
    B = y.shape[0]
    expected_token_labels = 8 * 33 * 3

    assert token_labels.shape == (B, expected_token_labels), (
        f"Expected token_labels shape {(B, expected_token_labels)}, "
        f"got {token_labels.shape}"
    )

    assert x.shape[0] == B * 33 * 3, (
        f"Expected x first dim {B * 33 * 3}, got {x.shape[0]}"
    )

    assert mask.shape[0] == x.shape[0], (
        f"Expected mask first dim {x.shape[0]}, got {mask.shape[0]}"
    )

    print("Passed shape checks.")

    print("\n=== Label checks ===")
    unique_token_labels, counts = torch.unique(token_labels, return_counts=True)
    print("unique token labels:", unique_token_labels.tolist())
    print("counts:", counts.tolist())

    print("\nFirst sample window label:", y[0].item())
    print("First sample first 40 token labels:")
    print(token_labels[0, :40].tolist())

    print("\nFirst sample token labels reshaped as [T=8, S=99]:")
    print(token_labels[0].reshape(8, 99)[:, :10])

    print("\nMetadata example:")
    print(metadata[0])

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()