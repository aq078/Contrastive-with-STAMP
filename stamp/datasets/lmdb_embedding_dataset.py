import json
import os
import pickle

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from CBraMod.utils.util import to_tensor
from stamp.datasets.utils import get_dataset_params


class CustomLMDBEmbeddingDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        data_dir,
        mode,
        n_temporal_channels,
        n_spatial_channels,
        tdr,
        seed,
        temporal_channel_selection=None,
    ):
        super(CustomLMDBEmbeddingDataset, self).__init__()

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.mode = mode
        self.n_temporal_channels = n_temporal_channels
        self.n_spatial_channels = n_spatial_channels
        self.channel_product = (
            self.n_temporal_channels
            * self.n_spatial_channels
        )
        self.tdr = tdr if tdr is not None else 1.0
        self.seed = seed
        self.temporal_channel_selection = (
            temporal_channel_selection
        )

        print(
            "Temporal channel selection: "
            f"{self.temporal_channel_selection}"
        )

        self.db = None
        self.sample_metadata = None

        self._load_sample_metadata()
        self._init_keys()

    def _split_dir(self):
        return os.path.join(
            self.data_dir,
            self.mode,
        )

    def _load_sample_metadata(self):
        """
        Load the metadata sidecar produced by the modified embeddings.py.

        Expected path:
            <embedding_data_dir>/<mode>/sample_metadata.pkl

        Expected structure:
            {
                sample_key: {
                    "sample_key": ...,
                    "sequence_id": ...,
                    "nframes": ...,
                    ...
                }
            }
        """
        metadata_path = os.path.join(
            self._split_dir(),
            "sample_metadata.pkl",
        )

        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as file:
                self.sample_metadata = pickle.load(file)

            if not isinstance(self.sample_metadata, dict):
                raise TypeError(
                    "Expected sample metadata to be a dictionary, "
                    f"got {type(self.sample_metadata).__name__}"
                )

            print(
                f"Loaded metadata for "
                f"{len(self.sample_metadata)} samples from "
                f"{metadata_path}"
            )

        elif self.mode == "test":
            print(
                "WARNING: test metadata sidecar was not found: "
                f"{metadata_path}"
            )

    def _init_keys(self):
        """Initialize keys from the LMDB database."""
        temp_db = lmdb.open(
            self._split_dir(),
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )

        with temp_db.begin(write=False) as txn:
            keys_bytes = txn.get(b"__keys__")

            if keys_bytes is None:
                raise RuntimeError(
                    f"{self._split_dir()} does not contain __keys__"
                )

            keys_str = json.loads(
                keys_bytes.decode()
            )
            self.keys = [
                key.encode()
                for key in keys_str
            ]

            if (
                self.tdr < 1.0
                and self.mode == "train"
            ):
                print(
                    "Using training data ratio of "
                    f"{self.tdr}"
                )

                length = len(self.keys)
                rng = np.random.default_rng(
                    self.seed
                )
                rng.shuffle(self.keys)
                self.keys = self.keys[
                    :int(length * self.tdr)
                ]

            self.length = len(self.keys)

            print(
                f"Loaded {self.length} keys "
                "from stored __keys__"
            )

        temp_db.close()

    def _get_db(self):
        """Lazy DB initialization for each DataLoader worker."""
        if self.db is None:
            self.db = lmdb.open(
                self._split_dir(),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=1024,
            )

        return self.db

    def _init_db(self):
        """Explicitly initialize the LMDB connection."""
        self.db = lmdb.open(
            self._split_dir(),
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )

        with self.db.begin(write=False) as txn:
            keys_bytes = txn.get(b"__keys__")

            if keys_bytes is None:
                raise RuntimeError(
                    f"{self._split_dir()} does not contain __keys__"
                )

            keys_str = json.loads(
                keys_bytes.decode()
            )
            self.keys = [
                key.encode()
                for key in keys_str
            ]
            self.length = len(self.keys)

            print(
                f"Loaded {self.length} keys "
                "from stored __keys__"
            )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return idx

    @staticmethod
    def _metadata_key_candidates(
        embedding_key,
    ):
        """
        Generate possible source-metadata keys for an embedding LMDB key.

        Example:
            embedding key:
                0001_w0000_00000000_y0

            candidates:
                0001_w0000_00000000_y0
                0001_w0000_00000000
                0001_w0000
        """
        candidates = [embedding_key]

        without_label = embedding_key
        if "_y" in without_label:
            without_label = without_label.rsplit(
                "_y",
                1,
            )[0]

        if without_label not in candidates:
            candidates.append(without_label)

        # LMDBWriter may append a unique numeric suffix after the
        # original PennAction window key. Preserve only:
        #     <sequence_id>_w<window_start>
        if "_w" in without_label:
            sequence_prefix, window_suffix = (
                without_label.split(
                    "_w",
                    1,
                )
            )
            window_start = window_suffix.split(
                "_",
                1,
            )[0]
            source_key = (
                f"{sequence_prefix}_w{window_start}"
            )

            if source_key not in candidates:
                candidates.append(source_key)

        return candidates

    def _lookup_metadata(self, embedding_key):
        if self.sample_metadata is None:
            raise RuntimeError(
                "Sequence-level test diagnostics require "
                "sample_metadata.pkl, but no metadata sidecar "
                f"was loaded for {self.mode!r}."
            )

        candidate_keys = (
            self._metadata_key_candidates(
                embedding_key
            )
        )

        for candidate_key in candidate_keys:
            metadata = self.sample_metadata.get(
                candidate_key
            )

            if metadata is not None:
                metadata = dict(metadata)
                metadata.setdefault(
                    "sample_key",
                    candidate_key,
                )
                metadata[
                    "embedding_sample_key"
                ] = embedding_key
                metadata[
                    "matched_metadata_key"
                ] = candidate_key
                return metadata

        raise KeyError(
            "No metadata found for embedding key "
            f"{embedding_key!r}. Tried: "
            f"{candidate_keys}. "
            "Example available metadata keys: "
            f"{list(self.sample_metadata.keys())[:5]}"
        )

    def collate(self, batch_indices):
        x_data = []
        y_labels = []
        sample_keys = []
        sample_metadata = []

        db = self._get_db()

        with db.begin(write=False) as txn:
            for idx in batch_indices:
                key = self.keys[idx]
                data_bytes = txn.get(key)

                if data_bytes is None:
                    raise IndexError(
                        f"Sample {idx} not found"
                    )

                sample = np.frombuffer(
                    data_bytes,
                    dtype=np.float32,
                )
                x_data.append(sample)

                key_str = key.decode()
                parts = key_str.split("_")
                label = int(
                    parts[-1][1:]
                )

                if self.dataset_name == "tuev":
                    label = label - 1

                y_labels.append(label)
                sample_keys.append(key_str)

                if self.mode == "test":
                    sample_metadata.append(
                        self._lookup_metadata(
                            key_str
                        )
                    )

        x_data = np.stack(
            x_data,
            axis=0,
        )
        y_label = np.asarray(
            y_labels,
            dtype=np.int64,
        )
        sample_keys = np.asarray(
            sample_keys,
            dtype=object,
        )

        if (
            x_data.shape[1]
            % self.channel_product
            != 0
        ):
            raise ValueError(
                "Embedding size is not divisible by "
                "n_temporal_channels * "
                "n_spatial_channels: "
                f"flat_size={x_data.shape[1]}, "
                f"channel_product={self.channel_product}"
            )

        embedding_dim = (
            x_data.shape[1]
            // self.channel_product
        )

        # [B, T*S*D] -> [B, S*T, D]
        x_data = x_data.reshape(
            x_data.shape[0],
            self.channel_product,
            embedding_dim,
        )

        # [B, S*T, D] -> [B, S, T, D]
        x_data = x_data.reshape(
            x_data.shape[0],
            self.n_spatial_channels,
            self.n_temporal_channels,
            embedding_dim,
        )

        # [B, S, T, D] -> [B, T, S, D]
        x_data = x_data.transpose(
            0,
            2,
            1,
            3,
        )

        if (
            self.temporal_channel_selection
            is not None
        ):
            x_data = x_data[
                :,
                self.temporal_channel_selection,
                :,
                :,
            ]

        x_data = to_tensor(x_data)
        y_label = to_tensor(
            y_label
        ).long()

        if self.mode == "test":
            return (
                x_data,
                y_label,
                sample_keys,
                sample_metadata,
            )

        return (
            x_data,
            y_label,
            sample_keys,
        )

    def __getstate__(self):
        """
        Do not pickle an open LMDB environment into worker processes.
        """
        state = self.__dict__.copy()
        state["db"] = None
        return state


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.dataset_dir = params.dataset_dir

        self.dataset_params = (
            get_dataset_params(
                dataset_name=params.dataset_name
            )
        )

        self.n_temporal_channels = (
            self.dataset_params[
                "n_temporal_channels"
            ]
        )
        self.n_spatial_channels = (
            self.dataset_params[
                "n_spatial_channels"
            ]
        )

        self.num_workers = (
            params.num_workers
            if hasattr(params, "num_workers")
            else 4
        )
        self.prefetch_factor = (
            params.prefetch_factor
            if hasattr(params, "prefetch_factor")
            else 2
        )
        self.tdr = (
            params.tdr
            if hasattr(params, "tdr")
            else 1.0
        )
        self.temporal_channel_selection = (
            params.temporal_channel_selection
            if hasattr(
                params,
                "temporal_channel_selection",
            )
            else None
        )

        if hasattr(self.params, "seed"):
            self.dataloader_rng = (
                torch.Generator()
            )
            self.dataloader_rng.manual_seed(
                params.seed
            )
        else:
            print(
                "WARNING: Seed was not given, "
                "so train generator will not be set."
            )

        self._cached_sample_orders = {}

    def _dataset(self, mode):
        return CustomLMDBEmbeddingDataset(
            dataset_name=self.params.dataset_name,
            data_dir=self.dataset_dir,
            mode=mode,
            n_temporal_channels=(
                self.n_temporal_channels
            ),
            n_spatial_channels=(
                self.n_spatial_channels
            ),
            tdr=self.tdr,
            seed=(
                self.params.seed
                if hasattr(
                    self.params,
                    "seed",
                )
                else None
            ),
            temporal_channel_selection=(
                self.temporal_channel_selection
            ),
        )

    def _loader_worker_params(self):
        if self.num_workers <= 0:
            return {
                "num_workers": 0,
                "pin_memory": True,
            }

        return {
            "num_workers": self.num_workers,
            "pin_memory": True,
            "prefetch_factor": (
                self.prefetch_factor
            ),
            "persistent_workers": True,
        }

    def get_data_loader(self):
        train_set = self._dataset("train")
        val_set = self._dataset("val")
        test_set = self._dataset("test")

        print(
            len(train_set),
            len(val_set),
            len(test_set),
        )
        print(
            len(train_set)
            + len(val_set)
            + len(test_set)
        )

        worker_params = (
            self._loader_worker_params()
        )

        data_loader = {
            "train": DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                generator=(
                    self.dataloader_rng
                    if hasattr(
                        self.params,
                        "seed",
                    )
                    else None
                ),
                **worker_params,
            ),
            "val": DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                **worker_params,
            ),
            "test": DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                **worker_params,
            ),
        }

        return data_loader