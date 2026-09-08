import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lmdb
import pickle
import pandas as pd
from functools import partial
from CBraMod.utils.util import to_tensor
from stamp.datasets.utils import get_dataset_params

class CustomLMDBPickleDataset(Dataset):
    def __init__(
            self,
            dataset_name,
            data_dir,
            mode,
            tdr,
            seed,
            pad_to_len=0,
            reshape_data=False,
            check_reshaped_data=False,
            temporal_channel_selection:list[int]=None
    ):
        super(CustomLMDBPickleDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.pad_to_len = pad_to_len
        self.reshape_data = reshape_data
        self.check_reshaped_data = check_reshaped_data
        self.mode = mode
        self.tdr = tdr if tdr is not None else 1.0
        self.seed = seed
        self.temporal_channel_selection = temporal_channel_selection

        valid_dataset_names = [
            'bciciv2a',
            'faced',
            'mumtaz',
            'physio',
            'seedv',
            'seedvig',
            'shu',
            'speech',
            'stress',
            'sere',
            'sere_video_comp',
            'sere_video_rom',
            'sere_video_smooth',
            'sere_video_spast',
            'sere_event_comp',
            'sere_comp_hybrid',
            'Penn_Action'
        ]

        assert self.dataset_name in valid_dataset_names, f'Given dataset name, {self.dataset_name}, is not in {valid_dataset_names}.'

        # Initialize LMDB connection
        self._init_db()

    def _init_db(self):
        """Initialize LMDB database connection"""
        self.db = lmdb.open(self.data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[self.mode]
            if self.tdr < 1.0 and self.mode == 'train':
                print(f"Using training data ratio of {self.tdr}")
                length = len(self.keys)
                # Shuffle keys
                rng = np.random.default_rng(self.seed)
                rng.shuffle(self.keys)
                self.keys = self.keys[:int(length * self.tdr)]

    def __getstate__(self):
        """Prepare object for pickling - remove unpicklable LMDB connection"""
        state = self.__dict__.copy()
        # Remove the LMDB database connection
        if 'db' in state:
            del state['db']
        return state

    def __setstate__(self, state):
        """Restore object after unpickling - recreate LMDB connection"""
        self.__dict__.update(state)
        # Recreate the LMDB database connection
        self._init_db()

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]

        with self.db.begin(write=False) as txn:
            pair_bytes = txn.get(key.encode())

        if pair_bytes is None:
            raise KeyError(
                f"Sample key {key!r} was not found in LMDB."
            )

        pair = pickle.loads(pair_bytes)

        if 'sample' not in pair or 'label' not in pair:
            raise KeyError(
                f"Sample {key!r} must contain 'sample' and 'label'. "
                f"Available keys: {sorted(pair.keys())}"
            )

        data = pair['sample']
        label = pair['label']
        sample_key = key

        original_metadata = pair.get('meta', {})
        if original_metadata is None:
            original_metadata = {}

        if not isinstance(original_metadata, dict):
            raise TypeError(
                f"Sample {key!r} has non-dict metadata of type "
                f"{type(original_metadata).__name__}."
            )

        original_metadata = dict(original_metadata)
        original_metadata.setdefault(
            'sample_key',
            sample_key,
        )
        original_metadata.setdefault(
            'label',
            int(label),
        )

        return (
            data / 100,
            label,
            sample_key,
            original_metadata,
        )

    def collate(self, batch, embedding_model_name=None):
        x_data = np.array(
            [x[0] for x in batch]
        )
        y_label = np.array(
            [x[1] for x in batch]
        )
        sample_keys = [
            x[2] for x in batch
        ]
        sample_metadata = [
            x[3] for x in batch
        ]

        # print(f'x_data shape: {x_data.shape}')
        # print(f'y_label shape: {y_label.shape}')
        # Handle padding
        if self.pad_to_len and x_data.shape[-1] < self.pad_to_len:
            pad_width = self.pad_to_len - x_data.shape[-1]
            x_data = np.pad(x_data, pad_width=((0, 0), (0, 0), (0, 0), (0, pad_width)), mode='constant') # Pad to (batch_size, n_spatial_channels, n_temporal_channels, pad_to_len)
            if self.check_reshaped_data:
                orig = x_data.copy()

        if self.temporal_channel_selection is not None:
            # Apply temporal channel selection
            x_data = x_data[:, :, self.temporal_channel_selection, :] # Shape: (batch_size, n_spatial_channels, selected_n_temporal, seq_len)
            # Update n_temporal_channels
            self.n_temporal_channels = len(self.temporal_channel_selection)

        batch_size, n_spatial_channels, n_temporal_channels, seq_len = x_data.shape
        if self.reshape_data:
            x_data = x_data.reshape(-1, x_data.shape[-1]) # Reshape to (batch_size * n_spatial_channels * n_temporal_channels, seq_len)
            x_data = np.expand_dims(x_data, axis=1) # Shape: (batch_size * n_spatial_channels * n_temporal_channels, 1, seq_len)

            # Make sure that orig and x_data match up correctly in values
            if self.check_reshaped_data:
                for i in range(batch_size):
                    for j in range(n_spatial_channels):
                        for k in range(n_temporal_channels):
                            assert np.all(orig[i, j, k, :] == x_data[i * n_spatial_channels * n_temporal_channels + j * n_temporal_channels + k, :]), f"Mismatch at {i}, {j}, {k}"

        trial_metadata = self._create_metadata_vectorized(
            batch_size=batch_size,
            n_spatial_channels=n_spatial_channels,
            n_temporal_channels=n_temporal_channels,
            sample_keys=sample_keys,
            y_label=y_label,
            sample_metadata=sample_metadata,
        )

        if embedding_model_name and 'chronos' in embedding_model_name.lower():
            # Squeeze dim 1 for Chronos models
            x_data = np.squeeze(x_data, axis=1)  # Shape: (batch_size * n_spatial_channels * n_temporal_channels, seq_len)

        if embedding_model_name and 'eeg_conformer' in embedding_model_name.lower():
            batch_size, n_spatial_channels, n_temporal_channels, seq_len = x_data.shape
            x_data = x_data.reshape(batch_size, n_spatial_channels, n_temporal_channels * seq_len) # Shape: (batch_size, n_spatial_channels, n_temporal_channels * seq_len)
            x_data = np.expand_dims(x_data, axis=1)  # Shape: (batch_size, 1, n_spatial_channels, n_temporal_channels * seq_len)

        return to_tensor(x_data), to_tensor(y_label).long(), trial_metadata

    def _create_metadata_vectorized(
        self,
        batch_size,
        n_spatial_channels,
        n_temporal_channels,
        sample_keys,
        y_label,
        sample_metadata,
    ):
        """
        Replicate each original window's metadata for every flattened
        spatial/temporal series while preserving the complete raw-LMDB
        metadata dictionary.

        For one original sample, this produces:
            n_spatial_channels * n_temporal_channels
        metadata dictionaries in the same order as x_data.reshape(...).
        """
        expected_batch_size = len(sample_keys)

        if batch_size != expected_batch_size:
            raise RuntimeError(
                "Batch size does not match sample-key count: "
                f"{batch_size} vs {expected_batch_size}"
            )

        if len(sample_metadata) != batch_size:
            raise RuntimeError(
                "Metadata count does not match batch size: "
                f"{len(sample_metadata)} vs {batch_size}"
            )

        metadata = []
        reshaped_idx = 0

        for batch_idx in range(batch_size):
            base_metadata = sample_metadata[batch_idx]

            if base_metadata is None:
                base_metadata = {}

            if not isinstance(base_metadata, dict):
                raise TypeError(
                    "Expected original metadata to be a dictionary, "
                    f"got {type(base_metadata).__name__} "
                    f"for batch index {batch_idx}."
                )

            for spatial_channel in range(
                n_spatial_channels
            ):
                for temporal_channel in range(
                    n_temporal_channels
                ):
                    series_metadata = dict(
                        base_metadata
                    )

                    series_metadata.update(
                        {
                            'sample_key': str(
                                sample_keys[batch_idx]
                            ),
                            'batch_idx': int(
                                batch_idx
                            ),
                            'spatial_channel': int(
                                spatial_channel
                            ),
                            'temporal_channel': int(
                                temporal_channel
                            ),
                            'original_label': int(
                                y_label[batch_idx]
                            ),
                            'reshaped_idx': int(
                                reshaped_idx
                            ),
                        }
                    )

                    metadata.append(
                        series_metadata
                    )
                    reshaped_idx += 1

        expected_total = (
            batch_size
            * n_spatial_channels
            * n_temporal_channels
        )

        if len(metadata) != expected_total:
            raise RuntimeError(
                "Unexpected metadata length after flattening: "
                f"{len(metadata)} vs {expected_total}"
            )

        return metadata


    def collate_with_mask(
        dataset,
        batch,
        orig_seq_len,
        embedding_model_name=None,
    ):
        x_data, y_label, trial_metadata = dataset.collate(
            batch,
            embedding_model_name=embedding_model_name,
        )
        # X_data shape: (batch_size, 1, seq_len)
        # y_label shape: (batch_size,)
        

        ### DEBUG MASK SOURCE ###
        # print("collate x_data shape:", getattr(x_data, "shape", None), "dtype:", getattr(x_data, "dtype", None))
        # # show what keys exist in a single raw item
        # if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], dict):
        #     print("raw item keys:", batch[0].keys())
        #     if "mask" in batch[0]:
        #         m0 = batch[0]["mask"]
        #         import numpy as np
        #         m0 = np.asarray(m0)
        #         print("raw item mask shape:", m0.shape, "mean:", float(m0.mean()), "unique:", np.unique(m0)[:5])
        ### END DEBUG MASK SOURCE ###


        # Create a mask for the sequence length
        mask = torch.ones(x_data.shape[0], x_data.shape[-1], dtype=torch.bool)  # Shape: (batch_size, seq_len)
        # Zero out the padding part of the mask
        pad_width = dataset.pad_to_len - orig_seq_len
        if pad_width > 0:#debug
            mask[:, -pad_width:] = 0

        if embedding_model_name and 'tspulse' in embedding_model_name.lower():
            # For TSPulse, it expects input shape (batch_size, seq_len, 1) and mask shape (batch_size, seq_len, 1)
            # Swap dimensions to (batch_size, seq_len, 1) for TSPulse
            x_data = x_data.permute(0, 2, 1)
            # Make mask shape (batch_size, seq_len, 1)
            mask = mask.unsqueeze(-1)

        return x_data, y_label, mask, trial_metadata

class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.dataset_dir = params.dataset_dir
        self.dataset_params = get_dataset_params(dataset_name=params.dataset_name)
        self.n_temporal_channels = self.dataset_params['n_temporal_channels']
        self.n_spatial_channels = self.dataset_params['n_spatial_channels']
        self.orig_seq_len = params.orig_seq_len
        self.tdr = params.tdr if hasattr(params, 'tdr') else 1.0
        self.embedding_model_name = params.embedding_model_name if hasattr(params, 'embedding_model_name') else None
        self.temporal_channel_selection = params.temporal_channel_selection if hasattr(params, 'temporal_channel_selection') else None

        # NOTE: This is important because it allows us to guarantee that the order is always the same,
        # invariant of advances in the original RNG state. For example, in the case we don't have this, say we
        # create our data_loader then initialize our model, the model initialization advances the RNG state before
        # we iterate through the data_loader so we will get an order A. Now suppose we don't initialize that same model,
        # then we would get an order B. Thus, we need a separate RNG that only handles the data loader. This ensures that
        # our pipeline and the Cbramod pipeline use the same train order.
        if hasattr(self.params, 'seed'):
            self.dataloader_rng = torch.Generator()
            self.dataloader_rng.manual_seed(params.seed)
        else:
            print('WARNING: Seed was not given, so train generator will not be set!!!')

        self._cached_sample_orders = {}

    def get_data_loader(self):
        train_set = CustomLMDBPickleDataset(
            self.params.dataset_name, self.dataset_dir, mode='train', 
            pad_to_len=self.params.pad_to_len, reshape_data=self.params.reshape_data,
            tdr=self.tdr, seed=self.params.seed if hasattr(self, 'seed') else None)
        val_set = CustomLMDBPickleDataset(
            self.params.dataset_name, self.dataset_dir, mode='val', 
            pad_to_len=self.params.pad_to_len, reshape_data=self.params.reshape_data,
            tdr=self.tdr, seed=self.params.seed if hasattr(self, 'seed') else None)
        test_set = CustomLMDBPickleDataset(
            self.params.dataset_name, self.dataset_dir, mode='test', 
            pad_to_len=self.params.pad_to_len, reshape_data=self.params.reshape_data,
            tdr=self.tdr, seed=self.params.seed if hasattr(self, 'seed') else None)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))

        if self.temporal_channel_selection is not None:
            # Update n_temporal_channels in case it was changed due to temporal_channel_selection
            self.n_temporal_channels = train_set.n_temporal_channels

        if self.params.return_mask:
            train_collate_fn = partial(train_set.collate_with_mask, orig_seq_len=self.orig_seq_len, embedding_model_name=self.embedding_model_name)
            val_collate_fn = partial(val_set.collate_with_mask, orig_seq_len=self.orig_seq_len, embedding_model_name=self.embedding_model_name)
            test_collate_fn = partial(test_set.collate_with_mask, orig_seq_len=self.orig_seq_len, embedding_model_name=self.embedding_model_name) 
        else:
            train_collate_fn = partial(train_set.collate, embedding_model_name=self.embedding_model_name)
            val_collate_fn = partial(val_set.collate, embedding_model_name=self.embedding_model_name)
            test_collate_fn = partial(test_set.collate, embedding_model_name=self.embedding_model_name)
            
        train_shuffle = True
        val_shuffle = False
        test_shuffle = False

        if self.params.dataset_name in ['shu', 'stress']:
            # NOTE: For some reason, CBraMod shuffles the val and test splits for the shu and stress datasets
            val_shuffle = True
            test_shuffle = True

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_collate_fn,
                shuffle=train_shuffle,
                generator=self.dataloader_rng if hasattr(self.params, 'seed') else None
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_collate_fn,
                shuffle=val_shuffle,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_collate_fn,
                shuffle=test_shuffle,
            ),
        }
        return data_loader