import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lmdb
import json
from CBraMod.utils.util import to_tensor
from stamp.datasets.utils import get_dataset_params
import pickle

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
        temporal_channel_selection=None
    ):
        super(CustomLMDBEmbeddingDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.mode = mode
        self.n_temporal_channels = n_temporal_channels
        self.n_spatial_channels = n_spatial_channels
        self.channel_product = self.n_temporal_channels * self.n_spatial_channels
        self.tdr = tdr if tdr is not None else 1.0
        self.seed = seed
        self.temporal_channel_selection = temporal_channel_selection
        print(f"Temporal channel selection: {self.temporal_channel_selection}")

        self.db = None

        self._init_keys()

    def _init_keys(self):
        """Initialize keys from LMDB database"""
        temp_db = lmdb.open(
            self.data_dir + f'/{self.mode}', 
            readonly=True, 
            lock=False, 
            readahead=True,
            meminit=False
        )
        with temp_db.begin(write=False) as txn:
            keys_bytes = txn.get(b'__keys__')
            keys_str = json.loads(keys_bytes.decode())
            self.keys = [k.encode() for k in keys_str]
            if self.tdr < 1.0 and self.mode == 'train':
                print(f"Using training data ratio of {self.tdr}")
                length = len(self.keys)
                # Shuffle keys
                rng = np.random.default_rng(self.seed)
                rng.shuffle(self.keys)
                self.keys = self.keys[:int(length * self.tdr)]
            self.length = len(self.keys)
            print(f"Loaded {self.length} keys from stored __keys__")
        temp_db.close()  # Close the temporary connection

    def _get_db(self):
        """Lazy initialization of DB connection for each worker"""
        if self.db is None:
            self.db = lmdb.open(
                self.data_dir + f'/{self.mode}', 
                readonly=True, 
                lock=False, 
                readahead=True,
                meminit=False,
                max_readers=1024
            )
            with self.db.begin(write=False) as txn:
                keys_bytes = txn.get(b'__keys__')
                keys_str = json.loads(keys_bytes.decode())
                self.keys = [k.encode() for k in keys_str]
        return self.db

    def _init_db(self):
        """Initialize LMDB database connection"""
        self.db = lmdb.open(self.data_dir + f'/{self.mode}', readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            keys_bytes = txn.get(b'__keys__')
            keys_str = json.loads(keys_bytes.decode())
            self.keys = [k.encode() for k in keys_str]  # Convert back to bytes
            self.length = len(self.keys)
            print(f"Loaded {self.length} keys from stored __keys__")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return idx

    def collate(self, batch_indices):
        x_data = [] # Shape: (batch_size, n_spatial_channels * n_temporal_channels * embedding_dim)
        y_labels = [] # Shape: (batch_size, n_classes) for multiclass; (batch_size,) for binary
        token_comp_labels_list = []
        temporal_token_comp_labels_list=[]
        sample_keys = [] # List of sample keys

        # Single transaction for the entire batch
        db = self._get_db()  # Get DB connection for this worker
        with db.begin(write=False) as txn:
            for idx in batch_indices:
                key = self.keys[idx]
                data_bytes = txn.get(key)
                
                if data_bytes is None:
                    raise IndexError(f"Sample {idx} not found")
                
                pair = pickle.loads(data_bytes)

                sample = pair["sample"]                  # [8, 99, 1024]
                label = pair["label"]
                token_comp_labels = pair["token_comp_labels"]
                temporal_token_comp_labels = pair["temporal_token_comp_labels"]

                x_data.append(sample)
                y_labels.append(label)
                token_comp_labels_list.append(token_comp_labels)
                temporal_token_comp_labels_list.append(temporal_token_comp_labels)
                sample_keys.append(key)

        x_data = np.stack(x_data, axis=0)  # Shape: (batch_size, n_spatial_channels * n_temporal_channels * embedding_dim)
        y_label = np.array(y_labels)  # Shape: (batch_size,)
        sample_keys = np.array(sample_keys)  # Shape: (batch_size,)
        token_comp_labels = np.stack(token_comp_labels_list, axis=0)
        temporal_token_comp_labels = np.stack(temporal_token_comp_labels_list, axis=0)

        

        if self.temporal_channel_selection is not None:
            # Apply temporal channel selection
            x_data = x_data[:, self.temporal_channel_selection, :, :] # Shape: (batch_size, selected_n_temporal, n_spatial, embedding_dim)

        x_data = to_tensor(x_data)

        return (x_data, to_tensor(y_label).long(), to_tensor(token_comp_labels).long(),to_tensor(temporal_token_comp_labels).long(), sample_keys)

class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.dataset_dir = params.dataset_dir
        self.dataset_params = get_dataset_params(dataset_name=params.dataset_name)
        self.n_temporal_channels = self.dataset_params['n_temporal_channels']
        self.n_spatial_channels = self.dataset_params['n_spatial_channels']
        self.num_workers = params.num_workers if hasattr(params, 'num_workers') else 4
        self.prefetch_factor = params.prefetch_factor if hasattr(params, 'prefetch_factor') else 2
        self.tdr = params.tdr if hasattr(params, 'tdr') else 1.0
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
        train_set = CustomLMDBEmbeddingDataset(
            self.params.dataset_name, self.dataset_dir, mode='train', 
            n_temporal_channels=self.n_temporal_channels, n_spatial_channels=self.n_spatial_channels, 
            tdr=self.tdr, seed=self.params.seed if hasattr(self, 'seed') else None,
            temporal_channel_selection=self.temporal_channel_selection)
        val_set = CustomLMDBEmbeddingDataset(
            self.params.dataset_name, self.dataset_dir, mode='val', 
            n_temporal_channels=self.n_temporal_channels, n_spatial_channels=self.n_spatial_channels, 
            tdr=self.tdr, seed=self.params.seed if hasattr(self, 'seed') else None,
            temporal_channel_selection=self.temporal_channel_selection)
        test_set = CustomLMDBEmbeddingDataset(
            self.params.dataset_name, self.dataset_dir, mode='test', 
            n_temporal_channels=self.n_temporal_channels, n_spatial_channels=self.n_spatial_channels,
            tdr=self.tdr, seed=self.params.seed if hasattr(self, 'seed') else None,
            temporal_channel_selection=self.temporal_channel_selection)
        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                generator=self.dataloader_rng if hasattr(self.params, 'seed') else None,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=True,  # Keep workers alive between epochs
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                num_workers=self.num_workers, 
                pin_memory=True, 
                prefetch_factor=self.prefetch_factor,
                persistent_workers=True,  # Keep workers alive between epochs
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
            ),
        }
        return data_loader