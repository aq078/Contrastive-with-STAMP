python - <<'PY'
from types import SimpleNamespace
from stamp.datasets.lmdb_embedding_dataset import LoadDataset

# EDIT THESE:
EMB_ROOT = "dataset/processed_sere/sere_framecomp_world_xyzv_L64_S16.lmdb"   # folder that contains train/ val/ test/
DATASET_NAME = "sere"                      # must exist in get_dataset_params

p = SimpleNamespace(
    dataset_name=DATASET_NAME,
    dataset_dir=EMB_ROOT,
    batch_size=2,
    seed=0,
    num_workers=0,       # keep 0 for debugging
    prefetch_factor=2,
    tdr=1.0,
    temporal_channel_selection=None,
)

dl = LoadDataset(p).get_data_loader()

x, y, keys = next(iter(dl["train"]))
print("x:", x.shape, x.dtype)
print("y:", y.shape, y[:10])
print("keys[0:2]:", keys[:2])
PY
