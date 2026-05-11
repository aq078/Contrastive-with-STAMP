import pickle, lmdb

env = lmdb.open("dataset/processed_sere/sere_comp_hybrid_train_event_valtest_random_xyz.lmdb", readonly=True, lock=False)
with env.begin() as txn:
    keys = pickle.loads(txn.get(b"__keys__"))

print({k: len(v) for k, v in keys.items()})
print("Total:", sum(len(v) for v in keys.values()))