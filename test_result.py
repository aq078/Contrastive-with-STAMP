import lmdb

path = "embeddings/Penn_Action/MOMENT-1-large/train"

env = lmdb.open(
    path,
    readonly=True,
    lock=False,
    readahead=False,
)

with env.begin() as txn:
    cursor = txn.cursor()

    for i, (key, value) in enumerate(cursor):
        print("key:", key)
        print("value bytes:", len(value))
        print("first 32 bytes:", value[:32])
        print()

        if i >= 4:
            break

env.close()