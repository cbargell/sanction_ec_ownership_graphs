# closest_embeddings.py
# Exact nearest neighbor per vector (cosine) using FAISS Flat.
# This will produce the full dataset for each node's closest neighbor

import os, math, gc
import numpy as np
import pandas as pd

EMB_CSV = "O:/scratch/yicheng/230/emb_vgae_directed.csv"
OUT_NN  = "O:/scratch/yicheng/230/nearest_neighbor_per_node_flat_exact.csv"

BATCH_ADD = 500_000     # add this many vectors per batch
BATCH_QRY = 200_000     # query this many per batch
USE_GPU   = True

def l2_normalize_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    m = (n.squeeze(-1) > 0)
    X[m] /= n[m]
    return X

def read_ids(csv_path):
    df = pd.read_csv(csv_path, usecols=["firm_id"])
    return df["firm_id"].astype(str).to_list()

def read_feats(csv_path, start, count, feat_cols=None):
    if feat_cols is None:
        head = pd.read_csv(csv_path, nrows=0)
        feat_cols = [c for c in head.columns if c != "firm_id"]
    df = pd.read_csv(csv_path,
                     skiprows=range(1, start+1),
                     nrows=count,
                     usecols=feat_cols)
    X = df.to_numpy(dtype=np.float32, copy=True)
    return X, feat_cols

# setup
import faiss
faiss.omp_set_num_threads(max(1, os.cpu_count() or 1))

# discover N, D and ids
hdr = pd.read_csv(EMB_CSV, nrows=1)
feat_cols = [c for c in hdr.columns if c != "firm_id"]
D = len(feat_cols)
ids = read_ids(EMB_CSV)
N = len(ids)
print(f"N={N:,}, D={D}")

# build flat exact index (inner product for cosine)
cpu_index = faiss.IndexFlatIP(D)

gpu_index = None
if USE_GPU:
    try:
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        index = gpu_index
        print("Using GPU Flat index.")
    except Exception as e:
        print(f"GPU not available ({e}); falling back to CPU.")
        index = cpu_index
else:
    index = cpu_index

# add database vectors in batches 
added = 0
while added < N:
    n = min(BATCH_ADD, N - added)
    Xb, _ = read_feats(EMB_CSV, added, n, feat_cols)
    l2_normalize_rows(Xb)
    index.add(Xb)  # IDs are implicit: [added, ..., added+n-1]
    added += n
    print(f"Added {added:,}/{N:,}")
    del Xb; gc.collect()

# query each vector against the DB (batched) 
# search k=2 and drop self (exact NN is the other one)
nn_idx = np.full(N, -1, dtype=np.int64)
nn_sim = np.full(N, -np.inf, dtype=np.float32)

q = 0
while q < N:
    n = min(BATCH_QRY, N - q)
    Xq, _ = read_feats(EMB_CSV, q, n, feat_cols)
    l2_normalize_rows(Xq)
    Dm, Im = index.search(Xq, k=2)  # (n,2)
    rows = np.arange(n)
    # if top-1 is self, pick column 2; else pick column 1
    pick = (Im[:,0] == (q + rows)).astype(np.int64)
    nn_idx[q:q+n] = Im[rows, pick ^ 1] if np.all(pick) else Im[rows, 1 - pick]
    nn_sim[q:q+n] = Dm[rows, 1 - pick]
    q += n
    print(f"Queried {q:,}/{N:,}")
    del Xq, Dm, Im; gc.collect()

out_df = pd.DataFrame({
    "i": np.arange(N, dtype=np.int64),
    "firm_i": ids,
    "nn_j": nn_idx,
    "nn_firm": [ids[j] if j >= 0 else None for j in nn_idx],
    "cosine_sim": nn_sim,
    "euclid_dist_on_unit": np.sqrt(np.maximum(0.0, 2.0 - 2.0 * nn_sim.astype(np.float64))),
})
out_df.to_csv(OUT_NN, index=False)
print(f"Wrote -> {OUT_NN}")
