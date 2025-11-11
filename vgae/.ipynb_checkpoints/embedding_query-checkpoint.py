# embedding_query.py
# ------------------------------------------------------------
# Exact cosine neighbors for a small set of firm_ids using FAISS Flat.
# - Streams embeddings from embedding outputs and L2-normalizes before indexing.
# - For each target firm_id, queries k=6 and drops self -> top 5.
# Requires: pip install faiss-cpu.
# ------------------------------------------------------------
import os, math, gc
import numpy as np
import pandas as pd
import faiss  # pip install faiss-cpu  (or faiss-gpu)

# path
EMB_CSV   = "C:/Users/yichengy/Downloads/emb_vgae_directed.csv"
OUT_CSV   = "C:/Users/yichengy/Downloads/top5_neighbors_for_targets.csv"

# edit here for target firm id
TARGET_FIRMS = [
    "AE*Z00061201",
    "AE0000003829",
    "AE0000040199",
    "AE0000116198",
    "AE0000350503",
]

# batching
BATCH_ADD = 500_000   # rows per chunk when adding to index
USE_GPU   = True

faiss.omp_set_num_threads(max(1, os.cpu_count() or 1))

def cos_to_euclid_on_unit(c):
    return math.sqrt(max(0.0, 2.0 - 2.0*float(c)))

def feature_chunk_iter(csv_path, feat_cols, chunksize):
    """
    Yields (offset, X_chunk) where X_chunk is C-contiguous float32 and L2-normalized.
    """
    offset = 0
    for df in pd.read_csv(csv_path, usecols=feat_cols, chunksize=chunksize, iterator=True):
        X = df.to_numpy(dtype=np.float32, copy=False)
        X = np.ascontiguousarray(X, dtype=np.float32)  # FAISS wants C-contiguous, writable
        faiss.normalize_L2(X)                          # cosine via inner product
        n = X.shape[0]
        yield offset, X
        offset += n

# discover schema and mapping
hdr = pd.read_csv(EMB_CSV, nrows=1)
all_cols = list(hdr.columns)
assert "firm_id" in all_cols, "CSV must have a 'firm_id' column"
FEAT_COLS = [c for c in all_cols if c != "firm_id"]
D = len(FEAT_COLS)

ids = pd.read_csv(EMB_CSV, usecols=["firm_id"])["firm_id"].astype(str).to_list()
N = len(ids)
print(f"N={N:,}, D={D}")

id2ix = {fid: i for i, fid in enumerate(ids)}

# validate targets exist
missing = [fid for fid in TARGET_FIRMS if fid not in id2ix]
if missing:
    raise ValueError(f"These firm_ids were not found in the CSV: {missing}")

anchor_idx = np.array([id2ix[fid] for fid in TARGET_FIRMS], dtype=np.int64)
anchor_pos = {int(idx): p for p, idx in enumerate(anchor_idx)}
Q = len(anchor_idx)

# placeholder for query rows
X_query = np.zeros((Q, D), dtype=np.float32)
have = np.zeros(Q, dtype=bool)

# build exact Flat index (cosine via IP)
cpu_index = faiss.IndexFlatIP(D)  # exact
index = cpu_index
if USE_GPU:
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("Using GPU Flat index.")
    except Exception as e:
        print(f"GPU not available ({e}); using CPU.")

print("Adding vectors and capturing target queries (streamed)...")
added = 0
for off, Xb in feature_chunk_iter(EMB_CSV, FEAT_COLS, BATCH_ADD):
    # Add to index
    index.add(Xb)  # implicit IDs are 0..N-1
    # If any targets live in this chunk, copy them into X_query
    # global indices in [off, off+len)
    if have.all():
        pass  # already have all targets
    else:
        # which anchors fall into this range?
        mask = (anchor_idx >= off) & (anchor_idx < off + Xb.shape[0])
        if mask.any():
            sel = anchor_idx[mask]
            for gidx in sel:
                pos = anchor_pos[int(gidx)]
                local = int(gidx - off)
                X_query[pos] = Xb[local]
                have[pos] = True

    added += Xb.shape[0]
    print(f"  added {added:,}/{N:,}")
    del Xb; gc.collect()

if not have.all():
    missing_q = [TARGET_FIRMS[p] for p in np.where(~have)[0]]
    raise RuntimeError(f"Failed to collect query vectors for: {missing_q}")

# query all targets at once, get top-5 (drop self)
print("Querying targets...")
# Ensure 2D contiguous float32
X_query = np.ascontiguousarray(X_query, dtype=np.float32)
# Search for k=6, then drop the self-match (exact index, so self will be included)
k = 6
Dm, Im = index.search(X_query, k)   # shapes: (Q, k)

rows = []
for p, fid in enumerate(TARGET_FIRMS):
    gidx = int(anchor_idx[p])
    nbrs = []
    for j, s in zip(Im[p], Dm[p]):
        j = int(j)
        if j == gidx:
            continue
        nbrs.append((j, float(s)))
        if len(nbrs) == 5:
            break
    # Fallback in rare case self wasn't returned: still take first 5 unique
    if len(nbrs) < 5:
        seen = set(j for j, _ in nbrs)
        for j, s in zip(Im[p], Dm[p]):
            j = int(j)
            if j == gidx or j in seen:
                continue
            nbrs.append((j, float(s)))
            seen.add(j)
            if len(nbrs) == 5:
                break

    for rank, (j, s) in enumerate(nbrs, start=1):
        rows.append({
            "anchor_i": gidx,
            "anchor_firm": fid,
            "rank": rank,
            "neighbor_j": j,
            "neighbor_firm": ids[j],
            "cosine_sim": s,
            "euclid_dist_on_unit": cos_to_euclid_on_unit(s),
        })

out_df = pd.DataFrame(rows, columns=[
    "anchor_i","anchor_firm","rank","neighbor_j","neighbor_firm","cosine_sim","euclid_dist_on_unit"
])
out_df.to_csv(OUT_CSV, index=False)
print(f"Wrote -> {OUT_CSV}")
print(out_df.to_string(index=False))
