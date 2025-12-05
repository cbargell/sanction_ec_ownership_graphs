import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import normalize

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

#####################################################################
# Without conglomerate information
#####################################################################

EMB_OUT = "O:/scratch/yicheng/230/emb_vgae_directed_10_cpu.csv"

# load embeddings
df = pd.read_csv(EMB_OUT)
firm_id = df["firm_id"].astype(str).to_numpy()

Z = df.drop(columns=["firm_id"]).to_numpy(dtype=np.float32)

# normalize
Z = normalize(Z, norm="l2")

MAX_PLOT = 200_000
if Z.shape[0] > MAX_PLOT:
    rng = np.random.default_rng(7)
    idx = rng.choice(Z.shape[0], size=MAX_PLOT, replace=False)
    firm_id_s = firm_id[idx]
    Z_s = Z[idx]
else:
    firm_id_s = firm_id
    Z_s = Z

#2D layout
import umap
reducer = umap.UMAP(
    n_neighbors=30,      
    min_dist=0.05,       
    metric="cosine",
    random_state=7,
)
Y = reducer.fit_transform(Z_s)
method = "UMAP"


K = 150  
km = MiniBatchKMeans(n_clusters=K, random_state=7, batch_size=4096, n_init="auto")
labels = km.fit_predict(Z_s)
algo = f"KMeans(K={K})"

plt.figure(figsize=(10, 8))
plt.scatter(Y[:, 0], Y[:, 1], s=2, c=labels, alpha=0.6)
plt.title(f"Firm embeddings clustered")
plt.xlabel(f"{method}-1")
plt.ylabel(f"{method}-2")
plt.tight_layout()
plt.show()

unique, counts = np.unique(labels, return_counts=True)
sizes = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)
print("Top cluster sizes:", sizes[:10])



#####################################################################
# With conglomerate information
#####################################################################

NODES_CSV = "O:/scratch/yicheng/230/nodes_unweighted_all.csv"
EDGES_CSV = "O:/scratch/yicheng/230/edges_unweighted_all.csv"
EMB_OUT   = "O:/scratch/yicheng/230/emb_vgae_directed_10_cpu.csv"

COMP_CACHE_CSV = "O:/scratch/yicheng/230/wcc_byrow_cache.csv"

#SEED = 7
EDGE_CHUNKSIZE = 2_000_000

MAX_PLOT = 200_000          # total points sent to UMAP
TOP_M = 60                 # number of non-giant conglomerates to color
EXCLUDE_GIANT_FROM_COLOR = True

# cap how many points from the giant WCC enter the plot
USE_GIANT_CAP = True
GIANT_CAP = 30_000          

# UMAP settings
N_NEIGHBORS = 30
MIN_DIST = 0.05

DOT_BG = 8
DOT_FG = 15

# union set
class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# nodes
nodes = pd.read_csv(NODES_CSV, dtype=str).fillna("")
ids = nodes["firm_id"].astype(str).to_numpy()
N = len(ids)

dup_nodes = int(pd.Series(ids).duplicated().sum())
print("Duplicate firm_id rows in nodes:", dup_nodes)

# mapping firm_id
id2ix = {fid: i for i, fid in enumerate(ids)}

# load cache
need_recompute = True
if os.path.exists(COMP_CACHE_CSV):
    try:
        cache = pd.read_csv(COMP_CACHE_CSV)
        if {"node_ix", "conglomerate"}.issubset(cache.columns) and len(cache) == N:
            cache = cache.sort_values("node_ix")
            comp_id = cache["conglomerate"].to_numpy(dtype=np.int64)
            need_recompute = False
            print("Loaded cache:", COMP_CACHE_CSV)
        else:
            print("Cache exists but format/length mismatch; recomputing WCC.")
    except Exception as e:
        print("Failed to load cache; recomputing. Reason:", repr(e))

if need_recompute:
    uf = UnionFind(N)

    for chunk in pd.read_csv(EDGES_CSV, dtype=str, usecols=["src", "dst"], chunksize=EDGE_CHUNKSIZE):
        chunk = chunk.fillna("")
        src_ix = chunk["src"].map(id2ix)
        dst_ix = chunk["dst"].map(id2ix)
        m = src_ix.notna() & dst_ix.notna()
        src_ix = src_ix[m].astype(np.int64).to_numpy()
        dst_ix = dst_ix[m].astype(np.int64).to_numpy()

        # weak connectivity
        for u, v in zip(src_ix, dst_ix):
            if u != v:
                uf.union(int(u), int(v))

    roots = np.fromiter((uf.find(i) for i in range(N)), dtype=np.int64, count=N)
    _, comp_id = np.unique(roots, return_inverse=True)  

    os.makedirs(os.path.dirname(COMP_CACHE_CSV), exist_ok=True)
    pd.DataFrame({"node_ix": np.arange(N, dtype=np.int64), "conglomerate": comp_id}).to_csv(COMP_CACHE_CSV, index=False)
    print("Saved WCC cache:", COMP_CACHE_CSV)

C = int(comp_id.max() + 1)
sizes_row = np.bincount(comp_id, minlength=C)
print(f"Number of conglomerates (WCC by row): {C:,}")
print("Top 10 component sizes (by row):", sorted(sizes_row.tolist(), reverse=True)[:10])

# firm_id may appear multiple times with different conglomerate labels
comp_df_all = pd.DataFrame({"firm_id": ids, "conglomerate": comp_id})
rng = np.random.default_rng(SEED)
comp_df_all = comp_df_all.iloc[rng.permutation(len(comp_df_all))].reset_index(drop=True)
comp_df = comp_df_all.drop_duplicates("firm_id", keep="first")  

# firm-level component sizes 
sizes_firm = comp_df["conglomerate"].value_counts()
giant_id = int(sizes_firm.idxmax())
print(f"Giant conglomerate (firm-level) id={giant_id}, size={int(sizes_firm.loc[giant_id]):,}")

# choose which components to color 
comp_order = sizes_firm.index.to_numpy(dtype=np.int64)  
if EXCLUDE_GIANT_FROM_COLOR:
    comp_order = comp_order[comp_order != giant_id]

top_ids = comp_order[:TOP_M]
top_set = set(top_ids.tolist())

if len(top_ids) > 0:
    print(f"Largest colored conglomerate id={int(top_ids[0])}, size={int(sizes_firm.loc[int(top_ids[0])]):,}")
print(f"Coloring TOP_M={TOP_M} (exclude_giant={EXCLUDE_GIANT_FROM_COLOR})")

# embeddings
emb = pd.read_csv(EMB_OUT)
emb["firm_id"] = emb["firm_id"].astype(str)

dup_emb = int(emb["firm_id"].duplicated().sum())
print("Duplicate firm_id rows in embeddings:", dup_emb)

emb = emb.iloc[rng.permutation(len(emb))].drop_duplicates("firm_id", keep="first").reset_index(drop=True)

emb = emb.merge(comp_df, on="firm_id", how="inner")
print("Embeddings after merge:", len(emb))

labels_full = emb["conglomerate"].to_numpy(dtype=np.int64)

if USE_GIANT_CAP:
    idx_g = np.where(labels_full == giant_id)[0]
    idx_o = np.where(labels_full != giant_id)[0]

    other_cap = max(0, MAX_PLOT - GIANT_CAP)
    take_g = min(GIANT_CAP, len(idx_g))
    take_o = min(other_cap, len(idx_o))

    pick_parts = []
    if take_g > 0:
        pick_parts.append(rng.choice(idx_g, size=take_g, replace=False))
    if take_o > 0:
        pick_parts.append(rng.choice(idx_o, size=take_o, replace=False))
    pick = np.concatenate(pick_parts) if pick_parts else np.array([], dtype=np.int64)
    rng.shuffle(pick)

    emb = emb.iloc[pick].reset_index(drop=True)
    labels = labels_full[pick]
    print(f"UMAP sample: total={len(emb):,} | giant={take_g:,} | non-giant={take_o:,}")
else:
    if len(emb) > MAX_PLOT:
        emb = emb.sample(n=MAX_PLOT, random_state=SEED).reset_index(drop=True)
    labels = emb["conglomerate"].to_numpy(dtype=np.int64)
    print(f"UMAP sample: total={len(emb):,} (pure random)")

# embedding matrix
Z = emb.drop(columns=["firm_id", "conglomerate"]).to_numpy(dtype=np.float32)
Z = normalize(Z, norm="l2")

# umap
import umap
reducer = umap.UMAP(
    n_neighbors=N_NEIGHBORS,
    min_dist=MIN_DIST,
    metric="cosine",
    init="random",
    random_state=None,   # allow parallel
    n_jobs=-1,
    low_memory=True,
)
Y = reducer.fit_transform(Z)


# color by conglomerate (top non-giant), others grey 
labels2 = np.array([lab if lab in top_set else -1 for lab in labels], dtype=np.int64)
remap = {cid: i for i, cid in enumerate(top_ids)}
labels_plot = np.array([remap[x] if x != -1 else -1 for x in labels2], dtype=np.int64)

plt.figure(figsize=(10, 8))

bg = labels_plot == -1
plt.scatter(Y[bg, 0], Y[bg, 1], s=DOT_BG, color="lightgray", alpha=0.25, linewidths=0)

fg = ~bg
cmap = plt.get_cmap("tab20", TOP_M) 
norm = mcolors.BoundaryNorm(np.arange(-0.5, TOP_M + 0.5, 1), cmap.N)

plt.scatter(
    Y[fg, 0], Y[fg, 1],
    s=DOT_FG,
    c=labels_plot[fg],
    cmap=cmap,
    norm=norm,
    alpha=0.90,
    linewidths=0
)

plt.title(
    f"{method} of firm embeddings (n={len(Z):,}); "
    f"top {TOP_M} conglomerates colored"
)
plt.xlabel(f"{method}-1")
plt.ylabel(f"{method}-2")
plt.tight_layout()
plt.show()
