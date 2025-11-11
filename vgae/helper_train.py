# helper_train.py
# ------------------------------------------------------------
# Inputs:
#   nodes_unweighted_all.csv : firm_id, iso, sic, sic_descr
#   edges_unweighted_all.csv : src, dst   (shareholder -> subsidiary)
#
# Features: ISO one-hot (sparse) + TF-IDF(sic_descr) (sparse) + bias (sparse)
# Encoder : 2-layer GraphSAGE-Mean with IN+OUT aggregation -> (mu, logstd)
# Decoder : asymmetric MLP on [z_u || z_v] (keeps direction)
# Loss    : BCE(pos vs sampled neg) + beta * KL, with KL warm-up
# Eval    : ROC-AUC / Average Precision on held-out edges
# ------------------------------------------------------------
import os
os.environ["PYTORCH_DISABLE_DYNAMO"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import math, random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse  # for CSR features & subgraph ops

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Config -----------------------------
NODES_CSV = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230/nodes_unweighted_all.csv"
EDGES_CSV = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230/edges_unweighted_all.csv"
EMB_OUT   = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230/emb_vgae_directed.csv"

HIDDEN = 128
LATENT = 64
LR = 1e-3
EPOCHS = 200
VAL_RATIO = 0.05
TEST_RATIO = 0.10
NEG_TRAIN_RATIO = 1.0          # negatives per positive per batch
BETA_MAX = 2e-2                # final KL weight (warm-up target)
WARMUP_EPOCHS = 20             # epochs to ramp beta from 0 -> BETA_MAX
TFIDF_MAX_FEATURES = 2000      # tune for RAM
EDGE_BATCH_SIZE = 8192         # edges per mini-batch (tune as needed)
ENC_NODE_CHUNK = 20000         # node chunk for full-graph encoding at eval/saving

SEED = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------------------------------------------------

def set_seed(seed=7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# ----------------------------- Load data -----------------------------
nodes = pd.read_csv(NODES_CSV, dtype=str).fillna("")
edges = pd.read_csv(EDGES_CSV, dtype=str).fillna("")

ids = nodes["firm_id"].astype(str).tolist()
id2ix = {k:i for i,k in enumerate(ids)}
N = len(ids)

# sanitize edges -> indices, drop self-loops & duplicates
E = (edges.assign(src_ix = edges["src"].map(id2ix),
                  dst_ix = edges["dst"].map(id2ix))
           .dropna(subset=["src_ix","dst_ix"]))
E["src_ix"] = E["src_ix"].astype(int)
E["dst_ix"] = E["dst_ix"].astype(int)
E = E[E["src_ix"] != E["dst_ix"]].drop_duplicates(subset=["src_ix","dst_ix"]).reset_index(drop=True)

E_np = E[["src_ix","dst_ix"]].to_numpy()
print(f"Graph: {N:,} nodes, {len(E_np):,} directed edges")

# edge split: train/val/test
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(E_np))
n_test = int(len(E_np) * TEST_RATIO)
n_val  = int(len(E_np) * VAL_RATIO)
test_idx = perm[:n_test]
val_idx  = perm[n_test:n_test+n_val]
train_idx = perm[n_test+n_val:]

E_train = E_np[train_idx]
E_val   = E_np[val_idx]
E_test  = E_np[test_idx]
print(f"Edges split -> train: {len(E_train):,}, val: {len(E_val):,}, test: {len(E_test):,}")

E_all_set = set(map(tuple, E_np))  # for negative sampling (avoid any true edge)

# csr
nodes["iso"] = nodes["iso"].astype(str)
nodes["sic_descr"] = nodes["sic_descr"].fillna("").astype(str)

# TF-IDF sparse features
tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=3, max_features=TFIDF_MAX_FEATURES)
tf = tfidf.fit_transform(nodes["sic_descr"])   # (N x F_txt) CSR

# ISO one-hot as sparse CSR
iso_cats = pd.Categorical(nodes["iso"])
iso_codes = iso_cats.codes.astype(np.int64)
num_iso = int(iso_codes.max() + 1)
rows = np.arange(N, dtype=np.int64)
X_iso_csr = sparse.csr_matrix((np.ones(N, dtype=np.float32), (rows, iso_codes)),
                              shape=(N, num_iso), dtype=np.float32)

# bias column as CSR
bias_csr = sparse.csr_matrix(np.ones((N, 1), dtype=np.float32))

# full sparse feature matrix: ISO one-hot + TF-IDF + bias
X_csr = sparse.hstack([X_iso_csr, tf.astype(np.float32), bias_csr], format="csr", dtype=np.float32)
IN_DIM = X_csr.shape[1]
print("Feature dim (sparse):", IN_DIM)

# sparse row-normalized adj (CSR) 
# CSR matrices for mean aggregation:
# A_in  rows = dst, cols = src (aggregate from incoming owners)
# A_out rows = src, cols = dst (aggregate from outgoing subsidiaries)
def make_row_norm_csr(num_nodes, edges_ix):
    if len(edges_ix) == 0:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    rows = edges_ix[:,1].astype(np.int64)  # dst
    cols = edges_ix[:,0].astype(np.int64)  # src
    data = np.ones(len(edges_ix), dtype=np.float32)
    M = sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    deg = np.asarray(M.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0
    inv_deg = 1.0 / deg
    # row-normalize: multiply by diagonal(inv_deg)
    Dinv = sparse.diags(inv_deg.astype(np.float32))
    return Dinv @ M  # CSR

A_in_csr  = make_row_norm_csr(N, E_train)
A_out_csr = make_row_norm_csr(N, np.c_[E_train[:,1], E_train[:,0]])  # reversed

# build neighbor lists for 1-hop expansion (for forming U)
in_neigh = [[] for _ in range(N)]
out_neigh = [[] for _ in range(N)]
for u, v in E_train:
    out_neigh[u].append(v)  # u -> v
    in_neigh[v].append(u)   # u -> v

in_neigh = [np.array(s, dtype=np.int64) if len(s) else np.empty(0, np.int64) for s in in_neigh]
out_neigh = [np.array(s, dtype=np.int64) if len(s) else np.empty(0, np.int64) for s in out_neigh]

# model module
class SAGEMean3(nn.Module):
    def __init__(self, in_dim, out_dim, act=F.relu):
        super().__init__()
        self.lin = nn.Linear(3*in_dim, out_dim)
        self.act = act

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden, latent):
        super().__init__()
        self.s1   = SAGEMean3(in_dim, hidden, act=F.relu)  # hidden with ReLU
        self.mu   = SAGEMean3(hidden, latent, act=None)    # linear head
        self.lstd = SAGEMean3(hidden, latent, act=None)    # linear head

class MLPDecoder(nn.Module):
    def __init__(self, latent, hidden=64):
        super().__init__()
        self.lin1 = nn.Linear(2*latent, hidden)
        self.lin2 = nn.Linear(hidden, 1)
    def forward(self, z, edge_ix_local):
        # z: (B_nodes x latent); edge_ix_local: indices into z (local to batch)
        if isinstance(edge_ix_local, np.ndarray):
            edge_ix_local = torch.tensor(edge_ix_local, dtype=torch.long, device=z.device)
        u = edge_ix_local[:,0]
        v = edge_ix_local[:,1]
        h = torch.cat([z[u], z[v]], dim=-1)
        h = F.relu(self.lin1(h))
        return self.lin2(h).view(-1)

# Adam 
class SimpleAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr; self.b1, self.b2 = betas; self.eps = eps; self.wd = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.b1, self.b2
        lr_t = self.lr * (math.sqrt(1 - b2**self.t) / (1 - b1**self.t))
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd != 0:
                g = g + self.wd * p
            self.m[i].mul_(b1).add_(g, alpha=1-b1)
            self.v[i].mul_(b2).addcmul_(g, g, value=1-b2)
            p.addcdiv_(self.m[i], self.v[i].sqrt().add(self.eps), value=-lr_t)

enc = Encoder(in_dim=IN_DIM, hidden=HIDDEN, latent=LATENT).to(DEVICE)
dec = MLPDecoder(LATENT, hidden=64).to(DEVICE)

# init- encourage small sigma (good for KL)
with torch.no_grad():
    nn.init.constant_(enc.lstd.lin.bias, -3.0)  # σ ≈ e^{-3} ≈ 0.05

opt = SimpleAdam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
bce = nn.BCEWithLogitsLoss()

def kld_normal(mu, logstd):
    # KL( N(mu, diag(sigma^2)) || N(0, I) ) summed over nodes in batch
    return -0.5 * torch.sum(1 + 2*logstd - mu.pow(2) - torch.exp(2*logstd))

# Utilities 
def sample_negatives(num_samples, forbidden_set, num_nodes, rng):
    neg = []
    seen = set()
    while len(neg) < num_samples:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u == v:
            continue
        tup = (u, v)
        if tup in forbidden_set or tup in seen:
            continue
        seen.add(tup)
        neg.append(tup)
    return np.asarray(neg, dtype=np.int64)

def csr_to_torch_coo(csr_mat, device):
    coo = csr_mat.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values  = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=csr_mat.shape, device=device).coalesce()

def build_batch_node_sets(pos_edges, neg_edges):
    # Nodes used by decoder in this batch:
    K_nodes = np.unique(np.concatenate([pos_edges.reshape(-1), neg_edges.reshape(-1)]).astype(np.int64))
    # One-hop neighborhoods for encoder first layer:
    U_set = set(int(k) for k in K_nodes)
    for k in K_nodes:
        for u in in_neigh[k]:
            U_set.add(int(u))
        for v in out_neigh[k]:
            U_set.add(int(v))
    U_nodes = np.array(sorted(U_set), dtype=np.int64)
    return K_nodes, U_nodes

def encode_batch_mu_logstd(K_nodes, U_nodes):
    """
    Compute (mu, logstd) for nodes in K, using U = K ∪ 1-hop neighbors
    First layer (hidden) computed for all nodes in U:
      h1_U = ReLU( W1 [ X_U ; (A_in[U,:] X) ; (A_out[U,:] X) ] )
    Second layer, only rows K:
      mu_K  = W_mu [ h_U[K] ; (A_in[K,U] h_U) ; (A_out[K,U] h_U) ]
      lstd_K analogous
    Notes:
      - X_* @ ... done with SciPy sparse; grads not needed w.r.t. input features.
      - (A_*[K,U] @ h_U) done as torch.sparse.mm to keep grads through h_U.
    """
    # Map nodes to local indices within U
    idx_in_U = {int(n): i for i, n in enumerate(U_nodes)}
    K_local = np.array([idx_in_U[int(k)] for k in K_nodes], dtype=np.int64)

    # --- First layer aggregations (SciPy) ---
    X_U = X_csr[U_nodes, :].toarray().astype(np.float32)                           # (|U| x IN_DIM)
    AinU_X = (A_in_csr[U_nodes, :]  @ X_csr).toarray().astype(np.float32)         # (|U| x IN_DIM)
    AoutU_X= (A_out_csr[U_nodes, :] @ X_csr).toarray().astype(np.float32)         # (|U| x IN_DIM)

    # Torch tensors
    XU_t   = torch.tensor(X_U,   dtype=torch.float32, device=DEVICE)
    AinU_t = torch.tensor(AinU_X,dtype=torch.float32, device=DEVICE)
    AoutU_t= torch.tensor(AoutU_X,dtype=torch.float32, device=DEVICE)

    H1 = torch.cat([XU_t, AinU_t, AoutU_t], dim=1)                                 # (|U| x 3*IN_DIM)
    h_U = enc.s1.lin(H1)
    h_U = F.relu(h_U)                                                              # (|U| x HIDDEN)

    # --- Second layer aggregations (torch sparse for grads) ---
    # Submatrices A_in[K, U] and A_out[K, U]
    Ain_KU = A_in_csr[K_nodes, :][:, U_nodes]
    Aout_KU= A_out_csr[K_nodes, :][:, U_nodes]

    Ain_KU_t = csr_to_torch_coo(Ain_KU, DEVICE)
    Aout_KU_t= csr_to_torch_coo(Aout_KU, DEVICE)

    h_K      = h_U[torch.tensor(K_local, dtype=torch.long, device=DEVICE)]         # (|K| x HIDDEN)
    Ain_h_K  = torch.sparse.mm(Ain_KU_t,  h_U)                                     # (|K| x HIDDEN)
    Aout_h_K = torch.sparse.mm(Aout_KU_t, h_U)                                     # (|K| x HIDDEN)

    H2 = torch.cat([h_K, Ain_h_K, Aout_h_K], dim=1)                                # (|K| x 3*HIDDEN)

    mu_K    = enc.mu.lin(H2)                                                       # (|K| x LATENT)
    logstd_K= enc.lstd.lin(H2)                                                     # (|K| x LATENT)
    logstd_K= torch.clamp(logstd_K, -5.0, 2.0)
    return mu_K, logstd_K, K_local  # return K_local to line up with z indices

def edges_to_local(edges, K_nodes, K_local):
    # map global node id -> local position in K
    g2l = {int(g): int(i) for i, g in enumerate(K_nodes)}
    # edges_local uses local indices (0..|K|-1) to index z in this batch
    el = np.empty_like(edges)
    for i, (u, v) in enumerate(edges):
        el[i,0] = g2l[int(u)]
        el[i,1] = g2l[int(v)]
    return el

# ------------------------------ Evaluation helpers ------------------------------
@torch.no_grad()
def encode_all_nodes_in_chunks(chunk=ENC_NODE_CHUNK):
    """Compute mu for ALL nodes by chunking the node set and using the same
       1-hop U expansion per chunk. Returns dense (N x LATENT) torch tensor on DEVICE."""
    mu_all = torch.empty((N, LATENT), dtype=torch.float32, device=DEVICE)
    # simple fixed order chunks
    order = np.arange(N, dtype=np.int64)
    for start in range(0, N, chunk):
        K_nodes = order[start:start+chunk]
        # U = K ∪ 1-hop neighbors
        U_set = set(int(k) for k in K_nodes)
        for k in K_nodes:
            for u in in_neigh[k]:
                U_set.add(int(u))
            for v in out_neigh[k]:
                U_set.add(int(v))
        U_nodes = np.array(sorted(U_set), dtype=np.int64)
        mu_K, _, K_local = encode_batch_mu_logstd(K_nodes, U_nodes)
        mu_all[K_nodes] = mu_K
    return mu_all

@torch.no_grad()
def evaluate(mu_all, pos_edges, all_forbidden, neg_mult=1.0):
    if len(pos_edges) == 0:
        return float("nan"), float("nan")
    n_pos = len(pos_edges)
    rng_local = np.random.default_rng(SEED+1234)
    neg_edges = sample_negatives(int(math.ceil(neg_mult*n_pos)), all_forbidden, N, rng_local)
    logits_pos = dec(mu_all, torch.tensor(pos_edges, dtype=torch.long, device=DEVICE))
    logits_neg = dec(mu_all, torch.tensor(neg_edges, dtype=torch.long, device=DEVICE))
    y_true  = np.r_[np.ones(len(logits_pos)), np.zeros(len(logits_neg))]
    y_score = torch.cat([logits_pos, logits_neg]).detach().cpu().numpy()
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    return auc, ap
