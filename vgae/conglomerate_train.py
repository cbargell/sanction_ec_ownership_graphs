# train_conglomerate_set_encoder.py
# ------------------------------------------------------------
# Train a non-contrastive (VICReg-style) set encoder to produce
# conglomerate embeddings from pretrained firm embeddings
#
# Features:
# - Conglomerates = weakly connected components on TRAIN graph (E_train)
# - Keeps a Python variable conglomerate_nodes: list[np.ndarray] node indices per conglomerate
# - Logs per-minibatch and per-epoch losses
# - Saves checkpoint + CSV loss snapshots EVERY 2 EPOCHS
# - Resume from latest checkpoint in OUT_DIR
#
# Outputs:
# -Conglomerate membership as CSR-like arrays (indptr, indices) in NPZ
# -Conglomerate embeddings g_C for every conglomerate (CSV) at the end
# -Checkpoints: setencoder_ckpt_epoch_XXX.pt + loss CSVs
# ------------------------------------------------------------

import os, random, glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

NODES_CSV = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230/nodes_unweighted_all.csv"
EDGES_CSV = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230/edges_unweighted_all.csv"

FIRM_EMB_CSV = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230/emb_vgae_directed_10_cpu.csv"

OUT_DIR = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230"
MEMBERSHIP_OUT_NPZ = os.path.join(OUT_DIR, "conglomerate_membership_trainWCC.npz")
CONG_EMB_OUT_CSV   = os.path.join(OUT_DIR, "conglomerate_embeddings_setencoder.csv")

SEED = 7
VAL_RATIO = 0.05
TEST_RATIO = 0.10

LATENT_DIM = 64
PHI_H = 256
DG = 64

EPOCHS = 1000
BATCH_CONG = 256
LR = 1e-3
WEIGHT_DECAY = 1e-6

GAMMA = 1.0
L_INV = 25.0
L_VAR = 25.0
L_COV = 1.0

MIN_FRAC = 0.2
MAX_FRAC = 0.8

SAVE_EVERY = 100  # epochs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UnionFind:
    __slots__ = ("p", "r")
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)
        self.r = np.zeros(n, dtype=np.int8)
    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa == pb:
            return
        ra, rb = self.r[pa], self.r[pb]
        if ra < rb:
            self.p[pa] = pb
        elif ra > rb:
            self.p[pb] = pa
        else:
            self.p[pb] = pa
            self.r[pa] += 1


def split_edges(E_np, seed, test_ratio, val_ratio):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(E_np))
    n_test = int(len(E_np) * test_ratio)
    n_val  = int(len(E_np) * val_ratio)
    test_idx = perm[:n_test]
    val_idx  = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return E_np[train_idx], E_np[val_idx], E_np[test_idx]


def build_train_wcc_components(num_nodes, E_train):
    uf = UnionFind(num_nodes)
    for u, v in E_train:
        uf.union(int(u), int(v))  # ignore direction

    roots = np.fromiter((uf.find(i) for i in range(num_nodes)),
                        dtype=np.int64, count=num_nodes)
    uniq_roots, inv = np.unique(roots, return_inverse=True)
    comp_id = inv.astype(np.int64)

    K = len(uniq_roots)
    buckets = [[] for _ in range(K)]
    for node, cid in enumerate(comp_id.tolist()):
        buckets[cid].append(node)

    components = [np.asarray(b, dtype=np.int64) for b in buckets]
    return components, comp_id


def save_membership_csr_like(components, out_npz_path):
    K = len(components)
    sizes = np.array([len(c) for c in components], dtype=np.int64)
    indptr = np.empty(K + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(sizes, out=indptr[1:])
    indices = np.concatenate(components, axis=0) if K > 0 else np.empty(0, dtype=np.int64)
    np.savez_compressed(out_npz_path, indptr=indptr, indices=indices, sizes=sizes)
    return indptr, indices, sizes


def load_firm_embeddings_aligned(firm_emb_csv, ids_in_node_order, latent_dim):
    emb_df = pd.read_csv(firm_emb_csv, dtype={"firm_id": str})
    if "firm_id" not in emb_df.columns:
        raise ValueError("Embedding file must contain a 'firm_id' column.")
    emb_df["firm_id"] = emb_df["firm_id"].astype(str)
    emb_df = emb_df.set_index("firm_id")

    expected = [str(i) for i in range(latent_dim)]
    if all(c in emb_df.columns for c in expected):
        cols = expected
    else:
        cols = list(emb_df.columns)
        if len(cols) < latent_dim:
            raise ValueError(f"Not enough embedding columns: found {len(cols)} but need {latent_dim}.")
        cols = cols[:latent_dim]

    missing = [fid for fid in ids_in_node_order if fid not in emb_df.index]
    if missing:
        raise ValueError(f"Embedding file missing {len(missing)} firm_id(s). Example: {missing[:5]}")

    Z = emb_df.loc[ids_in_node_order, cols].to_numpy(dtype=np.float32, copy=True)
    if Z.shape != (len(ids_in_node_order), latent_dim):
        raise ValueError(f"Aligned embedding matrix has shape {Z.shape}, expected {(len(ids_in_node_order), latent_dim)}.")
    return Z


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0):
        super().__init__()
        assert num_layers >= 1
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SetEncoder(nn.Module):
    """
    g(S) = rho( mean_{i in S} phi(mu_i) )
    """
    def __init__(self, d_in, h, d_g):
        super().__init__()
        self.d_g = int(d_g)
        self.phi = MLP(d_in, hidden_dim=h, out_dim=h, num_layers=2, dropout=0.0)
        self.rho = MLP(h, hidden_dim=h, out_dim=d_g, num_layers=2, dropout=0.0)

    def forward_views(self, mu, view_node_lists):
        """
        view_node_lists: list[np.ndarray] (variable-size sets), GLOBAL node indices.
        Returns: (B, d_g)
        """
        B = len(view_node_lists)
        if B == 0:
            return torch.empty((0, self.d_g), device=mu.device)

        lens = np.array([len(v) for v in view_node_lists], dtype=np.int64)
        if np.any(lens <= 0):
            raise ValueError("All views must be non-empty.")

        idx = np.concatenate(view_node_lists).astype(np.int64, copy=False)
        group = np.repeat(np.arange(B, dtype=np.int64), lens)

        idx_t = torch.tensor(idx, dtype=torch.long, device=mu.device)
        group_t = torch.tensor(group, dtype=torch.long, device=mu.device)
        counts = torch.tensor(lens, dtype=torch.float32, device=mu.device).unsqueeze(1)  # (B,1)

        h = self.phi(mu.index_select(0, idx_t))  # (total, h)
        agg = torch.zeros((B, h.shape[1]), dtype=h.dtype, device=mu.device)
        agg.index_add_(0, group_t, h)
        mean = agg / counts
        g = self.rho(mean)
        return g


def sample_two_disjoint_views(nodes_np, rng, min_frac=0.2, max_frac=0.8):
    n = int(len(nodes_np))
    if n < 2:
        return None, None

    frac = float(rng.uniform(min_frac, max_frac))
    m = int(max(1, round(frac * n)))
    m = min(m, n // 2)  # enforce disjoint
    if m < 1:
        return None, None

    perm = rng.permutation(n)
    s1 = nodes_np[perm[:m]]
    s2 = nodes_np[perm[m:2*m]]
    return s1, s2


def off_diagonal(x):
    d = x.shape[0]
    return x.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()


def vicreg_loss(g1, g2, gamma=1.0, lam_inv=25.0, lam_var=25.0, lam_cov=1.0, eps=1e-4):
    """
    Batchwise VICReg: var/cov across (2B) samples.
    g1, g2: (B, d_g)
    """
    inv = ((g1 - g2) ** 2).sum(dim=1).mean()

    y = torch.cat([g1, g2], dim=0)  # (2B, d)
    std = torch.sqrt(y.var(dim=0, unbiased=False) + eps)
    var = torch.mean(F.relu(gamma - std))

    y = y - y.mean(dim=0, keepdim=True)
    cov = (y.T @ y) / (y.shape[0] - 1)  # (d,d)
    cov_off = off_diagonal(cov)
    cov_loss = (cov_off ** 2).sum() / g1.shape[1]

    total = lam_inv * inv + lam_var * var + lam_cov * cov_loss
    return total, inv.detach(), var.detach(), cov_loss.detach()


def latest_checkpoint(dirpath: str):
    paths = sorted(glob.glob(os.path.join(dirpath, "setencoder_ckpt_epoch_*.pt")))
    return paths[-1] if paths else None



set_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# load nodes/edges and build index mapping
nodes = pd.read_csv(NODES_CSV, dtype=str).fillna("")
edges = pd.read_csv(EDGES_CSV, dtype=str).fillna("")

ids = nodes["firm_id"].astype(str).tolist()
id2ix = {k: i for i, k in enumerate(ids)}
N = len(ids)

E = (edges.assign(src_ix=edges["src"].map(id2ix),
                  dst_ix=edges["dst"].map(id2ix))
          .dropna(subset=["src_ix", "dst_ix"]))
E["src_ix"] = E["src_ix"].astype(int)
E["dst_ix"] = E["dst_ix"].astype(int)
E = E[E["src_ix"] != E["dst_ix"]].drop_duplicates(subset=["src_ix", "dst_ix"]).reset_index(drop=True)
E_np = E[["src_ix", "dst_ix"]].to_numpy(dtype=np.int64)

print(f"Graph: {N:,} nodes, {len(E_np):,} directed edges")

# split edges
E_train, E_val, E_test = split_edges(E_np, SEED, TEST_RATIO, VAL_RATIO)
print(f"Edges split -> train: {len(E_train):,}, val: {len(E_val):,}, test: {len(E_test):,}")

# conglomerates as weakly connected components on TRAIN graph
components, comp_id = build_train_wcc_components(N, E_train)
K = len(components)
sizes = np.array([len(c) for c in components], dtype=np.int64)
print(f"Train-WCC conglomerates: K={K:,} | size stats: min={sizes.min()}, med={int(np.median(sizes))}, max={sizes.max()}")

# list of node indices per conglomerate
conglomerate_nodes = components  # list[np.ndarray], length K

# save membership to disk (CSR-like)
indptr, indices, sizes = save_membership_csr_like(conglomerate_nodes, MEMBERSHIP_OUT_NPZ)
print(f"Saved membership -> {MEMBERSHIP_OUT_NPZ}")
print("Membership format: component k nodes are indices[indptr[k]:indptr[k+1]]")

# load firm embeddings mu_i
Z = load_firm_embeddings_aligned(FIRM_EMB_CSV, ids, LATENT_DIM)  # (N, LATENT_DIM)
mu = torch.tensor(Z, dtype=torch.float32, device=DEVICE)
print(f"Loaded firm embeddings: {mu.shape} on {DEVICE}")

# build training set of conglomerates with size >= 2 
train_cong_ids = np.where(sizes >= 2)[0].astype(np.int64)
if len(train_cong_ids) == 0:
    raise RuntimeError("No conglomerates with size >= 2. Cannot train set encoder.")

# optimizer 
model = SetEncoder(d_in=LATENT_DIM, h=PHI_H, d_g=DG).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


rng = np.random.default_rng(SEED + 999)

# loss histories (persisted in checkpoints + CSVs) 
batch_loss_history = []
epoch_loss_history = []

# resume
start_epoch = 1
ckpt_path = latest_checkpoint(OUT_DIR)
if ckpt_path is not None:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["opt_state_dict"])
    batch_loss_history = ckpt.get("batch_loss_history", [])
    epoch_loss_history = ckpt.get("epoch_loss_history", [])
    start_epoch = int(ckpt["epoch"]) + 1

    # restore states
    if "torch_rng_state" in ckpt and ckpt["torch_rng_state"] is not None:
        torch.random.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt and ckpt["cuda_rng_state_all"] is not None:
        torch.cuda.random.set_rng_state_all(ckpt["cuda_rng_state_all"])
    if "np_rng_state" in ckpt and ckpt["np_rng_state"] is not None:
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt["np_rng_state"]

    print(f"Resuming from {ckpt_path} -> starting at epoch {start_epoch}")
else:
    print("No set-encoder checkpoint found; starting fresh.")

    
    
# training loop
num_batches = (len(train_cong_ids) + BATCH_CONG - 1) // BATCH_CONG
print(f"Minibatches per epoch: {num_batches} (train conglomerates={len(train_cong_ids)}, batch size={BATCH_CONG})")


for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    perm = rng.permutation(len(train_cong_ids))

    running_loss = 0.0
    running_inv = 0.0
    running_var = 0.0
    running_cov = 0.0
    n_batches = 0

    for bi, start in enumerate(range(0, len(perm), BATCH_CONG)):
        #if bi % 50 == 0:
            #print(f"epoch {epoch}, minibatch {bi}/{num_batches-1}")
            
        batch_ids = train_cong_ids[perm[start:start + BATCH_CONG]]

        v1, v2 = [], []
        for cid in batch_ids:
            s1, s2 = sample_two_disjoint_views(conglomerate_nodes[int(cid)], rng, MIN_FRAC, MAX_FRAC)
            if s1 is None:
                continue
            v1.append(s1)
            v2.append(s2)

        B = len(v1)
        if B == 0:
            continue

        g1 = model.forward_views(mu, v1)
        g2 = model.forward_views(mu, v2)

        loss, inv, var, cov = vicreg_loss(
            g1, g2,
            gamma=GAMMA,
            lam_inv=L_INV, lam_var=L_VAR, lam_cov=L_COV
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # per-minibatch log
        batch_loss_history.append({
            "epoch": int(epoch),
            "batch": int(bi),
            "loss": float(loss.item()),
            "inv": float(inv.item()),
            "var": float(var.item()),
            "cov": float(cov.item()),
            "B_pairs": int(B),
        })

        running_loss += float(loss.item())
        running_inv += float(inv.item())
        running_var += float(var.item())
        running_cov += float(cov.item())
        n_batches += 1

    denom = max(1, n_batches)

    # per-epoch log
    epoch_loss_history.append({
        "epoch": int(epoch),
        "avg_loss": float(running_loss / denom),
        "avg_inv": float(running_inv / denom),
        "avg_var": float(running_var / denom),
        "avg_cov": float(running_cov / denom),
        "n_batches": int(n_batches),
    })

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"[{epoch:03d}] "
            f"loss={running_loss/denom:.4f} | inv={running_inv/denom:.4f} "
            f"| var={running_var/denom:.4f} | cov={running_cov/denom:.4f} "
            f"| batches={n_batches}"
        )

    # save checkpoint + CSV
    if epoch % SAVE_EVERY == 0:
        ckpt_out = os.path.join(OUT_DIR, f"setencoder_ckpt_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "opt_state_dict": opt.state_dict(),
            "batch_loss_history": batch_loss_history,
            "epoch_loss_history": epoch_loss_history,
            "config": {
                "SEED": SEED, "VAL_RATIO": VAL_RATIO, "TEST_RATIO": TEST_RATIO,
                "LATENT_DIM": LATENT_DIM, "PHI_H": PHI_H, "DG": DG,
                "EPOCHS": EPOCHS, "BATCH_CONG": BATCH_CONG, "LR": LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "GAMMA": GAMMA, "L_INV": L_INV, "L_VAR": L_VAR, "L_COV": L_COV,
                "MIN_FRAC": MIN_FRAC, "MAX_FRAC": MAX_FRAC,
            },
            # states for resume
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.random.get_rng_state_all() if torch.cuda.is_available() else None,
            "np_rng_state": rng.bit_generator.state,
        }, ckpt_out)
        print(f"Saved checkpoint -> {ckpt_out}")

        # CSV snapshots 
        pd.DataFrame(batch_loss_history).to_csv(
            os.path.join(OUT_DIR, f"setencoder_batch_loss_epoch_{epoch:03d}.csv"),
            index=False
        )
        pd.DataFrame(epoch_loss_history).to_csv(
            os.path.join(OUT_DIR, f"setencoder_epoch_loss_epoch_{epoch:03d}.csv"),
            index=False
        )

# final conglomerate embeddings g_C for ALL components
model.eval()
with torch.no_grad():
    comp_id_t = torch.tensor(comp_id, dtype=torch.long, device=DEVICE)  # (N,)

    phi_all = model.phi(mu)  # (N, PHI_H)

    sum_phi = torch.zeros((K, PHI_H), dtype=phi_all.dtype, device=DEVICE)
    sum_phi.index_add_(0, comp_id_t, phi_all)

    counts = torch.bincount(comp_id_t, minlength=K).to(phi_all.dtype).unsqueeze(1)
    mean_phi = sum_phi / counts.clamp_min(1.0)

    g_all = model.rho(mean_phi)  # (K, DG)
    g_np = g_all.detach().cpu().numpy()

# save conglomerate embeddings
out = pd.DataFrame(g_np, columns=[f"g_{j}" for j in range(DG)])
out.insert(0, "conglomerate_id", np.arange(K, dtype=np.int64))
out.insert(1, "size", sizes.astype(np.int64))

rep_node = np.array([c[0] for c in conglomerate_nodes], dtype=np.int64) if K > 0 else np.empty(0, np.int64)
rep_firm = [ids[int(i)] for i in rep_node.tolist()] if K > 0 else []
out.insert(2, "rep_firm_id", rep_firm)

out.to_csv(CONG_EMB_OUT_CSV, index=False)
print(f"Saved conglomerate embeddings -> {CONG_EMB_OUT_CSV}  (shape: {out.shape})")

