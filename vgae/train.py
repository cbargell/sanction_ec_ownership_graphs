# train.py
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

# import helper_train

# training
best_val_auc = -1.0
best_state = None
rng_train = np.random.default_rng(SEED)

num_batches = max(1, (len(E_train) + EDGE_BATCH_SIZE - 1) // EDGE_BATCH_SIZE)
print(f"Training with {num_batches} mini-batches per epoch (batch size ~ {EDGE_BATCH_SIZE} edges)")

for epoch in range(1, EPOCHS+1):
    print("current epoch:"+str(epoch))
    enc.train(); dec.train(); opt.zero_grad()

    # shuffle training edges each epoch
    perm_train = rng_train.permutation(len(E_train))
    E_train_shuf = E_train[perm_train]

    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0

    for b in range(num_batches):
        print("current batch:"+str(b))
        start = b * EDGE_BATCH_SIZE
        stop  = min(len(E_train_shuf), start + EDGE_BATCH_SIZE)
        pos_edges = E_train_shuf[start:stop]
        if len(pos_edges) == 0:
            continue

        neg_edges = sample_negatives(int(NEG_TRAIN_RATIO * len(pos_edges)), E_all_set, N, rng_train)

        # node sets for this batch
        K_nodes, U_nodes = build_batch_node_sets(pos_edges, neg_edges)

        # encode only for K (using U for 1-hop aggregations)
        mu_K, logstd_K, K_local = encode_batch_mu_logstd(K_nodes, U_nodes)
        eps = torch.randn_like(mu_K)
        z_K = mu_K + torch.exp(logstd_K) * eps

        # build local edges for decoder (indices into z_K)
        pos_local = edges_to_local(pos_edges, K_nodes, K_local)
        neg_local = edges_to_local(neg_edges, K_nodes, K_local)

        logits_pos = dec(z_K, pos_local)
        logits_neg = dec(z_K, neg_local)
        labels = torch.cat([torch.ones_like(logits_pos), torch.zeros_like(logits_neg)], dim=0)
        logits = torch.cat([logits_pos, logits_neg], dim=0)

        recon = bce(logits, labels)
        kl = kld_normal(mu_K, logstd_K) / max(1, len(K_nodes))  # avg per node in batch

        beta = BETA_MAX * min(1.0, epoch / WARMUP_EPOCHS)
        loss = recon + beta * kl

        loss.backward()
        opt.step()
        opt.zero_grad()

        running_loss += loss.item()
        running_recon += recon.item()
        running_kl += kl.item()

    # evaluation (compute mu for ALL nodes in chunks)
    enc.eval(); dec.eval()
    with torch.no_grad():
        mu_eval = encode_all_nodes_in_chunks(chunk=ENC_NODE_CHUNK)  # (N x LATENT) on DEVICE
        val_auc, val_ap = evaluate(mu_eval, E_val, E_all_set, neg_mult=1.0)
        test_auc, test_ap = evaluate(mu_eval, E_test, E_all_set, neg_mult=1.0)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {
            "enc": enc.state_dict(),
            "dec": dec.state_dict(),
            "mu": mu_eval.detach().cpu()
        }

    if epoch % 5 == 0 or epoch == 1:
        avg_loss = running_loss / num_batches
        avg_recon = running_recon / num_batches
        avg_kl = running_kl / num_batches
        print(f"[{epoch:03d}] loss={avg_loss:.4f}  recon={avg_recon:.4f}  KL={avg_kl:.4f}  "
              f"beta={beta:.4f} | val AUC={val_auc:.3f} AP={val_ap:.3f} | test AUC={test_auc:.3f} AP={test_ap:.3f}")

# save best embeddings
if best_state is None:
    with torch.no_grad():
        mu_final = encode_all_nodes_in_chunks(chunk=ENC_NODE_CHUNK)
        Z = mu_final.detach().cpu().numpy()
else:
    enc.load_state_dict(best_state["enc"]); dec.load_state_dict(best_state["dec"])
    Z = best_state["mu"].numpy()

emb = pd.DataFrame(Z)
emb.insert(0, "firm_id", ids)
emb.to_csv(EMB_OUT, index=False)
print(f"Saved embeddings -> {EMB_OUT}  (shape: {emb.shape})")
