import glob

def save_checkpoint(epoch,
                    enc,
                    dec,
                    best_val_auc,
                    best_state,
                    batch_loss_history,
                    epoch_loss_history):
    """
    save model parameters and training state at given epoch.
    """
    ckpt = {
        "epoch": int(epoch),
        "enc_state_dict": enc.state_dict(),
        "dec_state_dict": dec.state_dict(),
        "best_val_auc": float(best_val_auc),
        "best_state": best_state,
        "batch_loss_history": batch_loss_history,
        "epoch_loss_history": epoch_loss_history,
    }
    out_dir = os.path.dirname(EMB_OUT)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

def latest_checkpoint(dirpath: str):
    paths = sorted(glob.glob(os.path.join(dirpath, "checkpoint_epoch_*.pt")))
    return paths[-1] if paths else None

# training with auto-resume
best_val_auc = -1.0
best_state = None
rng_train = np.random.default_rng(SEED)

num_batches = max(1, (len(E_train) + EDGE_BATCH_SIZE - 1) // EDGE_BATCH_SIZE)
print(f"Training with {num_batches} mini-batches per epoch (batch size ~ {EDGE_BATCH_SIZE} edges)")

# containers to store loss histories
batch_loss_history = []  # per-mini-batch stats
epoch_loss_history = []  # per-epoch averages

# resume (load latest checkpoint from the same folder as EMB_OUT)
out_dir = os.path.dirname(EMB_OUT)
os.makedirs(out_dir, exist_ok=True)

start_epoch = 1
ckpt_path = latest_checkpoint(out_dir)
if ckpt_path is not None:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # models
    enc.load_state_dict(ckpt["enc_state_dict"])
    dec.load_state_dict(ckpt["dec_state_dict"])

    # trackers
    best_val_auc = ckpt.get("best_val_auc", best_val_auc)
    best_state   = ckpt.get("best_state",   best_state)
    batch_loss_history = ckpt.get("batch_loss_history", batch_loss_history)
    epoch_loss_history = ckpt.get("epoch_loss_history", epoch_loss_history)

    # continue from the next epoch
    start_epoch = int(ckpt["epoch"]) + 1
    print(f"Resuming from {ckpt_path} -> starting at epoch {start_epoch}")
else:
    print("No checkpoint found; starting fresh.")

for epoch in range(start_epoch, EPOCHS+1):
    enc.train(); dec.train(); opt.zero_grad()
    print("now epoch" + str(epoch))

    # shuffle training edges each epoch
    perm_train = rng_train.permutation(len(E_train))
    E_train_shuf = E_train[perm_train]

    running_loss = 0.0
    running_recon = 0.0
    running_kl = 0.0

    for b in range(num_batches):
        if b % 50 == 0:
            print("minibatch" + str(b))

        start = b * EDGE_BATCH_SIZE
        stop  = min(len(E_train_shuf), start + EDGE_BATCH_SIZE)
        pos_edges = E_train_shuf[start:stop]
        if len(pos_edges) == 0:
            continue

        neg_edges = sample_negatives(
            int(NEG_TRAIN_RATIO * len(pos_edges)),
            E_all_set,
            N,
            rng_train
        )

        # node sets for this batch
        K_nodes, U_nodes = build_batch_node_sets(pos_edges, neg_edges)

        mu_K, logstd_K, K_local = encode_batch_mu_logstd(K_nodes, U_nodes)
        eps = torch.randn_like(mu_K)
        z_K = mu_K + torch.exp(logstd_K) * eps

        # build local edges for decoder (indices into z_K)
        pos_local = edges_to_local(pos_edges, K_nodes, K_local)
        neg_local = edges_to_local(neg_edges, K_nodes, K_local)

        logits_pos = dec(z_K, pos_local)
        logits_neg = dec(z_K, neg_local)

        labels = torch.cat(
            [torch.ones_like(logits_pos), torch.zeros_like(logits_neg)],
            dim=0
        )
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

        # log per-mini-batch stats for plotting later
        batch_loss_history.append({
            "epoch":       int(epoch),
            "batch":       int(b),
            "loss":        float(loss.item()),
            "recon":       float(recon.item()),
            "kl":          float(kl.item()),
            "beta":        float(beta),
            "n_pos_edges": int(len(pos_edges)),
        })

    # evaluation (compute mu for all nodes in chunks) 
    enc.eval(); dec.eval()
    with torch.no_grad():
        mu_eval = encode_all_nodes_in_chunks(chunk=ENC_NODE_CHUNK)  # (N x LATENT) 
        val_auc, val_ap = evaluate(mu_eval, E_val, E_all_set, neg_mult=1.0)
        test_auc, test_ap = evaluate(mu_eval, E_test, E_all_set, neg_mult=1.0)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = {
            "enc": enc.state_dict(),
            "dec": dec.state_dict(),
            "mu": mu_eval.detach().cpu()
        }

    avg_loss = running_loss / num_batches
    avg_recon = running_recon / num_batches
    avg_kl = running_kl / num_batches

    # store per-epoch stats
    epoch_loss_history.append({
        "epoch":    int(epoch),
        "avg_loss": float(avg_loss),
        "avg_recon": float(avg_recon),
        "avg_kl":   float(avg_kl),
        "beta":     float(beta),      # last beta of this epoch
        "val_auc":  float(val_auc),
        "val_ap":   float(val_ap),
        "test_auc": float(test_auc),
        "test_ap":  float(test_ap),
    })

    if epoch % 2 == 0 or epoch == 1:
        print(
            f"[{epoch:03d}] loss={avg_loss:.4f}  recon={avg_recon:.4f}  KL={avg_kl:.4f}  "
            f"beta={beta:.4f} | val AUC={val_auc:.3f} AP={val_ap:.3f} | "
            f"test AUC={test_auc:.3f} AP={test_ap:.3f}"
        )

    # save checkpoint & CSV snapshots every 2 epochs
    if epoch % 2 == 0:
        save_checkpoint(
            epoch=epoch,
            enc=enc,
            dec=dec,
            best_val_auc=best_val_auc,
            best_state=best_state,
            batch_loss_history=batch_loss_history,
            epoch_loss_history=epoch_loss_history,
        )
        # CSV snapshots under the same folder as EMB_OUT
        pd.DataFrame(batch_loss_history).to_csv(
            os.path.join(out_dir, f"train_batch_loss_epoch_{epoch:03d}.csv"), index=False
        )
        pd.DataFrame(epoch_loss_history).to_csv(
            os.path.join(out_dir, f"train_epoch_loss_epoch_{epoch:03d}.csv"), index=False
        )

# save best embeddings
if best_state is None:
    with torch.no_grad():
        mu_final = encode_all_nodes_in_chunks(chunk=ENC_NODE_CHUNK)
        Z = mu_final.detach().cpu().numpy()
else:
    enc.load_state_dict(best_state["enc"])
    dec.load_state_dict(best_state["dec"])
    Z = best_state["mu"].numpy()

emb = pd.DataFrame(Z)
emb.insert(0, "firm_id", ids)
emb.to_csv(EMB_OUT, index=False)
print(f"Saved embeddings -> {EMB_OUT}  (shape: {emb.shape})")

# loss histories
pd.DataFrame(batch_loss_history).to_csv(
    os.path.join(out_dir, "train_batch_loss_cpu.csv"), index=False
)
pd.DataFrame(epoch_loss_history).to_csv(
    os.path.join(out_dir, "train_epoch_loss_cpu.csv"), index=False
)