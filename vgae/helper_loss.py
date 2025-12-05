import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


log_dir = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230/"  

cands = glob.glob(os.path.join(log_dir, "train_batch_loss_epoch_*"))
files = [f for f in cands if os.path.isfile(f) and re.search(r"epoch_(\d+)$|epoch_(\d+)\.csv$", f)]
files = sorted(
    files,
    key=lambda f: int(next(g for g in re.search(r"epoch_(\d+)", os.path.basename(f)).groups() if g))
)

dfs = []
for f in files:
    end_epoch = int(re.search(r"epoch_(\d+)", os.path.basename(f)).group(1))
    df = pd.read_csv(f)

    local_max = int(df["epoch"].max())
    if local_max <= 2 and int(df["epoch"].min()) == 1:
        df["global_epoch"] = df["epoch"] + (end_epoch - local_max)
    else:
        df["global_epoch"] = df["epoch"]

    df["src_file"] = os.path.basename(f)
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
all_df = all_df.sort_values(["global_epoch", "batch"]).reset_index(drop=True)

# global minibatch index (0,1,2,... across all epochs/files)
all_df["global_step"] = np.arange(len(all_df))

#window = 50
#all_df["loss_smooth"] = all_df["loss"].rolling(window, min_periods=1).mean()

plt.figure(figsize=(12, 5))
plt.plot(all_df["global_step"], all_df["loss"], linewidth=1, alpha=0.35, label="loss")
#plt.plot(all_df["global_step"], all_df["loss_smooth"], linewidth=2, label=f"loss (MA{window})")
plt.xlabel("Mini-batch")
plt.ylabel("Loss")
plt.title("Loss over mini-batches")
plt.legend()
plt.tight_layout()
plt.show()


log_dir = "/oak/stanford/groups/maggiori/GCAP/data/scratch/yicheng/230"  

# all epoch-metrics
files = sorted(glob.glob(os.path.join(log_dir, "train_epoch_loss_epoch_*.csv")))

def get_suffix_int(path):
    m = re.search(r"epoch_(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else -1

files = sorted(files, key=get_suffix_int)

dfs = []
for f in files:
    df = pd.read_csv(f)

    # each file contains 2 epochs. make a global epoch index.
    end_epoch = get_suffix_int(f)
    local_max = int(df["epoch"].max())

    if local_max <= 2 and int(df["epoch"].min()) == 1:
        df["global_epoch"] = df["epoch"] + (end_epoch - local_max)
    else:
        df["global_epoch"] = df["epoch"]

    df["src_file"] = os.path.basename(f)
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

all_df = (all_df.sort_values(["global_epoch", "src_file"])
                .drop_duplicates(subset=["global_epoch"], keep="last")
                .sort_values("global_epoch")
                .reset_index(drop=True))

plt.figure(figsize=(10, 5))
plt.plot(all_df["global_epoch"], all_df["val_auc"], marker="o", label="Validation AUC")
plt.plot(all_df["global_epoch"], all_df["test_auc"], marker="o", label="Test AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Validation vs Test AUC by Epoch")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(all_df["global_epoch"], all_df["val_ap"], marker="o", label="Validation AP")
plt.plot(all_df["global_epoch"], all_df["test_ap"], marker="o", label="Test AP")
plt.xlabel("Epoch")
plt.ylabel("Average Precision")
plt.title("Validation vs Test AP by Epoch")
plt.legend()
plt.tight_layout()
plt.show()
