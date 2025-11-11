import os
import pandas as pd

BASE = r"O:\scratch\yicheng\230"   
usecols = [
    "bvdid","shareholderbvdid",
    "sub_iso","shareholder_iso",
    "sub_sic","shareholder_sic",
    "sub_sic_descr","shareholder_sic_descr"
]

files = []
for i in range(1, 21):
    f1 = os.path.join(BASE, f"SHARE_CN_Links_2022_part{str(i).zfill(2)}of20.dta_with_industry.csv")
    f2 = os.path.join(BASE, f"SHARE_CN_Links_2022_part{i}of20.dta_with_industry.csv")
    if os.path.exists(f1): files.append(f1)
    elif os.path.exists(f2): files.append(f2)

print(f"Found {len(files)} files")
if not files:
    raise FileNotFoundError("No part files found under BASE")

all_nodes, all_edges = [], []

for PATH in files:
    print(f"[load] {PATH}")
    df = pd.read_csv(PATH, dtype=str, usecols=usecols).fillna("")
    for c in usecols:
        df[c] = df[c].astype(str).str.strip()

    # keep rows with both ends present and not identical
    df = df[(df["bvdid"]!="") & (df["shareholderbvdid"]!="")]
    df = df[df["bvdid"] != df["shareholderbvdid"]]

    # nodes as attributes (country + SIC)
    subs = df[["bvdid","sub_iso","sub_sic","sub_sic_descr"]].rename(
        columns={"bvdid":"firm_id","sub_iso":"iso","sub_sic":"sic","sub_sic_descr":"sic_descr"}
    )
    owners = df[["shareholderbvdid","shareholder_iso","shareholder_sic","shareholder_sic_descr"]].rename(
        columns={"shareholderbvdid":"firm_id","shareholder_iso":"iso","shareholder_sic":"sic","shareholder_sic_descr":"sic_descr"}
    )
    nodes = pd.concat([subs, owners], ignore_index=True)

    # prefer non-empty entries when the same firm appears multiple times
    nodes = (nodes.replace({"": None})
                  .sort_values(["firm_id"])
                  .groupby("firm_id", as_index=False)
                  .agg({
                      "iso":       lambda s: next((x for x in s if x), None),
                      "sic":       lambda s: next((x for x in s if x), None),
                      "sic_descr": lambda s: next((x for x in s if x), None),
                  })
                  .fillna(""))

    # unweighted directed edge list (shareholder -> subsidiary), collapse duplicates
    edges = (df[["shareholderbvdid","bvdid"]]
             .rename(columns={"shareholderbvdid":"src","bvdid":"dst"}))
    edges = edges.groupby(["src", "dst"]).size().reset_index(name="n_rows")
    edges["sample_rel"] = "unknown"

    all_nodes.append(nodes)
    all_edges.append(edges)
# combine across
nodes = pd.concat(all_nodes, ignore_index=True)
nodes = (nodes.replace({"": None})
              .sort_values(["firm_id"])
              .groupby("firm_id", as_index=False)
              .agg({
                  "iso":       lambda s: next((x for x in s if x), None),
                  "sic":       lambda s: next((x for x in s if x), None),
                  "sic_descr": lambda s: next((x for x in s if x), None),
              })
              .fillna(""))

edges = pd.concat(all_edges, ignore_index=True)
edges = (edges.groupby(["src","dst"], as_index=False)
              .agg(n_rows=("n_rows","sum")))
edges["sample_rel"] = "unknown"

# save by nodes and edges
nodes_out = os.path.join(BASE, "nodes_unweighted_all.csv")
edges_out = os.path.join(BASE, "edges_unweighted_all.csv")
nodes.to_csv(nodes_out, index=False)
edges[["src","dst"]].to_csv(edges_out, index=False)
print(f"[done] wrote:\n- {nodes_out}\n- {edges_out}")

elist_out = os.path.join(BASE, "graph_unweighted_all.edgelist")
with open(elist_out, "w", encoding="utf-8") as f:
    for r in edges.itertuples(index=False):
        f.write(f"{r.src} {r.dst}\n")
print(f"- {elist_out}")

# degree summaries (based on unique edges)
top_owners = (edges.groupby("src", as_index=False)["dst"]
                   .nunique().sort_values("dst", ascending=False).head(10))
top_subs   = (edges.groupby("dst", as_index=False)["src"]
                   .nunique().sort_values("src", ascending=False).head(10))
print("\nTop owners by out-degree (unique subsidiaries):")
for r in top_owners.itertuples(index=False):
    print(f"{r.src}\t{r.dst}")
print("\nTop subsidiaries by in-degree (unique owners):")
for r in top_subs.itertuples(index=False):
    print(f"{r.dst}\t{r.src}")
