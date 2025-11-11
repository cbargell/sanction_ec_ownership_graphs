import pandas as pd
import pandas as pd
import networkx as nx
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

gcap_data = "/oak/stanford/groups/maggiori/GCAP/data"

##### IMPORT and APPEND ALL SHAREHOLDER DATASETS 

filepaths = [
    f"{gcap_data}/scratch/yicheng/230/SHARE_CN_Links_2022_part{str(i).zfill(2)}of20.dta_with_industry.csv"
    for i in range(1, 21)
]

dataframes = [pd.read_csv(filepath) for filepath in filepaths]
combined_df = pd.concat(dataframes, ignore_index=True)

##### EXTRACT BIG G GRAPH OF ALL CONGLOMERATES

# load & minimal normalize
usecols = [
    "bvdid","shareholderbvdid",
    "sub_iso","shareholder_iso",
    "sub_sic","shareholder_sic",
    "sub_sic_descr","shareholder_sic_descr"
]

df = combined_df

# Convert all columns to strings first
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
                  "iso": lambda s: next((x for x in s if x), None),
                  "sic": lambda s: next((x for x in s if x), None),
                  "sic_descr": lambda s: next((x for x in s if x), None),
              })
              .fillna(""))

# unweighted directed edge list (shareholder -> subsidiary)
edges = (df[["shareholderbvdid","bvdid"]]
         .rename(columns={"shareholderbvdid":"src","bvdid":"dst"}))
                   
edges = edges.groupby(["src", "dst"]).size().reset_index(name="n_rows")
edges["sample_rel"] = "unknown"

# build DiGraph (no weights)
G = nx.DiGraph()
G.add_nodes_from(nodes["firm_id"])
nx.set_node_attributes(G, nodes.set_index("firm_id").to_dict(orient="index"))
for r in edges.itertuples(index=False):
    G.add_edge(r.src, r.dst, n_rows=int(r.n_rows), sample_rel=r.sample_rel)

print(f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# quick sanity checks
isolated = [n for n in G.nodes if G.degree(n)==0]
print(f"- Isolated nodes: {len(isolated):,}")

# strongly connected components (cycles are common in corp structures)
scc_sizes = sorted((len(c) for c in nx.strongly_connected_components(G)), reverse=True)
print(f"- Strongly connected components (top 3 sizes): {scc_sizes[:3]}")

# degree summaries
out_deg = G.out_degree()
in_deg = G.in_degree()

top_owners = sorted(out_deg, key=lambda x: x[1], reverse=True)[:10]
top_subs   = sorted(in_deg, key=lambda x: x[1], reverse=True)[:10]

print("\nTop owners by out-degree (edges to subsidiaries):")
for n, d in top_owners:
    print(f"{n}\t{d}")

print("\nTop subsidiaries by in-degree (owners pointing to them):")
for n, d in top_subs:
    print(f"{n}\t{d}")

# save
nodes.to_csv(f"{gcap_data}/scratch/chiara/cs230/nodes_unweighted.csv", index=False)
edges[["src","dst"]].to_csv(f"{gcap_data}/scratch/chiara/cs230/edges_unweighted.csv", index=False)
nx.write_edgelist(G, f"{gcap_data}/scratch/chiara/cs230/graph_unweighted.edgelist", data=False)


##### PLOT EXAMPLE GRAPH (CONGLOMERATE 9)

comp_dir = Path(f"{gcap_data}/scratch/chiara/cs230/orbis/temp/conglomerates/component_009")
gml = comp_dir / "graph.graphml"  # created earlier

H = nx.read_graphml(gml)

#  Layout: kamada_kawai often looks better than spring for paper figures ---
pos = nx.kamada_kawai_layout(H)  # or nx.spring_layout(H, seed=42, k=0.25)

# Node colors by an attribute (e.g. iso); fallback to 'UNK' ---
attr = "iso" 
values = [d.get(attr, "UNK") for _, d in H.nodes(data=True)]
unique_vals = sorted(set(values))

cmap = plt.get_cmap("tab20")
color_map = {v: cmap(i / max(1, len(unique_vals) - 1)) for i, v in enumerate(unique_vals)}
node_colors = [color_map[v] for v in values]

# Figure + style 
plt.figure(figsize=(6, 6), dpi=300)

# Edges: light, slightly curved, de-emphasized
nx.draw_networkx_edges(
    H,
    pos,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=8,
    width=0.4,
    alpha=0.25,
    edge_color="0.6",
    connectionstyle="arc3,rad=0.12",
)

# Nodes: small, with thin border
nx.draw_networkx_nodes(
    H,
    pos,
    node_size=18,
    node_color=node_colors,
    linewidths=0.2,
    edgecolors="black",
)

plt.title(
    f"{comp_dir.name.replace('_', ' ')}\n"
    f"nodes = {H.number_of_nodes()}, edges = {H.number_of_edges()}",
    fontsize=9,
)
plt.axis("off")
plt.tight_layout(pad=0.05)

# Optional small legend, but only if you have few categories
if len(unique_vals) <= 10:
    handles = [
        mpl.lines.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=4,
            markerfacecolor=color_map[v],
            markeredgecolor="black",
            label=v,
        )
        for v in unique_vals
    ]
    plt.legend(
        handles=handles,
        title=attr.upper(),
        fontsize=6,
        title_fontsize=7,
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

plt.show()