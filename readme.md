# Pipeline Overview

## 1. Pre-processing

The files should be run in the following order:

1. `industry_classifications.py`  
   - Generates unique mappings for each entity to industry codes of three different types: **NAICS**, **NACE**, and **SIC**.

2. `extract_ownership_chunks.py`  
   - Splits our ~300GB datasets into **20 subchunks**.  
   - There will be:
     - 20 chunks for all cases where a **Chinese company is a shareholder**, and  
     - 20 chunks where a **Chinese company is a subsidiary**.  
   - Drops duplicates when a Chinese company appears both as a shareholder and as a subsidiary.

3. `merge_metadata_ownership.py`  
   - Loops over the **40 chunks** (20 shareholder chunks + 20 subsidiary chunks).  
   - Merges **metadata** with **ownership structure** for each chunk.

4. `graph_prep.py`  
   - Harmonizes industry code/description fields (first non-missing industry code, longest industry description), renames ISO columns and drops rows with blank descriptions.
  
5. `graph_exmple.py`
   - Plots the example network graph of a conglomerate.
   - ![Plot](component_009_network_iso.pdf)
     
---

## 2. VGAE

The files should be run in the following order:

1. `build_graph.py`  
   - Builds a cleaned, deduplicated shareholder-to-subsidiary directed graph from the data (merging ISO/SIC node attributes) 
   - Writes nodes/edges and a .edgelist, and prints top in/out-degree summaries
       
2. `helper_train.py`  
   - Helper functions and parameters for training a directed VGAE (GraphSAGE encoder + asymmetric MLP decoder)
   - Helper functions for optimization (BCE+KL with warm-up) and evaluation (ROC-AUC/AP).
     
3. `train.py`  
   - Executes the training loop with mini-batch edge sampling and negative sampling on the shareholder-to-subsidiary graph (see report for details)
   - Selects the best epoch by AUC/AP, and writes the resulting node embeddings to dataset.

4. `embedding_query.py`  
   - Streams the embedding space, L2-normalizes, builds a FAISS Flat (CPU/GPU) index, and for specified firm_ids in query exports their top-5 cosine neighbors.

5. `closest_embedding.py`  
   - Performs batched FAISS-Flat cosine search to find each embedding’s exact nearest neighbor (self-dropped) and writes the per-node results (full dataset)
