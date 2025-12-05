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
   - ![Plot](pre_processing/component_009_network_iso.pdf)
  
6. `get_company_name.py`
   - Function extracting the company name for a given company id.
     
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

4. `train_resume.py`  
   - Adds checkpointing and auto-resume for VGAE training: it periodically saves encoder/decoder weights, the best validation AUC state, and loss histories, and if a checkpoint exists it reloads everything and continues from the next epoch (we used this as jobs on Sherlock have time limits, and we need to run for many epochs across multiple submissions).
   - The rest same as train.py

5. `conglomerate_train.py`  
   - Loads pretrained firm embeddings and trains a set-encoder that turns the firms inside each conglomerate into a single conglomerate embedding, using a VICReg-style objective.
   - computes conglomerate embeddings for all conglomerates and saves with each conglomerate’s size and representative firm IDs.

6. `embedding_query.py`  
   - Streams the embedding space, L2-normalizes, builds a FAISS Flat (CPU/GPU) index, and for specified firm_ids in query exports their top-5 cosine neighbors.

7. `closest_embedding.py`  
   - Performs batched FAISS-Flat cosine search to find each embedding’s exact nearest neighbor (self-dropped) and writes the per-node results (full dataset).

8. `visualization.py`  
   - Panel A: loads firm embeddings, normalizes them, randomly samples up to 200,000 firms, projects them to 2D with UMAP, clusters the sampled firms into 150 groups using MiniBatch K-means, and plots the 2D points colored by cluster.
   - Panel B (with conglomerates): computes weakly connected components as conglomerates, samples up to 200,000 firms, runs UMAP again, and plots the 2D points colored by the largest 60 conglomerates (others in grey).

9. `helper_loss.py`  
   - Documents scripts plotting loss over minibatches and plots validation vs. test AUC and average precision over epochs/minibatches.
  
---

## 3. Prediction

The files should be run in the following order:

1. `prediction_data.py'
   - Generates crosswalks across firm identifiers (factset entity id, isin, bvdid)
   - Merges the data on sanctions and export controls from Clayton et al. with out VGAE 64-d embeddings
  
2. `prediction.py'
   - Executes firm-level prediction of sanctions and export controls in the future
   - Contains all tested specifications:
     - shallow NN with future-adjusted cross-entropy loss and weighted classes
     - shallow NN with weighted classes (with and without firm-features, at the quarter, year and firm level)
     - logistic regression with weighted classes (with and without firm-features, at the quarter, year and firm level)
     - logistic regression
