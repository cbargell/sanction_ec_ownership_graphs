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

---

## 2. VGAE

The files should be run in the following order:

1. `...`  
2. `...`  
3. `...`  
